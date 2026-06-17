"""Coconut hidden-state feedback wrapper for causal language models."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from coconut_qwen.data import IGNORE_INDEX


@dataclass(frozen=True)
class CoconutOutput:
    """Output returned by `CoconutForCausalLM`."""

    loss: torch.Tensor | None
    logits: torch.Tensor
    inputs_embeds: torch.Tensor
    latent_positions: list[int]


class CoconutForCausalLM(nn.Module):
    """Wrap a causal LM with Coconut-style continuous latent thoughts."""

    def __init__(
        self,
        base_causallm: nn.Module,
        *,
        latent_token_id: int,
        eos_token_id: int | None,
    ) -> None:
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.embedding = base_causallm.get_input_embeddings()

        signature = inspect.signature(base_causallm.forward)
        self._forward_params = set(signature.parameters)
        self._forward_accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CoconutOutput:
        """Run a Coconut training/evaluation forward pass."""

        del kwargs
        self._validate_input_ids(input_ids)
        attention_mask = self._default_attention_mask(input_ids, attention_mask)
        position_ids = self._default_position_ids(input_ids, position_ids)

        inputs_embeds, latent_positions = self.prepare_inputs_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        outputs = self._call_base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
        )
        logits = outputs.logits
        loss = self._compute_loss(logits, labels) if labels is not None else None

        return CoconutOutput(
            loss=loss,
            logits=logits,
            inputs_embeds=inputs_embeds,
            latent_positions=latent_positions,
        )

    def prepare_inputs_embeds(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """Replace `<latent>` token embeddings with fed-back hidden states."""

        self._validate_input_ids(input_ids)
        attention_mask = self._default_attention_mask(input_ids, attention_mask)
        position_ids = self._default_position_ids(input_ids, position_ids)

        latent_positions = self._latent_positions(input_ids)
        inputs_embeds = self.embedding(input_ids)
        pieces = [
            inputs_embeds[:, pos : pos + 1, :]
            for pos in range(inputs_embeds.shape[1])
        ]

        for latent_pos in latent_positions:
            if latent_pos == 0:
                raise ValueError("latent token cannot be the first token")

            prefix_embeds = torch.cat(pieces[:latent_pos], dim=1)
            prefix_outputs = self._call_base(
                inputs_embeds=prefix_embeds,
                attention_mask=attention_mask[:, :latent_pos],
                position_ids=position_ids[:, :latent_pos],
                output_hidden_states=True,
            )
            hidden_states = getattr(prefix_outputs, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError("base model did not return hidden states")

            pieces[latent_pos] = hidden_states[-1][:, -1:, :]

        return torch.cat(pieces, dim=1), latent_positions

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """Greedy generation after filling latent embeddings."""

        self._validate_input_ids(input_ids)
        attention_mask = self._default_attention_mask(input_ids, attention_mask)
        position_ids = self._default_position_ids(input_ids, None)
        inputs_embeds, _ = self.prepare_inputs_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        token_ids = input_ids[0].detach().tolist()
        cur_embeds = inputs_embeds
        for _ in range(max_new_tokens):
            cur_attention = torch.ones(
                (1, cur_embeds.shape[1]),
                dtype=torch.long,
                device=cur_embeds.device,
            )
            cur_positions = torch.arange(
                cur_embeds.shape[1],
                dtype=torch.long,
                device=cur_embeds.device,
            ).unsqueeze(0)
            outputs = self._call_base(
                inputs_embeds=cur_embeds,
                attention_mask=cur_attention,
                position_ids=cur_positions,
                output_hidden_states=False,
            )
            next_token_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
            token_ids.append(next_token_id)

            if self.eos_token_id is not None and next_token_id == self.eos_token_id:
                break

            next_token = torch.tensor(
                [[next_token_id]],
                dtype=torch.long,
                device=cur_embeds.device,
            )
            next_embed = self.embedding(next_token)
            cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)

        return torch.tensor([token_ids], dtype=torch.long, device=input_ids.device)

    def _call_base(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        output_hidden_states: bool,
    ) -> Any:
        kwargs: dict[str, Any] = {"inputs_embeds": inputs_embeds}
        self._maybe_add_kwarg(kwargs, "attention_mask", attention_mask)
        self._maybe_add_kwarg(kwargs, "position_ids", position_ids)
        self._maybe_add_kwarg(kwargs, "output_hidden_states", output_hidden_states)
        self._maybe_add_kwarg(kwargs, "use_cache", False)
        return self.base_causallm(**kwargs)

    def _maybe_add_kwarg(
        self,
        kwargs: dict[str, Any],
        name: str,
        value: Any,
    ) -> None:
        if value is None:
            return
        if self._forward_accepts_kwargs or name in self._forward_params:
            kwargs[name] = value

    def _latent_positions(self, input_ids: torch.Tensor) -> list[int]:
        latent_positions = (
            input_ids[0]
            .eq(self.latent_token_id)
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        return [int(position) for position in latent_positions]

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if input_ids.shape[0] != 1:
            raise NotImplementedError("CoconutForCausalLM currently supports batch size 1")
        if input_ids.shape[1] < 2:
            raise ValueError("input_ids must contain at least two tokens")

    def _default_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            return attention_mask
        return torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    def _default_position_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if position_ids is not None:
            return position_ids
        return torch.arange(
            input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0)

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
