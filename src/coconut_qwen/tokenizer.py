"""Tokenizer helpers for Coconut special tokens."""

from __future__ import annotations

from typing import Any

from coconut_qwen.data import BOT_TOKEN, EOT_TOKEN, LATENT_TOKEN, SPECIAL_TOKENS


def add_coconut_special_tokens(
    tokenizer: Any,
    model: Any | None = None,
    *,
    init_from_token: str = "<<",
) -> dict[str, int]:
    """Add Coconut markers to a HuggingFace tokenizer and optional model.

    Returns the token ids for `<bot>`, `<latent>`, and `<eot>`.
    """

    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
        _initialize_new_token_embeddings(
            tokenizer,
            model,
            init_from_token=init_from_token,
        )

    return {
        "bot": tokenizer.convert_tokens_to_ids(BOT_TOKEN),
        "latent": tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
        "eot": tokenizer.convert_tokens_to_ids(EOT_TOKEN),
    }


def _initialize_new_token_embeddings(
    tokenizer: Any,
    model: Any,
    *,
    init_from_token: str,
) -> None:
    """Initialize new special tokens from a stable existing token embedding."""

    source_ids = tokenizer.encode(init_from_token, add_special_tokens=False)
    if not source_ids:
        return

    source_id = source_ids[0]
    embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()

    for token in [BOT_TOKEN, LATENT_TOKEN, EOT_TOKEN]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        embedding.weight.data[token_id].copy_(embedding.weight.data[source_id])
        if output_embedding is not None:
            output_embedding.weight.data[token_id].copy_(
                output_embedding.weight.data[source_id]
            )
