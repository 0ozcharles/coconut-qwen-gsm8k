"""Run the required single-example Coconut overfit experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coconut_qwen.data import ReasoningExample, build_coconut_sample  # noqa: E402
from coconut_qwen.model import CoconutForCausalLM  # noqa: E402
from coconut_qwen.tokenizer import add_coconut_special_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="outputs/overfit_one")
    parser.add_argument("--stages", default="0,1,2")
    parser.add_argument("--steps-per-stage", type=int, default=120)
    parser.add_argument("--latent-tokens-per-step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    token_ids = add_coconut_special_tokens(tokenizer, base_model)
    base_model.to(device)

    model = CoconutForCausalLM(
        base_model,
        latent_token_id=token_ids["latent"],
        eos_token_id=tokenizer.eos_token_id,
    )
    model.train()

    stages = [int(stage.strip()) for stage in args.stages.split(",") if stage.strip()]
    example = gsm8k_natalia_example()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float | int]] = []

    for stage in stages:
        batch = make_batch(
            example,
            tokenizer,
            stage=stage,
            latent_tokens_per_step=args.latent_tokens_per_step,
            device=device,
        )
        for local_step in range(1, args.steps_per_stage + 1):
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            assert output.loss is not None
            if not torch.isfinite(output.loss):
                print(
                    f"warning: non-finite loss at global step {len(history) + 1}; "
                    "skipping update"
                )
                continue
            output.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
            if not torch.isfinite(grad_norm):
                print(
                    f"warning: non-finite grad norm at global step {len(history) + 1}; "
                    "skipping update"
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

            row = {
                "stage": stage,
                "local_step": local_step,
                "global_step": len(history) + 1,
                "loss": float(output.loss.detach().cpu()),
            }
            history.append(row)

            if local_step == 1 or local_step % args.log_every == 0:
                print(
                    f"stage={stage} step={local_step:04d} "
                    f"loss={row['loss']:.6f}"
                )

    final_stage = stages[-1]
    final_sample = build_coconut_sample(
        example,
        tokenizer,
        stage=final_stage,
        latent_tokens_per_step=args.latent_tokens_per_step,
    )
    prompt_ids = torch.tensor(
        [final_sample.input_ids[: final_sample.latent_end]],
        dtype=torch.long,
        device=device,
    )
    model.eval()
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    write_history(output_dir / "loss_curve.csv", history)
    write_loss_plot(output_dir / "loss_curve.png", history)
    (output_dir / "generated.txt").write_text(generated_text, encoding="utf-8")
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print(generated_text)
    print("=" * 80)
    print(f"Wrote outputs to {output_dir.resolve()}")


def choose_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def make_batch(
    example: ReasoningExample,
    tokenizer: AutoTokenizer,
    *,
    stage: int,
    latent_tokens_per_step: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sample = build_coconut_sample(
        example,
        tokenizer,
        stage=stage,
        latent_tokens_per_step=latent_tokens_per_step,
    )
    return {
        "input_ids": torch.tensor([sample.input_ids], dtype=torch.long, device=device),
        "labels": torch.tensor([sample.labels], dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            [sample.attention_mask],
            dtype=torch.long,
            device=device,
        ),
        "position_ids": torch.tensor(
            [sample.position_ids],
            dtype=torch.long,
            device=device,
        ),
    }


def gsm8k_natalia_example() -> ReasoningExample:
    return ReasoningExample(
        question=(
            "Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell altogether?"
        ),
        steps=[
            "Natalia sold 48 / 2 = 24 clips in May.",
            "Natalia sold 48 + 24 = 72 clips altogether.",
        ],
        answer="72",
    )


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "local_step", "global_step", "loss"],
        )
        writer.writeheader()
        writer.writerows(history)


def write_loss_plot(path: Path, history: list[dict[str, float | int]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping loss_curve.png")
        return

    steps = [int(row["global_step"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses)
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.title("Single-example Coconut overfit")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
