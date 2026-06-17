"""Small-subset GSM8K bonus experiment for Coconut vs. plain CoT."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import torch
from datasets import DownloadConfig, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coconut_qwen.data import (  # noqa: E402
    ReasoningExample,
    build_coconut_sample,
    build_plain_cot_sample,
    extract_numeric_answer,
    gsm8k_record_to_example,
)
from coconut_qwen.model import CoconutForCausalLM  # noqa: E402
from coconut_qwen.tokenizer import add_coconut_special_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="models/Qwen3-0.6B")
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load HuggingFace datasets from the local cache without Hub checks.",
    )
    parser.add_argument("--output-dir", default="outputs/bonus_gsm8k_subset")
    parser.add_argument("--mode", choices=["coconut", "cot", "both"], default="both")
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--eval-size", type=int, default=16)
    parser.add_argument("--eval-source", choices=["test", "train"], default="test")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Use the first N GSM8K records instead of a seed-shuffled subset.",
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--stages", default="0,1,2")
    parser.add_argument("--epochs-per-stage", type=int, default=1)
    parser.add_argument(
        "--stage-epochs",
        default=None,
        help="Comma-separated Coconut epochs per stage, e.g. 8,8,12.",
    )
    parser.add_argument("--cot-epochs", type=int, default=3)
    parser.add_argument("--latent-tokens-per-step", type=int, default=1)
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="For Coconut, supervise only the final answer after the latent segment.",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-bad-updates", type=int, default=20)
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Freeze input/output embedding tables after adding Coconut tokens.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
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
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    train_examples, eval_examples = load_gsm8k_examples(args)
    summary: dict[str, object] = {
        "train_size": len(train_examples),
        "eval_size": len(eval_examples),
        "eval_source": args.eval_source,
        "shuffled": not args.no_shuffle,
    }

    if args.mode in {"coconut", "both"}:
        coconut_summary = run_coconut(args, train_examples, eval_examples, output_dir)
        summary["coconut"] = coconut_summary
        cleanup_cuda()

    if args.mode in {"cot", "both"}:
        cot_summary = run_cot(args, train_examples, eval_examples, output_dir)
        summary["cot"] = cot_summary
        cleanup_cuda()

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def load_gsm8k_examples(args: argparse.Namespace) -> tuple[list[ReasoningExample], list[ReasoningExample]]:
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        download_config=DownloadConfig(local_files_only=args.local_files_only),
    )
    train_split = dataset["train"]
    if not args.no_shuffle:
        train_split = train_split.shuffle(seed=args.seed)

    train_records = train_split.select(range(args.train_size))
    if args.eval_source == "train":
        eval_dataset = train_split
    elif args.no_shuffle:
        eval_dataset = dataset["test"]
    else:
        eval_dataset = dataset["test"].shuffle(seed=args.seed + 1)

    eval_records = eval_dataset.select(range(args.eval_size))
    train_examples = [gsm8k_record_to_example(record) for record in train_records]
    eval_examples = [gsm8k_record_to_example(record) for record in eval_records]
    return train_examples, eval_examples


def run_coconut(
    args: argparse.Namespace,
    train_examples: list[ReasoningExample],
    eval_examples: list[ReasoningExample],
    output_dir: Path,
) -> dict[str, float | int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, base_model = load_model_and_tokenizer(args, device)
    token_ids = add_coconut_special_tokens(tokenizer, base_model)
    if args.freeze_embeddings:
        freeze_embeddings(base_model)
    model = CoconutForCausalLM(
        base_model,
        latent_token_id=token_ids["latent"],
        eos_token_id=tokenizer.eos_token_id,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.lr,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    stages = parse_stages(args.stages)
    stage_epochs = parse_stage_epochs(args, stages)
    history: list[dict[str, float | int | str]] = []
    global_step = 0
    attempted_step = 0
    bad_updates = 0

    model.train()
    for stage, epochs in zip(stages, stage_epochs):
        stage_examples = [
            example for example in train_examples
            if coconut_len(example, tokenizer, stage, args) <= args.max_seq_len
        ]
        for epoch in range(1, epochs + 1):
            for example_idx, example in enumerate(stage_examples):
                attempted_step += 1
                batch = coconut_batch(example, tokenizer, stage, args, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(**batch)
                assert output.loss is not None
                if not torch.isfinite(output.loss):
                    print(
                        f"warning: non-finite coconut loss at attempted step {attempted_step}; "
                        "skipping update"
                    )
                    bad_updates += 1
                    check_bad_updates(bad_updates, args.max_bad_updates)
                    continue
                output.loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
                if not torch.isfinite(grad_norm):
                    print(
                        f"warning: non-finite coconut grad norm at attempted step {attempted_step}; "
                        "skipping update"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    bad_updates += 1
                    check_bad_updates(bad_updates, args.max_bad_updates)
                    continue
                optimizer.step()
                bad_updates = 0

                global_step += 1
                loss = float(output.loss.detach().cpu())
                history.append(
                    {
                        "mode": "coconut",
                        "stage": stage,
                        "epoch": epoch,
                        "example_idx": example_idx,
                        "global_step": global_step,
                        "loss": loss,
                    }
                )
                if global_step == 1 or global_step % args.log_every == 0:
                    print(f"coconut step={global_step:04d} stage={stage} loss={loss:.6f}")

    write_training_history(output_dir / "coconut_loss.csv", history)
    predictions = evaluate_coconut(model, tokenizer, eval_examples, args, device)
    write_predictions(output_dir / "coconut_predictions.csv", predictions)
    accuracy = mean([row["correct"] for row in predictions])
    return {
        "accuracy": accuracy,
        "correct": int(sum(row["correct"] for row in predictions)),
        "total": len(predictions),
        "train_steps": global_step,
    }


def run_cot(
    args: argparse.Namespace,
    train_examples: list[ReasoningExample],
    eval_examples: list[ReasoningExample],
    output_dir: Path,
) -> dict[str, float | int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_and_tokenizer(args, device)
    if args.freeze_embeddings:
        freeze_embeddings(model)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.lr,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float | int | str]] = []
    global_step = 0
    attempted_step = 0
    bad_updates = 0

    model.train()
    stage_examples = [
        example for example in train_examples
        if cot_len(example, tokenizer) <= args.max_seq_len
    ]
    for epoch in range(1, args.cot_epochs + 1):
        for example_idx, example in enumerate(stage_examples):
            attempted_step += 1
            batch = cot_batch(example, tokenizer, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            if not torch.isfinite(loss):
                print(
                    f"warning: non-finite cot loss at attempted step {attempted_step}; "
                    "skipping update"
                )
                bad_updates += 1
                check_bad_updates(bad_updates, args.max_bad_updates)
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
            if not torch.isfinite(grad_norm):
                print(
                    f"warning: non-finite cot grad norm at attempted step {attempted_step}; "
                    "skipping update"
                )
                optimizer.zero_grad(set_to_none=True)
                bad_updates += 1
                check_bad_updates(bad_updates, args.max_bad_updates)
                continue
            optimizer.step()
            bad_updates = 0

            global_step += 1
            loss_value = float(loss.detach().cpu())
            history.append(
                {
                    "mode": "cot",
                    "stage": -1,
                    "epoch": epoch,
                    "example_idx": example_idx,
                    "global_step": global_step,
                    "loss": loss_value,
                }
            )
            if global_step == 1 or global_step % args.log_every == 0:
                print(f"cot step={global_step:04d} loss={loss_value:.6f}")

    write_training_history(output_dir / "cot_loss.csv", history)
    predictions = evaluate_cot(model, tokenizer, eval_examples, args, device)
    write_predictions(output_dir / "cot_predictions.csv", predictions)
    accuracy = mean([row["correct"] for row in predictions])
    return {
        "accuracy": accuracy,
        "correct": int(sum(row["correct"] for row in predictions)),
        "total": len(predictions),
        "train_steps": global_step,
    }


def load_model_and_tokenizer(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    dtype = choose_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    return tokenizer, model


def freeze_embeddings(model: AutoModelForCausalLM) -> None:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_embeddings.weight.requires_grad_(False)
    if output_embeddings is not None:
        output_embeddings.weight.requires_grad_(False)


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def evaluate_coconut(
    model: CoconutForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[ReasoningExample],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    final_stage = parse_stages(args.stages)[-1]
    rows: list[dict[str, object]] = []
    for idx, example in enumerate(tqdm(examples, desc="eval coconut")):
        sample = build_coconut_sample(
            example,
            tokenizer,
            stage=final_stage,
            latent_tokens_per_step=args.latent_tokens_per_step,
            include_remaining_steps=not args.answer_only,
        )
        if len(sample.input_ids) > args.max_seq_len:
            continue
        prompt_ids = torch.tensor(
            [sample.input_ids[: sample.latent_end]],
            dtype=torch.long,
            device=device,
        )
        generated_ids = model.generate(prompt_ids, max_new_tokens=args.max_new_tokens)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        pred = extract_numeric_answer(generated_text)
        rows.append(prediction_row(idx, example, pred, generated_text))
    return rows


def evaluate_cot(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[ReasoningExample],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    for idx, example in enumerate(tqdm(examples, desc="eval cot")):
        prompt_ids = tokenizer.encode(
            example.question.rstrip() + "\n",
            add_special_tokens=True,
        )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        pred = extract_numeric_answer(generated_text)
        rows.append(prediction_row(idx, example, pred, generated_text))
    return rows


def coconut_batch(
    example: ReasoningExample,
    tokenizer: AutoTokenizer,
    stage: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sample = build_coconut_sample(
        example,
        tokenizer,
        stage=stage,
        latent_tokens_per_step=args.latent_tokens_per_step,
        include_remaining_steps=not args.answer_only,
    )
    return tensor_batch(
        sample.input_ids,
        sample.labels,
        sample.attention_mask,
        sample.position_ids,
        device,
    )


def cot_batch(
    example: ReasoningExample,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sample = build_plain_cot_sample(example, tokenizer)
    return tensor_batch(
        sample.input_ids,
        sample.labels,
        sample.attention_mask,
        sample.position_ids,
        device,
    )


def tensor_batch(
    input_ids: list[int],
    labels: list[int],
    attention_mask: list[int],
    position_ids: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=device),
        "position_ids": torch.tensor([position_ids], dtype=torch.long, device=device),
    }


def prediction_row(
    idx: int,
    example: ReasoningExample,
    pred: str | None,
    generated_text: str,
) -> dict[str, object]:
    correct = int(pred == example.answer)
    return {
        "idx": idx,
        "gold": example.answer,
        "pred": pred or "",
        "correct": correct,
        "question": example.question,
        "generated": generated_text.replace("\r", "\\r").replace("\n", "\\n"),
    }


def write_training_history(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mode", "stage", "epoch", "example_idx", "global_step", "loss"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["idx", "gold", "pred", "correct", "question", "generated"],
        )
        writer.writeheader()
        writer.writerows(rows)


def coconut_len(
    example: ReasoningExample,
    tokenizer: AutoTokenizer,
    stage: int,
    args: argparse.Namespace,
) -> int:
    return len(
        build_coconut_sample(
            example,
            tokenizer,
            stage=stage,
            latent_tokens_per_step=args.latent_tokens_per_step,
            include_remaining_steps=not args.answer_only,
        ).input_ids
    )


def cot_len(example: ReasoningExample, tokenizer: AutoTokenizer) -> int:
    return len(build_plain_cot_sample(example, tokenizer).input_ids)


def parse_stages(stages: str) -> list[int]:
    parsed = [int(stage.strip()) for stage in stages.split(",") if stage.strip()]
    if not parsed:
        raise ValueError("at least one stage is required")
    return parsed


def parse_stage_epochs(args: argparse.Namespace, stages: list[int]) -> list[int]:
    if args.stage_epochs is None:
        return [args.epochs_per_stage] * len(stages)

    parsed = [
        int(value.strip())
        for value in args.stage_epochs.split(",")
        if value.strip()
    ]
    if len(parsed) != len(stages):
        raise ValueError(
            "--stage-epochs must have the same number of entries as --stages"
        )
    if any(value < 1 for value in parsed):
        raise ValueError("--stage-epochs values must be positive")
    return parsed


def check_bad_updates(bad_updates: int, max_bad_updates: int) -> None:
    if bad_updates >= max_bad_updates:
        raise RuntimeError(
            f"stopping after {bad_updates} consecutive non-finite updates; "
            "lower --lr, lower --grad-clip, or reduce stage epochs"
        )


def choose_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
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


def mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
