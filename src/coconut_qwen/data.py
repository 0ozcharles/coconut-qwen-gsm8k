"""Data construction utilities for Coconut-style training examples."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol


IGNORE_INDEX = -100
BOT_TOKEN = "<bot>"
LATENT_TOKEN = "<latent>"
EOT_TOKEN = "<eot>"
SPECIAL_TOKENS = [BOT_TOKEN, LATENT_TOKEN, EOT_TOKEN]


class CoconutTokenizer(Protocol):
    """Small tokenizer interface used by the data builder."""

    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...

    def convert_tokens_to_ids(self, token: str) -> int:
        ...


@dataclass(frozen=True)
class ReasoningExample:
    """One math-reasoning example with decomposed solution steps."""

    question: str
    steps: list[str]
    answer: str


@dataclass(frozen=True)
class CoconutSample:
    """Tokenized sample consumed by the Coconut training forward pass."""

    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    position_ids: list[int]
    latent_start: int
    latent_end: int
    skipped_steps: int
    latent_token_count: int


@dataclass(frozen=True)
class PlainCOTSample:
    """Tokenized plain chain-of-thought causal-LM sample."""

    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    position_ids: list[int]
    prompt_end: int


def build_coconut_sample(
    example: ReasoningExample,
    tokenizer: CoconutTokenizer,
    *,
    stage: int,
    latent_tokens_per_step: int = 1,
    max_latent_steps: int | None = None,
    include_remaining_steps: bool = True,
) -> CoconutSample:
    """Build one curriculum-stage Coconut training sample.

    `stage` means how many textual reasoning steps are replaced by latent
    thoughts. Stage 0 keeps all reasoning text. Stage 1 replaces the first CoT
    step with latent token(s), stage 2 replaces the first two steps, and so on.
    """

    if stage < 0:
        raise ValueError("stage must be non-negative")
    if latent_tokens_per_step < 1:
        raise ValueError("latent_tokens_per_step must be at least 1")

    max_stage = len(example.steps)
    if max_latent_steps is not None:
        max_stage = min(max_stage, max_latent_steps)

    skipped_steps = min(stage, max_stage)
    latent_token_count = skipped_steps * latent_tokens_per_step

    question_ids = tokenizer.encode(example.question.rstrip() + "\n", add_special_tokens=True)
    remaining_step_ids = []
    if include_remaining_steps:
        remaining_step_ids = [
            tokenizer.encode(step.rstrip() + "\n", add_special_tokens=False)
            for step in example.steps[skipped_steps:]
        ]
    answer_ids = tokenizer.encode("### " + example.answer.strip(), add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    bot_id = tokenizer.convert_tokens_to_ids(BOT_TOKEN)
    latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    eot_id = tokenizer.convert_tokens_to_ids(EOT_TOKEN)

    latent_segment = [bot_id] + [latent_id] * latent_token_count + [eot_id]
    supervised_ids = _flatten(remaining_step_ids) + answer_ids
    input_ids = question_ids + latent_segment + supervised_ids

    masked_prefix_len = len(question_ids) + len(latent_segment)
    labels = [IGNORE_INDEX] * masked_prefix_len + supervised_ids

    return CoconutSample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=[1] * len(input_ids),
        position_ids=list(range(len(input_ids))),
        latent_start=len(question_ids),
        latent_end=len(question_ids) + len(latent_segment),
        skipped_steps=skipped_steps,
        latent_token_count=latent_token_count,
    )


def build_plain_cot_sample(
    example: ReasoningExample,
    tokenizer: CoconutTokenizer,
) -> PlainCOTSample:
    """Build a plain CoT fine-tuning sample for the baseline."""

    question_ids = tokenizer.encode(example.question.rstrip() + "\n", add_special_tokens=True)
    reasoning_ids = _flatten(
        [
            tokenizer.encode(step.rstrip() + "\n", add_special_tokens=False)
            for step in example.steps
        ]
    )
    answer_ids = tokenizer.encode("### " + example.answer.strip(), add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    supervised_ids = reasoning_ids + answer_ids
    input_ids = question_ids + supervised_ids
    labels = [IGNORE_INDEX] * len(question_ids) + supervised_ids

    return PlainCOTSample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=[1] * len(input_ids),
        position_ids=list(range(len(input_ids))),
        prompt_end=len(question_ids),
    )


def parse_gsm8k_answer(answer_text: str) -> tuple[list[str], str]:
    """Split GSM8K's answer field into reasoning steps and final answer."""

    if "####" in answer_text:
        reasoning_text, final_answer = answer_text.rsplit("####", 1)
    else:
        reasoning_text = answer_text
        final_answer = extract_numeric_answer(answer_text) or answer_text.strip()

    steps = [
        _clean_gsm8k_step(line)
        for line in reasoning_text.splitlines()
        if _clean_gsm8k_step(line)
    ]
    if not steps and reasoning_text.strip():
        steps = [_clean_gsm8k_step(reasoning_text)]

    return steps, normalize_numeric_answer(final_answer)


def gsm8k_record_to_example(record: dict[str, str]) -> ReasoningExample:
    """Convert a HuggingFace GSM8K record into a `ReasoningExample`."""

    steps, answer = parse_gsm8k_answer(record["answer"])
    return ReasoningExample(
        question=record["question"],
        steps=steps,
        answer=answer,
    )


def extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from generated or ground-truth text."""

    text = text.replace("\\n", "\n").replace("\\r", "\r")
    if EOT_TOKEN in text:
        text = text.split(EOT_TOKEN, 1)[1]
    text = text.replace("<|im_end|>", " ")

    for marker in ["####", "###"]:
        if marker not in text:
            continue
        for segment in reversed(text.split(marker)[1:]):
            answer = _last_numeric_answer(segment)
            if answer is not None:
                return answer

    return _last_numeric_answer(text)


def normalize_numeric_answer(text: str) -> str:
    """Normalize numeric answer strings for exact-match evaluation."""

    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return text.strip()

    value = matches[-1].replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    if value == "-0":
        value = "0"
    return value


def _flatten(nested: list[list[int]]) -> list[int]:
    return [item for group in nested for item in group]


def _clean_gsm8k_step(step: str) -> str:
    cleaned = re.sub(r"<<[^<>]*>>", "", step.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _last_numeric_answer(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return normalize_numeric_answer(matches[-1])
