# Coconut Qwen3-0.6B GSM8K Reproduction

I implemented a focused reproduction of Coconut-style continuous latent
reasoning on GSM8K with `Qwen/Qwen3-0.6B`.

My main goal was to validate the core mechanism rather than chase
state-of-the-art GSM8K accuracy: add latent markers, replace selected
chain-of-thought steps with continuous hidden states, keep gradients through the
fed-back hidden states, and show that the resulting model can overfit a GSM8K
example.

## What I Implemented

- A Coconut wrapper for HuggingFace causal LMs in `src/coconut_qwen/model.py`.
- GSM8K preprocessing and curriculum sample construction in
  `src/coconut_qwen/data.py`.
- Special-token setup for `<bot>`, `<latent>`, and `<eot>` in
  `src/coconut_qwen/tokenizer.py`.
- A required single-example overfit script in `scripts/overfit_one.py`.
- A small-subset bonus experiment script in `scripts/bonus_gsm8k_subset.py`.
- A self-contained report in `report.html`.

The current Coconut wrapper supports batch size 1. I kept it simple so the
hidden-state feedback path is easy to inspect.

## Environment

I ran the experiments locally with:

- Windows
- NVIDIA GeForce RTX 3070 Laptop GPU, 8GB VRAM
- Python 3.11 in a conda environment
- PyTorch CUDA build
- `Qwen/Qwen3-0.6B`

Install dependencies:

```bash
pip install -r requirements.txt
```

I downloaded the model to:

```text
models/Qwen3-0.6B
```

The `models/` and `outputs/` directories are intentionally ignored by Git.

## Required Experiment

Run:

```bash
python scripts/overfit_one.py --model-name models/Qwen3-0.6B --output-dir outputs/overfit_one_default
```

Result:

```text
stage 2 final loss: 0.000145
generated: <bot><latent><latent><eot>### 72<|im_end|>
```

This proves that the single-example Coconut feedback loop, special tokens,
loss masking, and stage curriculum are wired correctly.

## Bonus Experiments

I also ran small-subset GSM8K checks to compare Coconut with a plain CoT
baseline. These were not meant to be full GSM8K training runs.

Useful command pattern:

```bash
python scripts/bonus_gsm8k_subset.py --local-files-only --model-name models/Qwen3-0.6B --train-size 32 --eval-size 32 --eval-source train --mode both
```

Selected results:

| Experiment | Result |
| --- | ---: |
| 2-example train-set sanity, Coconut | 2/2 |
| 2-example train-set sanity, plain CoT | 2/2 |
| 32-example train-set, plain CoT | 32/32 |
| 32-example Coconut, remaining-CoT target | 8/32 |
| 32-example Coconut, answer-only target | 9/32 |

The small-subset experiments showed that plain CoT memorizes the tiny training
set much more easily than Coconut. Coconut remained harder to optimize under
full-parameter fp16 fine-tuning on an 8GB GPU.

## Notes on Limitations

The required single-example overfit worked cleanly. The bonus experiments were
more difficult:

- Later latent stages had noticeably higher and more volatile loss.
- Long fp16 full-parameter training sometimes produced non-finite gradients.
- Lower learning rates, gradient clipping, cleaner GSM8K preprocessing, local
  dataset loading, and answer-only ablations improved stability but did not make
  Coconut match the plain CoT baseline on the 32-example memorization check.

I describe these issues and the fixes I tried in `report.html`.
