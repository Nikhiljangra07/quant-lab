---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
base_model_relation: quantized
tags:
  - w4a16
  - gptq
  - int4
  - compressed-tensors
  - llm-compressor
  - vllm
  - qwen3
language:
  - en
---

# Qwen3-4B-Instruct-2507 · W4A16, MLP only

A GPTQ W4A16 quantization (INT4 weights, group size 128, symmetric; BF16 activations) of [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), revision `cdbee75f`, applied to the **MLP projections only**. The attention projections (q/k/v/o) and `lm_head` stay BF16. Produced with llm-compressor 0.12.0 in compressed-tensors format. 4.05 GB on disk against 8.05 GB for BF16.

Made in a one-afternoon pre-registered study: [github.com/Nikhiljangra07/quant-lab](https://github.com/Nikhiljangra07/quant-lab). The study's pre-registered recipe quantized every Linear layer and **failed** its quality bar; this checkpoint is the follow-up that explains why.

## Status: exploratory GO

Chosen after seeing the protocol result, so a single-attempt hypothesis rather than a pre-registered claim. Same base snapshot, calibration set, pinned task files, seeds and serving settings as the protocol run.

## Quality

lm-evaluation-harness 0.4.13 on vLLM 0.25.0, full test sets, chat template, greedy, seed 42. Bar: within 2.0 points of BF16 on both tasks.

| | GSM8K 5-shot, strict-match (n=1,319) | IFEval prompt-level strict (n=541) |
|---|---:|---:|
| BF16 reference | 80.52% | 82.26% |
| **W4A16 MLP-only (this checkpoint)** | **82.56%** (+2.0) | **82.44%** (+0.2) |
| W4A16 g128, all Linear (protocol recipe) | 72.55% (−8.0) | 81.52% (−0.7) |
| W4A16 g32, all Linear | 76.35% (−4.2) | 80.59% (−1.7) |

The GSM8K point estimate is above BF16, which is within the ±1.1 point standard error, so read it as "no measurable loss", not as an improvement. The eight-point loss in the all-Linear recipe comes from quantizing this model's attention projections to INT4; finer groups halve it, leaving attention in BF16 removes it.

## Throughput

One RTX PRO 6000 Blackwell (96 GB), vLLM 0.25.0, 1024-token input / 1024-token output, prefix caching off, single run per cell (BF16 and all-Linear rows are medians of three).

| concurrency | BF16 tok/s | all-Linear W4A16 | **MLP-only W4A16** |
|---|---:|---:|---:|
| 1 | 144 | 283 | 230 |
| 16 | 1,622 | 2,530 | 2,218 |
| 64 | 3,486 | 3,941 | 3,854 |

Keeping attention in BF16 gives back about a fifth of the single-stream speedup and a third of the size saving, in exchange for holding the quality bar.

## Serving

```bash
vllm serve Nikhil0097/Qwen3-4B-Instruct-2507-W4A16-MLP --dtype bfloat16
```

## Recipe

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
recipe = GPTQModifier(targets="Linear", scheme="W4A16", dampening_frac=0.01,
                      ignore=["lm_head", "re:.*self_attn.*"])
oneshot(model=model, dataset=calibration, recipe=recipe, max_seq_length=2048, num_calibration_samples=512)
```

Calibration: 512 chat-formatted conversations from `HuggingFaceH4/ultrachat_200k` (train_sft, revision `8049631c`, seeded shuffle), each capped at 2048 tokens, after rejecting any conversation sharing an 8-word shingle with a GSM8K test/few-shot question or an IFEval prompt (2 rejected).

## Caveats

- Single attempt; no repeats of the quantization or the eval.
- The −2.0 point bar is an acceptance rule from the study, not an equivalence test.
- Not tested: long context, multilingual, tool use, anything beyond the two evals above.

Base model license: Apache-2.0. This checkpoint: Apache-2.0.
