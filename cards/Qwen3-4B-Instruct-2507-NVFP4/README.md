---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
base_model_relation: quantized
tags:
  - nvfp4
  - fp4
  - compressed-tensors
  - llm-compressor
  - vllm
  - qwen3
language:
  - en
---

# Qwen3-4B-Instruct-2507 · NVFP4

An NVFP4 quantization (FP4 weights and FP4 activations, group size 16, FP8 block scales) of [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), revision `cdbee75f`. All Linear layers are quantized; `lm_head` stays BF16. Produced with llm-compressor 0.12.0 in compressed-tensors format. 2.82 GB on disk against 8.05 GB for BF16.

Made in a one-afternoon pre-registered study: [github.com/Nikhiljangra07/quant-lab](https://github.com/Nikhiljangra07/quant-lab). Read the status line below before using it.

## Status: exploratory GO

The study's frozen protocol pinned the vLLM backend for this checkpoint to `flashinfer_b12x`, which was not available on the pod image (its nvcc 12.8 cannot build FlashInfer kernels for SM120). Under the protocol this checkpoint is therefore **NO-RESULT**. The numbers below come from one exploratory attempt that served the same checkpoint through vLLM's Marlin FP4 kernel instead. Same base snapshot, calibration set, pinned task files, seeds and serving settings; chosen after the failure, so not a pre-registered claim.

## Quality

lm-evaluation-harness 0.4.13 on vLLM 0.25.0, full test sets, chat template, greedy, seed 42. Bar: within 2.0 points of BF16 on both tasks.

| | GSM8K 5-shot, strict-match (n=1,319) | IFEval prompt-level strict (n=541) |
|---|---:|---:|
| BF16 reference | 80.52% | 82.26% |
| **NVFP4 (this checkpoint)** | **78.62%** (−1.9) | **82.26%** (0.0) |
| W4A16 INT4 g128, all Linear (same study) | 72.55% (−8.0) | 81.52% (−0.7) |

On this 4B model NVFP4 held the bar that INT4 g128 missed by eight points; the study's follow-ups attribute the INT4 loss to the attention projections.

## Throughput

One RTX PRO 6000 Blackwell (96 GB), vLLM 0.25.0, Marlin FP4 kernel, 1024-token input / 1024-token output, prefix caching off, single run per cell (BF16 row is a median of three).

| concurrency | BF16 tok/s | NVFP4 tok/s |
|---|---:|---:|
| 1 | 144 | 262 |
| 16 | 1,622 | 2,436 |
| 64 | 3,486 | 3,957 |

## Serving

```bash
vllm serve Nikhil0097/Qwen3-4B-Instruct-2507-NVFP4 --dtype bfloat16
```

On SM120 (RTX PRO 6000 / RTX 50-series) with a vLLM build whose FlashInfer JIT cannot target the card, add `--linear-backend marlin`; that is how the numbers above were produced. On SM100 (B200/B300) vLLM's default kernel selection should apply.

## Recipe

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
oneshot(model=model, dataset=calibration, recipe=recipe, max_seq_length=2048, num_calibration_samples=512)
```

Calibration: 512 chat-formatted conversations from `HuggingFaceH4/ultrachat_200k` (train_sft, revision `8049631c`, seeded shuffle), each capped at 2048 tokens, after rejecting any conversation sharing an 8-word shingle with a GSM8K test/few-shot question or an IFEval prompt (2 rejected).

## Caveats

- Single attempt; no repeats of the quantization or the eval.
- The −2.0 point bar is an acceptance rule from the study, not an equivalence test.
- Not tested: long context, multilingual, tool use, anything beyond the two evals above.
- Served with FP4 activations; results may differ on kernels other than Marlin.

Base model license: Apache-2.0. This checkpoint: Apache-2.0.
