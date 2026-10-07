# Exploratory results (outside the pre-registered freeze)

Run `20261007T075434Z-3719`. Protocol rows are the frozen result; exploratory rows are single attempts chosen after seeing the protocol outcome and are reported as hypotheses, not pre-registered claims. Same base snapshot, calibration selection, task files, seeds and serving settings. Bars: within 2.0 pp of BF16 on both tasks.

| recipe | status | GSM8K strict | Δ | IFEval prompt-strict | Δ | weights GB | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| BF16 (reference) | protocol | 80.52% | +0.0 | 82.26% | +0.0 | 8.04 | reference |
| W4A16 GPTQ g128, all Linear | protocol | 72.55% | -8.0 | 81.52% | -0.7 | 2.65 | NO-GO |
| W4A16 GPTQ g32, all Linear | exploratory | 76.35% | -4.2 | 80.59% | -1.7 | 2.82 | NO-GO (exploratory) |
| W4A16 GPTQ g128, MLP only (attention BF16) | exploratory | 82.56% | +2.0 | 82.44% | +0.2 | 4.05 | GO (exploratory) |
| NVFP4 (W4A4 g16), Marlin kernel | exploratory | 78.62% | -1.9 | 82.26% | +0.0 | 2.82 | GO (exploratory) |
| FP8 dynamic (W8A8 float) | exploratory | 81.50% | +1.0 | 82.07% | -0.2 | 4.41 | GO (exploratory) |
| INT8 W8A8 | exploratory | — | — | — | — | 4.41 | unservable: vLLM CUTLASS INT8 not supported on SM120 |

## Throughput (output tok/s, 1024 in / 1024 out, prefix caching off; protocol rows are medians of 3, exploratory rows single runs)

| recipe | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| BF16 (reference) | 144 | 1622 | 3486 |
| W4A16 GPTQ g128, all Linear | 283 | 2530 | 3941 |
| W4A16 GPTQ g128, MLP only (attention BF16) | 230 | 2218 | 3854 |
| NVFP4 (W4A4 g16), Marlin kernel | 262 | 2436 | 3957 |

## What the series says

- The protocol W4A16 recipe (INT4 g128 on every Linear layer) fails the GSM8K bar by 8 points on this 4B model; IFEval barely moves.
- Quantizing only the MLP projections and leaving attention q/k/v/o in BF16 removes the loss entirely (GSM8K above the BF16 point estimate, within noise) at 4.05 GB instead of 2.65 GB and 1.59× instead of 1.97× single-stream.
- Finer INT4 groups (g32) halve the loss but do not clear the bar.
- NVFP4 (FP4 weights and activations, group 16) clears both bars at 2.82 GB once served through the Marlin kernel; the FlashInfer kernels vLLM prefers need a JIT this pod image's nvcc 12.8 cannot build for SM120.
- FP8 dynamic clears both bars at 4.41 GB with no calibration pass.
- INT8 W8A8 cannot be served on this card in vLLM 0.25.0 (`Int8 not supported on SM120`).

Caveats: single attempts; GSM8K strict-match depends on the harness's `#### N` answer format (flexible-extract gaps are smaller); the −2.0 pp bar is an acceptance rule, not an equivalence test.
