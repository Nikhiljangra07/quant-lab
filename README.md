# quant-lab

A one-afternoon, pre-registered quantization study of **Qwen3-4B-Instruct-2507** on a single RTX PRO 6000 Blackwell (96 GB, SM120), run on RunPod for about $5. Question: do W4A16 and NVFP4 quants of this model hold the BF16 reference on a reasoning eval and an instruction-following eval, and what do they buy in vLLM throughput?

The published record includes the frozen protocol, the go/no-go bars written before any GPU minute, the run that failed, the amendment that fixed it, the pre-registered result (a NO-GO), and an exploratory series investigating it. Solo work by Nikhil Jangra, October 7, 2026.

## The pre-registered result

| arm | GSM8K strict-match | IFEval prompt-strict | weights | verdict |
|---|---:|---:|---:|---|
| BF16 reference | 80.52% | 82.26% | 8.05 GB | reference |
| W4A16 GPTQ g128, all Linear (lm_head BF16) | 72.55% (−8.0) | 81.52% (−0.7) | 2.65 GB | **NO-GO** |
| NVFP4 (W4A4 g16), `flashinfer_b12x` backend | — | — | 2.82 GB | **NO-RESULT** (kernel unavailable on the pod image, see log) |

Bars, written in [RUNBOOK.md](RUNBOOK.md) before spend: each quant within 2.0 percentage points of BF16 on both tasks. Full test sets (GSM8K 1,319, IFEval 541), greedy, seed 42, chat template, pinned lm-eval 0.4.13 task files and dataset revisions, calibration set audited for 8-word overlap with both evals (2 of 514 candidates rejected). Paired per-sample differences, generation-cap counts and descriptive intervals: [results/runs/20261007T075434Z-3719/RESULTS.md](results/runs/20261007T075434Z-3719/RESULTS.md).

Throughput (vLLM 0.25.0, 1024 in / 1024 out, prefix caching off, medians of three repeats within 0.3%):

| arm | c=1 tok/s | c=16 | c=64 |
|---|---:|---:|---:|
| BF16 | 144 | 1,622 | 3,486 |
| W4A16 | 283 | 2,530 | 3,941 |

The W4A16 recipe doubles single-stream decode and cuts the checkpoint to a third. It still fails the quality bar, so under this protocol it does not ship.

## The exploratory series (outside the freeze)

Chosen after seeing the result, so reported as hypotheses, single attempts each. Same base snapshot, calibration selection, task files, seeds and serving settings. Full table: [RESULTS-EXPLORATORY.md](results/runs/20261007T075434Z-3719/RESULTS-EXPLORATORY.md).

| recipe | GSM8K | IFEval | weights | c=1 tok/s | verdict |
|---|---:|---:|---:|---:|---|
| W4A16 g32, all Linear | 76.35% (−4.2) | 80.59% (−1.7) | 2.82 GB | — | NO-GO |
| W4A16 g128, **MLP only** (attention BF16) | 82.56% (+2.0) | 82.44% (+0.2) | 4.05 GB | 230 | GO (exploratory) |
| NVFP4, **Marlin kernel** | 78.62% (−1.9) | 82.26% (0.0) | 2.82 GB | 262 | GO (exploratory) |
| FP8 dynamic | 81.50% (+1.0) | 82.07% (−0.2) | 4.41 GB | — | GO (exploratory) |
| INT8 W8A8 | — | — | 4.41 GB | — | unservable on SM120 in vLLM 0.25.0 |

In these exploratory runs, finer INT4 groups reduced the GSM8K strict-match deficit, and keeping attention q/k/v/o in BF16 removed it in the point estimate. This suggests sensitivity to the quantization recipe; it does not establish a single underlying cause. The MLP-only recipe measured 230 tok/s versus 144 for BF16 (about 1.60×), with a 4.05 GB checkpoint; all-Linear W4A16 measured 283 tok/s (about 1.97×) at 2.65 GB. Exploratory throughput rows are single measurements, while protocol throughput rows are medians of three.

GSM8K strict-match also depends on the `#### N` answer format: a lenient-extraction diagnostic reduced the protocol W4A16 gap to 1.1 points. This motivates a format-compliance hypothesis; it does not replace the registered strict-match result. NVFP4/Marlin and MLP-only INT4 cleared the specified point-estimate bars in this exploratory series. Passing those bars is not an equivalence test or a general deployment recommendation.

## What broke, and what was done about it

The engineering log is [LOG.md](LOG.md). Short version:

1. **FlashInfer vs the pod image.** The RunPod Ubuntu 24.04 image ships nvcc 12.8; FlashInfer 0.6.13 needs CUDA 12.9+ to target SM120, so every FlashInfer JIT path refused the card. This killed run 1 at the BF16 serving smoke (vLLM's default FlashInfer sampler), the pre-registered NVFP4 backend (`flashinfer_b12x` is only registered when that JIT works), and the auto-selected FlashInfer CUTLASS NVFP4 kernel. Amendment A1 switched vLLM to its native sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`); the exploratory NVFP4 runs forced the Marlin kernel, which needs no JIT. Run 1 is retained as NO-RESULT.
2. **INT8 on Blackwell.** vLLM's CUTLASS W8A8 INT8 path raises `Int8 not supported on SM120`; FP8 is the 8-bit option on this card.
3. **An operator mistake.** One exploratory bench was queued with the wrong wait marker and started on top of a running server; killed within a minute, partial output removed, logged.

## Reproduce

Protocol: [RUNBOOK.md](RUNBOOK.md) (v2; the pre-spend v1 is kept as [RUNBOOK.v1.md](RUNBOOK.v1.md)), settings in [protocol.json](protocol.json), hash-locked environments in [requirements/](requirements/), CPU-only regression tests in [tests/](tests/) (`python3 -m unittest discover -s tests`).

Rent one SM120 GPU on a Linux x86_64 host with Python 3.12 and an NVIDIA driver ≥ 580 (CUDA 13 wheels), with a 100 GB workspace. Upload the kit to `/workspace/quant-lab` (no `.git`, `results/`, `venv_*`, `cache/`) and run, in a supervised session:

```bash
bash scripts/run_all.sh --supervised --rate <USD per hour> --pod-started '<ISO time with zone>' --with-nvfp4
```

The runner stops compute at the budget deadline; it cannot stop the rental. Copy `results/runs/<id>/` off the pod, then stop the pod yourself. If your image's nvcc is below 12.9, expect the FlashInfer paths above to fail the same way; the A1 env var is already in the runner.

Exploratory: [explore/](explore/) holds `explore_quant.py`, the `run_explore*.sh` sequence exactly as run, and `summarize_explore.py`.

Every artifact under `results/runs/<id>/` is what the pod produced: stage logs, per-sample scores with document and prompt hashes, generation telemetry, benchmark JSON, `pip freeze`, `nvidia-smi`, the frozen source copy and its hashes. Checkpoints are not in git (sizes in the tables). The two exploratory GO candidates are on Hugging Face with their cards: [Nikhil0097/Qwen3-4B-Instruct-2507-NVFP4](https://huggingface.co/Nikhil0097/Qwen3-4B-Instruct-2507-NVFP4) and [Nikhil0097/Qwen3-4B-Instruct-2507-W4A16-MLP](https://huggingface.co/Nikhil0097/Qwen3-4B-Instruct-2507-W4A16-MLP).

## Credits and tooling

Protocol v2 was drafted with Codex and verified against the pinned package sources before the first GPU minute; the pod runs, amendments and write-up were driven with Claude Code. Design choices, the go/no-go bars and the interpretation are mine. Base model: Qwen3-4B-Instruct-2507 (Apache-2.0). Quantization: llm-compressor 0.12.0. Serving and benchmarks: vLLM 0.25.0. Evals: lm-evaluation-harness 0.4.13.
