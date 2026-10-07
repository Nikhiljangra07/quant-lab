# quant-lab — protocol v2, prepared October 7, 2026

Question: how do GPTQ W4A16 and optional NVFP4 of Qwen3-4B-Instruct-2507 compare with a BF16 reference on reasoning, instruction following, checkpoint size and vLLM serving performance on one 96GB RTX PRO 6000 Blackwell?

The original pre-spend plan is retained in RUNBOOK.v1.md. No GPU results existed when this amendment was prepared. Scientific settings live in protocol.json. Once a run begins, its source files, hashes and settings are copied into its own directory. A crash fix requires an explicit logged amendment and a fresh attempt; do not overwrite the previous attempt or claim it was the original protocol.

## Arms and scope

BF16 reference and GPTQ W4A16 are required. NVFP4 is optional and runs last, only when requested and at least 35 minutes remain. Optional failure is NO-RESULT; it does not erase completed required-arm evidence. A serving smoke precedes full evaluation of each arm.

Use exactly one full 96GB SM120 RTX PRO 6000 Blackwell. A dense 4B model is an approachable tooling exercise, not a reproduction of CQL's multi-GPU MoE or MTP work. No long-context benchmark, speculative decoding or KV-cache quantization is included. Performance can be worse after quantization.

## Frozen choices

| Item | Choice |
|---|---|
| Base | Qwen/Qwen3-4B-Instruct-2507 at cdbee75f17c01a7cc42f958dc650907174af0554; one local snapshot for all arms |
| Calibration | 512 seeded UltraChat train_sft conversations, each capped at 2048 tokens; save exact IDs, text, token IDs, lengths and SHA256 |
| Calibration audit | Reject any selected conversation sharing a normalized 8-word shingle with any GSM8K test/fewshot question or IFEval prompt; save rejected indices. No semantic/paraphrase contamination claim |
| W4A16 | llm-compressor GPTQ, all Linear layers except lm_head; symmetric INT4, group 128, damping .01; activations and lm_head BF16 |
| NVFP4 | QuantizationModifier NVFP4, all Linear except lm_head; FP4 weights/activations, group 16; explicit flashinfer_b12x serving backend on SM120 |
| Evals | Pinned lm_eval 0.4.13 task code plus pinned dataset revisions; GSM8K 5-shot strict-match, IFEval zero-shot prompt-strict; full test splits; chat templates; greedy, seed 42 |
| Output caps | GSM8K 1024, IFEval 1280. Count generation-cap stops; score all examples. Input truncation invalidates the evaluation |
| Serving | BF16 activation dtype, max_model_len 4096, GPU memory utilization .85, prefix caching disabled, same seed |
| Throughput | Fixed 1024-token input/output random workload, EOS ignored, greedy; c1/c16/c64, 16/128/256 requests respectively; 4 warmups, 3 repeats per cell |
| Evidence | Raw results, per-example scores and prompt/document hashes; length-stop telemetry; full server/stage logs; request completion and actual token counts; size/memory and environment records |

Custom tasks are named quant_gsm8k and quant_ifeval. The installed task files are copied and hashed before evaluation; only task name, dataset revision and explicit generation caps are amended. Reasoning eval strict-match depends on the harness's #### answer format; the raw harness result also preserves flexible-extract for diagnosis.

## Decision rule

A quant is GO only if both complete, paired full-set point estimates are no more than 2.0 percentage points below BF16. Scores, sample counts, example/prompt hashes, calibration provenance and generation telemetry must validate. Missing or mismatched evidence is NO-RESULT. Failure of a complete comparison is NO-GO.

This is an acceptance rule, not proof of equivalence. Report paired differences, disagreement counts and descriptive normal 95% intervals. This interval has a normal-approximation limitation, particularly for very rare differences; do not use it as a formal equivalence test. Keep generation-cap stops in the denominator.

Benchmark results are descriptive, separate from the quality rule. Report median and range across repeats plus raw p50/p95/p99 latencies. Reject any cell with failed requests or actual output lengths different from 1024. A partial repetition series is NO-RESULT, not a fabricated average. vLLM's reserved GPU memory is not the same as weight memory: report weight bytes separately and preserve server memory snapshots.

## Environments

Two Python 3.12 virtualenvs isolate the different Torch builds. Complete Linux x86_64 runtime locks, with package hashes, are in requirements/quant.lock and requirements/serve.lock. Quantization: llmcompressor 0.12.0, Torch 2.12.0, compressed-tensors 0.17.1. Serving: vLLM 0.25.0, Torch 2.11.0, compressed-tensors 0.17.0. Both use Transformers 5.10.1 and datasets 5.0.0. CUDA 13 wheels require a compatible NVIDIA host driver (580+).

The minor checkpoint-library patch difference is tested by loading each saved checkpoint before full evaluation. Actual GPU compatibility is not established by local dependency resolution; preflight and model smoke tests are required. No automatic source build of GPU packages and no silent upgrades/fallbacks. Some small pure-Python dependencies only ship source archives; their archives are hash-locked, while isolated build tools are supplied by pip. Setup saves complete installer output, pip check, pip freeze and BF16 CUDA checks.

## Spending and stopping

Budget remains $8 USD, including a $1 reserve. Supply the actual hourly GPU rate and the original pod rental start timestamp. The compute deadline accounts for time already spent on the rental. Setup has a 45-minute cutoff within that deadline. Every required stage fails closed; a timeout interrupts its entire process group, including server workers.

**This is a supervised run. The runner stops processes; it cannot stop Runpod rental or storage billing.** The operator must copy results and stop the pod in Runpod immediately at completion/failure, or before the allowance expires. Keep a console timer based on the displayed rate. Do not detach the run and leave the rental unattended. No Runpod account API credential is required by this kit.

The time estimate is not promised. If the allowance expires before required arms finish, preserve partial results as NO-RESULT and stop. A later retry must use the same original rental start time if the pod remained billable.

## Order

setup -> snapshot/calibration/task preparation -> BF16 load/generation smoke -> W4A16 quantization and smoke -> BF16 eval + benchmark -> W4A16 eval + benchmark -> optional NVFP4 quantization/smoke/eval/benchmark -> summary -> copy artifacts -> stop pod.

Each attempt writes results/runs/<id> independently; results/LATEST points to the newest attempt. No collector combines files across attempts. Protocol changes after looking at outcomes are exploratory amendments, not results under this freeze.
