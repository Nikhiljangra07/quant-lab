# quant-lab — RUNBOOK (frozen before any spend)

**Question.** Does a 4-bit weight-only quant (W4A16, GPTQ, g=128) and an NVFP4 quant of
Qwen3-4B-Instruct-2507 hold the BF16 reference on a reasoning eval and an instruction-following eval,
and what do they buy in vLLM throughput on one RTX PRO 6000 Blackwell?

**Why this model.** Same base as cube-program (runs 21–23), so the weights are already familiar; dense,
so the whole Linear stack quantizes (no router/expert carve-out needed). lm_head stays BF16.

**Why this card.** CQL published an RTX PRO 6000 (Blackwell) TP=4 cell for GLM-5.3-Flash; a single card
of the same family is the cheapest place to run NVFP4 natively.

## Pre-committed protocol (zero edits after first GPU minute)
| item | value |
|---|---|
| base | `Qwen/Qwen3-4B-Instruct-2507`, revision pinned in LOG.md at download |
| calibration | 512 × 2048-token samples, `HuggingFaceH4/ultrachat_200k` train_sft, shuffle seed 42. Chat-formatted. Disjoint from both evals by construction (no GSM8K / IFEval text). |
| W4A16 recipe | llm-compressor `GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"], dampening_frac=0.01)` → compressed-tensors, group 128, symmetric |
| NVFP4 recipe | llm-compressor `QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])`, same calibration set |
| evals | lm-eval-harness, vllm backend, `gsm8k` (5-shot, strict-match) + `ifeval` (prompt-level strict), chat template on, greedy, seed 42, full test sets |
| throughput | `vllm bench serve`, random dataset 1024 in / 1024 out, concurrency 1 / 16 / 64, same `max_model_len=4096`, same seed |
| arms | BF16 (reference) · W4A16 · NVFP4 — identical eval + bench commands, only the checkpoint path changes |

## Success bars (go / no-go), written before spend
- **GO for W4A16:** gsm8k strict-match and ifeval prompt-strict each within **−2.0 points** of BF16.
- **GO for NVFP4:** same bars. If vLLM cannot serve NVFP4 on sm_120 with the installed build, that is a
  recorded NO-RESULT with the error, not a silent drop.
- Throughput is **reported, not a bar** (one card, one model size; no pre-stated expectation).
- Anything outside the bars is published at the same volume as a pass.

## Budget
Cap **$8** on RunPod. Expected ~2 h on RTX PRO 6000 Blackwell (~$1.8/h → ~$3.6). If setup exceeds
45 min, stop and log. Pod log (`script` capture) is kept in `results/pod.log` — lesson from cube-program
audit #2, where an unlogged pod substitution cost a claim.

## Freeze record (filled at run time, never edited after)
- base revision sha: 
- quant safetensors md5 (W4A16 / NVFP4): 
- `pip freeze` snapshot: `results/env.txt`
- harness commit before results: 
