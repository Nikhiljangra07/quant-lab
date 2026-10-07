# quant-lab

A supervised, budget-limited Qwen3-4B quantization/evaluation experiment. Read RUNBOOK.md for the measured question and acceptance rule. No quantization results exist yet.

## What to rent

- **One RTX PRO 6000 Blackwell, full 96GB**, not RTX 6000 Ada, A6000 or a MIG partition.
- Linux x86_64, **Python 3.12**, an NVIDIA **580+ driver compatible with CUDA 13**. A recent Runpod PyTorch/CUDA image is suitable if it meets those requirements; the kit creates its own virtualenvs.
- **100GB workspace volume** with at least 55 GiB free when setup starts; allow around 30GB container disk for image and temporary package installation. Keep the kit and Hugging Face cache under /workspace. Volume storage is separately billed.
- SSH enabled, one GPU only. No public HTTP serving port is needed; servers bind to 127.0.0.1.
- Record the displayed hourly USD price and the time you click Deploy. Share the SSH connection command and hourly price with the operator. No private SSH key or account password needs to be pasted into chat.

Host driver compatibility is checked before installing environments, and each environment must pass a real BF16 CUDA test. Model loading is also smoke-tested before the full eval. Actual GPU execution remains to be verified on the rented pod.

## Run on the pod

Upload the kit to /workspace/quant-lab, excluding local .git, caches, virtualenvs and prior result directories. Use an interactive, supervised SSH session:

```bash
cd /workspace/quant-lab
bash scripts/run_all.sh --supervised --rate 2.09 \
  --pod-started '2026-10-07T08:00:00Z' --with-nvfp4
```

Replace the example rate and timestamp with the **actual price and rental start**, including timezone. Do not copy the example timestamp. Omit --with-nvfp4 to run just the required BF16/W4A16 comparison. Python 3.12 must be the python3 used to run the kit.

The runner prints stage progress. Full output is in results/runs/<id>/<stage>.log. Inspect from another SSH shell if needed:

```bash
cat results/LATEST
# Use the printed run ID below:
tail -f results/runs/RUN_ID/setup.log
```

Successful completion means the planned pipeline completed; inspect RESULTS.md for GO, NO-GO and NO-RESULT. A successful pipeline does not mean the quant passed the quality threshold. The --supervised flag acknowledges that a human will stop the rental when needed.

## Retrieve before stopping

From the local computer, use your supplied SSH hostname/port and the printed run ID:

```bash
scp -P SSH_PORT -r root@SSH_HOST:/workspace/quant-lab/results/runs/RUN_ID ./quant-results-RUN_ID
```

This directory includes both raw scientific evidence and quantized model files; copying several gigabytes takes time. Leave budget for retrieval. If time is tight, retrieve the small logs/reports/eval/bench/calibration/source files first, then weights if affordable. **Stop the pod in the Runpod console immediately after retrieval, including on failure. Stopping Python alone does not stop rental charges.** Do not terminate a pod before retrieving files from a non-persistent disk. Stopped persistent volumes can still incur storage charges.

## Local checks without a GPU

```bash
python3 -m unittest discover -s tests -v
```

Tests cover invalid/missing evidence, paired scoring, budget math, required-stage failure, process cleanup, HTTP health and shortened benchmark outputs. No GPU libraries or model weights are needed for these tests. See VALIDATION.md for completed checks and remaining GPU checks.

## Files

- protocol.json: exact model/dataset SHAs and scientific settings.
- requirements/: hash-locked Linux environments and direct dependency inputs.
- scripts/: portable paths, setup/preflight, calibration audit, quantization, measured eval, benchmarks, deadline runner and strict collector.
- results/runs/<id>/freeze.json and source/: actual settings plus complete frozen source for each attempt.
- results/runs/<id>/RESULTS.md and summary.json: human-readable and structured findings.

Sources used for the environment/API checks: [LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/), [vLLM 0.25 SM120 backend](https://github.com/vllm-project/vllm/blob/v0.25.0/vllm/model_executor/kernels/linear/nvfp4/flashinfer.py), [harness 0.4.13](https://pypi.org/project/lm-eval/0.4.13/). The package wheels and dataset revisions were checked against public registry metadata on October 7, 2026.
