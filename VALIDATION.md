# Local validation — October 7, 2026

- Linux x86_64/Python 3.12 quantization and serving dependency locks resolve successfully.
- Both locks pass hash-enforced installation dry runs; no GPU packages were installed on the Mac and no model weights were downloaded.
- Public model and dataset revisions resolved and fixed in protocol.json.
- Pinned lm_eval wheel task definitions and evaluator API inspected; pinned vLLM 0.25 benchmark/backend source inspected.
- 19 CPU-only regression checks pass: missing reference/metrics, NaN, changed calibration/holdout, incomplete examples, input truncation, quality failure, strict-filter selection, budget offset/invalid input, required/optional failures, timeout cleanup, HTTP errors/wrong model, shortened outputs, and a complete local HTTP server smoke/benchmark/cleanup lifecycle.
- Every Python source parses and all shell wrappers pass bash -n.

Required on the GPU: driver/CUDA and BF16 smoke in both virtualenvs; exact saved quantization schema; BF16/W4A16 model serving smoke; NVFP4 SM120 backend smoke if attempted; complete dataset counts and pair hashes; actual timings, scores, throughput and artifact sizes. These have not been run and are not claimed to pass.

The $8 limit is a supervised rental allowance with a compute-process watchdog and a $1 reserve. Pod lifecycle and storage billing remain under operator control.
