# Quantization results

Run: `20261007T074625Z-992`

Quality GO is a point-estimate rule (no more than 2 percentage points worse on either task). It is not proof of equivalence.

| arm | GSM8K strict | IFEval prompt strict | quality verdict | weight GB |
|---|---:|---:|---|---:|
| BF16 | — | — | NO-RESULT | 8.045 |
| W4A16 | — | — | NO-RESULT | — |
| NVFP4 | — | — | NO-RESULT | — |

## Paired differences and generation limits


## Serving results

Median and range across three repetitions; fixed 1024-token inputs/outputs, prefix caching disabled.

| arm | concurrency | output tok/s, median [min,max] | TTFT ms, median | TPOT ms, median |
|---|---:|---|---:|---:|
| BF16 | 1 | NO-RESULT | — | — |
| BF16 | 16 | NO-RESULT | — | — |
| BF16 | 64 | NO-RESULT | — | — |
| W4A16 | 1 | NO-RESULT | — | — |
| W4A16 | 16 | NO-RESULT | — | — |
| W4A16 | 64 | NO-RESULT | — | — |
| NVFP4 | 1 | NO-RESULT | — | — |
| NVFP4 | 16 | NO-RESULT | — | — |
| NVFP4 | 64 | NO-RESULT | — | — |

## Missing or invalid evidence

- BF16: gsm8k: FileNotFoundError: [Errno 2] No such file or directory: '/workspace/quant-lab/results/runs/20261007T074625Z-992/eval/BF16/results_gsm8k.json'
- BF16: ifeval: FileNotFoundError: [Errno 2] No such file or directory: '/workspace/quant-lab/results/runs/20261007T074625Z-992/eval/BF16/results_ifeval.json'
- W4A16: gsm8k: FileNotFoundError: [Errno 2] No such file or directory: '/workspace/quant-lab/results/runs/20261007T074625Z-992/eval/W4A16/results_gsm8k.json'
- W4A16: ifeval: FileNotFoundError: [Errno 2] No such file or directory: '/workspace/quant-lab/results/runs/20261007T074625Z-992/eval/W4A16/results_ifeval.json'
- W4A16: quantization: [Errno 2] No such file or directory: '/workspace/quant-lab/results/runs/20261007T074625Z-992/quant_W4A16.json'
- W4A16: BF16 reference incomplete
- NVFP4: gsm8k: FileNotFoundError: [Errno 2] No such file or directory: '/workspace/quant-lab/results/runs/20261007T074625Z-992/eval/NVFP4/results_gsm8k.json'
- NVFP4: ifeval: FileNotFoundError: [Errno 2] No such file or directory: '/workspace/quant-lab/results/runs/20261007T074625Z-992/eval/NVFP4/results_ifeval.json'
- NVFP4: quantization: [Errno 2] No such file or directory: '/workspace/quant-lab/results/runs/20261007T074625Z-992/quant_NVFP4.json'
- NVFP4: BF16 reference incomplete

Raw stage logs, sample scores, termination telemetry and benchmark JSON are retained alongside this report. Long-context evaluation and MTP preservation are outside this protocol.
