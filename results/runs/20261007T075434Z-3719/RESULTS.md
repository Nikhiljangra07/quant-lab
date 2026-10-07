# Quantization results

Run: `20261007T075434Z-3719`

Quality GO is a point-estimate rule (no more than 2 percentage points worse on either task). It is not proof of equivalence.

| arm | GSM8K strict | IFEval prompt strict | quality verdict | weight GB |
|---|---:|---:|---|---:|
| BF16 | 80.52% | 82.26% | REFERENCE | 8.045 |
| W4A16 | 72.55% | 81.52% | NO-GO | 2.652 |
| NVFP4 | — | — | NO-RESULT | 2.822 |

## Paired differences and generation limits

- BF16 generation-cap stops: {'gsm8k': 14, 'ifeval': 28}. These remain scored in the full-set evaluation.
- W4A16 gsm8k: -7.96 pp; descriptive paired normal 95% interval [-10.33, -5.59], n=1319.
- W4A16 ifeval: -0.74 pp; descriptive paired normal 95% interval [-2.91, +1.44], n=541.
- W4A16 generation-cap stops: {'gsm8k': 21, 'ifeval': 33}. These remain scored in the full-set evaluation.

## Serving results

Median and range across three repetitions; fixed 1024-token inputs/outputs, prefix caching disabled.

| arm | concurrency | output tok/s, median [min,max] | TTFT ms, median | TPOT ms, median |
|---|---:|---|---:|---:|
| BF16 | 1 | 144.0 [143.9,144.2] | 42.5 | 6.9 |
| BF16 | 16 | 1621.8 [1620.6,1623.1] | 266.9 | 9.6 |
| BF16 | 64 | 3485.8 [3478.1,3490.1] | 625.1 | 17.8 |
| W4A16 | 1 | 283.2 [280.0,283.7] | 38.1 | 3.5 |
| W4A16 | 16 | 2529.6 [2529.2,2533.4] | 303.6 | 6.0 |
| W4A16 | 64 | 3941.1 [3937.2,3947.9] | 705.2 | 15.6 |
| NVFP4 | 1 | NO-RESULT | — | — |
| NVFP4 | 16 | NO-RESULT | — | — |
| NVFP4 | 64 | NO-RESULT | — | — |

## Missing or invalid evidence

- NVFP4: gsm8k: FileNotFoundError: [Errno 2] No such file or directory: '/workspace/quant-lab/results/runs/20261007T075434Z-3719/eval/NVFP4/results_gsm8k.json'
- NVFP4: ifeval: FileNotFoundError: [Errno 2] No such file or directory: '/workspace/quant-lab/results/runs/20261007T075434Z-3719/eval/NVFP4/results_ifeval.json'

Raw stage logs, sample scores, termination telemetry and benchmark JSON are retained alongside this report. Long-context evaluation and MTP preservation are outside this protocol.
