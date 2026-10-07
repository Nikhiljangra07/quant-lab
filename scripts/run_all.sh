#!/usr/bin/env bash
# Full pipeline under a pod log. Order: setup → quantize both → eval 3 arms → bench 3 arms → collect.
set -uo pipefail
cd /workspace/quant-lab; mkdir -p results
exec > >(tee -a results/pod.log) 2>&1
echo "== START $(date -u +%FT%TZ)"
bash scripts/setup.sh
venv_quant/bin/python scripts/quantize.py --scheme W4A16 --out models/qwen3-4b-W4A16
venv_quant/bin/python scripts/quantize.py --scheme NVFP4 --out models/qwen3-4b-NVFP4 || echo "NVFP4 quantize FAILED (recorded)"
bash scripts/eval.sh BF16 Qwen/Qwen3-4B-Instruct-2507
bash scripts/eval.sh W4A16 models/qwen3-4b-W4A16
bash scripts/eval.sh NVFP4 models/qwen3-4b-NVFP4 || echo "NVFP4 eval FAILED (recorded)"
bash scripts/bench.sh BF16 Qwen/Qwen3-4B-Instruct-2507
bash scripts/bench.sh W4A16 models/qwen3-4b-W4A16
bash scripts/bench.sh NVFP4 models/qwen3-4b-NVFP4 || echo "NVFP4 bench FAILED (recorded)"
venv_vllm/bin/python scripts/collect.py
echo "== END $(date -u +%FT%TZ)"
