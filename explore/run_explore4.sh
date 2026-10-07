#!/usr/bin/env bash
set -uo pipefail
cd /workspace/quant-lab
RUN=results/runs/20261007T075434Z-3719
until grep -q "EXPLORE3 END" $RUN/explore/console.log 2>/dev/null; do sleep 30; done
exec > >(tee -a $RUN/explore/console.log) 2>&1
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/workspace/quant-lab/cache/huggingface VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
echo "== [7] fp8 quantize $(date -u +%T)"
venv_quant/bin/python explore/explore_quant.py --run $RUN --name fp8 --variant fp8 --stage quantize || echo "fp8 quantize FAILED"
echo "== [7] fp8 evaluate $(date -u +%T)"
venv_serve/bin/python explore/explore_quant.py --run $RUN --name fp8 --stage evaluate || echo "fp8 eval FAILED"
echo "== EXPLORE4 END $(date -u +%FT%TZ)"
