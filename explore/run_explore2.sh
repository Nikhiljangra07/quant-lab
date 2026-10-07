#!/usr/bin/env bash
set -uo pipefail
cd /workspace/quant-lab
RUN=results/runs/20261007T075434Z-3719
until grep -q "EXPLORE END" $RUN/explore/console.log 2>/dev/null; do sleep 30; done
exec > >(tee -a $RUN/explore/console.log) 2>&1
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/workspace/quant-lab/cache/huggingface VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
echo "== [5] NVFP4 eval, linear backend marlin $(date -u +%T)"
venv_serve/bin/python explore/explore_quant.py --run $RUN --name nvfp4-marlin --stage evaluate --model-path $RUN/models/NVFP4 --linear-backend marlin || echo "nvfp4-marlin eval FAILED"
echo "== EXPLORE2 END $(date -u +%FT%TZ)"
