#!/usr/bin/env bash
set -uo pipefail
cd /workspace/quant-lab
RUN=results/runs/20261007T075434Z-3719
mkdir -p $RUN/explore
exec > >(tee -a $RUN/explore/console.log) 2>&1
echo "== EXPLORE START $(date -u +%FT%TZ)"
QP=venv_quant/bin/python; SP=venv_serve/bin/python
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/workspace/quant-lab/cache/huggingface VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
echo "== [1] NVFP4 eval, linear backend auto $(date -u +%T)"
$SP explore/explore_quant.py --run $RUN --name nvfp4-auto --stage evaluate --model-path $RUN/models/NVFP4 --linear-backend auto || echo "nvfp4-auto eval FAILED"
for v in mlp-only w8a8 g32; do
  echo "== [$v] quantize $(date -u +%T)"
  $QP explore/explore_quant.py --run $RUN --name $v --variant $v --stage quantize || { echo "$v quantize FAILED"; continue; }
  echo "== [$v] evaluate $(date -u +%T)"
  $SP explore/explore_quant.py --run $RUN --name $v --stage evaluate || echo "$v eval FAILED"
done
echo "== EXPLORE END $(date -u +%FT%TZ)"
