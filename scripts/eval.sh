#!/usr/bin/env bash
# usage: scripts/eval.sh <label> <model_path_or_id>
set -euo pipefail
cd /workspace/quant-lab
L=$1; M=$2
venv_vllm/bin/lm_eval --model vllm \
  --model_args "pretrained=$M,dtype=auto,gpu_memory_utilization=0.85,max_model_len=4096,seed=42,enable_prefix_caching=True" \
  --tasks gsm8k,ifeval --batch_size auto --apply_chat_template --fewshot_as_multiturn \
  --seed 42 --output_path results/eval/$L --log_samples 2>&1 | tee results/eval_$L.log
