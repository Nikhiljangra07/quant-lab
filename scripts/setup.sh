#!/usr/bin/env bash
# Two venvs: llm-compressor and vLLM pin different torch builds; keep them apart.
set -euo pipefail
cd /workspace/quant-lab
nvidia-smi | tee results/nvidia-smi.txt
python -m venv venv_quant && venv_quant/bin/pip -q install -U pip
venv_quant/bin/pip -q install "llmcompressor" "transformers" "datasets" "accelerate" 2>&1 | tail -2
python -m venv venv_vllm && venv_vllm/bin/pip -q install -U pip
venv_vllm/bin/pip -q install "vllm" "lm_eval[vllm,ifeval]" "pandas" 2>&1 | tail -2
{ echo "== venv_quant =="; venv_quant/bin/pip freeze; echo "== venv_vllm =="; venv_vllm/bin/pip freeze; } > results/env.txt
echo "setup done $(date -u +%FT%TZ)"
