#!/usr/bin/env bash
# usage: scripts/bench.sh <label> <model_path_or_id>   — serves once, benches c=1/16/64, kills server
set -euo pipefail
cd /workspace/quant-lab
L=$1; M=$2; PORT=8000
mkdir -p results/bench/$L
venv_vllm/bin/vllm serve "$M" --port $PORT --max-model-len 4096 --seed 42 --gpu-memory-utilization 0.85 \
  --served-model-name m > results/bench/$L/server.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -s localhost:$PORT/health >/dev/null && break; sleep 5; done
curl -s localhost:$PORT/health >/dev/null || { echo "server failed to start"; tail -50 results/bench/$L/server.log; kill $SP; exit 1; }
for C in 1 16 64; do
  N=$(( C == 1 ? 8 : (C == 16 ? 64 : 128) ))
  venv_vllm/bin/vllm bench serve --backend vllm --base-url http://localhost:$PORT --model m --tokenizer "$M" \
    --dataset-name random --random-input-len 1024 --random-output-len 1024 --num-prompts $N --max-concurrency $C \
    --seed 42 --save-result --result-dir results/bench/$L --result-filename c$C.json 2>&1 | tee results/bench/$L/c$C.log
done
kill $SP; wait $SP 2>/dev/null || true
