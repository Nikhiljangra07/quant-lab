#!/usr/bin/env bash
set -uo pipefail
cd /workspace/quant-lab
RUN=results/runs/20261007T075434Z-3719
until grep -q "EXPLORE2 END" $RUN/explore/console.log 2>/dev/null; do sleep 30; done
exec > >(tee -a $RUN/explore/console.log) 2>&1
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/workspace/quant-lab/cache/huggingface VLLM_NO_USAGE_STATS=1
M=$RUN/explore/mlp-only/model; D=$RUN/explore/mlp-only/bench; mkdir -p $D
echo "== [6] bench mlp-only (1 repeat per cell, exploratory) $(date -u +%T)"
venv_serve/bin/vllm serve $M --host 127.0.0.1 --port 8123 --served-model-name m --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 42 --no-enable-prefix-caching --generation-config vllm > $D/server.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -s localhost:8123/health >/dev/null && break; sleep 5; done
if curl -s localhost:8123/health >/dev/null; then
  for C in 1 16 64; do N=$(( C == 1 ? 16 : (C == 16 ? 128 : 256) ))
    venv_serve/bin/vllm bench serve --backend vllm --base-url http://127.0.0.1:8123 --model m --tokenizer $M --dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.0 --num-prompts $N --max-concurrency $C --seed 42 --temperature 0 --ignore-eos --num-warmups 4 --save-result --result-dir $D --result-filename c$C.json > $D/c$C.log 2>&1 && echo "mlp-only bench c$C ok"
  done
else echo "mlp-only bench FAILED: server not ready"; tail -20 $D/server.log; fi
kill $SP 2>/dev/null; wait $SP 2>/dev/null
echo "== EXPLORE3 END $(date -u +%FT%TZ)"
