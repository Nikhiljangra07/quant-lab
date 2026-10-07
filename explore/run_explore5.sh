#!/usr/bin/env bash
set -uo pipefail
cd /workspace/quant-lab
RUN=results/runs/20261007T075434Z-3719
until grep -q "EXPLORE4 END" $RUN/explore/console.log 2>/dev/null; do sleep 30; done
exec > >(tee -a $RUN/explore/console.log) 2>&1
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/workspace/quant-lab/cache/huggingface VLLM_NO_USAGE_STATS=1
M=$RUN/models/NVFP4; D=$RUN/explore/nvfp4-marlin/bench; mkdir -p $D
echo "== [8] bench nvfp4-marlin (1 repeat per cell, exploratory) $(date -u +%T)"
venv_serve/bin/vllm serve $M --host 127.0.0.1 --port 8124 --served-model-name m --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 42 --no-enable-prefix-caching --generation-config vllm --linear-backend marlin > $D/server.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -s localhost:8124/health >/dev/null && break; sleep 5; done
if curl -s localhost:8124/health >/dev/null; then
  for C in 1 16 64; do N=$(( C == 1 ? 16 : (C == 16 ? 128 : 256) ))
    venv_serve/bin/vllm bench serve --backend vllm --base-url http://127.0.0.1:8124 --model m --tokenizer $M --dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.0 --num-prompts $N --max-concurrency $C --seed 42 --temperature 0 --ignore-eos --num-warmups 4 --save-result --result-dir $D --result-filename c$C.json > $D/c$C.log 2>&1 && echo "nvfp4-marlin bench c$C ok"
  done
else echo "nvfp4-marlin bench FAILED: server not ready"; tail -20 $D/server.log; fi
kill $SP 2>/dev/null; wait $SP 2>/dev/null
echo "== EXPLORE5 END $(date -u +%FT%TZ)"
