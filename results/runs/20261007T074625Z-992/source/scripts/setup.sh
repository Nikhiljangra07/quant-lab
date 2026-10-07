#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
: "${QUANT_RUN_DIR:?Run through scripts/run_all.sh so deadlines and logs are active}"
mkdir -p "$QUANT_RUN_DIR/env"
python3 - <<'PY'
import platform, sys, shutil
if sys.version_info[:2] != (3, 12) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('This lock targets Linux x86_64 and Python 3.12. Select that pod image.')
if shutil.disk_usage('.').free < 55 * 1024**3:
    raise SystemExit('Need at least 55 GiB free for CUDA environments, weights, datasets and results.')
PY
nvidia-smi | tee "$QUANT_RUN_DIR/env/nvidia-smi.txt"
python3 - <<'PY'
import subprocess
driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
if int(driver.split('.')[0]) < 580:
    raise SystemExit('Pinned CUDA 13 wheels require an NVIDIA 580+ host driver; choose a compatible pod host.')
PY
for ENV in quant serve; do
  python3 -m venv "venv_$ENV"
  "venv_$ENV/bin/python" -m pip install --disable-pip-version-check --require-hashes --only-binary=torch,vllm,llmcompressor,compressed-tensors,tokenizers,flashinfer-python -r "requirements/$ENV.lock"
  "venv_$ENV/bin/python" -m pip check
  "venv_$ENV/bin/python" -m pip freeze > "$QUANT_RUN_DIR/env/$ENV.txt"
  "venv_$ENV/bin/python" scripts/preflight.py "$ENV"
done
venv_serve/bin/python - <<'PY'
import nltk
for resource in ('punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'):
    if not nltk.download(resource, quiet=True):
        raise SystemExit(f'Failed to download NLTK resource: {resource}')
PY
