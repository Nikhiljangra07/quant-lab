#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$ROOT/venv_serve/bin/python" "$ROOT/scripts/evaluate.py" "$@"
