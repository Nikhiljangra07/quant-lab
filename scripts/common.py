"""Shared stdlib-only provenance, budget and validation utilities."""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("QUANT_RUN_DIR", str(ROOT / "results" / "manual"))).resolve()
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())

def now():
    return datetime.now(timezone.utc).isoformat()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def source_hashes():
    paths = [ROOT / "protocol.json", ROOT / "RUNBOOK.md"]
    for folder in ("scripts", "requirements"):
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}

def verify_freeze():
    freeze = json.loads((RUN / "freeze.json").read_text())
    if source_hashes() != freeze["source_sha256"]:
        raise RuntimeError("Frozen source changed; record an amendment and start a new run.")
    if (RUN / "prepared.json").exists():
        prepared = json.loads((RUN / "prepared.json").read_text())
        for name, digest in prepared["file_sha256"].items():
            if sha256(RUN / name) != digest:
                raise RuntimeError(f"Prepared artifact changed: {name}")

def budget_deadline(started, rate, cap, reserve):
    if not all(math.isfinite(x) for x in (rate, cap, reserve)) or rate <= 0 or cap <= reserve or reserve < 0:
        raise ValueError("Require finite positive hourly rate, cap > reserve >= 0.")
    started = datetime.fromisoformat(started.replace("Z", "+00:00"))
    if started.tzinfo is None:
        raise ValueError("Pod start time must include its timezone.")
    if started.timestamp() > datetime.now(timezone.utc).timestamp() + 5:
        raise ValueError("Pod start cannot be in the future.")
    return started.timestamp() + (cap - reserve) / rate * 3600

def shingles(text, size=8):
    words = re.findall(r"\w+", text.casefold())
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}

def valid_score(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1

def validate_bench(data, expected, output_tokens, input_tokens=1024):
    errors = []
    if data.get("completed") != expected or data.get("failed", 0) != 0:
        errors.append("incomplete or failed requests")
    lens = data.get("output_lens", [])
    if len(lens) != expected or any(n != output_tokens for n in lens):
        errors.append("output lengths differ from the fixed-length workload")
    if data.get("total_output_tokens") != expected * output_tokens:
        errors.append("total output token count mismatch")
    inputs = data.get("input_lens", [])
    if len(inputs) != expected or any(n != input_tokens for n in inputs):
        errors.append("input lengths differ from the fixed-length workload")
    if data.get("total_input_tokens") != expected * input_tokens:
        errors.append("total input token count mismatch")
    for name in ("duration", "output_throughput", "mean_ttft_ms", "median_tpot_ms"):
        x = data.get(name)
        if not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
            errors.append(f"invalid {name}")
    return errors
