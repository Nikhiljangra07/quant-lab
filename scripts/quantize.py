#!/usr/bin/env python
"""One-shot PTQ of Qwen3-4B-Instruct-2507 with llm-compressor. --scheme W4A16 | NVFP4."""
import argparse, hashlib, json, os, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--scheme", choices=["W4A16", "NVFP4"], required=True)
ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--seqlen", type=int, default=2048)
a = ap.parse_args()

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype="auto", device_map="cuda")
rev = getattr(model.config, "_commit_hash", None)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=42).select(range(a.n))
ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)})
ds = ds.map(lambda ex: tok(ex["text"], padding=False, max_length=a.seqlen, truncation=True, add_special_tokens=False),
            remove_columns=ds.column_names)

if a.scheme == "W4A16":
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"], dampening_frac=0.01)
else:
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=a.seqlen, num_calibration_samples=a.n)
model.save_pretrained(a.out, save_compressed=True)
tok.save_pretrained(a.out)

md5 = {}
for p in sorted(Path(a.out).glob("*.safetensors")):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    md5[p.name] = h.hexdigest()
size_gb = sum(p.stat().st_size for p in Path(a.out).glob("*.safetensors")) / 1e9
rec = {"scheme": a.scheme, "base": a.base, "base_revision": rev, "n_calib": a.n, "seqlen": a.seqlen,
       "safetensors_md5": md5, "size_gb": round(size_gb, 3), "seconds": round(time.time() - t0)}
Path("results").mkdir(exist_ok=True)
Path(f"results/quant_{a.scheme}.json").write_text(json.dumps(rec, indent=2))
print(json.dumps(rec, indent=2))
