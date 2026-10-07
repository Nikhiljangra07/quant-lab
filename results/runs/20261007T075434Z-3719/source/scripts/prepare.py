"""Save and audit identical calibration data for both quantization arms."""
import json
import random
import shutil
import subprocess
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from common import PROTOCOL as P, ROOT, RUN, sha256, shingles, verify_freeze, write_json

def main():
    verify_freeze()
    base = P["base"]
    base_path = Path(snapshot_download(repo_id=base["id"], revision=base["revision"],
                                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"]))
    tok = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True)
    protected = set()
    eval_counts = {}
    for name, spec in P["evals"].items():
        ds = load_dataset(spec["id"], spec["config"], revision=spec["revision"])
        eval_counts[name] = len(ds[spec["split"]])
        for split in {spec["split"], spec.get("fewshot_split", spec["split"])}:
            for row in ds[split]:
                protected.update(shingles(row.get("question", row.get("prompt", "")), P["calibration"]["shingle_words"]))
    c = P["calibration"]
    ds = load_dataset(c["id"], split=c["split"], revision=c["revision"])
    indices = list(range(len(ds)))
    random.Random(P["seed"]).shuffle(indices)
    rows, rejected = [], []
    for idx in indices:
        row = ds[idx]
        text = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        if shingles(text, c["shingle_words"]) & protected:
            rejected.append(idx)
            continue
        encoded = tok(text, truncation=True, max_length=c["max_tokens"], add_special_tokens=False)
        if not encoded["input_ids"]:
            continue
        rows.append({"source_index": idx, "prompt_id": row.get("prompt_id"), "text": text,
                     "input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]})
        if len(rows) == c["samples"]:
            break
    if len(rows) != c["samples"]:
        raise RuntimeError("Could not construct the full calibration selection.")
    calibration = RUN / "calibration.jsonl"
    calibration.write_text("".join(json.dumps(r) + "\n" for r in rows))
    task_root = Path(subprocess.check_output([str(ROOT / "venv_serve/bin/python"), "-c",
        "import lm_eval,pathlib; print(pathlib.Path(lm_eval.__file__).parent/'tasks')"], text=True).strip())
    for name, spec in P["evals"].items():
        dest = RUN / "tasks" / name
        dest.mkdir(parents=True)
        for file in (task_root / name).glob("*.py"):
            shutil.copy2(file, dest / file.name)
        content = (task_root / name / f"{name}.yaml").read_text()
        content = content.replace(f"task: {name}\n", f"task: quant_{name}\n")
        content += f"\ndataset_kwargs:\n  revision: {spec['revision']}\n"
        lines = content.splitlines()
        start = next(i for i, line in enumerate(lines) if line == "generation_kwargs:")
        end = start + 1
        while end < len(lines) and (lines[end].startswith(" ") or not lines[end]):
            end += 1
        stop = '["Question:", "</s>", "<|im_end|>"]' if name == "gsm8k" else '[]'
        lines[start:end] = ["generation_kwargs:", f"  until: {stop}", "  do_sample: false",
                            "  temperature: 0.0", f"  max_gen_toks: {spec['max_gen_toks']}"]
        (dest / f"{name}.yaml").write_text("\n".join(lines) + "\n")
    tracked = [calibration] + list((RUN / "tasks").rglob("*.yaml")) + list((RUN / "tasks").rglob("*.py"))
    lengths = [len(r["input_ids"]) for r in rows]
    write_json(RUN / "prepared.json", {
        "base_path": str(base_path), "base": base, "eval_counts": eval_counts,
        "calibration": {**c, "selected_indices": [r["source_index"] for r in rows],
            "rejected_overlap_indices": rejected, "min_tokens": min(lengths), "max_tokens": max(lengths),
            "total_tokens": sum(lengths), "audit": "zero normalized 8-word overlaps with eval/fewshot prompts; semantic overlap not tested"},
        "base_safetensors_sha256": {p.name: sha256(p) for p in base_path.glob("*.safetensors")},
        "base_size_bytes": sum(p.stat().st_size for p in base_path.glob("*.safetensors")),
        "file_sha256": {str(p.relative_to(RUN)): sha256(p) for p in tracked}})
    verify_freeze()

if __name__ == "__main__":
    main()
