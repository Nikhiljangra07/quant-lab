#!/usr/bin/env python
"""EXPLORATORY (not under the protocol freeze): quantize a recipe variant using a completed run's
prepared base snapshot and calibration selection, then evaluate it with the protocol's eval settings.

Writes to results/explore/<name>/ inside the given run directory. Nothing here feeds collect.py.
Variants:
  mlp-only  : W4A16 GPTQ g=128 on MLP projections only (attention q/k/v/o stay BF16)
  g32       : W4A16 GPTQ, group_size 32, all Linear except lm_head
  w8a8      : INT8 weights + dynamic INT8 activations (SmoothQuant-free), all Linear except lm_head
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def quantize(run, name, variant, seed):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    prepared = json.loads((run / "prepared.json").read_text())
    rows = [json.loads(l) for l in (run / "calibration.jsonl").read_text().splitlines()]
    ds = Dataset.from_list([{k: r[k] for k in ("input_ids", "attention_mask")} for r in rows])
    model = AutoModelForCausalLM.from_pretrained(prepared["base_path"], dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(prepared["base_path"], local_files_only=True)
    if variant == "mlp-only":
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", dampening_frac=0.01,
                              ignore=["lm_head", "re:.*self_attn.*"])
    elif variant == "g32":
        recipe = GPTQModifier(targets="Linear", dampening_frac=0.01, ignore=["lm_head"],
                              config_groups={"group_0": {"targets": ["Linear"], "weights": {
                                  "num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 32}}})
    elif variant == "w8a8":
        recipe = QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])
    elif variant == "fp8":
        recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    else:
        raise SystemExit(f"unknown variant {variant}")
    t0 = time.monotonic()
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=2048, num_calibration_samples=len(rows))
    out = run / "explore" / name / "model"
    model.save_pretrained(out, save_compressed=True); tok.save_pretrained(out)
    qc = json.loads((out / "config.json").read_text()).get("quantization_config", {})
    size = sum(p.stat().st_size for p in out.glob("*.safetensors"))
    (run / "explore" / name / "quant.json").write_text(json.dumps({
        "variant": variant, "seconds": round(time.monotonic() - t0), "size_bytes": size,
        "quantization_config": qc, "base": prepared["base"]}, indent=2))
    print(name, "quantized", round(size / 1e9, 3), "GB in", round(time.monotonic() - t0), "s")

def evaluate(run, name, seed, model_path=None, linear_backend=None):
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    from lm_eval.models.vllm_causallms import VLLM
    model_path = model_path or str(run / "explore" / name / "model")
    kw = dict(pretrained=model_path, dtype="bfloat16", max_model_len=4096, seed=seed,
              gpu_memory_utilization=0.85, enable_prefix_caching=False, max_gen_toks=1024)
    if linear_backend: kw["linear_backend"] = linear_backend
    lm = VLLM(**kw)
    tm = TaskManager(include_path=str(run / "tasks"))
    dest = run / "explore" / name / "eval"; dest.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task, metric in (("gsm8k", "exact_match,strict-match"), ("ifeval", "prompt_level_strict_acc,none")):
        res = simple_evaluate(model=lm, tasks=[f"quant_{task}"], task_manager=tm, batch_size="auto",
                              apply_chat_template=True, fewshot_as_multiturn=True, random_seed=seed,
                              numpy_random_seed=seed, torch_random_seed=seed, fewshot_random_seed=seed,
                              bootstrap_iters=1000, log_samples=True)
        samples = res.pop("samples")[f"quant_{task}"]
        (dest / f"samples_{task}.jsonl").write_text("".join(json.dumps(x, default=lambda v: v.item()) + "\n" for x in samples))
        (dest / f"results_{task}.json").write_text(json.dumps(res, indent=2, default=lambda v: v.item()))
        r = res["results"][f"quant_{task}"]
        summary[task] = {k: r[k] for k in r if isinstance(r[k], float)}
        print(name, task, metric, round(r[metric] * 100, 2))
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="completed protocol run directory")
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", choices=["mlp-only", "g32", "w8a8", "fp8"])
    ap.add_argument("--stage", choices=["quantize", "evaluate"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-path", help="evaluate an existing checkpoint instead of explore/<name>/model")
    ap.add_argument("--linear-backend", help="vLLM linear backend override, e.g. auto")
    a = ap.parse_args()
    run = Path(a.run).resolve()
    if a.stage == "quantize":
        if not a.variant: raise SystemExit("--variant required for quantize")
        quantize(run, a.name, a.variant, a.seed)
    else:
        evaluate(run, a.name, a.seed, a.model_path, a.linear_backend)
