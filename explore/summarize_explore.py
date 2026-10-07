#!/usr/bin/env python
"""Build RESULTS-EXPLORATORY.md from a completed protocol run + its explore/ directory.
Exploratory arms are single attempts outside the freeze; the table says so."""
import json, statistics, sys
from pathlib import Path

run = Path(sys.argv[1]).resolve()
P = json.loads((run / "freeze.json").read_text())["protocol"]
GS, IF = "exact_match,strict-match", "prompt_level_strict_acc,none"

def proto_eval(arm):
    out = {}
    for t, k in (("gsm8k", GS), ("ifeval", IF)):
        f = run / "eval" / arm / f"results_{t}.json"
        out[t] = json.load(open(f))["results"][f"quant_{t}"][k] if f.exists() else None
    return out

def expl_eval(name):
    f = run / "explore" / name / "eval" / "summary.json"
    if not f.exists(): return {"gsm8k": None, "ifeval": None}
    s = json.load(open(f)); return {"gsm8k": s.get("gsm8k", {}).get(GS), "ifeval": s.get("ifeval", {}).get(IF)}

def proto_bench(arm):
    r = {}
    for c in (1, 16, 64):
        fs = [run / "bench" / arm / f"c{c}-r{i}.json" for i in (1, 2, 3)]
        if all(f.exists() for f in fs):
            r[c] = statistics.median(json.load(open(f))["output_throughput"] for f in fs)
    return r

def expl_bench(name):
    r = {}
    for c in (1, 16, 64):
        f = run / "explore" / name / "bench" / f"c{c}.json"
        if f.exists(): r[c] = json.load(open(f))["output_throughput"]
    return r

def size(arm=None, name=None):
    if arm == "BF16": return json.load(open(run / "prepared.json"))["base_size_bytes"]
    if arm: return json.load(open(run / f"quant_{arm}.json"))["size_bytes"]
    f = run / "explore" / name / "quant.json"
    return json.load(open(f))["size_bytes"] if f.exists() else None

ref = proto_eval("BF16"); margin = P["acceptance_margin_pp"]
rows = [
    ("BF16 (reference)", "protocol", ref, size("BF16"), proto_bench("BF16"), "reference"),
    ("W4A16 GPTQ g128, all Linear", "protocol", proto_eval("W4A16"), size("W4A16"), proto_bench("W4A16"), None),
    ("W4A16 GPTQ g32, all Linear", "exploratory", expl_eval("g32"), size(name="g32"), expl_bench("g32"), None),
    ("W4A16 GPTQ g128, MLP only (attention BF16)", "exploratory", expl_eval("mlp-only"), size(name="mlp-only"), expl_bench("mlp-only"), None),
    ("NVFP4 (W4A4 g16), Marlin kernel", "exploratory", expl_eval("nvfp4-marlin"), size("NVFP4"), expl_bench("nvfp4-marlin"), None),
    ("FP8 dynamic (W8A8 float)", "exploratory", expl_eval("fp8"), size(name="fp8"), expl_bench("fp8"), None),
    ("INT8 W8A8", "exploratory", {"gsm8k": None, "ifeval": None}, size(name="w8a8"), {}, "unservable: vLLM CUTLASS INT8 not supported on SM120"),
]
L = ["# Exploratory results (outside the pre-registered freeze)", "",
     f"Run `{run.name}`. Protocol rows are the frozen result; exploratory rows are single attempts chosen after seeing the protocol outcome and are reported as hypotheses, not pre-registered claims. Same base snapshot, calibration selection, task files, seeds and serving settings. Bars: within {margin:.1f} pp of BF16 on both tasks.", "",
     "| recipe | status | GSM8K strict | Δ | IFEval prompt-strict | Δ | weights GB | verdict |", "|---|---|---:|---:|---:|---:|---:|---|"]
for name, st, e, sz, b, note in rows:
    if e["gsm8k"] is None:
        L.append(f"| {name} | {st} | — | — | — | — | {sz/1e9:.2f} | {note or 'NO-RESULT'} |"); continue
    dg, di = (e["gsm8k"] - ref["gsm8k"]) * 100, (e["ifeval"] - ref["ifeval"]) * 100
    v = note or ("GO" if dg >= -margin and di >= -margin else "NO-GO")
    if st == "exploratory" and not note: v += " (exploratory)"
    L.append(f"| {name} | {st} | {e['gsm8k']*100:.2f}% | {dg:+.1f} | {e['ifeval']*100:.2f}% | {di:+.1f} | {sz/1e9:.2f} | {v} |")
L += ["", "## Throughput (output tok/s, 1024 in / 1024 out, prefix caching off; protocol rows are medians of 3, exploratory rows single runs)", "",
      "| recipe | c=1 | c=16 | c=64 |", "|---|---:|---:|---:|"]
for name, st, e, sz, b, note in rows:
    if b: L.append(f"| {name} | {b.get(1, float('nan')):.0f} | {b.get(16, float('nan')):.0f} | {b.get(64, float('nan')):.0f} |")
L += ["", "## What the series says", "",
      "- The protocol W4A16 recipe (INT4 g128 on every Linear layer) fails the GSM8K bar by 8 points on this 4B model; IFEval barely moves.",
      "- Quantizing only the MLP projections and leaving attention q/k/v/o in BF16 removes the loss entirely (GSM8K above the BF16 point estimate, within noise) at 4.05 GB instead of 2.65 GB and 1.59× instead of 1.97× single-stream.",
      "- Finer INT4 groups (g32) halve the loss but do not clear the bar.",
      "- NVFP4 (FP4 weights and activations, group 16) clears both bars at 2.82 GB once served through the Marlin kernel; the FlashInfer kernels vLLM prefers need a JIT this pod image's nvcc 12.8 cannot build for SM120.",
      "- FP8 dynamic clears both bars at 4.41 GB with no calibration pass.",
      "- INT8 W8A8 cannot be served on this card in vLLM 0.25.0 (`Int8 not supported on SM120`).",
      "", "Caveats: single attempts; GSM8K strict-match depends on the harness's `#### N` answer format (flexible-extract gaps are smaller); the −2.0 pp bar is an acceptance rule, not an equivalence test."]
(run / "RESULTS-EXPLORATORY.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
