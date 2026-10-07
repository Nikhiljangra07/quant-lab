#!/usr/bin/env python
"""results/ → RESULTS.md: eval table (gsm8k strict, ifeval prompt-strict) + throughput table + bars verdict."""
import glob, json
from pathlib import Path
arms = ["BF16", "W4A16", "NVFP4"]
ev, th = {}, {}
for a in arms:
    fs = sorted(glob.glob(f"results/eval/{a}/**/results_*.json", recursive=True))
    if fs:
        r = json.load(open(fs[-1]))["results"]
        ev[a] = {"gsm8k": r.get("gsm8k", {}).get("exact_match,strict-match"),
                 "ifeval": r.get("ifeval", {}).get("prompt_level_strict_acc,none")}
    th[a] = {}
    for c in (1, 16, 64):
        p = Path(f"results/bench/{a}/c{c}.json")
        if p.exists():
            j = json.load(open(p)); th[a][c] = (j.get("output_throughput"), j.get("mean_ttft_ms"), j.get("median_tpot_ms"))
q = {a: json.load(open(f"results/quant_{a}.json")) for a in ("W4A16", "NVFP4") if Path(f"results/quant_{a}.json").exists()}
L = ["# Results\n", "| arm | gsm8k strict-match | ifeval prompt-strict | Δ gsm8k | Δ ifeval | size GB |", "|---|---|---|---|---|---|"]
b = ev.get("BF16", {})
for a in arms:
    e = ev.get(a)
    if not e: L.append(f"| {a} | — | — | — | — | — |"); continue
    dg = (e["gsm8k"] - b["gsm8k"]) * 100 if b.get("gsm8k") is not None and e["gsm8k"] is not None else None
    di = (e["ifeval"] - b["ifeval"]) * 100 if b.get("ifeval") is not None and e["ifeval"] is not None else None
    sz = q.get(a, {}).get("size_gb", "ref")
    L.append(f"| {a} | {e['gsm8k']:.4f} | {e['ifeval']:.4f} | {dg if dg is None else f'{dg:+.1f}'} | {di if di is None else f'{di:+.1f}'} | {sz} |")
L += ["", "| arm | c=1 tok/s | c=16 tok/s | c=64 tok/s | c=1 TTFT ms | c=1 TPOT ms |", "|---|---|---|---|---|---|"]
for a in arms:
    t = th.get(a, {})
    g = lambda c, i: (f"{t[c][i]:.1f}" if c in t and t[c][i] is not None else "—")
    L.append(f"| {a} | {g(1,0)} | {g(16,0)} | {g(64,0)} | {g(1,1)} | {g(1,2)} |")
L += ["", "## Go / no-go against the frozen bars (−2.0 points each)"]
for a in ("W4A16", "NVFP4"):
    e = ev.get(a)
    if not e or e["gsm8k"] is None: L.append(f"- {a}: NO-RESULT (see LOG.md)"); continue
    ok = (e["gsm8k"] - b["gsm8k"]) * 100 >= -2.0 and (e["ifeval"] - b["ifeval"]) * 100 >= -2.0
    L.append(f"- {a}: {'GO' if ok else 'NO-GO'}")
Path("RESULTS.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
