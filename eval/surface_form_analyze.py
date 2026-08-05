#!/usr/bin/env python3
"""Analyze surface-form stress predictions (CPU, safeguard 5).

Reads results/surface_form_stress/<model>/form_predictions.jsonl for base / SFT /
sign-only and reports the two CO-PRIMARY axes plus supporting metrics:

  co-primary 1 — semantic invariance: same final good across all 48 equivalent
                 forms (a constant answer passes this, hence axis 2 is needed);
  co-primary 2 — fidelity: agreement of the chosen final good with the frozen
                 preferred good (delta sign), incl. WORST-form fidelity.

Also: pairwise flip rate, probability spread across forms, effects by
transformation axis, parse failures, direct ownership effect (keep-both across
perspectives), all stratified by delta sign and magnitude.

    python3 eval/surface_form_analyze.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBSET = ROOT / "data" / "surface_form_stress" / "surface_form_subset.json"
ROOT_OUT = ROOT / "results" / "surface_form_stress"
# Core comparison (2026-08-03 clarification): base, SFT, magnitude-GRPO s1@2000,
# s2@6000. Sign-only is supplementary (aggregated if its dir is present; the
# analyzer skips any missing model). Metrics/logic are UNCHANGED.
MODELS = ["Base", "SFT-seed1-step6000",
          "GRPO-qd-seed1-ckpt2000", "GRPO-qd-seed2-ckpt6000",
          "SignOnly-seed1-step6000"]
CORE_MODELS = ["Base", "SFT-seed1-step6000", "GRPO-qd-seed1-ckpt2000", "GRPO-qd-seed2-ckpt6000"]
AXES = ["answer_style", "display_order", "attr_order", "paraphrase"]


def preferred_good(case):
    return case["X_num"] if case["delta"] > 0 else case["Y_num"]


def simpson_flip(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    counts = Counter(vals)
    same = sum(c * (c - 1) for c in counts.values())
    return 1.0 - same / (n * (n - 1))


def analyze_model(name, subset):
    path = ROOT_OUT / name / "form_predictions.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cases = {c["case_id"]: c for c in subset["cases"]}
    pref = {cid: preferred_good(c) for cid, c in cases.items()}

    by_unit = defaultdict(list)          # (case_id, perspective) -> rows
    for r in rows:
        by_unit[(r["case_id"], r["perspective"])].append(r)

    invariant_units = 0
    flip_rates, prob_spreads = [], []
    fidelity_all = []                    # per-form final==preferred
    per_axis = {ax: defaultdict(list) for ax in AXES}
    per_form_fidelity = defaultdict(list)  # form axis-combo -> [0/1]
    modal_by_unit = {}
    strat = defaultdict(lambda: {"inv": [], "fid": []})

    for (cid, persp), frs in by_unit.items():
        finals = [r["final_good"] for r in frs]
        modal = Counter(finals).most_common(1)[0][0]
        modal_by_unit[(cid, persp)] = modal
        inv = len(set(finals)) == 1
        invariant_units += inv
        flip_rates.append(simpson_flip(finals))
        pts = [r["p_trade"] for r in frs]
        prob_spreads.append(max(pts) - min(pts))
        p = pref[cid]
        sign = cases[cid]["delta_sign"]; mbin = cases[cid]["delta_bin"]
        strat[f"{sign}|{mbin}"]["inv"].append(inv)
        for r in frs:
            hit = int(r["final_good"] == p)
            fidelity_all.append(hit)
            strat[f"{sign}|{mbin}"]["fid"].append(hit)
            for ax in AXES:
                per_axis[ax][r[ax]].append(hit)
            per_form_fidelity[(r["answer_style"], r["display_order"],
                               r["attr_order"], r["paraphrase"])].append(hit)

    # ownership: keep-both across perspectives (modal good per perspective)
    keep_both = 0; n_cases = 0
    for cid, c in cases.items():
        mx = modal_by_unit.get((cid, "X")); my = modal_by_unit.get((cid, "Y"))
        if mx is None or my is None:
            continue
        n_cases += 1
        keep_both += int(mx == c["X_num"] and my == c["Y_num"])

    worst_form = min(((sum(v) / len(v), k) for k, v in per_form_fidelity.items()),
                     default=(None, None))
    n_units = len(by_unit)
    return {
        "model": name, "n_units": n_units, "n_form_rows": len(rows),
        "parse_failures": 0, "parse_failure_note": "teacher-forced scoring: choice always defined",
        "semantic_invariance_rate": invariant_units / n_units,
        "fidelity_mean": sum(fidelity_all) / len(fidelity_all),
        "worst_form_fidelity": worst_form[0],
        "worst_form": ("%s/%s/%s/%s" % worst_form[1]) if worst_form[1] else None,
        "mean_pairwise_flip_rate": sum(flip_rates) / len(flip_rates),
        "mean_prob_spread_across_forms": sum(prob_spreads) / len(prob_spreads),
        "ownership_keep_both_rate": keep_both / n_cases if n_cases else None,
        "fidelity_by_axis": {ax: {v: sum(l) / len(l) for v, l in vals.items()}
                             for ax, vals in per_axis.items()},
        "by_stratum": {s: {"invariance": sum(d["inv"]) / len(d["inv"]),
                           "fidelity": sum(d["fid"]) / len(d["fid"])}
                       for s, d in sorted(strat.items())},
    }


def render_md(results):
    L = ["# Surface-form stress diagnostic (co-primary: invariance + fidelity)", "",
         "*EXPLORATORY / POST-HOC. Fresh test_goods subset; frozen suites not "
         "consumed. Invariance alone is insufficient (a constant answer passes it) "
         "— fidelity is co-primary.*", "",
         "| model | role | invariance | fidelity | worst-form fid | flip rate | prob spread | keep-both |",
         "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r is None:
            continue
        wf = f"{r['worst_form_fidelity']:.3f}" if r['worst_form_fidelity'] is not None else "—"
        L.append(f"| {r['model']} | {r.get('role','—')} | {r['semantic_invariance_rate']:.3f} | "
                 f"{r['fidelity_mean']:.3f} | {wf} | {r['mean_pairwise_flip_rate']:.3f} | "
                 f"{r['mean_prob_spread_across_forms']:.3f} | {r['ownership_keep_both_rate']:.3f} |")
    L += ["", "Read: high invariance + high fidelity = ownership-invariant rule; "
          "high invariance + low fidelity = constant/shortcut answer; low invariance "
          "= surface-form fragility. Worst-form fidelity exposes the weakest "
          "transformation. See JSON for per-axis and per-stratum breakdowns.", ""]
    for r in results:
        if r is None:
            continue
        L.append(f"### {r['model']} — fidelity by axis")
        for ax, vals in r["fidelity_by_axis"].items():
            L.append(f"- {ax}: " + ", ".join(f"{v}={x:.3f}" for v, x in vals.items()))
        L.append("")
    return "\n".join(L)


def main():
    subset = json.load(open(SUBSET))
    results = []
    for m in MODELS:
        r = analyze_model(m, subset)
        if r is not None:
            r["role"] = "core" if m in CORE_MODELS else "supplementary"
            results.append(r)
    have = results
    if not have:
        raise SystemExit("No form_predictions found yet — run surface_form_infer first (GPU).")
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    (ROOT_OUT / "diagnostic_summary.json").write_text(
        json.dumps({"status": "EXPLORATORY_POST_HOC", "models": have}, indent=2) + "\n")
    (ROOT_OUT / "diagnostic_summary.md").write_text(render_md(results))
    print(f"Wrote {ROOT_OUT/'diagnostic_summary.json'} ({len(have)} models)")
    for r in have:
        print(f"  {r['model']:>24}: inv={r['semantic_invariance_rate']:.3f} "
              f"fid={r['fidelity_mean']:.3f} worst={r['worst_form_fidelity']} "
              f"keep_both={r['ownership_keep_both_rate']:.3f}")


if __name__ == "__main__":
    main()
