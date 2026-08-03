#!/usr/bin/env python3
"""Analyze neutral-preference predictions (CPU, safeguard 9).

Primary: preservation = tuned neutral preference == BASE neutral preference
(per case, modal over the 4 order/paraphrase forms). Secondary: agreement with
the frozen delta-preferred good, order (A/B position) invariance, stratified by
delta sign/magnitude.

    python3 eval/neutral_preference_analyze.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBSET = ROOT / "data" / "surface_form_stress" / "surface_form_subset.json"
ROOT_OUT = ROOT / "results" / "neutral_preference"
MODELS = ["Base", "SFT-seed1-step6000", "SignOnly-seed1-step6000"]


def load_prefs(name):
    p = ROOT_OUT / name / "neutral_predictions.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    by_case = defaultdict(list)
    order_choice = defaultdict(dict)
    for r in rows:
        by_case[r["case_id"]].append(r["chosen_good"])
        order_choice[r["case_id"]].setdefault(r["order"], []).append(r["chosen_good"])
    modal = {cid: Counter(v).most_common(1)[0][0] for cid, v in by_case.items()}
    # order invariance: same modal choice within each order block
    order_inv = {}
    for cid, od in order_choice.items():
        picks = [Counter(v).most_common(1)[0][0] for v in od.values()]
        order_inv[cid] = len(set(picks)) == 1
    return modal, order_inv


def main():
    subset = json.load(open(SUBSET))
    cases = {c["case_id"]: c for c in subset["cases"]}
    pref_target = {cid: (c["X_num"] if c["delta"] > 0 else c["Y_num"]) for cid, c in cases.items()}

    base = load_prefs("Base")
    if base is None:
        raise SystemExit("No neutral predictions yet — run neutral_preference_infer (GPU).")
    base_modal, _ = base

    out = {"status": "EXPLORATORY_POST_HOC", "models": []}
    for name in MODELS:
        r = load_prefs(name)
        if r is None:
            continue
        modal, order_inv = r
        ids = list(modal)
        preserve = sum(1 for cid in ids if modal[cid] == base_modal.get(cid)) / len(ids)
        tgt = sum(1 for cid in ids if modal[cid] == pref_target[cid]) / len(ids)
        oinv = sum(order_inv.values()) / len(order_inv)
        strat = defaultdict(lambda: {"preserve": [], "target": []})
        for cid in ids:
            s = f"{cases[cid]['delta_sign']}|{cases[cid]['delta_bin']}"
            strat[s]["preserve"].append(int(modal[cid] == base_modal.get(cid)))
            strat[s]["target"].append(int(modal[cid] == pref_target[cid]))
        out["models"].append({
            "model": name, "n_cases": len(ids),
            "preservation_vs_base": preserve if name != "Base" else 1.0,
            "agreement_with_frozen_target": tgt,
            "order_position_invariance": oinv,
            "by_stratum": {s: {"preservation": (sum(d["preserve"]) / len(d["preserve"])) if name != "Base" else 1.0,
                               "target": sum(d["target"]) / len(d["target"])}
                           for s, d in sorted(strat.items())},
        })
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    (ROOT_OUT / "neutral_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {ROOT_OUT/'neutral_summary.json'}")
    for m in out["models"]:
        print(f"  {m['model']:>24}: preserve_vs_base={m['preservation_vs_base']:.3f} "
              f"target={m['agreement_with_frozen_target']:.3f} order_inv={m['order_position_invariance']:.3f}")


if __name__ == "__main__":
    main()
