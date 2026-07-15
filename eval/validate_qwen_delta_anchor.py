#!/usr/bin/env python3
"""
Validate the Qwen-own pseudo-utility (delta_qwen_base.json) against Qwen's
FROZEN OWNERSHIP-FREE preferences (data/anchors/neutral_anchor_*.json).

This addresses PAPER_READINESS.md Priority-0 item #2 ("Reliability of Qwen's own
pseudo-utility"): the concern is that base Qwen answers "No" to ~99% of endowed
prompts, which might leave its item/attribute utilities weakly identified.

The two signals are derived INDEPENDENTLY:
  * delta_qwen_base.json — sign(delta) = which good the structural NLS (Model A,
    link scale T=1) says Qwen prefers, fit from ENDOWMENT-framed choices.
  * neutral_anchor_*.json — which good Qwen prefers when asked WITHOUT ownership
    framing, elicited over multiple paraphrases and both display orders and kept
    only when stable (elicit_neutral_anchor.py).

so their agreement is convergent validity, not a tautology. Agreement is also
reported by |delta| bucket: the reward weights by |delta|, so the relevant
question is whether the signal is trustworthy exactly where the reward is loud.

CPU-only, no GPU, seconds to run.

    python eval/validate_qwen_delta_anchor.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DELTA_FILE = PROJECT_ROOT / "data" / "deltas" / "delta_qwen_base.json"
SPLITS = {
    "test_goods": PROJECT_ROOT / "data" / "anchors" / "neutral_anchor_test_goods.json",
    "remaining_goods": PROJECT_ROOT / "data" / "anchors" / "neutral_anchor_remaining_goods.json",
}
BUCKETS = [("|d|>1.0", 1.0, float("inf")),
           ("0.5<|d|<=1.0", 0.5, 1.0),
           ("|d|<=0.5", 0.0, 0.5)]


def load_anchor(path: Path) -> dict[int, str]:
    raw = json.load(open(path))
    a = raw.get("anchors", raw)
    return {int(k): (v["pref"] if isinstance(v, dict) else v) for k, v in a.items()}


def delta_of(d: dict, cid: int):
    v = d.get(str(cid))
    if v is None:
        return None
    return float(v["mean_delta"] if isinstance(v, dict) else v)


def main():
    deltas = json.load(open(DELTA_FILE))
    report = {}
    for split, anchor_path in SPLITS.items():
        anchor = load_anchor(anchor_path)
        agree = tot = 0
        by_bucket = {name: [0, 0] for name, _, _ in BUCKETS}
        for cid, pref in anchor.items():
            if pref not in ("X", "Y"):
                continue                      # no stable ownership-free preference
            dv = delta_of(deltas, cid)
            if dv is None or abs(dv) < 1e-9:
                continue
            ok = (("X" if dv > 0 else "Y") == pref)
            tot += 1
            agree += ok
            for name, lo, hi in BUCKETS:
                if lo < abs(dv) <= hi:
                    by_bucket[name][1] += 1
                    by_bucket[name][0] += ok
                    break
        report[split] = {
            "n_compared": tot,
            "sign_agreement": agree / tot if tot else float("nan"),
            "by_delta_magnitude": {
                k: {"n": v[1], "agreement": (v[0] / v[1]) if v[1] else float("nan")}
                for k, v in by_bucket.items()
            },
        }
        print(f"{split}: sign agreement = {agree}/{tot} = {agree/tot:.1%}")
        for name, _, _ in BUCKETS:
            c, n = by_bucket[name]
            if n:
                print(f"    {name:<14} {c}/{n} = {c/n:.1%}")

    out = PROJECT_ROOT / "results" / "qwen_delta_anchor_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    print("Chance level is 50%; agreement well above chance indicates the "
          "pseudo-utility carries real preference signal despite the ~99% No rate.")


if __name__ == "__main__":
    main()
