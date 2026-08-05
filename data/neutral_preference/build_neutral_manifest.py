#!/usr/bin/env python3
"""Freeze the neutral-preference test DESIGN before any GPU inference (safeguard 9).

Reuses the SAME frozen surface-form subset (no new case selection, so nothing is
chosen on OOD/framing/surface-form outcomes). Freezes the neutral templates,
axes, form count, output locations, and input hashes into an immutable manifest.

    python3 data/neutral_preference/build_neutral_manifest.py
    python3 data/neutral_preference/build_neutral_manifest.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from neutral_preference import ORDERS, PARAPHRASES, _TEMPLATE  # noqa: E402

SUBSET = ROOT / "data" / "surface_form_stress" / "surface_form_subset.json"
OUT = ROOT / "data" / "neutral_preference" / "neutral_preference.manifest.json"


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            d.update(c)
    return d.hexdigest()


def build() -> bytes:
    subset = json.load(open(SUBSET))
    doc = {
        "schema_version": 1,
        "status": "FROZEN_PRE_GPU",
        "purpose": "Neutral (no-ownership) preference elicitation to test whether tuning preserved the base model's preferences or overwrote them. Exploratory/post-hoc; cannot change selection.",
        "reuses_subset": {
            "path": str(SUBSET.relative_to(ROOT)),
            "sha256": sha256_file(SUBSET),
            "n_cases": subset["n_cases"],
            "note": "Same frozen pre-training-metadata subset as surface-form; NOT reselected on any result.",
        },
        "design": {
            "framing": "neutral preference between Good A and Good B; NO endowment/keep/trade",
            "orders": ORDERS,
            "paraphrases": PARAPHRASES,
            "forms_per_case": len(ORDERS) * len(PARAPHRASES),
            "total_forms_per_model": subset["n_cases"] * len(ORDERS) * len(PARAPHRASES),
            "answer_tokens": ["A", "B"],
            "templates": _TEMPLATE,
            "scoring": "teacher-forced candidate log-prob (as surface-form), argmax",
        },
        "metrics": {
            "primary": "preservation = tuned neutral preference == BASE neutral preference (per case, modal over forms)",
            "secondary": ["agreement with frozen delta-preferred good",
                          "order-invariance (A/B position bias)",
                          "per-form choice stability"],
        },
        "models": ["Base", "SFT-seed1-step6000", "SignOnly-seed1-step6000"],
        "output_root": "results/neutral_preference",
        "gpu_submission": "DEFERRED (design frozen now; harness eval/neutral_preference_infer.py; PBS train/submit_eval_neutral_pilot.pbs)",
    }
    return (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    payload = build()
    if args.check:
        if not OUT.exists() or OUT.read_bytes() != payload:
            raise SystemExit("CHECK FAILED: neutral manifest drifted or missing.")
        print("CHECK PASSED: neutral-preference manifest byte-identical.")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(payload)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
