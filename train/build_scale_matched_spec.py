#!/usr/bin/env python3
"""Deterministically compute the scale-matched reward constant (Task 3 / ABLATION-001).

Characterises the magnitude-reward distribution |δ̃| over the FROZEN training
deltas (the same rows GRPO/SFT train on) and derives the scale-matching
constant c for the `scale_matched` reward weighting. Writes a machine-readable
spec and (optionally) checks it regenerates byte-identically.

The scale_matched control keeps sign-only's per-case-uniform magnitude but sets
c so the GLOBAL reward scale matches ±|δ̃|, isolating per-case magnitude
information from global scale (see ABLATION-001).

No training is run and no scale is tuned on evaluation results.

Usage (repository root):
    python3 train/build_scale_matched_spec.py
    python3 train/build_scale_matched_spec.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_functions import characterize_deltas, compute_scale_constant  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DELTA_FILE = PROJECT_ROOT / "data" / "deltas" / "delta_qwen_base.json"
OUT = PROJECT_ROOT / "results" / "scale_matched_reward" / "scale_matched_reward_spec.json"

# remaining_goods.json (training) occupies global case IDs 9,951..59,400.
TRAIN_ID_MIN = 9_951
TRAIN_ID_MAX = 59_400

DEFAULT_RULE = "mean_abs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def training_deltas() -> list[float]:
    with DELTA_FILE.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    vals = []
    for k, v in raw.items():
        cid = int(k)
        if TRAIN_ID_MIN <= cid <= TRAIN_ID_MAX:
            d = float(v["mean_delta"])
            if d != 0.0:
                vals.append(d)
    if not vals:
        raise SystemExit("No training deltas found in expected ID range.")
    return vals


def build_spec() -> dict:
    deltas = training_deltas()
    stats = characterize_deltas(deltas)
    c_mean_abs = compute_scale_constant(deltas, "mean_abs")
    c_rms = compute_scale_constant(deltas, "rms")
    chosen = compute_scale_constant(deltas, DEFAULT_RULE)
    return {
        "schema_version": 1,
        "purpose": "Scale-matched sign control for ABLATION-001 (reward-magnitude ablation).",
        "delta_file": str(DELTA_FILE.relative_to(PROJECT_ROOT)),
        "delta_file_sha256": sha256_file(DELTA_FILE),
        "training_id_range": [TRAIN_ID_MIN, TRAIN_ID_MAX],
        "magnitude_reward_distribution": stats,
        "scale_matching": {
            "default_rule": DEFAULT_RULE,
            "c_mean_abs": c_mean_abs,
            "c_rms": c_rms,
            "chosen_constant": chosen,
            "mean_abs_justification":
                "c = E[|delta|]; matches the first absolute moment / mean reward magnitude.",
            "rms_justification":
                "c = sqrt(E[delta^2]); matches the second moment / RMS effective advantage scale.",
        },
        "limitations": (
            "GRPO mean-centres advantages by the group mean (scale_rewards='none') "
            "and ~80% of groups carry zero task-reward advantage, so no constant c "
            "reproduces the realized per-step gradient of the magnitude reward; c "
            "matches only the chosen moment of the reward-magnitude distribution. "
            "The scale_matched control removes per-case weighting while matching "
            "global scale; sign_only removes both."
        ),
        "usage": {
            "reward_weighting": "scale_matched",
            "scale_constant": chosen,
            "config": "train/configs/qwen25_7b_qwen_delta_scale_matched.yaml",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    spec = build_spec()
    spec_bytes = canonical_json_bytes(spec)

    if args.check:
        if not OUT.exists() or OUT.read_bytes() != spec_bytes:
            raise SystemExit("CHECK FAILED: scale-matched spec drifted or missing.")
        print("CHECK PASSED: scale-matched spec regenerates byte-identically.")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_bytes(spec_bytes)
    tmp.replace(OUT)
    print(f"Wrote {OUT.relative_to(PROJECT_ROOT)}")
    print(f"  n_nonzero={spec['magnitude_reward_distribution']['n_nonzero']:,}")
    print(f"  mean_abs (c_mean_abs) = {spec['scale_matching']['c_mean_abs']:.6f}")
    print(f"  rms      (c_rms)      = {spec['scale_matching']['c_rms']:.6f}")
    print(f"  chosen constant ({DEFAULT_RULE}) = {spec['scale_matching']['chosen_constant']:.6f}")


if __name__ == "__main__":
    main()
