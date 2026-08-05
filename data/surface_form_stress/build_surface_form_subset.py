#!/usr/bin/env python3
"""Freeze the surface-form stress SUBSET before any GPU inference (safeguard 1).

Selection uses ONLY pre-training metadata — the sign and magnitude of the FROZEN
base pseudo-utility delta (`data/deltas/delta_qwen_base.json`) — plus a fixed
seed. It does NOT use any SFT or sign-only model output. Cases are drawn from
`test_goods` (already-opened validation), so the frozen semantic and
method-comparison suites are NOT consumed.

Stratify by delta sign (pos/neg) x magnitude bin (<=0.5, 0.5-1, 1-2, >2) = 8
strata; within each, hash-rank case IDs by sha256("{seed}:{case_id}") and take
the first N. Saves case IDs + full (X_num, Y_num, attr_code, delta) mapping, the
selection rule, seed, and a SHA-256 manifest.

    python3 data/surface_form_stress/build_surface_form_subset.py
    python3 data/surface_form_stress/build_surface_form_subset.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DELTA_FILE = ROOT / "data" / "deltas" / "delta_qwen_base.json"
GOODS_FILES = ["trial_goods", "test_goods"]  # order fixes the global case-ID offset
OUT_DIR = ROOT / "data" / "surface_form_stress"
OUT_SUBSET = OUT_DIR / "surface_form_subset.json"
OUT_MANIFEST = OUT_DIR / "surface_form_subset.manifest.json"

FROZEN_SEED = 20260803
N_PER_STRATUM = 12
BIN_EDGES = [0.0, 0.5, 1.0, 2.0, float("inf")]
BIN_LABELS = ["<=0.5", "0.5-1", "1-2", ">2"]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            d.update(c)
    return d.hexdigest()


def canon(v) -> bytes:
    return (json.dumps(v, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def build_case_map() -> dict[int, dict]:
    """Replicate the global case-ID assignment (trial then test) so every
    test_goods case_id maps to (X_num, Y_num, attr_code)."""
    cmap: dict[int, dict] = {}
    gid = 0
    for gs in GOODS_FILES:
        rows = json.load(open(ROOT / "data" / f"{gs}.json"))
        for entry in rows:
            X, Y, codes = entry[0], entry[1], entry[2]
            if not isinstance(codes, list):
                codes = [codes]
            for code in codes:
                gid += 1
                if gs == "test_goods":
                    cmap[gid] = {"case_id": gid, "X_num": X, "Y_num": Y, "attr_code": code}
    return cmap


def mag_bin(x: float) -> str:
    a = abs(x)
    for i in range(len(BIN_LABELS)):
        if (BIN_EDGES[i] <= a < BIN_EDGES[i + 1]) or (i == 0 and a == 0.0):
            return BIN_LABELS[i]
    return BIN_LABELS[-1]


def build():
    deltas = json.load(open(DELTA_FILE))
    cmap = build_case_map()
    # attach delta + strata (pre-training metadata only)
    cases = []
    for cid, rec in cmap.items():
        d = deltas.get(str(cid))
        if d is None:
            continue
        dv = float(d["mean_delta"])
        if dv == 0.0:
            continue
        rec = dict(rec)
        rec["delta"] = dv
        rec["delta_sign"] = "pos" if dv > 0 else "neg"
        rec["delta_bin"] = mag_bin(dv)
        rec["stratum"] = f"{rec['delta_sign']}|{rec['delta_bin']}"
        cases.append(rec)

    by_stratum: dict[str, list] = {}
    for c in cases:
        by_stratum.setdefault(c["stratum"], []).append(c)

    selected = []
    per_stratum = {}
    for stratum in sorted(by_stratum):
        ranked = sorted(by_stratum[stratum],
                        key=lambda c: hashlib.sha256(f"{FROZEN_SEED}:{c['case_id']}".encode()).digest())
        take = ranked[:N_PER_STRATUM]
        per_stratum[stratum] = len(take)
        selected.extend(take)
    selected.sort(key=lambda c: c["case_id"])

    subset = {
        "status": "FROZEN_PRE_GPU",
        "purpose": "Surface-form stress subset for the SFT-vs-sign-only diagnostic (rule vs shortcut). Frozen before inference; selected from pre-training metadata only.",
        "source": "test_goods.json (already-opened validation); frozen semantic/method-comparison suites NOT consumed",
        "selection_rule": "stratify by delta sign x magnitude bin; hash-rank by sha256('{seed}:{case_id}'); take first N per stratum",
        "response_independent": True,
        "uses_no_sft_or_sign_output": True,
        "frozen_seed": FROZEN_SEED,
        "n_per_stratum_target": N_PER_STRATUM,
        "per_stratum_selected": per_stratum,
        "n_cases": len(selected),
        "bin_edges": BIN_EDGES,
        "cases": selected,
    }
    subset_bytes = canon(subset)
    manifest = {
        "schema_version": 1,
        "dataset": OUT_SUBSET.name,
        "status": "FROZEN_PRE_GPU",
        "frozen_seed": FROZEN_SEED,
        "inputs": {
            "delta_qwen_base.json": {"sha256": sha256_file(DELTA_FILE)},
            "test_goods.json": {"sha256": sha256_file(ROOT / "data" / "test_goods.json")},
            "trial_goods.json": {"sha256": sha256_file(ROOT / "data" / "trial_goods.json")},
        },
        "n_cases": len(selected),
        "per_stratum_selected": per_stratum,
        "sign_balance": dict(Counter(c["delta_sign"] for c in selected)),
        "output_sha256": sha256_bytes(subset_bytes),
        "construction_script": "data/surface_form_stress/build_surface_form_subset.py",
    }
    return subset_bytes, canon(manifest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    subset_bytes, manifest_bytes = build()
    if args.check:
        bad = []
        if not OUT_SUBSET.exists() or OUT_SUBSET.read_bytes() != subset_bytes:
            bad.append("subset")
        if not OUT_MANIFEST.exists() or OUT_MANIFEST.read_bytes() != manifest_bytes:
            bad.append("manifest")
        if bad:
            raise SystemExit("CHECK FAILED: " + ", ".join(bad))
        print("CHECK PASSED: surface-form subset regenerates byte-identically.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SUBSET.write_bytes(subset_bytes)
    OUT_MANIFEST.write_bytes(manifest_bytes)
    subset = json.loads(subset_bytes)
    print(f"Wrote {OUT_SUBSET.relative_to(ROOT)}  ({subset['n_cases']} cases)")
    print(f"  per stratum: {subset['per_stratum_selected']}")


if __name__ == "__main__":
    main()
