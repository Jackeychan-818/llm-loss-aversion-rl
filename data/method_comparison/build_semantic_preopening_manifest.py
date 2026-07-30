#!/usr/bin/env python3
"""Separate pre-opening evaluation manifest for the semantic-counterbalancing
suite (correction #2).

Concrete checkpoint IDs are kept OUT of the frozen suite JSON. This manifest
resolves the model set to evaluate — matched base + each method family's
frozen-selected seed checkpoint — by REFERENCING the frozen selector manifests
and the frozen suite's SHA-256. Families whose selections do not yet exist (SFT,
sign-only, scale-matched) are recorded as unresolved with the selector manifest
path that will fix them.

This resolves NO model and evaluates nothing; it is the plan that will be
frozen/checked before the suite is opened once on GPU.

    python3 data/method_comparison/build_semantic_preopening_manifest.py
    python3 data/method_comparison/build_semantic_preopening_manifest.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MC = ROOT / "data" / "method_comparison"
SUITE = MC / "semantic_counterbalancing.json"
SELECT_DIR = ROOT / "results" / "checkpoint_selection"
REPLICATION = ROOT / "results" / "seed_replication_report.json"
OUT = MC / "semantic_preopening_eval_manifest.json"


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def selector_ref(name: str) -> dict:
    p = SELECT_DIR / name
    if p.exists():
        return {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)}
    return {"path": str((SELECT_DIR / name).relative_to(ROOT)), "sha256": None,
            "status": "missing"}


def build() -> dict:
    return {
        "schema_version": 1,
        "role": "pre-opening resolution of the semantic suite's model set (no eval run)",
        "frozen_suite": {
            "path": str(SUITE.relative_to(ROOT)),
            "sha256": sha256_file(SUITE) if SUITE.exists() else None,
        },
        "resolution_rule": (
            "Evaluate the matched base plus each method family's frozen-selected "
            "seed checkpoint. Concrete IDs are taken from the referenced frozen "
            "selector manifests; none are stored in the frozen suite file."
        ),
        "models": {
            "matched_base": {"model": "Qwen-7B-Base-Local", "resolved": True},
            "magnitude_grpo": {
                "resolved": True,
                "selected": {
                    "seed1": {"checkpoint": "Qwen-7B-GRPO-qd-seed1-ckpt2000",
                              "selector_manifest": selector_ref("qwen_delta_seed1.json")},
                    "seed2": {"checkpoint": "Qwen-7B-GRPO-qd-seed2-ckpt6000",
                              "selector_manifest": selector_ref("qwen_delta_seed2.json")},
                },
                "replication_report": {
                    "path": str(REPLICATION.relative_to(ROOT)),
                    "sha256": sha256_file(REPLICATION) if REPLICATION.exists() else None,
                },
            },
            "sft": {"resolved": False,
                    "selector_manifest_when_selected": "results/baselines_selection/sft_seed{N}.json (TBD)"},
            "sign_only": {"resolved": False,
                          "selector_manifest_when_selected": "results/baselines_selection/sign_seed{N}.json (TBD)"},
            "scale_matched": {"resolved": False,
                              "selector_manifest_when_selected": "results/baselines_selection/scale_seed{N}.json (TBD)"},
        },
        "open_rule": (
            "Freeze this manifest (all families resolved) BEFORE opening the "
            "semantic suite once, under METHOD_COMPARISON_PROTOCOL.md."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    doc = build()
    payload = json.dumps(doc, indent=2) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != payload:
            raise SystemExit("CHECK FAILED: pre-opening manifest drifted or missing.")
        print("CHECK PASSED: semantic pre-opening manifest byte-identical.")
        return
    OUT.write_text(payload)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
