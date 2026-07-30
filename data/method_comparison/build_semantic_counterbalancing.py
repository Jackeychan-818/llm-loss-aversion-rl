#!/usr/bin/env python3
"""Build the FROZEN semantic-counterbalancing component of the method-comparison
suite (Task 1).

This selects a moderate, stratified subset of the frozen untouched
method-comparison cases and freezes the surface-form variant axes that a later
GPU evaluation will expand. It materialises the subset (stable case IDs) and
the exact variant-axis definitions so expansion is deterministic; it does NOT
evaluate any model.

Variant axes (5), each preserving the underlying decision:
    1. response_mode : Yes/No     vs  keep/trade
    2. item_labels   : X/Y        vs  A/B
    3. display_order : normal      vs  reversed (offered good shown first)
    4. attr_order    : normal      vs  reversed (attribute positions swapped)
    5. paraphrase    : a fixed set of 3 prompt paraphrases

Stratification: within each predicted-|delta| bin (from the frozen base
utility table, a deterministic non-response quantity), cases are hash-ranked
with a frozen seed and the first N_PER_BIN are taken. Sample sizes are fixed in
advance and do not inspect any model response.

Usage (from repository root):
    python3 data/method_comparison/build_semantic_counterbalancing.py
    python3 data/method_comparison/build_semantic_counterbalancing.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "method_comparison"
SUITE = OUT_DIR / "method_comparison_suite.json"
STRATA = OUT_DIR / "method_comparison_strata.json"
OUTPUT = OUT_DIR / "semantic_counterbalancing.json"
MANIFEST = OUT_DIR / "semantic_counterbalancing.manifest.json"

FROZEN_SEED = 20260730
FROZEN_ON = "2026-07-30"
N_PER_BIN = 40  # per predicted-|delta| bin

RESPONSE_MODES = ["yes_no", "keep_trade"]
ITEM_LABELS = ["XY", "AB"]
DISPLAY_ORDERS = ["normal", "reversed"]
ATTR_ORDERS = ["normal", "reversed"]
PARAPHRASES = [
    "baseline_v1",   # exact eval-pipeline wording (canonical)
    "concise_v1",    # shorter phrasing, same decision
    "explicit_v1",   # spells out keep-vs-swap, same decision
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_key(case_id: int) -> bytes:
    return hashlib.sha256(f"sem:{FROZEN_SEED}:{case_id}".encode("ascii")).digest()


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    if not STRATA.exists():
        raise SystemExit("Strata file missing; run build_method_comparison_suite.py first.")
    strata = load_json(STRATA)
    labels = strata["bin_labels"]

    by_bin: dict[str, list[int]] = {label: [] for label in labels}
    for rec in strata["per_case"]:
        if rec["bin"] is not None:
            by_bin[rec["bin"]].append(rec["case_id"])

    selected: list[dict[str, Any]] = []
    per_bin_counts = {}
    for label in labels:
        ranked = sorted(by_bin[label], key=rank_key)
        take = ranked[:N_PER_BIN]
        per_bin_counts[label] = len(take)
        for cid in take:
            selected.append({"case_id": cid, "pred_delta_bin": label})
    selected.sort(key=lambda r: r["case_id"])

    n_forms = (len(RESPONSE_MODES) * len(ITEM_LABELS) * len(DISPLAY_ORDERS)
               * len(ATTR_ORDERS) * len(PARAPHRASES))

    doc = {
        "schema_version": 1,
        "status": "FROZEN_UNTOUCHED_UNEVALUATED",
        "frozen_on": FROZEN_ON,
        "frozen_seed": FROZEN_SEED,
        "parent_suite": SUITE.name,
        "selection_rule": (
            "Within each predicted-|delta| bin, hash-rank case_ids by "
            "sha256('sem:{seed}:{case_id}') and take the first N_PER_BIN. "
            "Response-independent."
        ),
        "n_per_bin": N_PER_BIN,
        "per_bin_selected": per_bin_counts,
        "n_cases": len(selected),
        "variant_axes": {
            "response_mode": RESPONSE_MODES,
            "item_labels": ITEM_LABELS,
            "display_order": DISPLAY_ORDERS,
            "attr_order": ATTR_ORDERS,
            "paraphrase": PARAPHRASES,
        },
        "forms_per_case": n_forms,
        "total_prompt_forms_per_perspective": len(selected) * n_forms,
        "primary_outcome": (
            "Whether the same final good is selected across semantically "
            "equivalent forms; also report response-token rates and paired "
            "structural outcomes where estimable."
        ),
        "permitted_models": [
            "Qwen-7B-Base-Local",
            "Qwen-7B-GRPO-qd-seed1-ckpt2000",
            "Qwen-7B-GRPO-qd-seed2-ckpt6000",
        ],
        "cases": selected,
    }
    manifest = {
        "schema_version": 1,
        "dataset": OUTPUT.name,
        "status": "FROZEN_UNTOUCHED_UNEVALUATED",
        "frozen_on": FROZEN_ON,
        "inputs": {
            SUITE.name: {"sha256": sha256_file(SUITE)},
            STRATA.name: {"sha256": sha256_file(STRATA)},
        },
        "n_cases": len(selected),
        "per_bin_selected": per_bin_counts,
        "forms_per_case": n_forms,
        "freeze_policy": {
            "template_text": "FROZEN",
            "subset_ids": "FROZEN",
            "decoding": "greedy, do_sample=False, structural link scale T=1",
            "analysis": "FROZEN before opening trained-model results",
        },
    }
    return doc, manifest


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(value)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    doc, manifest = build()
    doc_bytes = canonical_json_bytes(doc)
    # manifest records the output hash after doc is finalised
    manifest["output_sha256"] = sha256_bytes(doc_bytes)
    manifest_bytes = canonical_json_bytes(manifest)

    if args.check:
        problems = []
        if not OUTPUT.exists() or OUTPUT.read_bytes() != doc_bytes:
            problems.append("semantic subset drifted or missing")
        if not MANIFEST.exists() or MANIFEST.read_bytes() != manifest_bytes:
            problems.append("semantic manifest drifted or missing")
        if problems:
            raise SystemExit("CHECK FAILED: " + "; ".join(problems))
        print("CHECK PASSED: semantic-counterbalancing regenerates byte-identically.")
        return

    write_bytes(OUTPUT, doc_bytes)
    write_bytes(MANIFEST, manifest_bytes)
    print(f"Wrote {OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {MANIFEST.relative_to(PROJECT_ROOT)}")
    print(f"{doc['n_cases']} cases; {doc['forms_per_case']} forms/case; "
          f"per-bin {doc['per_bin_selected']}")


if __name__ == "__main__":
    main()
