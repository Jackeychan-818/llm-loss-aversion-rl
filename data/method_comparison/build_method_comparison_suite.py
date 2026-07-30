#!/usr/bin/env python3
"""Build the FROZEN, UNTOUCHED method-comparison suite (Task 1).

Purpose
-------
A new evaluation suite that has NOT been used for training, reward
construction, checkpoint selection, or any prior method development. Its only
intended use is the *final* comparison among the matched base, magnitude-
weighted GRPO, matched SFT, and sign-only / scale-matched GRPO controls.

Why this can be genuinely untouched
-----------------------------------
Every one of the 4,945 main goods pairs has 81 joint attribute codes (3^4).
The already-used codes per pair are:

    test_goods.json          2 codes  (validation)
    remaining_goods.json    10 codes  (training)
    frozen_unused_test...   10 codes  (already-OPENED prospective suite)
    -----------------------------------
    total used             22 codes

leaving 81 - 22 = 59 codes per pair that have never appeared in any split,
reward file, or opened suite. This generator selects K of those 59 per pair by
a deterministic SHA-256 rank shuffle with a NEW frozen seed, so the suite is
byte-reproducible from tracked inputs alone and provably disjoint from all
prior configurations.

The 50-good OOD suite (`data/ood_new_goods_50.json`) uses a *different* goods
population (unseen goods), so it shares no (pair, code) tuples with the 100
main goods by construction; the manifest records this explicitly.

Determinism
-----------
Selection depends ONLY on the three tracked config files and FROZEN_SEED. The
SHA-256 rank shuffle is independent of Python's ``random`` module, so the same
inputs and seed produce byte-identical output on any machine. Predicted-delta
stratification (from the frozen base utility table) is written to a SEPARATE
enrichment file and is not part of the frozen suite hash.

Usage (from the repository root):

    python3 data/method_comparison/build_method_comparison_suite.py
    python3 data/method_comparison/build_method_comparison_suite.py --check

`--check` regenerates in memory and asserts byte-identity with the committed
suite and manifest, and hard-fails on any drift.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "method_comparison"

TEST_INPUT = DATA_DIR / "test_goods.json"
TRAIN_INPUT = DATA_DIR / "remaining_goods.json"
FROZEN_UNUSED_INPUT = DATA_DIR / "frozen_unused_test_goods.json"
OOD_INPUT = DATA_DIR / "ood_new_goods_50.json"
UTILITY_CSV = PROJECT_ROOT / "baseline" / "Qwen-7B" / "Model_1" / "Qwen-7B_utility_of_each_goods_Model_A.csv"

SUITE_OUTPUT = OUT_DIR / "method_comparison_suite.json"
MANIFEST_OUTPUT = OUT_DIR / "method_comparison_suite.manifest.json"
STRATA_OUTPUT = OUT_DIR / "method_comparison_strata.json"

# NEW frozen seed, distinct from the frozen-unused seed (20260723).
FROZEN_SEED = 20260730
FROZEN_ON = "2026-07-30"

EXPECTED_PAIRS = 4_945
N_JOINT_CODES = 81
N_ATTRIBUTE_POSITIONS = 4
N_LEVELS = 3

TEST_CODES_PER_PAIR = 2
TRAIN_CODES_PER_PAIR = 10
FROZEN_UNUSED_CODES_PER_PAIR = 10
USED_CODES_PER_PAIR = (
    TEST_CODES_PER_PAIR + TRAIN_CODES_PER_PAIR + FROZEN_UNUSED_CODES_PER_PAIR
)  # 22
AVAILABLE_CODES_PER_PAIR = N_JOINT_CODES - USED_CODES_PER_PAIR  # 59

# Number of previously-unused configurations selected per pair for this suite.
# Chosen well below the 59 available so a further untouched suite is still
# possible if ever needed.
SELECTED_CODES_PER_PAIR = 4

# Balance gates: verify the selection is not grossly skewed. They only reject
# gross imbalance; they do NOT search a seed for unusually good balance.
MAX_JOINT_RELATIVE_DEVIATION = 0.15
MAX_MARGINAL_SHARE_DEVIATION = 0.01

# Case-ID namespace: a large offset that cannot collide with the 1..59,400
# trial/test/remaining global IDs or the frozen-unused suite.
CASE_ID_OFFSET = 2_000_000

# Predicted-|delta| strata edges (for reporting; not used in selection).
DELTA_BIN_EDGES = [0.0, 0.5, 1.0, 2.0, float("inf")]
DELTA_BIN_LABELS = ["|d|<=0.5", "0.5<|d|<=1.0", "1.0<|d|<=2.0", "|d|>2.0"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true",
                   help="Verify committed suite/manifest regenerate byte-identically.")
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


def parse_pair_map(raw: Any, source: Path, expected_codes: int) -> dict[tuple[int, int], list[int]]:
    if not isinstance(raw, list):
        raise ValueError(f"{source}: top-level value must be a list")
    pairs: dict[tuple[int, int], list[int]] = {}
    for row_number, entry in enumerate(raw, start=1):
        if not isinstance(entry, list) or len(entry) != 3:
            raise ValueError(f"{source}: row {row_number} must be [X, Y, [codes]]")
        x, y, codes = entry
        if not isinstance(x, int) or not isinstance(y, int) or not 0 <= x < y < 100:
            raise ValueError(f"{source}: invalid goods pair at row {row_number}: {(x, y)}")
        if not isinstance(codes, list) or len(codes) != expected_codes:
            raise ValueError(
                f"{source}: pair {(x, y)} must have {expected_codes} codes, "
                f"found {len(codes) if isinstance(codes, list) else 'non-list'}"
            )
        if any(not isinstance(c, int) or not 0 <= c < N_JOINT_CODES for c in codes):
            raise ValueError(f"{source}: pair {(x, y)} has a code outside 0..80")
        if len(set(codes)) != len(codes):
            raise ValueError(f"{source}: pair {(x, y)} has duplicate codes")
        if (x, y) in pairs:
            raise ValueError(f"{source}: duplicate goods pair {(x, y)}")
        pairs[(x, y)] = codes
    return pairs


def shuffle_key(pair: tuple[int, int], code: int) -> bytes:
    payload = f"{FROZEN_SEED}:{pair[0]}:{pair[1]}:{code}".encode("ascii")
    return hashlib.sha256(payload).digest()


def decode_code(code: int) -> tuple[int, int, int, int]:
    """Decode a joint code into (i, j, k, l), 0-indexed. Matches prompt_builder
    / build_delta_qwen_base decode_attr: X gets (i, j), Y gets (k, l)."""
    i = code % N_LEVELS
    j = (code // N_LEVELS) % N_LEVELS
    k = (code // (N_LEVELS ** 2)) % N_LEVELS
    l = (code // (N_LEVELS ** 3)) % N_LEVELS
    return i, j, k, l


def build_suite() -> tuple[list[list[Any]], dict[str, Any]]:
    test_pairs = parse_pair_map(load_json(TEST_INPUT), TEST_INPUT, TEST_CODES_PER_PAIR)
    train_pairs = parse_pair_map(load_json(TRAIN_INPUT), TRAIN_INPUT, TRAIN_CODES_PER_PAIR)
    fu_pairs = parse_pair_map(load_json(FROZEN_UNUSED_INPUT), FROZEN_UNUSED_INPUT,
                              FROZEN_UNUSED_CODES_PER_PAIR)

    if not (set(test_pairs) == set(train_pairs) == set(fu_pairs)):
        raise ValueError("Input pair sets differ across test/remaining/frozen-unused.")
    if len(test_pairs) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} pairs, found {len(test_pairs)}")

    rows: list[list[Any]] = []
    joint_counts: Counter[int] = Counter()
    marginal_counts = [Counter() for _ in range(N_ATTRIBUTE_POSITIONS)]

    for pair in sorted(test_pairs):
        used = set(test_pairs[pair]) | set(train_pairs[pair]) | set(fu_pairs[pair])
        if len(used) != USED_CODES_PER_PAIR:
            raise ValueError(
                f"Pair {pair}: expected {USED_CODES_PER_PAIR} distinct used codes, "
                f"found {len(used)} (test/train/frozen-unused overlap detected)."
            )
        available = [c for c in range(N_JOINT_CODES) if c not in used]
        if len(available) != AVAILABLE_CODES_PER_PAIR:
            raise ValueError(
                f"Pair {pair}: expected {AVAILABLE_CODES_PER_PAIR} untouched codes, "
                f"found {len(available)}."
            )
        shuffled = sorted(available, key=lambda c: shuffle_key(pair, c))
        selected = sorted(shuffled[:SELECTED_CODES_PER_PAIR])
        if set(selected) & used:
            raise AssertionError(f"Pair {pair}: selected code overlaps a used code.")
        if len(set(selected)) != SELECTED_CODES_PER_PAIR:
            raise AssertionError(f"Pair {pair}: selected codes not unique.")
        rows.append([pair[0], pair[1], selected])
        joint_counts.update(selected)
        for c in selected:
            for pos, lvl in enumerate(decode_code(c)):
                marginal_counts[pos][lvl] += 1

    n_cases = EXPECTED_PAIRS * SELECTED_CODES_PER_PAIR
    expected_joint = n_cases / N_JOINT_CODES
    max_joint_dev = max(
        abs(joint_counts[c] - expected_joint) / expected_joint for c in range(N_JOINT_CODES)
    )
    expected_share = 1 / N_LEVELS
    max_marg_dev = max(
        abs(marginal_counts[pos][lvl] / n_cases - expected_share)
        for pos in range(N_ATTRIBUTE_POSITIONS) for lvl in range(N_LEVELS)
    )
    if max_joint_dev > MAX_JOINT_RELATIVE_DEVIATION:
        raise ValueError(f"Joint-code balance gate failed: {max_joint_dev:.4f} "
                         f"> {MAX_JOINT_RELATIVE_DEVIATION}")
    if max_marg_dev > MAX_MARGINAL_SHARE_DEVIATION:
        raise ValueError(f"Marginal-share balance gate failed: {max_marg_dev:.4f} "
                         f"> {MAX_MARGINAL_SHARE_DEVIATION}")

    validation = {
        "pair_count": len(rows),
        "selected_codes_per_pair": SELECTED_CODES_PER_PAIR,
        "case_count": n_cases,
        "prompts_per_model": n_cases * 2,
        "used_codes_per_pair": USED_CODES_PER_PAIR,
        "untouched_candidates_per_pair": AVAILABLE_CODES_PER_PAIR,
        "zero_overlap_with_test_goods": True,
        "zero_overlap_with_remaining_goods": True,
        "zero_overlap_with_frozen_unused": True,
        "ood_50_shares_no_goods": True,
        "all_selected_codes_unique_within_pair": True,
        "joint_code_counts_0_to_80": [joint_counts[c] for c in range(N_JOINT_CODES)],
        "joint_code_count_min": min(joint_counts.values()),
        "joint_code_count_max": max(joint_counts.values()),
        "joint_code_expected_count": expected_joint,
        "max_joint_relative_deviation": max_joint_dev,
        "max_joint_relative_deviation_gate": MAX_JOINT_RELATIVE_DEVIATION,
        "attribute_position_level_counts": [
            [marginal_counts[pos][lvl] for lvl in range(N_LEVELS)]
            for pos in range(N_ATTRIBUTE_POSITIONS)
        ],
        "max_marginal_share_deviation": max_marg_dev,
        "max_marginal_share_deviation_gate": MAX_MARGINAL_SHARE_DEVIATION,
        "balance_gates_pass": True,
    }
    return rows, validation


def assign_case_ids(rows: list[list[Any]]) -> list[dict[str, Any]]:
    """Deterministic stable case IDs. Cases are ordered by (X, Y, code)."""
    records: list[dict[str, Any]] = []
    cid = CASE_ID_OFFSET
    for x, y, codes in rows:
        for code in codes:
            cid += 1
            records.append({"case_id": cid, "X": x, "Y": y, "code": code})
    return records


def build_manifest(suite_bytes: bytes, validation: dict[str, Any],
                   case_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": SUITE_OUTPUT.name,
        "status": "FROZEN_UNTOUCHED_UNEVALUATED",
        "frozen_on": FROZEN_ON,
        "purpose": (
            "Final method-comparison suite for matched base, magnitude-weighted "
            "GRPO, matched SFT, and sign-only/scale-matched GRPO. Never used for "
            "training, reward construction, checkpoint selection, or method "
            "development."
        ),
        "selection": {
            "algorithm": "sha256_rank_shuffle_v1",
            "seed": FROZEN_SEED,
            "codes_per_pair": SELECTED_CODES_PER_PAIR,
            "candidate_codes": "integers 0..80 excluding each pair's 22 used codes",
            "rule": "sort untouched codes by SHA-256 key, take the first K, then sort",
            "hash_payload": "{seed}:{x}:{y}:{code}",
            "response_independent": True,
        },
        "case_id_scheme": {
            "offset": CASE_ID_OFFSET,
            "order": "sequential over (X, Y, code) ascending",
            "first_case_id": case_records[0]["case_id"],
            "last_case_id": case_records[-1]["case_id"],
            "n_case_ids": len(case_records),
            "paired_perspectives": "each case_id yields one X-endowed and one Y-endowed prompt",
        },
        "inputs": {
            TEST_INPUT.name: {"role": "validation configs (2/pair) excluded",
                              "sha256": sha256_file(TEST_INPUT)},
            TRAIN_INPUT.name: {"role": "training configs (10/pair) excluded",
                               "sha256": sha256_file(TRAIN_INPUT)},
            FROZEN_UNUSED_INPUT.name: {"role": "already-opened prospective configs (10/pair) excluded",
                                       "sha256": sha256_file(FROZEN_UNUSED_INPUT)},
            OOD_INPUT.name: {"role": "unseen-goods OOD suite; different goods population, no code overlap",
                             "sha256": sha256_file(OOD_INPUT)},
        },
        "output": {
            "path": str(SUITE_OUTPUT.relative_to(PROJECT_ROOT)),
            "sha256": sha256_bytes(suite_bytes),
        },
        "validation": validation,
        "freeze_policy": {
            "training_or_reward_construction": "PROHIBITED",
            "checkpoint_selection": "PROHIBITED",
            "method_development_or_tuning": "PROHIBITED",
            "open_rule": (
                "Open ONCE, after all per-seed checkpoint selections are frozen on "
                "test_goods validation, for the pre-registered method comparison in "
                "METHOD_COMPARISON_PROTOCOL.md."
            ),
            "scope": "Same 100 goods and 4,945 pairs; untouched configurations, not unseen goods.",
        },
    }


def compute_strata(case_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Predicted-|delta| strata from the frozen base utility table. Enrichment
    only; does not affect selection or the frozen suite hash. Returns None if
    the utility CSV is unavailable."""
    if not UTILITY_CSV.exists():
        return None
    util: dict[tuple[int, int, int], float] = {}
    with UTILITY_CSV.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            util[(int(row["index"]), int(row["attr_1"]), int(row["attr_2"]))] = float(row["utility"])

    bins = Counter()
    per_case = []
    missing = 0
    for rec in case_records:
        i, j, k, l = decode_code(rec["code"])
        ux = util.get((rec["X"] + 1, i + 1, j + 1))
        uy = util.get((rec["Y"] + 1, k + 1, l + 1))
        if ux is None or uy is None:
            missing += 1
            per_case.append({"case_id": rec["case_id"], "pred_delta": None, "bin": None})
            continue
        d = ux - uy
        ad = abs(d)
        label = DELTA_BIN_LABELS[-1]
        for bi in range(len(DELTA_BIN_LABELS)):
            if DELTA_BIN_EDGES[bi] <= ad < DELTA_BIN_EDGES[bi + 1] or (
                bi == 0 and ad == 0.0
            ):
                label = DELTA_BIN_LABELS[bi]
                break
        bins[label] += 1
        per_case.append({"case_id": rec["case_id"], "pred_delta": d, "bin": label})

    return {
        "note": (
            "Predicted delta = U_X - U_Y from the FROZEN base utility table "
            "(baseline/Qwen-7B/Model_1/Qwen-7B_utility_of_each_goods_Model_A.csv). "
            "Deterministic function of already-fitted parameters; NOT a model "
            "response and NOT used in selection. For stratified reporting only."
        ),
        "utility_csv": str(UTILITY_CSV.relative_to(PROJECT_ROOT)),
        "utility_csv_sha256": sha256_file(UTILITY_CSV),
        "bin_edges": DELTA_BIN_EDGES,
        "bin_labels": DELTA_BIN_LABELS,
        "bin_counts": {label: bins[label] for label in DELTA_BIN_LABELS},
        "missing_utility_lookups": missing,
        "per_case": per_case,
    }


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(value)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    rows, validation = build_suite()
    case_records = assign_case_ids(rows)
    suite_bytes = canonical_json_bytes(rows)
    manifest = build_manifest(suite_bytes, validation, case_records)
    manifest_bytes = canonical_json_bytes(manifest)

    if args.check:
        problems = []
        if not SUITE_OUTPUT.exists() or SUITE_OUTPUT.read_bytes() != suite_bytes:
            problems.append("suite JSON drifted or missing")
        if not MANIFEST_OUTPUT.exists() or MANIFEST_OUTPUT.read_bytes() != manifest_bytes:
            problems.append("manifest JSON drifted or missing")
        if problems:
            raise SystemExit("CHECK FAILED: " + "; ".join(problems))
        print("CHECK PASSED: suite and manifest regenerate byte-identically.")
        return

    write_bytes(SUITE_OUTPUT, suite_bytes)
    write_bytes(MANIFEST_OUTPUT, manifest_bytes)
    strata = compute_strata(case_records)
    if strata is not None:
        strata_doc = {
            "generated_from": SUITE_OUTPUT.name,
            "suite_sha256": sha256_bytes(suite_bytes),
            "frozen_seed": FROZEN_SEED,
            "generated_on_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **strata,
        }
        write_bytes(STRATA_OUTPUT, canonical_json_bytes(strata_doc))

    print(f"Wrote {SUITE_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {MANIFEST_OUTPUT.relative_to(PROJECT_ROOT)}")
    if strata is not None:
        print(f"Wrote {STRATA_OUTPUT.relative_to(PROJECT_ROOT)}  (enrichment)")
        print(f"  predicted-|delta| bins: {strata['bin_counts']}")
    else:
        print("Utility CSV absent; strata enrichment skipped.")
    print(f"{validation['pair_count']:,} pairs; {validation['case_count']:,} cases; "
          f"{validation['prompts_per_model']:,} X/Y prompts per model")
    print(f"joint-code counts {validation['joint_code_count_min']}.."
          f"{validation['joint_code_count_max']}; "
          f"max rel dev {validation['max_joint_relative_deviation']:.3%}")


if __name__ == "__main__":
    main()
