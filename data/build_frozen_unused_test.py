#!/usr/bin/env python3
"""Build the frozen prospective unused-configuration test set.

For every one of the 4,945 main goods pairs:

1. remove the 2 codes in ``test_goods.json`` and the 10 codes in
   ``remaining_goods.json``;
2. deterministically hash-shuffle the remaining 69 joint attribute codes;
3. take the first 10.

The hash shuffle is independent of Python's ``random`` implementation, so the
same inputs and seed produce byte-identical output on different machines.

Run from the repository root:

    python3 data/build_frozen_unused_test.py
    python3 data/build_frozen_unused_test.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_INPUT = PROJECT_ROOT / "data" / "test_goods.json"
DEFAULT_TRAIN_INPUT = PROJECT_ROOT / "data" / "remaining_goods.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "frozen_unused_test_goods.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "frozen_unused_test_goods.manifest.json"

FROZEN_SEED = 20260723
EXPECTED_PAIRS = 4_945
EXISTING_CODES_PER_PAIR = 12
AVAILABLE_CODES_PER_PAIR = 69
SELECTED_CODES_PER_PAIR = 10
N_JOINT_CODES = 81
N_ATTRIBUTE_POSITIONS = 4
N_LEVELS = 3

# These gates check gross imbalance without selecting a seed for unusually good
# balance. The committed seed passes with 10.40% and 0.387 percentage points.
MAX_JOINT_RELATIVE_DEVIATION = 0.12
MAX_MARGINAL_SHARE_DEVIATION = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the committed output and manifest regenerate exactly.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
        if any(not isinstance(code, int) or not 0 <= code < N_JOINT_CODES for code in codes):
            raise ValueError(f"{source}: pair {(x, y)} has a code outside 0..80")
        if len(set(codes)) != len(codes):
            raise ValueError(f"{source}: pair {(x, y)} has duplicate codes")
        pair = (x, y)
        if pair in pairs:
            raise ValueError(f"{source}: duplicate goods pair {pair}")
        pairs[pair] = codes
    return pairs


def shuffle_key(pair: tuple[int, int], code: int) -> bytes:
    payload = f"{FROZEN_SEED}:{pair[0]}:{pair[1]}:{code}".encode("ascii")
    return hashlib.sha256(payload).digest()


def decode_code(code: int) -> tuple[int, int, int, int]:
    levels = []
    for _ in range(N_ATTRIBUTE_POSITIONS):
        levels.append(code % N_LEVELS)
        code //= N_LEVELS
    return tuple(levels)  # type: ignore[return-value]


def build_dataset(
    test_pairs: dict[tuple[int, int], list[int]],
    train_pairs: dict[tuple[int, int], list[int]],
) -> tuple[list[list[Any]], dict[str, Any]]:
    if set(test_pairs) != set(train_pairs):
        only_test = sorted(set(test_pairs) - set(train_pairs))[:5]
        only_train = sorted(set(train_pairs) - set(test_pairs))[:5]
        raise ValueError(
            "Input pair sets differ; "
            f"test-only examples={only_test}, train-only examples={only_train}"
        )
    if len(test_pairs) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} pairs, found {len(test_pairs)}")

    selected_rows: list[list[Any]] = []
    joint_counts: Counter[int] = Counter()
    marginal_counts = [Counter() for _ in range(N_ATTRIBUTE_POSITIONS)]

    for pair in sorted(test_pairs):
        test_codes = set(test_pairs[pair])
        train_codes = set(train_pairs[pair])
        if test_codes & train_codes:
            raise ValueError(f"Existing test/train overlap for pair {pair}")
        used_codes = test_codes | train_codes
        if len(used_codes) != EXISTING_CODES_PER_PAIR:
            raise ValueError(
                f"Pair {pair}: expected {EXISTING_CODES_PER_PAIR} existing codes, "
                f"found {len(used_codes)}"
            )

        available = [code for code in range(N_JOINT_CODES) if code not in used_codes]
        if len(available) != AVAILABLE_CODES_PER_PAIR:
            raise ValueError(
                f"Pair {pair}: expected {AVAILABLE_CODES_PER_PAIR} unused codes, "
                f"found {len(available)}"
            )
        shuffled = sorted(available, key=lambda code: shuffle_key(pair, code))
        selected = shuffled[:SELECTED_CODES_PER_PAIR]

        if set(selected) & used_codes:
            raise AssertionError(f"Pair {pair}: selected codes overlap existing codes")
        if len(set(selected)) != SELECTED_CODES_PER_PAIR:
            raise AssertionError(f"Pair {pair}: selected codes are not unique")

        selected_rows.append([pair[0], pair[1], selected])
        joint_counts.update(selected)
        for code in selected:
            for position, level in enumerate(decode_code(code)):
                marginal_counts[position][level] += 1

    n_cases = EXPECTED_PAIRS * SELECTED_CODES_PER_PAIR
    expected_joint_count = n_cases / N_JOINT_CODES
    joint_relative_deviations = {
        code: abs(joint_counts[code] - expected_joint_count) / expected_joint_count
        for code in range(N_JOINT_CODES)
    }
    max_joint_relative_deviation = max(joint_relative_deviations.values())

    expected_marginal_share = 1 / N_LEVELS
    marginal_share_deviations = {
        f"position_{position + 1}_level_{level}": abs(
            marginal_counts[position][level] / n_cases - expected_marginal_share
        )
        for position in range(N_ATTRIBUTE_POSITIONS)
        for level in range(N_LEVELS)
    }
    max_marginal_share_deviation = max(marginal_share_deviations.values())
    joint_uniform_chi_square = sum(
        (joint_counts[code] - expected_joint_count) ** 2 / expected_joint_count
        for code in range(N_JOINT_CODES)
    )
    marginal_uniform_chi_square = []
    for position in range(N_ATTRIBUTE_POSITIONS):
        expected_position_count = sum(marginal_counts[position].values()) / N_LEVELS
        marginal_uniform_chi_square.append(
            sum(
                (marginal_counts[position][level] - expected_position_count) ** 2
                / expected_position_count
                for level in range(N_LEVELS)
            )
        )

    if max_joint_relative_deviation > MAX_JOINT_RELATIVE_DEVIATION:
        raise ValueError(
            "Joint-code balance gate failed: "
            f"{max_joint_relative_deviation:.6f} > {MAX_JOINT_RELATIVE_DEVIATION:.6f}"
        )
    if max_marginal_share_deviation > MAX_MARGINAL_SHARE_DEVIATION:
        raise ValueError(
            "Marginal-level balance gate failed: "
            f"{max_marginal_share_deviation:.6f} > "
            f"{MAX_MARGINAL_SHARE_DEVIATION:.6f}"
        )

    validation = {
        "pair_count": len(selected_rows),
        "selected_codes_per_pair": SELECTED_CODES_PER_PAIR,
        "case_count": n_cases,
        "prompts_per_model": n_cases * 2,
        "existing_codes_excluded_per_pair": EXISTING_CODES_PER_PAIR,
        "unused_candidates_per_pair": AVAILABLE_CODES_PER_PAIR,
        "zero_overlap_with_test_goods": True,
        "zero_overlap_with_remaining_goods": True,
        "all_selected_codes_unique_within_pair": True,
        "all_pairs_have_exactly_10_selected_codes": True,
        "joint_code_counts_0_to_80": [joint_counts[code] for code in range(N_JOINT_CODES)],
        "joint_code_count_min": min(joint_counts.values()),
        "joint_code_count_max": max(joint_counts.values()),
        "joint_code_expected_count": expected_joint_count,
        "joint_uniform_chi_square": joint_uniform_chi_square,
        "joint_uniform_chi_square_df": N_JOINT_CODES - 1,
        "max_joint_relative_deviation": max_joint_relative_deviation,
        "max_joint_relative_deviation_gate": MAX_JOINT_RELATIVE_DEVIATION,
        "attribute_position_level_counts": [
            [marginal_counts[position][level] for level in range(N_LEVELS)]
            for position in range(N_ATTRIBUTE_POSITIONS)
        ],
        "attribute_position_uniform_chi_square": marginal_uniform_chi_square,
        "attribute_position_uniform_chi_square_df_each": N_LEVELS - 1,
        "max_marginal_share_deviation": max_marginal_share_deviation,
        "max_marginal_share_deviation_gate": MAX_MARGINAL_SHARE_DEVIATION,
        "balance_gates_pass": True,
    }
    return selected_rows, validation


def build_manifest(
    args: argparse.Namespace,
    output_bytes: bytes,
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": args.output.name,
        "status": "FROZEN_PROSPECTIVE_UNEVALUATED",
        "frozen_on": "2026-07-23",
        "purpose": (
            "Supplemental within-benchmark generalization test on previously unused "
            "attribute configurations for the same 100 goods and 4,945 goods pairs."
        ),
        "selection": {
            "algorithm": "sha256_rank_shuffle_v1",
            "seed": FROZEN_SEED,
            "candidate_codes": "integers 0..80 excluding each pair's 12 existing codes",
            "rule": "sort the remaining 69 by SHA-256 key and take the first 10",
            "hash_payload": "{seed}:{x}:{y}:{code}",
        },
        "inputs": {
            args.test_input.name: {
                "role": "existing validation configurations; 2 codes per pair",
                "sha256": sha256_file(args.test_input),
            },
            args.train_input.name: {
                "role": "existing training configurations; 10 codes per pair",
                "sha256": sha256_file(args.train_input),
            },
        },
        "output": {
            "path": str(args.output.relative_to(PROJECT_ROOT)),
            "sha256": sha256_bytes(output_bytes),
        },
        "validation": validation,
        "freeze_policy": {
            "checkpoint_selection": "PROHIBITED",
            "training_or_reward_construction": "PROHIBITED",
            "permitted_primary_models": [
                "Qwen-7B-Base-Local",
                "Qwen-7B-GRPO-qd-seed1-ckpt2000",
                "Qwen-7B-GRPO-qd-seed2-ckpt6000",
            ],
            "exploratory_model_rule": (
                "Other already-trained checkpoints may be evaluated only if labelled "
                "exploratory and may not change the frozen selections or confirmatory verdict."
            ),
            "scope": (
                "Same goods and goods-pair identities; this is not an unseen-goods OOD test."
            ),
        },
    }


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.test_input = args.test_input.resolve()
    args.train_input = args.train_input.resolve()
    args.output = args.output.resolve()
    args.manifest = args.manifest.resolve()

    test_pairs = parse_pair_map(load_json(args.test_input), args.test_input, expected_codes=2)
    train_pairs = parse_pair_map(load_json(args.train_input), args.train_input, expected_codes=10)
    dataset, validation = build_dataset(test_pairs, train_pairs)
    output_bytes = canonical_json_bytes(dataset)
    manifest = build_manifest(args, output_bytes, validation)
    manifest_bytes = canonical_json_bytes(manifest)

    if args.check:
        if not args.output.exists() or not args.manifest.exists():
            raise SystemExit("CHECK FAILED: output or manifest is missing")
        if args.output.read_bytes() != output_bytes:
            raise SystemExit("CHECK FAILED: dataset does not match deterministic regeneration")
        if args.manifest.read_bytes() != manifest_bytes:
            raise SystemExit("CHECK FAILED: manifest does not match deterministic regeneration")
        print("CHECK PASSED")
    else:
        write_bytes(args.output, output_bytes)
        write_bytes(args.manifest, manifest_bytes)
        print(f"Wrote {args.output.relative_to(PROJECT_ROOT)}")
        print(f"Wrote {args.manifest.relative_to(PROJECT_ROOT)}")

    print(
        f"{validation['pair_count']:,} pairs; "
        f"{validation['case_count']:,} cases; "
        f"{validation['prompts_per_model']:,} X/Y prompts per model"
    )
    print(
        "Joint-code counts "
        f"{validation['joint_code_count_min']}..{validation['joint_code_count_max']}; "
        "max relative deviation "
        f"{validation['max_joint_relative_deviation']:.3%}"
    )
    print(
        "Attribute-position counts "
        f"{validation['attribute_position_level_counts']}; "
        "max marginal share deviation "
        f"{validation['max_marginal_share_deviation']:.3%}"
    )
    print("Zero overlap with all 12 existing codes per pair: PASSED")


if __name__ == "__main__":
    main()
