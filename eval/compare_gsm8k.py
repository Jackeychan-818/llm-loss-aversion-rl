#!/usr/bin/env python3
"""Paired comparison of two GSM8K prediction JSONL files."""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare paired GSM8K predictions")
    parser.add_argument("--base", required=True, help="Base predictions.jsonl")
    parser.add_argument("--tuned", required=True, help="Tuned predictions.jsonl")
    parser.add_argument("--output", default="results/gsm8k/comparison.json")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require_count",
        type=int,
        default=1319,
        help="Expected paired count; set 0 to disable the check.",
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_predictions(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError("Predictions file not found: {}".format(path))
    rows: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id", ""))
            if not row_id:
                raise ValueError("Missing id at {}:{}".format(path, line_number))
            if row_id in rows:
                raise ValueError("Duplicate id {} in {}".format(row_id, path))
            if not isinstance(row.get("correct"), bool):
                raise ValueError("Missing boolean correct field for {}".format(row_id))
            rows[row_id] = row
    return rows


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_ci(
    differences: Sequence[int], samples: int, seed: int
) -> Tuple[float, float]:
    if samples <= 0:
        raise ValueError("--bootstrap_samples must be positive")
    rng = random.Random(seed)
    count = len(differences)
    estimates: List[float] = []
    for _ in range(samples):
        total = 0
        for _item in range(count):
            total += differences[rng.randrange(count)]
        estimates.append(total / count)
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def exact_mcnemar_p_value(base_only: int, tuned_only: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant pairs."""
    discordant = base_only + tuned_only
    if discordant == 0:
        return 1.0
    tail_end = min(base_only, tuned_only)
    # Keep the denominator as an integer.  A float such as ``2.0 ** 1319``
    # overflows even though the final binomial tail probability is representable.
    probability = sum(
        math.comb(discordant, k) for k in range(tail_end + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    base_path = resolve(args.base)
    tuned_path = resolve(args.tuned)
    output_path = resolve(args.output)

    base = read_predictions(base_path)
    tuned = read_predictions(tuned_path)
    if set(base) != set(tuned):
        missing_from_tuned = sorted(set(base) - set(tuned))
        missing_from_base = sorted(set(tuned) - set(base))
        raise ValueError(
            "Prediction IDs do not match. Missing from tuned: {}; missing from "
            "base: {}".format(missing_from_tuned[:10], missing_from_base[:10])
        )

    ids = sorted(base)
    if args.require_count and len(ids) != args.require_count:
        raise ValueError(
            "Expected {} paired items, found {}".format(args.require_count, len(ids))
        )
    if not ids:
        raise ValueError("No paired predictions found")

    both_correct = 0
    base_only = 0
    tuned_only = 0
    both_wrong = 0
    differences: List[int] = []
    parse_failures_base = 0
    parse_failures_tuned = 0

    for row_id in ids:
        base_row = base[row_id]
        tuned_row = tuned[row_id]
        if str(base_row.get("target")) != str(tuned_row.get("target")):
            raise ValueError("Target mismatch for {}".format(row_id))
        base_correct = bool(base_row["correct"])
        tuned_correct = bool(tuned_row["correct"])
        differences.append(int(tuned_correct) - int(base_correct))
        if base_correct and tuned_correct:
            both_correct += 1
        elif base_correct:
            base_only += 1
        elif tuned_correct:
            tuned_only += 1
        else:
            both_wrong += 1
        parse_failures_base += base_row.get("predicted_answer") is None
        parse_failures_tuned += tuned_row.get("predicted_answer") is None

    count = len(ids)
    base_correct_count = both_correct + base_only
    tuned_correct_count = both_correct + tuned_only
    base_accuracy = base_correct_count / count
    tuned_accuracy = tuned_correct_count / count
    delta = tuned_accuracy - base_accuracy
    ci_low, ci_high = paired_bootstrap_ci(
        differences, args.bootstrap_samples, args.seed
    )
    mcnemar_p = exact_mcnemar_p_value(base_only, tuned_only)

    if ci_low > 0:
        conclusion = "clear_increase"
    elif ci_high < 0:
        conclusion = "clear_decrease"
    else:
        conclusion = "no_clear_change"

    result = {
        "paired_items": count,
        "base": {
            "correct": base_correct_count,
            "accuracy": base_accuracy,
            "parse_failures": parse_failures_base,
            "predictions": str(base_path),
        },
        "tuned": {
            "correct": tuned_correct_count,
            "accuracy": tuned_accuracy,
            "parse_failures": parse_failures_tuned,
            "predictions": str(tuned_path),
        },
        "paired_change": {
            "accuracy_delta": delta,
            "percentage_point_delta": 100.0 * delta,
            "paired_bootstrap_95_ci": [ci_low, ci_high],
            "paired_bootstrap_95_ci_percentage_points": [
                100.0 * ci_low,
                100.0 * ci_high,
            ],
            "bootstrap_samples": args.bootstrap_samples,
            "mcnemar_exact_p_value": mcnemar_p,
            "conclusion": conclusion,
        },
        "paired_table": {
            "both_correct": both_correct,
            "base_only_correct": base_only,
            "tuned_only_correct": tuned_only,
            "both_wrong": both_wrong,
        },
    }
    dump_json(output_path, result)

    print("=" * 68)
    print("PAIRED GSM8K COMPARISON")
    print("=" * 68)
    print("Paired items:        {:,}".format(count))
    print("Base accuracy:       {:.2%} ({}/{})".format(
        base_accuracy, base_correct_count, count
    ))
    print("Tuned accuracy:      {:.2%} ({}/{})".format(
        tuned_accuracy, tuned_correct_count, count
    ))
    print("Change:              {:+.2f} percentage points".format(100.0 * delta))
    print("Paired bootstrap CI: [{:+.2f}, {:+.2f}] pp".format(
        100.0 * ci_low, 100.0 * ci_high
    ))
    print("McNemar exact p:     {:.6g}".format(mcnemar_p))
    print("Base-only correct:   {}".format(base_only))
    print("Tuned-only correct:  {}".format(tuned_only))
    print("Conclusion:          {}".format(conclusion))
    print("Saved:               {}".format(output_path))


if __name__ == "__main__":
    main()
