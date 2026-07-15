#!/usr/bin/env python3
"""Measure attribute-profile sensitivity on the held-out GRPO evaluation.

The primary specification compares the two held-out configurations for the
same unordered goods pair and the same endowment perspective. First
differencing removes a pair-by-perspective fixed effect, leaving only changes
in the eight non-reference 3x3 ordinal attribute profiles.

The script also reports an item-fixed-effect levels diagnostic, grouped
cross-validation, and direct answer-flip rates. It intentionally depends only
on NumPy and the Python standard library.

Usage:
    python eval/analyze_attribute_effects.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES = [(i, j) for i in range(1, 4) for j in range(1, 4) if (i, j) != (1, 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--x-file",
        default="baseline/Qwen-7B-GRPO/loss_aversion_X.json",
        help="X-perspective held-out response file, relative to the project root.",
    )
    parser.add_argument(
        "--y-file",
        default="baseline/Qwen-7B-GRPO/loss_aversion_Y.json",
        help="Y-perspective held-out response file, relative to the project root.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Pair-grouped CV folds.")
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_by_case(path: Path) -> Dict[int, dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    result = {int(row["case_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate case_id values in {path}")
    return result


def no_probability(row: dict) -> float:
    probabilities = row["Yes / No prob"]
    if len(probabilities) != 2:
        raise ValueError("Expected [P(Yes), P(No)]")
    return float(probabilities[1])


def attribute_difference(attr: Sequence[int]) -> np.ndarray:
    """Return endowed-minus-offered indicators for 8 non-reference profiles."""
    i, j, k, l = (int(value) for value in attr)
    endowed_profile = 3 * i + j
    offered_profile = 3 * k + l
    difference = np.zeros(8, dtype=float)
    if endowed_profile:
        difference[endowed_profile - 1] += 1.0
    if offered_profile:
        difference[offered_profile - 1] -= 1.0
    return difference


def prepare_cases(x_rows: Dict[int, dict], y_rows: Dict[int, dict]) -> List[dict]:
    case_ids = sorted(set(x_rows) & set(y_rows))
    if len(case_ids) != len(x_rows) or len(case_ids) != len(y_rows):
        raise ValueError("X and Y files do not contain identical case_id sets")

    cases = []
    for case_id in case_ids:
        row_x = x_rows[case_id]
        row_y = y_rows[case_id]
        x_num = int(row_x["X_num"])
        y_num = int(row_x["Y_num"])
        if row_y["X_num"] != x_num or row_y["Y_num"] != y_num:
            raise ValueError(f"X/Y metadata mismatch for case {case_id}")
        pair = (min(x_num, y_num), max(x_num, y_num))
        cases.append(
            {
                "case_id": case_id,
                "pair": pair,
                "x_num": x_num,
                "y_num": y_num,
                "attr_difference": attribute_difference(row_x["attr"]),
                "no_x": no_probability(row_x),
                "no_y": no_probability(row_y),
            }
        )
    return cases


def ols(design: np.ndarray, outcome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residuals = outcome - design @ coefficients
    return coefficients, residuals


def cluster_covariance(
    design: np.ndarray,
    residuals: np.ndarray,
    clusters: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """One-way cluster-robust covariance with the usual finite-sample correction."""
    grouped: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped[cluster].append(index)

    n_obs, n_params = design.shape
    n_clusters = len(grouped)
    bread = np.linalg.pinv(design.T @ design)
    meat = np.zeros((n_params, n_params), dtype=float)
    for indices in grouped.values():
        group_indices = np.asarray(indices)
        score = design[group_indices].T @ residuals[group_indices]
        meat += np.outer(score, score)

    correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
    return correction * bread @ meat @ bread


def leave_one_good_out_covariance(
    design: np.ndarray,
    outcome: np.ndarray,
    pairs: np.ndarray,
    parameter_indices: np.ndarray,
    num_goods: int,
) -> np.ndarray:
    """Delete every observation involving each good and jackknife the target coefficients."""
    estimates = []
    for good in range(num_goods):
        keep = (pairs[:, 0] != good) & (pairs[:, 1] != good)
        coefficients = np.linalg.lstsq(design[keep], outcome[keep], rcond=None)[0]
        estimates.append(coefficients[parameter_indices])

    estimates_array = np.asarray(estimates)
    mean_estimate = estimates_array.mean(axis=0)
    centered = estimates_array - mean_estimate
    return ((num_goods - 1) / num_goods) * centered.T @ centered


def chi_square_df8_log10_survival(statistic: float) -> float:
    """Log10 survival probability for chi-square(8), using its finite sum."""
    if statistic <= 0.0:
        return 0.0
    half = statistic / 2.0
    log_terms = [j * math.log(half) - math.lgamma(j + 1) for j in range(4)]
    maximum = max(log_terms)
    log_survival = -half + maximum + math.log(
        sum(math.exp(term - maximum) for term in log_terms)
    )
    return log_survival / math.log(10.0)


def joint_wald(coefficients: np.ndarray, covariance: np.ndarray) -> dict:
    statistic = float(coefficients @ np.linalg.pinv(covariance) @ coefficients)
    return {
        "chi_square": statistic,
        "df": 8,
        "log10_p": chi_square_df8_log10_survival(statistic),
    }


def paired_first_difference(cases: Sequence[dict], num_goods: int) -> dict:
    """Estimate attribute effects after removing pair-by-perspective fixed effects."""
    by_pair: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for case in cases:
        by_pair[case["pair"]].append(case)

    design_rows = []
    outcomes = []
    clusters = []
    x_probability_changes = []
    y_probability_changes = []
    x_flips = 0
    y_flips = 0
    either_flips = 0
    both_flips = 0

    for pair, pair_cases in sorted(by_pair.items()):
        if len(pair_cases) != 2:
            raise ValueError(f"Expected exactly 2 held-out configurations for pair {pair}")
        first, second = sorted(pair_cases, key=lambda row: row["case_id"])
        delta_design = second["attr_difference"] - first["attr_difference"]
        delta_x = second["no_x"] - first["no_x"]
        delta_y = second["no_y"] - first["no_y"]

        # The Y perspective reverses endowed and offered profiles.
        design_rows.extend((delta_design, -delta_design))
        outcomes.extend((delta_x, delta_y))
        clusters.extend((pair, pair))

        x_probability_changes.append(abs(delta_x))
        y_probability_changes.append(abs(delta_y))
        flip_x = (first["no_x"] >= 0.5) != (second["no_x"] >= 0.5)
        flip_y = (first["no_y"] >= 0.5) != (second["no_y"] >= 0.5)
        x_flips += int(flip_x)
        y_flips += int(flip_y)
        either_flips += int(flip_x or flip_y)
        both_flips += int(flip_x and flip_y)

    design = np.asarray(design_rows)
    outcome = np.asarray(outcomes)
    pair_array = np.asarray(clusters)
    coefficients, residuals = ols(design, outcome)

    pair_covariance = cluster_covariance(design, residuals, clusters)
    pair_standard_errors = np.sqrt(np.maximum(np.diag(pair_covariance), 0.0))
    parameter_indices = np.arange(8)
    good_covariance = leave_one_good_out_covariance(
        design, outcome, pair_array, parameter_indices, num_goods
    )
    good_standard_errors = np.sqrt(np.maximum(np.diag(good_covariance), 0.0))

    residual_sum_squares = float(residuals @ residuals)
    total_sum_squares = float(outcome @ outcome)
    pooled_changes = x_probability_changes + y_probability_changes
    n_pairs = len(by_pair)

    effects = []
    for profile, estimate, pair_se, good_se in zip(
        PROFILES, coefficients, pair_standard_errors, good_standard_errors
    ):
        effects.append(
            {
                "profile": str(profile),
                "estimate_probability_points": float(estimate),
                "pair_cluster_se": float(pair_se),
                "pair_cluster_z": float(estimate / pair_se),
                "leave_one_good_out_se": float(good_se),
                "leave_one_good_out_z": float(estimate / good_se),
            }
        )

    return {
        "observations_after_differencing": len(outcome),
        "unordered_pair_clusters": n_pairs,
        "within_r_squared": 1.0 - residual_sum_squares / total_sum_squares,
        "effects": effects,
        "joint_wald_pair_clustered": joint_wald(coefficients, pair_covariance),
        "joint_wald_leave_one_good_out": joint_wald(coefficients, good_covariance),
        "sensitivity": {
            "goods_pairs": n_pairs,
            "perspective_comparisons": 2 * n_pairs,
            "mean_absolute_probability_change": float(np.mean(pooled_changes)),
            "median_absolute_probability_change": float(np.median(pooled_changes)),
            "x_perspective_flip_count": x_flips,
            "x_perspective_flip_rate": x_flips / n_pairs,
            "y_perspective_flip_count": y_flips,
            "y_perspective_flip_rate": y_flips / n_pairs,
            "total_perspective_flip_count": x_flips + y_flips,
            "either_perspective_flip_count": either_flips,
            "either_perspective_flip_rate": either_flips / n_pairs,
            "both_perspectives_flip_count": both_flips,
            "both_perspectives_flip_rate": both_flips / n_pairs,
        },
    }


def levels_design(cases: Sequence[dict], num_goods: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build item-difference FE plus attribute-difference indicators in levels."""
    n_parameters = 1 + (num_goods - 1) + 8
    design = np.zeros((2 * len(cases), n_parameters), dtype=float)
    outcome = np.zeros(2 * len(cases), dtype=float)
    pairs = []

    for case_index, case in enumerate(cases):
        x_num = case["x_num"]
        y_num = case["y_num"]
        attr = case["attr_difference"]
        specifications = (
            (x_num, y_num, attr, case["no_x"]),
            (y_num, x_num, -attr, case["no_y"]),
        )
        for perspective, (endowed, offered, attr_diff, no_prob) in enumerate(specifications):
            row = 2 * case_index + perspective
            design[row, 0] = 1.0
            if endowed:
                design[row, endowed] += 1.0
            if offered:
                design[row, offered] -= 1.0
            design[row, num_goods:] = attr_diff
            outcome[row] = no_prob
            pairs.append(case["pair"])

    return design, outcome, np.asarray(pairs)


def grouped_cv_mse(
    full_design: np.ndarray,
    restricted_design: np.ndarray,
    outcome: np.ndarray,
    pairs: np.ndarray,
    folds: int,
) -> Tuple[float, float]:
    unique_pairs = sorted({tuple(pair) for pair in pairs})
    fold_by_pair = {pair: index % folds for index, pair in enumerate(unique_pairs)}
    full_squared_error = 0.0
    restricted_squared_error = 0.0
    count = 0

    for fold in range(folds):
        test = np.asarray([fold_by_pair[tuple(pair)] == fold for pair in pairs])
        train = ~test
        full_coefficients = np.linalg.lstsq(
            full_design[train], outcome[train], rcond=None
        )[0]
        restricted_coefficients = np.linalg.lstsq(
            restricted_design[train], outcome[train], rcond=None
        )[0]
        full_residuals = outcome[test] - full_design[test] @ full_coefficients
        restricted_residuals = (
            outcome[test] - restricted_design[test] @ restricted_coefficients
        )
        full_squared_error += float(full_residuals @ full_residuals)
        restricted_squared_error += float(restricted_residuals @ restricted_residuals)
        count += int(test.sum())

    return full_squared_error / count, restricted_squared_error / count


def levels_diagnostic(cases: Sequence[dict], num_goods: int, folds: int) -> dict:
    design, outcome, pairs = levels_design(cases, num_goods)
    restricted_design = design[:, :num_goods]
    full_coefficients, full_residuals = ols(design, outcome)
    _, restricted_residuals = ols(restricted_design, outcome)

    full_rss = float(full_residuals @ full_residuals)
    restricted_rss = float(restricted_residuals @ restricted_residuals)
    centered_tss = float(((outcome - outcome.mean()) ** 2).sum())
    full_cv_mse, restricted_cv_mse = grouped_cv_mse(
        design, restricted_design, outcome, pairs, folds
    )

    attribute_indices = np.arange(num_goods, num_goods + 8)
    good_covariance = leave_one_good_out_covariance(
        design, outcome, pairs, attribute_indices, num_goods
    )
    attribute_coefficients = full_coefficients[attribute_indices]

    return {
        "observations": len(outcome),
        "item_only_r_squared": 1.0 - restricted_rss / centered_tss,
        "item_plus_attribute_r_squared": 1.0 - full_rss / centered_tss,
        "attribute_partial_r_squared": (restricted_rss - full_rss) / restricted_rss,
        "pair_grouped_cv_folds": folds,
        "item_only_cv_mse": restricted_cv_mse,
        "item_plus_attribute_cv_mse": full_cv_mse,
        "cv_mse_reduction": (restricted_cv_mse - full_cv_mse) / restricted_cv_mse,
        "joint_wald_leave_one_good_out": joint_wald(
            attribute_coefficients, good_covariance
        ),
    }


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    x_rows = load_by_case(resolve(args.x_file))
    y_rows = load_by_case(resolve(args.y_file))
    cases = prepare_cases(x_rows, y_rows)
    num_goods = 1 + max(max(case["pair"]) for case in cases)

    results = {
        "inputs": {"x_file": args.x_file, "y_file": args.y_file},
        "sample": {
            "goods": num_goods,
            "held_out_configurations": len(cases),
            "perspective_observations": 2 * len(cases),
            "unordered_goods_pairs": len({case["pair"] for case in cases}),
        },
        "paired_first_difference": paired_first_difference(cases, num_goods),
        "levels_item_fixed_effects": levels_diagnostic(cases, num_goods, args.folds),
        "interpretation": {
            "estimand": "generic ordinal 3x3 attribute-level combinations",
            "not_estimand": "separate named semantic attributes such as flavor or scent",
            "scope": "held-out configurations of familiar goods and goods pairs",
        },
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
