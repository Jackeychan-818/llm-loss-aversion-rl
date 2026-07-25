#!/usr/bin/env python3
"""
Pseudo-utility alignment W — a third reported quantity alongside lambda and eta.

Definition:

    w_q = u_chosen / max(u_1, u_2)
        = 1                              if the higher-utility good is chosen,
          min(u_1, u_2) / max(u_1, u_2) otherwise,
    W   = (1/N) sum_q w_q.

The frozen Model-A utility table contains strictly positive utility levels, so
the fraction is defined and lies in (0, 1]. W is a magnitude-weighted
rational-choice score: 1.0 means the model always picks the higher-utility good,
while a lower-utility choice receives that good's utility as a fraction of the
best available utility. W is DESCRIPTIVE (reported with lambda/eta); it is not a
success gate and was added after the seed pre-registration was frozen.

Reference pseudo-utility = the positive Qwen-base Model-A utility table used to
construct `data/deltas/delta_qwen_base.json`. Using one shared utility reference
makes W comparable ACROSS models — unlike each model's own fitted utilities,
which would be a different yardstick per model. Choices are read as the hard
argmax of P(Yes)/P(No) (the formula selects a definite chosen good).

Case/perspective -> chosen good (endowment task):
  X-perspective (endowed X, offered Y): "No" -> keep X (chosen X); "Yes" -> Y
  Y-perspective (endowed Y, offered X): "No" -> keep Y (chosen Y); "Yes" -> X

    python eval/pseudo_utility_alignment.py \\
        --eval_dir baseline/Qwen-7B-GRPO-qd-ckpt8000 --name qd-8k
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from sweep_partition_estimate import load_rows, choice_from_row

DEFAULT_UTILITY_FILE = (
    PROJECT_ROOT
    / "baseline"
    / "Qwen-7B"
    / "Model_1"
    / "Qwen-7B_utility_of_each_goods_Model_A.csv"
)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
GOODS_RUN_ORDER = ("trial_goods", "test_goods", "remaining_goods",
                   "frozen_unused_test_goods")


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def decode_attr(attr_code: int) -> tuple[int, int, int, int]:
    i = attr_code % 3
    attr_code //= 3
    j = attr_code % 3
    attr_code //= 3
    k = attr_code % 3
    attr_code //= 3
    l = attr_code % 3
    return i, j, k, l


def load_utility_table(path: Path) -> dict[tuple[int, int, int], float]:
    utilities = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["index"]), int(row["attr_1"]), int(row["attr_2"]))
            utility = float(row["utility"])
            if utility <= 0:
                raise ValueError(
                    f"utility must be positive for the fraction score: {key}={utility}"
                )
            utilities[key] = utility
    if not utilities:
        raise RuntimeError(f"no utilities loaded from {path}")
    return utilities


def load_case_utilities(
    data_dir: Path, utility_file: Path
) -> dict[int, tuple[float, float]]:
    utility_table = load_utility_table(utility_file)
    case_utilities = {}
    case_id = 0

    for goods_stem in GOODS_RUN_ORDER:
        with (data_dir / f"{goods_stem}.json").open("r", encoding="utf-8") as f:
            goods_data = json.load(f)

        for entry in goods_data:
            x_num, y_num, attr_values = entry[0], entry[1], entry[2]
            if not isinstance(attr_values, list):
                attr_values = [attr_values]

            for attr_code in attr_values:
                case_id += 1
                i, j, k, l = decode_attr(int(attr_code))
                x_key = (int(x_num) + 1, i + 1, j + 1)
                y_key = (int(y_num) + 1, k + 1, l + 1)
                try:
                    case_utilities[case_id] = (
                        utility_table[x_key],
                        utility_table[y_key],
                    )
                except KeyError as error:
                    raise KeyError(
                        f"missing utility for case_id={case_id}: "
                        f"X={x_key}, Y={y_key}"
                    ) from error

    return case_utilities


def score_choice(chosen: str, u_x: float, u_y: float) -> tuple[float, bool]:
    if chosen not in {"X", "Y"}:
        raise ValueError(f"chosen good must be X or Y, got {chosen!r}")
    if u_x <= 0 or u_y <= 0:
        raise ValueError(f"utilities must be positive, got u_x={u_x}, u_y={u_y}")

    chosen_utility = u_x if chosen == "X" else u_y
    best_utility = max(u_x, u_y)
    return chosen_utility / best_utility, chosen_utility == best_utility


def load_structural_estimates(estimation_file: Path) -> dict:
    with estimation_file.open("r", encoding="utf-8", newline="") as f:
        rows = {row["Parameter"]: row for row in csv.DictReader(f)}

    missing = {"lambda", "eta"} - set(rows)
    if missing:
        raise KeyError(
            f"missing structural parameters in {estimation_file}: {sorted(missing)}"
        )

    return {
        "lambda": float(rows["lambda"]["Estimate"]),
        "lambda_se": float(rows["lambda"]["Std. Err."]),
        "eta": float(rows["eta"]["Estimate"]),
        "eta_se": float(rows["eta"]["Std. Err."]),
        "estimator": "Model A (NLS), structural link scale T=1",
    }


def resolve_estimation_file(eval_dir: Path, requested: str | None) -> Path:
    if requested:
        estimation_file = resolve_project_path(requested)
        if not estimation_file.exists():
            raise FileNotFoundError(f"estimation CSV not found: {estimation_file}")
        return estimation_file

    candidates = sorted(
        (eval_dir / "Model_1").glob("*_NLS_estimation_T1(Model A).csv")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one Model-A estimation CSV under {eval_dir / 'Model_1'}, "
            f"found {len(candidates)}; pass --estimation_file explicitly"
        )
    return candidates[0]


def compute_W(
    eval_dir: Path, case_utilities: dict[int, tuple[float, float]]
) -> dict:
    xr = load_rows(eval_dir / "loss_aversion_X.json")
    yr = load_rows(eval_dir / "loss_aversion_Y.json")
    common = sorted(set(xr) & set(yr))
    ws, rational, n, skipped = [], 0, 0, 0

    for cid in common:
        utilities = case_utilities.get(cid)
        if utilities is None:
            skipped += 1
            continue

        u_x, u_y = utilities
        for persp, row in (("X", xr[cid]), ("Y", yr[cid])):
            resp = choice_from_row(row)  # argmax Yes/No
            if persp == "X":
                chosen = "X" if resp == "No" else "Y"
            else:
                chosen = "Y" if resp == "No" else "X"

            w_q, hit = score_choice(chosen, u_x, u_y)
            ws.append(w_q)
            rational += hit
            n += 1

    if n == 0:
        raise RuntimeError(
            f"no scorable choices in {eval_dir} "
            "(utility coverage or case-id overlap problem?)"
        )

    return {
        "W": sum(ws) / n,
        "N": n,
        "rational_choice_rate": rational / n,
        "cases_scored": len(common) - skipped,
        "cases_skipped_missing_utility": skipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument(
        "--utility_file",
        default=str(DEFAULT_UTILITY_FILE.relative_to(PROJECT_ROOT)),
        help="shared positive utility table used as the W reference",
    )
    ap.add_argument(
        "--data_dir",
        default=str(DEFAULT_DATA_DIR.relative_to(PROJECT_ROOT)),
        help="directory containing trial/test/remaining goods JSON files",
    )
    ap.add_argument(
        "--estimation_file",
        default=None,
        help="Model-A estimation CSV; auto-detected under EVAL_DIR/Model_1 by default",
    )
    ap.add_argument("--out", default=None, help="optional results/*.json to write")
    args = ap.parse_args()

    utility_file = resolve_project_path(args.utility_file)
    data_dir = resolve_project_path(args.data_dir)
    eval_dir = resolve_project_path(args.eval_dir)
    estimation_file = resolve_estimation_file(eval_dir, args.estimation_file)
    case_utilities = load_case_utilities(data_dir, utility_file)
    res = compute_W(eval_dir, case_utilities)
    res.update(load_structural_estimates(estimation_file))
    name = args.name or Path(args.eval_dir).name
    res.update(
        {
            "name": name,
            "eval_dir": args.eval_dir,
            "utility_file": args.utility_file,
            "data_dir": args.data_dir,
            "estimation_file": str(
                estimation_file.relative_to(PROJECT_ROOT)
                if estimation_file.is_relative_to(PROJECT_ROOT)
                else estimation_file
            ),
            "score_definition": "u_chosen / max(u_X, u_Y)",
        }
    )
    print(
        f"{name}: lambda = {res['lambda']:.4f} (SE {res['lambda_se']:.4f}), "
        f"eta = {res['eta']:.4f} (SE {res['eta_se']:.4f}), "
        f"W = {res['W']:.4f} (N={res['N']}, "
        f"rational-choice rate = {res['rational_choice_rate']:.4f})"
    )
    if args.out:
        out = resolve_project_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
            f.write("\n")
        try:
            display_path = out.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = out
        print(f"  wrote {display_path}")


if __name__ == "__main__":
    main()
