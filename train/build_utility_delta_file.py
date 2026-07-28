#!/usr/bin/env python3
"""
Build a GRPO delta file from a model's own estimated goods utilities.

The training reward expects case_id -> {"mean_delta": U_X - U_Y}.  The
consensus delta file uses frontier-model utilities; this script creates the
same shape from a single utility CSV, e.g. Qwen-7B's baseline Model A utility
table.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


GOODS_RUN_ORDER = ["trial_goods", "test_goods", "remaining_goods"]


UtilityKey = Tuple[int, int, int]


def decode_attr(attr_code: int) -> Tuple[int, int, int, int]:
    i = attr_code % 3
    attr_code //= 3
    j = attr_code % 3
    attr_code //= 3
    k = attr_code % 3
    attr_code //= 3
    l = attr_code % 3
    return i, j, k, l


def load_utilities(path: Path) -> Dict[UtilityKey, float]:
    utilities: Dict[UtilityKey, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["index"]), int(row["attr_1"]), int(row["attr_2"]))
            utilities[key] = float(row["utility"])
    return utilities


def iter_cases(data_dir: Path) -> Iterable[Tuple[int, int, int, Tuple[int, int, int, int]]]:
    case_id = 0
    for goods_stem in GOODS_RUN_ORDER:
        goods_path = data_dir / f"{goods_stem}.json"
        with goods_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        for entry in raw:
            if not (isinstance(entry, list) and len(entry) >= 3 and isinstance(entry[2], list)):
                continue
            x_num, y_num, attr_list = entry[0], entry[1], entry[2]
            for attr_code in attr_list:
                case_id += 1
                yield case_id, int(x_num), int(y_num), decode_attr(int(attr_code))


def build_delta_file(data_dir: Path, utility_csv: Path) -> Dict[str, dict]:
    utilities = load_utilities(utility_csv)
    deltas: Dict[str, dict] = {}

    for case_id, x_num, y_num, (i, j, k, l) in iter_cases(data_dir):
        u_x = utilities[(x_num + 1, i + 1, j + 1)]
        u_y = utilities[(y_num + 1, k + 1, l + 1)]
        delta = u_x - u_y
        deltas[str(case_id)] = {
            "mean_delta": delta,
            "source": "single_model_utility",
            "utility_model": utility_csv.stem,
            "U_X": u_x,
            "U_Y": u_y,
        }

    return deltas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data", help="Directory containing trial/test/remaining goods JSON files.")
    parser.add_argument(
        "--utility_csv",
        default="baseline/Qwen-7B/Model_1/Qwen-7B_utility_of_each_goods_Model_A.csv",
        help="Utility CSV with columns index, attr_1, attr_2, utility.",
    )
    parser.add_argument(
        "--output",
        default="data/deltas/delta_qwen_base_model_a.json",
        help="Output delta JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    utility_csv = Path(args.utility_csv)
    output = Path(args.output)

    deltas = build_delta_file(data_dir, utility_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(deltas, f, indent=2)
        f.write("\n")

    nonzero = sum(1 for row in deltas.values() if row["mean_delta"] != 0.0)
    print(f"Wrote {len(deltas)} deltas to {output} ({nonzero} nonzero).")


if __name__ == "__main__":
    main()
