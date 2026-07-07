#!/usr/bin/env python3
"""
Build delta_qwen_base.json — utility differences δ = U_X - U_Y
using Qwen-7B-Instruct's own NLS utility estimates (baseline treatment, Model A).

Output format matches delta_consensus_v3.json:
  { "case_id_str": { "mean_delta": float, "source": "qwen_base" }, ... }

Case IDs and attr encoding follow the same logic as prompt_builder.py.
"""

import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # lambda-zero/

# ── Load Qwen utility table ─────────────────────────────────────────────────
util_path = ROOT / "baseline/Qwen-7B/Model_1/Qwen-7B_utility_of_each_goods_Model_A.csv"
df = pd.read_csv(util_path)

# Build lookup: (item_index, attr_1, attr_2) -> utility
# item_index is 1-based in the table; attr values are 1,2,3
util_lookup = {}
for _, row in df.iterrows():
    key = (int(row["index"]), int(row["attr_1"]), int(row["attr_2"]))
    util_lookup[key] = float(row["utility"])

print(f"Loaded {len(util_lookup)} utility entries for {df['index'].nunique()} items")

# ── Decode attr code (matches prompt_builder.py decode_attr) ────────────────
def decode_attr(attr_code: int):
    i = attr_code % 3; attr_code //= 3
    j = attr_code % 3; attr_code //= 3
    k = attr_code % 3; attr_code //= 3
    l = attr_code % 3
    return i, j, k, l  # 0-indexed; add 1 for utility table lookup

# ── Process each goods file in run order ────────────────────────────────────
GOODS_FILES = ["trial_goods", "test_goods", "remaining_goods"]
DATA_DIR = ROOT / "data"

deltas = {}
global_id = 0
missing = 0

for gs in GOODS_FILES:
    path = DATA_DIR / f"{gs}.json"
    with open(path) as f:
        goods_data = json.load(f)

    local_id = 0
    for entry in goods_data:
        X_num, Y_num, attr_list = entry[0], entry[1], entry[2]
        if not isinstance(attr_list, list):
            attr_list = [attr_list]

        for attr_code in attr_list:
            local_id += 1
            global_id += 1
            key = str(global_id)

            i, j, k, l = decode_attr(attr_code)
            # attr values in utility table are 1-indexed
            ux_key = (X_num + 1, i + 1, j + 1)
            uy_key = (Y_num + 1, k + 1, l + 1)

            ux = util_lookup.get(ux_key)
            uy = util_lookup.get(uy_key)

            if ux is None or uy is None:
                missing += 1
                continue

            deltas[key] = {
                "mean_delta": ux - uy,
                "source": "qwen_base"
            }

    print(f"  {gs}: {local_id} cases processed (running global_id={global_id})")

print(f"\nTotal delta entries: {len(deltas)}")
print(f"Missing utility lookups: {missing}")

# ── Stats ────────────────────────────────────────────────────────────────────
import statistics
vals = [v["mean_delta"] for v in deltas.values()]
pos = sum(1 for v in vals if v > 0)
neg = sum(1 for v in vals if v < 0)
print(f"δ > 0: {pos} ({pos/len(vals)*100:.1f}%)  δ < 0: {neg} ({neg/len(vals)*100:.1f}%)")
print(f"δ mean={statistics.mean(vals):.4f}  std={statistics.stdev(vals):.4f}  "
      f"min={min(vals):.4f}  max={max(vals):.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
out = DATA_DIR / "deltas/delta_qwen_base.json"
with open(out, "w") as f:
    json.dump(deltas, f)
print(f"\nSaved: {out}")
