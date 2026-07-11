#!/usr/bin/env python3
"""
Coverage audit for the Reward Design v2 components (REWARD_DESIGN_V2.md).

For each proposed non-cardinal reward component, count how many of the ACTUAL
training/eval cases can carry that signal without constructing new data:

  R_pair          — matched X/Y perspectives (should be 100%)
  R_dominance     — GENUINE Pareto: the pair's two goods must share comparable
                    attribute dimensions AND one config dominate the other
  R_monotonicity  — single-attribute +1 counterfactuals CONSTRUCTIBLE from a case
                    (an attribute currently below its top level)

R_neutral coverage depends on model-elicited anchor stability and cannot be
counted statically, so it is reported as "requires model pass".

Usage:  python data/audit_reward_v2_coverage.py
"""

import json
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent.parent
GOODS_FILES = ["trial_goods", "test_goods", "remaining_goods"]


def decode_attr(code: int):
    """attr_code (0..80) -> (i, j, k, l) 0-indexed levels; i,j = X's attrs, k,l = Y's."""
    i = code % 3; code //= 3
    j = code % 3; code //= 3
    k = code % 3; code //= 3
    l = code % 3
    return i, j, k, l


def flatten_goods(goods_full):
    """Flat list in the same order build_delta_qwen_base.py assumes: category order, then good order."""
    flat = []
    for cat, goods in goods_full.items():
        for name, attrs in goods.items():
            attr_names = tuple(attrs.keys())          # 2 attribute dimension names
            flat.append((name, attr_names))
    return flat


def main():
    goods_full = json.load(open(ROOT / "everyday_goods_full.json"))
    flat = flatten_goods(goods_full)
    n_goods = len(flat)

    # ── Do any two DISTINCT goods share both attribute-dimension names? ──
    shared_dim_pairs = 0
    for a in range(n_goods):
        for b in range(a + 1, n_goods):
            if set(flat[a][1]) == set(flat[b][1]):
                shared_dim_pairs += 1

    # ── Walk the real cases ──
    total_cases = 0
    total_pairs = 0
    genuine_pareto_cases = 0          # X,Y share dims AND one dominates
    mono_counterfactuals = 0          # constructible one-level-up variants
    cases_with_any_mono = 0

    for gs in GOODS_FILES:
        path = ROOT / "data" / f"{gs}.json"
        if not path.exists():
            continue
        data = json.load(open(path))
        for entry in data:
            X_num, Y_num, attr_list = entry[0], entry[1], entry[2]
            if not isinstance(attr_list, list):
                attr_list = [attr_list]
            total_pairs += 1
            same_dims = set(flat[X_num][1]) == set(flat[Y_num][1])
            for code in attr_list:
                total_cases += 1
                i, j, k, l = decode_attr(code)
                # monotonicity: any of the 4 attribute levels below top (level<2 0-indexed)
                improvable = sum(1 for v in (i, j, k, l) if v < 2)
                mono_counterfactuals += improvable
                if improvable:
                    cases_with_any_mono += 1
                # genuine cross-good Pareto requires shared dims (never true here)
                if same_dims:
                    xcfg, ycfg = (i, j), (k, l)
                    dom = (all(a >= b for a, b in zip(xcfg, ycfg)) and xcfg != ycfg) or \
                          (all(b >= a for a, b in zip(xcfg, ycfg)) and xcfg != ycfg)
                    if dom:
                        genuine_pareto_cases += 1

    # ── Same-good dominance CONSTRUCTION potential (synthetic) ──
    # each good has 2 attrs x 3 levels = 9 configs; count ordered dominance pairs
    cfgs = list(product(range(3), range(3)))
    dom_pairs_per_good = sum(
        1 for a in cfgs for b in cfgs
        if a != b and all(x >= y for x, y in zip(a, b))
    )
    same_good_dom_potential = dom_pairs_per_good * n_goods

    pct = lambda n: f"{n/total_cases*100:.2f}%" if total_cases else "n/a"
    print("=" * 66)
    print("Reward Design v2 — coverage audit on ACTUAL data")
    print("=" * 66)
    print(f"Goods (flat):                 {n_goods}")
    print(f"Goods-pairs sharing both attribute dims: {shared_dim_pairs}")
    print(f"Total goods-pairs in data:    {total_pairs:,}")
    print(f"Total structural cases:       {total_cases:,}")
    print("-" * 66)
    print(f"R_pair            applicable: {total_cases:,}  ({pct(total_cases)})  [every matched X/Y]")
    print(f"R_dominance  genuine in-data: {genuine_pareto_cases:,}  ({pct(genuine_pareto_cases)})  [needs shared dims]")
    print(f"R_monotonicity constructible: cases {cases_with_any_mono:,} ({pct(cases_with_any_mono)}),"
          f" total +1 variants {mono_counterfactuals:,}")
    print(f"R_neutral         applicable: requires a model pass (anchor stability) — not counted here")
    print("-" * 66)
    print(f"Same-good Pareto CONSTRUCTION potential (synthetic, off the base data):")
    print(f"   dominance-ordered config pairs per good: {dom_pairs_per_good}")
    print(f"   x {n_goods} goods = {same_good_dom_potential:,} constructible dominance comparisons")
    print("=" * 66)


if __name__ == "__main__":
    main()
