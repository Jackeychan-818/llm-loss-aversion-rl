#!/usr/bin/env python3
"""
Dataset construction for lambda-zero GRPO training.

Replicates generate_prompt() from eval/run_all_models.py exactly (baseline treatment).
Builds paired X/Y-perspective examples with pre-looked-up δ̃ values.

Case ID mapping (from baseline/Qwen-7B/completed_index.json):
  trial_goods.json      →  60 cases, IDs    1–60
  test_goods.json       →9890 cases, IDs   61–9950
  remaining_goods.json  →49450 cases, IDs 9951–59400

Usage:
    python train/prompt_builder.py data/trial_goods.json
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

logger = logging.getLogger(__name__)

# Matches GOODS_RUN_ORDER in eval/run_all_models.py exactly.
GOODS_RUN_ORDER = ["trial_goods", "test_goods", "remaining_goods"]


# ─────────────────────────────────────────────────────────────────────────────
# Goods data helpers — must match eval/run_all_models.py exactly
# ─────────────────────────────────────────────────────────────────────────────

class GoodsData:
    """Wraps everyday_goods_full.json for attribute/value lookups."""

    def __init__(self, goods_json_path: str):
        with open(goods_json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.cats: List[str] = list(self.data.keys())
        self.items: List[str] = []
        self.item_to_cat: Dict[str, str] = {}
        for cat in self.cats:
            for item in self.data[cat]:
                self.items.append(item)
                self.item_to_cat[item] = cat

    def attribute(self, item: str, num: int) -> str:
        """Return the num-th attribute name (1-indexed) of item, lowercased."""
        return list(self.data[self.item_to_cat[item]][item].keys())[num - 1].lower()

    def value(self, item: str, num: int, key: int) -> str:
        """Return the key-th value of item's num-th attribute, lowercased."""
        attr = list(self.data[self.item_to_cat[item]][item].keys())[num - 1]
        return str(self.data[self.item_to_cat[item]][item][attr][key]).lower()


def decode_attr(attr_code: int) -> Tuple[int, int, int, int]:
    """Decode base-3 encoded attr_code into (i, j, k, l). Matches run_all_models.py."""
    i = attr_code % 3; attr_code //= 3
    j = attr_code % 3; attr_code //= 3
    k = attr_code % 3; attr_code //= 3
    l = attr_code % 3
    return i, j, k, l


def generate_prompt(goods: GoodsData, X: str, Y: str,
                    i: int, j: int, k: int, l: int) -> str:
    """
    Generate the baseline endowment-effect prompt.

    Exactly matches eval/run_all_models.py generate_prompt() with TREATMENT="baseline".
    X is the endowed item (attributes i, j); Y is the offered item (attributes k, l).
    If you change this, also change run_all_models.py — they must stay in sync.
    """
    base = (
        f"Suppose that, by chance, you receive a {X.lower()} out of many possible goods "
        f"that you could have received. \n"
        f"In terms of {goods.attribute(X, 1)}, the {X.lower()} is {goods.value(X, 1, i)}. "
        f"In terms of {goods.attribute(X, 2)}, it is {goods.value(X, 2, j)}.\n"
        f"You can keep the {X.lower()} or trade it for a {Y.lower()}. \n"
        f"In terms of {goods.attribute(Y, 1)}, the {Y.lower()} is {goods.value(Y, 1, k)}. "
        f"In terms of {goods.attribute(Y, 2)}, it is {goods.value(Y, 2, l)}."
    )
    return base + "\nWould you trade it? Answer with the single word Yes or No."


# ─────────────────────────────────────────────────────────────────────────────
# Case ID offset
# ─────────────────────────────────────────────────────────────────────────────

def compute_case_id_offset(goods_stem: str, data_dir: str) -> int:
    """
    Compute the global case_id offset for a goods file by summing the case
    counts of all files that precede it in GOODS_RUN_ORDER.

    Mirrors the offset logic in eval/run_all_models.py so that case IDs
    here map 1-to-1 onto keys in delta_consensus_v3.json.
    """
    offset = 0
    for gs in GOODS_RUN_ORDER:
        if gs == goods_stem:
            break
        path = Path(data_dir) / f"{gs}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                prior = json.load(f)
            count = sum(
                len(e[2]) for e in prior
                if isinstance(e, (list, tuple)) and len(e) >= 3 and isinstance(e[2], list)
            )
            offset += count
            logger.info(f"  {gs}.json: {count} cases (running offset → {offset})")
        else:
            logger.warning(
                f"  Prior file {gs}.json not found under {data_dir} — "
                "assuming 0 cases. Case IDs may be wrong if this file exists elsewhere."
            )
    return offset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_grpo_dataset(
    goods_path: str,
    goods_json_path: str,
    delta_path: str,
    data_dir: str,
) -> "datasets.Dataset":
    """
    Build a HuggingFace Dataset for GRPOTrainer from a goods file.

    Each goods pair (X_num, Y_num) with an attr_code generates two examples:
      - X-perspective: generate_prompt(X, Y, i, j, k, l)
      - Y-perspective: generate_prompt(Y, X, k, l, i, j)
    Both perspectives share the same case_id and delta (δ̃ = U_X − U_Y).

    Cases not in the delta file, or with δ̃ == 0.0, are skipped.

    Returned dataset columns:
      prompt      : str  — plain user message (wrap in messages format in grpo_train.py)
      perspective : str  — "X" or "Y"
      delta       : float — δ̃ = U_X − U_Y from delta_consensus_v3.json
      case_id     : int  — global case ID (matches delta file keys)
    """
    goods_path = Path(goods_path)
    goods_stem = goods_path.stem

    logger.info(f"Loading goods pairs from {goods_path}")
    with open(goods_path, "r", encoding="utf-8") as f:
        dataset_raw = json.load(f)

    logger.info(f"Loading item attributes from {goods_json_path}")
    goods = GoodsData(goods_json_path)

    logger.info(f"Loading delta file from {delta_path}")
    with open(delta_path, "r", encoding="utf-8") as f:
        deltas = json.load(f)

    case_id_offset = compute_case_id_offset(goods_stem, data_dir)
    logger.info(f"Case ID offset for '{goods_stem}': {case_id_offset}")

    examples = []
    local_id = 0
    n_missing_delta = 0
    n_zero_delta = 0

    for entry in dataset_raw:
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
            continue
        X_num, Y_num, attr_list = entry[0], entry[1], entry[2]
        if not isinstance(attr_list, list):
            continue
        if not (0 <= X_num < len(goods.items) and 0 <= Y_num < len(goods.items)):
            continue

        X = goods.items[X_num]
        Y = goods.items[Y_num]

        for attr_code in attr_list:
            local_id += 1
            global_id = case_id_offset + local_id
            key = str(global_id)

            if key not in deltas:
                n_missing_delta += 1
                continue

            delta = float(deltas[key]["mean_delta"])

            if delta == 0.0:
                n_zero_delta += 1
                continue  # zero gradient signal either way

            i, j, k, l = decode_attr(attr_code)

            # X-perspective: endowed with X, offered Y
            examples.append({
                "prompt": generate_prompt(goods, X, Y, i, j, k, l),
                "perspective": "X",
                "delta": delta,
                "case_id": global_id,
            })

            # Y-perspective: endowed with Y, offered X
            # Argument order matches run_all_models.py: generate_prompt(Y, X, k, l, i, j)
            examples.append({
                "prompt": generate_prompt(goods, Y, X, k, l, i, j),
                "perspective": "Y",
                "delta": delta,
                "case_id": global_id,
            })

    n_cases = local_id
    n_examples = len(examples)
    logger.info(
        f"Dataset built: {n_cases} cases → {n_examples} examples "
        f"({n_missing_delta} skipped: missing delta, {n_zero_delta} skipped: zero delta)"
    )

    if n_examples == 0:
        raise RuntimeError(
            f"No training examples built from {goods_path}.\n"
            "Likely causes:\n"
            "  1. everyday_goods_full.json is a broken symlink — copy the real file.\n"
            "  2. delta_consensus_v3.json doesn't cover this file's case ID range.\n"
            "  3. Case ID offset is wrong — check GOODS_RUN_ORDER and prior file sizes."
        )

    return datasets.Dataset.from_list(examples)


def build_grpo_dataset_v2(
    goods_path: str,
    goods_json_path: str,
    data_dir: str,
    anchor_map: "Dict[int, str] | None" = None,
) -> "datasets.Dataset":
    """
    Dataset builder for Reward Design v2 (gated paired reward).

    Unlike build_grpo_dataset, this does NOT read or filter by delta — the v2
    reward is derived from ownership consistency gated by the frozen neutral
    anchor. Columns:

      prompt      : str  — plain user message
      perspective : str  — "X" or "Y"
      case_id     : int  — global 1-indexed case ID (matches anchor / delta keys)

    X and Y perspectives of a case are emitted consecutively (paired), so
    build_pairs_from_columns() over (case_id, perspective) recovers the pairs.

    anchor_map: if given ({case_id: "X"/"Y"/None}), the dataset is FILTERED to
    cases with a STABLE anchor (pref in {"X","Y"}). The gated reward needs a
    stable anchor to grade preference direction, so training only on stable-anchor
    cases makes anchor coverage exactly 100% and every pair fully graded. Cases
    with pref None / missing are dropped. If None, every case is kept.
    """
    goods_path = Path(goods_path)
    with open(goods_path, "r", encoding="utf-8") as f:
        dataset_raw = json.load(f)
    goods = GoodsData(goods_json_path)

    case_id_offset = compute_case_id_offset(goods_path.stem, data_dir)
    logger.info(f"[v2] Case ID offset for '{goods_path.stem}': {case_id_offset}")

    def _stable(cid: int) -> bool:
        if anchor_map is None:
            return True
        return anchor_map.get(int(cid)) in ("X", "Y")

    examples = []
    local_id = 0
    n_cases = n_kept = 0
    for entry in dataset_raw:
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
            continue
        X_num, Y_num, attr_list = entry[0], entry[1], entry[2]
        if not isinstance(attr_list, list):
            continue
        if not (0 <= X_num < len(goods.items) and 0 <= Y_num < len(goods.items)):
            continue
        X, Y = goods.items[X_num], goods.items[Y_num]
        for attr_code in attr_list:
            local_id += 1
            global_id = case_id_offset + local_id   # 1-indexed, matches anchors
            n_cases += 1
            if not _stable(global_id):
                continue
            n_kept += 1
            i, j, k, l = decode_attr(attr_code)
            examples.append({
                "prompt": generate_prompt(goods, X, Y, i, j, k, l),
                "perspective": "X", "case_id": global_id,
            })
            examples.append({
                "prompt": generate_prompt(goods, Y, X, k, l, i, j),
                "perspective": "Y", "case_id": global_id,
            })

    filt = "no anchor filter" if anchor_map is None \
        else f"stable-anchor filter: {n_kept}/{n_cases} cases kept"
    logger.info(f"[v2] Dataset built: {n_kept} cases → {len(examples)} examples ({filt})")
    if not examples:
        raise RuntimeError(f"No v2 examples built from {goods_path} — check goods file / offset / anchor filter.")
    return datasets.Dataset.from_list(examples)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Resolve paths relative to project root (one level up from train/)
    project_root = Path(__file__).resolve().parent.parent
    goods_file = sys.argv[1]

    ds = build_grpo_dataset(
        goods_path=goods_file,
        goods_json_path=str(project_root / "everyday_goods_full.json"),
        delta_path=str(project_root / "data" / "deltas" / "delta_consensus_v3.json"),
        data_dir=str(project_root / "data"),
    )

    print(f"\nBuilt {len(ds)} examples")
    print("\nFirst 3 examples:")
    for i in range(min(3, len(ds))):
        ex = ds[i]
        print(f"  [{i}] case_id={ex['case_id']}  perspective={ex['perspective']}  delta={ex['delta']:.4f}")
        print(f"       {ex['prompt'][:100]!r}...")
