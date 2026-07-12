#!/usr/bin/env python3
"""
lambda-zero GRPO training — Reward Design v2 core (R_pair + R_neutral).

Distinct from grpo_train.py (the legacy consensus-delta run). This entry point:
  * builds a delta-free dataset (every case kept; build_grpo_dataset_v2),
  * loads the frozen neutral anchor for the data file,
  * trains with PairedGRPOTrainer so each case's X/Y perspectives are scored
    together (R_pair) alongside the anchor-direction reward (R_neutral),
  * logs `reward=v2_core` plus per-component rewards and v2 diagnostics.

Usage:
    # Sanity (30 steps on the training set; throwaway checkpoint)
    python train/grpo_train_v2.py \\
        --config train/configs/qwen25_7b_v2core.yaml \\
        --mode sanity --max_steps 30 \\
        --data_file data/remaining_goods.json \\
        --output_dir checkpoints/sanity_v2core

    # Full run
    python train/grpo_train_v2.py \\
        --config train/configs/qwen25_7b_v2core.yaml \\
        --mode train \\
        --data_file data/remaining_goods.json \\
        --output_dir checkpoints/grpo_v2core
"""

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from prompt_builder import build_grpo_dataset_v2, compute_case_id_offset
from paired_grpo import (
    build_pairs_from_columns,
    make_r_gated_reward,
    build_paired_grpo_trainer_cls,
)
# reuse the vetted helpers from the legacy entry point
from grpo_train import (
    resolve, load_config, find_resume_checkpoint,
    build_lora_config, to_messages_format, filter_supported_config_kwargs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("grpo_v2")

# Hard-coded 1-indexed global case-ID ranges per goods file (CLAUDE.md, verified
# against baseline/Qwen-7B/completed_index.json). Startup checks assert against these.
EXPECTED_ID_RANGE = {
    "trial_goods": (1, 60),
    "test_goods": (61, 9950),
    "remaining_goods": (9951, 59400),
}


def validate_anchor_json(anchor_raw, amap, stem, anchor_file):
    """Validate the anchor JSON CONTENTS (not just its filename), before any
    dataset build or model load. Raises AssertionError on the first violation."""
    errs = []
    # Structure: must be a flat map or {"anchors": {...}} (+ optional _summary).
    if not isinstance(anchor_raw, dict):
        raise AssertionError(f"anchor JSON root is {type(anchor_raw).__name__}, expected object")
    if "anchors" in anchor_raw and not isinstance(anchor_raw["anchors"], dict):
        errs.append("'anchors' present but is not an object")
    if not amap:
        errs.append("anchor map is empty after parsing")

    # Every value must be a valid preference.
    bad_vals = {k: v for k, v in amap.items() if v not in ("X", "Y", None)}
    if bad_vals:
        errs.append(f"{len(bad_vals)} anchor entries have invalid pref (not X/Y/None), "
                    f"e.g. {dict(list(bad_vals.items())[:5])}")

    # Keys must fall inside the file's expected global case-ID range.
    lo, hi = EXPECTED_ID_RANGE.get(stem, (None, None))
    if lo is not None:
        oor = [k for k in amap if not (lo <= k <= hi)]
        if oor:
            errs.append(f"{len(oor)} anchor keys outside expected range [{lo},{hi}] "
                        f"for '{stem}', e.g. {sorted(oor)[:5]}")

    n_stable = sum(1 for v in amap.values() if v in ("X", "Y"))
    n_amb = sum(1 for v in amap.values() if v is None)
    if n_stable == 0:
        errs.append("no stable (X/Y) anchors — nothing to train on")

    # Cross-check the _summary block if the elicitation wrote one.
    summ = anchor_raw.get("_summary") if isinstance(anchor_raw, dict) else None
    if isinstance(summ, dict):
        for key in ("n_stable", "stable", "n_frozen", "frozen"):
            if key in summ and isinstance(summ[key], int) and summ[key] != n_stable:
                errs.append(f"_summary['{key}']={summ[key]} disagrees with counted stable={n_stable}")

    if errs:
        for e in errs:
            logger.error(f"[ANCHOR CHECK FAILED] {e}")
        raise AssertionError(f"{len(errs)} anchor-content check(s) failed for {Path(anchor_file).name}.")

    logger.info(f"[checks] anchor JSON contents valid: {len(amap)} entries "
                f"({n_stable} stable X/Y, {n_amb} ambiguous), keys in [{lo},{hi}], all prefs valid")
    return n_stable


def run_startup_checks(stem, anchor_file, amap, train_ds, pairs, data_dir):
    """Fail LOUD and EARLY (before the model trains) on any structural mismatch:
    anchor file, case-ID range, one-X-one-Y-per-case, pair count, anchor coverage.
    Raises AssertionError on the first violation."""
    case_ids = [int(c) for c in train_ds["case_id"]]
    persp = list(train_ds["perspective"])
    uniq = sorted(set(case_ids))
    errs = []

    # (1) Correct anchor file for this data file.
    if f"neutral_anchor_{stem}" not in Path(anchor_file).stem:
        errs.append(f"anchor file {Path(anchor_file).name} does not match data stem '{stem}'")

    # (2) Case-ID range matches the file's known global range.
    lo, hi = EXPECTED_ID_RANGE.get(stem, (None, None))
    if lo is not None:
        if min(uniq) < lo or max(uniq) > hi:
            errs.append(f"case-ID range [{min(uniq)},{max(uniq)}] outside expected [{lo},{hi}] for '{stem}'")
    else:
        logger.warning(f"[checks] no expected ID range registered for stem '{stem}' — skipping range check")

    # (3) Exactly one X row and one Y row per case_id.
    from collections import Counter
    persp_by_case: dict[int, Counter] = {}
    for c, p in zip(case_ids, persp):
        persp_by_case.setdefault(int(c), Counter())[p] += 1
    bad = [c for c, cc in persp_by_case.items() if cc.get("X", 0) != 1 or cc.get("Y", 0) != 1
           or sum(cc.values()) != 2]
    if bad:
        errs.append(f"{len(bad)} cases do NOT have exactly one X + one Y row (e.g. {bad[:5]})")
    if len(case_ids) != 2 * len(uniq):
        errs.append(f"dataset has {len(case_ids)} rows but {len(uniq)} unique cases (expected 2x)")

    # (4) Expected number of pairs == number of unique stable-anchor cases.
    stable_in_range = sum(1 for c in uniq if amap.get(c) in ("X", "Y"))
    if len(pairs) != len(uniq):
        errs.append(f"pairs built ({len(pairs)}) != unique cases ({len(uniq)})")
    if len(pairs) != stable_in_range:
        errs.append(f"pairs built ({len(pairs)}) != stable-anchor cases in dataset ({stable_in_range})")

    # (5) Anchor coverage on the (already-filtered) training set must be 100%.
    covered = sum(1 for c in uniq if amap.get(c) in ("X", "Y"))
    coverage = covered / max(len(uniq), 1)
    if coverage < 1.0 - 1e-9:
        missing = [c for c in uniq if amap.get(c) not in ("X", "Y")][:5]
        errs.append(f"anchor coverage on filtered train set is {coverage:.4%} (<100%); "
                    f"{len(uniq) - covered} cases lack a stable anchor (e.g. {missing})")

    if errs:
        for e in errs:
            logger.error(f"[STARTUP CHECK FAILED] {e}")
        raise AssertionError(f"{len(errs)} startup check(s) failed — refusing to train. See errors above.")

    logger.info("[checks] all startup checks passed:")
    logger.info(f"[checks]   anchor file matches stem '{stem}'")
    logger.info(f"[checks]   case-ID range [{min(uniq)},{max(uniq)}] within expected [{lo},{hi}]")
    logger.info(f"[checks]   one X + one Y per case for all {len(uniq)} cases")
    logger.info(f"[checks]   pairs = unique cases = stable-anchor cases = {len(pairs)}")
    logger.info(f"[checks]   anchor coverage on filtered train set = 100.0%")
    return coverage


def parse_args():
    p = argparse.ArgumentParser(description="GRPO v2-core training for lambda-zero")
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["sanity", "train"], default="train")
    p.add_argument("--data_file", required=True)
    p.add_argument("--anchor_file", default=None,
                   help="Frozen anchor JSON; default data/anchors/neutral_anchor_<datastem>.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", default=None)
    return p.parse_args()


def build_grpo_config_v2(cfg, args, resume_checkpoint):
    from trl import GRPOConfig

    max_steps  = args.max_steps  if args.max_steps  is not None else cfg.get("max_steps", -1)
    save_steps = args.save_steps if args.save_steps is not None else cfg.get("save_steps", 200)

    kwargs = dict(
        output_dir=args.output_dir,
        # GRPO algorithm.
        # NOTE: do NOT set generation_batch_size AND steps_per_generation — TRL
        # forbids both. We set neither and let TRL derive
        #   generation_batch_size = per_device_train_batch_size * num_proc * steps_per_generation
        # where steps_per_generation defaults to gradient_accumulation_steps.
        # With per_device=1, grad_accum=32 → generation_batch_size=32 = one whole
        # X/Y pair (2 prompts x G=16). PairedGRPOTrainer reads the derived value.
        num_generations=cfg.get("num_generations", 16),
        temperature=cfg.get("temperature", 1.5),
        epsilon=cfg.get("epsilon", 0.2),
        beta=cfg.get("beta", 0.04),
        max_completion_length=cfg.get("max_completion_length", 4),
        loss_type=cfg.get("loss_type", "dapo"),
        scale_rewards=cfg.get("scale_rewards", "none"),
        mask_truncated_completions=cfg.get("mask_truncated_completions", True),
        log_completions=cfg.get("log_completions", True),
        # v2-core: ONE gated paired reward (r_gated). No sub-weights.
        reward_weights=[1.0],
        # keep X/Y pairs adjacent — pairing relies on the PairedRepeatSampler
        shuffle_dataset=cfg.get("shuffle_dataset", True),
        # Training — geometry controls generation_batch_size (see note above)
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 32),
        learning_rate=cfg.get("learning_rate", 1e-6),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=max_steps,
        max_grad_norm=cfg.get("max_grad_norm", 0.1),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        bf16=cfg.get("bf16", True),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        remove_unused_columns=False,   # reward fns need perspective + case_id
        logging_steps=cfg.get("logging_steps", 5),
        save_steps=save_steps,
        report_to=cfg.get("report_to", "none"),
        use_vllm=cfg.get("use_vllm", False),
        resume_from_checkpoint=resume_checkpoint,
    )
    return GRPOConfig(**filter_supported_config_kwargs(GRPOConfig, kwargs))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.chdir(PROJECT_ROOT)

    stem = Path(args.data_file).stem
    anchor_file = resolve(args.anchor_file) if args.anchor_file \
        else PROJECT_ROOT / "data" / "anchors" / f"neutral_anchor_{stem}.json"

    model_path = resolve(cfg["model_name_or_path"])
    goods_json = resolve(cfg.get("goods_file", "everyday_goods_full.json"))
    data_file  = resolve(args.data_file)

    for pth, what in [(model_path, "model"), (goods_json, "everyday_goods_full.json"),
                      (data_file, "data file"), (anchor_file, "anchor file")]:
        if not Path(pth).exists():
            logger.error(f"{what} not found at {pth}")
            sys.exit(1)

    reward_mode = cfg.get("reward_mode", "shaped")

    # ══ STRUCTURAL VALIDATION FIRST — everything below happens BEFORE the model
    #    is loaded, so any structural failure aborts in seconds without paying the
    #    model-load / GPU-allocation cost. ═══════════════════════════════════════

    # ── Anchor: load + validate CONTENTS (not just filename) ─────────────
    anchor_raw = json.load(open(anchor_file))
    anchor = anchor_raw.get("anchors", anchor_raw)   # accept {"anchors": {...}} or flat
    amap = {int(k): (v["pref"] if isinstance(v, dict) else v) for k, v in anchor.items()}
    n_stable_anchors = validate_anchor_json(anchor_raw, amap, stem, anchor_file)
    logger.info(f"Loaded anchor {anchor_file.name}: {n_stable_anchors} stable / {len(amap)} entries")

    # ── Dataset — FILTERED to stable-anchor cases only ───────────────────
    logger.info("Building v2 dataset (filtered to stable anchors)...")
    train_ds = build_grpo_dataset_v2(
        goods_path=str(data_file),
        goods_json_path=str(goods_json),
        data_dir=str(resolve("data")),
        anchor_map=amap,
    )
    train_ds = to_messages_format(train_ds)

    case_ids = [int(c) for c in train_ds["case_id"]]
    pairs = build_pairs_from_columns(case_ids, train_ds["perspective"])

    # ── HARD STRUCTURAL CHECKS (still before any GPU/model cost) ─────────
    coverage = run_startup_checks(stem, anchor_file, amap, train_ds, pairs,
                                  data_dir=str(resolve("data")))

    if reward_mode not in ("shaped", "hard"):
        raise AssertionError(f"reward_mode must be 'shaped' or 'hard', got {reward_mode!r}")

    # ══ Only now pay for the model. ══════════════════════════════════════
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Structural checks passed. Loading tokenizer + model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.bfloat16)

    # ── GRPO config + single gated reward func ───────────────────────────
    resume_checkpoint = find_resume_checkpoint(args)
    grpo_config = build_grpo_config_v2(cfg, args, resume_checkpoint)
    lora_config = build_lora_config(cfg)
    reward_funcs = [make_r_gated_reward(anchor, mode=reward_mode)]

    table = ("keep/trade-both=-1 | consistent-wrong-pref=0 | consistent-anchored=+1"
             if reward_mode == "shaped"
             else "consistent+anchored=+1 | everything else=-1 (hard/binary)")
    logger.info("=" * 64)
    logger.info(f"reward=v2_core (single GATED paired reward: r_gated, mode={reward_mode})")
    logger.info(f"gated table: {table}")
    logger.info(f"anchor file: {anchor_file.name}")
    logger.info(f"anchor coverage on filtered train set: {len(pairs)}/{len(set(case_ids))} = {coverage:.1%}")
    logger.info(f"pairs (X/Y cases): {len(pairs)}  |  dataset rows: {len(train_ds)}")
    logger.info(f"GRPO: G={grpo_config.num_generations}, gen_batch={grpo_config.generation_batch_size}, "
                f"temp={grpo_config.temperature}, β={grpo_config.beta}, "
                f"loss_type={getattr(grpo_config, 'loss_type', '?')}, "
                f"shuffle_dataset={getattr(grpo_config, 'shuffle_dataset', '?')}")
    logger.info("=" * 64)

    PairedGRPOTrainer = build_paired_grpo_trainer_cls()
    trainer = PairedGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        pairs=pairs,
    )

    logger.info("Starting v2-core GRPO training")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Final v2-core LoRA checkpoint saved to {final_dir}")


if __name__ == "__main__":
    main()
