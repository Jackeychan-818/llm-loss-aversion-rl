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

from prompt_builder import build_grpo_dataset_v2
from paired_grpo import (
    build_pairs_from_columns,
    make_r_pair_reward,
    make_r_neutral_reward,
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
    w_pair    = cfg.get("w_pair", 1.0)
    w_neutral = cfg.get("w_neutral", 0.5)

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
        # v2: two reward functions [r_pair, r_neutral]
        reward_weights=[w_pair, w_neutral],
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
    return GRPOConfig(**filter_supported_config_kwargs(GRPOConfig, kwargs)), (w_pair, w_neutral)


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

    # ── Model ────────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading tokenizer + model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.bfloat16)

    # ── Dataset (delta-free) + anchor ────────────────────────────────────
    logger.info("Building v2 dataset (no delta filtering)...")
    train_ds = build_grpo_dataset_v2(
        goods_path=str(data_file),
        goods_json_path=str(goods_json),
        data_dir=str(resolve("data")),
    )
    train_ds = to_messages_format(train_ds)

    anchor_raw = json.load(open(anchor_file))
    anchor = anchor_raw.get("anchors", anchor_raw)   # accept {"anchors": {...}} or flat
    amap = {int(k): (v["pref"] if isinstance(v, dict) else v) for k, v in anchor.items()}

    case_ids = [int(c) for c in train_ds["case_id"]]
    pairs = build_pairs_from_columns(case_ids, train_ds["perspective"])
    uniq_cases = set(case_ids)
    frozen = sum(1 for c in uniq_cases if amap.get(c) is not None)
    coverage = frozen / max(len(uniq_cases), 1)

    # ── GRPO config + reward funcs ───────────────────────────────────────
    resume_checkpoint = find_resume_checkpoint(args)
    grpo_config, (w_pair, w_neutral) = build_grpo_config_v2(cfg, args, resume_checkpoint)
    lora_config = build_lora_config(cfg)
    reward_funcs = [make_r_pair_reward(), make_r_neutral_reward(anchor)]

    logger.info("=" * 64)
    logger.info("reward=v2_core (R_pair + R_neutral)")
    logger.info(f"reward_weights: R_pair={w_pair}, R_neutral={w_neutral}")
    logger.info(f"anchor file: {anchor_file.name}")
    logger.info(f"anchor coverage on {stem}: {frozen}/{len(uniq_cases)} cases = {coverage:.1%}")
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
