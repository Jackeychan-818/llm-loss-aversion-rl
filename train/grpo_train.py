#!/usr/bin/env python3
"""
lambda-zero GRPO training script.

Fine-tunes Qwen2.5-7B-Instruct with LoRA using TRL's GRPOTrainer to reduce
loss aversion (λ → 0). Reward: utility-weighted using consensus δ̃ from 6
frontier models. See CLAUDE.md for full rationale and parameter choices.

Usage:
    # Sanity check (100 steps on trial_goods.json, ~1.5h on gdev)
    python train/grpo_train.py \\
        --config train/configs/qwen25_7b.yaml \\
        --mode sanity \\
        --max_steps 100 \\
        --data_file data/trial_goods.json \\
        --output_dir checkpoints/sanity_test

    # Full training (remaining_goods.json, ~24h on g1)
    python train/grpo_train.py \\
        --config train/configs/qwen25_7b.yaml \\
        --mode train \\
        --data_file data/remaining_goods.json \\
        --output_dir checkpoints/grpo_run1 \\
        --eval_data_file data/trial_goods.json

PBS submission: see train/submit_sanity.pbs and train/submit_train.pbs
"""

import argparse
import dataclasses
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from prompt_builder import build_grpo_dataset
from reward_functions import make_reward_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training for lambda-zero")
    p.add_argument("--config",          required=True,  help="Path to YAML config file")
    p.add_argument("--mode",            choices=["sanity", "train"], default="train",
                   help="'sanity' for quick smoke test, 'train' for full run")
    p.add_argument("--data_file",       required=True,
                   help="Goods data file (e.g. data/remaining_goods.json)")
    p.add_argument("--eval_data_file",  default=None,
                   help="Eval goods file for periodic reward monitoring (e.g. data/trial_goods.json)")
    p.add_argument("--output_dir",      required=True,  help="Directory for LoRA checkpoints")
    p.add_argument("--max_steps",       type=int, default=None,
                   help="Override max_steps (use 100 for sanity check)")
    p.add_argument("--save_steps",      type=int, default=None)
    p.add_argument("--eval_steps",      type=int, default=None)
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Checkpoint path to resume from, or 'auto' to use the latest checkpoint")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    """Resolve path relative to PROJECT_ROOT if not absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ─────────────────────────────────────────────────────────────────────────────
# GRPO config builder
# ─────────────────────────────────────────────────────────────────────────────

def latest_checkpoint(path: Path) -> Optional[Path]:
    """Return the numerically latest checkpoint-* directory under path."""
    if not path.exists():
        return None
    checkpoints = []
    for child in path.glob("checkpoint-*"):
        if child.is_dir():
            try:
                step = int(child.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, child))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def find_resume_checkpoint(args: argparse.Namespace) -> Optional[str]:
    """Resolve explicit or automatic checkpoint resume target."""
    requested = args.resume_from_checkpoint
    if not requested:
        return None

    output_dir = resolve(args.output_dir)
    if requested == "auto":
        ckpt = latest_checkpoint(output_dir)
        if ckpt is None:
            logger.info(f"No checkpoint found under {output_dir}; starting from scratch.")
            return None
        logger.info(f"Auto-resuming from latest checkpoint: {ckpt}")
        return str(ckpt)

    ckpt = resolve(requested)
    if not ckpt.exists():
        logger.error(f"Requested checkpoint not found: {ckpt}")
        sys.exit(1)
    return str(ckpt)


def filter_supported_config_kwargs(config_cls, kwargs: dict) -> dict:
    """Drop GRPOConfig kwargs unsupported by the installed TRL version."""
    supported = {field.name for field in dataclasses.fields(config_cls)}
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    dropped = sorted(set(kwargs) - supported)
    if dropped:
        logger.warning(
            "Installed TRL GRPOConfig does not support these settings; ignoring: "
            + ", ".join(dropped)
        )
    return filtered


def build_grpo_config(cfg: dict, args: argparse.Namespace, resume_checkpoint: Optional[str]):
    from trl import GRPOConfig

    max_steps  = args.max_steps  if args.max_steps  is not None else cfg.get("max_steps", -1)
    save_steps = args.save_steps if args.save_steps is not None else cfg.get("save_steps", 200)
    eval_steps = args.eval_steps if args.eval_steps is not None else cfg.get("eval_steps", 200)

    kwargs = dict(
        output_dir=args.output_dir,
        # GRPO algorithm
        num_generations=cfg.get("num_generations", 16),
        temperature=cfg.get("temperature", 1.5),
        epsilon=cfg.get("epsilon", 0.2),
        beta=cfg.get("beta", 0.04),
        max_completion_length=cfg.get("max_completion_length", 4),
        loss_type=cfg.get("loss_type", "dapo"),
        scale_rewards=cfg.get("scale_rewards", "none"),
        mask_truncated_completions=cfg.get("mask_truncated_completions", True),
        log_completions=cfg.get("log_completions", True),
        # Training
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 1e-6),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=max_steps,
        max_grad_norm=cfg.get("max_grad_norm", 0.1),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        bf16=cfg.get("bf16", True),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        # Critical: keep perspective/delta/case_id columns so reward fn gets them
        remove_unused_columns=False,
        # Logging / saving
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=save_steps,
        eval_steps=eval_steps,
        report_to=cfg.get("report_to", "none"),
        # vLLM
        use_vllm=cfg.get("use_vllm", False),
        resume_from_checkpoint=resume_checkpoint,
    )

    if cfg.get("use_vllm"):
        # TRL's vLLM API changed across releases. Keep every known knob here and
        # filter against the installed GRPOConfig fields below.
        kwargs.update(
            vllm_mode=cfg.get("vllm_mode", "colocate"),
            vllm_device=cfg.get("vllm_device", "auto"),
            vllm_gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.3),
            vllm_server_base_url=cfg.get("vllm_server_base_url", None),
            vllm_server_host=cfg.get("vllm_server_host", "0.0.0.0"),
            vllm_server_port=cfg.get("vllm_server_port", 8000),
            vllm_server_timeout=cfg.get("vllm_server_timeout", 240.0),
            vllm_max_model_length=cfg.get("vllm_max_model_length", None),
            vllm_max_model_len=cfg.get("vllm_max_model_len", cfg.get("vllm_max_model_length", None)),
        )

    return GRPOConfig(**filter_supported_config_kwargs(GRPOConfig, kwargs))


# ─────────────────────────────────────────────────────────────────────────────
# LoRA config builder
# ─────────────────────────────────────────────────────────────────────────────

def build_lora_config(cfg: dict):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias="none",
        task_type="CAUSAL_LM",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_messages_format(dataset):
    """Wrap plain-string prompts in user-message format for Qwen's chat template."""
    return dataset.map(
        lambda ex: {"prompt": [{"role": "user", "content": ex["prompt"]}]},
        desc="Formatting prompts",
    )


def log_sample_examples(dataset, n: int = 3) -> None:
    logger.info("─" * 64)
    logger.info("Sample training examples:")
    for i in range(min(n, len(dataset))):
        ex = dataset[i]
        content = ex["prompt"][0]["content"] if isinstance(ex["prompt"], list) else ex["prompt"]
        logger.info(
            f"  [{i}] case_id={ex['case_id']:>6}  "
            f"perspective={ex['perspective']}  delta={ex['delta']:+.4f}"
        )
        logger.info(f"       {content[:110]!r}")
    logger.info("─" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    os.chdir(PROJECT_ROOT)
    logger.info(f"Working directory : {PROJECT_ROOT}")
    logger.info(f"Mode              : {args.mode}")
    logger.info(f"Config            : {args.config}")
    logger.info(f"Data file         : {args.data_file}")
    logger.info(f"Output dir        : {args.output_dir}")

    # ── Validate paths ────────────────────────────────────────────────────
    model_path     = resolve(cfg["model_name_or_path"])
    delta_path     = resolve(cfg.get("delta_file", "data/deltas/delta_consensus_v3.json"))
    goods_json     = resolve(cfg.get("goods_file", "everyday_goods_full.json"))
    data_dir       = resolve("data")
    data_file      = resolve(args.data_file)
    eval_data_file: Optional[Path] = resolve(args.eval_data_file) if args.eval_data_file else None

    if not model_path.exists():
        logger.error(
            f"Model not found at {model_path}.\n"
            "Run train/setup_nscc.sh first to download Qwen2.5-7B-Instruct."
        )
        sys.exit(1)

    if not goods_json.exists():
        logger.error(
            f"everyday_goods_full.json not found at {goods_json}.\n"
            "It is a symlink locally — copy the actual file to NSCC before training.\n"
            "See CLAUDE.md Critical item 11."
        )
        sys.exit(1)

    if not delta_path.exists():
        logger.error(f"Delta file not found at {delta_path}.")
        sys.exit(1)

    if not data_file.exists():
        logger.error(f"Data file not found at {data_file}.")
        sys.exit(1)

    # ── Tokenizer + model ────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {model_path} (bfloat16)")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {n_params:,} parameters")

    # ── Datasets ─────────────────────────────────────────────────────────
    logger.info("Building training dataset...")
    train_ds = build_grpo_dataset(
        goods_path=str(data_file),
        goods_json_path=str(goods_json),
        delta_path=str(delta_path),
        data_dir=str(data_dir),
    )
    train_ds = to_messages_format(train_ds)
    logger.info(f"Training examples: {len(train_ds)}")

    eval_ds = None
    if eval_data_file and eval_data_file.exists():
        logger.info("Building eval dataset...")
        eval_ds = build_grpo_dataset(
            goods_path=str(eval_data_file),
            goods_json_path=str(goods_json),
            delta_path=str(delta_path),
            data_dir=str(data_dir),
        )
        eval_ds = to_messages_format(eval_ds)
        logger.info(f"Eval examples    : {len(eval_ds)}")

    log_sample_examples(train_ds)

    # ── Trainer setup ────────────────────────────────────────────────────
    from trl import GRPOTrainer

    resume_checkpoint = find_resume_checkpoint(args)
    grpo_config = build_grpo_config(cfg, args, resume_checkpoint)
    lora_config = build_lora_config(cfg)
    reward_fn   = make_reward_fn()

    logger.info(
        f"GRPO config: G={grpo_config.num_generations}, "
        f"temp={grpo_config.temperature}, β={grpo_config.beta}, "
        f"ε={grpo_config.epsilon}, max_completion={grpo_config.max_completion_length}, "
        f"loss_type={getattr(grpo_config, 'loss_type', 'unsupported-by-installed-trl')}"
    )
    if getattr(grpo_config, "use_vllm", False):
        logger.info(
            "vLLM enabled: "
            f"mode={getattr(grpo_config, 'vllm_mode', 'legacy')}, "
            f"device={getattr(grpo_config, 'vllm_device', 'n/a')}, "
            f"gpu_memory={getattr(grpo_config, 'vllm_gpu_memory_utilization', 'n/a')}"
        )
    logger.info(
        f"Training: lr={grpo_config.learning_rate}, "
        f"batch={grpo_config.per_device_train_batch_size}×"
        f"{grpo_config.gradient_accumulation_steps} (eff.), "
        f"max_steps={grpo_config.max_steps}"
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # ── Train ────────────────────────────────────────────────────────────
    logger.info("=" * 64)
    logger.info("Starting GRPO training")
    logger.info("=" * 64)
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ── Save final checkpoint ────────────────────────────────────────────
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Final LoRA checkpoint saved to {final_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
