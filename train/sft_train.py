#!/usr/bin/env python3
"""
lambda-zero matched SFT baseline (Priority 1A, CAUSAL_BASELINE_PROTOCOL.md).

Supervised fine-tuning of Qwen2.5-7B-Instruct with LoRA on the SAME Qwen-own-delta
rational-choice targets that the GRPO reward rewards. Answers: is RL necessary,
or does ordinary supervised learning already remove the endowment effect?

MATCHED to the confirmatory GRPO run: same base weights, same prompt builder,
same frozen δ̃ signs, same LoRA, same optimizer/schedule, independent seeds. The
SFT target is the rational answer from reward_functions.rational_choice (single
source of truth), so SFT and GRPO cannot silently diverge on "which answer is
correct". Loss is COMPLETION-ONLY: only the assistant Yes/No tokens contribute;
prompt tokens are masked with -100 (mirrors GRPO's policy gradient on generated
tokens only).

Data exposure (primary matching rule): unique prompts seen. With batch 1 /
grad-accum 1, 1 step = 1 unique prompt = 1 optimizer update, so MAX_STEPS matches
the GRPO endpoint on the unique-prompt axis. See CAUSAL_BASELINE_PROTOCOL.md.

Usage:
    # Dataset/target validation (NO GPU, NO training) + manifest:
    python train/sft_train.py --config train/configs/qwen25_7b_sft_qwen_delta.yaml \
        --data_file data/remaining_goods.json --output_dir checkpoints/sft_qwen_delta_seed1 \
        --seed 1 --validate

    # Training (GPU):
    python train/sft_train.py --config train/configs/qwen25_7b_sft_qwen_delta.yaml \
        --data_file data/remaining_goods.json --output_dir checkpoints/sft_qwen_delta_seed1 \
        --seed 1 --max_steps 30000 --save_steps 2000
"""
import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from prompt_builder import build_grpo_dataset          # reuse GRPO data utilities
from reward_functions import rational_choice           # single source of truth

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matched SFT baseline for lambda-zero")
    p.add_argument("--config", required=True)
    p.add_argument("--data_file", required=True, help="Training goods file (data/remaining_goods.json)")
    p.add_argument("--output_dir", required=True, help="Per-seed LoRA output dir (checkpoints/sft_qwen_delta_seed{N})")
    p.add_argument("--seed", type=int, default=None, help="Training seed (RNG + data order). Overrides YAML.")
    p.add_argument("--max_steps", type=int, default=None, help="Optimizer steps (=unique prompts at batch 1).")
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", default=None, help="Checkpoint path or 'auto'")
    p.add_argument("--validate", action="store_true",
                   help="Build the dataset, write the manifest, print counts, and EXIT. No GPU, no training.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── dataset: reuse GRPO builder, map to SFT rational-choice targets ───────────
def build_sft_records(data_file: Path, goods_json: Path, delta_path: Path, data_dir: Path):
    """Return a list of {prompt, perspective, delta, case_id, target} records.
    target = rational_choice(perspective, delta) — the frozen preferred answer.
    Pure (no tokenizer/GPU) so it is unit-testable and drives --validate."""
    ds = build_grpo_dataset(goods_path=str(data_file), goods_json_path=str(goods_json),
                            delta_path=str(delta_path), data_dir=str(data_dir))
    records = []
    for ex in ds:
        target = rational_choice(ex["perspective"], float(ex["delta"]))
        if target not in ("Yes", "No"):
            continue  # δ̃==0 filtered upstream; guard anyway
        records.append({"prompt": ex["prompt"], "perspective": ex["perspective"],
                        "delta": float(ex["delta"]), "case_id": int(ex["case_id"]), "target": target})
    return records


def dataset_stats(records) -> dict:
    return {
        "n_examples": len(records),
        "label_counts": dict(Counter(r["target"] for r in records)),
        "perspective_counts": dict(Counter(r["perspective"] for r in records)),
        "n_distinct_case_ids": len({r["case_id"] for r in records}),
    }


def build_sft_example(tokenizer, prompt: str, target: str) -> dict:
    """Tokenize one (prompt, target) into {input_ids, labels} with COMPLETION-ONLY
    masking: prompt tokens get label -100, only the assistant answer tokens train."""
    user = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(user, add_generation_prompt=True, tokenize=True)
    full_ids = tokenizer.apply_chat_template(
        user + [{"role": "assistant", "content": target}], tokenize=True)
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    return {"input_ids": list(full_ids), "labels": labels}


class CompletionOnlyCollator:
    """Pad input_ids (with pad_token) and labels (with -100) to the batch max."""
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            n = len(f["input_ids"]); pad = maxlen - n
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def latest_checkpoint(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    cks = [(int(c.name.rsplit("-", 1)[1]), c) for c in path.glob("checkpoint-*")
           if c.is_dir() and c.name.rsplit("-", 1)[1].isdigit()]
    return max(cks, key=lambda t: t[0])[1] if cks else None


FORBIDDEN_TRAINING_SUBSTRINGS = ("test_goods", "frozen_unused", "ood_new_goods", "ood_")


def assert_clean_training_data(data_file) -> None:
    """Training may use ONLY remaining_goods (+ trial for smoke). It must never
    read test_goods (validation), frozen-unused, or OOD data."""
    s = str(data_file).lower()
    for bad in FORBIDDEN_TRAINING_SUBSTRINGS:
        if bad in s:
            raise SystemExit(f"REFUSED: training data '{data_file}' contains forbidden set '{bad}' "
                             "(test_goods is validation-only; frozen-unused/OOD are closed).")


def guard_output_dir(output_dir: str) -> None:
    """Refuse to write into any confirmatory GRPO / other-treatment directory."""
    name = Path(output_dir).name
    forbidden = ("grpo_qwen_delta", "grpo_20260427_1724", "grpo_qwen_delta_sign")
    if name in ("grpo_qwen_delta",) or name.startswith(("grpo_qwen_delta_seed",
                                                        "grpo_qwen_delta_sign")):
        raise SystemExit(f"REFUSED: SFT output_dir '{output_dir}' collides with a GRPO run dir.")
    for f in forbidden:
        if f in output_dir and "sft" not in name:
            raise SystemExit(f"REFUSED: SFT output_dir '{output_dir}' looks like a non-SFT run dir.")


def write_manifest(output_dir: Path, cfg: dict, args, stats: dict, sources: dict, max_steps: int, seed: int):
    manifest = {
        "baseline": "matched SFT (Qwen-own-delta rational-choice targets)",
        "protocol": "CAUSAL_BASELINE_PROTOCOL.md",
        "output_dir": str(output_dir), "seed": seed,
        "loss": "completion-only (prompt tokens masked with -100)",
        "target_rule": "reward_functions.rational_choice(perspective, delta)",
        "dataset": stats,
        "sources": sources,
        "matching": {"primary_rule": "unique prompt/data exposure",
                     "max_steps": max_steps, "per_device_train_batch_size": cfg.get("per_device_train_batch_size", 1),
                     "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 1),
                     "unique_prompts_at_endpoint": max_steps * cfg.get("per_device_train_batch_size", 1)
                                                   * cfg.get("gradient_accumulation_steps", 1),
                     "note": "1 step = 1 unique prompt = 1 optimizer update at batch 1 / accum 1"},
        "lora": {"r": cfg.get("lora_r", 16), "alpha": cfg.get("lora_alpha", 32),
                 "dropout": cfg.get("lora_dropout", 0.05),
                 "target_modules": cfg.get("lora_target_modules")},
        "optimizer": {"learning_rate": cfg.get("learning_rate", 1e-6),
                      "lr_scheduler_type": cfg.get("lr_scheduler_type", "cosine"),
                      "warmup_ratio": cfg.get("warmup_ratio", 0.05),
                      "max_grad_norm": cfg.get("max_grad_norm", 0.1)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "sft_dataset_manifest.json"
    json.dump(manifest, open(out, "w"), indent=2)
    open(out, "a").write("\n")
    logger.info(f"Wrote manifest: {out}")
    return manifest


def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.chdir(PROJECT_ROOT)

    guard_output_dir(args.output_dir)
    assert_clean_training_data(args.data_file)
    model_path = resolve(cfg["model_name_or_path"])
    delta_path = resolve(cfg.get("delta_file", "data/deltas/delta_qwen_base.json"))
    goods_json = resolve(cfg.get("goods_file", "everyday_goods_full.json"))
    data_dir = resolve("data")
    data_file = resolve(args.data_file)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    max_steps = args.max_steps if args.max_steps is not None else cfg.get("max_steps", 30000)
    save_steps = args.save_steps if args.save_steps is not None else cfg.get("save_steps", 2000)

    for pth in (delta_path, goods_json, data_file):
        if not pth.exists():
            raise SystemExit(f"Missing required input: {pth}")

    logger.info(f"Building SFT records from {data_file.name} (seed {seed})")
    records = build_sft_records(data_file, goods_json, delta_path, data_dir)
    stats = dataset_stats(records)
    logger.info(f"SFT dataset: {stats}")

    sources = {"data_file": {"path": str(data_file.relative_to(PROJECT_ROOT)), "sha256": sha256_of(data_file)},
               "delta_file": {"path": str(delta_path.relative_to(PROJECT_ROOT)), "sha256": sha256_of(delta_path)},
               "goods_file": {"path": str(goods_json.relative_to(PROJECT_ROOT)),
                              "sha256": (sha256_of(goods_json) if goods_json.is_file() else "symlink/unavailable")}}
    write_manifest(resolve(args.output_dir), cfg, args, stats, sources, max_steps, seed)

    if args.validate:
        logger.info("--validate: dataset + manifest built, no training. Exiting.")
        # (training-data contamination already checked by assert_clean_training_data above)
        print(json.dumps({"validate_ok": True, **stats}, indent=2))
        return

    # ── training path (GPU) ──────────────────────────────────────────────────
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Tokenizing (completion-only) ...")
    import datasets
    tok_ds = datasets.Dataset.from_list(records).map(
        lambda r: build_sft_example(tokenizer, r["prompt"], r["target"]),
        remove_columns=["prompt", "perspective", "delta", "case_id", "target"],
        desc="Tokenizing SFT examples")

    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj",
                                                       "gate_proj", "up_proj", "down_proj"]),
        bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir, seed=seed, data_seed=cfg.get("data_seed", seed),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg.get("learning_rate", 1e-6),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05), max_grad_norm=cfg.get("max_grad_norm", 0.1),
        max_steps=max_steps, num_train_epochs=cfg.get("num_train_epochs", 1),
        bf16=cfg.get("bf16", True), logging_steps=cfg.get("logging_steps", 10),
        save_steps=save_steps, report_to=cfg.get("report_to", "none"),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0), remove_unused_columns=False)

    trainer = Trainer(model=model, args=targs, train_dataset=tok_ds,
                      data_collator=CompletionOnlyCollator(tokenizer.pad_token_id))

    resume = args.resume_from_checkpoint
    if resume == "auto":
        ck = latest_checkpoint(resolve(args.output_dir))
        resume = str(ck) if ck else None
        logger.info(f"Auto-resume: {resume or 'none (fresh start)'}")

    logger.info(f"Starting SFT: max_steps={max_steps}, batch={targs.per_device_train_batch_size}"
                f"×{targs.gradient_accumulation_steps}, lr={targs.learning_rate}, seed={seed}")
    trainer.train(resume_from_checkpoint=resume)
    final = Path(args.output_dir) / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    logger.info(f"Final SFT LoRA saved to {final}. Done.")


if __name__ == "__main__":
    main()
