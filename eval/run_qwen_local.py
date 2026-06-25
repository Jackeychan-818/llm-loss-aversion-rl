#!/usr/bin/env python3
"""
vLLM-free local evaluator for Qwen2.5 checkpoints.

Runs the loss-aversion prompts with plain Transformers + optional PEFT LoRA and
writes the same JSON files consumed by core_exp_refactored.py:

    baseline/Qwen-7B-GRPO/loss_aversion_X.json
    baseline/Qwen-7B-GRPO/loss_aversion_Y.json

Examples:
    python eval/run_qwen_local.py \\
        --model_path models/Qwen2.5-7B-Instruct \\
        --adapter_path checkpoints/grpo_20260427_1724/final \\
        --model_name Qwen-7B-GRPO \\
        --data_file data/test_goods.json \\
        --treatment baseline \\
        --batch_size 8

    # Smoke test
    python eval/run_qwen_local.py --adapter_path checkpoints/foo/final --limit 20 --yes
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

GOODS_RUN_ORDER = ["trial_goods", "test_goods", "remaining_goods"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline local Qwen evaluator")
    p.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter_path", default=None, help="LoRA adapter directory")
    p.add_argument("--model_name", default="Qwen-7B-GRPO")
    p.add_argument("--data_file", default="data/test_goods.json")
    p.add_argument("--treatment", choices=["baseline", "debias", "forced"], default="baseline")
    p.add_argument("--goods_json", default="everyday_goods_full.json")
    p.add_argument("--output_root", default=None, help="Defaults to treatment name")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="Limit number of cases for smoke tests")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--yes", action="store_true", help="Run without interactive confirmation")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    tmp.replace(path)


class GoodsData:
    def __init__(self, goods_json_path: Path):
        self.data = load_json(goods_json_path)
        self.cats = list(self.data.keys())
        self.items: List[str] = []
        self.item_to_cat: Dict[str, str] = {}
        for cat in self.cats:
            for item in self.data[cat]:
                self.items.append(item)
                self.item_to_cat[item] = cat

    def attribute(self, item: str, num: int) -> str:
        return list(self.data[self.item_to_cat[item]][item].keys())[num - 1].lower()

    def value(self, item: str, num: int, key: int) -> str:
        attr = list(self.data[self.item_to_cat[item]][item].keys())[num - 1]
        return str(self.data[self.item_to_cat[item]][item][attr][key]).lower()


def decode_attr(attr_code: int) -> Tuple[int, int, int, int]:
    i = attr_code % 3; attr_code //= 3
    j = attr_code % 3; attr_code //= 3
    k = attr_code % 3; attr_code //= 3
    l = attr_code % 3
    return i, j, k, l


def generate_prompt(goods: GoodsData, treatment: str, X: str, Y: str,
                    i: int, j: int, k: int, l: int) -> str:
    base = (
        f"Suppose that, by chance, you receive a {X.lower()} out of many possible goods "
        f"that you could have received. \n"
        f"In terms of {goods.attribute(X, 1)}, the {X.lower()} is {goods.value(X, 1, i)}. "
        f"In terms of {goods.attribute(X, 2)}, it is {goods.value(X, 2, j)}.\n"
        f"You can keep the {X.lower()} or trade it for a {Y.lower()}. \n"
        f"In terms of {goods.attribute(Y, 1)}, the {Y.lower()} is {goods.value(Y, 1, k)}. "
        f"In terms of {goods.attribute(Y, 2)}, it is {goods.value(Y, 2, l)}."
    )

    if treatment == "baseline":
        return base + "\nWould you trade it? Answer with the single word Yes or No."
    if treatment == "debias":
        return base + (
            "\nWould you trade it? Please ensure that your answer is not affected by "
            "the status quo (an overall tendency to trade or not to trade), or gain/loss "
            "attitudes (the tendency to value attributes of the good you have differently "
            "from the good you can trade it for). Answer with the single word Yes or No."
        )
    if treatment == "forced":
        return base + (
            f"\nBut even if you choose to keep the {X.lower()}, there is a 50% probability "
            f"that the {X.lower()} will be switched for the {Y.lower()}. "
            "Would you trade it? Answer with the single word Yes or No."
        )
    raise ValueError(f"Unknown treatment: {treatment}")


def count_cases(goods_file: Path) -> int:
    data = load_json(goods_file)
    return sum(
        len(e[2]) for e in data
        if isinstance(e, (list, tuple)) and len(e) >= 3 and isinstance(e[2], list)
    )


def compute_case_id_offset(goods_path: Path) -> int:
    stem = goods_path.stem
    data_dir = goods_path.parent
    offset = 0
    for prior in GOODS_RUN_ORDER:
        if prior == stem:
            break
        prior_path = data_dir / f"{prior}.json"
        if prior_path.exists():
            offset += count_cases(prior_path)
    return offset


def build_cases(goods: GoodsData, goods_path: Path, treatment: str, limit: Optional[int]) -> List[dict]:
    raw = load_json(goods_path)
    offset = compute_case_id_offset(goods_path)
    cases = []
    local_id = 0

    for entry in raw:
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
            if limit is not None and len(cases) >= limit:
                return cases
            i, j, k, l = decode_attr(attr_code)
            case_id = offset + local_id
            cases.append({
                "case_id": case_id,
                "X_num": X_num,
                "Y_num": Y_num,
                "attr": [i, j, k, l],
                "prompt_X": generate_prompt(goods, treatment, X, Y, i, j, k, l),
                "prompt_Y": generate_prompt(goods, treatment, Y, X, k, l, i, j),
            })

    return cases


def clean_answer(text: str) -> str:
    text = text.strip()
    first = text.split()[0] if text.split() else ""
    return re.sub(r"[^a-zA-Z]", "", first).lower()


def dtype_from_name(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def candidate_token_ids(tokenizer, text: str) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Candidate {text!r} encoded to no tokens")
    return ids


@torch.no_grad()
def score_yes_no(model, tokenizer, prompt_texts: List[str]) -> List[Tuple[float, float]]:
    """Return normalized P(Yes), P(No) by scoring candidate continuations."""
    yes_ids = candidate_token_ids(tokenizer, "Yes")
    no_ids = candidate_token_ids(tokenizer, "No")
    encoded_prompts = [
        tokenizer(t, add_special_tokens=False)["input_ids"]
        for t in prompt_texts
    ]

    full_ids = []
    meta = []
    for idx, prompt_ids in enumerate(encoded_prompts):
        for label, cand_ids in (("yes", yes_ids), ("no", no_ids)):
            full_ids.append(prompt_ids + cand_ids)
            meta.append((idx, label, len(prompt_ids), len(cand_ids)))

    pad_id = tokenizer.pad_token_id
    max_len = max(len(ids) for ids in full_ids)
    input_ids = torch.full((len(full_ids), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(full_ids), max_len), dtype=torch.long)
    for row, ids in enumerate(full_ids):
        input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :len(ids)] = 1

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = torch.log_softmax(logits, dim=-1)

    scores = [{"yes": float("-inf"), "no": float("-inf")} for _ in prompt_texts]
    for row, (idx, label, prompt_len, cand_len) in enumerate(meta):
        score = 0.0
        for offset in range(cand_len):
            token_pos = prompt_len + offset
            token_id = int(input_ids[row, token_pos])
            score += float(log_probs[row, token_pos - 1, token_id])
        scores[idx][label] = score

    probs = []
    for score in scores:
        m = max(score["yes"], score["no"])
        yes = math.exp(score["yes"] - m)
        no = math.exp(score["no"] - m)
        z = yes + no
        probs.append((yes / z, no / z))
    return probs


@torch.no_grad()
def generate_answers(model, tokenizer, prompt_texts: List[str], max_new_tokens: int) -> List[str]:
    tokenizer.padding_side = "left"
    batch = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model.generate(
        **batch,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = batch["input_ids"].shape[1]
    generated = outputs[:, prompt_len:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def load_existing(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return load_json(path)


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    goods_path = resolve(args.data_file)
    goods_json_path = resolve(args.goods_json)
    model_path = resolve(args.model_path)
    adapter_path = resolve(args.adapter_path) if args.adapter_path else None
    output_root = Path(args.output_root) if args.output_root else Path(args.treatment)
    output_dir = output_root / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = output_dir / "loss_aversion_X.json"
    y_path = output_dir / "loss_aversion_Y.json"
    completed_path = output_dir / "completed_index.json"

    print("=" * 70)
    print("LOCAL QWEN EVAL")
    print("=" * 70)
    print(f"Model path:    {model_path}")
    print(f"Adapter path:  {adapter_path or '(none)'}")
    print(f"Data file:     {goods_path}")
    print(f"Treatment:     {args.treatment}")
    print(f"Output dir:    {output_dir}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Limit:         {args.limit or '(none)'}")

    goods = GoodsData(goods_json_path)
    cases = build_cases(goods, goods_path, args.treatment, args.limit)
    existing_x = load_existing(x_path)
    existing_y = load_existing(y_path)
    done_x = {row["case_id"] for row in existing_x}
    done_y = {row["case_id"] for row in existing_y}
    done = done_x & done_y
    cases = [case for case in cases if case["case_id"] not in done]

    print(f"Already complete: {len(done)}")
    print(f"Remaining cases:  {len(cases)}")
    if not cases:
        print("Nothing to do.")
        return
    if not args.yes:
        confirm = input("Proceed? Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype_from_name(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    out_x = existing_x[:]
    out_y = existing_y[:]
    processed = 0

    for start in tqdm(range(0, len(cases), args.batch_size), desc="Evaluating"):
        batch_cases = cases[start:start + args.batch_size]
        flat_prompts = []
        flat_meta = []
        for case in batch_cases:
            flat_prompts.append(apply_chat_template(tokenizer, case["prompt_X"]))
            flat_meta.append((case, "X"))
            flat_prompts.append(apply_chat_template(tokenizer, case["prompt_Y"]))
            flat_meta.append((case, "Y"))

        probs = score_yes_no(model, tokenizer, flat_prompts)
        answers = generate_answers(model, tokenizer, flat_prompts, args.max_new_tokens)

        for (case, label), (p_yes, p_no), answer in zip(flat_meta, probs, answers):
            cleaned = clean_answer(answer)
            if cleaned not in ("yes", "y", "no", "n"):
                answer = "Yes" if p_yes >= p_no else "No"
            row = {
                "case_id": case["case_id"],
                "X_num": case["X_num"],
                "Y_num": case["Y_num"],
                "attr": case["attr"],
                "Yes / No prob": [p_yes, p_no],
                "output": case[f"prompt_{label}"] + " " + answer.strip(),
            }
            if label == "X":
                out_x.append(row)
            else:
                out_y.append(row)

        processed += len(batch_cases)
        if processed % 100 == 0 or start + args.batch_size >= len(cases):
            out_x = sorted({row["case_id"]: row for row in out_x}.values(), key=lambda r: r["case_id"])
            out_y = sorted({row["case_id"]: row for row in out_y}.values(), key=lambda r: r["case_id"])
            dump_json(x_path, out_x)
            dump_json(y_path, out_y)
            dump_json(completed_path, {
                "file_case_counts": {goods_path.stem: count_cases(goods_path)},
                "completed_ids": {goods_path.stem: sorted(set(done) | {c["case_id"] for c in cases[:processed]})},
            })

    print(f"Saved {len(out_x)} X rows and {len(out_y)} Y rows to {output_dir}")


if __name__ == "__main__":
    main()
