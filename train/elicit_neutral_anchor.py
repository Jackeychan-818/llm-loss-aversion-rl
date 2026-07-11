#!/usr/bin/env python3
"""
Elicit the FROZEN ownership-free preference anchor for Reward Design v2
(REWARD_DESIGN_V2.md §2 → R_neutral).

For each case we ask the base model, with NO ownership framing, which good it
prefers — as an A/B forced choice read from the single-token log-prob of "A"
vs "B". To cancel prompt/order artifacts we average over:
  * several neutral paraphrases, and
  * both display orders (good X shown as A, and good X shown as B).

A case is frozen to pref "X"/"Y" only if the averaged P(prefer X) clears a
margin AND the per-variant choices are stable; otherwise it is left ambiguous
(no R_neutral signal). The output is keyed by GLOBAL case_id so it lines up
with prompt_builder / the delta files.

    python train/elicit_neutral_anchor.py --data_file data/test_goods.json --dry_run
    # then, on a GPU node:
    python train/elicit_neutral_anchor.py --data_file data/test_goods.json \
        --model_path models/Qwen2.5-7B-Instruct --batch_size 32

Design note: the anchor is a FROZEN NEUTRAL PSEUDO-LABEL, not ground truth.
Source it from the frozen base model (default), and validate a sample against
human labels before trusting it (REWARD_DESIGN_V2.md reviewer note 4).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from prompt_builder import GoodsData, decode_attr, compute_case_id_offset

ROOT = Path(__file__).resolve().parent.parent

# Ownership-free paraphrases. {A}/{B} are the two good descriptions; the answer
# is always a single letter so we can read it from one token's log-prob.
PARAPHRASES = [
    ("Consider two goods.\nOption A: {A}\nOption B: {B}\n"
     "Which would you prefer for yourself? Answer with the single letter A or B."),
    ("Here are two options.\nA) {A}\nB) {B}\n"
     "If you had to pick one for your own use, which is better? Reply with just A or B."),
]


def describe_good(goods: GoodsData, item: str, a1: int, a2: int) -> str:
    """Ownership-free description of a good at attribute levels (a1, a2)."""
    return (
        f"a {item.lower()} — in terms of {goods.attribute(item, 1)}, "
        f"{goods.value(item, 1, a1)}; in terms of {goods.attribute(item, 2)}, "
        f"{goods.value(item, 2, a2)}"
    )


def build_variants(goods: GoodsData, X: str, Y: str, i: int, j: int, k: int, l: int):
    """Return list of (prompt_text, x_is_A). Both orders × all paraphrases."""
    desc_x = describe_good(goods, X, i, j)
    desc_y = describe_good(goods, Y, k, l)
    variants = []
    for tmpl in PARAPHRASES:
        variants.append((tmpl.format(A=desc_x, B=desc_y), True))   # X shown as A
        variants.append((tmpl.format(A=desc_y, B=desc_x), False))  # X shown as B
    return variants


# ── Model scoring (lazy imports so --dry_run needs no torch/GPU) ─────────────
def load_model(model_path: str, adapter_path: str | None, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_path)
    # score_batch reads the logit at attention_mask.sum()-1, which is the last
    # real token only under RIGHT padding — pin it so batching stays correct.
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map="auto"
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def letter_token_ids(tok):
    """Candidate token ids for 'A' and 'B' (with and without leading space)."""
    ids = {"A": set(), "B": set()}
    for letter in ("A", "B"):
        for s in (letter, " " + letter):
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids[letter].add(enc[0])
    if not ids["A"] or not ids["B"]:
        raise RuntimeError("Could not resolve single-token ids for 'A'/'B'.")
    return ids


def score_batch(model, tok, prompts, ab_ids):
    """Return list of (p_A, p_B) normalized over the A/B letters, teacher-forced."""
    import torch
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits
    # last real token position per row (left/right padding safe via attention mask)
    last = enc["attention_mask"].sum(dim=1) - 1
    out = []
    a_ids = torch.tensor(sorted(ab_ids["A"]), device=model.device)
    b_ids = torch.tensor(sorted(ab_ids["B"]), device=model.device)
    for r in range(logits.size(0)):
        probs = torch.softmax(logits[r, last[r]].float(), dim=-1)
        pa = probs[a_ids].sum().item()
        pb = probs[b_ids].sum().item()
        tot = pa + pb
        out.append((pa / tot, pb / tot) if tot > 0 else (0.5, 0.5))
    return out


def iter_cases(goods: GoodsData, data_file: Path, offset: int, limit: int | None):
    """Yield (case_id, X, Y, i, j, k, l) matching prompt_builder's case ordering."""
    data = json.load(open(data_file))
    local = 0
    for entry in data:
        X_num, Y_num, attr_list = entry[0], entry[1], entry[2]
        if not isinstance(attr_list, list):
            attr_list = [attr_list]
        X, Y = goods.items[X_num], goods.items[Y_num]
        for code in attr_list:
            i, j, k, l = decode_attr(code)
            yield offset + local, X, Y, i, j, k, l
            local += 1
            if limit and local >= limit:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", default=None, help="Leave empty for the frozen BASE model")
    ap.add_argument("--goods_json", default="everyday_goods_full.json")
    ap.add_argument("--data_file", default="data/trial_goods.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="Freeze only if |mean P(prefer X) - 0.5| > margin")
    ap.add_argument("--agree", type=float, default=0.75,
                    help="Freeze only if >= this fraction of variants agree on direction")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true", help="Build prompts only; no model")
    args = ap.parse_args()

    goods = GoodsData(str(ROOT / args.goods_json))
    stem = Path(args.data_file).stem
    offset = compute_case_id_offset(stem, str(ROOT / "data"))
    data_file = ROOT / args.data_file
    out_path = Path(args.out) if args.out else ROOT / "data" / "anchors" / f"neutral_anchor_{stem}.json"

    if args.dry_run:
        print(f"[dry-run] case_id offset for {stem}: {offset}")
        for cid, X, Y, i, j, k, l in iter_cases(goods, data_file, offset, limit=2):
            variants = build_variants(goods, X, Y, i, j, k, l)
            print(f"\n=== case_id {cid}: {X} vs {Y}  levels=({i},{j},{k},{l}) — {len(variants)} variants ===")
            for text, x_is_A in variants[:2]:
                print(f"  [X is {'A' if x_is_A else 'B'}] {text!r}")
        print("\n[dry-run] prompt construction OK — no model loaded.")
        return

    model, tok = load_model(args.model_path, args.adapter_path, args.dtype)
    ab_ids = letter_token_ids(tok)

    anchors, frozen, ambiguous = {}, 0, 0
    # flatten (case, variants) then batch across everything for GPU efficiency
    pending_prompts, pending_meta = [], []   # meta: (case_id, x_is_A)

    def flush():
        nonlocal frozen, ambiguous
        if not pending_prompts:
            return
        scored = score_batch(model, tok, pending_prompts, ab_ids)
        by_case: dict[int, list[float]] = {}
        for (cid, x_is_A), (pa, pb) in zip(pending_meta, scored):
            p_x = pa if x_is_A else pb          # P(prefer X) for this variant
            by_case.setdefault(cid, []).append(p_x)
        for cid, ps in by_case.items():
            mean_p = statistics.fmean(ps)
            agree = statistics.fmean([1.0 if (p > 0.5) == (mean_p > 0.5) else 0.0 for p in ps])
            stable = abs(mean_p - 0.5) > args.margin and agree >= args.agree
            pref = ("X" if mean_p > 0.5 else "Y") if stable else None
            anchors[str(cid)] = {"pref": pref, "p_x": round(mean_p, 4),
                                 "agree": round(agree, 3), "n_variants": len(ps)}
            frozen += stable
            ambiguous += (not stable)
        pending_prompts.clear()
        pending_meta.clear()

    n = 0
    for cid, X, Y, i, j, k, l in iter_cases(goods, data_file, offset, args.limit):
        for text, x_is_A in build_variants(goods, X, Y, i, j, k, l):
            pending_prompts.append(text)
            pending_meta.append((cid, x_is_A))
        n += 1
        # flush on batch boundary (keep all variants of a case together)
        if len(pending_prompts) >= args.batch_size:
            flush()
        if n % 500 == 0:
            print(f"  ...{n} cases  (frozen={frozen}, ambiguous={ambiguous})")
    flush()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"_summary": {"data_file": stem, "cases": len(anchors),
                            "frozen": frozen, "ambiguous": ambiguous,
                            "margin": args.margin, "agree": args.agree},
               "anchors": anchors}, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")
    print(f"  cases={len(anchors)}  frozen={frozen} ({frozen/max(len(anchors),1)*100:.1f}%)  ambiguous={ambiguous}")


if __name__ == "__main__":
    main()
