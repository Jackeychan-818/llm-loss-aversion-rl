#!/usr/bin/env python3
"""Surface-form stress INFERENCE harness (GPU).

For one model (base or base+adapter), evaluates every frozen-subset case across
both perspectives and all 48 equivalent forms, using the SAME teacher-forced
candidate scoring as run_qwen_local (log-prob of each answer token, normalized).
Each form's chosen token is canonicalized to the physical final good via
surface_form_transforms.canonical_final_good.

Writes results/surface_form_stress/<model_name>/form_predictions.jsonl (one row
per form) + run_metadata.json. Raw predictions are NOT committed.

    python eval/surface_form_infer.py --model_name Base --adapter_path "" ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from surface_form_transforms import Goods, build_forms, canonical_final_good  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SUBSET = ROOT / "data" / "surface_form_stress" / "surface_form_subset.json"


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            d.update(c)
    return d.hexdigest()


def apply_chat_template(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_forms(model, tokenizer, jobs, batch_rows=64):
    """jobs: list of (prompt_text, [candidate_strings]). Returns list of dicts
    {cand: prob} (normalized over that job's candidates). Teacher-forced, mirrors
    run_qwen_local.score_yes_no."""
    # Flatten to (job_idx, cand, ids)
    rows = []
    cand_ids_cache: dict[str, list[int]] = {}
    prompt_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p, _ in jobs]
    for ji, (_, cands) in enumerate(jobs):
        for c in cands:
            if c not in cand_ids_cache:
                cand_ids_cache[c] = tokenizer.encode(c, add_special_tokens=False)
            rows.append((ji, c, prompt_ids[ji], cand_ids_cache[c]))

    scores = [dict() for _ in jobs]
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start:start + batch_rows]
        full = [pi + ci for _, _, pi, ci in chunk]
        max_len = max(len(x) for x in full)
        input_ids = torch.full((len(full), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(full), max_len), dtype=torch.long)
        for r, ids in enumerate(full):
            input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[r, :len(ids)] = 1
        input_ids = input_ids.to(device); attn = attn.to(device)
        logp = torch.log_softmax(model(input_ids=input_ids, attention_mask=attn).logits, dim=-1)
        for r, (ji, c, pi, ci) in enumerate(chunk):
            s = 0.0
            for off in range(len(ci)):
                pos = len(pi) + off
                s += float(logp[r, pos - 1, int(input_ids[r, pos])])
            scores[ji][c] = s
    # normalize per job
    out = []
    for sc in scores:
        m = max(sc.values())
        exp = {k: math.exp(v - m) for k, v in sc.items()}
        z = sum(exp.values())
        out.append({k: v / z for k, v in exp.items()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", default="")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--output_root", default="results/surface_form_stress")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--batch_rows", type=int, default=64)
    args = ap.parse_args()

    subset = json.load(open(SUBSET))
    goods = Goods()
    out_dir = ROOT / args.output_root / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume guard: if this model's predictions are already complete, skip it so
    # a resubmitted (timed-out) job never re-runs or mixes a finished model dir.
    expected_forms = len(subset["cases"]) * 2 * 48
    pred_path = out_dir / "form_predictions.jsonl"
    if pred_path.exists():
        have = sum(1 for _ in open(pred_path))
        if have == expected_forms:
            print(f"RESUME: {args.model_name} already complete ({have} rows) — skipping.")
            return
        print(f"RESUME: {args.model_name} partial ({have}/{expected_forms}) — recomputing.")

    print(f"Loading {args.model_path} (adapter={args.adapter_path or 'none'})")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dt,
                                                 device_map="auto", trust_remote_code=True)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # Build all forms + chat-templated prompts
    all_forms = []
    for c in subset["cases"]:
        for persp in ("X", "Y"):
            all_forms.extend(build_forms(goods, c["case_id"], c["X_num"], c["Y_num"],
                                         c["attr_code"], persp))
    jobs = [(apply_chat_template(tok, f.prompt), f.answer_tokens) for f in all_forms]
    print(f"Scoring {len(all_forms)} forms ({len(subset['cases'])} cases x 2 persp x 48)")
    probs = score_forms(model, tok, jobs, batch_rows=args.batch_rows)

    delta_by_case = {c["case_id"]: c for c in subset["cases"]}
    tmp_path = out_dir / "form_predictions.jsonl.tmp"   # atomic: rename only on success
    with open(tmp_path, "w") as fh:
        for f, pr in zip(all_forms, probs):
            # keep/trade action masses
            p_keep = sum(p for t, p in pr.items()
                         if f.token_to_action.get(_n(t)) == "keep")
            p_trade = sum(p for t, p in pr.items()
                          if f.token_to_action.get(_n(t)) == "trade")
            chosen = max(pr, key=pr.get)
            final = canonical_final_good(f, chosen)
            fh.write(json.dumps({
                "case_id": f.case_id, "perspective": f.perspective, "form_id": f.form_id,
                "answer_style": f.answer_style, "display_order": f.display_order,
                "attr_order": f.attr_order, "paraphrase": f.paraphrase,
                "keep_good": f.keep_good, "trade_good": f.trade_good,
                "chosen_token": chosen, "final_good": final,
                "p_keep": p_keep, "p_trade": p_trade, "p_chosen": pr[chosen],
                "delta": delta_by_case[f.case_id]["delta"],
                "delta_sign": delta_by_case[f.case_id]["delta_sign"],
                "delta_bin": delta_by_case[f.case_id]["delta_bin"],
            }) + "\n")
    tmp_path.replace(pred_path)   # atomic completion

    meta = {
        "model_name": args.model_name, "model_path": args.model_path,
        "adapter_path": args.adapter_path or None,
        "adapter_sha256": (sha256_file(Path(args.adapter_path) / "adapter_model.safetensors")
                           if args.adapter_path else None),
        "subset_sha256": sha256_file(SUBSET),
        "n_cases": len(subset["cases"]), "n_forms": len(all_forms),
        "scoring": "teacher-forced candidate log-prob (as run_qwen_local), greedy argmax",
        "dtype": args.dtype,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {pred_path} ({len(all_forms)} rows)")


def _n(tok):
    from surface_form_transforms import _norm
    return _norm(tok)


if __name__ == "__main__":
    main()
