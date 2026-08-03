#!/usr/bin/env python3
"""Neutral-preference INFERENCE harness (GPU; submission deferred).

Scores the frozen 96-case subset under neutral (no-ownership) A/B preference
forms for one model, using the same teacher-forced candidate scoring as the
surface-form harness. Writes results/neutral_preference/<model>/neutral_predictions.jsonl.

    python eval/neutral_preference_infer.py --model_name Base --adapter_path "" ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from surface_form_infer import (  # noqa: E402  (torch loaded here)
    apply_chat_template, score_forms, sha256_file,
)
from neutral_preference import Goods, build_neutral_forms, canonical_choice, _norm  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SUBSET = ROOT / "data" / "surface_form_stress" / "surface_form_subset.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", default="")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--output_root", default="results/neutral_preference")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--batch_rows", type=int, default=64)
    args = ap.parse_args()

    subset = json.load(open(SUBSET))
    goods = Goods()
    out_dir = ROOT / args.output_root / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "neutral_predictions.jsonl"

    all_forms = []
    for c in subset["cases"]:
        all_forms.extend(build_neutral_forms(goods, c["case_id"], c["X_num"], c["Y_num"], c["attr_code"]))
    expected = len(all_forms)
    if pred_path.exists() and sum(1 for _ in open(pred_path)) == expected:
        print(f"RESUME: {args.model_name} complete ({expected}) — skipping.")
        return

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

    jobs = [(apply_chat_template(tok, f.prompt), f.answer_tokens) for f in all_forms]
    print(f"Scoring {len(all_forms)} neutral forms ({len(subset['cases'])} cases x 4)")
    probs = score_forms(model, tok, jobs, batch_rows=args.batch_rows)

    by_case = {c["case_id"]: c for c in subset["cases"]}
    tmp = pred_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as fh:
        for f, pr in zip(all_forms, probs):
            chosen = max(pr, key=pr.get)
            fh.write(json.dumps({
                "case_id": f.case_id, "order": f.order, "paraphrase": f.paraphrase,
                "form_id": f.form_id, "good_for_A": f.good_for_A, "good_for_B": f.good_for_B,
                "chosen_token": chosen, "chosen_good": canonical_choice(f, chosen),
                "p_A": pr.get("A", 0.0), "p_B": pr.get("B", 0.0),
                "delta": by_case[f.case_id]["delta"],
                "delta_sign": by_case[f.case_id]["delta_sign"],
                "delta_bin": by_case[f.case_id]["delta_bin"],
            }) + "\n")
    tmp.replace(pred_path)
    (out_dir / "run_metadata.json").write_text(json.dumps({
        "model_name": args.model_name, "adapter_path": args.adapter_path or None,
        "adapter_sha256": (sha256_file(Path(args.adapter_path) / "adapter_model.safetensors")
                           if args.adapter_path else None),
        "subset_sha256": sha256_file(SUBSET), "n_forms": expected,
        "scoring": "teacher-forced A/B candidate log-prob, argmax",
    }, indent=2) + "\n")
    print(f"Wrote {pred_path} ({expected} rows)")


if __name__ == "__main__":
    main()
