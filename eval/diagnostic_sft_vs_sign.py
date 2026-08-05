#!/usr/bin/env python3
"""SFT vs sign-only pilot diagnostic — structural + direct ownership axis (CPU).

EXPLORATORY / POST-HOC. A diagnostic of reward-hacking vs task-simplicity, NOT a
method-winner test. This module covers the parts computable WITHOUT new
inference, from existing test_goods (validation) artifacts:

  - lambda, eta (+ SE), d = sqrt(lambda^2 + eta^2)   [NLS Model-A CSV]
  - direct ownership effects: keep-both / trade-both rates, consistency
  - target agreement vs the frozen rational choice
  - hard-choice yes-rate, parse failures

for: matched base, SFT seed1 step6000, sign-only GRPO seed1 step6000.

The surface-form stress tests (label swaps, keep/trade wording, order reversal,
paraphrases) and genuinely-new-goods (OOD-50) require GPU inference and are run
separately; this file is the immediate CPU portion.

    python3 eval/diagnostic_sft_vs_sign.py
    python3 eval/diagnostic_sft_vs_sign.py --check
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DELTA_FILE = ROOT / "data" / "deltas" / "delta_qwen_base.json"
OUT_DIR = ROOT / "results" / "sft_sign_diagnostic"
OUT_JSON = OUT_DIR / "structural_ownership.json"
OUT_MD = OUT_DIR / "structural_ownership.md"

MODELS = [
    ("matched_base", "baseline/Qwen-7B-Base-Local", "Qwen-7B-Base-Local"),
    ("sft_seed1_step6000", "baselines/Qwen-7B-SFT-qd-seed1-ckpt6000", "Qwen-7B-SFT-qd-seed1-ckpt6000"),
    ("sign_only_seed1_step6000", "baselines/Qwen-7B-GRPO-sign-seed1-ckpt6000", "Qwen-7B-GRPO-sign-seed1-ckpt6000"),
]


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            d.update(c)
    return d.hexdigest()


def read_nls(csv_path: Path):
    lam = eta = lse = ese = None
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            n = r["Parameter"].strip()
            if n == "lambda":
                lam, lse = float(r["Estimate"]), float(r["Std. Err."])
            elif n == "eta":
                eta, ese = float(r["Estimate"]), float(r["Std. Err."])
    return lam, eta, lse, ese


def choice(e):
    if "Yes / No prob" not in e:
        return ""
    yes, no = e["Yes / No prob"]
    return "Yes" if float(yes) > float(no) else "No"


def rational(persp, delta):
    if delta == 0.0:
        return ""
    if persp == "X":
        return "No" if delta > 0 else "Yes"
    return "Yes" if delta > 0 else "No"


def ownership(model_dir: Path, deltas: dict):
    xp = model_dir / "Model_1" / "loss_aversion_X_for_Model_A.json"
    yp = model_dir / "Model_1" / "loss_aversion_Y_for_Model_A.json"
    xs = {int(e["case_id"]): e for e in json.load(open(xp))}
    ys = {int(e["case_id"]): e for e in json.load(open(yp))}
    ids = sorted(set(xs) & set(ys))
    n = keep = trade = cons = pf = 0
    tgt_ok = tgt_tot = yes_ct = 0
    for cid in ids:
        cx, cy = choice(xs[cid]), choice(ys[cid])
        if not cx or not cy:
            pf += 1
            continue
        n += 1
        yes_ct += (cx == "Yes")
        if cx == "No" and cy == "No":
            keep += 1
        elif cx == "Yes" and cy == "Yes":
            trade += 1
        else:
            cons += 1
        d = deltas.get(str(cid))
        if d is not None:
            dv = float(d["mean_delta"])
            for p, ch in (("X", cx), ("Y", cy)):
                r = rational(p, dv)
                if r:
                    tgt_tot += 1
                    tgt_ok += (ch == r)
    return {
        "n": n, "keep_both_rate": keep / n, "trade_both_rate": trade / n,
        "consistency": cons / n, "hard_choice_yes_rate": yes_ct / n,
        "target_agreement": (tgt_ok / tgt_tot) if tgt_tot else None,
        "parse_failure_rate": pf / len(ids),
        "x_sha256": sha256_file(xp), "y_sha256": sha256_file(yp),
    }


def build():
    deltas = json.load(open(DELTA_FILE))
    rows = []
    for key, d, name in MODELS:
        md = ROOT / d
        nls = md / "Model_1" / f"{name}_NLS_estimation_T1(Model A).csv"
        lam, eta, lse, ese = read_nls(nls)
        own = ownership(md, deltas)
        rows.append({
            "model": key, "dir": d,
            "lambda": lam, "lambda_se": lse, "eta": eta, "eta_se": ese,
            "d": math.sqrt(lam * lam + eta * eta),
            **own,
            "nls_csv": str(nls.relative_to(ROOT)),
        })
    base = rows[0]
    for r in rows:
        r["lambda_vs_base_pp_of_base"] = None if base["lambda"] == 0 else r["lambda"] / base["lambda"]
        r["keep_both_delta_vs_base"] = r["keep_both_rate"] - base["keep_both_rate"]
    return {
        "status": "EXPLORATORY_POST_HOC",
        "purpose": "Diagnostic (reward-hacking vs task-simplicity), structural + direct-ownership axis on test_goods (validation). NOT a method-winner test.",
        "dataset": "test_goods.json (validation, already informs reward/selection)",
        "delta_file_sha256": sha256_file(DELTA_FILE),
        "models": rows,
        "reads_only_existing_artifacts": True,
        "surface_form_and_new_goods": "require GPU inference; run separately",
    }


def render_md(doc):
    L = ["# SFT vs sign-only pilot diagnostic — structural + ownership (CPU)", "",
         "*EXPLORATORY / POST-HOC. test_goods validation. Reward-hacking vs "
         "task-simplicity diagnostic, NOT a method winner. Surface-form stress "
         "tests and new goods (OOD-50) are GPU-pending.*", "",
         "| model | λ (SE) | η (SE) | d | keep-both | trade-both | consistency | target-agree | yes-rate | parse |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["models"]:
        ta = f"{r['target_agreement']:.3f}" if r["target_agreement"] is not None else "—"
        L.append(f"| {r['model']} | {r['lambda']:.3f} ({r['lambda_se']:.3f}) | "
                 f"{r['eta']:.3f} ({r['eta_se']:.3f}) | {r['d']:.3f} | "
                 f"{r['keep_both_rate']:.3f} | {r['trade_both_rate']:.3f} | "
                 f"{r['consistency']:.3f} | {ta} | {r['hard_choice_yes_rate']:.3f} | "
                 f"{r['parse_failure_rate']:.3f} |")
    L += ["", "Base keep-both is the loss-averse signature (endowed good kept from "
          "both sides). Lower keep-both + lower λ + higher consistency = reduced "
          "ownership dependence.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    doc = build()
    outs = {OUT_JSON: json.dumps(doc, indent=2) + "\n", OUT_MD: render_md(doc)}
    if args.check:
        bad = [p.name for p, s in outs.items() if not p.exists() or p.read_text() != s]
        if bad:
            raise SystemExit("CHECK FAILED: " + ", ".join(bad))
        print("CHECK PASSED: diagnostic outputs byte-identical.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, s in outs.items():
        p.write_text(s)
        print(f"Wrote {p.relative_to(ROOT)}")
    for r in doc["models"]:
        print(f"  {r['model']:>26}: lam={r['lambda']:.3f} eta={r['eta']:.3f} "
              f"keep={r['keep_both_rate']:.3f} cons={r['consistency']:.3f}")


if __name__ == "__main__":
    main()
