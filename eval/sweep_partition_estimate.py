#!/usr/bin/env python3
"""
Held-out checkpoint-selection helper for the v2-core GRPO sweep.

Given one checkpoint's FULL test_goods eval output
(``{feature}/{model_name}/loss_aversion_X.json`` + ``_Y.json``, with global test
case_ids 61-9950), this:

  1. Partitions the per-case results into a VALIDATION half and a FROZEN half by
     case_id parity (even = validation / used for SELECTION, odd = frozen / used
     only for CONFIRMATION). test_goods was never in training, so both halves are
     held out; the parity split just keeps selection and confirmation disjoint.
  2. Runs the vetted Model A (NLS, T=1) estimator on each half via
     estimate_qwen_checkpoint.py (subprocess — core_exp_refactored is untouched)
     and parses lambda_hat, eta_hat and their standard errors.
  3. Computes the guardrails from the test neutral anchor: rational-choice
     accuracy (consistent pair that picks the anchored good) and the keep-both
     (status-quo) rate.

Writes results/v2core_sweep/{model_name}.json with val + frozen blocks.

Usage:
    python eval/sweep_partition_estimate.py --feature val_sweep --model_name v2core-ckpt6600
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANCHOR_FILE = PROJECT_ROOT / "data" / "anchors" / "neutral_anchor_test_goods.json"


def load_rows(path: Path) -> dict[int, dict]:
    rows = json.load(open(path))
    out = {}
    for r in rows:
        out[int(r["case_id"])] = r
    return out


def choice_from_row(r: dict) -> str:
    """Model's Yes/No choice from the [P(Yes), P(No)] field."""
    p = r["Yes / No prob"]
    if isinstance(p, str):
        p = json.loads(p)
    return "Yes" if float(p[0]) >= float(p[1]) else "No"


def load_anchor() -> dict[int, str]:
    raw = json.load(open(ANCHOR_FILE))
    a = raw.get("anchors", raw)
    return {int(k): (v["pref"] if isinstance(v, dict) else v) for k, v in a.items()}


def write_subset(feature: str, model_name: str, base_x: dict, base_y: dict,
                 case_ids: list[int]) -> None:
    d = PROJECT_ROOT / feature / model_name
    d.mkdir(parents=True, exist_ok=True)
    json.dump([base_x[c] for c in case_ids], open(d / "loss_aversion_X.json", "w"))
    json.dump([base_y[c] for c in case_ids], open(d / "loss_aversion_Y.json", "w"))


def estimate(feature: str, model_name: str) -> dict:
    """Run Model A NLS T=1 via the vetted CLI, parse lambda/eta from the CSV."""
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "eval" / "estimate_qwen_checkpoint.py"),
         "--feature", feature, "--model_name", model_name, "--temperature", "1"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    if r.returncode != 0:
        raise RuntimeError(f"estimator failed for {feature}/{model_name}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    csv = PROJECT_ROOT / feature / model_name / "Model_1" / f"{model_name}_NLS_estimation_T1(Model A).csv"
    import csv as _csv
    with open(csv) as f:
        rows = list(_csv.DictReader(f))
    def grab(i):
        return float(rows[i]["Estimate"]), float(rows[i]["Std. Err."])
    lam, lam_se = grab(0)   # row 0 = lambda
    eta, eta_se = grab(1)   # row 1 = eta
    return {"lambda": lam, "lambda_se": lam_se, "eta": eta, "eta_se": eta_se}


def guardrails(xrows: dict, yrows: dict, case_ids: list[int], anchor: dict) -> dict:
    """Rational-choice accuracy + keep-both rate over cases WITH a stable anchor."""
    consistent = keep_both = trade_both = correct = judged = 0
    for c in case_ids:
        rx = choice_from_row(xrows[c]); ry = choice_from_row(yrows[c])
        if rx == ry:
            if rx == "No":
                keep_both += 1
            else:
                trade_both += 1
            continue
        consistent += 1
        pref = anchor.get(c)
        if pref in ("X", "Y"):
            judged += 1
            picked = "X" if rx == "No" else "Y"
            if picked == pref:
                correct += 1
    n = len(case_ids)
    return {
        "n": n,
        "consistent_rate": consistent / n if n else 0.0,
        "keep_both_rate": keep_both / n if n else 0.0,
        "trade_both_rate": trade_both / n if n else 0.0,
        # rational accuracy: of anchor-judgeable cases, fraction that are consistent+correct
        "rational_acc": correct / judged if judged else float("nan"),
        "n_judged": judged,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", default="val_sweep")
    ap.add_argument("--model_name", required=True, help="base name, e.g. v2core-ckpt6600")
    ap.add_argument("--estimate", default="val",
                    help="comma-list of halves to run the (slow) NLS on: 'val' (sweep/"
                         "selection, default) or 'val,frozen' (winner confirmation). "
                         "Guardrails are always computed for both halves.")
    args = ap.parse_args()
    est_halves = set(h.strip() for h in args.estimate.split(",") if h.strip())

    base_dir = PROJECT_ROOT / args.feature / args.model_name
    xrows = load_rows(base_dir / "loss_aversion_X.json")
    yrows = load_rows(base_dir / "loss_aversion_Y.json")
    common = sorted(set(xrows) & set(yrows))
    anchor = load_anchor()

    val_ids = [c for c in common if c % 2 == 0]     # validation (selection)
    frz_ids = [c for c in common if c % 2 == 1]     # frozen (confirmation)

    out = {"model_name": args.model_name, "n_total": len(common),
           "val_ids": len(val_ids), "frozen_ids": len(frz_ids)}
    for half, ids in (("val", val_ids), ("frozen", frz_ids)):
        mn = f"{args.model_name}-{half}"
        gr = guardrails(xrows, yrows, ids, anchor)
        if half in est_halves:
            write_subset(args.feature, mn, xrows, yrows, ids)
            est = estimate(args.feature, mn)
            out[half] = {**est, **gr}
            print(f"[{half}] lambda={est['lambda']:+.4f} (SE {est['lambda_se']:.4f})  "
                  f"eta={est['eta']:+.4f}  rational_acc={gr['rational_acc']:.3f}  "
                  f"consistent={gr['consistent_rate']:.3f}  keep_both={gr['keep_both_rate']:.3f}  n={gr['n']}")
        else:
            out[half] = {"lambda": None, "lambda_se": None, "eta": None, "eta_se": None, **gr}
            print(f"[{half}] (NLS skipped)  rational_acc={gr['rational_acc']:.3f}  "
                  f"consistent={gr['consistent_rate']:.3f}  keep_both={gr['keep_both_rate']:.3f}  n={gr['n']}")

    res_dir = PROJECT_ROOT / "results" / "v2core_sweep"
    res_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(res_dir / f"{args.model_name}.json", "w"), indent=2)
    print(f"wrote {res_dir / (args.model_name + '.json')}")


if __name__ == "__main__":
    main()
