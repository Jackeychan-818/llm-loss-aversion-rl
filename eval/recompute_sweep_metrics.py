#!/usr/bin/env python3
"""
Recompute the CORRECTED v2-core sweep metrics for every evaluated checkpoint,
from cached artifacts only (no GPU, no re-running the NLS — login-node safe):

  * behavioral metrics (consistency, keep-both, trade-both, the corrected
    anchor_match_given_consistent + unconditional joint_anchored_success) are
    recomputed from the cached full-test eval rows, restricted to the VALIDATION
    half (even case_ids) — Phase-2 selection uses even-ID data only;
  * lambda_hat, eta_hat and their SEs are read from the cached val-half NLS CSV
    (val_sweep/v2core-ckptN-val/Model_1/*_NLS_estimation_T1(Model A).csv);
  * checkpoint KL is read from checkpoints/grpo_v2core/checkpoint-N/trainer_state.json
    (mean of the last KL_WINDOW logged kl values up to the checkpoint step);
  * derived: lambda 95% CI, Euclidean distance of (lambda,eta) from (0,0), and a
    standardized (Wald-style, diagonal-only) distance sqrt((l/SEl)^2+(e/SEe)^2).

Writes the enriched results/v2core_sweep/v2core-ckptN.json (val block).

    python eval/recompute_sweep_metrics.py                # all evaluated checkpoints
    python eval/recompute_sweep_metrics.py 4000 5200      # specific steps
"""
from __future__ import annotations

import csv as _csv
import glob
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_partition_estimate import load_rows, load_anchor, guardrails

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_SWEEP = PROJECT_ROOT / "val_sweep"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "grpo_v2core"
RES_DIR = PROJECT_ROOT / "results" / "v2core_sweep"
KL_WINDOW = 20


def read_lambda_eta(step: int) -> dict:
    csvp = VAL_SWEEP / f"v2core-ckpt{step}-val" / "Model_1" / f"v2core-ckpt{step}-val_NLS_estimation_T1(Model A).csv"
    if not csvp.exists():
        return {}
    rows = list(_csv.DictReader(open(csvp)))
    lam, lam_se = float(rows[0]["Estimate"]), float(rows[0]["Std. Err."])
    eta, eta_se = float(rows[1]["Estimate"]), float(rows[1]["Std. Err."])
    ci = 1.96 * lam_se
    return {
        "lambda": lam, "lambda_se": lam_se,
        "lambda_ci95": [lam - ci, lam + ci],
        "eta": eta, "eta_se": eta_se,
        "dist_euclid": math.hypot(lam, eta),
        "dist_wald_diag": math.hypot(lam / lam_se if lam_se else 0.0,
                                     eta / eta_se if eta_se else 0.0),
    }


def read_kl(step: int) -> float:
    ts = CKPT_DIR / f"checkpoint-{step}" / "trainer_state.json"
    if not ts.exists():
        return float("nan")
    lh = json.load(open(ts)).get("log_history", [])
    kls = [e["kl"] for e in lh if "kl" in e and e.get("step", 0) <= step]
    kls = kls[-KL_WINDOW:]
    return sum(kls) / len(kls) if kls else float("nan")


def steps_available() -> list[int]:
    out = []
    for d in glob.glob(str(VAL_SWEEP / "v2core-ckpt*")):
        m = re.fullmatch(r"v2core-ckpt(\d+)", Path(d).name)
        if m and (Path(d) / "loss_aversion_X.json").exists():
            out.append(int(m.group(1)))
    return sorted(set(out))


def main():
    want = [int(a) for a in sys.argv[1:]] or steps_available()
    anchor = load_anchor()
    RES_DIR.mkdir(parents=True, exist_ok=True)
    for step in want:
        base = VAL_SWEEP / f"v2core-ckpt{step}"
        xr = load_rows(base / "loss_aversion_X.json")
        yr = load_rows(base / "loss_aversion_Y.json")
        common = sorted(set(xr) & set(yr))
        val_ids = [c for c in common if c % 2 == 0]     # validation half
        gr = guardrails(xr, yr, val_ids, anchor)
        le = read_lambda_eta(step)
        kl = read_kl(step)
        out = {"model_name": f"v2core-ckpt{step}", "step": step,
               "split": "val=even case_ids of test_goods (selection); frozen=odd (deferred)",
               "kl": kl, "val": {**le, **gr}}
        json.dump(out, open(RES_DIR / f"v2core-ckpt{step}.json", "w"), indent=2)
        v = out["val"]
        lam = v.get("lambda")
        lam_s = f"{lam:+.4f}" if lam is not None else "  n/a"
        print(f"ckpt{step:>5}: λ̂={lam_s}  η̂={v.get('eta', float('nan')):+.4f}  "
              f"joint_success={v['joint_anchored_success']:.3f}  "
              f"match|cons={v['anchor_match_given_consistent']:.3f}  "
              f"consist={v['consistent_rate']:.3f}  keep={v['keep_both_rate']:.3f}  "
              f"trade={v['trade_both_rate']:.3f}  KL={kl:.3f}")


if __name__ == "__main__":
    main()
