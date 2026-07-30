#!/usr/bin/env python3
"""Estimator-recovery simulation for the robustness (lambda, eta) estimator (Task 4).

Simulates paired endowment data from KNOWN (lambda*, eta*) using the frozen
structural link, refits with eval/robust_inference.fit_lambda_eta, and reports
bias, pair-clustered bootstrap CI coverage, failure rate, and weak-identification
diagnostics across a grid. Validates that the robustness estimator recovers the
truth and that its clustered intervals have approximately nominal coverage.

Deterministic given --seed. The full grid is a CPU-only PBS job
(train/submit_cpu_recovery.pbs); a small grid runs interactively.

Usage:
    python3 eval/estimator_recovery.py --quick
    python3 eval/estimator_recovery.py --n_cases 4945 --reps 200 --n_boot 300 \
        --out results/estimator_recovery/recovery.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_inference import fit_lambda_eta, pair_clustered_bootstrap, PairedCase  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def simulate(n, lam, eta, rng, util_sd=0.4, binary=False, n_resp=1):
    """Generate V_X, V_Y and observed P(No) per perspective from the link."""
    vx = np.exp(rng.normal(0, util_sd, n))
    vy = np.exp(rng.normal(0, util_sd, n))
    zx = (1 + lam) * vx - vy + eta
    zy = (1 + lam) * vy - vx + eta
    px = 1 / (1 + np.exp(-zx))
    py = 1 / (1 + np.exp(-zy))
    if binary:
        # Model B regime: observed choice is a noisy draw -> P(No) in {0,1} avg
        px = rng.binomial(n_resp, px) / n_resp
        py = rng.binomial(n_resp, py) / n_resp
    return vx, vy, px, py


def run_cell(n, lam, eta, reps, n_boot, base_seed, binary):
    biases_lam, biases_eta = [], []
    cover_lam = cover_eta = 0
    failures = 0
    jac_conds, lam_ses = [], []
    for r in range(reps):
        rng = np.random.default_rng(base_seed + r)
        vx, vy, px, py = simulate(n, lam, eta, rng, binary=binary)
        fit = fit_lambda_eta(vx, vy, px, py)
        if not (fit.success and np.isfinite(fit.lam)):
            failures += 1
            continue
        biases_lam.append(fit.lam - lam)
        biases_eta.append(fit.eta - eta)
        jac_conds.append(fit.jac_cond)
        # coverage via pair-clustered bootstrap on this replicate
        cases = [PairedCase(i, 0, 1, (), px[i], py[i]) for i in range(n)]
        boot = pair_clustered_bootstrap(cases, vx, vy, n_boot=n_boot, seed=base_seed + 10000 + r)
        lam_ses.append(boot.lam_se)
        if boot.lam_ci[0] <= lam <= boot.lam_ci[1]:
            cover_lam += 1
        if boot.eta_ci[0] <= eta <= boot.eta_ci[1]:
            cover_eta += 1
    ok = reps - failures
    return {
        "n_cases": n, "lambda_true": lam, "eta_true": eta, "reps": reps,
        "failures": failures, "failure_rate": failures / reps,
        "lambda_bias_mean": float(np.mean(biases_lam)) if biases_lam else None,
        "lambda_bias_sd": float(np.std(biases_lam)) if biases_lam else None,
        "eta_bias_mean": float(np.mean(biases_eta)) if biases_eta else None,
        "lambda_ci_coverage": cover_lam / ok if ok else None,
        "eta_ci_coverage": cover_eta / ok if ok else None,
        "jac_cond_median": float(np.median(jac_conds)) if jac_conds else None,
        "lam_se_median": float(np.median(lam_ses)) if lam_ses else None,
        "weak_identification": bool(np.median(jac_conds) > 1e6) if jac_conds else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny grid for a fast check")
    ap.add_argument("--n_cases", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--binary", action="store_true", help="Model-B binary-response regime")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.quick:
        grid = [(200, 0.1, 0.0), (200, 2.0, 0.5)]
        reps, n_boot = 20, 80
    else:
        grid = [(args.n_cases, lam, eta)
                for lam in (0.05, 0.5, 2.0, 5.0) for eta in (0.0, 0.5, 1.0)]
        reps, n_boot = args.reps, args.n_boot

    cells = []
    for (n, lam, eta) in grid:
        cell = run_cell(n, lam, eta, reps, n_boot, args.seed, args.binary)
        cells.append(cell)
        print(f"  lam*={lam:<4} eta*={eta:<3} n={n:<5} | "
              f"bias(lam)={cell['lambda_bias_mean']:+.3f} "
              f"cov(lam)={cell['lambda_ci_coverage']} "
              f"fail={cell['failure_rate']:.2f} "
              f"jac_cond~{cell['jac_cond_median']:.0f}")

    doc = {
        "purpose": "Estimator-recovery for the robustness (lambda,eta) estimator (Task 4).",
        "regime": "binary(Model-B-like)" if args.binary else "probabilistic(Model-A-like)",
        "seed": args.seed, "reps": reps, "n_boot": n_boot,
        "nominal_ci": 0.95, "cells": cells,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
