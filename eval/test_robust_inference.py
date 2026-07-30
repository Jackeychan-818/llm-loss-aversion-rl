#!/usr/bin/env python3
"""Tests for the robustness inference layer (Task 4).

Covers strict ID-join integrity (PAIR-001 hard-fails), point recovery, pair-
clustered bootstrap, leave-one-good-out, and Jacobian conditioning. Run from the
repository root:

    python3 eval/test_robust_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from robust_inference import (  # noqa: E402
    join_paired_records, fit_lambda_eta, pair_clustered_bootstrap,
    leave_one_good_out, PairedCase,
)

_fail: list[str] = []
_pass = 0


def check(name, cond, detail=""):
    global _pass
    if cond:
        _pass += 1
        print(f"  PASS  {name}")
    else:
        _fail.append(f"{name}: {detail}")
        print(f"  FAIL  {name}: {detail}")


def rec(cid, xnum, ynum, no):
    return {"case_id": cid, "X_num": xnum, "Y_num": ynum, "attr": [0, 0, 0, 0],
            "Yes / No prob": [1 - no, no]}


# --- 1. valid join ----------------------------------------------------------
xs = [rec(1, 0, 1, 0.9), rec(2, 0, 2, 0.8)]
ys = [rec(1, 0, 1, 0.2), rec(2, 0, 2, 0.3)]
cases = join_paired_records(xs, ys)
check("valid join length", len(cases) == 2, str(len(cases)))
check("valid join pno", abs(cases[0].pno_x - 0.9) < 1e-9 and abs(cases[0].pno_y - 0.2) < 1e-9, "")

# --- 2. hard-fail: duplicate ID ---------------------------------------------
try:
    join_paired_records([rec(1, 0, 1, 0.9), rec(1, 0, 2, 0.8)], [rec(1, 0, 1, 0.2)])
    check("dup ID hard-fails", False, "no raise")
except ValueError:
    check("dup ID hard-fails", True)

# --- 3. hard-fail: asymmetric (missing) ID ----------------------------------
try:
    join_paired_records([rec(1, 0, 1, 0.9), rec(2, 0, 2, 0.8)], [rec(1, 0, 1, 0.2)])
    check("asymmetric IDs hard-fail", False, "no raise")
except ValueError:
    check("asymmetric IDs hard-fail", True)

# --- 4. hard-fail: goods mismatch between X and Y ---------------------------
try:
    join_paired_records([rec(1, 0, 1, 0.9)], [rec(1, 0, 2, 0.2)])
    check("goods mismatch hard-fails", False, "no raise")
except ValueError:
    check("goods mismatch hard-fails", True)

# --- 5. point recovery on clean synthetic -----------------------------------
rng = np.random.default_rng(3)
n = 500
vx = np.exp(rng.normal(0, 0.4, n)); vy = np.exp(rng.normal(0, 0.4, n))
lam_true, eta_true = 3.0, 0.8
zx = (1 + lam_true) * vx - vy + eta_true
zy = (1 + lam_true) * vy - vx + eta_true
pno_x = 1 / (1 + np.exp(-zx)); pno_y = 1 / (1 + np.exp(-zy))
fit = fit_lambda_eta(vx, vy, pno_x, pno_y)
check("recovers lambda", abs(fit.lam - lam_true) < 0.05, f"{fit.lam}")
check("recovers eta", abs(fit.eta - eta_true) < 0.05, f"{fit.eta}")
check("jacobian condition finite", np.isfinite(fit.jac_cond), str(fit.jac_cond))
check("multistart stable", fit.multistart_lam_sd < 1e-3, str(fit.multistart_lam_sd))

# --- 6. pair-clustered bootstrap --------------------------------------------
syn_cases = [PairedCase(i, int(i) % 10, int(i) % 10 + 1, (), pno_x[i], pno_y[i]) for i in range(n)]
boot = pair_clustered_bootstrap(syn_cases, vx, vy, n_boot=150, seed=7)
check("bootstrap CI finite", np.isfinite(boot.lam_ci[0]) and np.isfinite(boot.lam_ci[1]), str(boot.lam_ci))
check("bootstrap CI ordered", boot.lam_ci[0] <= boot.lam_ci[1], str(boot.lam_ci))
check("bootstrap CI covers truth", boot.lam_ci[0] <= lam_true <= boot.lam_ci[1], str(boot.lam_ci))
check("bootstrap few failures", boot.n_failed <= 0.1 * 150, str(boot.n_failed))

# --- 7. leave-one-good-out --------------------------------------------------
logo = leave_one_good_out(syn_cases, vx, vy)
check("LOGO returns per-good rows", len(logo["per_good"]) > 0, "empty")
check("LOGO lambda range finite", np.isfinite(logo["lam_range"]), str(logo["lam_range"]))
check("LOGO stable (small range on clean data)", logo["lam_range"] < 0.5, str(logo["lam_range"]))

print(f"\n{_pass} passed, {len(_fail)} failed")
if _fail:
    for f in _fail:
        print("  -", f)
    raise SystemExit(1)
print("ALL ROBUST-INFERENCE TESTS PASSED")
