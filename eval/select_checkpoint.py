#!/usr/bin/env python3
"""
Mechanical checkpoint selection under the FROZEN protocol (CHECKPOINT_PROTOCOL.md).

Answers PAPER_READINESS.md #4: the rule is fixed in advance and applied with no
human judgement at selection time. Selection uses VALIDATION data only
(`test_goods`); the frozen final suite must not be opened until after this runs.

Rule (frozen 2026-07-15):
  1. Eligibility: consistency >= CONSISTENCY_FLOOR, and finite non-zero SEs for
     lambda and eta (NLS converged).
  2. Primary: minimise d = sqrt(lambda^2 + eta^2) — the joint distance from the
     rational ideal (0, 0). lambda alone is NOT the criterion: rigidity can be
     shifted onto eta (step 20,000: |lambda|=0.083 smallest, eta=1.263 worst).
  3. Tie-break 1: if d within D_TOL, prefer higher consistency.
  4. Tie-break 2: if consistency within CONS_TOL, prefer the earliest checkpoint.
  5. If nothing is eligible, report it. Do NOT relax thresholds post hoc.

    python eval/select_checkpoint.py --feature baseline --pattern 'Qwen-7B-GRPO-qd-ckpt*'
"""
from __future__ import annotations

import argparse
import csv as _csv
import glob
import json
import math
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from sweep_partition_estimate import load_rows, choice_from_row

# ── FROZEN CONSTANTS — do not tune per run (CHECKPOINT_PROTOCOL.md) ──────────
CONSISTENCY_FLOOR = 0.50
D_TOL = 0.05
CONS_TOL = 0.02
PROTOCOL_FROZEN = "2026-07-15"

# The frozen candidate grid: every 2,000 steps from 2k to 30k (15 checkpoints).
# ANY matched eval dir off this grid — e.g. the diagnostic early checkpoints
# (step 200..1800) added for the training-trajectory plots — is shown for
# visibility but can NEVER be a candidate. This encodes the CHECKPOINT_PROTOCOL
# rule ("Saved adapters outside this grid ... must never enter the candidate
# set, alter a tie-break, or receive OOD evaluation for checkpoint choice") so
# selection is safe regardless of which eval dirs happen to exist on disk.
FROZEN_GRID = frozenset(range(2000, 30001, 2000))


def read_estimates(d: Path) -> dict | None:
    csvs = glob.glob(str(d / "Model_1" / "*NLS_estimation*.csv"))
    if not csvs:
        return None
    rows = list(_csv.DictReader(open(csvs[0])))
    lam, lse = float(rows[0]["Estimate"]), float(rows[0]["Std. Err."])
    eta, ese = float(rows[1]["Estimate"]), float(rows[1]["Std. Err."])
    return {"lambda": lam, "lambda_se": lse, "eta": eta, "eta_se": ese}


def behavior(d: Path) -> dict | None:
    try:
        xr = load_rows(d / "loss_aversion_X.json")
        yr = load_rows(d / "loss_aversion_Y.json")
    except Exception:
        return None
    common = sorted(set(xr) & set(yr))
    if not common:
        return None
    c = k = t = 0
    for cid in common:
        a, b = choice_from_row(xr[cid]), choice_from_row(yr[cid])
        if a != b:
            c += 1
        elif a == "No":
            k += 1
        else:
            t += 1
    n = len(common)
    return {"n": n, "consistency": c / n, "keep_both_rate": k / n, "trade_both_rate": t / n}


def eligible(r: dict) -> tuple[bool, str]:
    if r.get("consistency") is None:
        return False, "no behavioral data"
    if r["consistency"] < CONSISTENCY_FLOOR:
        return False, f"consistency {r['consistency']:.3f} < floor {CONSISTENCY_FLOOR}"
    for k in ("lambda_se", "eta_se"):
        v = r.get(k)
        if v is None or not math.isfinite(v) or v <= 0:
            return False, f"{k} not finite/positive (NLS did not converge)"
    return True, ""


def build_row(step: int, model_name: str, est: dict, beh: dict | None) -> dict:
    """Assemble one candidate row and classify it. Off-grid steps (e.g. the
    diagnostic early checkpoints) are marked ineligible so they can never be
    selected; on-grid steps go through the frozen eligibility rule."""
    r = {"step": step, "model_name": model_name, **est, **(beh or {})}
    r["d"] = math.hypot(r["lambda"], r["eta"])
    r["_on_grid"] = step in FROZEN_GRID
    if not r["_on_grid"]:
        r["_eligible"], r["_why"] = False, "off frozen 2k-30k grid (diagnostic only)"
    else:
        ok, why = eligible(r)
        r["_eligible"], r["_why"] = ok, why
    return r


def grid_status(rows: list[dict]) -> tuple[list[int], list[int]]:
    """(present_on_grid, missing_from_grid) against the frozen 15-step grid."""
    present = sorted({r["step"] for r in rows if r.get("_on_grid")})
    missing = sorted(FROZEN_GRID - set(present))
    return present, missing


def select(rows: list[dict]) -> tuple[dict | None, str]:
    # HARD REQUIREMENT: the exact 15-step grid must be complete. A partial grid
    # is a hard failure, not a "pick the best of what ran" — selecting on a
    # subset silently reintroduces the degree of freedom the protocol removes.
    _, missing = grid_status(rows)
    if missing:
        return None, (f"INCOMPLETE GRID: missing frozen checkpoint(s) {missing}. The "
                      f"frozen protocol requires all 15 of {sorted(FROZEN_GRID)} before "
                      "any selection. Evaluate the missing step(s); do NOT select on a "
                      "partial grid.")
    cand = [r for r in rows if r["_eligible"]]
    if not cand:
        return None, "no eligible checkpoint (do NOT relax thresholds post hoc)"
    best = min(cand, key=lambda r: r["d"])
    # tie-break 1: higher consistency among d-ties
    ties = [r for r in cand if abs(r["d"] - best["d"]) < D_TOL]
    if len(ties) > 1:
        best = max(ties, key=lambda r: r["consistency"])
        # tie-break 2: earliest among consistency-ties
        ties2 = [r for r in ties if abs(r["consistency"] - best["consistency"]) < CONS_TOL]
        if len(ties2) > 1:
            best = min(ties2, key=lambda r: r["step"])
    return best, f"min d among {[r['step'] for r in cand]}"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", default="baseline")
    ap.add_argument("--pattern", required=True, help="e.g. 'Qwen-7B-GRPO-qd-ckpt*'")
    ap.add_argument("--run_name", default=None, help="label for the manifest")
    args = ap.parse_args()

    rows = []
    for d in glob.glob(str(PROJECT_ROOT / args.feature / args.pattern)):
        d = Path(d)
        m = re.search(r"(\d+)$", d.name)
        if not m:
            continue
        est = read_estimates(d)
        beh = behavior(d)
        if est is None:
            continue
        rows.append(build_row(int(m.group(1)), d.name, est, beh))
    rows.sort(key=lambda r: r["step"])
    if not rows:
        print(f"No checkpoints matched {args.feature}/{args.pattern}")
        return

    print(f"\nFROZEN checkpoint-selection protocol (frozen {PROTOCOL_FROZEN})")
    print(f"  eligibility: consistency >= {CONSISTENCY_FLOOR}, finite non-zero SEs")
    print(f"  primary: min d = sqrt(lambda^2 + eta^2);  tie-breaks: consistency (d within "
          f"{D_TOL}), then earliest (consistency within {CONS_TOL})")
    print("  SELECTION USES VALIDATION DATA ONLY — frozen final suite stays closed.\n")
    hdr = f"{'step':>7} | {'lambda':>8} | {'eta':>8} | {'consist':>7} | {'d':>6} | eligible"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if not r["_on_grid"]:
            flag = "diagnostic (off-grid, not a candidate)"
        else:
            flag = "yes" if r["_eligible"] else f"NO ({r['_why']})"
        print(f"{r['step']:>7} | {r['lambda']:>+8.4f} | {r['eta']:>+8.4f} | "
              f"{r.get('consistency', float('nan')):>7.3f} | {r['d']:>6.3f} | {flag}")
    off_grid = [r["step"] for r in rows if not r["_on_grid"]]
    if off_grid:
        print(f"\n  ({len(off_grid)} off-grid diagnostic checkpoint(s) shown but excluded "
              f"from selection: {off_grid})")

    # Hard-fail on an incomplete frozen grid (exit 2, no manifest written).
    present, missing = grid_status(rows)
    print(f"\n  frozen grid: {len(present)}/{len(FROZEN_GRID)} present"
          + (f"; MISSING {missing}" if missing else " (complete)"))
    if missing:
        print(f"\nHARD FAILURE: incomplete grid, missing {missing}. Evaluate the missing "
              "step(s) before selecting. No selection written.")
        sys.exit(2)

    best, reason = select(rows)
    if best is None:
        print(f"\nNO SELECTION: {reason}")
        sys.exit(1)
    print(f"\n{'='*66}")
    print(f"SELECTED: step {best['step']}  (d = {best['d']:.3f})")
    print(f"  lambda = {best['lambda']:+.4f} (SE {best['lambda_se']:.4f})")
    print(f"  eta    = {best['eta']:+.4f} (SE {best['eta_se']:.4f})   <-- report jointly with lambda")
    print(f"  consistency = {best['consistency']:.3f}, keep-both = {best['keep_both_rate']:.3f}, "
          f"trade-both = {best['trade_both_rate']:.3f}")
    print(f"  rule: {reason}")
    print("  NOTE: this lambda is a VALIDATION estimate, not final-test performance.")
    print("=" * 66)

    run = args.run_name or args.pattern.replace("*", "").rstrip("-_")
    out = PROJECT_ROOT / "results" / "checkpoint_selection" / f"{run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "run": run, "feature": args.feature, "pattern": args.pattern,
        "protocol_frozen": PROTOCOL_FROZEN, "code_commit": git_commit(),
        "rule": {"consistency_floor": CONSISTENCY_FLOOR, "primary": "min sqrt(lambda^2+eta^2)",
                 "d_tol": D_TOL, "cons_tol": CONS_TOL},
        "selection_data": "validation (test_goods); frozen final suite not opened",
        "selected_step": best["step"], "selected": best,
        "grid": rows,
    }, open(out, "w"), indent=2, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
