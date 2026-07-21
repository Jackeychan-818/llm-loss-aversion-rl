#!/usr/bin/env python3
"""
Tests for the FROZEN checkpoint-selection guard (select_checkpoint.py).

Runs on a login node under the venv (imports the estimator's helper module):

    module load pytorch/...; source .../venv/bin/activate
    python eval/test_select_checkpoint.py

Covers the four guarantees required by CHECKPOINT_PROTOCOL.md:
  1. an off-grid diagnostic checkpoint (step 600) can never be selected;
  2. a missing frozen checkpoint is a HARD FAILURE, not "best of what ran";
  3. the complete 15-step grid permits selection (min-d is applied);
  4. the real committed seed-1 selection is still step 2000 (no regression).
"""
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # eval/
import select_checkpoint as sc  # noqa: E402


def est(lam, eta, lam_se=0.02, eta_se=0.02):
    return {"lambda": lam, "lambda_se": lam_se, "eta": eta, "eta_se": eta_se}


def beh(cons=0.70, keep=0.20, trade=0.10, n=100):
    return {"n": n, "consistency": cons, "keep_both_rate": keep, "trade_both_rate": trade}


def full_grid_rows(lam=0.30, eta=0.40):
    return [sc.build_row(s, f"m{s}", est(lam, eta), beh()) for s in sorted(sc.FROZEN_GRID)]


def test_frozen_grid_is_the_15_step_2k_to_30k():
    assert sorted(sc.FROZEN_GRID) == list(range(2000, 30001, 2000))
    assert len(sc.FROZEN_GRID) == 15
    print("test_frozen_grid_is_the_15_step_2k_to_30k: OK")


def test_step600_off_grid_cannot_be_selected():
    # step 600 has the BEST possible metrics; it must still be excluded.
    rows = full_grid_rows() + [sc.build_row(600, "m600-diag", est(0.0, 0.0), beh(cons=0.99))]
    r600 = next(r for r in rows if r["step"] == 600)
    assert r600["_on_grid"] is False, "step 600 wrongly marked on-grid"
    assert r600["_eligible"] is False, "step 600 wrongly eligible"
    best, reason = sc.select(rows)
    assert best is not None, reason
    assert best["step"] != 600, "off-grid step 600 was selected"
    assert best["step"] in sc.FROZEN_GRID
    print("test_step600_off_grid_cannot_be_selected: OK")


def test_missing_frozen_checkpoint_hard_fails():
    rows = [r for r in full_grid_rows() if r["step"] != 16000]  # drop one on-grid step
    present, missing = sc.grid_status(rows)
    assert missing == [16000], missing
    best, reason = sc.select(rows)
    assert best is None, "selected on an incomplete grid"
    assert "INCOMPLETE GRID" in reason, reason
    print("test_missing_frozen_checkpoint_hard_fails: OK")


def test_missing_stays_hard_fail_even_with_off_grid_diagnostics():
    # adding off-grid diagnostics must NOT paper over a missing frozen step.
    rows = [r for r in full_grid_rows() if r["step"] != 4000]
    rows += [sc.build_row(s, f"m{s}-diag", est(0.0, 0.0), beh()) for s in (600, 800, 1000)]
    best, reason = sc.select(rows)
    assert best is None and "INCOMPLETE GRID" in reason, reason
    print("test_missing_stays_hard_fail_even_with_off_grid_diagnostics: OK")


def test_complete_grid_permits_selection_and_applies_min_d():
    rows = []
    for s in sorted(sc.FROZEN_GRID):
        lam, eta = (0.02, 0.03) if s == 8000 else (0.50, 0.60)  # 8000 = clear min d
        rows.append(sc.build_row(s, f"m{s}", est(lam, eta), beh()))
    present, missing = sc.grid_status(rows)
    assert missing == [], missing
    best, reason = sc.select(rows)
    assert best is not None and best["step"] == 8000, (best, reason)
    print("test_complete_grid_permits_selection_and_applies_min_d: OK")


def test_ineligible_low_consistency_excluded():
    rows = []
    for s in sorted(sc.FROZEN_GRID):
        c = 0.40 if s == 2000 else 0.70  # 2000 below the 0.50 floor
        rows.append(sc.build_row(s, f"m{s}", est(0.10, 0.20), beh(cons=c)))
    r2000 = next(r for r in rows if r["step"] == 2000)
    assert r2000["_eligible"] is False
    best, _ = sc.select(rows)
    assert best is not None and best["step"] != 2000
    print("test_ineligible_low_consistency_excluded: OK")


def test_real_seed1_selection_is_step_2000():
    rows = []
    for d in glob.glob(str(sc.PROJECT_ROOT / "baseline" / "Qwen-7B-GRPO-qd-seed1-ckpt*")):
        d = Path(d)
        m = re.search(r"(\d+)$", d.name)
        if not m:
            continue
        e = sc.read_estimates(d)
        b = sc.behavior(d)
        if e is None:
            continue
        rows.append(sc.build_row(int(m.group(1)), d.name, e, b))
    if not rows:
        print("test_real_seed1_selection_is_step_2000: SKIP (no seed1 eval dirs on this host)")
        return
    present, missing = sc.grid_status(rows)
    assert missing == [], f"seed1 grid incomplete, missing {missing}"
    best, reason = sc.select(rows)
    assert best is not None, reason
    assert best["step"] == 2000, f"expected step 2000, got {best['step']} ({reason})"
    print("test_real_seed1_selection_is_step_2000: OK")


if __name__ == "__main__":
    test_frozen_grid_is_the_15_step_2k_to_30k()
    test_step600_off_grid_cannot_be_selected()
    test_missing_frozen_checkpoint_hard_fails()
    test_missing_stays_hard_fail_even_with_off_grid_diagnostics()
    test_complete_grid_permits_selection_and_applies_min_d()
    test_ineligible_low_consistency_excluded()
    test_real_seed1_selection_is_step_2000()
    print("\nAll select_checkpoint guard tests passed.")
