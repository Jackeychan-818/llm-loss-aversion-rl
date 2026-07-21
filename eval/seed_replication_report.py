#!/usr/bin/env python3
"""
Seed replication report — mechanically applies the FROZEN pre-registration
(PRE_REGISTRATION.md) to the confirmatory seeds and prints the verdict.

This exists so the final read-out cannot drift from what was pre-registered: the
thresholds below are copied from the frozen doc and the verdict is computed, not
judged. It reads only artifacts produced by the eval pipeline; anything missing
is reported as PENDING rather than guessed.

Per seed it checks:
  S1  eligible checkpoint exists              (selection produced a winner)
  S2  |lambda_OOD| <= 0.5                      (OOD-50 suite, selected checkpoint)
  S3  consistency >= 0.50 and keep-both <= 0.50 and trade-both <= 0.50   (OOD)
  S5  GSM8K paired 95% CI lower bound >= -3 pp (vs exact local base 86.88%)
  S4  eta reported jointly, NOT gated
A seed PASSES if S1 & S2 & S3 & S5. Bar: 2/2 new seeds (SEED=1,2). seed=42 is
exploratory and excluded from the denominator.

    module load pytorch/...; source .../venv/bin/activate
    python eval/seed_replication_report.py
"""
from __future__ import annotations

import csv as _csv
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from sweep_partition_estimate import load_rows, choice_from_row  # noqa: E402

# ── Frozen thresholds (copied from PRE_REGISTRATION.md; do not tune here) ─────
CONFIRMATORY_SEEDS = [1, 2]
S2_LAMBDA_OOD_MAX = 0.5
S3_CONSISTENCY_MIN = 0.50
S3_KEEP_MAX = 0.50
S3_TRADE_MAX = 0.50
S5_CI_LOWER_MIN_PP = -3.0
GSM8K_BASE_ACC = 0.8688  # exact local base, for reference in the report


def read_nls(csv_path: Path):
    if not csv_path.exists():
        return None
    rows = list(_csv.DictReader(open(csv_path)))
    return {"lambda": float(rows[0]["Estimate"]), "lambda_se": float(rows[0]["Std. Err."]),
            "eta": float(rows[1]["Estimate"]), "eta_se": float(rows[1]["Std. Err."])}


def behavior(eval_dir: Path):
    xf, yf = eval_dir / "loss_aversion_X.json", eval_dir / "loss_aversion_Y.json"
    if not (xf.exists() and yf.exists()):
        return None
    xr, yr = load_rows(xf), load_rows(yf)
    common = sorted(set(xr) & set(yr))
    if not common:
        return None
    c = k = t = 0
    for cid in common:
        a, b = choice_from_row(xr[cid]), choice_from_row(yr[cid])
        if a != b: c += 1
        elif a == "No": k += 1
        else: t += 1
    n = len(common)
    return {"n": n, "consistency": c / n, "keep_both_rate": k / n, "trade_both_rate": t / n}


def find_ood_dir(seed: int, sel: int) -> Path:
    # matches submit_eval_ood_seed.pbs naming
    name = f"Qwen-7B-GRPO-qd-seed{seed}-ckpt{sel}-OOD50"
    return PROJECT_ROOT / "ood" / name


def gsm8k_ci_lower_pp(seed: int):
    comp = PROJECT_ROOT / f"results/gsm8k_seed{seed}/comparison.json"
    if not comp.exists():
        return None, "no comparison.json"
    d = json.load(open(comp))
    pc = d.get("paired_change", {})
    ci = pc.get("paired_bootstrap_95_ci_percentage_points")
    if not ci:
        return None, "no CI in comparison.json"
    return ci[0], f"delta={pc.get('percentage_point_delta'):+.2f}pp CI[{ci[0]:+.2f},{ci[1]:+.2f}]"


def assess_seed(seed: int) -> dict:
    out = {"seed": seed, "checks": {}, "values": {}}
    # selection
    selp = PROJECT_ROOT / f"results/checkpoint_selection/qwen_delta_seed{seed}.json"
    if not selp.exists():
        out["status"] = "PENDING: selection not run"
        out["checks"]["S1"] = "pending"
        return out
    sel = json.load(open(selp))
    grid_steps = sorted(r["step"] for r in sel.get("grid", []))
    expected = list(range(2000, 30001, 2000))
    out["values"]["grid_complete"] = (grid_steps == expected)
    step = sel.get("selected_step")
    out["values"]["selected_step"] = step
    out["checks"]["S1"] = "PASS" if step is not None else "FAIL (no eligible checkpoint)"
    if step is None:
        out["status"] = "FAIL: no eligible checkpoint (S1)"
        return out
    if not out["values"]["grid_complete"]:
        out["status"] = f"PENDING: grid incomplete {grid_steps}"
        return out

    # OOD (S2, S3) at the selected checkpoint
    ood_dir = find_ood_dir(seed, step)
    ood_csv = ood_dir / "Model_1" / f"Qwen-7B-GRPO-qd-seed{seed}-ckpt{step}-OOD50_NLS_estimation_T1(Model A).csv"
    nls = read_nls(ood_csv)
    beh = behavior(ood_dir)
    if nls is None or beh is None:
        out["checks"]["S2"] = out["checks"]["S3"] = "pending"
        out["status"] = "PENDING: OOD eval not done"
    else:
        out["values"].update({"lambda_OOD": nls["lambda"], "lambda_OOD_se": nls["lambda_se"],
                               "eta_OOD": nls["eta"], **beh})
        s2 = abs(nls["lambda"]) <= S2_LAMBDA_OOD_MAX
        s3 = (beh["consistency"] >= S3_CONSISTENCY_MIN and beh["keep_both_rate"] <= S3_KEEP_MAX
              and beh["trade_both_rate"] <= S3_TRADE_MAX)
        out["checks"]["S2"] = f"{'PASS' if s2 else 'FAIL'} (|lambda_OOD|={abs(nls['lambda']):.3f} vs <= {S2_LAMBDA_OOD_MAX})"
        out["checks"]["S3"] = f"{'PASS' if s3 else 'FAIL'} (consist={beh['consistency']:.2f}, keep={beh['keep_both_rate']:.2f}, trade={beh['trade_both_rate']:.2f})"

    # GSM8K (S5)
    lo, msg = gsm8k_ci_lower_pp(seed)
    if lo is None:
        out["checks"]["S5"] = "pending"
        out["values"]["gsm8k"] = msg
    else:
        s5 = lo >= S5_CI_LOWER_MIN_PP
        out["checks"]["S5"] = f"{'PASS' if s5 else 'FAIL'} (CI lower {lo:+.2f}pp vs >= {S5_CI_LOWER_MIN_PP})"
        out["values"]["gsm8k"] = msg

    decided = [c for c in (out["checks"].get(k) for k in ("S1", "S2", "S3", "S5"))]
    if any(c == "pending" for c in decided):
        out["status"] = "PENDING"
    elif all(str(c).startswith("PASS") for c in decided):
        out["status"] = "PASS"
    else:
        out["status"] = "FAIL"
    return out


def main():
    seeds = [assess_seed(s) for s in CONFIRMATORY_SEEDS]
    print("=" * 74)
    print("SEED REPLICATION — frozen pre-registration verdict (PRE_REGISTRATION.md)")
    print(f"Confirmatory seeds: {CONFIRMATORY_SEEDS}  (seed 42 = exploratory, excluded)")
    print("=" * 74)
    for s in seeds:
        print(f"\nseed {s['seed']}: {s['status']}")
        for k in ("S1", "S2", "S3", "S5"):
            if k in s["checks"]:
                print(f"    {k}: {s['checks'][k]}")
        v = s["values"]
        if "lambda_OOD" in v:
            print(f"    values: sel_step={v.get('selected_step')}, lambda_OOD={v['lambda_OOD']:+.3f}, "
                  f"eta_OOD={v['eta_OOD']:+.3f} (reported, not gated), {v.get('gsm8k','')}")

    statuses = [s["status"] for s in seeds]
    n_pass = sum(st == "PASS" for st in statuses)
    n_pending = sum(st.startswith("PENDING") for st in statuses)
    print("\n" + "=" * 74)
    if n_pending:
        print(f"VERDICT: PENDING — {n_pending}/{len(seeds)} seeds not fully evaluated yet.")
    elif n_pass == 2:
        print("VERDICT: 2/2 confirmatory seeds PASS → method replicates. "
              "Report seed 42 as supporting exploratory evidence.")
    elif n_pass == 1:
        print("VERDICT: 1/2 — seeds DISAGREE. Per pre-registration: run a THIRD new "
              "seed and require 2/3 fresh (do NOT use seed 42 to break the tie).")
    else:
        print("VERDICT: 0/2 PASS → report that the method did not replicate.")
    print("=" * 74)
    print("Reminder: report EVERY seed's numbers, never the best alone. eta is "
          "reported jointly with lambda but is not a gate.")

    outp = PROJECT_ROOT / "results" / "seed_replication_report.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"seeds": seeds, "n_pass": n_pass, "n_pending": n_pending}, open(outp, "w"), indent=2)
    print(f"\nwrote {outp.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
