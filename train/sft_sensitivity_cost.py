#!/usr/bin/env python3
"""GPU-hour / SU estimate for the SFT batch-LR sensitivity experiment.

Every rate is MEASURED from artefacts already in this repository, not guessed;
the source of each is named in the output so the estimate can be audited.

    python train/sft_sensitivity_cost.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import sft_sensitivity_plan as P  # noqa: E402

# ── measured rates ──────────────────────────────────────────────────────────
# Training: the two completed full matched-SFT runs, 30,000 prompts each at
# per_device_batch=1 (results/training_dynamics/sft/sft_training_manifest.json).
# Effective batch is realised by ACCUMULATION, so the number of forward/backward
# passes per prompt is identical at every batch: training cost tracks prompts,
# not optimizer steps.
TRAIN_S_PER_PROMPT_FAST = 4668 / 30000      # seed 2 = 0.1556
TRAIN_S_PER_PROMPT_SLOW = 5422 / 30000      # seed 1 = 0.1807

# Evaluation: one test_goods checkpoint eval = 1,237 batches of 8.
# Observed 1.15 it/s (2026-07-27 pilot) and 1.40 it/s (2026-08-18 base job),
# plus NLS estimation and model load.
EVAL_BATCHES = 1237
EVAL_IT_S_SLOW, EVAL_IT_S_FAST = 1.15, 1.40
EVAL_OVERHEAD_S = 300                        # model load + NLS + PBS start-up

SU_PER_GPU_HOUR = 64                         # myprojects, 2026-08-18
SU_BALANCE = 1933.353                        # myprojects, 2026-08-18


def eval_hours(n: int, fast: bool) -> float:
    rate = EVAL_IT_S_FAST if fast else EVAL_IT_S_SLOW
    return n * (EVAL_BATCHES / rate + EVAL_OVERHEAD_S) / 3600


def train_hours(prompts: int, fast: bool) -> float:
    return prompts * (TRAIN_S_PER_PROMPT_FAST if fast else TRAIN_S_PER_PROMPT_SLOW) / 3600


def scenario(name: str, prompts: int, evals: int, note: str) -> dict:
    lo = train_hours(prompts, True) + eval_hours(evals, True)
    hi = train_hours(prompts, False) + eval_hours(evals, False)
    return {
        "scenario": name, "note": note,
        "training_prompt_exposures": prompts,
        "checkpoint_evaluations": evals,
        "gpu_hours_low": round(lo, 2), "gpu_hours_high": round(hi, 2),
        "su_low": round(lo * SU_PER_GPU_HOUR), "su_high": round(hi * SU_PER_GPU_HOUR),
        "fits_in_balance": bool(hi * SU_PER_GPU_HOUR <= SU_BALANCE),
        "pct_of_balance_high": round(hi * SU_PER_GPU_HOUR / SU_BALANCE * 100, 1),
    }


def main() -> int:
    n_seeds = len(P.SEEDS)
    a_cells = len(P.BATCHES) * n_seeds                      # 12
    b_new = (len(P.PHASE_B_LRS) - 1) * n_seeds              # 9 (1e-6 reused)
    full_cells = n_seeds                                    # 3
    n_ckpt_pilot = len(P.PILOT_CKPT_EXPOSURES)              # 3
    n_ckpt_full = len(P.checkpoint_steps(P.FULL_EXPOSURE, 32))   # 15

    a_prompts = a_cells * P.PILOT_EXPOSURE
    b_prompts = b_new * P.PILOT_EXPOSURE
    f_prompts = full_cells * P.FULL_EXPOSURE

    rows = [
        scenario("Phase A, endpoint evaluation only", a_prompts, a_cells,
                 "12 cells; enough to APPLY the frozen batch-selection rule, "
                 "which needs only endpoint validation cross-entropy"),
        scenario("Phase A, all 3 exposure checkpoints", a_prompts,
                 a_cells * n_ckpt_pilot,
                 "adds the within-run behavioural trajectory for each cell"),
        scenario("Phase A + Phase B, endpoint only", a_prompts + b_prompts,
                 a_cells + b_new,
                 "both selection rules applied; no per-checkpoint trajectories"),
        scenario("Phase A + Phase B, all checkpoints",
                 a_prompts + b_prompts,
                 (a_cells + b_new) * n_ckpt_pilot,
                 "full pilot diagnostics, no 30,016-prompt runs"),
        scenario("Everything incl. full 30,016-prompt runs",
                 a_prompts + b_prompts + f_prompts,
                 (a_cells + b_new) * n_ckpt_pilot + full_cells * n_ckpt_full,
                 "the complete program as specified"),
    ]

    doc = {
        "title": "SFT batch/LR sensitivity — measured GPU-hour and SU estimate",
        "su_rate_su_per_gpu_hour": SU_PER_GPU_HOUR,
        "su_balance_at_estimate": SU_BALANCE,
        "gpu_hours_affordable": round(SU_BALANCE / SU_PER_GPU_HOUR, 2),
        "rate_sources": {
            "training": "results/training_dynamics/sft/sft_training_manifest.json "
                        f"— {TRAIN_S_PER_PROMPT_FAST:.4f}–{TRAIN_S_PER_PROMPT_SLOW:.4f} s/prompt "
                        "at per_device_batch=1; accumulation does not change the "
                        "per-prompt forward/backward cost",
            "evaluation": f"{EVAL_BATCHES} batches at {EVAL_IT_S_SLOW}–{EVAL_IT_S_FAST} it/s "
                          f"plus {EVAL_OVERHEAD_S}s load/NLS overhead, from logs/eval_*.log",
            "su_rate_and_balance": "`myprojects` on aspire2a, 2026-08-18",
        },
        "cost_driver": (
            "Evaluation, not training. Training the entire program costs "
            f"{round(train_hours(a_prompts + b_prompts + f_prompts, False), 1)} GPU-h; "
            "the 108 test_goods checkpoint evaluations cost roughly three times that."
        ),
        "validation_cross_entropy_note": (
            "The frozen selection rule needs completion-only validation "
            "cross-entropy, which is recoverable on CPU from the teacher-forced "
            "Yes/No probabilities the standard evaluation already writes into "
            "loss_aversion_X/Y.json. It therefore needs NO extra GPU pass and is "
            "not costed separately."
        ),
        "scenarios": rows,
    }
    out = P.RESULT_ROOT / "cost_estimate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")

    print(f"SU balance {SU_BALANCE:,.0f}  =  {SU_BALANCE / SU_PER_GPU_HOUR:.1f} GPU-h "
          f"at {SU_PER_GPU_HOUR} SU/GPU-h\n")
    hdr = f"{'scenario':<46} {'GPU-h':>13} {'SU':>13}  {'%bal':>6}  fits"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['scenario']:<46} {r['gpu_hours_low']:>5.1f}-{r['gpu_hours_high']:<7.1f} "
              f"{r['su_low']:>5,}-{r['su_high']:<7,} {r['pct_of_balance_high']:>6.0f}  "
              f"{'YES' if r['fits_in_balance'] else 'NO'}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
