#!/usr/bin/env python3
"""Guard tests for SFT grid verification and the verified-grid plotting path.

The regression that motivates this file: `Model_1/` is created by
estimate_qwen_checkpoint.py BEFORE the NLS CSV is written, so a directory-
existence test reports a checkpoint as complete while its estimation is still
running. On 2026-08-06 seed-1 step 24,000 was in exactly that state. Anything
that decides "the grid is ready" must look past the directory.

    python eval/test_verify_sft_grid.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

import verify_sft_grid as vsg  # noqa: E402

_pass, _fail = 0, []


def check(cond, msg):
    global _pass
    if cond:
        _pass += 1
    else:
        _fail.append(msg)


NLS_CSV = (
    "Parameter,Estimate,Std. Err.,Variance\n"
    "lambda,0.12345678,0.01400000,0.00019600\n"
    "eta,-0.20000000,0.03300000,0.00108900\n"
    "alpha_2,-0.30000000,0.02000000,0.00040000\n"
)


def write_rows(path: Path, n: int, first_id: int = 61) -> None:
    rows = [{"case_id": first_id + i, "X_num": 0, "Y_num": 1, "attr": [1, 2, 1, 0],
             "Yes / No prob": [0.3, 0.7], "output": "No"} for i in range(n)]
    path.write_text(json.dumps(rows))


def make_cell(root: Path, seed: int, step: int, *, n: int = vsg.EXPECTED_N,
              with_csv: bool = True, with_model_1: bool = True,
              bad_probs: bool = False) -> None:
    """One synthetic grid cell: adapter + evaluation output."""
    ad = root / f"checkpoints/sft_qwen_delta_seed{seed}/checkpoint-{step}"
    ad.mkdir(parents=True, exist_ok=True)
    (ad / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 16, "lora_alpha": 32,
        "base_model_name_or_path": "models/Qwen2.5-7B-Instruct"}))
    (ad / "adapter_model.safetensors").write_bytes(b"w" * 64)

    d = root / "baselines" / f"Qwen-7B-SFT-qd-seed{seed}-ckpt{step}"
    d.mkdir(parents=True, exist_ok=True)
    write_rows(d / "loss_aversion_X.json", n)
    write_rows(d / "loss_aversion_Y.json", n)
    if bad_probs:
        rows = json.loads((d / "loss_aversion_X.json").read_text())
        rows[0]["Yes / No prob"] = ["nan", "nan"]
        (d / "loss_aversion_X.json").write_text(json.dumps(rows))
    if with_model_1:
        m = d / "Model_1"
        m.mkdir(exist_ok=True)
        # The PNGs land before the CSV — this is what makes the directory test lie.
        (m / f"Qwen-7B-SFT-qd-seed{seed}-ckpt{step}-alpha-estimation-Model-A.png"
         ).write_bytes(b"\x89PNG")
        if with_csv:
            (m / f"Qwen-7B-SFT-qd-seed{seed}-ckpt{step}_NLS_estimation_T1(Model A).csv"
             ).write_text(NLS_CSV)


def with_root(fn):
    """Run fn against a synthetic PROJECT_ROOT, restoring the real one after."""
    real = vsg.PROJECT_ROOT
    with tempfile.TemporaryDirectory() as td:
        vsg.PROJECT_ROOT = Path(td)
        try:
            fn(Path(td))
        finally:
            vsg.PROJECT_ROOT = real


# ── the core regression ──────────────────────────────────────────────────────
def t_model_1_without_csv_is_not_complete(root: Path):
    make_cell(root, 1, 2000, with_csv=False)
    ev = vsg.verify_evaluation(1, 2000)
    check(ev["model_1_dir_exists"] is True,
          "fixture should present a populated Model_1 directory")
    check(any("estimation incomplete" in p for p in ev["problems"]),
          "Model_1 without an NLS CSV must fail verification")


def t_complete_cell_verifies(root: Path):
    make_cell(root, 1, 2000)
    ev = vsg.verify_evaluation(1, 2000)
    check(ev["problems"] == [], f"complete cell should verify, got {ev['problems']}")
    check(abs(ev["lambda"] - 0.12345678) < 1e-9, "lambda parsed from CSV")
    check(abs(ev["eta"] + 0.2) < 1e-9, "eta parsed from CSV")
    check(ev["lambda_se"] > 0 and ev["eta_se"] > 0, "SEs must be positive")
    check(ev["n_cases"] == vsg.EXPECTED_N, "N must be 9,890 paired cases")
    check(ev["parse_failures"] == 0, "no parse failures expected")
    check("T1" in Path(ev["nls_csv"]).name, "CSV must record link scale T=1")


def t_partial_rows_fail(root: Path):
    make_cell(root, 1, 4000, n=3000)
    ev = vsg.verify_evaluation(1, 4000)
    check(any("9890" in p for p in ev["problems"]),
          "a partially-written evaluation must fail on N")


def t_parse_failures_detected(root: Path):
    make_cell(root, 1, 6000, bad_probs=True)
    ev = vsg.verify_evaluation(1, 6000)
    check(ev["parse_failures"] >= 1, "non-finite probabilities must be counted")
    check(any("unparseable" in p for p in ev["problems"]),
          "parse failures must be reported as problems")


def t_duplicate_roots_fail(root: Path):
    make_cell(root, 1, 8000)
    dup = root / "baseline" / "Qwen-7B-SFT-qd-seed1-ckpt8000"
    dup.mkdir(parents=True, exist_ok=True)
    ev = vsg.verify_evaluation(1, 8000)
    check(any("DUPLICATE" in p for p in ev["problems"]),
          "the same seed/step under two roots must hard-fail")


def t_missing_adapter_fails(root: Path):
    d = root / "baselines" / "Qwen-7B-SFT-qd-seed2-ckpt2000"
    d.mkdir(parents=True, exist_ok=True)
    write_rows(d / "loss_aversion_X.json", vsg.EXPECTED_N)
    write_rows(d / "loss_aversion_Y.json", vsg.EXPECTED_N)
    ad = vsg.verify_adapter(2, 2000)
    check(any("adapter directory absent" in p for p in ad["problems"]),
          "an evaluation without its adapter cannot prove identity")


def t_partial_grid_hard_fails(root: Path):
    for step in vsg.GRID_STEPS:
        make_cell(root, 1, step)
    make_cell(root, 2, 2000)          # seed 2 only 1/15
    man = vsg.verify_grid()
    check(man["complete"] is False, "a 16/30 grid must not be complete")
    check(man["identity_verified"] is False, "partial grid is not identity-verified")
    check(man["n_verified"] == 16, f"expected 16 verified, got {man['n_verified']}")
    check(len(man["problems"]) > 0, "problems must be reported for a partial grid")


def t_full_grid_verifies(root: Path):
    for seed in vsg.SEEDS:
        for step in vsg.GRID_STEPS:
            make_cell(root, seed, step)
    man = vsg.verify_grid()
    check(man["complete"] is True, f"30/30 grid should be complete: {man['problems'][:3]}")
    check(man["identity_verified"] is True, "30/30 grid should be identity-verified")
    check(man["n_verified"] == 30, f"expected 30 verified, got {man['n_verified']}")
    check(man["protocol"]["expected_n_per_checkpoint"] == 9890, "N recorded")
    check("baseline" in man["protocol"]["comparator"],
          "manifest must name the plain-prompt matched comparator")


# ── the plotting path ────────────────────────────────────────────────────────
def t_plotting_refuses_unverified_grid():
    import plot_sft_training_dynamics as P
    # complete=True but identity_verified=False must NOT produce a full trajectory.
    df, status = P.build_behavioural({"complete": True, "identity_verified": False})
    check(status["full_grid_identity_verified"] is False,
          "status must record that identity verification did not pass")
    check(df is None or str(df["run_role"].iloc[0]) != "full",
          "an unverified grid must never yield a full-run trajectory")


def t_plotting_accepts_verified_grid():
    import plot_sft_training_dynamics as P
    rows = []
    for seed in (1, 2):
        for step in range(2000, 30001, 2000):
            rows.append({
                "seed": seed, "step": step,
                "model_name": f"Qwen-7B-SFT-qd-seed{seed}-ckpt{step}",
                "adapter_path": f"checkpoints/sft_qwen_delta_seed{seed}/checkpoint-{step}",
                "adapter_weight_sha256": "0" * 64,
                "eval_dir": f"baselines/Qwen-7B-SFT-qd-seed{seed}-ckpt{step}",
                "nls_csv": "x.csv", "nls_csv_sha256": "1" * 64,
                "lambda": 0.1, "lambda_se": 0.014, "eta": -0.2, "eta_se": 0.033,
                "d": 0.2236, "consistency": 0.8, "keep_both": 0.1,
                "trade_both": 0.1, "W": 0.9, "n_cases": 9890, "parse_failures": 0,
            })
    scan = {"complete": True, "identity_verified": True, "verified_rows": rows,
            "required": {"n_evaluations_required": 30},
            "verification_manifest": "results/sft_grid_verification.json"}
    df, status = P.build_behavioural(scan)
    check(df is not None, "a verified grid must yield a trajectory")
    if df is None:
        return
    check(len(df) == 30, f"expected 30 rows, got {len(df)}")
    check(str(df["run_role"].iloc[0]) == "full", "rows must be marked full-run")
    check(sorted(df["seed"].unique().tolist()) == [1, 2], "both seeds present")
    check(status["full_trajectory"]["duplicate_checkpoint_rows"] == 0,
          "no duplicate seed/step rows")
    check(status["base_comparator"]["treatment"].startswith("baseline"),
          "comparator must be the plain-prompt matched base")
    check("winner" in status["method_comparison"],
          "status must state that no method winner is declared")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fig.png"
        P.plot_behavioural(df, p, status)
        check(p.exists() and p.stat().st_size > 10000,
              "six-panel figure should render to a non-trivial PNG")


def t_partial_verified_rows_refused():
    import plot_sft_training_dynamics as P
    scan = {"complete": True, "identity_verified": True,
            "verified_rows": [], "required": {"n_evaluations_required": 30}}
    df, status = P.build_behavioural(scan)
    check(df is None or str(df["run_role"].iloc[0]) != "full",
          "row count below the required grid must refuse the full path")


def main() -> int:
    for fn in (t_model_1_without_csv_is_not_complete, t_complete_cell_verifies,
               t_partial_rows_fail, t_parse_failures_detected,
               t_duplicate_roots_fail, t_missing_adapter_fails,
               t_partial_grid_hard_fails, t_full_grid_verifies):
        with_root(fn)
        print(f"{fn.__name__}: done")
    for fn in (t_plotting_refuses_unverified_grid, t_plotting_accepts_verified_grid,
               t_partial_verified_rows_refused):
        fn()
        print(f"{fn.__name__}: done")

    if _fail:
        print(f"\nFAILED {len(_fail)} / {_pass + len(_fail)} checks:")
        for f in _fail:
            print(f"  ✗ {f}")
        return 1
    print(f"\nOK — {_pass} checks passed (SFT grid verification + plotting guard).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
