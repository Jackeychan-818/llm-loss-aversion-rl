#!/usr/bin/env python3
"""SFT TRAINING METRICS for the matched supervised fine-tuning baselines.

Companion to eval/plot_training_dynamics.py (GRPO). It does NOT touch, read or
rewrite any GRPO output. The title/caption wording is deliberately "SFT training
metrics", never bare "training dynamics", so the two figure families cannot be
confused in a slide deck or a paper draft.

WHAT SFT LOGS NATIVELY
    completion-only cross-entropy loss, gradient norm, learning rate, training
    progress (step / epoch), and end-of-run runtime + throughput.

WHAT SFT DOES NOT HAVE
    task reward, reward std, zero-task-reward-advantage fraction, GRPO policy
    loss, KL penalty, group-relative advantages. No artificial SFT stand-ins for
    those are constructed here.

THE OBJECTIVE
    L_SFT = - sum over answer tokens of log p_theta(target token | prompt,
            previous target tokens)
    Only the assistant Yes/No completion tokens contribute; prompt tokens are
    masked with label -100. Falling SFT loss means the model assigns more
    probability to the prescribed Yes/No completion on TRAINING prompts. It does
    not measure lambda or eta, ownership invariance, OOD generalization, framing
    robustness, preference preservation, or capability retention. SFT
    cross-entropy and the GRPO policy loss optimize different objectives, so
    their numerical values must never be compared directly.

TWO MODES, DELIBERATELY SEPARATED
    1. --refresh : raw NSCC trainer_state.json (+ run logs) -> canonical compact
                   CSV/JSON snapshot under results/training_dynamics/sft/.
    2. (default) : tracked snapshot -> figures, summary, README.
    3. --check   : verify the tracked figures/tables agree with the tracked
                   snapshot, re-rendering into a temporary directory.

    Rendering and checking the SFT curves is repository-reproducible;
    regenerating the underlying logged rows requires the raw NSCC
    trainer_state.json sources.

PILOT vs FULL
    The pilot used a cosine schedule ending at 6,000. The full run used a cosine
    schedule ending at 30,000. Their step-6,000 weights are not interchangeable.
    The two histories are never merged or overlaid as one trajectory.

Run from the repository root:

    python eval/plot_sft_training_dynamics.py --refresh
    python eval/plot_sft_training_dynamics.py
    python eval/plot_sft_training_dynamics.py --check
"""
from __future__ import annotations

import argparse
import csv as csv_module
import filecmp
import glob
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PARSER_VERSION = "1.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "training_dynamics" / "sft"
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
# Same choice/row helpers the frozen selector uses, so the base comparator and
# the grid rows are computed by one code path rather than two.
from sweep_partition_estimate import (  # noqa: E402
    choice_from_row as choice_helper, load_rows as load_rows_helper)

# ── run registry ──────────────────────────────────────────────────────────────
# run_id -> identity. `state` / `runlog` are RAW NSCC sources, needed only by
# --refresh; they are gitignored and absent from a clean `git archive`.
RUNS: dict[str, dict] = {
    "sft_full_seed1": {
        "run_role": "full",
        "seed": 1,
        "expected_max_steps": 30000,
        "state": "checkpoints/sft_qwen_delta_seed1/checkpoint-30000/trainer_state.json",
        "runlog": "logs/train_sft_qd_seed1_run.log",
        "manifest": "checkpoints/sft_qwen_delta_seed1/sft_dataset_manifest.json",
        "ckpt_dir": "checkpoints/sft_qwen_delta_seed1",
        "label": "full seed 1 (cosine -> 30k)",
        "color": "#1f77b4",
    },
    "sft_full_seed2": {
        "run_role": "full",
        "seed": 2,
        "expected_max_steps": 30000,
        "state": "checkpoints/sft_qwen_delta_seed2/checkpoint-30000/trainer_state.json",
        "runlog": "logs/train_sft_qd_seed2_run.log",
        "manifest": "checkpoints/sft_qwen_delta_seed2/sft_dataset_manifest.json",
        "ckpt_dir": "checkpoints/sft_qwen_delta_seed2",
        "label": "full seed 2 (cosine -> 30k)",
        "color": "#d62728",
    },
    "sft_pilot6k_seed1": {
        "run_role": "pilot",
        "seed": 1,
        "expected_max_steps": 6000,
        # The pilot originally trained into checkpoints/sft_qwen_delta_seed1 and
        # was renamed to *_pilot6k before the full run reused the old path. Its
        # dataset manifest therefore still records the pre-rename output_dir.
        "state": "checkpoints/sft_qwen_delta_seed1_pilot6k/checkpoint-6000/trainer_state.json",
        "runlog": "logs/train_sft_qd_seed1_run.log",
        "manifest": "checkpoints/sft_qwen_delta_seed1_pilot6k/sft_dataset_manifest.json",
        "ckpt_dir": "checkpoints/sft_qwen_delta_seed1_pilot6k",
        "label": "pilot seed 1 (cosine -> 6k)",
        "color": "#7f7f7f",
    },
}
FULL_RUNS = [r for r, m in RUNS.items() if m["run_role"] == "full"]
PILOT_RUNS = [r for r, m in RUNS.items() if m["run_role"] == "pilot"]

METRIC_COLS = ["loss", "grad_norm", "learning_rate"]
TIDY_COLS = ["run_id", "run_role", "seed", "max_steps", "step", "epoch",
             "loss", "grad_norm", "learning_rate"]

# Relative LR deviations land at ~1e-16 — pure floating-point noise whose exact
# value shifts with the NumPy/BLAS build. Anything below this floor is reported
# as a canonical 0.0 so the tracked summary is byte-stable across environments;
# --check additionally compares the summary JSON numerically, not byte-wise.
LR_DEVIATION_FLOOR = 1e-12
LR_MATCH_TOLERANCE = 1e-6   # what counts as "matches the configured schedule"
CHECK_RTOL, CHECK_ATOL = 1e-9, 1e-12   # tolerance for the --check JSON compare

SMOOTH_WINDOW = 50          # observations, causal trailing mean
LOG_SPACING = 10            # trainer logging_steps -> ~500 training steps
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 0.1         # configured clipping threshold (sft_dataset_manifest)
SELECTION_GRID_STRIDE = 2000

# outputs (snapshot-rendered unless flagged raw-log-derived)
F_FULL_CSV = OUT / "sft_full_metrics.csv"
F_PILOT_CSV = OUT / "sft_pilot_metrics.csv"
F_MANIFEST = OUT / "sft_training_manifest.json"
F_SUMMARY_JSON = OUT / "sft_training_summary.json"
F_SUMMARY_MD = OUT / "sft_training_summary.md"
F_README = OUT / "README.md"
F_FULL_PNG = OUT / "sft_full_training_curves.png"
F_FULL_LOG_PNG = OUT / "sft_full_training_curves_logscale.png"
F_PILOT_PNG = OUT / "sft_pilot_training_curves.png"
F_PILOT_LOG_PNG = OUT / "sft_pilot_training_curves_logscale.png"
F_BEHAV_CSV = OUT / "sft_behavioral_trajectory.csv"
F_BEHAV_PNG = OUT / "sft_behavioral_trajectory.png"

# behavioural sources (tracked, so render/--check need no NSCC access)
PILOT_CORE = PROJECT_ROOT / "results" / "causal_baseline_pilot" / "pilot_core.json"
PILOT_TABLE = PROJECT_ROOT / "results" / "causal_baseline_pilot" / "pilot_table.md"
# where a FULL behavioural grid would live if it had been evaluated
FULL_GRID_PREFIX = {1: "Qwen-7B-SFT-qd-seed1-ckpt", 2: "Qwen-7B-SFT-qd-seed2-ckpt"}
FULL_GRID_ROOTS = ["baselines", "baseline"]

VERBATIM_BOUNDARY = (
    "Training loss measures fit to the supervised target tokens, whereas "
    "lambda, eta, consistency, and preference preservation are behavioral "
    "estimands computed after inference. A lower SFT loss is therefore not "
    "itself evidence of lower ownership dependence."
)
REPRO_STATEMENT = (
    "Rendering and checking the SFT curves is repository-reproducible; "
    "regenerating the underlying logged rows requires the raw NSCC "
    "trainer_state.json sources."
)
PILOT_SEPARATION = (
    "The pilot used a cosine schedule ending at 6,000. The full run used a "
    "cosine schedule ending at 30,000. Their step-6,000 weights are not "
    "interchangeable."
)
NO_GPU_STATEMENT = (
    "CPU-only rendering. This script ran no GPU/PBS job, no inference and no "
    "checkpoint evaluation, and opened no frozen or untouched suite. It also "
    "selects nothing: where a frozen checkpoint selection is shown it is READ "
    "from manifests written earlier by eval/select_checkpoint.py, and this "
    "script neither re-ranks nor breaks a tie. The underlying checkpoint "
    "evaluations were produced by separate GPU jobs "
    "(train/submit_eval_baseline_ckpt.pbs)."
)


# ── small utilities ───────────────────────────────────────────────────────────
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=PROJECT_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def canonical_deviation(x: float) -> float:
    """Collapse machine-precision noise to a cross-environment-stable value.

    Relative deviations below LR_DEVIATION_FLOOR are reported as exactly 0.0;
    anything larger is rounded to 6 significant digits. Without this the tracked
    summary carries a ~1e-16 value that differs between NumPy builds and makes
    an otherwise identical --check fail on nothing.
    """
    if not math.isfinite(x):
        return x
    if abs(x) < LR_DEVIATION_FLOOR:
        return 0.0
    return float(f"{x:.6g}")


def numerically_equal(a, b, rtol: float = CHECK_RTOL,
                      atol: float = CHECK_ATOL) -> list[str]:
    """Compare two JSON structures, allowing float slack. Returns difference paths."""
    diffs: list[str] = []

    def walk(x, y, path: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    diffs.append(f"{path}.{k} (present on one side only)")
                else:
                    walk(x[k], y[k], f"{path}.{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                diffs.append(f"{path} (length {len(x)} vs {len(y)})")
            else:
                for i, (xi, yi) in enumerate(zip(x, y)):
                    walk(xi, yi, f"{path}[{i}]")
        elif isinstance(x, bool) or isinstance(y, bool):
            if x != y:
                diffs.append(f"{path} ({x!r} vs {y!r})")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if not math.isclose(float(x), float(y), rel_tol=rtol, abs_tol=atol):
                diffs.append(f"{path} ({x!r} vs {y!r})")
        elif x != y:
            diffs.append(f"{path} ({x!r} vs {y!r})")

    walk(a, b, "")
    return diffs


def causal_rolling(series: pd.Series, window: int) -> pd.Series:
    """Trailing (causal) rolling mean: uses only current and PRECEDING points."""
    return series.rolling(window=window, min_periods=1).mean()


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=False) + "\n")


# ── (1) REFRESH: raw trainer_state.json -> tidy rows + source audit ───────────
def parse_trainer_state(path: Path, run_id: str, meta: dict) -> tuple[pd.DataFrame, dict]:
    """Parse one trainer_state.json into tidy rows plus a source-integrity audit.

    Never fabricates a value: a field absent from a log entry stays NaN and is
    reported in `missing_fields`. Duplicated / out-of-order / non-finite records
    are counted, not silently repaired.
    """
    raw = json.loads(path.read_text())
    history = raw.get("log_history", []) or []

    rows, malformed = [], []
    for i, entry in enumerate(history):
        if not isinstance(entry, dict) or "step" not in entry:
            malformed.append({"index": i, "reason": "no step field"})
            continue
        try:
            step = int(entry["step"])
        except (TypeError, ValueError):
            malformed.append({"index": i, "reason": "non-integer step"})
            continue
        row = {"step": step, "epoch": np.nan}
        for col in ["epoch"] + METRIC_COLS:
            val = entry.get(col, None)
            if val is None:
                row[col] = np.nan
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                malformed.append({"index": i, "step": step,
                                  "reason": f"non-numeric {col}: {val!r}"})
                row[col] = np.nan
                continue
            if not math.isfinite(fval):
                malformed.append({"index": i, "step": step,
                                  "reason": f"non-finite {col}: {val!r}"})
            row[col] = fval
        rows.append(row)

    df = pd.DataFrame(rows, columns=["step", "epoch"] + METRIC_COLS)

    steps = df["step"].to_numpy() if not df.empty else np.array([], dtype=int)
    monotonic = bool(np.all(np.diff(steps) > 0)) if steps.size > 1 else True
    dup_counts = pd.Series(steps).value_counts()
    duplicated = sorted(int(s) for s in dup_counts[dup_counts > 1].index)

    logging_steps = raw.get("logging_steps")
    max_steps = raw.get("max_steps")
    missing_intervals: list[int] = []
    if logging_steps and max_steps:
        expected = set(range(int(logging_steps), int(max_steps) + 1, int(logging_steps)))
        missing_intervals = sorted(expected - set(int(s) for s in steps))

    nonfinite = {c: int((~np.isfinite(df[c].to_numpy(dtype=float))).sum() -
                        int(df[c].isna().sum()))
                 for c in METRIC_COLS} if not df.empty else {c: 0 for c in METRIC_COLS}
    missing_fields = {c: int(df[c].isna().sum()) for c in ["epoch"] + METRIC_COLS} \
        if not df.empty else {c: 0 for c in ["epoch"] + METRIC_COLS}

    ckpt_dir = PROJECT_ROOT / meta["ckpt_dir"]
    ckpts = sorted(int(p.name.split("-")[1]) for p in ckpt_dir.glob("checkpoint-*")
                   if p.is_dir() and p.name.split("-")[-1].isdigit()) \
        if ckpt_dir.is_dir() else []

    audit = {
        "run_id": run_id,
        "run_role": meta["run_role"],
        "seed": meta["seed"],
        "source_path_absolute": str(path.resolve()),
        "source_path_repo_relative": (
            str(path.resolve().relative_to(PROJECT_ROOT))
            if str(path.resolve()).startswith(str(PROJECT_ROOT)) else None),
        "source_is_git_tracked": False,   # checkpoints/ and logs/ are gitignored
        "sha256": sha256_of(path),
        "size_bytes": path.stat().st_size,
        "max_steps": max_steps,
        "expected_max_steps": meta["expected_max_steps"],
        "max_steps_matches_expected_identity": max_steps == meta["expected_max_steps"],
        "global_step": raw.get("global_step"),
        "logging_steps": logging_steps,
        "save_steps": raw.get("save_steps"),
        "num_train_epochs": raw.get("num_train_epochs"),
        "final_epoch": raw.get("epoch"),
        "total_flos": raw.get("total_flos"),
        "train_batch_size": raw.get("train_batch_size"),
        "n_log_history_entries": len(history),
        "n_entries_with_loss": int(df["loss"].notna().sum()) if not df.empty else 0,
        "last_logged_step": int(steps.max()) if steps.size else None,
        "first_logged_step": int(steps.min()) if steps.size else None,
        "checkpoint_dirs_found": ckpts,
        "steps_monotonically_increasing": monotonic,
        "duplicated_steps": duplicated,
        "n_missing_expected_logging_intervals": len(missing_intervals),
        "missing_expected_logging_intervals": missing_intervals[:50],
        "malformed_or_nonfinite_records": malformed[:50],
        "n_malformed_or_nonfinite_records": len(malformed),
        "n_nonfinite_by_field": nonfinite,
        "n_missing_by_field": missing_fields,
        # These live in the end-of-run summary line, not in a checkpoint state.
        "final_train_loss_in_state": None,
        "train_runtime_in_state": None,
        "train_steps_per_second_in_state": None,
    }
    for entry in history:
        if isinstance(entry, dict) and "train_runtime" in entry:
            audit["train_runtime_in_state"] = entry.get("train_runtime")
            audit["train_steps_per_second_in_state"] = entry.get("train_steps_per_second")
            audit["final_train_loss_in_state"] = entry.get("train_loss")

    df.insert(0, "max_steps", max_steps)
    df.insert(0, "seed", meta["seed"])
    df.insert(0, "run_role", meta["run_role"])
    df.insert(0, "run_id", run_id)
    return df[TIDY_COLS], audit


_SUMMARY_RE = re.compile(r"\{'train_runtime':[^}]*\}")


def parse_run_summary(log_path: Path, expected_epoch: float | None,
                      tol: float = 5e-4) -> dict:
    """Recover the Trainer end-of-run summary from a text run log.

    A checkpoint-N/trainer_state.json never contains train_runtime, so runtime
    and throughput can only come from the job log. The seed-1 log holds BOTH the
    pilot and the full run, so summaries are disambiguated by final epoch; an
    ambiguous match is reported rather than guessed.
    """
    out = {"source": str(log_path), "found": False, "n_summaries_in_log": 0,
           "ambiguous": False, "train_runtime_s": None, "train_loss": None,
           "train_steps_per_second": None, "train_samples_per_second": None,
           "matched_epoch": None, "note": None}
    if not log_path.exists():
        out["note"] = "run log absent (gitignored raw NSCC source)"
        return out
    text = log_path.read_text(errors="replace")
    cands = []
    for m in _SUMMARY_RE.finditer(text):
        blob = m.group(0)
        fields = dict(re.findall(r"'([a-z_]+)':\s*'?([-0-9.eE+]+)'?", blob))
        try:
            cands.append({k: float(v) for k, v in fields.items()})
        except ValueError:
            continue
    out["n_summaries_in_log"] = len(cands)
    if not cands:
        out["note"] = "no train_runtime summary line found"
        return out
    if expected_epoch is None:
        matches = cands
    else:
        matches = [c for c in cands
                   if "epoch" in c and abs(c["epoch"] - expected_epoch) <= tol]
    if not matches:
        out["note"] = (f"no summary matched final epoch {expected_epoch}; "
                       f"{len(cands)} summaries present — not guessed")
        return out
    if len(matches) > 1:
        out["ambiguous"] = True
        out["note"] = (f"{len(matches)} summaries matched final epoch "
                       f"{expected_epoch}; ambiguous — not used")
        return out
    c = matches[0]
    out.update(found=True, train_runtime_s=c.get("train_runtime"),
               train_loss=c.get("train_loss"),
               train_steps_per_second=c.get("train_steps_per_second"),
               train_samples_per_second=c.get("train_samples_per_second"),
               matched_epoch=c.get("epoch"),
               note="matched by final epoch in the job log")
    return out


def read_config_manifest(path: Path) -> dict:
    if not path.exists():
        return {"present": False}
    d = json.loads(path.read_text())
    opt = d.get("optimizer", {})
    return {
        "present": True,
        "path": str(path.resolve()),
        "sha256": sha256_of(path),
        "git_commit_recorded": d.get("git_commit"),
        "seed": d.get("seed"),
        "loss": d.get("loss"),
        "max_steps": d.get("matching", {}).get("max_steps"),
        "learning_rate": opt.get("learning_rate"),
        "lr_scheduler_type": opt.get("lr_scheduler_type"),
        "warmup_ratio": opt.get("warmup_ratio"),
        "max_grad_norm": opt.get("max_grad_norm"),
        "output_dir_recorded": d.get("output_dir"),
    }


def do_refresh(args) -> dict:
    """Raw NSCC sources -> canonical compact CSV snapshot + source audit."""
    overrides = {"sft_full_seed1": args.seed1_state,
                 "sft_full_seed2": args.seed2_state,
                 "sft_pilot6k_seed1": args.pilot_state}
    frames: dict[str, pd.DataFrame] = {}
    refresh: dict = {"parser_version": PARSER_VERSION,
                     "extraction_date": date.today().isoformat(),
                     "git_commit_at_extraction": git_commit(),
                     "runs": {}}
    missing = []
    for run_id, meta in RUNS.items():
        path = Path(overrides[run_id]) if overrides[run_id] else PROJECT_ROOT / meta["state"]
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            missing.append(f"{run_id}: {path}")
            refresh["runs"][run_id] = {"run_id": run_id, "run_role": meta["run_role"],
                                       "seed": meta["seed"], "available": False,
                                       "source_path_absolute": str(path)}
            continue
        df, audit = parse_trainer_state(path, run_id, meta)
        audit["available"] = True
        audit["config_manifest"] = read_config_manifest(PROJECT_ROOT / meta["manifest"])
        audit["end_of_run_summary"] = parse_run_summary(
            PROJECT_ROOT / meta["runlog"], audit.get("final_epoch"))
        refresh["runs"][run_id] = audit
        frames[run_id] = df
        print(f"[refresh] {run_id:<18} {len(df):>5} rows  steps "
              f"{audit['first_logged_step']}..{audit['last_logged_step']}  "
              f"max_steps={audit['max_steps']}  sha256={audit['sha256'][:12]}…")

    if missing:
        print("[refresh] MISSING raw sources (no curve fabricated):")
        for m in missing:
            print(f"          {m}")

    OUT.mkdir(parents=True, exist_ok=True)
    full = [frames[r] for r in FULL_RUNS if r in frames]
    pilot = [frames[r] for r in PILOT_RUNS if r in frames]
    if full:
        pd.concat(full, ignore_index=True).to_csv(F_FULL_CSV, index=False)
        print(f"[refresh] wrote {F_FULL_CSV.relative_to(PROJECT_ROOT)}")
    if pilot:
        pd.concat(pilot, ignore_index=True).to_csv(F_PILOT_CSV, index=False)
        print(f"[refresh] wrote {F_PILOT_CSV.relative_to(PROJECT_ROOT)}")
    if not full and not pilot:
        raise SystemExit("[refresh] no SFT trainer_state.json source found — "
                         "nothing written, nothing fabricated.")
    refresh["missing_sources"] = missing
    refresh["full_behavioral_grid_scan"] = scan_full_behavioural_grid()
    return refresh


# ── (2) snapshot -> descriptive summaries ────────────────────────────────────
def load_snapshot() -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for csv in (F_FULL_CSV, F_PILOT_CSV):
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        for run_id, sub in df.groupby("run_id", sort=False):
            dfs[str(run_id)] = sub.sort_values("step").reset_index(drop=True)
    return dfs


def theoretical_lr(steps: np.ndarray, max_steps: int, base_lr: float,
                   warmup_ratio: float) -> np.ndarray:
    """HF linear-warmup + cosine-decay LambdaLR, evaluated the way it is logged.

    Trainer logs the scheduler value produced AFTER stepping, i.e. the factor at
    `step - 1`. That -1 offset is applied here so `recorded` and `configured`
    are compared on the same clock.
    """
    warm = int(round(warmup_ratio * max_steps))
    s = np.asarray(steps, dtype=float) - 1.0
    out = np.empty_like(s)
    rise = s < warm
    out[rise] = base_lr * s[rise] / max(1, warm)
    prog = (s[~rise] - warm) / max(1, max_steps - warm)
    out[~rise] = base_lr * 0.5 * (1.0 + np.cos(math.pi * prog))
    return out


def describe_run(run_id: str, df: pd.DataFrame, source_audit: dict) -> dict:
    meta = RUNS.get(run_id, {})
    loss = df["loss"].to_numpy(dtype=float)
    grad = df["grad_norm"].to_numpy(dtype=float)
    lr = df["learning_rate"].to_numpy(dtype=float)
    steps = df["step"].to_numpy(dtype=int)
    max_steps = int(df["max_steps"].iloc[0])
    fin = np.isfinite(loss)
    k = max(1, int(round(0.10 * fin.sum())))

    d = {
        "run_id": run_id,
        "run_role": meta.get("run_role", df["run_role"].iloc[0]),
        "seed": int(df["seed"].iloc[0]),
        "max_steps": max_steps,
        "n_plotted_observations": int(len(df)),
        "step_range": [int(steps.min()), int(steps.max())],
        "logging_stride_steps": int(np.median(np.diff(steps))) if len(steps) > 1 else None,
        "first_logged_loss": float(loss[fin][0]),
        "final_logged_loss": float(loss[fin][-1]),
        "min_logged_loss": float(np.nanmin(loss)),
        "step_of_min_loss": int(steps[int(np.nanargmin(loss))]),
        "max_logged_loss": float(np.nanmax(loss)),
        "step_of_max_loss": int(steps[int(np.nanargmax(loss))]),
        "median_loss_first_10pct": float(np.median(loss[fin][:k])),
        "median_loss_last_10pct": float(np.median(loss[fin][-k:])),
        "median_grad_norm": float(np.nanmedian(grad)),
        "max_grad_norm_logged": float(np.nanmax(grad)),
        "min_grad_norm_logged": float(np.nanmin(grad)),
        "final_learning_rate": float(lr[np.isfinite(lr)][-1]),
        "max_learning_rate_logged": float(np.nanmax(lr)),
        "step_of_max_learning_rate": int(steps[int(np.nanargmax(lr))]),
        "n_nonfinite_records": int((~np.isfinite(loss)).sum() +
                                   (~np.isfinite(grad)).sum() +
                                   (~np.isfinite(lr)).sum()),
    }
    d["absolute_loss_change_early_to_late"] = (
        d["median_loss_last_10pct"] - d["median_loss_first_10pct"])
    d["relative_loss_change_early_to_late"] = (
        d["absolute_loss_change_early_to_late"] / d["median_loss_first_10pct"]
        if d["median_loss_first_10pct"] else None)

    diffs = np.diff(loss[fin])
    if diffs.size:
        j = int(np.argmax(diffs))
        fsteps = steps[fin]
        d["largest_one_step_loss_increase"] = {
            "delta": float(diffs[j]),
            "from_step": int(fsteps[j]),
            "to_step": int(fsteps[j + 1]),
            "from_loss": float(loss[fin][j]),
            "to_loss": float(loss[fin][j + 1]),
        }

    # gradient clipping: HF logs the PRE-clipping norm returned by
    # clip_grad_norm_, so logged > max_grad_norm implies clipping was applied.
    gfin = np.isfinite(grad)
    d["configured_max_grad_norm"] = MAX_GRAD_NORM
    d["frac_logged_obs_above_clip_threshold"] = float((grad[gfin] > MAX_GRAD_NORM).mean())
    d["grad_norm_clipping_note"] = (
        "HF Trainer logs the pre-clipping global gradient norm; the fraction "
        "above max_grad_norm is over LOGGED steps only (1 in "
        f"{d['logging_stride_steps']}), not all optimizer steps.")

    # learning-rate schedule fidelity
    cfg = (source_audit or {}).get("config_manifest", {})
    base_lr = cfg.get("learning_rate")
    warm_ratio = cfg.get("warmup_ratio", WARMUP_RATIO)
    sched = cfg.get("lr_scheduler_type")
    lr_block = {"configured_learning_rate": base_lr,
                "configured_scheduler": sched,
                "configured_warmup_ratio": warm_ratio,
                "expected_warmup_boundary_step": (
                    int(round((warm_ratio or WARMUP_RATIO) * max_steps))),
                "recorded_peak_lr": d["max_learning_rate_logged"],
                "recorded_peak_step": d["step_of_max_learning_rate"]}
    if base_lr and sched == "cosine":
        theo = theoretical_lr(steps, max_steps, float(base_lr), float(warm_ratio))
        ok = np.isfinite(lr) & (theo > 0)
        rel = np.abs(lr[ok] - theo[ok]) / theo[ok]
        lr_block.update({
            "compared_against": "linear warmup + cosine decay (HF LambdaLR), "
                                "evaluated at step-1 to match Trainer logging",
            "max_relative_deviation": canonical_deviation(float(rel.max())),
            "median_relative_deviation": canonical_deviation(float(np.median(rel))),
            "deviation_floor": LR_DEVIATION_FLOOR,
            "deviation_below_floor": bool(rel.max() < LR_DEVIATION_FLOOR),
            "deviation_canonicalization": (
                f"Deviations below {LR_DEVIATION_FLOOR:g} are reported as 0.0 and "
                f"larger ones are rounded to 6 significant digits; the raw values "
                f"are floating-point noise whose exact magnitude varies with the "
                f"NumPy/BLAS build."),
            "match_tolerance": LR_MATCH_TOLERANCE,
            "matches_configured_schedule": bool(rel.max() < LR_MATCH_TOLERANCE),
        })
    else:
        lr_block["compared_against"] = None
        lr_block["matches_configured_schedule"] = None
    d["learning_rate_schedule"] = lr_block

    # end-of-run runtime / throughput (job log, not the checkpoint state)
    summ = (source_audit or {}).get("end_of_run_summary", {})
    d["train_runtime_s"] = summ.get("train_runtime_s")
    d["train_steps_per_second"] = summ.get("train_steps_per_second")
    d["trainer_aggregate_train_loss"] = summ.get("train_loss")
    d["runtime_source"] = summ.get("source") if summ.get("found") else None
    d["runtime_note"] = summ.get("note")
    if d["trainer_aggregate_train_loss"] is not None:
        d["final_logged_loss_vs_aggregate_train_loss"] = {
            "final_logged_loss": d["final_logged_loss"],
            "trainer_aggregate_train_loss": d["trainer_aggregate_train_loss"],
            "mean_of_logged_losses": float(np.mean(loss[fin])),
            "note": "The last logged point is one logging-interval mean, not "
                    "the run aggregate. The Trainer's train_loss equals the "
                    "mean over all logging intervals; both are reported.",
        }
    return d


# ── (3) figures ───────────────────────────────────────────────────────────────
PANELS = [("loss", "completion-only cross-entropy loss"),
          ("grad_norm", "gradient norm (pre-clipping)"),
          ("learning_rate", "learning rate")]

# Fixed viewing windows for the LINEAR figure only. Both quantities have long
# thin tails (loss to 2.76, gradient norm to 543) that compress the bulk of the
# data into a sliver under autoscale. Cropping is a display choice, not a data
# change: every value stays in the CSV, the log-scale companion shows the full
# range, and each cropped panel prints how many observations fall outside the
# window so the crop can never be mistaken for the data ending there.
PANEL_YLIM = {"loss": (0.0, 1.5), "grad_norm": (0.0, 100.0)}


def _panel(ax, col, title, runs, window, logscale, mark_ckpts, warmups):
    for run_id, df in runs.items():
        color = RUNS[run_id]["color"]
        label = RUNS[run_id]["label"]
        y = df[col]
        if col == "learning_rate":
            ax.plot(df["step"], y, color=color, lw=1.6, label=f"{label} (recorded)")
        else:
            ax.plot(df["step"], y, color=color, alpha=0.16, lw=0.6)
            ax.plot(df["step"], causal_rolling(y, window), color=color, lw=1.9,
                    label=f"{label} — causal mean, w={window} obs")
    if col == "learning_rate":
        for run_id, df in runs.items():
            ms = int(df["max_steps"].iloc[0])
            base = df["learning_rate"].max()
            theo = theoretical_lr(df["step"].to_numpy(), ms, float(base), WARMUP_RATIO)
            ax.plot(df["step"], theo, color="k", ls=":", lw=0.9, alpha=0.55,
                    label="configured schedule (reference overlay)"
                    if run_id == list(runs)[0] else None)
    if mark_ckpts:
        for s in mark_ckpts:
            ax.axvline(s, color="0.55", lw=0.5, alpha=0.20, zorder=0)
    if logscale:
        ax.set_yscale("log")
    elif col == "learning_rate":
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if col == "learning_rate":
        # warmup boundaries last, so the annotation sits inside the final ylim.
        # Runs sharing a boundary (both full seeds -> 1,500) are drawn once.
        for wb in sorted(set(warmups.values())):
            if not wb:
                continue
            ax.axvline(wb, color="#444444", ls="--", lw=1.1, alpha=0.85)
            ax.annotate(f"expected warmup end {wb:,}", xy=(wb, 0.02),
                        xycoords=("data", "axes fraction"),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=7.5, color="#444444", rotation=90,
                        va="bottom", ha="left")
    if not logscale and col in PANEL_YLIM:
        lo, hi = PANEL_YLIM[col]
        ax.set_ylim(lo, hi)
        # Disclose the crop: count raw observations outside the window, per run.
        total_out = total_n = 0
        for df in runs.values():
            v = df[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            total_out += int(((v < lo) | (v > hi)).sum())
            total_n += v.size
        if total_out:
            ax.annotate(
                f"y-axis cropped to [{lo:g}, {hi:g}]\n"
                f"{total_out:,}/{total_n:,} points ({total_out / total_n:.1%}) "
                f"above it — full range in the log companion",
                xy=(0.5, -0.16), xycoords="axes fraction", ha="center",
                va="top", fontsize=7.5, color="#7a2020", linespacing=1.35)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("training step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="best", framealpha=0.85)


def plot_runs(runs: dict[str, pd.DataFrame], path: Path, window: int,
              suptitle: str, logscale: bool, ckpt_marks: list[int]) -> None:
    warmups = {r: int(round(WARMUP_RATIO * int(df["max_steps"].iloc[0])))
               for r, df in runs.items()}
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))
    for ax, (col, title) in zip(axes, PANELS):
        _panel(ax, col, title, runs, window, logscale, ckpt_marks, warmups)
    fig.suptitle(suptitle, fontsize=11.5, y=0.985, va="top")
    fig.tight_layout()
    fig.subplots_adjust(top=0.815, bottom=0.175 if not logscale else 0.115)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def full_suptitle(window: int, logscale: bool) -> str:
    scale = "log-scale companion — " if logscale else ""
    return (
        f"SFT training metrics — matched supervised fine-tuning, Qwen-own-delta "
        f"targets, full runs (seed 1 & seed 2, cosine -> 30,000)\n"
        f"{scale}faint = raw logged observations; bold = CAUSAL trailing mean, "
        f"window = {window} observations (~{window * LOG_SPACING} steps), identical "
        f"for both seeds; grey verticals = 2k-stride checkpoints\n"
        f"Optimization diagnostics only — NOT behavioral results, NOT comparable "
        f"to GRPO loss values")


def pilot_suptitle(window: int, logscale: bool) -> str:
    scale = "log-scale companion — " if logscale else ""
    return (
        f"SFT training metrics — EXPLORATORY seed-1 PILOT (cosine -> 6,000), plotted "
        f"separately from the full runs\n"
        f"{scale}faint = raw logged observations; bold = CAUSAL trailing mean, "
        f"window = {window} observations (~{window * LOG_SPACING} steps)\n"
        f"{PILOT_SEPARATION}")


# ── (4) behavioural trajectory (already-recorded results only) ────────────────
_MD_ROW = re.compile(r"^\|\s*SFT\s*\|\s*(\d+)\s*\|(.+)$")


def _num(tok: str) -> float:
    return float(tok.strip().replace("−", "-").replace("−", "-").split(" ")[0])


def parse_pilot_W() -> dict[int, float]:
    """Read the W column for the SFT pilot rows out of the tracked pilot table."""
    if not PILOT_TABLE.exists():
        return {}
    out: dict[int, float] = {}
    for line in PILOT_TABLE.read_text().splitlines():
        m = _MD_ROW.match(line.strip())
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        # columns after ckpt: lambda, eta, d, consist, keep, trade, W, cond(J)
        if len(cells) >= 7:
            out[int(m.group(1))] = _num(cells[6])
    return out


def scan_full_behavioural_grid() -> dict:
    """Read-only VERIFICATION of an already-evaluated FULL SFT grid.

    A working-tree observation, so it runs in --refresh (the mode allowed to
    look at the machine) and is recorded in the manifest; render and --check
    then consume the recorded scan, which keeps them snapshot-reproducible.

    Completion is decided by eval/verify_sft_grid.py, NOT by the presence of a
    `Model_1` directory. `estimate_qwen_checkpoint.py` creates `Model_1/` and
    writes its PNGs before the NLS CSV lands, so a directory test marks a
    checkpoint complete while its estimation is still running — observed
    directly on 2026-08-06, when seed-1 step 24,000 had a populated `Model_1/`
    and no CSV. The verifier instead requires, per checkpoint: the adapter
    (hashed), both perspectives, N = 9,890 paired cases, zero parse failures, a
    T=1 Model-A CSV, and finite estimates with positive standard errors.

    Only the FLAT `<root>/<prefix><step>` layout counts. The seed-1 pilot's own
    2k/4k/6k evaluations used exactly these names and now live one level down in
    `baselines/pilot6k/` (see that directory's PROVENANCE.md), so they are
    structurally excluded rather than excluded by a hardcoded step list.
    """
    from verify_sft_grid import (EVAL_ROOTS, GRID_STEPS, MANIFEST, SEEDS,
                                 verify_grid)

    man = verify_grid()
    # --refresh is the mode allowed to touch the working tree, so persist the
    # verification the scan block points at; otherwise `verification_manifest`
    # would name a file that only exists if the verifier was run by hand.
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2, default=str))
    found: dict[int, list[int]] = {s: [] for s in SEEDS}
    rows: list[dict] = []
    for c in man["checkpoints"]:
        if c["status"] != "VERIFIED":
            continue
        found[c["seed"]].append(c["step"])
        e, a = c["evaluation"], c["adapter"]
        rows.append({
            "seed": c["seed"], "step": c["step"], "model_name": e["model_name"],
            "adapter_path": a["adapter_path"],
            "adapter_weight_sha256": a.get("adapter_weight_sha256"),
            "eval_dir": e["eval_dir"], "nls_csv": e["nls_csv"],
            "nls_csv_sha256": e["nls_csv_sha256"],
            "lambda": e["lambda"], "lambda_se": e["lambda_se"],
            "eta": e["eta"], "eta_se": e["eta_se"], "d": e["d"],
            "consistency": e["consistency"], "keep_both": e["keep_both_rate"],
            "trade_both": e["trade_both_rate"], "W": e.get("W"),
            "n_cases": e["n_cases"], "parse_failures": e["parse_failures"],
        })
    pilot_dir = PROJECT_ROOT / "baselines" / "pilot6k"
    return {
        "verifier": "eval/verify_sft_grid.py",
        "verification_manifest": "results/sft_grid_verification.json",
        "completion_test": (
            "per-checkpoint verification (adapter hash, both perspectives, "
            "N=9,890 paired cases, zero parse failures, T=1 Model-A CSV, finite "
            "estimates with positive SEs) — NOT Model_1 directory existence"),
        "scanned_roots": list(EVAL_ROOTS),
        "scan_is_recursive": False,
        "dir_prefixes": {str(k): v for k, v in FULL_GRID_PREFIX.items()},
        "required": {"seeds": list(SEEDS), "checkpoints": list(GRID_STEPS),
                     "n_evaluations_required": man["n_required"]},
        "directories_found": {str(k): sorted(v) for k, v in found.items()},
        "n_attributable_to_full_runs": man["n_verified"],
        "pilot_evaluations_quarantined_at": (
            str(pilot_dir.relative_to(PROJECT_ROOT)) if pilot_dir.is_dir() else None),
        "pilot_exclusion_mechanism": (
            "structural: the seed-1 pilot evaluations were moved into "
            "baselines/pilot6k/ on 2026-08-06 so they cannot occupy the flat "
            "full-run names, and this non-recursive scan cannot reach them."),
        "problems": man["problems"],
        "complete": man["complete"],
        "identity_verified": man["identity_verified"],
        "verified_rows": rows,
        "protocol": man["protocol"],
    }


def load_frozen_selection() -> dict:
    """Selected step per seed, from the frozen selector's own manifests.

    Read-only: the marks come from what eval/select_checkpoint.py already wrote.
    This function never selects, re-ranks, or breaks a tie itself.
    """
    out: dict[int, dict] = {}
    for seed in FULL_GRID_PREFIX:
        p = (PROJECT_ROOT / "results" / "checkpoint_selection" /
             f"Qwen-7B-SFT-qd-seed{seed}.json")
        if not p.is_file():
            continue
        try:
            sel = json.loads(p.read_text())
        except Exception:
            continue
        out[seed] = {
            "selected_step": sel.get("selected_step"),
            "protocol_frozen": sel.get("protocol_frozen"),
            "rule": sel.get("rule"),
            "selection_data": sel.get("selection_data"),
            "provenance_note": sel.get("provenance_note"),
            "manifest": str(p.relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256_of(p),
            "manifest_mtime_utc": datetime.fromtimestamp(
                p.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
            "read_only": ("selection performed earlier by "
                          "eval/select_checkpoint.py; this script only reads it"),
        }
    return {str(k): v for k, v in out.items()}


def load_base_comparator() -> dict:
    """The matched local base under the SAME plain `baseline` prompt.

    The SFT grid was evaluated with `--treatment baseline`
    (train/submit_eval_baseline_ckpt.pbs), so the matched comparator is the
    plain-prompt base — NOT the debias- or forced-treated base. Same weights
    family, same 9,890 rows, same teacher-forced scorer, same Model-A NLS at
    T=1; only the training differs.
    """
    d = PROJECT_ROOT / "baseline" / "Qwen-7B-Base-Local"
    csvs = sorted(glob.glob(str(d / "Model_1" / "*NLS_estimation_T1*.csv")))
    if not csvs:
        return {"available": False, "reason": f"{d} has no T=1 Model-A CSV"}
    rows = list(csv_module.DictReader(open(csvs[0])))
    est = {r["Parameter"]: r for r in rows}
    try:
        xr = load_rows_helper(d / "loss_aversion_X.json")
        yr = load_rows_helper(d / "loss_aversion_Y.json")
    except Exception as exc:
        return {"available": False, "reason": f"raw rows unreadable: {exc}"}
    common = sorted(set(xr) & set(yr))
    con = keep = trade = 0
    for cid in common:
        a, b = choice_helper(xr[cid]), choice_helper(yr[cid])
        if a != b:
            con += 1
        elif a == "No":
            keep += 1
        else:
            trade += 1
    n = len(common) or 1
    # W on the same shared reference table the grid rows use, so the sixth panel
    # has a comparable base line rather than an empty one.
    w_block: dict = {}
    try:
        from pseudo_utility_alignment import (DEFAULT_DATA_DIR,
                                              DEFAULT_UTILITY_FILE, compute_W,
                                              load_case_utilities)
        w = compute_W(d, load_case_utilities(DEFAULT_DATA_DIR, DEFAULT_UTILITY_FILE))
        w_block = {"W": w["W"], "rational_choice_rate": w["rational_choice_rate"]}
    except Exception as exc:
        w_block = {"W": None, "W_unavailable_reason": str(exc)}
    return {
        "available": True,
        **w_block,
        "model_name": "Qwen-7B-Base-Local",
        "treatment": "baseline (plain prompt) — matched to the SFT grid",
        "eval_dir": str(d.relative_to(PROJECT_ROOT)),
        "nls_csv": str(Path(csvs[0]).relative_to(PROJECT_ROOT)),
        "lambda": float(est["lambda"]["Estimate"]),
        "lambda_se": float(est["lambda"]["Std. Err."]),
        "eta": float(est["eta"]["Estimate"]),
        "eta_se": float(est["eta"]["Std. Err."]),
        "consistency": con / n, "keep_both": keep / n, "trade_both": trade / n,
        "n_cases": len(common),
        "note": ("Matched on weights family, rows, scorer and estimator; the SFT "
                 "grid and this base share the plain `baseline` prompt, so the "
                 "contrast is training alone."),
    }


def build_behavioural(grid_scan: dict | None) -> tuple[pd.DataFrame | None, dict]:
    """Assemble the ONLY already-evaluated SFT structural trajectory.

    Read-only: no inference, no estimator run, no selector. Uses the recorded
    full-grid scan and reports honestly when the grid is absent.
    """
    scan = grid_scan or {"complete": False, "note": "no grid scan recorded"}
    grid_complete = bool(scan.get("complete"))

    grid_verified = bool(scan.get("identity_verified"))
    if grid_verified:
        statement = ("A complete 2-seed x 15-checkpoint SFT grid was located and "
                     "every cell passed per-checkpoint identity and completeness "
                     "verification; it is plotted read-only from the recorded "
                     "verification manifest.")
    elif grid_complete:
        statement = ("A complete-looking grid was found but identity "
                     "verification did not pass, so no full-run trajectory is "
                     "plotted.")
    else:
        statement = ("The two full SFT runs have completed training, but their "
                     "behavioral checkpoint grid is not fully evaluated and "
                     "verified. Therefore, no full-run lambda/eta trajectory is "
                     "available.")
    status = {
        "full_grid_scan": scan,
        "full_grid_complete": grid_complete,
        "full_grid_identity_verified": grid_verified,
        "frozen_selector_run_for_sft": False,
        "statement": statement,
    }

    # The verified-grid path. A complete grid is only plotted when every cell
    # passed eval/verify_sft_grid.py — identity, completeness and estimator
    # checks — never on directory existence alone. `identity_verified` is set by
    # the verifier, so a merely-complete-looking scan still falls through to the
    # pilot branch below.
    if grid_complete and scan.get("identity_verified"):
        rows = scan.get("verified_rows") or []
        n_req = scan.get("required", {}).get("n_evaluations_required")
        if n_req is not None and len(rows) != n_req:
            status["note"] = (f"verified-row count {len(rows)} != required {n_req}; "
                              "refusing to plot a partial grid.")
            return None, status

        recs = []
        for r in sorted(rows, key=lambda x: (x["seed"], x["step"])):
            recs.append({
                "run_id": f"sft_full_seed{r['seed']}",
                "run_role": "full",
                "seed": int(r["seed"]),
                "step": int(r["step"]),
                "lambda": r["lambda"], "lambda_se": r["lambda_se"],
                "eta": r["eta"], "eta_se": r["eta_se"],
                "d": r["d"], "consistency": r["consistency"],
                "keep_both": r["keep_both"], "trade_both": r["trade_both"],
                "W": r.get("W", np.nan),
                "n_cases": int(r["n_cases"]),
                "adapter_path": r["adapter_path"],
                "adapter_weight_sha256": r.get("adapter_weight_sha256"),
                "nls_csv_sha256": r.get("nls_csv_sha256"),
                "dataset": "test_goods (VALIDATION)",
                "estimator": "Model A NLS, structural link scale T=1",
                "status": "VERIFIED",
            })
        df = pd.DataFrame(recs)
        df["d_recomputed"] = np.sqrt(df["lambda"] ** 2 + df["eta"] ** 2)
        dupes = int(len(df) - df.groupby(["seed", "step"]).ngroups)
        if dupes:
            status["note"] = f"{dupes} duplicate seed/step row(s); refusing to plot."
            return None, status

        status["full_trajectory"] = {
            "available": True,
            "seeds_covered": sorted(int(s) for s in df["seed"].unique()),
            "checkpoints_per_seed": {
                str(s): sorted(int(x) for x in df.loc[df["seed"] == s, "step"])
                for s in sorted(df["seed"].unique())},
            "n_checkpoints": int(len(df)),
            "duplicate_checkpoint_rows": dupes,
            "d_max_abs_discrepancy_vs_sqrt_lambda2_eta2":
                float(np.abs(df["d"] - df["d_recomputed"]).max()),
            "W_available": bool(df["W"].notna().all()),
            "adapter_hash_available": bool(
                df["adapter_weight_sha256"].notna().all()),
            "identity_verification": scan.get("verification_manifest"),
            "label": ("Full 2-seed x 15-checkpoint SFT grid, every cell verified; "
                      "test_goods VALIDATION estimates used for checkpoint "
                      "selection — not final-test performance."),
        }
        status["frozen_selection"] = load_frozen_selection()
        status["base_comparator"] = load_base_comparator()
        status["frozen_selector_run_for_sft"] = bool(status["frozen_selection"])
        status["method_comparison"] = (
            "No SFT-versus-GRPO winner is declared or plotted. This figure shows "
            "SFT against its own matched plain-prompt base only; GRPO outputs are "
            "not read by this script (see module docstring), and a cross-method "
            "claim would additionally require the untouched method-comparison "
            "suite under METHOD_COMPARISON_PROTOCOL.md.")
        return df, status

    if not PILOT_CORE.exists():
        status["pilot_trajectory"] = {"available": False,
                                      "reason": f"{PILOT_CORE} absent"}
        return None, status

    core = json.loads(PILOT_CORE.read_text())
    rows = [r for r in core if r.get("meth") == "SFT"]
    if not rows:
        status["pilot_trajectory"] = {"available": False,
                                      "reason": "no SFT rows in pilot_core.json"}
        return None, status
    W = parse_pilot_W()
    recs = []
    for r in sorted(rows, key=lambda x: x["ck"]):
        recs.append({
            "run_id": "sft_pilot6k_seed1",
            "run_role": "pilot",
            "seed": 1,
            "step": int(r["ck"]),
            "lambda": r["lam"], "lambda_se": r["lse"],
            "eta": r["eta"], "eta_se": r["ese"],
            "d": r["d"], "consistency": r["cons"],
            "keep_both": r["keep"], "trade_both": r["trade"],
            "W": W.get(int(r["ck"]), np.nan),
            "n_cases": int(r["n"]),
            "dataset": "test_goods (VALIDATION)",
            "estimator": "Model A NLS, structural link scale T=1",
            "status": "EXPLORATORY",
        })
    df = pd.DataFrame(recs)
    # cross-check d against sqrt(lambda^2 + eta^2) as recorded
    df["d_recomputed"] = np.sqrt(df["lambda"] ** 2 + df["eta"] ** 2)
    dmax = float(np.abs(df["d"] - df["d_recomputed"]).max())
    status["pilot_trajectory"] = {
        "available": True,
        "source_files": {
            "pilot_core.json": {"path": str(PILOT_CORE.relative_to(PROJECT_ROOT)),
                                "sha256": sha256_of(PILOT_CORE)},
            "pilot_table.md": {"path": str(PILOT_TABLE.relative_to(PROJECT_ROOT)),
                               "sha256": sha256_of(PILOT_TABLE)},
        },
        "checkpoint_identity": "checkpoints/sft_qwen_delta_seed1_pilot6k/checkpoint-{2000,4000,6000}",
        "evaluation_outputs": "baselines/pilot6k/Qwen-7B-SFT-qd-seed1-ckpt{2000,4000,6000}",
        "provenance_note": (
            "The July-27 evaluation logs record the adapter path as "
            "checkpoints/sft_qwen_delta_seed1/checkpoint-N. The pilot trained "
            "into that path on July 26 and was renamed to *_pilot6k before the "
            "full run reused the name on July 29, so those logs refer to the "
            "PILOT adapters. Directory mtimes and the pilot's own manifest "
            "(max_steps=6000) confirm the identity. On 2026-08-06 the pilot's "
            "evaluation outputs were moved from the flat baselines/ names into "
            "baselines/pilot6k/ so a full-run evaluation cannot resume from them "
            "and silently return pilot numbers; see that directory's "
            "PROVENANCE.md."),
        "adapter_hash_available": False,
        "seeds_covered": [1],
        "checkpoints_covered": sorted(int(x) for x in df["step"]),
        "n_checkpoints": int(len(df)),
        "duplicate_checkpoint_rows": int(len(df) - df["step"].nunique()),
        "d_max_abs_discrepancy_vs_sqrt_lambda2_eta2": dmax,
        "W_recovered_from_table": bool(df["W"].notna().all()),
        "label": ("Exploratory seed-1 pilot; test_goods validation; incomplete "
                  "three-point grid; no frozen selector; not a full-run result."),
    }
    return df, status


# Six panels: the two structural estimands, three DIRECT behaviour rates, and
# the preference/alignment diagnostic W. They are deliberately different kinds
# of quantity and are labelled as such — lambda/eta are model-fitted, the middle
# three are raw choice rates, W scores choices against a shared utility table.
BEHAV_PANELS = [
    ("lambda", "λ  (loss aversion)", "structural", True),
    ("eta", "η  (status-quo bias)", "structural", True),
    ("consistency", "paired consistency", "direct behavior", False),
    ("keep_both", "keep-both fraction", "direct behavior", False),
    ("trade_both", "trade-both fraction", "direct behavior", False),
    ("W", "W  (pseudo-utility alignment)", "preference diagnostic", False),
]


def _mark_base(ax, col: str, base: dict) -> None:
    """Draw the matched plain-prompt base as a reference line.

    The base sits far outside the trained range on the structural panels
    (lambda = 7.64 against |lambda| < 0.4), so forcing a shared linear scale
    would flatten the trajectory into a straight line. Where the value is off
    the current view it is annotated at the panel edge instead of rescaling.
    """
    key = {"keep_both": "keep_both", "trade_both": "trade_both"}.get(col, col)
    v = base.get(key)
    if v is None or not np.isfinite(v):
        return
    lo, hi = ax.get_ylim()
    if lo <= v <= hi:
        ax.axhline(v, color="#7a2020", lw=1.2, ls=":", zorder=1)
        ax.annotate(f"base {v:.3g}", xy=(0.99, v), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7.5, color="#7a2020")
    else:
        edge, va = (hi, "top") if v > hi else (lo, "bottom")
        ax.annotate(f"matched base = {v:.3g}  (off scale)",
                    xy=(0.99, edge), xycoords=("axes fraction", "data"),
                    ha="right", va=va, fontsize=7.5, color="#7a2020",
                    fontweight="bold")


def plot_behavioural(df: pd.DataFrame, path: Path, status: dict | None = None) -> None:
    # d = sqrt(lambda^2 + eta^2) is retained in the CSV but no longer plotted:
    # it is a deterministic function of the first two panels, so the panel added
    # no information the reader could not already see.
    status = status or {}
    is_full = str(df["run_role"].iloc[0]) == "full"
    base = (status.get("base_comparator") or {}) if is_full else {}
    selection = (status.get("frozen_selection") or {}) if is_full else {}

    if is_full:
        series = [(int(s), RUNS[f"sft_full_seed{s}"]["color"],
                   f"full seed {int(s)}") for s in sorted(df["seed"].unique())]
        xlabel = "SFT training step"
    else:
        series = [(int(df["seed"].iloc[0]), RUNS["sft_pilot6k_seed1"]["color"],
                   "pilot seed 1")]
        xlabel = "pilot training step"

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    for ax, (col, title, kind, zero) in zip(axes.ravel(), BEHAV_PANELS):
        for seed, colour, label in series:
            sub = df[df["seed"] == seed].sort_values("step")
            if sub[col].isna().all():
                continue
            if col in ("lambda", "eta"):
                ax.errorbar(sub["step"], sub[col], yerr=sub[f"{col}_se"],
                            color=colour, marker="o", lw=1.8, ms=4.5, capsize=3,
                            label=label, zorder=3)
            else:
                ax.plot(sub["step"], sub[col], color=colour, marker="o", lw=1.8,
                        ms=4.5, label=label, zorder=3)
            # Frozen selected checkpoint — marked, never chosen here.
            picked = (selection.get(str(seed)) or {}).get("selected_step")
            if picked is not None:
                hit = sub[sub["step"] == picked]
                if not hit.empty and np.isfinite(hit[col].iloc[0]):
                    ax.plot(hit["step"], hit[col], marker="*", ms=17,
                            mfc="none", mec=colour, mew=1.9, ls="none",
                            zorder=5,
                            label=f"frozen selection: step {picked:,}")
        if zero:
            ax.axhline(0.0, color="0.4", lw=0.8, ls="--", zorder=1)
        # Margins must be applied BEFORE the base is drawn: _mark_base decides
        # "on scale" vs "off scale" from the current y-limits, and a base value
        # just outside the autoscaled range (trade-both = 0) becomes visible once
        # the margin is added — annotating it "off scale" would then be wrong.
        ax.margins(y=0.16)
        ax.autoscale_view()
        if base.get("available"):
            _mark_base(ax, col, base)
        ax.set_title(f"{title}\n[{kind}]", fontsize=10.5)
        ax.set_xlabel(xlabel)
        if is_full:
            ax.set_xticks(range(0, 30001, 6000))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
        else:
            ax.set_xticks(sorted(df["step"]))
        ax.grid(alpha=0.25)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    if base.get("available"):
        h2.append(plt.Line2D([], [], color="#7a2020", lw=1.2, ls=":"))
        l2.append(f"matched base, plain prompt (λ={base['lambda']:.3g}, "
                  f"η={base['eta']:.3g})")
    if h2:
        fig.legend(h2, l2, loc="lower center", ncol=min(len(h2), 4),
                   frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.055))

    if is_full:
        picks = ", ".join(
            f"seed {k} -> step {(v or {}).get('selected_step'):,}"
            for k, v in sorted(selection.items())
            if (v or {}).get("selected_step") is not None) or "not yet run"
        fig.text(0.5, 0.012,
                 "test_goods = VALIDATION / checkpoint-selection estimates, "
                 "N = 9,890 paired cases per point — NOT final-test performance "
                 "and NOT the 49,450-case prospective configuration suite.\n"
                 "Comparator: matched local base under the SAME plain `baseline` "
                 "prompt (same rows, scorer and Model-A NLS at T=1).  "
                 f"Frozen selection: {picks}.  "
                 "No SFT-versus-GRPO winner is shown or implied.",
                 fontsize=8.5, ha="center", va="bottom", color="#7a2020")
        fig.suptitle(
            "SFT behavioral trajectory — full runs, seeds 1 and 2, all 15 frozen "
            "checkpoints (2k–30k), every cell identity-verified.\n"
            "Structural estimands (λ, η) computed after inference — separate "
            "from, and not implied by, the SFT training-metric curves.",
            fontsize=11, y=0.985, va="top")
    else:
        fig.text(0.5, 0.012,
                 "EXPLORATORY seed-1 PILOT  ·  test_goods = VALIDATION  ·  incomplete "
                 "3-point grid  ·  no frozen selector run  ·  NOT a full-run result"
                 "     |     Model A NLS, structural link scale T=1  ·  N = 9,890 "
                 "cases per point  ·  error bars = ±1 SE",
                 fontsize=9, ha="center", va="bottom", color="#7a2020")
        fig.suptitle(
            "SFT behavioral trajectory — EXPLORATORY seed-1 pilot (cosine -> 6,000), "
            "test_goods VALIDATION, incomplete three-point grid, no frozen selector; "
            "not a full-run result.\n"
            "Structural estimands computed after inference — separate from, and not "
            "implied by, the SFT training-metric curves.", fontsize=11, y=0.985,
            va="top")
    fig.tight_layout()
    fig.subplots_adjust(top=0.875, bottom=0.155 if is_full else 0.10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── (5) prose outputs ────────────────────────────────────────────────────────
def render_summary_md(summary: dict) -> str:
    L = ["# SFT training metrics — descriptive summary", "",
         "Matched supervised fine-tuning baseline (Qwen-own-delta rational-choice "
         "targets), completion-only cross-entropy. **Optimization diagnostics "
         "only; not behavioral results.**", "",
         f"> {VERBATIM_BOUNDARY}", "",
         f"Smoothing in the figures: causal trailing rolling mean, window = "
         f"{summary['smoothing']['window_obs']} observations "
         f"(~{summary['smoothing']['window_steps']} training steps), identical for "
         f"every run; only current and preceding observations are used. All "
         f"unsmoothed values are retained in the CSVs.", "",
         "## Per-run descriptive statistics", "",
         "| quantity | " + " | ".join(summary["runs"][r]["run_id"]
                                      for r in summary["run_order"]) + " |",
         "|---|" + "---|" * len(summary["run_order"])]

    def row(name, fn):
        vals = []
        for r in summary["run_order"]:
            try:
                vals.append(fn(summary["runs"][r]))
            except Exception:
                vals.append("—")
        L.append(f"| {name} | " + " | ".join(vals) + " |")

    f4 = lambda x: "—" if x is None else f"{x:.4f}"
    f3g = lambda x: "—" if x is None else f"{x:.3g}"
    row("run role", lambda d: d["run_role"])
    row("seed", lambda d: str(d["seed"]))
    row("max_steps (cosine endpoint)", lambda d: f"{d['max_steps']:,}")
    row("plotted observations", lambda d: f"{d['n_plotted_observations']:,}")
    row("logging stride (steps)", lambda d: str(d["logging_stride_steps"]))
    row("first logged loss", lambda d: f4(d["first_logged_loss"]))
    row("final logged loss", lambda d: f4(d["final_logged_loss"]))
    row("minimum logged loss", lambda d: f3g(d["min_logged_loss"]))
    row("step of minimum loss", lambda d: f"{d['step_of_min_loss']:,}")
    row("median loss, first 10%", lambda d: f4(d["median_loss_first_10pct"]))
    row("median loss, last 10%", lambda d: f4(d["median_loss_last_10pct"]))
    row("absolute change (early→late)", lambda d: f4(d["absolute_loss_change_early_to_late"]))
    row("relative change (early→late)", lambda d: f"{d['relative_loss_change_early_to_late']:+.1%}")
    row("largest one-step loss increase",
        lambda d: (f"+{d['largest_one_step_loss_increase']['delta']:.4f} "
                   f"({d['largest_one_step_loss_increase']['from_step']:,}→"
                   f"{d['largest_one_step_loss_increase']['to_step']:,})"))
    row("Trainer aggregate `train_loss`", lambda d: f4(d["trainer_aggregate_train_loss"]))
    row("median gradient norm", lambda d: f3g(d["median_grad_norm"]))
    row("maximum gradient norm", lambda d: f3g(d["max_grad_norm_logged"]))
    row("logged obs above clip threshold (0.1)",
        lambda d: f"{d['frac_logged_obs_above_clip_threshold']:.1%}")
    row("non-finite records", lambda d: str(d["n_nonfinite_records"]))
    row("peak learning rate (step)",
        lambda d: f"{d['max_learning_rate_logged']:.4g} ({d['step_of_max_learning_rate']:,})")
    row("expected warmup boundary",
        lambda d: f"{d['learning_rate_schedule']['expected_warmup_boundary_step']:,}")
    row("LR matches configured schedule",
        lambda d: ("yes" if d["learning_rate_schedule"]["matches_configured_schedule"]
                   else "NO" if d["learning_rate_schedule"]["matches_configured_schedule"] is False
                   else "not checked"))
    row("max relative LR deviation",
        lambda d: (f"< {d['learning_rate_schedule']['deviation_floor']:g}"
                   if d["learning_rate_schedule"].get("deviation_below_floor")
                   else f3g(d["learning_rate_schedule"].get("max_relative_deviation"))))
    row("final learning rate", lambda d: f3g(d["final_learning_rate"]))
    row("train runtime (s)", lambda d: "—" if d["train_runtime_s"] is None
        else f"{d['train_runtime_s']:,.0f}")
    row("train steps / second", lambda d: f3g(d["train_steps_per_second"]))

    L += ["", "### Reading the loss column", "",
          "Each logged `loss` is the Trainer's mean over one logging interval "
          "(10 optimizer steps at batch 1 / accumulation 1), so the **final "
          "logged loss is one interval mean, not the run aggregate**. The "
          "Trainer's `train_loss` is the mean over all intervals; both are "
          "reported above and they are not interchangeable. With one training "
          "example per step the per-interval loss is extremely noisy, so early "
          "vs late medians — not any single point — carry the signal, and no "
          "convergence claim is made from a single log point.", ""]

    L += ["## What these curves do and do not license", "",
          "**May be read from these curves:** whether logged training loss "
          "generally decreased; whether the two seeds trace similar "
          "trajectories; whether either seed shows large spikes or numerical "
          "instability; how the gradient norm evolved; whether clipping was "
          "frequently active; whether the recorded learning-rate schedule "
          "matches the configured one; runtime and throughput.", "",
          "**May NOT be read from these curves:** that full SFT reduced λ; that "
          "full SFT beats GRPO; that step 6,000 is the best full-run "
          "checkpoint; that the full seed-1 result equals the pilot result; "
          "that the model generalizes; that preferences are preserved; that "
          "training loss selects a checkpoint.", ""]

    obs = summary.get("observations", [])
    if obs:
        L += ["## Observations (descriptive)", ""] + [f"- {o}" for o in obs] + [""]

    b = summary.get("behavioral", {})
    L += ["## Behavioral / structural trajectory", "", b.get("statement", ""), ""]
    if b.get("full_trajectory", {}).get("available"):
        ft = b["full_trajectory"]
        base = b.get("base_comparator") or {}
        L += [f"*{ft['label']}*", "",
              f"Identity and completeness verified per checkpoint by "
              f"`eval/verify_sft_grid.py` → `{ft['identity_verification']}`: "
              f"adapter hash, both perspectives, N = 9,890 paired cases, zero "
              f"parse failures, a T=1 Model-A CSV, and finite estimates with "
              f"positive standard errors. A `Model_1/` directory alone is not a "
              f"completion test.", ""]
        if base.get("available"):
            L += [f"Comparator — matched local base under the **same plain "
                  f"`baseline` prompt** (same rows, scorer and estimator; only "
                  f"training differs): λ = {base['lambda']:.3f} "
                  f"(SE {base['lambda_se']:.3f}), η = {base['eta']:.3f} "
                  f"(SE {base['eta_se']:.3f}), consistency = "
                  f"{base['consistency']:.3f}, keep-both = "
                  f"{base['keep_both']:.3f}, trade-both = "
                  f"{base['trade_both']:.3f}"
                  + (f", W = {base['W']:.3f}" if base.get("W") is not None else "")
                  + ".", ""]
        L += ["| seed | step | λ (SE) | η (SE) | consistency | keep-both | "
              "trade-both | W |",
              "|---:|---:|---|---|---:|---:|---:|---:|"]
        for r in summary["behavioral_rows"]:
            w = f"{r['W']:.3f}" if r.get("W") is not None else "—"
            L.append(f"| {r['seed']} | {r['step']:,} | "
                     f"{r['lambda']:+.3f} ({r['lambda_se']:.3f}) | "
                     f"{r['eta']:+.3f} ({r['eta_se']:.3f}) | "
                     f"{r['consistency']:.3f} | {r['keep_both']:.3f} | "
                     f"{r['trade_both']:.3f} | {w} |")
        sel = b.get("frozen_selection") or {}
        picks = ", ".join(
            f"seed {k} → step {(v or {}).get('selected_step'):,}"
            for k, v in sorted(sel.items())
            if (v or {}).get("selected_step") is not None)
        L += ["", f"Frozen checkpoint selection: {picks or 'not yet run'}. "
              "This table reports the trajectory only; nothing here selects a "
              "checkpoint.", "",
              b.get("method_comparison", ""), ""]
    elif b.get("pilot_trajectory", {}).get("available"):
        L += [f"*{b['pilot_trajectory']['label']}*", "",
              "| pilot step | λ (SE) | η (SE) | d | consistency | keep-both | "
              "trade-both | W |", "|---:|---|---|---:|---:|---:|---:|---:|"]
        for r in summary["behavioral_rows"]:
            L.append(f"| {r['step']:,} | {r['lambda']:+.3f} ({r['lambda_se']:.3f}) | "
                     f"{r['eta']:+.3f} ({r['eta_se']:.3f}) | {r['d']:.3f} | "
                     f"{r['consistency']:.3f} | {r['keep_both']:.3f} | "
                     f"{r['trade_both']:.3f} | {r['W']:.3f} |")
        L += ["", b["pilot_trajectory"]["provenance_note"], ""]
    L += [f"{PILOT_SEPARATION}", "", f"{NO_GPU_STATEMENT}", ""]
    return "\n".join(L)


def render_readme(summary: dict, manifest: dict) -> str:
    w = summary["smoothing"]
    return f"""# SFT training metrics

Native training curves for the two **completed full matched-SFT runs** (seed 1
and seed 2, cosine schedule ending at 30,000) and, separately, the exploratory
**seed-1 pilot** (cosine schedule ending at 6,000).

Generated by `eval/plot_sft_training_dynamics.py` (parser version
{manifest['refresh']['parser_version']}). The GRPO plots in
`results/training_dynamics/` are produced by a different script and are not
touched here.

## Two modes

```
python eval/plot_sft_training_dynamics.py --refresh   # raw NSCC logs -> compact snapshot
python eval/plot_sft_training_dynamics.py             # snapshot -> figures/tables
python eval/plot_sft_training_dynamics.py --check     # figures/tables agree with snapshot
```

{REPRO_STATEMENT}

`--refresh` reads gitignored NSCC sources
(`checkpoints/sft_qwen_delta_*/checkpoint-*/trainer_state.json` plus the job
logs for the end-of-run runtime line). The default render and `--check` read
only the tracked CSV/JSON snapshot in this directory and work in a clean
`git archive` extraction with no checkpoint directories present.

`--check` is tolerant of floating-point noise rather than demanding a bit-for-bit
match: `sft_training_summary.json` is compared numerically (rtol
{manifest['reproducibility_tolerances']['summary_json_rtol']:g}), relative
learning-rate deviations below
{manifest['reproducibility_tolerances']['lr_deviation_floor']:g} are canonicalised
to `0.0` before they are written, and a figure-byte mismatch is downgraded to a
warning when the running NumPy/pandas/matplotlib versions differ from the ones
recorded in the manifest. Text, CSV and Markdown outputs are still compared
byte-for-byte.

## Files

| file | contents | provenance |
|---|---|---|
| `sft_full_metrics.csv` | tidy per-log-entry rows, both full seeds, unsmoothed | raw-log-derived |
| `sft_pilot_metrics.csv` | tidy per-log-entry rows, seed-1 pilot, unsmoothed | raw-log-derived |
| `sft_full_training_curves.png` | 3-panel full-run figure (linear axes) | snapshot-rendered |
| `sft_full_training_curves_logscale.png` | log-y companion | snapshot-rendered |
| `sft_pilot_training_curves.png` | 3-panel pilot figure (linear axes) | snapshot-rendered |
| `sft_pilot_training_curves_logscale.png` | log-y companion | snapshot-rendered |
| `sft_training_summary.json` / `.md` | descriptive statistics + interpretation bounds | snapshot-rendered |
| `sft_behavioral_trajectory.csv` / `.png` | exploratory pilot λ/η/d/consistency/keep/trade/W | derived from tracked `results/causal_baseline_pilot/` |
| `sft_training_manifest.json` | sources, hashes, versions, limitations | mixed |

## What SFT logs, and what it does not

SFT natively logs **completion-only cross-entropy loss**, **gradient norm**,
**learning rate**, **training progress** (step / epoch) and, at the end of the
run, **runtime and throughput**.

SFT has **no** task reward, reward standard deviation,
zero-task-reward-advantage fraction, GRPO policy loss, KL penalty, or
group-relative advantages. No artificial SFT versions of those variables are
constructed anywhere in this directory.

## The SFT objective

```
L_SFT = - sum over answer tokens of log p_theta(target token | prompt, previous target tokens)
```

Only the assistant Yes/No completion tokens contribute; prompt tokens are masked
with label `-100`.

A falling SFT loss means the model assigns more probability to the prescribed
Yes/No completion **on training prompts**. It does **not** directly measure λ or
η, does **not** prove ownership invariance, OOD generalization, framing
robustness, preference preservation, or capability retention.

SFT cross-entropy and the GRPO policy loss optimize **different objectives**;
their numerical loss values must not be compared directly.

> {VERBATIM_BOUNDARY}

## Smoothing rule

Causal trailing rolling mean over **{w['window_obs']} observations**
(~{w['window_steps']} training steps at a {w['step_spacing']}-step logging
stride), `min_periods=1`. Only the current and preceding observations enter each
smoothed point — no future observation is ever used. The **identical window is
applied to both seeds**. Raw observations are drawn faintly beneath the smoothed
line, and every unsmoothed value is retained in the CSVs.

The learning-rate panel plots the **recorded** schedule; the configured
linear-warmup + cosine curve is overlaid only as a clearly labelled dotted
reference and never substituted for the recorded values. The expected warmup
boundary (5% of `max_steps`: 1,500 for the full runs, 300 for the pilot) is
marked.

## Pilot vs full — never merged

{PILOT_SEPARATION}

The pilot is plotted in its own figure and its own CSV. It is never overlaid as
though it were the first 6,000 steps of the full seed-1 trajectory.

## Behavioral status

{summary['behavioral']['statement']}

The only recorded SFT structural trajectory is the **exploratory seed-1 pilot**
(`sft_behavioral_trajectory.csv` / `.png`): `test_goods` **validation**, an
incomplete three-point grid (2k / 4k / 6k), Model A NLS at link scale T=1, no
frozen selector, **not a full-run result**. It is read verbatim from the tracked
`results/causal_baseline_pilot/` outputs; nothing was re-estimated here.

The evaluation directories named `Qwen-7B-SFT-qd-seed1-ckpt{{2000,4000,6000}}`
were produced from the **pilot** adapters, before the pilot directory was renamed
and the full run reused its path. They are therefore reported by the grid scan
but excluded from any full-run grid count.

## Governance

{NO_GPU_STATEMENT}

The existing SFT training manifests record `git_commit: "unknown"`
(`SFTPROV-001`), so the exact code state at training time is not recoverable
from the checkpoints themselves; the training logs and dataset manifests are the
available provenance.
"""


# ── (6) render + check ───────────────────────────────────────────────────────
def do_render(out_dir: Path, refresh_block: dict | None) -> dict:
    """Tracked snapshot -> figures, summary, README, manifest. No NSCC access."""
    dfs = load_snapshot()
    if not dfs:
        raise SystemExit("No tracked SFT snapshot found. Run --refresh on a host "
                         "with the NSCC checkpoint directories first.")

    manifest_path = F_MANIFEST
    if refresh_block is None:
        if not manifest_path.exists():
            raise SystemExit(f"{manifest_path} missing — run --refresh first.")
        refresh_block = json.loads(manifest_path.read_text())["refresh"]

    # redirect outputs when rendering into a temp dir for --check
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = {
        "full_csv": out_dir / F_FULL_CSV.name,
        "pilot_csv": out_dir / F_PILOT_CSV.name,
        "full_png": out_dir / F_FULL_PNG.name,
        "full_log_png": out_dir / F_FULL_LOG_PNG.name,
        "pilot_png": out_dir / F_PILOT_PNG.name,
        "pilot_log_png": out_dir / F_PILOT_LOG_PNG.name,
        "summary_json": out_dir / F_SUMMARY_JSON.name,
        "summary_md": out_dir / F_SUMMARY_MD.name,
        "readme": out_dir / F_README.name,
        "behav_csv": out_dir / F_BEHAV_CSV.name,
        "behav_png": out_dir / F_BEHAV_PNG.name,
        "manifest": out_dir / F_MANIFEST.name,
    }
    if out_dir != OUT:
        for key, csv in (("full_csv", F_FULL_CSV), ("pilot_csv", F_PILOT_CSV)):
            if csv.exists():
                shutil.copyfile(csv, outs[key])

    ckpt_marks = list(range(SELECTION_GRID_STRIDE, 30001, SELECTION_GRID_STRIDE))

    full = {r: dfs[r] for r in FULL_RUNS if r in dfs}
    pilot = {r: dfs[r] for r in PILOT_RUNS if r in dfs}
    if full:
        plot_runs(full, outs["full_png"], SMOOTH_WINDOW,
                  full_suptitle(SMOOTH_WINDOW, False), False, ckpt_marks)
        plot_runs(full, outs["full_log_png"], SMOOTH_WINDOW,
                  full_suptitle(SMOOTH_WINDOW, True), True, ckpt_marks)
    if pilot:
        pmarks = [s for s in ckpt_marks if s <= 6000]
        plot_runs(pilot, outs["pilot_png"], SMOOTH_WINDOW,
                  pilot_suptitle(SMOOTH_WINDOW, False), False, pmarks)
        plot_runs(pilot, outs["pilot_log_png"], SMOOTH_WINDOW,
                  pilot_suptitle(SMOOTH_WINDOW, True), True, pmarks)

    run_order = [r for r in RUNS if r in dfs]
    summary = {
        "title": "SFT training metrics — descriptive summary",
        "parser_version": PARSER_VERSION,
        "smoothing": {"method": "causal trailing rolling mean (min_periods=1)",
                      "window_obs": SMOOTH_WINDOW,
                      "step_spacing": LOG_SPACING,
                      "window_steps": SMOOTH_WINDOW * LOG_SPACING,
                      "uses_future_observations": False,
                      "identical_window_for_all_runs": True},
        "objective": ("L_SFT = - sum over answer tokens of log p_theta(target "
                      "token | prompt, previous target tokens); prompt tokens "
                      "masked with label -100 (completion-only)"),
        "sft_native_metrics": ["completion-only cross-entropy loss", "gradient norm",
                               "learning rate", "training progress (step/epoch)",
                               "runtime and throughput (end of run)"],
        "metrics_sft_does_not_have": ["task reward", "reward standard deviation",
                                      "zero-task-reward-advantage fraction",
                                      "GRPO policy loss", "KL penalty",
                                      "group-relative advantages"],
        "cross_objective_warning": ("SFT cross-entropy and the GRPO policy loss "
                                    "optimize different objectives; their "
                                    "numerical loss values must not be compared "
                                    "directly."),
        "interpretation_boundary": VERBATIM_BOUNDARY,
        "pilot_vs_full": PILOT_SEPARATION,
        "run_order": run_order,
        "runs": {r: describe_run(r, dfs[r], refresh_block["runs"].get(r, {}))
                 for r in run_order},
    }

    # descriptive observations, derived — never asserted beyond the numbers
    obs = []
    for r in run_order:
        d = summary["runs"][r]
        obs.append(
            f"`{r}`: median loss fell from {d['median_loss_first_10pct']:.4f} "
            f"(first 10% of logged training) to {d['median_loss_last_10pct']:.4f} "
            f"(last 10%), a relative change of "
            f"{d['relative_loss_change_early_to_late']:+.1%}; the raw trace is "
            f"highly dispersed (single logged points span "
            f"{d['min_logged_loss']:.3g}–{d['max_logged_loss']:.3g}).")
    fulls = [r for r in run_order if summary["runs"][r]["run_role"] == "full"]
    if len(fulls) == 2:
        a, b = (summary["runs"][r] for r in fulls)
        obs.append(
            f"The two full seeds track each other closely on the smoothed loss "
            f"summaries (late medians {a['median_loss_last_10pct']:.4f} vs "
            f"{b['median_loss_last_10pct']:.4f}); this is a similarity of "
            f"optimization trajectories only, not of behavior.")
    for r in run_order:
        d = summary["runs"][r]
        obs.append(
            f"`{r}`: gradient norm is strongly bimodal (median "
            f"{d['median_grad_norm']:.3g}, max {d['max_grad_norm_logged']:.3g}); "
            f"{d['frac_logged_obs_above_clip_threshold']:.1%} of LOGGED steps "
            f"exceed the configured clip threshold "
            f"{d['configured_max_grad_norm']}, so clipping was frequently "
            f"active. Expected at batch size 1: each step's norm comes from a "
            f"single example, and an already-correct Yes/No target yields a "
            f"near-zero gradient.")
        d["n_nonfinite_records"] == 0 and obs.append(
            f"`{r}`: no non-finite loss / gradient-norm / learning-rate record.")
        lrs = d["learning_rate_schedule"]
        if lrs.get("matches_configured_schedule"):
            dev = (f"below the {lrs['deviation_floor']:g} machine-precision floor"
                   if lrs.get("deviation_below_floor")
                   else f"{lrs['max_relative_deviation']:.3g}")
            obs.append(
                f"`{r}`: the recorded learning rate reproduces the configured "
                f"linear-warmup + cosine schedule to a maximum relative deviation "
                f"{dev}; peak at step {d['step_of_max_learning_rate']:,}, "
                f"consistent with the "
                f"{lrs['expected_warmup_boundary_step']:,}-step warmup boundary.")
        if d["train_runtime_s"]:
            obs.append(
                f"`{r}`: {d['train_runtime_s']:,.0f} s wall clock "
                f"({d['train_runtime_s']/3600:.2f} h) at "
                f"{d['train_steps_per_second']:.3g} steps/s.")
    summary["observations"] = obs

    # Machine-readable record of what the linear figures crop, per run.
    crop: dict = {}
    for col, (lo, hi) in PANEL_YLIM.items():
        per_run = {}
        for r in run_order:
            v = dfs[r][col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            per_run[r] = {"n_outside": int(((v < lo) | (v > hi)).sum()),
                          "n_total": int(v.size),
                          "max_observed": float(v.max()) if v.size else None}
        crop[col] = {"ylim": [lo, hi], "by_run": per_run}
    summary["axis_crop_points_outside"] = crop

    behav_df, behav_status = build_behavioural(
        refresh_block.get("full_behavioral_grid_scan"))
    summary["behavioral"] = behav_status
    if behav_df is not None:
        behav_df.to_csv(outs["behav_csv"], index=False)
        plot_behavioural(behav_df, outs["behav_png"], behav_status)
        summary["behavioral_rows"] = behav_df.drop(columns=["d_recomputed"]) \
            .to_dict(orient="records")

    write_json(outs["summary_json"], summary)
    outs["summary_md"].write_text(render_summary_md(summary))

    manifest = {
        "title": "SFT training metrics — provenance manifest",
        "generated_by": "eval/plot_sft_training_dynamics.py",
        "parser_version": PARSER_VERSION,
        "script_sha256": sha256_of(Path(__file__).resolve()),
        "git_commit_at_render": git_commit(),
        "reproducibility_statement": REPRO_STATEMENT,
        "no_gpu_statement": NO_GPU_STATEMENT,
        "pilot_vs_full": PILOT_SEPARATION,
        "interpretation_boundary": VERBATIM_BOUNDARY,
        "known_provenance_limitations": [
            "SFTPROV-001: the existing SFT training manifests "
            "(checkpoints/sft_qwen_delta_*/sft_dataset_manifest.json) record "
            "git_commit: \"unknown\", so the exact training-time code state is "
            "not recoverable from the checkpoints.",
            "checkpoint-N/trainer_state.json carries no train_runtime; runtime "
            "and throughput come from the gitignored job log and are matched to "
            "a run by its final epoch.",
            "The pilot's sft_dataset_manifest.json still records the pre-rename "
            "output_dir checkpoints/sft_qwen_delta_seed1; its max_steps=6000 and "
            "the *_pilot6k path are the authoritative identity.",
            "Raw trainer_state.json sources and job logs are gitignored, so the "
            "logged rows can be re-derived only on NSCC.",
            "git_commit_at_render / git_commit_at_extraction necessarily name the "
            "commit that was checked out when this file was written, i.e. the "
            "PARENT of the commit that contains it — a manifest cannot record its "
            "own commit hash. --check is unaffected: it reuses the tracked refresh "
            "block rather than recomputing these fields.",
        ],
        "smoothing": summary["smoothing"],
        "axis_cropping": {
            "applies_to": "the LINEAR training-metric figures only",
            "limits": {k: list(v) for k, v in PANEL_YLIM.items()},
            "rationale": (
                "Loss and gradient norm have long thin tails (loss to ~2.76, "
                "gradient norm to ~543) that compress the bulk of the data into "
                "a sliver under autoscale. Cropping is a display choice only: no "
                "value is altered or removed, the CSVs retain every observation, "
                "the log-scale companion figures show the full range, and each "
                "cropped panel states how many logged points fall outside."),
            "points_outside": summary.get("axis_crop_points_outside", {}),
        },
        "reproducibility_tolerances": {
            "summary_json_rtol": CHECK_RTOL,
            "summary_json_atol": CHECK_ATOL,
            "lr_deviation_floor": LR_DEVIATION_FLOOR,
            "lr_match_tolerance": LR_MATCH_TOLERANCE,
            "note": ("--check compares sft_training_summary.json numerically, not "
                     "byte-wise, and canonicalises sub-floor learning-rate "
                     "deviations to 0.0, so a ~1e-16 difference between NumPy/BLAS "
                     "builds does not fail an otherwise identical render. Figure "
                     "bytes are compared exactly but downgraded to a warning when "
                     "the environment differs from the recorded one; CSV and "
                     "Markdown outputs are always compared byte-for-byte."),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "refresh": refresh_block,
        "render": {
            "behavioral": behav_status,
            "outputs": {},
        },
    }
    outs["readme"].write_text(render_readme(summary, manifest))

    provenance = {
        F_FULL_CSV.name: "raw-log-derived",
        F_PILOT_CSV.name: "raw-log-derived",
        F_FULL_PNG.name: "snapshot-rendered",
        F_FULL_LOG_PNG.name: "snapshot-rendered",
        F_PILOT_PNG.name: "snapshot-rendered",
        F_PILOT_LOG_PNG.name: "snapshot-rendered",
        F_SUMMARY_JSON.name: "snapshot-rendered",
        F_SUMMARY_MD.name: "snapshot-rendered",
        F_README.name: "snapshot-rendered",
        F_BEHAV_CSV.name: "derived from tracked results/causal_baseline_pilot/",
        F_BEHAV_PNG.name: "derived from tracked results/causal_baseline_pilot/",
    }
    for name, kind in provenance.items():
        p = out_dir / name
        if p.exists():
            manifest["render"]["outputs"][name] = {
                "sha256": sha256_of(p), "size_bytes": p.stat().st_size,
                "provenance": kind}
    write_json(outs["manifest"], manifest)

    print(f"[render] wrote {len(manifest['render']['outputs'])} artifacts to "
          f"{out_dir}")
    for name in sorted(manifest["render"]["outputs"]):
        print(f"         {name}")
    return manifest


def do_check() -> int:
    """Verify the tracked figures/tables agree with the tracked snapshot."""
    if not F_MANIFEST.exists():
        print("[check] FAIL: no manifest — run --refresh then a plain render.")
        return 1
    tracked = json.loads(F_MANIFEST.read_text())
    fails, warns = [], []

    env_now = {"python": sys.version.split()[0], "numpy": np.__version__,
               "pandas": pd.__version__, "matplotlib": matplotlib.__version__}
    env_rec = tracked.get("environment", {})
    env_match = env_now == env_rec
    if not env_match:
        warns.append(f"environment differs from the recorded one "
                     f"(recorded {env_rec}, current {env_now}); byte-identical "
                     f"figure bytes are not guaranteed across versions")

    # 1. tracked CSVs must hash to what the manifest recorded
    for name in (F_FULL_CSV.name, F_PILOT_CSV.name):
        rec = tracked["render"]["outputs"].get(name)
        p = OUT / name
        if rec is None or not p.exists():
            fails.append(f"{name}: missing from manifest or from disk")
            continue
        if sha256_of(p) != rec["sha256"]:
            fails.append(f"{name}: sha256 does not match the manifest")

    # 2. re-render everything from the tracked snapshot into a temp dir
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fresh = do_render(tmp, tracked["refresh"])
        for name, rec in tracked["render"]["outputs"].items():
            p = tmp / name
            if not p.exists():
                fails.append(f"{name}: re-render did not produce it")
                continue
            if name.endswith(".png"):
                same = sha256_of(p) == rec["sha256"]
                if not same:
                    (fails if env_match else warns).append(
                        f"{name}: re-rendered bytes differ from the tracked figure")
            elif name == F_SUMMARY_JSON.name:
                pass    # compared numerically below, not byte-wise
            else:
                if not filecmp.cmp(p, OUT / name, shallow=False):
                    fails.append(f"{name}: re-rendered content differs from tracked")
        # 3. the descriptive statistics must be reproducible from the CSVs alone.
        # Compared with float tolerance (rtol=CHECK_RTOL): a bit-for-bit match of
        # every derived statistic is not portable across NumPy/BLAS builds, and a
        # ~1e-16 wobble is not a reproducibility failure.
        a = json.loads((tmp / F_SUMMARY_JSON.name).read_text())
        b = json.loads(F_SUMMARY_JSON.read_text())
        diffs = numerically_equal(a, b)
        if diffs:
            fails.append(f"sft_training_summary.json: recomputed statistics differ "
                         f"beyond rtol={CHECK_RTOL:g} at "
                         f"{', '.join(diffs[:5])}"
                         + (f" (+{len(diffs) - 5} more)" if len(diffs) > 5 else ""))
        for k in ("smoothing", "refresh"):
            d = numerically_equal(fresh.get(k), tracked.get(k))
            if d:
                fails.append(f"manifest.{k}: differs after re-render at "
                             f"{', '.join(d[:3])}")

    for w in warns:
        print(f"[check] WARN  {w}")
    for f in fails:
        print(f"[check] FAIL  {f}")
    if fails:
        print(f"[check] FAILED ({len(fails)} problem(s))")
        return 1
    n = len(tracked["render"]["outputs"])
    print(f"[check] PASS — {n} tracked artifacts agree with the tracked snapshot; "
          f"no NSCC checkpoint or trainer_state.json was read.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SFT training metrics: refresh the compact snapshot from raw "
                    "NSCC logs, render figures/tables from the tracked snapshot, "
                    "or check that they agree.")
    ap.add_argument("--refresh", action="store_true",
                    help="re-read the raw NSCC trainer_state.json sources and "
                         "rewrite the compact tracked snapshot (then render)")
    ap.add_argument("--check", action="store_true",
                    help="verify tracked figures/tables against the tracked "
                         "snapshot; reads no NSCC source")
    ap.add_argument("--seed1-state", default=None,
                    help="explicit path to the full seed-1 trainer_state.json")
    ap.add_argument("--seed2-state", default=None,
                    help="explicit path to the full seed-2 trainer_state.json")
    ap.add_argument("--pilot-state", default=None,
                    help="explicit path to the seed-1 pilot trainer_state.json")
    args = ap.parse_args()

    if args.check and args.refresh:
        ap.error("--check and --refresh are mutually exclusive")
    if args.check:
        return do_check()

    refresh_block = do_refresh(args) if args.refresh else None
    do_render(OUT, refresh_block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
