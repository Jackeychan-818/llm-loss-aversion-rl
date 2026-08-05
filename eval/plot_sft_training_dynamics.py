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
import filecmp
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PARSER_VERSION = "1.0.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "training_dynamics" / "sft"

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
    "CPU-only. No GPU/PBS job, no inference, no checkpoint evaluation, no "
    "frozen selector run, and no frozen/untouched suite was opened to produce "
    "these artifacts."
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
    fig.subplots_adjust(top=0.815)
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
    """Read-only filesystem scan for an already-evaluated FULL SFT grid.

    A working-tree observation, so it runs in --refresh (the mode allowed to
    look at the machine) and is recorded in the manifest; render and --check
    then consume the recorded scan, which keeps them snapshot-reproducible.

    Directory NAMES are not identity. The seed-1 2k/4k/6k evaluation directories
    were produced on 2026-07-27 from the PILOT adapters, back when the pilot
    still occupied checkpoints/sft_qwen_delta_seed1 (renamed to *_pilot6k before
    the full run reused that path on 2026-07-29). They are therefore listed but
    excluded from the full-run grid count.
    """
    grid_steps = list(range(SELECTION_GRID_STRIDE, 30001, SELECTION_GRID_STRIDE))
    found: dict[int, list[int]] = {}
    for seed, prefix in FULL_GRID_PREFIX.items():
        hits = []
        for root in FULL_GRID_ROOTS:
            base = PROJECT_ROOT / root
            if not base.is_dir():
                continue
            for step in grid_steps:
                if (base / f"{prefix}{step}" / "Model_1").is_dir():
                    hits.append(step)
        found[seed] = sorted(set(hits))
    known_pilot = {1: [2000, 4000, 6000]}
    attributable = {s: [k for k in v if k not in known_pilot.get(s, [])]
                    for s, v in found.items()}
    n_needed = len(grid_steps) * len(FULL_GRID_PREFIX)
    n_attr = sum(len(v) for v in attributable.values())
    return {
        "scanned_roots": FULL_GRID_ROOTS,
        "dir_prefixes": {str(k): v for k, v in FULL_GRID_PREFIX.items()},
        "required": {"seeds": list(FULL_GRID_PREFIX), "checkpoints": grid_steps,
                     "n_evaluations_required": n_needed},
        "directories_found": {str(k): v for k, v in found.items()},
        "excluded_as_pilot_evaluations": {str(k): v for k, v in known_pilot.items()},
        "attributable_to_full_runs": {str(k): v for k, v in attributable.items()},
        "n_attributable_to_full_runs": n_attr,
        "complete": n_attr == n_needed,
        "identity_verified": False,
    }


def build_behavioural(grid_scan: dict | None) -> tuple[pd.DataFrame | None, dict]:
    """Assemble the ONLY already-evaluated SFT structural trajectory.

    Read-only: no inference, no estimator run, no selector. Uses the recorded
    full-grid scan and reports honestly when the grid is absent.
    """
    scan = grid_scan or {"complete": False, "note": "no grid scan recorded"}
    grid_complete = bool(scan.get("complete"))

    status = {
        "full_grid_scan": scan,
        "full_grid_complete": grid_complete,
        "frozen_selector_run_for_sft": False,
        "statement": (
            "The two full SFT runs have completed training, but their "
            "behavioral checkpoint grid has not been evaluated. Therefore, no "
            "full-run lambda/eta trajectory is available."
            if not grid_complete else
            "A complete 2-seed x 15-checkpoint SFT grid was located; it is "
            "plotted read-only and no selector was run."),
    }

    if grid_complete:
        # Deliberately not implemented as an implicit path: a complete grid is a
        # project-state change that must be wired in explicitly, with checkpoint
        # identity / adapter hash / estimator verification, rather than picked up
        # silently by a plotting script.
        status["note"] = ("Complete grid detected — extend this function with "
                          "explicit per-checkpoint identity verification before "
                          "plotting it. Nothing was fabricated.")
        return None, status

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
        "provenance_note": (
            "The July-27 evaluation logs record the adapter path as "
            "checkpoints/sft_qwen_delta_seed1/checkpoint-N. The pilot trained "
            "into that path on July 26 and was renamed to *_pilot6k before the "
            "full run reused the name on July 29, so those logs refer to the "
            "PILOT adapters. Directory mtimes and the pilot's own manifest "
            "(max_steps=6000) confirm the identity."),
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


def plot_behavioural(df: pd.DataFrame, path: Path) -> None:
    panels = [("lambda", "λ  (loss aversion)", True),
              ("eta", "η  (status-quo bias)", True),
              ("d", "d = sqrt(λ² + η²)", False),
              ("consistency", "paired consistency", False),
              ("keep_both", "keep-both fraction", False),
              ("trade_both", "trade-both fraction", False),
              ("W", "W", False)]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    c = RUNS["sft_pilot6k_seed1"]["color"]
    for ax, (col, title, zero) in zip(axes.ravel(), panels):
        if col in ("lambda", "eta"):
            se = df[f"{col}_se"]
            ax.errorbar(df["step"], df[col], yerr=se, color=c, marker="o",
                        lw=1.8, ms=5, capsize=3)
        else:
            ax.plot(df["step"], df[col], color=c, marker="o", lw=1.8, ms=5)
        if zero:
            ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("pilot training step")
        ax.set_xticks(sorted(df["step"]))
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(
        0.0, 0.5,
        "EXPLORATORY seed-1 PILOT\n"
        "test_goods = VALIDATION\n"
        "incomplete 3-point grid\n"
        "no frozen selector run\n"
        "NOT a full-run result\n\n"
        "Model A NLS, link scale T=1\n"
        "N = 9,890 cases per point",
        fontsize=9.5, va="center", ha="left", color="#7a2020")
    fig.suptitle(
        "SFT behavioral trajectory — EXPLORATORY seed-1 pilot (cosine -> 6,000), "
        "test_goods VALIDATION, incomplete three-point grid, no frozen selector; "
        "not a full-run result.\n"
        "Structural estimands computed after inference — separate from, and not "
        "implied by, the SFT training-metric curves.", fontsize=11, y=0.985,
        va="top")
    fig.tight_layout()
    fig.subplots_adjust(top=0.885)
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
    if b.get("pilot_trajectory", {}).get("available"):
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

    behav_df, behav_status = build_behavioural(
        refresh_block.get("full_behavioral_grid_scan"))
    summary["behavioral"] = behav_status
    if behav_df is not None:
        behav_df.to_csv(outs["behav_csv"], index=False)
        plot_behavioural(behav_df, outs["behav_png"])
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
