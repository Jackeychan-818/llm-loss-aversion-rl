#!/usr/bin/env python3
"""First-2,000-step training curves for the three Qwen-own-delta GRPO runs.

Zooms into the early window that the 2k-30k @ 2k structural grid cannot resolve.
Two layers, kept strictly separate (same convention as plot_training_dynamics.py):

  (A) GRPO TRAINING optimization — reward, reward_std, KL from the reference
      policy, entropy, zero task-reward advantage fraction, grad norm, loss, lr.
      Logged every 10 steps, so the 0-2k window already holds 200 points per run.
      Read from the committed results/training_dynamics/grpo_metrics_seed*.csv,
      falling back to each run's trainer_state.json log_history.
      These are NOISY TRAINING DIAGNOSTICS, never results.

  (B) STRUCTURAL NLS estimation — lambda, eta, and alpha / beta / utility
      Spearman vs the exact local step-0 base. One point per EVALUATED
      checkpoint. Sub-2k adapters exist (200..1800, saved every 200 steps) but
      are only evaluated once train/submit_eval_early_ckpt.pbs has been run;
      this script reports exactly which are missing and the qsub lines to get
      them, and NEVER fabricates or interpolates a missing point.

Scientific guards:
  * Everything below step 2,000 is EXPLORATORY and NON-GATING. The frozen
    confirmatory selection grid is 2k-30k @ 2k (selected: seed1 -> 2,000,
    seed2 -> 6,000). Nothing plotted here may inform checkpoint selection.
  * The logged `reward` is a CONDITIONAL statistic: zero-diversity groups are
    recorded as 0.0 rather than their true +/-|delta|. It is therefore always
    plotted with frac_reward_zero_std directly beneath it. See KNOWN_ISSUES.md #4.
  * `kl` is the GRPO KL from the reference policy (coefficient beta=0.04) — a
    training-time quantity on training prompts. Structural beta is an
    attribute-profile effect. Different quantities; never merge the panels.

Outputs under results/training_dynamics/early_2k/:
  early_training_curves.png   (A)
  early_structural.png        (B, only when >=1 sub-2k evaluation exists)
  early_metrics.csv           (A, tidy: seed, step, metric columns)
  early_manifest.json         (inputs, sha256, what is missing and how to get it)

Run:
    module load pytorch/...; source .../venv/bin/activate
    python eval/plot_early_training.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "training_dynamics" / "early_2k"
SRC = PROJECT_ROOT / "results" / "training_dynamics"

MAX_STEP = 2000                      # the early window, inclusive
EARLY_ADAPTER_STEPS = list(range(200, 2000, 200))   # 200 .. 1800
LADDER = [200, 400, 800, 1400, 1800]                # cheaper log-spaced subset

RUNS = {  # seed -> (committed compact csv, trainer_state fallback, label, colour)
    "seed1":  (SRC / "grpo_metrics_seed1.csv",
               PROJECT_ROOT / "checkpoints/grpo_qwen_delta_seed1/checkpoint-30000/trainer_state.json",
               "seed 1", "#2c5f8a"),
    "seed2":  (SRC / "grpo_metrics_seed2.csv",
               PROJECT_ROOT / "checkpoints/grpo_qwen_delta_seed2/checkpoint-30000/trainer_state.json",
               "seed 2", "#2a7a2a"),
    "seed42": (SRC / "grpo_metrics_seed42.csv",
               PROJECT_ROOT / "checkpoints/grpo_qwen_delta/checkpoint-41200/trainer_state.json",
               "seed 42 (original)", "#b3541e"),
}
RUN_DIR = {"seed1": "checkpoints/grpo_qwen_delta_seed1",
           "seed2": "checkpoints/grpo_qwen_delta_seed2",
           "seed42": "checkpoints/grpo_qwen_delta"}
SEED_ARG = {"seed1": "1", "seed2": "2", "seed42": "42"}

BASE_MODEL = "Qwen-7B-Base-Local"    # exact local step-0 comparator


# ── provenance ────────────────────────────────────────────────────────────────
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


# ── (A) training metrics, first 2k steps ──────────────────────────────────────
def load_early_metrics() -> tuple[pd.DataFrame, dict, list]:
    frames, inputs, notes = [], {}, []
    for seed, (compact, state, _label, _c) in RUNS.items():
        if compact.exists():
            df = pd.read_csv(compact)
            src, path = "committed_compact_csv", compact
        elif state.exists():
            hist = json.load(open(state)).get("log_history", [])
            df = pd.DataFrame([h for h in hist if "step" in h])
            src, path = "trainer_state.json", state
        else:
            notes.append(f"{seed}: no metrics source found "
                         f"({compact.name} and {state} both missing)")
            continue
        df = df[df["step"] <= MAX_STEP].copy()
        df.insert(0, "seed", seed)
        frames.append(df)
        inputs[f"metrics:{seed}"] = {"path": str(path.relative_to(PROJECT_ROOT)),
                                     "sha256": sha256_of(path), "parser": src,
                                     "rows_in_window": int(len(df))}
    if not frames:
        raise SystemExit("No training metrics available for any run.")
    return pd.concat(frames, ignore_index=True), inputs, notes


# ── (B) structural points below 2k ────────────────────────────────────────────
def load_nls(csv_path: Path) -> dict:
    import csv as _csv
    return {r["Parameter"]: float(r["Estimate"]) for r in _csv.DictReader(open(csv_path))}


def find_nls_csv(model_dir: Path) -> Path | None:
    hits = glob.glob(str(model_dir / "Model_1" / "*NLS_estimation*.csv"))
    return Path(sorted(hits)[0]) if hits else None


def param_block(p: dict, prefix: str) -> dict:
    """Named parameters for one block, excluding the fixed reference levels."""
    return {k: v for k, v in p.items()
            if k.startswith(prefix) and k not in ("alpha_1", "beta_1,1")}


def spearman_vs_base(p: dict, base: dict, prefix: str) -> float:
    """Tie-aware rank correlation, aligning parameters BY NAME (not row order)."""
    from scipy.stats import spearmanr
    a, b = param_block(p, prefix), param_block(base, prefix)
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return float("nan")
    return float(spearmanr([a[k] for k in keys], [b[k] for k in keys]).statistic)


def utility_spearman(p: dict, base: dict) -> float:
    """Spearman on fitted utilities U = exp(alpha + beta) over shared names."""
    from scipy.stats import spearmanr
    pa, ba = param_block(p, "alpha"), param_block(base, "alpha")
    keys = sorted(set(pa) & set(ba))
    if len(keys) < 3:
        return float("nan")
    return float(spearmanr([math.exp(pa[k]) for k in keys],
                           [math.exp(ba[k]) for k in keys]).statistic)


def load_early_structural() -> tuple[pd.DataFrame, dict, list]:
    """Collect step-0 base + every EVALUATED sub-2k checkpoint. Never interpolates."""
    inputs, rows, notes = {}, [], []

    base_csv = find_nls_csv(PROJECT_ROOT / "baseline" / BASE_MODEL)
    base = None
    if base_csv is None:
        notes.append(f"step-0 base {BASE_MODEL} has no Model-A NLS CSV; "
                     "structural panel will have no anchor.")
    else:
        base = load_nls(base_csv)
        inputs["structural:base"] = {"path": str(base_csv.relative_to(PROJECT_ROOT)),
                                     "sha256": sha256_of(base_csv)}
        rows.append({"seed": "base", "step": 0, "model_name": BASE_MODEL,
                     "lambda": base["lambda"], "eta": base["eta"],
                     "spearman_alpha": 1.0, "spearman_beta": 1.0,
                     "spearman_utility": 1.0})

    found = {s: set() for s in RUNS}
    for feature in ("diagnostics/early", "diagnostics/step600"):
        for d in sorted(glob.glob(str(PROJECT_ROOT / feature / "*"))):
            name = Path(d).name
            m = re.search(r"-step(\d+)-EXPLORATORY", name)
            step = int(m.group(1)) if m else (600 if "step600" in feature else None)
            if step is None or step > MAX_STEP:
                continue
            seed = "seed1" if "seed1" in name else "seed2" if "seed2" in name else "seed42"
            csv = find_nls_csv(Path(d))
            if csv is None:
                notes.append(f"{name}: inference present but no NLS CSV — "
                             f"estimation stage did not finish.")
                continue
            if base is None:
                continue
            p = load_nls(csv)
            inputs[f"structural:{name}"] = {"path": str(csv.relative_to(PROJECT_ROOT)),
                                            "sha256": sha256_of(csv)}
            found[seed].add(step)
            rows.append({"seed": seed, "step": step, "model_name": name,
                         "lambda": p["lambda"], "eta": p["eta"],
                         "spearman_alpha": spearman_vs_base(p, base, "alpha"),
                         "spearman_beta": spearman_vs_base(p, base, "beta"),
                         "spearman_utility": utility_spearman(p, base)})

    missing = {s: [k for k in EARLY_ADAPTER_STEPS if k not in found[s]] for s in RUNS}
    df = (pd.DataFrame(rows).sort_values(["seed", "step"]).reset_index(drop=True)
          if rows else pd.DataFrame(columns=["seed", "step", "model_name", "lambda", "eta",
                                             "spearman_alpha", "spearman_beta",
                                             "spearman_utility"]))
    return df, inputs, notes, missing


def qsub_lines(missing: dict, ladder_only: bool = False) -> list[str]:
    out = []
    for seed, steps in missing.items():
        steps = [s for s in steps if (s in LADDER)] if ladder_only else steps
        if not steps:
            continue
        out.append(f"# {seed} ({RUN_DIR[seed]})")
        out.append(f"for S in {' '.join(str(s) for s in steps)}; do "
                   f"qsub -v SEED={SEED_ARG[seed]},STEPS=$S train/submit_eval_early_ckpt.pbs; done")
    return out


# ── plotting ──────────────────────────────────────────────────────────────────
def smooth(y: np.ndarray, w: int = 5) -> np.ndarray:
    """Centered moving average with shrinking end windows.

    np.convolve(mode='same') zero-pads the ends, which fabricates a dip at
    step 0 and at the right edge — exactly where the early trajectory is most
    interesting. A min_periods rolling mean averages over whatever is there.
    """
    if len(y) < w:
        return y
    return pd.Series(y).rolling(w, center=True, min_periods=1).mean().to_numpy()


def plot_training(df: pd.DataFrame, path: Path) -> None:
    panels = [
        ("reward",               "mean reward  (CONDITIONAL — see below)", None),
        ("frac_reward_zero_std", "zero task-reward advantage fraction", (0, 1.05)),
        ("kl",                   "KL from reference policy  (beta=0.04)", None),
        ("entropy",              "entropy", None),
        ("reward_std",           "reward std (within-group)", None),
        ("grad_norm",            "gradient norm", None),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11.5))
    fig.suptitle("GRPO training dynamics — first 2,000 steps, Qwen-own-delta runs\n"
                 "EXPLORATORY / NON-GATING: the frozen selection grid starts at step 2,000",
                 fontsize=12.5, y=0.985)

    for ax, (col, title, ylim) in zip(axes.ravel(), panels):
        for seed, (_c, _s, label, colour) in RUNS.items():
            d = df[df["seed"] == seed]
            if col not in d or d[col].isna().all():
                continue
            x, y = d["step"].to_numpy(), d[col].to_numpy(dtype=float)
            ax.plot(x, y, color=colour, alpha=0.22, linewidth=0.9)
            ax.plot(x, smooth(y), color=colour, linewidth=1.9, label=label)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("training step")
        ax.set_xlim(0, MAX_STEP)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.axvline(2000, color="grey", linestyle="--", linewidth=1)
        for s in EARLY_ADAPTER_STEPS:
            ax.axvline(s, color="grey", alpha=0.13, linewidth=0.6)
    axes[0, 0].legend(fontsize=9, loc="best")
    axes[0, 0].axhline(0, color="black", linewidth=0.8)

    fig.text(0.5, 0.012,
             "Faint line = raw (logged every 10 steps); solid = 5-point moving average. "
             "Light verticals = saved adapters (every 200 steps); dashed = first frozen grid point.\n"
             "Mean reward is a CONDITIONAL statistic: zero-diversity groups are logged as 0.0, "
             "not their true +/-|delta| — read it against the panel beside it.",
             ha="center", fontsize=8.6, color="#444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_structural(df: pd.DataFrame, path: Path, missing: dict) -> None:
    panels = [("lambda", "lambda (loss aversion)"),
              ("eta", "eta (status-quo bias)"),
              ("spearman_alpha", "alpha Spearman vs local base"),
              ("spearman_beta", "beta Spearman vs local base")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Structural parameters — first 2,000 steps (EXPLORATORY / NON-GATING)",
                 fontsize=12.5)
    base = df[df["seed"] == "base"]
    for ax, (col, title) in zip(axes.ravel(), panels):
        for seed, (_c, _s, label, colour) in RUNS.items():
            d = df[(df["seed"] == seed)].sort_values("step")
            if d.empty:
                continue
            # prepend the shared step-0 anchor so each run's curve starts at base
            x = np.concatenate([base["step"].to_numpy(), d["step"].to_numpy()])
            y = np.concatenate([base[col].to_numpy(dtype=float), d[col].to_numpy(dtype=float)])
            ax.plot(x, y, "o-", color=colour, linewidth=1.8, markersize=5, label=label)
        if not base.empty:
            ax.plot(base["step"], base[col].astype(float), "k*", markersize=13,
                    label="step 0 (local base)" if col == "lambda" else None)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("training step")
        ax.set_xlim(-40, MAX_STEP + 40)
        ax.grid(alpha=0.25, linewidth=0.6)
        for s in EARLY_ADAPTER_STEPS:
            ax.axvline(s, color="grey", alpha=0.13, linewidth=0.6)
    axes[0, 0].legend(fontsize=9, loc="best")

    n_missing = sum(len(v) for v in missing.values())
    if n_missing:
        fig.text(0.5, 0.012,
                 f"INCOMPLETE: {n_missing} sub-2k checkpoints are saved but not yet "
                 f"evaluated — no point is interpolated for them.\n"
                 f"Fill in with: qsub -v SEED=<1|2|42>,STEPS=<step> train/submit_eval_early_ckpt.pbs",
                 ha="center", fontsize=8.8, color="#8a2c2c")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notes = []

    metrics, m_inputs, m_notes = load_early_metrics()
    notes += m_notes
    struct, s_inputs, s_notes, missing = load_early_structural()
    notes += s_notes

    metrics.to_csv(OUT / "early_metrics.csv", index=False)
    plot_training(metrics, OUT / "early_training_curves.png")
    print(f"[A] training curves: {len(metrics)} rows "
          f"({', '.join(f'{s}={int((metrics.seed == s).sum())}' for s in RUNS)}) "
          f"-> {OUT / 'early_training_curves.png'}")

    evaluated = struct[struct["seed"] != "base"]
    if not evaluated.empty:
        struct.to_csv(OUT / "early_structural.csv", index=False)
        plot_structural(struct, OUT / "early_structural.png", missing)
        print(f"[B] structural: {len(evaluated)} sub-2k checkpoints evaluated "
              f"-> {OUT / 'early_structural.png'}")
    else:
        print("[B] structural: NO sub-2k checkpoint has been evaluated yet — "
              "panel skipped (nothing is interpolated).")

    n_missing = sum(len(v) for v in missing.values())
    if n_missing:
        print(f"\n{n_missing} saved sub-2k adapters are not yet evaluated.")
        print("Full set (9 per run, 27 jobs):")
        for line in qsub_lines(missing):
            print("  " + line)
        print("Log-spaced ladder (5 per run, 15 jobs):")
        for line in qsub_lines(missing, ladder_only=True):
            print("  " + line)

    manifest = {
        "generated_by": "eval/plot_early_training.py",
        "git_commit": git_commit(),
        "window": {"max_step": MAX_STEP, "adapter_steps": EARLY_ADAPTER_STEPS},
        "status": "EXPLORATORY / NON-GATING — below the frozen 2k-30k @ 2k grid",
        "inputs": {**m_inputs, **s_inputs},
        "structural_missing": {s: v for s, v in missing.items() if v},
        "notes": notes,
    }
    (OUT / "early_manifest.json").write_text(json.dumps(manifest, indent=2))
    for n in notes:
        print(f"NOTE: {n}")


if __name__ == "__main__":
    main()
