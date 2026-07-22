#!/usr/bin/env python3
"""Training-process curves for the Qwen-own-delta GRPO replication runs.

Two clearly separated layers (never conflated):

  (A) GRPO TRAINING optimization — reward, reward_std, DAPO/policy loss, KL from
      base, entropy, zero-reward/DAPO filtering fraction, gradient norm, learning
      rate. Parsed from trainer_state.json log_history (explicit training steps),
      with a text-log fallback. These are NOISY TRAINING DIAGNOSTICS.

  (B) STRUCTURAL NLS estimation — lambda, eta, d=sqrt(lambda^2+eta^2), and the
      preservation of alpha / beta / utility (U=exp(alpha+beta)) versus the EXACT
      LOCAL step-0 base (baseline/Qwen-7B-Base-Local). lambda/eta/d and the
      correlations are read from the committed Model-A NLS CSVs. RSS and Jacobian
      conditioning come ONLY from eval/estimator_diagnostics.py outputs when they
      exist; missing ones are marked unavailable (NOT fabricated) and the qsub
      command is reported.

Scientific guards:
  * The exact local step-0 comparator is baseline/Qwen-7B-Base-Local (NOT
    baseline/Qwen-7B/, which is the historical Together-API model).
  * The frozen confirmatory SELECTION grid is 2k..30k @ 2k. Selected: seed1->2000,
    seed2->6000. Early/other checkpoints are exploratory, non-gating.
  * GRPO beta=0.04 is the KL COEFFICIENT; structural beta is an attribute-profile
    effect. They are different quantities.
  * Correlations align parameters BY NAME (not row), use tie-aware ranks
    (scipy.stats.spearmanr), and exclude the reference levels alpha_1=0, beta_1,1=0.

Outputs under results/training_dynamics/:
  grpo_metrics_seed{1,2,42}.csv, grpo_training_curves.png,
  structural_trajectory.csv, structural_training_curves.png,
  parameter_correlations.csv, parameter_preservation.png, manifest.json

    module load pytorch/...; source .../venv/bin/activate
    python eval/plot_training_dynamics.py
"""
from __future__ import annotations

import ast
import csv as _csv
import glob
import hashlib
import json
import math
import subprocess
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "results" / "training_dynamics"

BASE_FEATURE, BASE_MODEL = "baseline", "Qwen-7B-Base-Local"   # exact local step 0
FROZEN_SELECTION_GRID = set(range(2000, 30001, 2000))          # confirmatory grid
SELECTED = {"seed1": 2000, "seed2": 6000}
QUASI_SEP = {"alpha_37", "alpha_51"}                           # goods 37/51
SMOOTH_WINDOW = 50   # observations (causal trailing mean); spacing is 10 steps
GRPO_METRICS = ["reward", "reward_std", "loss", "kl", "entropy",
                "frac_reward_zero_std", "grad_norm", "learning_rate"]
# TRL field aliases -> our canonical name (first present wins)
ALIASES = {
    "reward": ["reward", "rewards/reward_fn/mean"],
    "reward_std": ["reward_std", "rewards/reward_fn/std"],
    "loss": ["loss"],
    "kl": ["kl"],
    "entropy": ["entropy"],
    "frac_reward_zero_std": ["frac_reward_zero_std"],
    "grad_norm": ["grad_norm"],
    "learning_rate": ["learning_rate"],
    "epoch": ["epoch"],
}
RUNS = {  # seed -> (trainer_state, text_log)
    "seed1": ("checkpoints/grpo_qwen_delta_seed1/checkpoint-30000/trainer_state.json",
              "logs/train_qd_seed1_run.log"),
    "seed2": ("checkpoints/grpo_qwen_delta_seed2/checkpoint-30000/trainer_state.json",
              "logs/train_qd_seed2_run.log"),
    "seed42": ("checkpoints/grpo_qwen_delta/checkpoint-30000/trainer_state.json",
               "logs/train_qd_run.log"),
}
# eval-dir prefixes for the NLS structural trajectory
GRID_PREFIX = {"seed1": "Qwen-7B-GRPO-qd-seed1-ckpt",
               "seed2": "Qwen-7B-GRPO-qd-seed2-ckpt",
               "seed42": "Qwen-7B-GRPO-qd-ckpt"}
DIAG_QSUB = "qsub train/submit_diagnostics_cpu.pbs"


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


# ── (A) GRPO training metrics ─────────────────────────────────────────────────
def parse_trainer_state(path: Path) -> tuple[pd.DataFrame, dict]:
    hist = json.load(open(path)).get("log_history", [])
    rows = []
    for e in hist:
        if "step" not in e or "loss" not in e:   # keep training-step rows only
            continue
        row = {"step": int(e["step"])}
        for canon, keys in ALIASES.items():
            val = next((e[k] for k in keys if k in e and e[k] is not None), None)
            row[canon] = float(val) if val is not None else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    present = {m: bool(df[m].notna().any()) for m in GRPO_METRICS}
    return df, present


def parse_text_log(path: Path) -> tuple[pd.DataFrame, dict]:
    """Fallback: best-effort parse of TRL dict lines like {'loss':..,'step':..}."""
    rows = []
    for line in open(path, errors="ignore"):
        s = line.strip()
        i = s.find("{'")
        if i < 0 or "'loss'" not in s:
            continue
        try:
            e = ast.literal_eval(s[i:s.rfind("}") + 1])
        except Exception:
            continue
        if "step" not in e:
            continue
        row = {"step": int(e["step"])}
        for canon, keys in ALIASES.items():
            val = next((e[k] for k in keys if k in e and e[k] is not None), None)
            row[canon] = float(val) if val is not None else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True) if rows else pd.DataFrame()
    present = {m: bool(len(df) and df[m].notna().any()) for m in GRPO_METRICS}
    return df, present


def parse_compact_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read a previously-written compact metrics CSV back into the same frame."""
    df = pd.read_csv(path)
    for m in GRPO_METRICS + ["epoch"]:
        if m not in df.columns:
            df[m] = np.nan
    df = df.sort_values("step").reset_index(drop=True)
    present = {m: bool(df[m].notna().any()) for m in GRPO_METRICS}
    return df, present


def load_grpo(seed: str) -> tuple[pd.DataFrame, dict, dict]:
    ts, log = RUNS[seed]
    ts_p, log_p = PROJECT_ROOT / ts, PROJECT_ROOT / log
    compact_p = OUT / f"grpo_metrics_{seed}.csv"   # committed; regenerable source
    if ts_p.exists():
        df, present = parse_trainer_state(ts_p)
        src = {"source": str(ts_p.relative_to(PROJECT_ROOT)), "sha256": sha256_of(ts_p),
               "parser": "trainer_state.json"}
    elif compact_p.exists():
        # trainer_state.json / checkpoints are NOT committed; fall back to the
        # committed compact CSV so the figures regenerate from tracked artifacts.
        df, present = parse_compact_csv(compact_p)
        src = {"source": str(compact_p.relative_to(PROJECT_ROOT)), "sha256": sha256_of(compact_p),
               "parser": "committed_compact_csv"}
    elif log_p.exists():
        df, present = parse_text_log(log_p)
        src = {"source": str(log_p.relative_to(PROJECT_ROOT)), "sha256": sha256_of(log_p),
               "parser": "text_log_fallback"}
    else:
        return pd.DataFrame(), {m: False for m in GRPO_METRICS}, {"source": None}
    src.update({"n_observations": int(len(df)),
                "step_range": [int(df["step"].min()), int(df["step"].max())] if len(df) else None,
                "metrics_present": present})
    return df, present, src


def causal_rolling(series: pd.Series, window: int) -> pd.Series:
    """Trailing (causal) rolling mean; no zero padding (min_periods=1)."""
    return series.rolling(window=window, min_periods=1).mean()


# ── (B) structural NLS parameters ─────────────────────────────────────────────
def load_nls(csv_path: Path) -> dict:
    return {r["Parameter"]: float(r["Estimate"]) for r in _csv.DictReader(open(csv_path))}


def find_nls_csv(feature: str, model_name: str) -> Path | None:
    hits = glob.glob(str(PROJECT_ROOT / feature / model_name / "Model_1" / "*NLS_estimation*.csv"))
    return Path(hits[0]) if hits else None


def alpha_beta_keys(params: dict) -> tuple[list, list]:
    a = sorted((k for k in params if k.startswith("alpha_")),
               key=lambda k: int(k.split("_")[1]))
    b = sorted(k for k in params if k.startswith("beta_"))
    return a, b


def utility_vector(params: dict) -> tuple[np.ndarray, list]:
    """U=exp(alpha_item+beta_combo) over the full item x (3x3) grid, with the
    references alpha_1=0 and beta_1,1=0 supplied explicitly. Returns (values,
    ordered keys) so two models align element-for-element."""
    a_keys, _ = alpha_beta_keys(params)
    n_items = max(int(k.split("_")[1]) for k in a_keys) if a_keys else 1
    alpha = {1: 0.0}
    alpha.update({int(k.split("_")[1]): params[k] for k in a_keys})
    beta = {(1, 1): 0.0}
    for k in params:
        if k.startswith("beta_"):
            r, c = (int(x) for x in k.split("_")[1].split(","))
            beta[(r, c)] = params[k]
    vals, keys = [], []
    for i in range(1, n_items + 1):
        for (r, c) in product(range(1, 4), range(1, 4)):
            vals.append(math.exp(alpha.get(i, 0.0) + beta.get((r, c), 0.0)))
            keys.append(f"u_{i}_{r}{c}")
    return np.asarray(vals), keys


def _metrics(t: np.ndarray, b: np.ndarray) -> dict:
    out = {"n": int(t.size)}
    out["pearson"] = float(pearsonr(t, b)[0]) if t.size >= 2 else None
    out["spearman"] = float(spearmanr(t, b).statistic) if t.size >= 2 else None
    out["mae"] = float(np.mean(np.abs(t - b)))
    out["rmse"] = float(np.sqrt(np.mean((t - b) ** 2)))
    return out


def compare_params(target: dict, base: dict) -> dict:
    """All alignments are BY NAME. alpha/beta exclude reference levels (they are
    absent from the CSV). Also reports alpha excluding quasi-separated 37/51."""
    a_keys = [k for k in alpha_beta_keys(target)[0] if k in base]
    b_keys = [k for k in alpha_beta_keys(target)[1] if k in base]
    ta, ba = np.array([target[k] for k in a_keys]), np.array([base[k] for k in a_keys])
    tb, bb = np.array([target[k] for k in b_keys]), np.array([base[k] for k in b_keys])
    keep = [i for i, k in enumerate(a_keys) if k not in QUASI_SEP]
    tut, but_ = utility_vector(target), utility_vector(base)
    # align utility by key
    kmap = {k: v for k, v in zip(but_[1], but_[0])}
    ut_keys = [k for k in tut[1] if k in kmap]
    tu = np.array([v for k, v in zip(tut[1], tut[0]) if k in kmap])
    bu = np.array([kmap[k] for k in ut_keys])
    res = {"alpha": _metrics(ta, ba), "beta": _metrics(tb, bb), "utility": _metrics(tu, bu),
           "alpha_excl_37_51": _metrics(ta[keep], ba[keep])}
    return res


def load_diag_rss_cond() -> dict:
    """model_name -> {rss, cond_jacobian} from existing estimator_diagnostics JSONs
    (skip alternative-solution and multistart files)."""
    out = {}
    for p in glob.glob(str(PROJECT_ROOT / "results" / "estimator_diagnostics" / "*.json")):
        name = Path(p).name
        if "ALT_SOLUTION" in name or "multistart" in name:
            continue
        try:
            d = json.load(open(p))
            sol = d.get("solution_diagnostics", {})
            out[d["model_name"]] = {"rss": sol.get("rss_at_solution"),
                                    "cond_jacobian": sol.get("cond_jacobian"),
                                    "csv_sha256": d.get("committed_csv_sha256")}
        except Exception:
            continue
    return out


def build_structural(base_params: dict, base_sha: str) -> tuple[pd.DataFrame, dict, list]:
    diag = load_diag_rss_cond()
    inputs = {}
    rows = []

    def add(seed, step, feature, model_name, exploratory):
        csv = find_nls_csv(feature, model_name)
        if csv is None:
            return
        p = load_nls(csv)
        inputs[model_name] = {"path": str(csv.relative_to(PROJECT_ROOT)), "sha256": sha256_of(csv)}
        cmp = compare_params(p, base_params)
        dd = diag.get(model_name, {})
        cond = dd.get("cond_jacobian")
        rows.append({
            "seed": seed, "step": step, "model_name": model_name,
            "on_selection_grid": step in FROZEN_SELECTION_GRID,
            "selected": (SELECTED.get(seed) == step), "exploratory": exploratory,
            "lambda": p["lambda"], "eta": p["eta"],
            "d": math.hypot(p["lambda"], p["eta"]),
            "rss": dd.get("rss"),
            "log10_cond_jacobian": (math.log10(cond) if cond and cond > 0 and math.isfinite(cond) else np.nan),
            "spearman_alpha": cmp["alpha"]["spearman"],
            "spearman_beta": cmp["beta"]["spearman"],
            "spearman_utility": cmp["utility"]["spearman"],
        })

    # step-0 base row (self-comparison = 1.0)
    bl, be = base_params["lambda"], base_params["eta"]
    bd = diag.get(BASE_MODEL, {})
    bcond = bd.get("cond_jacobian")
    rows.append({"seed": "base", "step": 0, "model_name": BASE_MODEL,
                 "on_selection_grid": False, "selected": False, "exploratory": False,
                 "lambda": bl, "eta": be, "d": math.hypot(bl, be), "rss": bd.get("rss"),
                 "log10_cond_jacobian": (math.log10(bcond) if bcond and bcond > 0 and math.isfinite(bcond) else np.nan),
                 "spearman_alpha": 1.0, "spearman_beta": 1.0, "spearman_utility": 1.0})

    for seed in ("seed1", "seed2", "seed42"):
        for csv in sorted(glob.glob(str(PROJECT_ROOT / "baseline" / f"{GRID_PREFIX[seed]}*"))):
            name = Path(csv).name
            try:
                step = int(name.rsplit("ckpt", 1)[1])
            except (IndexError, ValueError):
                continue
            add(seed, step, "baseline", name, exploratory=(seed == "seed42"))
    # optional step-600 exploratory outputs (diagnostics/step600)
    for csv in sorted(glob.glob(str(PROJECT_ROOT / "diagnostics" / "step600" / "*"))):
        name = Path(csv).name
        seed = "seed1" if "seed1" in name else "seed2" if "seed2" in name else "seed42"
        add(seed, 600, "diagnostics/step600", name, exploratory=True)

    df = pd.DataFrame(rows).sort_values(["seed", "step"]).reset_index(drop=True)
    missing_rss = df["rss"].isna().sum()
    return df, inputs, [] if missing_rss == 0 else [
        f"{int(missing_rss)}/{len(df)} checkpoints lack RSS/cond(J) "
        f"(estimator_diagnostics.py not yet run for them). Run: {DIAG_QSUB}"]


def build_param_correlations(base_params: dict) -> pd.DataFrame:
    rows = []
    for seed in ("seed1", "seed2", "seed42"):
        for csv in sorted(glob.glob(str(PROJECT_ROOT / "baseline" / f"{GRID_PREFIX[seed]}*"))):
            name = Path(csv).name
            try:
                step = int(name.rsplit("ckpt", 1)[1])
            except (IndexError, ValueError):
                continue
            p = find_nls_csv("baseline", name)
            if p is None:
                continue
            cmp = compare_params(load_nls(p), base_params)
            rows.append({
                "seed": seed, "step": step, "model_name": name,
                "exploratory": (seed == "seed42"),
                "alpha_pearson": cmp["alpha"]["pearson"], "alpha_spearman": cmp["alpha"]["spearman"],
                "alpha_mae": cmp["alpha"]["mae"], "alpha_rmse": cmp["alpha"]["rmse"],
                "alpha_pearson_excl_37_51": cmp["alpha_excl_37_51"]["pearson"],
                "alpha_spearman_excl_37_51": cmp["alpha_excl_37_51"]["spearman"],
                "beta_pearson": cmp["beta"]["pearson"], "beta_spearman": cmp["beta"]["spearman"],
                "beta_mae": cmp["beta"]["mae"], "beta_rmse": cmp["beta"]["rmse"],
                "utility_pearson": cmp["utility"]["pearson"], "utility_spearman": cmp["utility"]["spearman"],
                "utility_mae": cmp["utility"]["mae"], "utility_rmse": cmp["utility"]["rmse"],
            })
    return pd.DataFrame(rows).sort_values(["seed", "step"]).reset_index(drop=True)


# ── validation (assert, do not hardcode) ──────────────────────────────────────
def validate_reference(base_params: dict) -> dict:
    """The matched-base vs exploratory step-8k comparison should reproduce known
    values; mismatch => a source-path / alignment bug, so stop."""
    p = find_nls_csv("baseline", "Qwen-7B-GRPO-qd-ckpt8000")
    if p is None:
        return {"ran": False, "reason": "step-8k CSV absent"}
    cmp = compare_params(load_nls(p), base_params)
    expect = {"alpha_pearson": (cmp["alpha"]["pearson"], 0.240),
              "alpha_spearman": (cmp["alpha"]["spearman"], 0.843),
              "alpha_excl_pearson": (cmp["alpha_excl_37_51"]["pearson"], 0.773),
              "beta_pearson": (cmp["beta"]["pearson"], 0.976),
              "utility_pearson": (cmp["utility"]["pearson"], 0.806),
              "utility_spearman": (cmp["utility"]["spearman"], 0.839)}
    checks = {k: {"got": round(g, 4), "expect": e, "ok": abs(g - e) <= 0.02}
              for k, (g, e) in expect.items()}
    return {"ran": True, "all_ok": all(c["ok"] for c in checks.values()), "checks": checks}


# ── plotting ──────────────────────────────────────────────────────────────────
SEED_COLOR = {"seed1": "#1f77b4", "seed2": "#d62728", "seed42": "#7f7f7f"}


def plot_grpo(dfs: dict, window: int, path: Path):
    panels = [("reward", "mean reward"), ("reward_std", "reward std"),
              ("loss", "GRPO/DAPO policy loss"), ("kl", "KL from base"),
              ("entropy", "entropy"), ("frac_reward_zero_std", "zero-reward/DAPO filter frac"),
              ("grad_norm", "gradient norm"), ("learning_rate", "learning rate")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (col, title) in zip(axes.ravel(), panels):
        any_data = False
        for seed, df in dfs.items():
            if seed == "seed42" or df.empty or col not in df or not df[col].notna().any():
                continue
            c = SEED_COLOR[seed]
            ax.plot(df["step"], df[col], color=c, alpha=0.18, lw=0.7)
            ax.plot(df["step"], causal_rolling(df[col], window), color=c, lw=1.9,
                    label=f"{seed} (causal mean w={window})")
            any_data = True
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
        if col == "learning_rate":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if not any_data:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=ax.transAxes)
    axes.ravel()[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"GRPO training dynamics (Qwen-own-delta seeds 1 & 2)  |  light=raw, "
                 f"bold=causal trailing mean, window={window} obs (~{window*10} steps)  |  "
                 f"noisy TRAINING diagnostics, not validation estimates", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_structural(df: pd.DataFrame, path: Path, mode_note: str = ""):
    panels = [("lambda", "lambda (loss aversion)"), ("eta", "eta (status-quo bias)"),
              ("d", "selection objective d = sqrt(lambda^2+eta^2)"), ("rss", "NLS RSS at solution"),
              ("log10_cond_jacobian", "log10 cond(Jacobian)"),
              ("spearman_alpha", "alpha Spearman vs local base"),
              ("spearman_beta", "beta Spearman vs local base"),
              ("spearman_utility", "utility-grid Spearman vs local base")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (col, title) in zip(axes.ravel(), panels):
        drew = False
        for seed in ("seed1", "seed2"):
            sub = df[(df["seed"] == seed) & (~df["exploratory"])].dropna(subset=[col])
            if not sub.empty:
                ax.plot(sub["step"], sub[col], "o-", color=SEED_COLOR[seed], lw=1.6,
                        ms=4, label=seed)
                drew = True
                sel = sub[sub["selected"]]
                if not sel.empty:
                    ax.scatter(sel["step"], sel[col], s=140, facecolors="none",
                               edgecolors=SEED_COLOR[seed], linewidths=2.2, zorder=5)
        expl = df[df["exploratory"]].dropna(subset=[col])
        if not expl.empty:
            ax.plot(expl["step"], expl[col], "x--", color=SEED_COLOR["seed42"], lw=1.0,
                    ms=6, label="exploratory, non-gating")
            drew = True
        base = df[df["seed"] == "base"].dropna(subset=[col])
        if not base.empty:
            ax.scatter(base["step"], base[col], marker="*", s=200, color="k",
                       zorder=6, label="local base (step 0)")
            drew = True
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
        if not drew or df[col].dropna().empty:
            ax.text(0.5, 0.5, f"unavailable\n(run: {DIAG_QSUB})" if col in ("rss", "log10_cond_jacobian")
                    else "unavailable", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    axes.ravel()[0].legend(fontsize=8, loc="best")
    fig.suptitle("Structural NLS trajectory vs exact local base (Qwen-7B-Base-Local).  "
                 "Circled = frozen selection (seed1@2000, seed2@6000).  "
                 "RSS/cond(J) shown only where estimator_diagnostics has run." + mode_note,
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_preservation(dfc: pd.DataFrame, path: Path, mode_note: str = ""):
    families = [("alpha", "alpha (item FE)"), ("beta", "beta (attribute-profile FE)"),
                ("utility", "utility grid U=exp(alpha+beta)")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, (fam, title) in zip(axes, families):
        drew = False
        for seed in ("seed1", "seed2"):
            sub = dfc[(dfc["seed"] == seed) & (~dfc["exploratory"])]
            if sub.empty:
                continue
            c = SEED_COLOR[seed]
            ax.plot(sub["step"], sub[f"{fam}_pearson"], "o-", color=c, lw=1.6, ms=4,
                    label=f"{seed} Pearson")
            ax.plot(sub["step"], sub[f"{fam}_spearman"], "s--", color=c, lw=1.3, ms=3,
                    alpha=0.8, label=f"{seed} Spearman")
            drew = True
        if fam == "alpha":
            for seed in ("seed1", "seed2"):
                sub = dfc[(dfc["seed"] == seed) & (~dfc["exploratory"])]
                if not sub.empty:
                    ax.plot(sub["step"], sub["alpha_pearson_excl_37_51"], ":", color=SEED_COLOR[seed],
                            lw=1.2, alpha=0.7, label=f"{seed} Pearson excl 37/51")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.set_ylabel("correlation vs local base")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        if drew:
            ax.legend(fontsize=7, loc="lower right")
        else:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("Parameter preservation vs exact local base — aligned by name, tie-aware ranks, "
                 "references alpha_1/beta_1,1 excluded" + mode_note, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def finite_guard(df: pd.DataFrame, cols: list, label: str):
    bad = {}
    for c in cols:
        if c in df:
            v = pd.to_numeric(df[c], errors="coerce")
            n = int(np.isinf(v).sum())
            if n:
                bad[c] = n
    if bad:
        print(f"  [warn] {label}: non-finite values present (kept out of plots): {bad}")
    return bad


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"git_commit": git_commit(), "generated_by": "eval/plot_training_dynamics.py",
                "exact_local_base": f"{BASE_FEATURE}/{BASE_MODEL}",
                "frozen_selection_grid": sorted(FROZEN_SELECTION_GRID),
                "selected_checkpoints": SELECTED,
                "smoothing": {"method": "causal trailing rolling mean", "window_obs": SMOOTH_WINDOW,
                              "step_spacing": 10},
                "grpo_note": "GRPO beta=0.04 is the KL coefficient; distinct from structural beta.",
                "inputs": {}, "analyses": {}, "pending": []}

    # (A) GRPO metrics
    print("== GRPO training metrics ==")
    grpo_dfs = {}
    for seed in ("seed1", "seed2", "seed42"):
        df, present, src = load_grpo(seed)
        manifest["inputs"][f"grpo_{seed}"] = src
        if df.empty:
            print(f"  {seed}: no metrics"); continue
        p = OUT / f"grpo_metrics_{seed}.csv"
        df.to_csv(p, index=False)
        grpo_dfs[seed] = df
        finite_guard(df, GRPO_METRICS, f"grpo_{seed}")
        miss = [m for m, ok in present.items() if not ok]
        print(f"  {seed}: {len(df)} obs, steps {df['step'].min()}-{df['step'].max()}"
              + (f", MISSING {miss}" if miss else ", all metrics present"))
    manifest["analyses"]["grpo_training_dynamics"] = {
        "type": "post-hoc / non-gating (training diagnostic)",
        "seeds": list(grpo_dfs.keys()), "metrics": GRPO_METRICS}
    plot_grpo(grpo_dfs, SMOOTH_WINDOW, OUT / "grpo_training_curves.png")
    print(f"  wrote grpo_training_curves.png")

    # (B) structural — recompute from the Model-A NLS CSVs when they exist, else
    # REPLOT FROM DERIVED ARTIFACTS (the committed structural_trajectory.csv /
    # parameter_correlations.csv), so figures regenerate on a clean clone where
    # the raw per-checkpoint NLS CSVs are not present.
    print("== structural trajectory ==")
    base_csv = find_nls_csv(BASE_FEATURE, BASE_MODEL)
    seed_nls = glob.glob(str(PROJECT_ROOT / "baseline" / f"{GRID_PREFIX['seed1']}*"
                             / "Model_1" / "*NLS_estimation*.csv"))
    derived_strat, derived_corr = OUT / "structural_trajectory.csv", OUT / "parameter_correlations.csv"
    mode_note = ""

    if base_csv is not None and seed_nls:
        structural_mode = "recomputed_from_nls_csvs"
        base_params = load_nls(base_csv)
        base_sha = sha256_of(base_csv)
        manifest["inputs"]["structural_base"] = {"path": str(base_csv.relative_to(PROJECT_ROOT)),
                                                 "sha256": base_sha}
        val = validate_reference(base_params)
        manifest["analyses"]["reference_validation"] = val
        if val.get("ran") and not val["all_ok"]:
            for k, c in val["checks"].items():
                print(f"    {k}: got {c['got']} expect {c['expect']} ok={c['ok']}")
            raise SystemExit("FATAL: reference validation (base vs step-8k) drifted >0.02 — "
                             "investigate source paths / parameter alignment before trusting output.")
        print(f"  reference validation vs step-8k: {'PASS' if val.get('all_ok') else val.get('reason')}")
        strat, s_inputs, pending = build_structural(base_params, base_sha)
        manifest["inputs"]["structural_checkpoints"] = s_inputs
        manifest["pending"] += pending
        strat.to_csv(derived_strat, index=False)
        dfc = build_param_correlations(base_params)
        dfc.to_csv(derived_corr, index=False)
    elif derived_strat.exists() and derived_corr.exists():
        structural_mode = "replot_from_derived_artifacts"
        mode_note = "   [replot from derived artifacts]"
        print("  seed NLS CSVs absent -> REPLOT FROM DERIVED ARTIFACTS "
              "(committed structural_trajectory.csv / parameter_correlations.csv; not overwritten)")
        strat = pd.read_csv(derived_strat)
        dfc = pd.read_csv(derived_corr)
        manifest["inputs"]["structural_base"] = {"path": None,
            "note": "NLS CSVs absent; replot from committed derived artifacts"}
        manifest["analyses"]["reference_validation"] = {
            "ran": False, "reason": "replot from derived artifacts (NLS CSVs absent)"}
    else:
        raise SystemExit("FATAL: no NLS CSVs and no committed derived structural artifacts to plot from")

    manifest["analyses"]["structural_mode"] = structural_mode
    finite_guard(strat, ["lambda", "eta", "d", "rss", "log10_cond_jacobian"], "structural")
    manifest["analyses"]["structural_trajectory"] = {
        "type": "lambda/eta/d = validation estimates; RSS/cond(J) = post-hoc diagnostic",
        "mode": structural_mode, "n_rows": int(len(strat)),
        "rss_cond_available_for": strat.dropna(subset=["rss"])["model_name"].tolist()}
    plot_structural(strat, OUT / "structural_training_curves.png", mode_note)
    print(f"  wrote structural_training_curves.png ({len(strat)} rows, {structural_mode})")

    manifest["analyses"]["parameter_preservation"] = {
        "type": "post-hoc / non-gating", "mode": structural_mode, "alignment": "by Parameter name",
        "rank_method": "scipy.stats.spearmanr (tie-aware)",
        "references_excluded": ["alpha_1=0", "beta_1,1=0"], "n_checkpoints": int(len(dfc))}
    plot_preservation(dfc, OUT / "parameter_preservation.png", mode_note)
    print(f"  wrote parameter_preservation.png ({len(dfc)} rows)")

    # multistart status (do not fabricate)
    ms = glob.glob(str(PROJECT_ROOT / "results" / "estimator_diagnostics" / "*_multistart.json"))
    manifest["analyses"]["multistart"] = {
        "type": "post-hoc / non-gating", "present": bool(ms),
        "targets": [BASE_MODEL, "Qwen-7B-GRPO-qd-ckpt8000",
                    "Qwen-7B-GRPO-qd-seed1-ckpt2000", "Qwen-7B-GRPO-qd-seed2-ckpt6000"]}
    if not ms:
        manifest["pending"].append(f"multistart diagnostics not run. Run: {DIAG_QSUB}")
        print(f"  multistart: PENDING (run: {DIAG_QSUB})")

    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2, default=str)
    open(OUT / "manifest.json", "a").write("\n")
    print(f"== wrote {OUT.relative_to(PROJECT_ROOT)}/manifest.json ==")
    if manifest["pending"]:
        print("PENDING:")
        for p in manifest["pending"]:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
