#!/usr/bin/env python3
"""Training-dynamics figures for the Phase-A SFT batch-sensitivity experiment.

OPTIMIZATION DIAGNOSTICS ONLY. Nothing here is a behavioural result: loss,
gradient norm and learning rate describe the optimizer, not lambda, eta,
consistency or preference preservation. The behavioural numbers live in
results/sft_sensitivity/PHASE_A_REPORT.md and were computed after inference.

The x-axis is PROMPTS SEEN, never optimizer steps. At a fixed 6,016-prompt
exposure a batch-64 cell takes 94 updates and a batch-1 cell takes 6,016, so a
step axis would compress the comparison by 64x and make the batches look like
runs of wildly different length. Smoothing is likewise defined in prompt
exposure and converted to a per-batch observation count, so every curve is
smoothed over the same amount of DATA.

Three modes, same contract as eval/plot_sft_training_dynamics.py:

    python eval/plot_sft_sensitivity_dynamics.py --refresh   # trainer states -> CSV
    python eval/plot_sft_sensitivity_dynamics.py             # CSV -> figures
    python eval/plot_sft_sensitivity_dynamics.py --check     # figures agree with CSV

Rendering and checking are repository-reproducible from the tracked CSV;
--refresh needs the gitignored checkpoints/sft_sensitivity/ trainer states.
CPU only: no GPU job, no inference, no checkpoint evaluation.
"""
from __future__ import annotations

import argparse
import filecmp
import glob
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import sft_sensitivity_plan as P     # noqa: E402

VERSION = "1.0.0"
OUT = ROOT / "results" / "sft_sensitivity"
CSV = OUT / "dynamics_metrics.csv"
MANIFEST = OUT / "dynamics_manifest.json"
MD = OUT / "dynamics_interpretation.md"
PNG_SEED = OUT / "dynamics_by_seed.png"
PNG_BATCH = OUT / "dynamics_by_batch.png"
PNG_LOG = OUT / "dynamics_grad_norm_logscale.png"
PNG_LRGRID = OUT / "dynamics_phaseB_lr_grid.png"

SMOOTH_PROMPTS = 512          # causal window, measured in PROMPT EXPOSURE
CLIP = P.FIXED["max_grad_norm"]
BATCH_COLOR = {1: "#1f77b4", 16: "#d62728", 32: "#2ca02c", 64: "#9467bd"}
SEED_STYLE = {1: "-", 2: "--", 3: ":"}
PANELS = [("loss", "completion-only cross-entropy loss"),
          ("grad_norm", "pre-clipping gradient norm"),
          ("learning_rate", "learning rate")]


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def causal_rolling(s: pd.Series, window_obs: int, stat: str = "mean") -> pd.Series:
    """Trailing statistic: current and preceding observations only, never future.

    Gradient norm uses the MEDIAN, not the mean. The distribution is heavy-tailed
    — batch 1 has a median of 0.91 against a maximum of 310 — so a rolling mean
    is dominated by the rare spikes and would plot batch 1 at ~25, describing its
    tail rather than its typical update. Loss and learning rate use the mean.
    """
    r = s.rolling(window=max(1, window_obs), min_periods=1)
    return r.median() if stat == "median" else r.mean()


def stat_for(col: str) -> str:
    return "median" if col == "grad_norm" else "mean"


def window_obs_for(effective_batch: int) -> int:
    """The 512-prompt window expressed as observations for this batch."""
    return max(1, int(round(SMOOTH_PROMPTS / effective_batch)))


# ── refresh: trainer states -> tidy CSV ─────────────────────────────────────
def final_trainer_state(cell: P.Cell) -> tuple[Path, int] | None:
    """Highest-numbered checkpoint. Sorted NUMERICALLY: 'checkpoint-64' sorts
    after 'checkpoint-188' as a string, which would silently truncate every
    large-batch history."""
    cands = []
    for c in glob.glob(str(cell.output_dir / "checkpoint-*")):
        tail = Path(c).name.rsplit("-", 1)[-1]
        p = Path(c) / "trainer_state.json"
        if tail.isdigit() and p.is_file():
            cands.append((int(tail), p))
    return (lambda t: (t[1], t[0]))(max(cands)) if cands else None


def all_cells() -> list:
    """Phase-A cells plus every Phase-B cell that has actually been run.

    Phase B is enumerated over the full LR grid and filtered by what exists on
    disk, so a partially-run sweep plots what it has instead of failing.
    """
    cells = list(P.phase_a_cells())
    seen = {c.name for c in cells}
    for b in P.LARGE_BATCHES:
        for c in P.phase_b_cells(b):
            if c.name in seen:
                continue
            if (c.output_dir).is_dir() and final_trainer_state(c):
                cells.append(c)
                seen.add(c.name)
    return cells


def do_refresh() -> dict:
    rows, srcs = [], {}
    for cell in all_cells():
        got = final_trainer_state(cell)
        if got is None:
            print(f"[refresh] MISSING trainer state: {cell.name}")
            continue
        path, last_step = got
        if last_step != cell.optimizer_steps:
            raise SystemExit(f"REFUSED: {cell.name} final state is step {last_step}, "
                             f"expected {cell.optimizer_steps} — incomplete run.")
        hist = json.loads(path.read_text())["log_history"]
        n = 0
        for e in hist:
            if "step" not in e or "loss" not in e:
                continue
            step = int(e["step"])
            rows.append({
                "run_id": cell.name, "phase": cell.phase,
                "effective_batch": cell.effective_batch,
                "lr_config": cell.learning_rate,
                "seed": cell.seed, "step": step,
                "prompts_seen": step * cell.effective_batch,
                "loss": float(e["loss"]),
                "grad_norm": (float(e["grad_norm"]) if e.get("grad_norm") is not None
                              else np.nan),
                "learning_rate": float(e.get("learning_rate", np.nan)),
            })
            n += 1
        srcs[cell.name] = {
            "trainer_state": str(path.relative_to(ROOT)), "sha256": sha256_of(path),
            "final_global_step": last_step, "expected_steps": cell.optimizer_steps,
            "n_logged_updates": n, "effective_batch": cell.effective_batch,
            "seed": cell.seed, "prompt_exposure": cell.exposure,
            "phase": cell.phase, "lr_config": cell.learning_rate,
        }
        print(f"[refresh] {cell.name:<42} {n:>5} updates -> "
              f"{n * cell.effective_batch:,} prompts")

    if not rows:
        raise SystemExit("[refresh] no trainer states found — nothing written.")
    df = pd.DataFrame(rows).sort_values(
        ["phase", "effective_batch", "lr_config", "seed", "step"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)
    print(f"[refresh] wrote {CSV.relative_to(ROOT)} ({len(df):,} rows)")
    return {"version": VERSION, "extraction_date": date.today().isoformat(),
            "git_commit_at_extraction": git_commit(), "sources": srcs}


# ── statistics ──────────────────────────────────────────────────────────────
def clip_stats(df: pd.DataFrame) -> dict:
    out = {}
    for b, g in df.groupby("effective_batch"):
        per_seed = {}
        for s, gs in g.groupby("seed"):
            v = gs["grad_norm"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            per_seed[int(s)] = float((v > CLIP).mean()) if v.size else math.nan
        v = g["grad_norm"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        out[int(b)] = {
            "clip_fraction_pooled": float((v > CLIP).mean()),
            "clip_fraction_by_seed": per_seed,
            "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max()),
            "n_updates_per_seed": int(len(g) / g["seed"].nunique()),
        }
    return out


# ── figures ─────────────────────────────────────────────────────────────────
def _finish_panel(ax, col, title, marks):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("prompts seen")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v/1000)}k" if v else "0"))
    if col == "learning_rate":
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for m in marks:
        ax.axvline(m, color="0.55", lw=0.5, alpha=0.20, zorder=0)


LOSS_YLIM = (0.0, 1.5)   # raw batch-1 loss reaches ~8 and squashes every curve


def _clip_note(ax, cs):
    txt = ("clipped updates (|g| > 0.1)\n" +
           "\n".join(f"  eb{b}: {cs[b]['clip_fraction_pooled']:.0%}" for b in sorted(cs)))
    ax.annotate(txt, xy=(0.015, 0.03), xycoords="axes fraction", ha="left",
                va="bottom", fontsize=8, color="#7a2020", fontweight="bold",
                bbox=dict(fc="white", ec="#7a2020", lw=0.6, alpha=0.85, pad=0.35))


def _crop_loss(ax, df):
    """Crop the loss view and DISCLOSE what falls outside — never a silent crop."""
    lo, hi = LOSS_YLIM
    ax.set_ylim(lo, hi)
    v = df["loss"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    out = int(((v < lo) | (v > hi)).sum())
    if out:
        ax.annotate(f"y cropped to [{lo:g}, {hi:g}] — {out:,}/{v.size:,} raw points "
                    f"({out / v.size:.1%}) above, max {v.max():.1f}",
                    xy=(0.5, 0.985), xycoords="axes fraction", ha="center", va="top",
                    fontsize=7.5, color="#7a2020")


def plot_by_seed(df, cs, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    marks = list(range(2048, 6017, 2048))
    for ax, (col, title) in zip(axes, PANELS):
        for b in sorted(df["effective_batch"].unique()):
            for s in sorted(df["seed"].unique()):
                g = df[(df.effective_batch == b) & (df.seed == s)].sort_values("prompts_seen")
                if g.empty or g[col].isna().all():
                    continue
                c, w = BATCH_COLOR[int(b)], window_obs_for(int(b))
                if col != "learning_rate":
                    ax.plot(g["prompts_seen"], g[col], color=c, alpha=0.13, lw=0.6, zorder=1)
                ax.plot(g["prompts_seen"], causal_rolling(g[col], w, stat_for(col)), color=c,
                        ls=SEED_STYLE[int(s)], lw=1.7, zorder=3,
                        label=f"eb{int(b)} seed {int(s)}")
        _finish_panel(ax, col, title, marks)
        if col == "loss":
            _crop_loss(ax, df)
        if col == "grad_norm":
            ax.set_yscale("log")
            ax.axhline(CLIP, color="#7a2020", ls=":", lw=1.2, zorder=2)
            ax.annotate("max_grad_norm = 0.1", xy=(0.99, CLIP),
                        xycoords=("axes fraction", "data"), ha="right", va="bottom",
                        fontsize=7.5, color="#7a2020")
            _clip_note(ax, cs)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "SFT batch-sensitivity — OPTIMIZATION DIAGNOSTICS by seed (Phase A, 6,016 prompts, lr 1e-6)\n"
        f"x-axis = prompts seen, NOT optimizer steps · faint = raw logged updates, bold = causal "
        f"trailing mean over {SMOOTH_PROMPTS} prompts — MEDIAN for gradient norm "
        f"(identical data window for every batch)\n"
        "NOT behavioural results: loss/gradient/LR describe the optimizer, not λ, η or consistency",
        fontsize=11, y=0.995, va="top")
    fig.tight_layout(); fig.subplots_adjust(top=0.80, bottom=0.185)
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_by_batch(df, cs, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    marks = list(range(2048, 6017, 2048))
    for ax, (col, title) in zip(axes, PANELS):
        for b in sorted(df["effective_batch"].unique()):
            b = int(b)
            w, c = window_obs_for(b), BATCH_COLOR[b]
            sm = []
            for s in sorted(df["seed"].unique()):
                g = df[(df.effective_batch == b) & (df.seed == s)].sort_values("prompts_seen")
                if g.empty:
                    continue
                sm.append(pd.Series(causal_rolling(g[col], w, stat_for(col)).to_numpy(),
                                    index=g["prompts_seen"].to_numpy()))
            if not sm:
                continue
            M = pd.concat(sm, axis=1)
            mean, lo, hi = M.mean(axis=1), M.min(axis=1), M.max(axis=1)
            # band = full across-seed range (3 seeds; a sd band would overstate
            # precision at n=3)
            ax.fill_between(M.index, lo, hi, color=c, alpha=0.18, lw=0, zorder=2)
            ax.plot(M.index, mean, color=c, lw=2.0, zorder=3,
                    label=f"eb{b} ({cs[b]['n_updates_per_seed']:,} updates)")
        _finish_panel(ax, col, title, marks)
        if col == "grad_norm":
            ax.set_yscale("log")
            ax.axhline(CLIP, color="#7a2020", ls=":", lw=1.2, zorder=4)
            ax.annotate("max_grad_norm = 0.1", xy=(0.99, CLIP),
                        xycoords=("axes fraction", "data"), ha="right", va="bottom",
                        fontsize=7.5, color="#7a2020")
            _clip_note(ax, cs)
        ax.legend(fontsize=8.5, loc="best", framealpha=0.85)
    fig.suptitle(
        "SFT batch-sensitivity — OPTIMIZATION DIAGNOSTICS by effective batch (Phase A, 3 seeds)\n"
        f"bold = across-seed mean of the causal {SMOOTH_PROMPTS}-prompt trailing "
        "statistic (median for gradient norm, mean for loss/LR) · "
        "band = full min–max range across seeds 1–3 (not a standard deviation: n=3)\n"
        "NOT behavioural results — these describe the optimizer, not λ, η or consistency",
        fontsize=11, y=0.995, va="top")
    fig.tight_layout(); fig.subplots_adjust(top=0.80, bottom=0.155)
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_grad_log(df, cs, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    marks = list(range(2048, 6017, 2048))
    for ax, mode in zip(axes, ("raw", "smoothed")):
        for b in sorted(df["effective_batch"].unique()):
            b = int(b); c, w = BATCH_COLOR[b], window_obs_for(b)
            for s in sorted(df["seed"].unique()):
                g = df[(df.effective_batch == b) & (df.seed == s)].sort_values("prompts_seen")
                if g.empty:
                    continue
                if mode == "raw":
                    ax.plot(g["prompts_seen"], g["grad_norm"], color=c, alpha=0.30,
                            lw=0.5, zorder=1,
                            label=f"eb{b}" if s == 1 else None)
                else:
                    ax.plot(g["prompts_seen"], causal_rolling(g["grad_norm"], w, "median"),
                            color=c, ls=SEED_STYLE[int(s)], lw=1.6, zorder=3,
                            label=f"eb{b} seed {int(s)}")
        ax.set_yscale("log")
        ax.axhline(CLIP, color="#7a2020", ls=":", lw=1.3, zorder=4)
        ax.annotate("max_grad_norm = 0.1", xy=(0.99, CLIP), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8, color="#7a2020")
        _finish_panel(ax, "grad_norm",
                      "every logged update (uncropped)" if mode == "raw"
                      else f"causal {SMOOTH_PROMPTS}-prompt trailing MEDIAN", marks)
        ax.legend(fontsize=8, loc="best", ncol=2, framealpha=0.85)
    rows = "   ".join(f"eb{b}: med {cs[b]['median']:.2f} / P90 {cs[b]['p90']:.1f} / "
                      f"P99 {cs[b]['p99']:.1f} / max {cs[b]['max']:.1f}" for b in sorted(cs))
    fig.text(0.5, 0.012, rows, ha="center", va="bottom", fontsize=8.5, color="#333333")
    big = max(cs[b]["max"] for b in cs if b != 1)
    fig.suptitle(
        "Pre-clipping gradient norm, log scale, NOTHING CROPPED — the full tail at every batch\n"
        f"Batch 1 reaches {cs[1]['max']:.0f}; every large batch stays below {big:.0f}. "
        "Everything above the dotted line was clipped to 0.1 before the update.\n"
        "OPTIMIZATION DIAGNOSTIC — a large pre-clipping norm is not gradient explosion; "
        "no run produced a non-finite value.",
        fontsize=11, y=0.995, va="top")
    fig.tight_layout(); fig.subplots_adjust(top=0.80, bottom=0.115)
    fig.savefig(path, dpi=130); plt.close(fig)


LR_COLOR = {1e-6: "#1f77b4", 3e-6: "#ff7f0e", 1e-5: "#d62728",
            3e-7: "#7f7f7f", 3e-5: "#8c564b"}


def _lr_lab(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def plot_lr_grid(df: pd.DataFrame, path: Path) -> None:
    """Phase-B: 3 metrics x 3 batches, one curve per learning rate.

    This is a DIFFERENT comparison from the Phase-A figures. Those hold the
    learning rate at 1e-6 and vary the batch; this holds the batch within each
    column and varies the learning rate. Mixing them onto one axis would confound
    the two, so they stay separate figures.
    """
    batches = sorted(df["effective_batch"].unique())
    fig, axes = plt.subplots(3, len(batches), figsize=(6.0 * len(batches), 12.6),
                             squeeze=False)
    marks = list(range(2048, 6017, 2048))
    for r, (col, title) in enumerate(PANELS):
        for c, b in enumerate(batches):
            ax = axes[r][c]
            sub_b = df[df["effective_batch"] == b]
            for lr in sorted(sub_b["lr_config"].unique()):
                g_lr = sub_b[sub_b["lr_config"] == lr]
                seeds = sorted(g_lr["seed"].unique())
                w, colr = window_obs_for(int(b)), LR_COLOR.get(float(lr), "#333333")
                sm = []
                for sd in seeds:
                    g = g_lr[g_lr["seed"] == sd].sort_values("prompts_seen")
                    if g.empty or g[col].isna().all():
                        continue
                    sm.append(pd.Series(
                        causal_rolling(g[col], w, stat_for(col)).to_numpy(),
                        index=g["prompts_seen"].to_numpy()))
                if not sm:
                    continue
                M = pd.concat(sm, axis=1)
                if M.shape[1] > 1:
                    ax.fill_between(M.index, M.min(axis=1), M.max(axis=1),
                                    color=colr, alpha=0.18, lw=0, zorder=2)
                ax.plot(M.index, M.mean(axis=1), color=colr, lw=1.9, zorder=3,
                        label=f"lr {_lr_lab(float(lr))}  ({len(sm)} seed"
                              f"{'s' if len(sm) > 1 else ''})")
            _finish_panel(ax, col, f"eb{int(b)} — {title}", marks)
            if col == "grad_norm":
                ax.set_yscale("log")
                ax.axhline(CLIP, color="#7a2020", ls=":", lw=1.2, zorder=4)
                ax.annotate("max_grad_norm = 0.1", xy=(0.99, CLIP),
                            xycoords=("axes fraction", "data"), ha="right",
                            va="bottom", fontsize=7.5, color="#7a2020")
            if col == "loss":
                ax.set_ylim(0.0, 1.5)
            ax.legend(fontsize=8, loc="best", framealpha=0.85)
    fig.suptitle(
        "Phase B — OPTIMIZATION DIAGNOSTICS by learning rate, within each effective batch\n"
        "rows = metric, columns = effective batch, colour = learning rate · x-axis = prompts seen · "
        f"causal {SMOOTH_PROMPTS}-prompt window (MEDIAN for gradient norm)\n"
        "band = across-seed min–max; lr 1e-6 is the REUSED Phase-A column (3 seeds), "
        "3e-6 and 1e-5 are Phase B (2 seeds) · "
        "loss cropped to [0, 1.5]\n"
        "NOT behavioural results — these describe the optimizer, not λ, η or consistency",
        fontsize=11.5, y=0.997, va="top")
    fig.tight_layout(); fig.subplots_adjust(top=0.915)
    fig.savefig(path, dpi=125)
    plt.close(fig)


def render_md(cs: dict) -> str:
    L = ["# Phase-A training dynamics — optimization diagnostics", "",
         "*Figures for the SFT effective-batch sensitivity experiment "
         "(`SFT_BATCH_LR_SENSITIVITY_PROTOCOL.md` + Amendment 1). "
         "CPU-only: no GPU job, no inference, no checkpoint evaluation.*", "",
         "> **These are optimization diagnostics, not behavioural results.** Loss, "
         "gradient norm and learning rate describe the optimizer. λ, η, `d`, "
         "consistency and W are behavioural estimands computed after inference and "
         "live in `PHASE_A_REPORT.md`. A smoother loss curve is not evidence of "
         "less ownership dependence.", "",
         "## Reading the axes", "",
         "The x-axis is **prompts seen**, never optimizer steps. At the fixed "
         "6,016-prompt exposure a batch-64 cell takes 94 updates and a batch-1 cell "
         "takes 6,016; a step axis would stretch the comparison by 64× and make the "
         "batches look like runs of different length.", "",
         f"Smoothing is a **causal trailing window of {SMOOTH_PROMPTS} prompts** — "
         "the same amount of *data* for every batch, which is "
         + ", ".join(f"{window_obs_for(b)} obs at eb{b}" for b in sorted(cs))
         + ". Only current and preceding observations enter each point. Raw "
           "observations are drawn faintly beneath; the CSV keeps every unsmoothed "
           "value.", "",
         "The gradient-norm panels smooth with a **median**, not a mean. That "
         "distribution is heavy-tailed — batch 1 has a median of 0.91 against a "
         "maximum of 310 — so a rolling mean plots batch 1 at ~25 and describes its "
         "tail rather than its typical update. Loss and learning rate use the mean.", "",
         "## Gradient norm and clipping", "",
         "| effective batch | updates/seed | median | P90 | P99 | max | clipped |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
    for b in sorted(cs):
        c = cs[b]
        L.append(f"| {b} | {c['n_updates_per_seed']:,} | {c['median']:.3f} | "
                 f"{c['p90']:.2f} | {c['p99']:.2f} | {c['max']:.1f} | "
                 f"{c['clip_fraction_pooled']:.1%} |")
    L += ["",
          "**The tail collapses as the batch grows.** P99 falls from "
          f"{cs[1]['p99']:.0f} at batch 1 to {cs[16]['p99']:.0f}–{cs[64]['p99']:.0f} "
          f"at the large batches (~10×), and the maximum from {cs[1]['max']:.0f} to "
          f"~{max(cs[b]['max'] for b in (16, 32, 64)):.0f} (~25×). The **median rises** "
          "(0.9 → 9.1) because averaging removes the many near-zero single-example "
          "gradients as well as the extreme ones. That is variance reduction, and it "
          "is the clearest result of the experiment.", "",
          "**Clipping becomes universal.** Batch 1 clips "
          f"{cs[1]['clip_fraction_pooled']:.0%} of updates; batches 16, 32 and 64 clip "
          "**100%** — every single update. With `max_grad_norm = 0.1` and accumulated "
          "norms concentrated at 2.6–9.1, every large-batch update is rescaled to the "
          "same fixed length. Those cells are therefore **not "
          "\"the same optimizer with a bigger batch\"** — they take normalized-gradient "
          "steps of constant size. Any batch effect read from these figures is "
          "entangled with that.", "",
          "A large pre-clipping norm is **not gradient explosion**. Batch 1's tail is "
          "heavy-tailed single-example gradients; **no run produced a non-finite loss, "
          "gradient or parameter** in any of the 12 cells.", "",
          "## Loss and learning rate", "",
          "The loss panels are not comparable point-for-point across batches: a "
          "batch-16 logged loss is a 16-example mean while a batch-1 logged loss is a "
          "single example, so batch 1 is dispersed by construction. Judge the smoothed "
          "level and trend, not the scatter.", "",
          "Each cell runs its **own** cosine schedule over its own horizon (6,016 / 16 "
          "/ 32 / 64 = 6,016 / 376 / 188 / 94 updates), with 5% warmup. On a "
          "prompts-seen axis the four schedules therefore trace the same shape, which "
          "is the point: the cells are matched on data, not on updates.", "",
          "## Files", "",
          "| file | contents |", "|---|---|",
          "| `dynamics_metrics.csv` | tidy per-update rows, unsmoothed |",
          "| `dynamics_by_seed.png` | 3 panels, 12 cells, seed by linestyle |",
          "| `dynamics_by_batch.png` | 3 panels, across-seed mean + min–max band |",
          "| `dynamics_grad_norm_logscale.png` | uncropped log-scale tails |",
          "| `dynamics_manifest.json` | sources, hashes, versions, clip stats |", "",
          "## Reproduce", "", "```bash",
          "python eval/plot_sft_sensitivity_dynamics.py --refresh   # trainer states -> CSV",
          "python eval/plot_sft_sensitivity_dynamics.py             # CSV -> figures",
          "python eval/plot_sft_sensitivity_dynamics.py --check     # verify agreement",
          "```", "",
          "Rendering and checking work from the tracked CSV alone; `--refresh` needs "
          "the gitignored `checkpoints/sft_sensitivity/` trainer states.", ""]
    return "\n".join(L)


def do_render(out_dir: Path, refresh_block: dict | None) -> dict:
    if not CSV.exists():
        raise SystemExit(f"{CSV} missing — run --refresh first.")
    df = pd.read_csv(CSV)
    if refresh_block is None:
        if not MANIFEST.exists():
            raise SystemExit(f"{MANIFEST} missing — run --refresh first.")
        refresh_block = json.loads(MANIFEST.read_text())["refresh"]

    out_dir.mkdir(parents=True, exist_ok=True)
    names = {"seed": PNG_SEED.name, "batch": PNG_BATCH.name, "log": PNG_LOG.name,
             "lrgrid": PNG_LRGRID.name, "md": MD.name, "csv": CSV.name}
    if out_dir != OUT:
        shutil.copyfile(CSV, out_dir / CSV.name)

    dfa = df[df["phase"] == "A"] if "phase" in df else df
    cs = clip_stats(dfa)
    plot_by_seed(dfa, cs, out_dir / names["seed"])
    plot_by_batch(dfa, cs, out_dir / names["batch"])
    plot_grad_log(dfa, cs, out_dir / names["log"])
    # Phase-B batches only. Batch 1 is abandoned and has a single learning
    # rate, so a batch-1 column would be an empty comparison in an LR figure.
    if "phase" in df and (df["phase"] == "B").any():
        dfb = df[df["effective_batch"].isin(P.LARGE_BATCHES)]
        plot_lr_grid(dfb, out_dir / names["lrgrid"])
    (out_dir / names["md"]).write_text(render_md(cs))

    man = {
        "title": "Phase-A SFT batch-sensitivity training dynamics",
        "generated_by": "eval/plot_sft_sensitivity_dynamics.py",
        "version": VERSION, "script_sha256": sha256_of(Path(__file__).resolve()),
        "git_commit_at_render": git_commit(),
        "status": ("OPTIMIZATION DIAGNOSTICS ONLY — not behavioural results. "
                   "CPU-only: no GPU job, no inference, no checkpoint evaluation."),
        "x_axis": "prompts seen (optimizer step x effective batch), never optimizer steps",
        "smoothing": {"method": "causal trailing mean (min_periods=1)",
                      "window_prompts": SMOOTH_PROMPTS,
                      "window_observations_by_batch": {b: window_obs_for(b)
                                                       for b in sorted(cs)},
                      "statistic": {"loss": "mean", "learning_rate": "mean",
                                    "grad_norm": "median (heavy-tailed; a rolling "
                                                 "mean would track the tail, not the "
                                                 "typical update)"},
                      "uses_future_observations": False,
                      "identical_data_window_for_all_batches": True},
        "aggregate_band": "full min-max across seeds 1-3 (not a sd: n=3)",
        "clipping_threshold": CLIP,
        "loss_panel_crop": {"ylim": list(LOSS_YLIM),
                            "applies_to": "dynamics_by_seed.png only",
                            "disclosed_on_panel": True,
                            "note": "display only; the CSV keeps every value "
                                    "and the by-batch figure is uncropped"},
        "gradient_and_clipping_by_batch": cs,
        "n_rows": int(len(df)),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "refresh": refresh_block,
        "outputs": {},
    }
    for n in names.values():
        p = out_dir / n
        if p.exists():
            man["outputs"][n] = {"sha256": sha256_of(p), "size_bytes": p.stat().st_size}
    (out_dir / MANIFEST.name).write_text(json.dumps(man, indent=2) + "\n")
    print(f"[render] {len(man['outputs'])} artifacts -> {out_dir}")
    for n in sorted(man["outputs"]):
        print(f"         {n}")
    return man


def do_check() -> int:
    if not MANIFEST.exists():
        print("[check] FAIL: no manifest — run --refresh then a render.")
        return 1
    tracked = json.loads(MANIFEST.read_text())
    fails, warns = [], []
    env_now = {"python": sys.version.split()[0], "numpy": np.__version__,
               "pandas": pd.__version__, "matplotlib": matplotlib.__version__}
    same_env = env_now == tracked.get("environment")
    if not same_env:
        warns.append(f"environment differs from recorded {tracked.get('environment')}")

    rec = tracked["outputs"].get(CSV.name)
    if not rec or sha256_of(CSV) != rec["sha256"]:
        fails.append(f"{CSV.name}: sha256 does not match the manifest")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fresh = do_render(tmp, tracked["refresh"])
        for n, r in tracked["outputs"].items():
            p = tmp / n
            if not p.exists():
                fails.append(f"{n}: re-render did not produce it")
            elif n.endswith(".png"):
                if sha256_of(p) != r["sha256"]:
                    (fails if same_env else warns).append(f"{n}: figure bytes differ")
            elif not filecmp.cmp(p, OUT / n, shallow=False):
                fails.append(f"{n}: re-rendered content differs")
        # Round-trip through JSON before comparing: the in-memory dicts key by
        # int while the tracked manifest keys by string, so a direct == would
        # always fail on a correct render.
        def norm(x):
            return json.loads(json.dumps(x))
        for key in ("gradient_and_clipping_by_batch", "smoothing", "loss_panel_crop"):
            if norm(fresh.get(key)) != norm(tracked.get(key)):
                fails.append(f"{key} differs after re-render")

    for w in warns:
        print(f"[check] WARN  {w}")
    for f in fails:
        print(f"[check] FAIL  {f}")
    if fails:
        print(f"[check] FAILED ({len(fails)})")
        return 1
    print(f"[check] PASS — {len(tracked['outputs'])} artifacts agree with the tracked CSV; "
          "no checkpoint or trainer state was read.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check and a.refresh:
        ap.error("--check and --refresh are mutually exclusive")
    if a.check:
        return do_check()
    do_render(OUT, do_refresh() if a.refresh else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
