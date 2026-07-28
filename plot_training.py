#!/usr/bin/env python3
"""
Plot GRPO training convergence from log file.

Usage:
    python plot_training.py                          # ablation run (train_qd_run.log)
    python plot_training.py train_long_run           # main run
    python plot_training.py train_qd_run --smooth 50 # heavier smoothing

Outputs:  logs/<stem>_convergence.png
"""

import re
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("log_stem", nargs="?", default="train_qd_run")
parser.add_argument("--smooth", type=int, default=30,
                    help="Rolling window for smoothing (steps)")
parser.add_argument("--out", default=None)
args = parser.parse_args()

ROOT   = Path(__file__).resolve().parent
log    = ROOT / "logs" / f"{args.log_stem}.log"
outpng = Path(args.out) if args.out else ROOT / "logs" / f"{args.log_stem}_convergence.png"

if not log.exists():
    sys.exit(f"Log not found: {log}")

# ── Parse ─────────────────────────────────────────────────────────────────────
log_text = log.read_text(errors="replace")

# Infer total steps from progress bar — use max denominator to skip batch-level bars
total_steps = 98900
bar_hits = re.findall(r"\| \d+/(\d+) \[", log_text)
if bar_hits:
    total_steps = max(int(x) for x in bar_hits)

KEYS = ["reward", "reward_std", "frac_reward_zero_std", "kl", "entropy", "loss"]
records = []

pat = re.compile(r"\{[^{}]+\}")   # match each JSON-like metric dict
for m in pat.finditer(log_text):
    blob = m.group()
    if "'reward'" not in blob:
        continue
    row = {"step": None}
    for key in KEYS:
        hit = re.search(rf"'{key}': '([^']+)'", blob)
        if hit:
            try:
                row[key] = float(hit.group(1))
            except ValueError:
                pass
    # epoch → approx step
    ep_hit = re.search(r"'epoch': '([^']+)'", blob)
    if ep_hit:
        row["step"] = float(ep_hit.group(1)) * total_steps
    if row.get("reward") is not None:
        records.append(row)

if not records:
    sys.exit("No metric records found in log.")

steps   = np.array([r["step"] for r in records if r["step"] is not None])
reward  = np.array([r.get("reward", np.nan) for r in records])
rstd    = np.array([r.get("reward_std", np.nan) for r in records])
dapo    = np.array([r.get("frac_reward_zero_std", np.nan) for r in records])
kl      = np.array([r.get("kl", np.nan) for r in records])
entropy = np.array([r.get("entropy", np.nan) for r in records])

# align lengths (epoch may be missing on first record)
n = min(len(steps), len(reward))
steps, reward, rstd, dapo, kl, entropy = (
    steps[:n], reward[:n], rstd[:n], dapo[:n], kl[:n], entropy[:n]
)

def smooth(y, w):
    """Causal rolling mean — no edge zero-padding artifacts."""
    if w <= 1 or len(y) < w:
        return y
    out = np.full_like(y, np.nan, dtype=float)
    for i in range(len(y)):
        start = max(0, i - w + 1)
        out[i] = np.nanmean(y[start:i + 1])
    return out

W = args.smooth

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "#333333",
    "xtick.color":      "#555555",
    "ytick.color":      "#555555",
    "text.color":       "#222222",
    "grid.color":       "#dddddd",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
    "font.size":        9,
})

ACCENT  = "#1f77b4"
GREEN   = "#2ca02c"
ORANGE  = "#ff7f0e"
RED     = "#d62728"
PURPLE  = "#9467bd"

fig = plt.figure(figsize=(14, 10), facecolor="white")
fig.suptitle(
    f"GRPO Training Convergence — {args.log_stem}\n"
    f"{len(records):,} metric records  |  smoothing window = {W}",
    fontsize=13, color="#111111", y=0.98
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                       left=0.07, right=0.97, top=0.92, bottom=0.06)

def make_ax(pos, title, ylabel, y, color, fill_std=None, hline=None, hline_label=None, ylim=None):
    ax = fig.add_subplot(pos)
    ax.set_title(title, fontsize=10, color="#aabbcc", pad=6)
    ax.set_xlabel("Training step", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True)
    if ylim:
        ax.set_ylim(*ylim)

    # raw (faded)
    ax.plot(steps, y, color=color, alpha=0.18, linewidth=0.6)
    # smoothed
    ys = smooth(y, W)
    ax.plot(steps, ys, color=color, linewidth=1.8, label="smoothed")

    if fill_std is not None:
        ys_std = smooth(fill_std, W)
        ax.fill_between(steps, ys - ys_std, ys + ys_std,
                        color=color, alpha=0.12)

    if hline is not None:
        ax.axhline(hline, color="#ff6b6b", linewidth=1.0, linestyle=":",
                   label=hline_label or str(hline))
        ax.legend(fontsize=7, framealpha=0.2)
    return ax

# 1 — Mean reward
ax1 = make_ax(gs[0, 0],
              "Mean Reward  (want → positive & stable)",
              "reward",
              reward, ACCENT,
              hline=0, hline_label="zero (break-even)")
ax1.axhspan(ax1.get_ylim()[0], 0, color=RED, alpha=0.04)
ax1.axhspan(0, ax1.get_ylim()[1], color=GREEN, alpha=0.04)

# 2 — zero task-reward advantage rate (nothing is filtered out; see KNOWN_ISSUES #4)
ax2 = make_ax(gs[0, 1],
              "Zero-Advantage Group Rate  (want → 0)",
              "frac_reward_zero_std",
              dapo, ORANGE,
              hline=0.5, hline_label="50% threshold",
              ylim=(0, 1.05))
ax2.fill_between(steps, smooth(dapo, W), 1.0,
                 color=RED, alpha=0.07)

# 3 — Reward std (diversity)
ax3 = make_ax(gs[1, 0],
              "Reward Std  (diversity in G=16 group)",
              "reward std",
              rstd, GREEN)

# 4 — KL divergence
ax4 = make_ax(gs[1, 1],
              "KL Divergence from Base Model",
              "KL",
              kl, PURPLE)

# 5 — Entropy
ax5 = make_ax(gs[2, 0],
              "Entropy  (want to stay > 0 — not collapsed)",
              "entropy",
              entropy, "#79c0ff")

# 6 — Reward rolling mean with 3 reference lines
ax6 = fig.add_subplot(gs[2, 1])
ax6.set_title("Reward — zoomed moving avg  (convergence guide)", fontsize=10, color="#aabbcc", pad=6)
ax6.set_xlabel("Training step", fontsize=8)
ax6.set_ylabel("reward (smoothed)", fontsize=8)
ax6.grid(True)

windows = [10, 100, 500]
colors  = ["#4ea8de", "#56d364", "#ffa657"]
for w, c in zip(windows, colors):
    ax6.plot(steps, smooth(reward, w), color=c, linewidth=1.5, label=f"w={w}")
ax6.axhline(0, color=RED, linewidth=0.8, linestyle=":")
ax6.legend(fontsize=7, framealpha=0.2)

# ── Progress annotation ───────────────────────────────────────────────────────
if len(steps) > 0:
    pct = steps[-1] / total_steps * 100
    fig.text(0.99, 0.01, f"Progress: {steps[-1]:,.0f} / {total_steps:,} steps  ({pct:.1f}%)",
             ha="right", va="bottom", fontsize=8, color="#666677")

plt.savefig(outpng, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpng}")
print(f"Parsed {len(records):,} log entries  |  step range: {steps[0]:.0f} – {steps[-1]:.0f}")
