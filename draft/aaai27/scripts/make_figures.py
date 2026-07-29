#!/usr/bin/env python3
"""Generate vector figures for the AAAI manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


def make_task_overview() -> None:
    """Show the response mapping as the 2x2 structure it actually is."""
    fig, ax = plt.subplots(figsize=(3.25, 1.42))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x0, x1, x2, x3 = 0.02, 0.32, 0.65, 0.98
    y0, y1, y2, y3 = 0.04, 0.36, 0.68, 0.94
    header_fill = "#f0f0f0"
    consistent_fill = "#e2e8ec"
    rule_color = "#4a4a4a"

    # Header cells and the two ownership-invariant outcomes.
    for x, y, w, h, face in [
        (x0, y2, x1 - x0, y3 - y2, header_fill),
        (x1, y2, x2 - x1, y3 - y2, header_fill),
        (x2, y2, x3 - x2, y3 - y2, header_fill),
        (x0, y1, x1 - x0, y2 - y1, header_fill),
        (x0, y0, x1 - x0, y1 - y0, header_fill),
        (x2, y1, x3 - x2, y2 - y1, consistent_fill),
        (x1, y0, x2 - x1, y1 - y0, consistent_fill),
    ]:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor="none"))

    # The corner cell contains the two perspective labels, separated by a
    # backslash diagonal: X indexes rows and Y indexes columns.
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x3 - x0,
            y3 - y0,
            facecolor="none",
            edgecolor=rule_color,
            linewidth=0.75,
        )
    )
    for x in (x1, x2):
        ax.plot([x, x], [y0, y3], color=rule_color, linewidth=0.65)
    for y in (y1, y2):
        ax.plot([x0, x3], [y, y], color=rule_color, linewidth=0.65)
    ax.plot([x0, x1], [y3, y2], color=rule_color, linewidth=0.65)
    ax.text(x0 + 0.27 * (x1 - x0), y2 + 0.27 * (y3 - y2), "X", ha="center", va="center", fontsize=8.5)
    ax.text(x0 + 0.73 * (x1 - x0), y2 + 0.73 * (y3 - y2), "Y", ha="center", va="center", fontsize=8.5)
    ax.text((x1 + x2) / 2, (y2 + y3) / 2, r"No $\rightarrow$ Y", ha="center", va="center", fontsize=8.3)
    ax.text((x2 + x3) / 2, (y2 + y3) / 2, r"Yes $\rightarrow$ X", ha="center", va="center", fontsize=8.3)
    ax.text((x0 + x1) / 2, (y1 + y2) / 2, r"No $\rightarrow$ X", ha="center", va="center", fontsize=8.3)
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, r"Yes $\rightarrow$ Y", ha="center", va="center", fontsize=8.3)

    ax.text((x1 + x2) / 2, (y1 + y2) / 2, "Keep both", ha="center", va="center", fontsize=8.5)
    ax.text((x2 + x3) / 2, (y1 + y2) / 2 + 0.035, "Choose X", ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text((x2 + x3) / 2, (y1 + y2) / 2 - 0.075, "consistent", ha="center", va="center", fontsize=7.4)
    ax.text((x1 + x2) / 2, (y0 + y1) / 2 + 0.035, "Choose Y", ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text((x1 + x2) / 2, (y0 + y1) / 2 - 0.075, "consistent", ha="center", va="center", fontsize=7.4)
    ax.text((x2 + x3) / 2, (y0 + y1) / 2, "Trade both", ha="center", va="center", fontsize=8.5)

    fig.savefig(FIGURES / "task_overview.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / "task_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_checkpoint_curve() -> None:
    data = json.loads((ROOT / "data" / "checkpoint_seed42_curve.json").read_text())
    points = data["points"]
    steps = [point["step"] / 1000 for point in points]

    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.axvline(data["selected_step"] / 1000, color="#999999", linewidth=0.8, linestyle="--")
    ax.plot(
        steps,
        [point["lambda"] for point in points],
        color="black",
        marker="o",
        markersize=3.8,
        linewidth=1.0,
        label=r"$\widehat{\lambda}$",
    )
    ax.plot(
        steps,
        [point["eta"] for point in points],
        color="#555555",
        marker="s",
        markersize=3.5,
        linewidth=1.0,
        linestyle="--",
        label=r"$\widehat{\eta}$",
    )
    ax.plot(
        steps,
        [point["d"] for point in points],
        color="#888888",
        marker="^",
        markersize=3.7,
        linewidth=1.0,
        linestyle=":",
        label=r"$d$",
    )
    ax.annotate(
        "selected",
        xy=(8, 0.516),
        xytext=(10.2, 0.36),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "black"},
        fontsize=8.5,
        ha="left",
    )
    ax.set_xlabel("Training step (thousands)")
    ax.set_ylabel("Validation estimate")
    ax.set_xlim(4, 31)
    ax.set_ylim(-0.16, 1.36)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.legend(frameon=False, ncol=3, loc="upper left", columnspacing=0.8, handlelength=1.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.35)
    fig.savefig(FIGURES / "checkpoint_curve.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / "checkpoint_curve.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    make_task_overview()
    make_checkpoint_curve()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
