#!/usr/bin/env python3
"""Post-hoc SFT answer-space divergence from the matched base (CPU-only).

Measures how far each full-run SFT checkpoint moved from the exact matched local
Qwen base, in the BINARY Yes/No answer space, from EXISTING teacher-forced
`test_goods` validation predictions. This is NOT the SFT training objective (the
SFT runs had NO explicit KL penalty and nothing like this was logged during
training), NOT full-vocabulary KL, and NOT general-language policy divergence.

Primary statistic, per prompt, in natural-log units (nats):
    KL(SFT || base) = P_yes*log(P_yes/Q_yes) + P_no*log(P_no/Q_no)
with P = SFT [Yes,No], Q = base [Yes,No].

Two stages:
  --refresh : read the untracked NSCC prediction arrays, strictly validate +
              join by (perspective, case_id), compute per-checkpoint statistics
              + pair-clustered bootstrap CI, and write the canonical snapshot +
              manifest (with source paths & SHA-256).
  default   : render CSV / Markdown / PNG from the tracked snapshot (no NSCC).
  --check   : render in memory and verify CSV/MD are byte-identical + snapshot
              present + PNG renders; works from a clean git archive.

No model is loaded; no inference is run; the frozen estimator is untouched;
pilot / OOD / frozen-unused / semantic suites are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "baseline" / "Qwen-7B-Base-Local"
SFT_ROOT = ROOT / "baselines"
OUT_DIR = ROOT / "results" / "training_dynamics" / "sft"
SNAPSHOT = OUT_DIR / "sft_answer_divergence_snapshot.json"
MANIFEST = OUT_DIR / "sft_answer_divergence_manifest.json"
CSV = OUT_DIR / "sft_answer_divergence.csv"
MD = OUT_DIR / "sft_answer_divergence.md"
PNG = OUT_DIR / "sft_answer_divergence.png"

SEEDS = [1, 2]
STEPS = list(range(2000, 30001, 2000))       # 2000..30000 by 2000 => 15
EXPECTED_PROMPTS = 19_780                     # 9890 X + 9890 Y
EPS = 1e-12
BOOT_SEED = 20260814
BOOT_N = 2000
SELECTED = {1: 4000, 2: 6000}                # frozen-selected checkpoints (markers)

PROB_KEY = "Yes / No prob"


# --------------------------------------------------------------------------- #
# Core math (importable, unit-tested)
# --------------------------------------------------------------------------- #
def validate_prob(arr) -> tuple[float, float]:
    """Validate a [P(Yes), P(No)] array and return it normalized. Raises on any
    violation. Does NOT clip here — clipping happens only in KL, near zeros."""
    if not isinstance(arr, (list, tuple)) or len(arr) != 2:
        raise ValueError(f"prob must be a 2-element array, got {arr!r}")
    y, n = arr
    if not (isinstance(y, (int, float)) and isinstance(n, (int, float))):
        raise ValueError(f"prob entries must be numbers, got {arr!r}")
    if not (math.isfinite(y) and math.isfinite(n)):
        raise ValueError(f"prob entries must be finite, got {arr!r}")
    if y < 0 or n < 0:
        raise ValueError(f"prob entries must be nonnegative, got {arr!r}")
    total = y + n
    if not (total > 0):
        raise ValueError(f"prob total must be positive, got {arr!r}")
    return y / total, n / total


def _clip_renorm(y: float, n: float, eps: float) -> tuple[float, float, bool]:
    clipped = (y < eps) or (n < eps)
    if clipped:
        y = max(y, eps)
        n = max(n, eps)
        t = y + n
        y, n = y / t, n / t
    return y, n, clipped


def kl_binary(p_yes, p_no, q_yes, q_no, eps=EPS):
    """KL(P || Q) in nats for two-class distributions. Returns (kl, clipped)."""
    py, pn, c1 = _clip_renorm(p_yes, p_no, eps)
    qy, qn, c2 = _clip_renorm(q_yes, q_no, eps)
    kl = py * math.log(py / qy) + pn * math.log(pn / qn)
    return kl, (c1 or c2)


def js_binary(p_yes, p_no, q_yes, q_no, eps=EPS):
    """Jensen-Shannon divergence (nats) — SECONDARY, symmetric. NOT a KL."""
    py, pn, _ = _clip_renorm(p_yes, p_no, eps)
    qy, qn, _ = _clip_renorm(q_yes, q_no, eps)
    my, mn = 0.5 * (py + qy), 0.5 * (pn + qn)
    def _kl(ay, an, by, bn):
        return ay * math.log(ay / by) + an * math.log(an / bn)
    return 0.5 * _kl(py, pn, my, mn) + 0.5 * _kl(qy, qn, my, mn)


# --------------------------------------------------------------------------- #
# Strict loading + join
# --------------------------------------------------------------------------- #
def _reject_pilot(path: str) -> None:
    if "pilot" in str(path).lower():
        raise ValueError(f"PILOT path is explicitly forbidden: {path}")


def load_perspective(path: Path, perspective: str) -> dict:
    """Load one loss_aversion_{X,Y}.json into {(perspective, case_id): rec}."""
    _reject_pilot(path)
    if perspective not in ("X", "Y"):
        raise ValueError(f"perspective must be X or Y, got {perspective!r}")
    with open(path) as fh:
        rows = json.load(fh)
    out: dict = {}
    for r in rows:
        for k in ("case_id", "X_num", "Y_num", PROB_KEY):
            if k not in r:
                raise ValueError(f"{path}: row missing required key {k!r}")
        cid = int(r["case_id"])
        key = (perspective, cid)
        if key in out:
            raise ValueError(f"{path}: duplicate key {key}")
        y, n = validate_prob(r[PROB_KEY])
        out[key] = {"X_num": int(r["X_num"]), "Y_num": int(r["Y_num"]),
                    "yes": y, "no": n}
    return out


def load_model(x_path: Path, y_path: Path) -> dict:
    d = load_perspective(x_path, "X")
    dy = load_perspective(y_path, "Y")
    inter = set(d) & set(dy)
    if inter:
        raise ValueError(f"X/Y key collision (perspective tag failed): {list(inter)[:3]}")
    d.update(dy)
    nx = sum(1 for k in d if k[0] == "X")
    ny = sum(1 for k in d if k[0] == "Y")
    if nx != 9890 or ny != 9890:
        raise ValueError(f"expected 9890 X and 9890 Y rows, got X={nx} Y={ny}")
    if len(d) != EXPECTED_PROMPTS:
        raise ValueError(f"expected {EXPECTED_PROMPTS} prompts, got {len(d)}")
    return d


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Per-checkpoint computation
# --------------------------------------------------------------------------- #
def _clip_renorm_vec(y, n, eps):
    clipped = (y < eps) | (n < eps)
    y2 = np.maximum(y, eps); n2 = np.maximum(n, eps)
    t = y2 + n2
    return y2 / t, n2 / t, clipped


def compute_checkpoint(base: dict, ckpt: dict) -> dict:
    if set(base) != set(ckpt):
        miss = list(set(base) - set(ckpt))[:3]
        extra = list(set(ckpt) - set(base))[:3]
        raise ValueError(f"key set mismatch: missing_in_ckpt={miss} extra_in_ckpt={extra}")
    keys = sorted(base)
    b_xn = np.array([base[k]["X_num"] for k in keys])
    b_yn = np.array([base[k]["Y_num"] for k in keys])
    c_xn = np.array([ckpt[k]["X_num"] for k in keys])
    c_yn = np.array([ckpt[k]["Y_num"] for k in keys])
    if not (np.array_equal(b_xn, c_xn) and np.array_equal(b_yn, c_yn)):
        i = int(np.argmax((b_xn != c_xn) | (b_yn != c_yn)))
        raise ValueError(f"X_num/Y_num mismatch at {keys[i]}: base=({b_xn[i]},{b_yn[i]}) "
                         f"ckpt=({c_xn[i]},{c_yn[i]})")
    q_yes = np.array([base[k]["yes"] for k in keys]); q_no = np.array([base[k]["no"] for k in keys])
    p_yes = np.array([ckpt[k]["yes"] for k in keys]); p_no = np.array([ckpt[k]["no"] for k in keys])

    py, pn, cp = _clip_renorm_vec(p_yes, p_no, EPS)
    qy, qn, cq = _clip_renorm_vec(q_yes, q_no, EPS)
    kl = py * np.log(py / qy) + pn * np.log(pn / qn)
    my, mn = 0.5 * (py + qy), 0.5 * (pn + qn)
    js = 0.5 * (py * np.log(py / my) + pn * np.log(pn / mn)) \
        + 0.5 * (qy * np.log(qy / my) + qn * np.log(qn / mn))
    disagree = (p_yes >= p_no) != (q_yes >= q_no)
    clipped = int(np.count_nonzero(cp | cq))

    pair_ids = np.minimum(b_xn, b_yn) * 100 + np.maximum(b_xn, b_yn)   # encode unordered pair
    ci_lo, ci_hi = pair_clustered_bootstrap_ci(kl, pair_ids)
    n_clusters = int(len(np.unique(pair_ids)))
    return {
        "n_prompts": int(len(keys)),
        "n_pair_clusters": n_clusters,
        "clipped_rows": clipped,
        "mean_kl": float(np.mean(kl)),
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "median_kl": float(np.median(kl)),
        "p90_kl": float(np.percentile(kl, 90)),
        "max_kl": float(np.max(kl)),
        "js_mean": float(np.mean(js)),
        "hard_choice_disagreement": float(np.mean(disagree)),
    }


def pair_clusters(pair_ids) -> tuple[list, list]:
    """Group row indices into goods-pair clusters keyed by the unordered pair
    (min,max)(X_num,Y_num). Every repeated configuration and both X/Y
    perspectives of a pair land in ONE cluster. Returns (keys, index_arrays)."""
    clusters: dict = {}
    for i, pid in enumerate(pair_ids):
        clusters.setdefault(tuple(pid), []).append(i)
    keys = sorted(clusters)
    return keys, [np.asarray(clusters[k], dtype=int) for k in keys]


def pair_resample_index(cluster_arrays: list, rng) -> np.ndarray:
    """Resample WHOLE clusters with replacement and concatenate member indices."""
    n = len(cluster_arrays)
    chosen = rng.integers(0, n, size=n)
    return np.concatenate([cluster_arrays[c] for c in chosen])


def pair_clustered_bootstrap_ci(values: np.ndarray, pair_ids: np.ndarray,
                                n_boot=BOOT_N, seed=BOOT_SEED, alpha=0.05):
    """95% percentile CI for the mean, resampling WHOLE goods-pair clusters.

    Uses per-cluster sums/counts: the mean of a cluster-resample equals
    (sum of chosen cluster sums) / (sum of chosen cluster counts) — identical to
    concatenating each chosen cluster's rows and averaging, but vectorized."""
    values = np.asarray(values, dtype=float)
    uniq, inv = np.unique(pair_ids, return_inverse=True)
    ncl = len(uniq)
    csum = np.zeros(ncl); ccnt = np.zeros(ncl)
    np.add.at(csum, inv, values)
    np.add.at(ccnt, inv, 1.0)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, ncl, size=(n_boot, ncl))
    means = csum[chosen].sum(axis=1) / ccnt[chosen].sum(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


# --------------------------------------------------------------------------- #
# Refresh (reads untracked NSCC)  /  render / check (snapshot only)
# --------------------------------------------------------------------------- #
def sft_paths(seed: int, step: int) -> tuple[Path, Path]:
    d = SFT_ROOT / f"Qwen-7B-SFT-qd-seed{seed}-ckpt{step}"
    _reject_pilot(d)
    return d / "loss_aversion_X.json", d / "loss_aversion_Y.json"


def refresh() -> dict:
    bx, by = BASE_DIR / "loss_aversion_X.json", BASE_DIR / "loss_aversion_Y.json"
    for p in (bx, by):
        if not p.exists():
            raise SystemExit(f"MISSING required base file: {p}")
    base = load_model(bx, by)

    sources = {"base": {"X": {"path": str(bx.relative_to(ROOT)), "sha256": sha256_file(bx)},
                        "Y": {"path": str(by.relative_to(ROOT)), "sha256": sha256_file(by)}}}
    rows = []
    missing = []
    for seed in SEEDS:
        for step in STEPS:
            xp, yp = sft_paths(seed, step)
            if not xp.exists() or not yp.exists():
                missing.append(str(xp)); missing.append(str(yp)); continue
    if missing:
        raise SystemExit("MISSING required SFT files:\n  " + "\n  ".join(missing))

    for seed in SEEDS:
        for step in STEPS:
            xp, yp = sft_paths(seed, step)
            ckpt = load_model(xp, yp)
            stats = compute_checkpoint(base, ckpt)
            stats.update({"seed": seed, "step": step})
            rows.append(stats)
            sources[f"seed{seed}_ckpt{step}"] = {
                "X": {"path": str(xp.relative_to(ROOT)), "sha256": sha256_file(xp)},
                "Y": {"path": str(yp.relative_to(ROOT)), "sha256": sha256_file(yp)}}
            print(f"  seed{seed} step{step:>5}: mean_kl={stats['mean_kl']:.4f} "
                  f"[{stats['ci_lo']:.4f},{stats['ci_hi']:.4f}] "
                  f"hard_disagree={stats['hard_choice_disagreement']:.4f} "
                  f"clipped={stats['clipped_rows']}")

    if len(rows) != len(SEEDS) * len(STEPS):
        raise SystemExit(f"expected {len(SEEDS)*len(STEPS)} checkpoint rows, got {len(rows)}")

    snapshot = {
        "title": "Post-hoc SFT answer-space divergence from matched base",
        "definition": "KL(SFT || base) on binary Yes/No teacher-forced probabilities",
        "units": "nats (natural log)",
        "dataset_role": "test_goods validation (post-hoc); not a training loss",
        "epsilon": EPS,
        "bootstrap": {"method": "pair-clustered (unordered goods pair)",
                      "seed": BOOT_SEED, "resamples": BOOT_N, "interval": "percentile 95%"},
        "grid": {"seeds": SEEDS, "steps": STEPS},
        "selected_checkpoints": SELECTED,
        "base_row": {"seed": None, "step": 0, "mean_kl": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                     "median_kl": 0.0, "p90_kl": 0.0, "max_kl": 0.0, "js_mean": 0.0,
                     "hard_choice_disagreement": 0.0,
                     "n_prompts": EXPECTED_PROMPTS,
                     "n_pair_clusters": rows[0]["n_pair_clusters"], "clipped_rows": 0},
        "checkpoints": sorted(rows, key=lambda r: (r["seed"], r["step"])),
    }
    manifest = build_manifest(sources)
    return snapshot, manifest


def build_manifest(sources: dict) -> dict:
    import matplotlib
    return {
        "analysis": "Post-hoc SFT answer-space divergence from the exact matched local base.",
        "kl_direction": "KL(SFT || base)  (P=SFT, Q=base)",
        "units": "nats (natural logarithm)",
        "epsilon": EPS,
        "epsilon_use": "protect exact zero endpoints only; renormalize after clipping; clipped_rows recorded",
        "bootstrap": {"method": "pair-clustered on unordered goods pair (min,max)(X_num,Y_num)",
                      "seed": BOOT_SEED, "resamples": BOOT_N, "interval": "percentile 95%"},
        "required_grid": {"seeds": SEEDS, "steps": STEPS, "checkpoints_total": len(SEEDS) * len(STEPS)},
        "dataset_role": "test_goods VALIDATION / post-hoc (informed reward + selection)",
        "matched_base_path": str((BASE_DIR).relative_to(ROOT)),
        "raw_inputs": sources,
        "git_commit_at_refresh": _git_commit(),
        "versions": {"python": sys.version.split()[0],
                     "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "limitations": [
            "binary Yes/No answer-space divergence only",
            "not full-vocabulary KL",
            "not part of the SFT training objective (no explicit KL penalty was used or logged)",
            "not a confirmatory or untouched-suite result",
            "historical raw predictions lack cryptographic adapter-to-prediction binding",
        ],
    }


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# Rendering from snapshot
# --------------------------------------------------------------------------- #
CSV_COLS = ["seed", "step", "mean_kl", "ci_lo", "ci_hi", "median_kl", "p90_kl",
            "max_kl", "js_mean", "hard_choice_disagreement", "n_prompts",
            "n_pair_clusters", "clipped_rows"]


def render_csv(snap: dict) -> str:
    import io, csv as _csv
    buf = io.StringIO()
    w = _csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLS)
    base = dict(snap["base_row"]); base["seed"] = "base"; base["step"] = 0
    w.writerow([base[c] for c in CSV_COLS])
    for r in snap["checkpoints"]:
        w.writerow([r[c] for c in CSV_COLS])
    return buf.getvalue()


def render_md(snap: dict) -> str:
    rows = {(r["seed"], r["step"]): r for r in snap["checkpoints"]}
    def line(label, seed, step):
        r = rows[(seed, step)]
        return (f"| {label} | {seed} | {step} | {r['mean_kl']:.4f} "
                f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['median_kl']:.4f} | "
                f"{r['hard_choice_disagreement']:.4f} |")
    L = ["# Post-hoc SFT answer-space divergence from matched base", "",
         "Binary `KL(SFT || base)` on `test_goods` teacher-forced Yes/No "
         "probabilities, in **nats**. **This is not a training loss** (the SFT "
         "runs used no explicit KL penalty) and **not full-vocabulary policy KL**.",
         "", "## Key checkpoints",
         "", "| checkpoint | seed | step | mean KL [95% CI] | median KL | hard-choice disagree |",
         "|---|---|---|---|---|---|",
         line("seed 1 selected", 1, SELECTED[1]),
         line("seed 2 selected", 2, SELECTED[2]),
         line("seed 1 endpoint", 1, 30000),
         line("seed 2 endpoint", 2, 30000),
         "", "## Full grid", "",
         "| seed | step | mean KL | 95% CI | median | p90 | max | JS (2ndary) | hard-disagree | clipped |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    L.append("| base | 0 | 0.0000 | [0,0] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |")
    for r in snap["checkpoints"]:
        L.append(f"| {r['seed']} | {r['step']} | {r['mean_kl']:.4f} | "
                 f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['median_kl']:.4f} | "
                 f"{r['p90_kl']:.4f} | {r['max_kl']:.4f} | {r['js_mean']:.4f} | "
                 f"{r['hard_choice_disagreement']:.4f} | {r['clipped_rows']} |")
    L += ["", f"Prompts/model: {snap['checkpoints'][0]['n_prompts']}; "
          f"goods-pair clusters: {snap['checkpoints'][0]['n_pair_clusters']}; "
          f"bootstrap: {snap['bootstrap']['resamples']} pair-clustered resamples, "
          f"seed {snap['bootstrap']['seed']}.", ""]
    return "\n".join(L)


def render_png(snap: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = snap["checkpoints"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {1: "#1f5fd0", 2: "#d02b2b"}
    for seed in SEEDS:
        rs = sorted([r for r in rows if r["seed"] == seed], key=lambda r: r["step"])
        steps = [0] + [r["step"] for r in rs]
        mean = [0.0] + [r["mean_kl"] for r in rs]
        lo = [0.0] + [r["ci_lo"] for r in rs]
        hi = [0.0] + [r["ci_hi"] for r in rs]
        med = [0.0] + [r["median_kl"] for r in rs]
        ax.plot(steps, mean, "-", color=colors[seed], lw=2, label=f"seed {seed} mean KL")
        ax.fill_between(steps, lo, hi, color=colors[seed], alpha=0.15)
        ax.plot(steps, med, "--", color=colors[seed], lw=0.9, alpha=0.8)
        sel = SELECTED[seed]
        sr = next(r for r in rs if r["step"] == sel)
        ax.scatter([sel], [sr["mean_kl"]], color=colors[seed], edgecolor="black",
                   zorder=5, s=55, marker="o")
        ax.annotate(f"seed {seed} selected @{sel}", (sel, sr["mean_kl"]),
                    textcoords="offset points", xytext=(6, 8), fontsize=8, color=colors[seed])
    ax.set_xlim(0, 30000)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean binary KL(SFT || base)  [nats]")
    ax.set_title("Post-hoc SFT answer-space divergence from matched base")
    ax.text(0.5, 1.005, "Binary KL(SFT || base) on test_goods; not a training loss and "
            "not full-vocabulary policy KL",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#444")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_snapshot() -> dict:
    if not SNAPSHOT.exists():
        raise SystemExit(f"Snapshot missing: {SNAPSHOT.relative_to(ROOT)} — run --refresh on NSCC.")
    return json.loads(SNAPSHOT.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="Read untracked NSCC predictions and rewrite the snapshot/manifest.")
    ap.add_argument("--check", action="store_true",
                    help="Verify CSV/MD render byte-identically from the tracked snapshot; PNG re-renders.")
    args = ap.parse_args()

    if args.refresh:
        snap, manifest = refresh()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(snap, indent=2) + "\n")
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {SNAPSHOT.relative_to(ROOT)} + manifest")
    snap = load_snapshot()
    csv_s, md_s = render_csv(snap), render_md(snap)
    if args.check:
        bad = []
        if not CSV.exists() or CSV.read_text() != csv_s:
            bad.append("csv")
        if not MD.exists() or MD.read_text() != md_s:
            bad.append("md")
        render_png(snap, PNG)          # PNG re-renders from snapshot (no NSCC)
        if not PNG.exists() or PNG.stat().st_size == 0:
            bad.append("png")
        if bad:
            raise SystemExit("CHECK FAILED: " + ", ".join(bad))
        print("CHECK PASSED: snapshot present; CSV/MD byte-identical; PNG renders.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV.write_text(csv_s); MD.write_text(md_s); render_png(snap, PNG)
    print(f"Wrote {CSV.relative_to(ROOT)}, {MD.relative_to(ROOT)}, {PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
