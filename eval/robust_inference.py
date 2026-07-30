#!/usr/bin/env python3
"""Robust structural inference for lambda-zero (Task 4).

This is a SEPARATE, documented robustness layer. It does NOT modify or replace
the frozen headline estimator (`eval/core_exp_refactored.py`, Model A full fixed
effects), which remains the reference for point estimates. It adds three things
the frozen estimator lacks (INFER-001, PAIR-001):

  1. ID-based X/Y joins with strict integrity assertions.
  2. Pair-clustered bootstrap intervals for (lambda, eta).
  3. Leave-one-good-out / good-aware robustness, plus Jacobian conditioning and
     multi-start diagnostics.

Design / documented limitation
-------------------------------
The frozen estimator jointly fits (lambda, eta, alpha item FE, beta attribute
FE). Re-fitting ~109 parameters inside every bootstrap replicate is expensive
and couples to the frozen engine's file I/O. Instead, this robustness layer
CONDITIONS on plug-in per-case utilities V_X = exp(U_X), V_Y = exp(U_Y) (the same
alpha+beta object the frozen estimator produces), and re-estimates only
(lambda, eta) via the identical structural link:

    X-perspective (endowed X): z = (1+lambda) * V_X - V_Y + eta,  target = P(No|X)
    Y-perspective (endowed Y): z = (1+lambda) * V_Y - V_X + eta,  target = P(No|Y)
    P(No) = sigmoid(z / T)          (T = 1, matching Model A)

This isolates exactly what INFER-001 is about: (lambda, eta) uncertainty under
paired perspectives, repeated pairs, and shared goods, which the frozen
iid `curve_fit` covariance ignores. Fully joint FE + clustered inference is a
further extension noted in the module docstring, not claimed here.

Small checks run interactively; the full high-replicate bootstrap and the
recovery grid are CPU-only PBS jobs (train/submit_cpu_bootstrap.pbs,
train/submit_cpu_recovery.pbs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

T_LINK = 1.0


# ---------------------------------------------------------------------------
# 1. Strict ID-based X/Y join (PAIR-001)
# ---------------------------------------------------------------------------

@dataclass
class PairedCase:
    case_id: int
    x_num: int
    y_num: int
    attr: tuple
    pno_x: float   # P(No) when endowed with X (keep X)
    pno_y: float   # P(No) when endowed with Y (keep Y)


def _parse_pno(entry: dict) -> float:
    """P(No) from a loss_aversion_{X,Y} record. Accepts an explicit
    'Yes / No prob' = [yes, no] or a textual response."""
    if "Yes / No prob" in entry:
        yes, no = entry["Yes / No prob"]
        return float(no)
    resp = str(entry.get("response", "")).strip().lower()
    if resp.startswith("no") or resp.startswith("n"):
        return 0.75
    if resp.startswith("yes") or resp.startswith("y"):
        return 0.25
    raise ValueError(f"case {entry.get('case_id')}: cannot parse P(No)")


def load_paired_predictions(x_path: str | Path, y_path: str | Path,
                            id_key: str = "case_id") -> list[PairedCase]:
    """Join X-perspective and Y-perspective prediction files by stable case ID
    with hard integrity assertions (PAIR-001). Raises on: missing IDs, duplicate
    IDs, asymmetric pairing, or mismatched goods/attributes."""
    with open(x_path) as fh:
        xs = json.load(fh)
    with open(y_path) as fh:
        ys = json.load(fh)
    return join_paired_records(xs, ys, id_key=id_key)


def join_paired_records(xs: Sequence[dict], ys: Sequence[dict],
                        id_key: str = "case_id") -> list[PairedCase]:
    def index(records, side):
        out = {}
        for r in records:
            if id_key not in r:
                raise ValueError(f"{side} record missing '{id_key}': {r!r:.120}")
            cid = int(r[id_key])
            if cid in out:
                raise ValueError(f"{side} has duplicate {id_key}={cid}")
            out[cid] = r
        return out

    xi, yi = index(xs, "X"), index(ys, "Y")
    if set(xi) != set(yi):
        only_x = sorted(set(xi) - set(yi))[:5]
        only_y = sorted(set(yi) - set(xi))[:5]
        raise ValueError(f"X/Y case-ID sets differ: X-only={only_x}, Y-only={only_y}")

    cases: list[PairedCase] = []
    for cid in sorted(xi):
        xr, yr = xi[cid], yi[cid]
        for f in ("X_num", "Y_num", "attr"):
            if f in xr and f in yr and xr[f] != yr[f]:
                raise ValueError(f"case {cid}: {f} mismatch between X and Y records "
                                 f"({xr[f]!r} vs {yr[f]!r})")
        x_num = int(xr["X_num"]); y_num = int(xr["Y_num"])
        if not 0 <= x_num < y_num:
            raise ValueError(f"case {cid}: expected 0<=X_num<Y_num, got {x_num},{y_num}")
        cases.append(PairedCase(
            case_id=cid, x_num=x_num, y_num=y_num,
            attr=tuple(xr.get("attr", ())),
            pno_x=_parse_pno(xr), pno_y=_parse_pno(yr),
        ))
    if not cases:
        raise ValueError("no paired cases after join")
    return cases


# ---------------------------------------------------------------------------
# 2. Conditional (lambda, eta) estimator with the frozen link
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    lam: float
    eta: float
    success: bool
    cost: float
    n_cases: int
    jac_cond: float = np.nan
    multistart_lam_sd: float = np.nan
    multistart_eta_sd: float = np.nan
    message: str = ""


def _residuals(params, vx, vy, pno_x, pno_y):
    lam, eta = params
    zx = (1.0 + lam) * vx - vy + eta
    zy = (1.0 + lam) * vy - vx + eta
    px = 1.0 / (1.0 + np.exp(-zx / T_LINK))
    py = 1.0 / (1.0 + np.exp(-zy / T_LINK))
    return np.concatenate([px - pno_x, py - pno_y])


def fit_lambda_eta(vx, vy, pno_x, pno_y,
                   starts: Sequence[tuple] = ((0.0, 0.0), (1.0, 0.0), (5.0, 1.0)),
                   ) -> FitResult:
    """NLS fit of (lambda, eta) given plug-in utilities. Multi-start; reports the
    best fit, Jacobian condition number, and multi-start dispersion."""
    vx = np.asarray(vx, float); vy = np.asarray(vy, float)
    pno_x = np.asarray(pno_x, float); pno_y = np.asarray(pno_y, float)
    n = len(vx)
    best = None
    lams, etas = [], []
    for s in starts:
        try:
            r = least_squares(_residuals, x0=np.array(s, float),
                              args=(vx, vy, pno_x, pno_y), method="lm", max_nfev=2000)
        except Exception as exc:  # numerical failure on a start
            continue
        lams.append(r.x[0]); etas.append(r.x[1])
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return FitResult(np.nan, np.nan, False, np.inf, n, message="all starts failed")
    try:
        _, sv, _ = np.linalg.svd(best.jac, full_matrices=False)
        jac_cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    except Exception:
        jac_cond = np.nan
    return FitResult(
        lam=float(best.x[0]), eta=float(best.x[1]), success=bool(best.success),
        cost=float(best.cost), n_cases=n, jac_cond=jac_cond,
        multistart_lam_sd=float(np.std(lams)) if len(lams) > 1 else 0.0,
        multistart_eta_sd=float(np.std(etas)) if len(etas) > 1 else 0.0,
        message=str(best.message),
    )


# ---------------------------------------------------------------------------
# 3. Pair-clustered bootstrap + leave-one-good-out
# ---------------------------------------------------------------------------

@dataclass
class BootResult:
    point: FitResult
    lam_ci: tuple = (np.nan, np.nan)
    eta_ci: tuple = (np.nan, np.nan)
    lam_se: float = np.nan
    eta_se: float = np.nan
    n_replicates: int = 0
    n_failed: int = 0
    n_clusters: int = 0
    replicate_lams: list = field(default_factory=list)
    replicate_etas: list = field(default_factory=list)


def build_pair_clusters(cases: list[PairedCase]) -> tuple[list, list]:
    """Group case indices into goods-pair CLUSTERS keyed by (X_num, Y_num).

    Every configuration of a pair, with BOTH perspectives, lives in one cluster,
    so a pair's repeated configurations can never be split across bootstrap
    replicates. Returns (cluster_keys, list_of_index_arrays)."""
    clusters: dict[tuple, list[int]] = {}
    for i, c in enumerate(cases):
        clusters.setdefault((c.x_num, c.y_num), []).append(i)
    keys = sorted(clusters)
    return keys, [np.array(clusters[k], dtype=int) for k in keys]


def pair_cluster_resample_index(cluster_index_arrays: list, rng) -> np.ndarray:
    """Resample WHOLE goods-pair clusters with replacement and concatenate their
    member indices. A selected pair contributes all its configurations/perspectives
    together (and its full block again if drawn more than once)."""
    n_clusters = len(cluster_index_arrays)
    chosen = rng.integers(0, n_clusters, size=n_clusters)
    return np.concatenate([cluster_index_arrays[c] for c in chosen])


def pair_clustered_bootstrap(cases: list[PairedCase], vx, vy,
                             n_boot: int = 1000, seed: int = 0,
                             alpha: float = 0.05) -> BootResult:
    """Pair-clustered bootstrap: the cluster is one goods pair keyed by
    (X_num, Y_num). Every replicate resamples WHOLE pairs with replacement, so
    all configurations and both perspectives of a pair move together and are
    never split across clusters. This respects the paired/repeated structure the
    frozen iid covariance ignores (INFER-001)."""
    vx = np.asarray(vx, float); vy = np.asarray(vy, float)
    pno_x = np.array([c.pno_x for c in cases]); pno_y = np.array([c.pno_y for c in cases])
    point = fit_lambda_eta(vx, vy, pno_x, pno_y)
    _, cluster_arrays = build_pair_clusters(cases)
    rng = np.random.default_rng(seed)
    lams, etas, failed = [], [], 0
    for _ in range(n_boot):
        idx = pair_cluster_resample_index(cluster_arrays, rng)  # whole-pair resample
        r = fit_lambda_eta(vx[idx], vy[idx], pno_x[idx], pno_y[idx])
        if r.success and np.isfinite(r.lam):
            lams.append(r.lam); etas.append(r.eta)
        else:
            failed += 1
    lams = np.array(lams); etas = np.array(etas)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return BootResult(
        point=point,
        lam_ci=(float(np.percentile(lams, lo)), float(np.percentile(lams, hi))) if lams.size else (np.nan, np.nan),
        eta_ci=(float(np.percentile(etas, lo)), float(np.percentile(etas, hi))) if etas.size else (np.nan, np.nan),
        lam_se=float(np.std(lams, ddof=1)) if lams.size > 1 else np.nan,
        eta_se=float(np.std(etas, ddof=1)) if etas.size > 1 else np.nan,
        n_replicates=int(lams.size), n_failed=int(failed),
        n_clusters=len(cluster_arrays),
        replicate_lams=lams.tolist(), replicate_etas=etas.tolist(),
    )


def leave_one_good_out(cases: list[PairedCase], vx, vy) -> dict:
    """Drop every case involving each good in turn; refit; report the (lambda,
    eta) spread. A good whose removal moves lambda a lot is influential."""
    vx = np.asarray(vx, float); vy = np.asarray(vy, float)
    goods = sorted({c.x_num for c in cases} | {c.y_num for c in cases})
    rows = []
    for g in goods:
        keep = np.array([c.x_num != g and c.y_num != g for c in cases])
        if keep.sum() < 10:
            continue
        pno_x = np.array([c.pno_x for c in cases])[keep]
        pno_y = np.array([c.pno_y for c in cases])[keep]
        r = fit_lambda_eta(vx[keep], vy[keep], pno_x, pno_y)
        rows.append({"good": g, "n_cases": int(keep.sum()), "lam": r.lam,
                     "eta": r.eta, "success": r.success})
    lams = [r["lam"] for r in rows if r["success"] and np.isfinite(r["lam"])]
    return {
        "per_good": rows,
        "lam_min": float(np.min(lams)) if lams else np.nan,
        "lam_max": float(np.max(lams)) if lams else np.nan,
        "lam_range": float(np.max(lams) - np.min(lams)) if lams else np.nan,
        "most_influential_good": max(
            rows, key=lambda r: abs(r["lam"]) if r["success"] and np.isfinite(r["lam"]) else -1,
            default=None),
    }


if __name__ == "__main__":  # tiny self-demo on synthetic data
    rng = np.random.default_rng(1)
    n = 400
    vx = np.exp(rng.normal(0, 0.4, n)); vy = np.exp(rng.normal(0, 0.4, n))
    lam_true, eta_true = 2.0, 0.5
    zx = (1 + lam_true) * vx - vy + eta_true
    zy = (1 + lam_true) * vy - vx + eta_true
    pno_x = 1 / (1 + np.exp(-zx)); pno_y = 1 / (1 + np.exp(-zy))
    fit = fit_lambda_eta(vx, vy, pno_x, pno_y)
    print(f"recovered lambda={fit.lam:.3f} (true 2.0), eta={fit.eta:.3f} (true 0.5), "
          f"jac_cond={fit.jac_cond:.1f}")
