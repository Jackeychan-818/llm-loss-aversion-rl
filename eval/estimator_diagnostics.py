#!/usr/bin/env python3
"""Non-gating structural-estimator diagnostics for one evaluated checkpoint.

Answers PAPER_READINESS #5's "optimization path and structural utility
diagnostics (post-hoc)" and #9's multi-start/conditioning requirement WITHOUT
modifying eval/core_exp_refactored.py and WITHOUT overwriting the
claim-carrying estimate CSVs.

Two tiers, because one residual evaluation costs ~0.3 s over ~20k rows and the
finite-difference Jacobian needs ~n+1 of them (~40 s), so a full re-optimization
from many starts is a batch job, not an interactive one:

  DEFAULT (fast, ~40 s) — read the ORIGINAL committed fit (the NLS estimation
  CSV) and report, at that solution:
    * OBJECTIVE (#2): RSS at the solution, the Jacobian condition number
      cond(J) (the identification signal), reconstructed pcov / standard
      errors, and a check that they match the committed SEs;
    * ALPHA / BETA / UTILITY (#3): ranges + percentiles of the item FE (alpha),
      attribute-profile FE (beta), and utility grid U = exp(alpha + beta),
      i.e. the "original" fitted structural quantities;
    * STARTING POINTS: the objective at each numerical start (ols / zero /
      perturbed) — how far each start sits from the solution.

  --multistart (slow, minutes/checkpoint) — actually RE-OPTIMIZE from each
  start and report whether they converge to the same (lambda, eta, alpha). This
  is the true starting-point-sensitivity test; run it as a batch job.

NON-GATING: never selects a checkpoint, never opens OOD, never alters a frozen
threshold. Reads teacher-forced eval output + the estimation CSV that already
exist under FEATURE/MODEL_NAME.

    module load pytorch/...; source .../venv/bin/activate
    # fast solution diagnostics:
    python eval/estimator_diagnostics.py \
        --feature baseline --model_name Qwen-7B-GRPO-qd-ckpt8000 \
        --base_feature baseline --base_model Qwen-7B-base-matched \
        --out results/estimator_diagnostics/qd_ckpt8000.json
    # add the slow multi-start sensitivity (batch):
    python eval/estimator_diagnostics.py ... --multistart --n_perturb 5
"""
from __future__ import annotations

import argparse
import csv as _csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from core_exp_refactored import LossAversionModel  # noqa: E402

LINK_SCALE = 1.0  # Model A structural link scale T=1 (NOT a sampling temperature)


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_model(feature: str, model_name: str) -> LossAversionModel:
    for fname in ("loss_aversion_X.json", "loss_aversion_Y.json"):
        p = PROJECT_ROOT / feature / model_name / fname
        if not p.exists():
            raise FileNotFoundError(f"missing evaluator output: {p}")
    return LossAversionModel(
        Model_name=model_name, feature=feature, robust_model=1,
        input_X="loss_aversion_X.json", input_Y="loss_aversion_Y.json",
        T=LINK_SCALE, starting="ols",
    )


def read_committed_fit(feature: str, model_name: str):
    """Return (popt, committed_se, param_names) from the claim-carrying NLS CSV.

    Row order is [lambda, eta, alpha_2..alpha_n, beta_1,2..beta_3,3], matching
    the packed vector core_exp_refactored expects."""
    hits = glob.glob(str(PROJECT_ROOT / feature / model_name /
                        "Model_1" / "*NLS_estimation*.csv"))
    if not hits:
        raise FileNotFoundError(f"no NLS estimation CSV under {feature}/{model_name}/Model_1")
    rows = list(_csv.DictReader(open(hits[0])))
    popt = np.array([float(r["Estimate"]) for r in rows])
    se = np.array([float(r["Std. Err."]) for r in rows])
    names = [r["Parameter"] for r in rows]
    return popt, se, names, Path(hits[0])


def residual_fn(model: LossAversionModel):
    def resid(theta):
        return model.model_function_nonlinear_curvefit(model.W_data, *theta) - model.y_data
    return resid


def unpack_alpha_beta(model: LossAversionModel, popt: np.ndarray):
    """Prepend reference levels alpha_1 = 0 and beta_1,1 = 0 to get the full
    identified vectors used to build utilities."""
    n = model.num_items
    alphas = np.concatenate([[0.0], popt[2:2 + n - 1]])
    betas = np.concatenate([[0.0], popt[2 + n - 1:]])
    return alphas, betas


def utility_grid(alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """U(item, attr) = exp(alpha_item + beta_attr) over the full item x
    attribute-profile grid (matches U = exp(alpha + beta) in the spec)."""
    return np.exp(alphas[:, None] + betas[None, :]).ravel()


def summarize_vector(v: np.ndarray) -> dict:
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return {"min": None, "max": None, "n_nonfinite": int(v.size)}
    return {"min": float(np.min(finite)), "max": float(np.max(finite)),
            "p01": float(np.percentile(finite, 1)), "p50": float(np.percentile(finite, 50)),
            "p99": float(np.percentile(finite, 99)), "n_nonfinite": int(v.size - finite.size)}


def covariance_from_jac(jac: np.ndarray, sse: float, m: int, n: int):
    """cov = (J^T J)^-1 * s^2, s^2 = SSE / (m - n) — the curve_fit covariance."""
    dof = max(1, m - n)
    jtj = jac.T @ jac
    try:
        cov = np.linalg.inv(jtj) * (sse / dof)
        cond_pcov = float(np.linalg.cond(cov))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(jtj) * (sse / dof)
        cond_pcov = None
    diag = np.diag(cov)
    se = np.sqrt(np.where(diag >= 0, diag, np.nan))
    return se, cond_pcov


def bounds(n: int):
    return ([-1.0, -200.0] + [-np.inf] * (n - 2),
            [200.0, 200.0] + [np.inf] * (n - 2))


# ── solution-level diagnostics on the ORIGINAL committed fit ──────────────────
def solution_diagnostics(model, popt, committed_se):
    resid = residual_fn(model)
    m, n = model.y_data.shape[0], popt.size
    r0 = resid(popt)
    sse = float(np.sum(r0 ** 2))
    jac = approx_derivative(resid, popt, method="2-point")     # one ~n-eval Jacobian
    try:
        cond_jac = float(np.linalg.cond(jac))
    except np.linalg.LinAlgError:
        cond_jac = None
    se, cond_pcov = covariance_from_jac(jac, sse, m, n)
    alphas, betas = unpack_alpha_beta(model, popt)
    util = utility_grid(alphas, betas)
    se_finite = np.isfinite(se)
    committed = committed_se[se_finite]
    ours = se[se_finite]
    se_rel = float(np.max(np.abs(ours - committed) / (np.abs(committed) + 1e-12))) if ours.size else None
    return {
        "rss_at_solution": sse,
        "lambda": float(popt[0]), "eta": float(popt[1]),
        "cond_jacobian": cond_jac, "cond_pcov": cond_pcov,
        "se_all_finite_positive": bool(np.all(np.isfinite(se)) and np.all(se > 0)),
        "se_max_rel_diff_vs_committed": se_rel,
        "alpha": summarize_vector(alphas), "beta": summarize_vector(betas),
        "utility": summarize_vector(util),
        "_alphas": alphas, "_utility": util,
    }


def make_starts(ols_init, n_perturb, seed, scale):
    starts = [("ols", ols_init.copy()), ("zero", np.zeros_like(ols_init))]
    rng = np.random.default_rng(seed)
    for k in range(n_perturb):
        starts.append((f"perturb{k + 1}", ols_init + rng.normal(0, scale, ols_init.shape)))
    return starts


def start_objectives(model, starts):
    """Cheap: objective at each start (1 eval each), no optimization."""
    resid = residual_fn(model)
    out = []
    for name, s in starts:
        lo, up = bounds(s.size)
        sc = np.clip(s, lo, up)
        out.append({"start": name, "start_rss": float(np.sum(resid(sc) ** 2)),
                    "lambda_start": float(sc[0]), "eta_start": float(sc[1])})
    return out


def multistart_refits(model, starts, max_nfev):
    """Slow: fully re-optimize from each start; the true sensitivity test."""
    resid = residual_fn(model)
    m = model.y_data.shape[0]
    fits = []
    for name, s in starts:
        n = s.size
        lo, up = bounds(n)
        t0 = time.time()
        res = least_squares(resid, x0=np.clip(s, lo, up), bounds=(lo, up),
                            method="trf", max_nfev=max_nfev)
        sse = float(np.sum(res.fun ** 2))
        se, cond_pcov = covariance_from_jac(res.jac, sse, m, n)
        try:
            cond_jac = float(np.linalg.cond(res.jac))
        except np.linalg.LinAlgError:
            cond_jac = None
        alphas, betas = unpack_alpha_beta(model, res.x)
        fits.append({
            "start": name, "converged": bool(res.status > 0), "status": int(res.status),
            "hit_nfev_cap": bool(res.status == 0), "nfev": int(res.nfev),
            "seconds": round(time.time() - t0, 1),
            "final_rss": sse, "lambda": float(res.x[0]), "eta": float(res.x[1]),
            "cond_jacobian": cond_jac, "cond_pcov": cond_pcov,
            "se_all_finite_positive": bool(np.all(np.isfinite(se)) and np.all(se > 0)),
            "alpha": summarize_vector(alphas), "utility": summarize_vector(utility_grid(alphas, betas)),
        })
    return fits


def spread(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return {"n": len(vals), "min": (vals[0] if vals else None),
                "max": (vals[0] if vals else None), "spread": 0.0 if vals else None}
    return {"n": len(vals), "min": float(min(vals)), "max": float(max(vals)),
            "spread": float(max(vals) - min(vals))}


def spearman(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    ra = np.argsort(np.argsort(a[mask])).astype(float)
    rb = np.argsort(np.argsort(b[mask])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    return float(np.sum(ra * rb) / denom) if denom > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature", default="baseline")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--base_feature", default=None)
    ap.add_argument("--base_model", default=None,
                    help="comparator checkpoint for utility rank preservation (e.g. matched base)")
    ap.add_argument("--multistart", action="store_true",
                    help="also re-optimize from every start (SLOW; batch job)")
    ap.add_argument("--n_perturb", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb_scale", type=float, default=0.5)
    ap.add_argument("--max_nfev", type=int, default=3000,
                    help="per-refit residual-eval cap for --multistart (status=0 = cap hit)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = build_model(args.feature, args.model_name)
    popt, committed_se, names, csv_path = read_committed_fit(args.feature, args.model_name)
    if popt.size != model.num_items + 9:
        raise ValueError(f"CSV has {popt.size} params but model expects {model.num_items + 9}; "
                         "num_items/CSV mismatch")

    sol = solution_diagnostics(model, popt, committed_se)

    model.initialize_parameters()
    ols_init = np.asarray(model.initial_params, float).copy()
    starts = make_starts(ols_init, args.n_perturb, args.seed, args.perturb_scale)
    starts_obj = start_objectives(model, starts)

    multistart = None
    if args.multistart:
        fits = multistart_refits(model, starts, args.max_nfev)
        multistart = {
            "fits": fits,
            "lambda_spread": spread([f["lambda"] for f in fits]),
            "eta_spread": spread([f["eta"] for f in fits]),
            "alpha_max_spread": spread([f["alpha"]["max"] for f in fits]),
            "n_converged": sum(f["converged"] for f in fits),
            "n_starts": len(fits),
            "all_starts_agree_lambda_0p05": bool(
                spread([f["lambda"] for f in fits])["spread"] is not None
                and spread([f["lambda"] for f in fits])["spread"] <= 0.05),
        }

    rank_pres = None
    if args.base_feature and args.base_model:
        base = build_model(args.base_feature, args.base_model)
        bpopt, _, _, _ = read_committed_fit(args.base_feature, args.base_model)
        b_alphas, b_betas = unpack_alpha_beta(base, bpopt)
        rank_pres = {
            "base": f"{args.base_feature}/{args.base_model}",
            "spearman_alpha_items": spearman(sol["_alphas"], b_alphas),
            "spearman_utility_grid": spearman(sol["_utility"], utility_grid(b_alphas, b_betas)),
        }

    sol.pop("_alphas", None); sol.pop("_utility", None)
    result = {
        "feature": args.feature, "model_name": args.model_name,
        "estimator": "Model A (NLS), structural link scale T=1",
        "committed_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "num_items": int(model.num_items), "num_attr_combos": int(model.num_attr_combos),
        "n_params": int(popt.size), "n_obs": int(model.y_data.shape[0]),
        "gating": "NON-GATING diagnostic; does not select checkpoints or touch frozen thresholds",
        "solution_diagnostics": sol,
        "start_objectives": starts_obj,
        "multistart": multistart,
        "rank_preservation": rank_pres,
        "config": {"n_perturb": args.n_perturb, "seed": args.seed,
                   "perturb_scale": args.perturb_scale, "max_nfev": args.max_nfev,
                   "multistart": args.multistart, "link_scale": LINK_SCALE},
    }

    # ── readable summary ──────────────────────────────────────────────────────
    print("=" * 78)
    print(f"ESTIMATOR DIAGNOSTICS (non-gating) — {args.feature}/{args.model_name}")
    print(f"n_obs={result['n_obs']}  n_params={result['n_params']}  (original committed fit)")
    print("=" * 78)
    print(f"OBJECTIVE / CONDITIONING at the solution:")
    print(f"  lambda={sol['lambda']:+.4f}  eta={sol['eta']:+.4f}  RSS={sol['rss_at_solution']:.4f}")
    print(f"  cond(Jacobian)={sol['cond_jacobian']:.3e}  cond(pcov)="
          f"{(sol['cond_pcov'] if sol['cond_pcov'] is not None else float('nan')):.3e}")
    print(f"  SEs finite&positive={sol['se_all_finite_positive']}  "
          f"max rel diff vs committed SE={sol['se_max_rel_diff_vs_committed']}")
    print(f"ALPHA / BETA / UTILITY (original fit):")
    print(f"  alpha   range [{sol['alpha']['min']:.3f}, {sol['alpha']['max']:.3f}]  "
          f"(p01={sol['alpha']['p01']:.3f}, p99={sol['alpha']['p99']:.3f})")
    print(f"  beta    range [{sol['beta']['min']:.3f}, {sol['beta']['max']:.3f}]")
    print(f"  utility range [{sol['utility']['min']:.3f}, {sol['utility']['max']:.3f}]  "
          f"(p99={sol['utility']['p99']:.3f})")
    print(f"STARTING-POINT objectives (start RSS; +--multistart to re-optimize):")
    for s in starts_obj:
        print(f"  {s['start']:<9} start_rss={s['start_rss']:.4f}")
    if multistart:
        print("MULTI-START re-optimization:")
        print(f"  {'start':<9}{'lambda':>9}{'eta':>9}{'finalRSS':>11}{'nfev':>7}{'sec':>7}{'conv':>6}")
        for f in multistart["fits"]:
            print(f"  {f['start']:<9}{f['lambda']:>9.3f}{f['eta']:>9.3f}{f['final_rss']:>11.3f}"
                  f"{f['nfev']:>7}{f['seconds']:>7.0f}{('yes' if f['converged'] else 'CAP'):>6}")
        print(f"  lambda spread across starts={multistart['lambda_spread']['spread']}  "
              f"agree<=0.05: {multistart['all_starts_agree_lambda_0p05']}")
    if rank_pres:
        print(f"RANK PRESERVATION vs {rank_pres['base']}: "
              f"spearman(alpha)={rank_pres['spearman_alpha_items']}, "
              f"spearman(utility)={rank_pres['spearman_utility_grid']}")

    if args.out:
        out = resolve_project_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
        print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
