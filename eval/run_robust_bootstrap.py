#!/usr/bin/env python3
"""Driver: pair-clustered bootstrap + leave-one-good-out on EXISTING predictions.

Loads an already-generated loss_aversion_X / loss_aversion_Y pair and the frozen
base utility table, builds plug-in utilities V = exp(U) per case, and runs the
robustness (lambda, eta) inference from eval/robust_inference.py. Generates NO
model responses and evaluates no checkpoint; it only reads files that exist.

    python3 eval/run_robust_bootstrap.py \
        --x baseline/.../loss_aversion_X_for_Model_A.json \
        --y baseline/.../loss_aversion_Y_for_Model_A.json \
        --utility baseline/Qwen-7B/Model_1/Qwen-7B_utility_of_each_goods_Model_A.csv \
        --n_boot 2000 --tag base --out results/robust_inference/bootstrap_base.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_inference import (  # noqa: E402
    load_paired_predictions, pair_clustered_bootstrap, leave_one_good_out,
)


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def load_utility(path: Path) -> dict:
    util = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            util[(int(row["index"]), int(row["attr_1"]), int(row["attr_2"]))] = float(row["utility"])
    return util


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--utility", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = load_paired_predictions(args.x, args.y)
    util = load_utility(Path(args.utility))

    vx, vy, missing = [], [], 0
    kept = []
    for c in cases:
        if len(c.attr) != 4:
            missing += 1
            continue
        i, j, k, l = c.attr
        ux = util.get((c.x_num + 1, i + 1, j + 1))
        uy = util.get((c.y_num + 1, k + 1, l + 1))
        if ux is None or uy is None:
            missing += 1
            continue
        vx.append(math.exp(ux)); vy.append(math.exp(uy)); kept.append(c)
    if not kept:
        raise SystemExit("No cases with a utility lookup; cannot bootstrap.")
    vx = np.array(vx); vy = np.array(vy)

    boot = pair_clustered_bootstrap(kept, vx, vy, n_boot=args.n_boot, seed=args.seed)
    logo = leave_one_good_out(kept, vx, vy)

    doc = {
        "tag": args.tag,
        "estimator": "conditional (lambda,eta) robustness layer; plug-in V=exp(U); T=1",
        "note": ("Separate from the frozen full-FE headline estimator. Bootstrap "
                 "resamples WHOLE goods-pair clusters keyed by (X_num,Y_num) — all "
                 "configurations and both perspectives of a pair move together — "
                 "addressing INFER-001 for (lambda,eta)."),
        "inputs": {
            "x": {"path": args.x, "sha256": sha256_file(Path(args.x))},
            "y": {"path": args.y, "sha256": sha256_file(Path(args.y))},
            "utility": {"path": args.utility, "sha256": sha256_file(Path(args.utility))},
        },
        "n_cases_joined": len(cases),
        "n_cases_used": len(kept),
        "n_missing_utility": missing,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "point": {"lambda": boot.point.lam, "eta": boot.point.eta,
                  "jac_cond": boot.point.jac_cond,
                  "multistart_lam_sd": boot.point.multistart_lam_sd},
        "pair_clustered": {
            "cluster_key": "(X_num, Y_num) goods pair",
            "n_clusters": boot.n_clusters,
            "lambda_ci95": list(boot.lam_ci), "eta_ci95": list(boot.eta_ci),
            "lambda_se": boot.lam_se, "eta_se": boot.eta_se,
            "n_replicates": boot.n_replicates, "n_failed": boot.n_failed,
        },
        "leave_one_good_out": {
            "lambda_min": logo["lam_min"], "lambda_max": logo["lam_max"],
            "lambda_range": logo["lam_range"],
            "most_influential_good": logo["most_influential_good"],
        },
    }
    print(json.dumps({k: doc[k] for k in ("tag", "n_cases_used", "point", "pair_clustered")},
                     indent=2, default=str))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2, default=str) + "\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
