#!/usr/bin/env python3
"""
Pseudo-utility alignment W — a third reported quantity alongside lambda and eta.

Definition (tau fixed at 1; alternative tau not tuned or discussed):

    w_q = exp( U_chosen - max(U_1, U_2) )
        = 1                       if the higher-pseudo-utility good is chosen,
          exp(-|U_1 - U_2|)       otherwise,
    W   = (1/N) sum_q w_q.

Because U_1 - U_2 = delta (the pseudo-utility gap delta = U_X - U_Y), w_q needs
only |delta| and which good was chosen. W in (0, 1] is a magnitude-weighted
rational-choice rate: 1.0 = always picks the pseudo-utility-preferred good; an
error costs exp(-|delta|), so giving up a LARGE utility gap is penalised more
than a near-tie. W is DESCRIPTIVE (reported with lambda/eta); it is not a
success gate and was added after the seed pre-registration was frozen.

Reference pseudo-utility = delta (default `data/deltas/delta_qwen_base.json`,
the primary reward signal). Using one shared delta reference makes W comparable
ACROSS models — unlike each model's own fitted utilities, which would be a
different yardstick per model. Choices are read as the hard argmax of P(Yes)/P(No)
(the formula selects a definite chosen good).

Case/perspective -> chosen good (endowment task):
  X-perspective (endowed X, offered Y): "No" -> keep X (chosen X); "Yes" -> Y
  Y-perspective (endowed Y, offered X): "No" -> keep Y (chosen Y); "Yes" -> X
Higher-utility good = X if delta > 0 else Y.

    python eval/pseudo_utility_alignment.py \\
        --eval_dir baseline/Qwen-7B-GRPO-qd-ckpt8000 --name qd-8k
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from sweep_partition_estimate import load_rows, choice_from_row

DEFAULT_DELTA = PROJECT_ROOT / "data" / "deltas" / "delta_qwen_base.json"


def load_delta(path: Path) -> dict[int, float]:
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        try:
            out[int(k)] = float(v["mean_delta"] if isinstance(v, dict) else v)
        except (ValueError, KeyError, TypeError):
            continue
    return out


def compute_W(eval_dir: Path, delta: dict[int, float], eps: float = 1e-12) -> dict:
    xr = load_rows(eval_dir / "loss_aversion_X.json")
    yr = load_rows(eval_dir / "loss_aversion_Y.json")
    common = sorted(set(xr) & set(yr))
    ws, rational, n, skipped = [], 0, 0, 0
    for cid in common:
        dv = delta.get(cid)
        if dv is None or abs(dv) < eps:
            skipped += 1
            continue
        higher = "X" if dv > 0 else "Y"
        for persp, row in (("X", xr[cid]), ("Y", yr[cid])):
            resp = choice_from_row(row)  # argmax Yes/No
            if persp == "X":
                chosen = "X" if resp == "No" else "Y"
            else:
                chosen = "Y" if resp == "No" else "X"
            hit = chosen == higher
            ws.append(1.0 if hit else math.exp(-abs(dv)))
            rational += hit
            n += 1
    if n == 0:
        raise RuntimeError(f"no scorable choices in {eval_dir} (delta coverage? case-id overlap?)")
    return {
        "W": sum(ws) / n,
        "N": n,
        "rational_choice_rate": rational / n,
        "cases_scored": n // 2,
        "cases_skipped_no_delta_or_zero": skipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--delta_file", default=str(DEFAULT_DELTA),
                    help="reference pseudo-utility delta (default delta_qwen_base.json)")
    ap.add_argument("--out", default=None, help="optional results/*.json to write")
    args = ap.parse_args()

    delta = load_delta(Path(args.delta_file))
    res = compute_W(PROJECT_ROOT / args.eval_dir, delta)
    name = args.name or Path(args.eval_dir).name
    res.update({"name": name, "eval_dir": args.eval_dir,
                "delta_file": args.delta_file})
    print(f"{name}: W = {res['W']:.4f}  (N={res['N']}, "
          f"rational-choice rate = {res['rational_choice_rate']:.4f})")
    if args.out:
        out = PROJECT_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out, "w"), indent=2)
        print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
