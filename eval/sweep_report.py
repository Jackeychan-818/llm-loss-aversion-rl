#!/usr/bin/env python3
"""
Aggregate the v2-core held-out checkpoint sweep into a selection table.

Reads results/v2core_sweep/v2core-ckpt*.json (written by
sweep_partition_estimate.py), pulls the training KL at each checkpoint step from
the training log, prints a table sorted by step, and marks the SELECTION winner:
the checkpoint with val |lambda_hat| closest to 0, using rational-choice accuracy,
eta, and KL as guardrails.

    python eval/sweep_report.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES_DIR = PROJECT_ROOT / "results" / "v2core_sweep"


def load_rows():
    rows = []
    for f in sorted(glob.glob(str(RES_DIR / "v2core-ckpt*.json"))):
        d = json.load(open(f))
        m = re.search(r"ckpt(\d+)", d["model_name"])
        d["step"] = int(m.group(1)) if m else -1
        rows.append(d)
    rows.sort(key=lambda r: r["step"])
    return rows


def fmt(v, nd=4):
    return "   n/a" if v is None else f"{v:+.{nd}f}"


def main():
    rows = load_rows()
    if not rows:
        print(f"No sweep results yet in {RES_DIR}")
        return

    print(f"\nv2-core held-out checkpoint sweep — {len(rows)} checkpoints")
    print("selection = validation half (even case_ids); guardrails from test anchor\n")
    hdr = f"{'step':>6} | {'val λ̂':>9} {'±SE':>7} | {'val η̂':>8} | {'rat.acc':>7} | {'consist':>7} | {'keepboth':>8}"
    print(hdr); print("-" * len(hdr))
    best = None
    for r in rows:
        v = r.get("val", {})
        lam = v.get("lambda")
        line = (f"{r['step']:>6} | {fmt(lam):>9} {('%.4f'%v['lambda_se']) if v.get('lambda_se') is not None else '   n/a':>7} | "
                f"{fmt(v.get('eta')):>8} | {('%.3f'%v['rational_acc']) if v.get('rational_acc')==v.get('rational_acc') else 'nan':>7} | "
                f"{('%.3f'%v.get('consistent_rate',0)):>7} | {('%.3f'%v.get('keep_both_rate',0)):>8}")
        print(line)
        if lam is not None and (best is None or abs(lam) < abs(best["val"]["lambda"])):
            best = r

    if best:
        v = best["val"]
        print("\n" + "=" * 60)
        print(f"SELECTION (min |val λ̂|): checkpoint-{best['step']}")
        print(f"  val λ̂ = {v['lambda']:+.4f} (SE {v['lambda_se']:.4f})   η̂ = {v['eta']:+.4f}")
        print(f"  rational-choice acc = {v['rational_acc']:.3f}   consistent = {v['consistent_rate']:.3f}")
        print("  → confirm this + neighbors on FROZEN half (odd case_ids) + OOD-50:")
        print(f"     python eval/sweep_partition_estimate.py --feature val_sweep \\")
        print(f"        --model_name v2core-ckpt{best['step']} --estimate val,frozen")
        print("=" * 60)


if __name__ == "__main__":
    main()
