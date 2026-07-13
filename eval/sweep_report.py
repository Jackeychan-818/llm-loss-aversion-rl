#!/usr/bin/env python3
"""
Aggregate the v2-core held-out sweep and LOCK a single winner by a written rule.

Reads results/v2core_sweep/v2core-ckpt*.json (enriched by
recompute_sweep_metrics.py), prints the corrected selection table (validation half
= even case_ids of test_goods), applies the written selection rule, and — with
--lock — writes a committed selection manifest.

WRITTEN SELECTION RULE (applied in order; thresholds are explicit + auditable):
  1. Practically near-zero λ̂: keep only checkpoints with |λ̂| <= LAMBDA_TOL.
  2. Small |η̂|: among those, prefer smaller |η̂| (ties within ETA_TOL pass on).
  3. Balanced errors: prefer smaller |keep_both_rate - trade_both_rate|
     (ties within IMBAL_TOL pass on).
  4. Statistical tie-break: prefer the earliest checkpoint; if two are within
     STEP_TOL steps, prefer the lower-KL one.
Guardrail (not a chooser): winner's λ̂ 95% CI is reported; joint_anchored_success
and consistency are shown so a low-consistency checkpoint can never hide behind a
high conditional anchor-match.

    python eval/sweep_report.py           # print table + provisional winner
    python eval/sweep_report.py --lock    # also write the selection manifest
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES_DIR = PROJECT_ROOT / "results" / "v2core_sweep"
MANIFEST = RES_DIR / "SELECTION_MANIFEST.md"

# ── written-rule thresholds ─────────────────────────────────────────────
LAMBDA_TOL = 0.05    # |λ̂| <= this counts as practically near-zero (<5% asymmetry on 1+λ)
ETA_TOL = 0.05       # |η̂| differences below this are treated as tied
IMBAL_TOL = 0.03     # keep/trade imbalance differences below this are treated as tied
STEP_TOL = 400       # checkpoints within this many steps are "statistically tied" for KL tie-break


def load_rows():
    rows = []
    for f in sorted(glob.glob(str(RES_DIR / "v2core-ckpt*.json"))):
        rows.append(json.load(open(f)))
    rows.sort(key=lambda r: r["step"])
    return rows


def imbalance(v):
    return abs(v.get("keep_both_rate", 0.0) - v.get("trade_both_rate", 0.0))


def select(rows):
    """Apply the written rule. Returns (winner, candidates, reason)."""
    cand = [r for r in rows if r["val"].get("lambda") is not None
            and abs(r["val"]["lambda"]) <= LAMBDA_TOL]
    if not cand:
        return None, [], "no checkpoint has |λ̂| <= %.2f" % LAMBDA_TOL

    def key(r):
        v = r["val"]
        # bucketed lexicographic so small differences count as ties and pass to the next rule
        return (
            round(abs(v["eta"]) / ETA_TOL),        # small |η̂| first (bucketed)
            round(imbalance(v) / IMBAL_TOL),        # balanced errors next (bucketed)
            round(r["step"] / STEP_TOL),            # earliest (bucketed) — statistical tie
            r.get("kl", float("inf")),              # then lower KL
            r["step"],                              # final determinism
        )
    ranked = sorted(cand, key=key)
    winner = ranked[0]
    reason = (f"|λ̂|<= {LAMBDA_TOL}: {[r['step'] for r in cand]}; "
              f"then min|η̂|, then balanced keep/trade, then earliest/lower-KL")
    return winner, cand, reason


def fmt(v, nd=4):
    return "   n/a" if v is None else f"{v:+.{nd}f}"


def print_table(rows):
    hdr = (f"{'step':>5} | {'λ̂':>8} {'95% CI':>17} | {'η̂':>8} | {'joint':>6} "
           f"| {'m|cons':>6} | {'consis':>6} | {'keep':>5} {'trade':>5} | {'KL':>5}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        v = r["val"]; lam = v.get("lambda")
        ci = v.get("lambda_ci95")
        ci_s = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "        —        "
        print(f"{r['step']:>5} | {fmt(lam):>8} {ci_s:>17} | {fmt(v.get('eta')):>8} "
              f"| {v['joint_anchored_success']:.3f} | {v['anchor_match_given_consistent']:.3f} "
              f"| {v['consistent_rate']:.3f} | {v['keep_both_rate']:.3f} {v['trade_both_rate']:.3f} "
              f"| {r.get('kl', float('nan')):.3f}")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_manifest(winner, cand, reason, rows):
    v = winner["val"]
    ci = v.get("lambda_ci95")
    lines = [
        "# v2-core checkpoint selection manifest",
        "",
        f"- **Chosen checkpoint:** `checkpoints/grpo_v2core/checkpoint-{winner['step']}`",
        f"- **Code commit:** `{git_commit()}`",
        f"- **Data split:** {winner['split']}",
        f"- **Selection metric source:** validation half only (even case_ids of test_goods); "
        "frozen odd half and OOD-50 NOT yet opened at selection time.",
        "",
        "## Written rule (applied in order)",
        f"1. Practically near-zero λ̂: |λ̂| ≤ {LAMBDA_TOL}.",
        f"2. Small |η̂| (ties within {ETA_TOL}).",
        f"3. Balanced keep-both/trade-both errors, i.e. min |keep−trade| (ties within {IMBAL_TOL}).",
        f"4. Statistical tie-break: earliest checkpoint (within {STEP_TOL} steps), then lower KL.",
        "",
        f"- **Candidate set (passed step 1):** {[r['step'] for r in cand]}",
        f"- **Reason:** {reason}",
        "",
        "## Chosen checkpoint metrics (validation half)",
        f"- λ̂ = {v['lambda']:+.4f}  (95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], SE {v['lambda_se']:.4f})",
        f"- η̂ = {v['eta']:+.4f}  (SE {v['eta_se']:.4f})",
        f"- distance of (λ,η) from (0,0): Euclidean {v['dist_euclid']:.4f}, "
        f"Wald-diag {v['dist_wald_diag']:.3f}",
        f"- consistency = {v['consistent_rate']:.3f}, keep-both = {v['keep_both_rate']:.3f}, "
        f"trade-both = {v['trade_both_rate']:.3f}",
        f"- anchor_match_given_consistent = {v['anchor_match_given_consistent']:.3f} (conditional)",
        f"- joint_anchored_success = {v['joint_anchored_success']:.3f} (unconditional)",
        f"- checkpoint KL = {winner.get('kl', float('nan')):.4f}",
        "",
        "## Full sweep (validation half)",
        "| step | λ̂ | η̂ | joint_success | consistency | keep | trade | KL |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        vv = r["val"]
        lam = vv.get("lambda")
        lam_s = f"{lam:+.4f}" if lam is not None else "n/a"
        lines.append(f"| {r['step']} | {lam_s} | {vv.get('eta', float('nan')):+.4f} | "
                     f"{vv['joint_anchored_success']:.3f} | {vv['consistent_rate']:.3f} | "
                     f"{vv['keep_both_rate']:.3f} | {vv['trade_both_rate']:.3f} | {r.get('kl', float('nan')):.3f} |")
    lines += [
        "",
        "## Confirmation protocol (post-lock)",
        f"- Open the FROZEN half (odd case_ids) and run OOD-50 on checkpoint-{winner['step']} ONLY.",
        "- If confirmation disappoints, REPORT it. Do NOT silently re-select on frozen/OOD.",
        "",
    ]
    MANIFEST.write_text("\n".join(lines))
    print(f"\nwrote manifest: {MANIFEST}")


def main():
    rows = load_rows()
    if not rows:
        print(f"No sweep results in {RES_DIR}")
        return
    print(f"\nv2-core held-out sweep — {len(rows)} checkpoints (validation = even case_ids)\n")
    print_table(rows)
    winner, cand, reason = select(rows)
    if winner is None:
        print(f"\nNo winner: {reason}")
        return
    print(f"\n{'='*64}")
    print(f"WINNER (written rule): checkpoint-{winner['step']}")
    print(f"  candidates (|λ̂|<= {LAMBDA_TOL}): {[r['step'] for r in cand]}")
    v = winner["val"]
    print(f"  λ̂={v['lambda']:+.4f}  η̂={v['eta']:+.4f}  keep/trade={v['keep_both_rate']:.3f}/{v['trade_both_rate']:.3f}"
          f"  joint_success={v['joint_anchored_success']:.3f}  KL={winner.get('kl', float('nan')):.3f}")
    print("="*64)
    if "--lock" in sys.argv:
        write_manifest(winner, cand, reason, rows)


if __name__ == "__main__":
    main()
