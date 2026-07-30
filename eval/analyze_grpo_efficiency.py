#!/usr/bin/env python3
"""GRPO signal / efficiency analysis (Task 6).

Uses ONLY existing logs, manifests, and the frozen training deltas — no model
inference, no checkpoint or frozen-suite evaluation. Quantifies where GRPO's
learning signal goes and separates MEASURED quantities (from logged metrics)
from ESTIMATES (from the exposure calculation and the delta distribution).

It does NOT conclude GRPO is useless; it states the conditions under which SFT
could dominate this deterministic one-token task and names the untouched tests
needed to tell efficient learning from shortcut/template learning.

Terminology: the ~80% zero-reward-std groups are "zero task-reward advantage
groups" (frac_reward_zero_std). This is NOT DAPO generation-level filtering:
loss_type='dapo' fixes length bias, but generation-level dynamic sampling is
NOT implemented (KNOWN_ISSUES.md #4) — nothing is resampled or skipped.

    python3 eval/analyze_grpo_efficiency.py
    python3 eval/analyze_grpo_efficiency.py --check
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TD = PROJECT_ROOT / "results" / "training_dynamics"
DELTA_FILE = PROJECT_ROOT / "data" / "deltas" / "delta_qwen_base.json"
OUT_DIR = PROJECT_ROOT / "results" / "grpo_efficiency"
OUT_JSON = OUT_DIR / "grpo_efficiency.json"
OUT_MD = OUT_DIR / "grpo_efficiency.md"

TRAIN_ID_MIN, TRAIN_ID_MAX = 9_951, 59_400

# GRPO config (measured/known from qwen25_7b_qwen_delta.yaml + CAUSAL_BASELINE_PROTOCOL).
G = 16
MAX_STEPS = 30_000
TRAIN_POOL = 98_900
# Measured throughput (CAUSAL_BASELINE_PROTOCOL append record / smoke tests).
GRPO_S_PER_STEP = 6.1      # HF generate + update
SFT_S_PER_STEP = 0.28      # supervised forward/backward, no generation
DELTA_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf"))]
DELTA_LABELS = ["|d|<=0.5", "0.5<|d|<=1.0", "1.0<|d|<=2.0", "|d|>2.0"]


def load_metrics(path: Path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if v not in ("", None) else math.nan)
                         for k, v in r.items()})
    return rows


def seed_summary(rows):
    def col(name):
        return [r[name] for r in rows if not math.isnan(r.get(name, math.nan))]
    reward, kl, zero = col("reward"), col("kl"), col("frac_reward_zero_std")
    ent, gn = col("entropy"), col("grad_norm")
    return {
        "n_logged_steps": len(rows),
        "last_step": int(rows[-1]["step"]) if rows else None,
        "reward_mean": sum(reward) / len(reward) if reward else None,
        "reward_first": reward[0] if reward else None,
        "reward_last": reward[-1] if reward else None,
        "kl_first": kl[0] if kl else None,
        "kl_last": kl[-1] if kl else None,
        "kl_max": max(kl) if kl else None,
        "frac_zero_adv_mean": sum(zero) / len(zero) if zero else None,
        "frac_zero_adv_first": zero[0] if zero else None,
        "frac_zero_adv_last": zero[-1] if zero else None,
        "entropy_first": ent[0] if ent else None,
        "entropy_last": ent[-1] if ent else None,
        "grad_norm_mean": sum(gn) / len(gn) if gn else None,
    }


def delta_concentration():
    with open(DELTA_FILE) as fh:
        raw = json.load(fh)
    vals = [abs(float(v["mean_delta"])) for k, v in raw.items()
            if TRAIN_ID_MIN <= int(k) <= TRAIN_ID_MAX and float(v["mean_delta"]) != 0.0]
    total_mass = sum(vals)
    n = len(vals)
    out = []
    for (lo, hi), label in zip(DELTA_BINS, DELTA_LABELS):
        binned = [v for v in vals if (lo < v <= hi) or (lo == 0.0 and v <= hi)]
        cnt = len(binned)
        mass = sum(binned)
        out.append({
            "bin": label,
            "case_share": cnt / n,
            "abs_delta_mass_share": mass / total_mass if total_mass else 0.0,
        })
    top = out[-2]["abs_delta_mass_share"] + out[-1]["abs_delta_mass_share"]
    top_cases = out[-2]["case_share"] + out[-1]["case_share"]
    return {
        "n_nonzero": n,
        "bins": out,
        "high_delta_case_share_gt1": top_cases,
        "high_delta_mass_share_gt1": top,
        "interpretation": (
            "ESTIMATE (from the frozen training delta distribution, not logged "
            "per-step): under magnitude weighting the within-group advantage "
            "magnitude is proportional to |delta|, so update magnitude concentrates "
            f"in |delta|>1 cases ({top:.1%} of total |delta| mass from "
            f"{top_cases:.1%} of cases). Sign-only removes this concentration; "
            "scale_matched keeps it uniform-per-case at matched global scale."
        ),
    }


def exposure_table():
    completions = MAX_STEPS * G
    return {
        "measured_or_derived": "DERIVED from config (1 step = 1 unique prompt at this TRL config)",
        "grpo": {
            "unique_prompts": MAX_STEPS,
            "epoch_fraction": MAX_STEPS / TRAIN_POOL,
            "optimizer_updates": MAX_STEPS,
            "completions_generated": completions,
            "completion_tokens_trained_est": f"~{completions}*(1-2) tokens",
            "runtime_hours_est": MAX_STEPS * GRPO_S_PER_STEP / 3600,
            "s_per_step_measured": GRPO_S_PER_STEP,
        },
        "sft": {
            "unique_prompts": MAX_STEPS,
            "epoch_fraction": MAX_STEPS / TRAIN_POOL,
            "optimizer_updates": MAX_STEPS,
            "completions_generated": 0,
            "runtime_hours_est": MAX_STEPS * SFT_S_PER_STEP / 3600,
            "s_per_step_measured": SFT_S_PER_STEP,
        },
        "asymmetry_warning": (
            "GRPO generates 480,000 completions; SFT generates 0. Runtime and "
            "completion counts are FUNDAMENTALLY different between RL and SFT and "
            "are reported, NOT matched. Only unique-prompt exposure (30,000, "
            "0.303 epoch) and optimizer updates (30,000) are matched. Do NOT treat "
            "the unequal completion counts as a matched comparison."
        ),
    }


def build():
    seeds = {}
    for name, fname in [("seed1", "grpo_metrics_seed1.csv"),
                        ("seed2", "grpo_metrics_seed2.csv"),
                        ("seed42", "grpo_metrics_seed42.csv")]:
        p = TD / fname
        if p.exists():
            seeds[name] = seed_summary(load_metrics(p))

    # cross-seed stability on confirmatory seeds 1 & 2
    conf = [seeds[s] for s in ("seed1", "seed2") if s in seeds]
    stability = None
    if len(conf) == 2:
        def spread(key):
            vals = [c[key] for c in conf if c[key] is not None]
            return (max(vals) - min(vals)) if len(vals) == 2 else None
        stability = {
            "reward_last_spread": spread("reward_last"),
            "kl_last_spread": spread("kl_last"),
            "frac_zero_adv_mean_spread": spread("frac_zero_adv_mean"),
        }

    return {
        "purpose": "GRPO signal/efficiency analysis from existing logs (Task 6).",
        "terminology_note": (
            "frac_reward_zero_std = fraction of prompt groups with zero task-reward "
            "advantage (all G completions identical). NOT DAPO generation-level "
            "filtering; generation-level dynamic sampling is not implemented "
            "(KNOWN_ISSUES.md #4)."
        ),
        "measured_per_seed": seeds,
        "cross_seed_stability_conf": stability,
        "update_signal_by_delta_bin": delta_concentration(),
        "exposure": exposure_table(),
        "why_sft_could_dominate": [
            "The task is a deterministic one-token (Yes/No) mapping with a fixed "
            "rational target, so SFT has a dense per-example gradient on every prompt.",
            "GRPO wastes signal: ~80% of groups are zero task-reward advantage "
            "(measured frac_reward_zero_std), so most generation+backward compute "
            "carries only a KL-only update; SFT has no such waste.",
            "SFT reaches the endpoint in ~1.4 h/seed vs GRPO ~51 h/seed at 30k "
            "(measured s/step), a large efficiency gap for the SAME unique-prompt "
            "exposure.",
        ],
        "why_this_is_not_proof_grpo_is_useless": [
            "Efficiency is not effectiveness: the method winner is decided on the "
            "untouched suite under METHOD_COMPARISON_PROTOCOL.md, not on training "
            "cost or training reward (which is a conditional diagnostic).",
            "SFT on a fixed target risks shortcut/template learning (answer-token "
            "or label heuristics) that GRPO's exploration might avoid.",
        ],
        "untouched_tests_needed": [
            "The frozen untouched method-comparison suite (data/method_comparison/) "
            "for the confirmatory SFT-vs-GRPO lambda/eta comparison.",
            "The frozen semantic-counterbalancing component (Yes/No vs keep/trade, "
            "X/Y vs A/B, reversed order/attr, paraphrases) to distinguish an "
            "ownership-invariant policy from a Yes/No/label/template heuristic.",
            "Capability retention (GSM8K/IFEval) to check SFT's dense fit did not "
            "degrade general behavior more than GRPO.",
        ],
    }


def render_md(doc) -> str:
    L = ["# GRPO signal / efficiency analysis (Task 6)", "",
         "*From existing logs/manifests + frozen deltas only. Measured vs estimated "
         "are labelled. Not a claim that GRPO is useless.*", "",
         "## Measured per seed (logged every 10 steps)", "",
         "| seed | steps | reward last | KL first→last (max) | zero-adv mean (first→last) | entropy first→last |",
         "|---|---|---|---|---|---|"]
    for s, d in doc["measured_per_seed"].items():
        L.append(f"| {s} | {d['last_step']} | {d['reward_last']:.3f} | "
                 f"{d['kl_first']:.3f}→{d['kl_last']:.3f} ({d['kl_max']:.3f}) | "
                 f"{d['frac_zero_adv_mean']:.2f} ({d['frac_zero_adv_first']:.2f}→{d['frac_zero_adv_last']:.2f}) | "
                 f"{d['entropy_first']:.3f}→{d['entropy_last']:.3f} |")
    dc = doc["update_signal_by_delta_bin"]
    L += ["", "## Update-signal concentration by |δ̃| bin (ESTIMATE)", "",
          "| bin | case share | |δ̃| mass share |", "|---|---|---|"]
    for b in dc["bins"]:
        L.append(f"| {b['bin']} | {b['case_share']:.3f} | {b['abs_delta_mass_share']:.3f} |")
    L += ["", f"High-|δ̃| (>1) cases carry **{dc['high_delta_mass_share_gt1']:.1%}** of "
          f"total |δ̃| mass from **{dc['high_delta_case_share_gt1']:.1%}** of cases → "
          "magnitude weighting concentrates updates on high-|δ̃| cases.", ""]
    ex = doc["exposure"]
    L += ["## Exposure (derived) + measured throughput", "",
          f"- GRPO: {ex['grpo']['unique_prompts']:,} prompts "
          f"({ex['grpo']['epoch_fraction']:.3f} epoch), "
          f"{ex['grpo']['completions_generated']:,} completions, "
          f"~{ex['grpo']['runtime_hours_est']:.1f} h/seed ({ex['grpo']['s_per_step_measured']} s/step).",
          f"- SFT: same {ex['sft']['unique_prompts']:,} prompts, "
          f"{ex['sft']['completions_generated']} completions, "
          f"~{ex['sft']['runtime_hours_est']:.1f} h/seed ({ex['sft']['s_per_step_measured']} s/step).",
          "", f"> {ex['asymmetry_warning']}", "",
          "## Interpretation", "",
          "**Why SFT could dominate this task:**"]
    L += [f"- {x}" for x in doc["why_sft_could_dominate"]]
    L += ["", "**Why that is NOT proof GRPO is useless:**"]
    L += [f"- {x}" for x in doc["why_this_is_not_proof_grpo_is_useless"]]
    L += ["", "**Untouched tests needed to decide:**"]
    L += [f"- {x}" for x in doc["untouched_tests_needed"]]
    L += ["", f"_{doc['terminology_note']}_", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    doc = build()
    outputs = {OUT_JSON: json.dumps(doc, indent=2) + "\n", OUT_MD: render_md(doc)}
    if args.check:
        bad = [p.name for p, s in outputs.items() if not p.exists() or p.read_text() != s]
        if bad:
            raise SystemExit("CHECK FAILED: drifted/missing " + ", ".join(bad))
        print("CHECK PASSED: GRPO-efficiency outputs byte-identical.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, s in outputs.items():
        p.write_text(s)
        print(f"Wrote {p.relative_to(PROJECT_ROOT)}")
    dc = doc["update_signal_by_delta_bin"]
    print(f"seeds analyzed: {list(doc['measured_per_seed'])}")
    print(f"high-|delta|(>1) mass share: {dc['high_delta_mass_share_gt1']:.1%} "
          f"from {dc['high_delta_case_share_gt1']:.1%} of cases")


if __name__ == "__main__":
    main()
