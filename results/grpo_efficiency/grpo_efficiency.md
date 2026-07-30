# GRPO signal / efficiency analysis (Task 6)

*From existing logs/manifests + frozen deltas only. Measured vs estimated are labelled. Not a claim that GRPO is useless.*

## Measured per seed (logged every 10 steps)

| seed | steps | reward last | KL first→last (max) | zero-adv mean (first→last) | entropy first→last |
|---|---|---|---|---|---|
| seed1 | 30000 | 0.054 | 0.000→0.713 (7.589) | 0.57 (0.90→0.30) | 0.070→0.193 |
| seed2 | 30000 | 0.128 | 0.000→0.768 (9.997) | 0.56 (0.60→0.70) | 0.152→0.116 |
| seed42 | 30000 | 0.245 | 0.000→0.643 (3.959) | 0.57 (0.80→0.70) | 0.095→0.081 |

## Update-signal concentration by |δ̃| bin (ESTIMATE)

| bin | case share | |δ̃| mass share |
|---|---|---|
| |d|<=0.5 | 0.530 | 0.158 |
| 0.5<|d|<=1.0 | 0.199 | 0.211 |
| 1.0<|d|<=2.0 | 0.219 | 0.450 |
| |d|>2.0 | 0.052 | 0.181 |

High-|δ̃| (>1) cases carry **63.1%** of total |δ̃| mass from **27.1%** of cases → magnitude weighting concentrates updates on high-|δ̃| cases.

## Exposure (derived) + measured throughput

- GRPO: 30,000 prompts (0.303 epoch), 480,000 completions, ~50.8 h/seed (6.1 s/step).
- SFT: same 30,000 prompts, 0 completions, ~1.51 h/seed (0.181 s/step, measured 5,422 s).

> GRPO generates 480,000 completions; SFT generates 0. Runtime and completion counts are FUNDAMENTALLY different between RL and SFT and are reported, NOT matched. Only unique-prompt exposure (30,000, 0.303 epoch) and optimizer updates (30,000) are matched. Do NOT treat the unequal completion counts as a matched comparison.

## Interpretation

**Why SFT could dominate this task:**
- The task is a deterministic one-token (Yes/No) mapping with a fixed rational target, so SFT has a dense per-example gradient on every prompt.
- GRPO wastes signal: ~56-57% of groups are zero task-reward advantage on the full runs (measured mean frac_reward_zero_std; ~80% is only an early-step observation), so a large share of generation+backward compute carries only a KL-only update; SFT has no such waste.
- SFT reaches the endpoint in ~1.51 h/seed (measured 5,422 s) vs GRPO ~51 h/seed at 30k, a large efficiency gap for the SAME unique-prompt exposure.

**Why that is NOT proof GRPO is useless:**
- Efficiency is not effectiveness: the method winner is decided on the untouched suite under METHOD_COMPARISON_PROTOCOL.md, not on training cost or training reward (which is a conditional diagnostic).
- SFT on a fixed target risks shortcut/template learning (answer-token or label heuristics) that GRPO's exploration might avoid.

**Untouched tests needed to decide:**
- The frozen untouched method-comparison suite (data/method_comparison/) for the confirmatory SFT-vs-GRPO lambda/eta comparison.
- The frozen semantic-counterbalancing component (Yes/No vs keep/trade, X/Y vs A/B, reversed order/attr, paraphrases) to distinguish an ownership-invariant policy from a Yes/No/label/template heuristic.
- Capability retention (GSM8K/IFEval) to check SFT's dense fit did not degrade general behavior more than GRPO.

_frac_reward_zero_std = fraction of prompt groups with zero task-reward advantage (all G completions identical). NOT DAPO generation-level filtering; generation-level dynamic sampling is not implemented (KNOWN_ISSUES.md #4)._
