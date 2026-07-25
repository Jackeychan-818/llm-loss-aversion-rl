# llm-loss-aversion-rl

Fine-tuning Qwen2.5-7B-Instruct with GRPO to reduce measured loss aversion in an endowment-effect task.

## Headline Results

- **Primary paper model:** Qwen-own-delta GRPO, checkpoint 8,000.
- Primary-model validation: λ̂ = **0.111**, η̂ = **0.504** on
  `test_goods.json`. This set was used for checkpoint selection, so 0.111 is a
  validation result rather than untouched final-test performance.
- Primary-model OOD result reported in the current draft: λ̂ = **0.226**,
  η̂ = **0.790**, consistency = **49.46%**. Raw outputs and estimator artifacts
  still need to be archived for paper-ready reproducibility.
- **Reward-source ablation:** frontier-consensus-delta GRPO, with ID
  λ̂ = **0.177** (SE = 0.005), η̂ = **-0.048** and reported OOD
  λ̂ = **0.395**, η̂ = **0.502**, consistency = **64.89%**.
- Matched local base Qwen: λ̂ = **7.637** (SE = 0.627), η̂ = **1.007**, using
  the same local weights, rows, scorer, and estimator as the tuned model. The
  historical Together-hosted estimate of 11.75 is not the matched comparator.
- Attribute profiles explain **48.8%** of within-pair/perspective variation
- Changing the held-out configuration flips at least one answer for **42.2%** of goods pairs

Training and evaluation contain no repeated prompts or exact configurations.
They use the same 100 goods and 4,945 goods pairs, with 10 configurations per
pair used for training and two disjoint configurations reserved for evaluation.
This supports within-benchmark configuration generalization; it is not an
unseen-goods claim. Because Qwen-own reward construction and checkpoint
selection used information from `test_goods`, the paper treats this set as
validation and retains a separately frozen evaluation for final claims. See
`ATTRIBUTE_EFFECTS.md` for the split analysis and `PAPER_READINESS.md` for the
full methodological audit.

### Frozen prospective configuration test — EVALUATED

`data/frozen_unused_test_goods.json` (10 new configs/pair, **49,450 cases**,
**98,900 X/Y prompts/model**, zero overlap with validation/training) was opened
once on 2026-07-24 for the three policy-allowed models. **λ collapses from the
matched base 5.946 → ≈0 on both confirmatory seeds** (seed1 0.031, seed2 −0.053),
with SEs ~3× tighter than `test_goods` and η/consistency/W matching the ID
pattern (W within 0.001 of ID for every model):

| model | λ (SE) | η | consistency | W |
|---|---|---|---|---|
| matched base | 5.946 (0.192) | 1.458 | 0.008 | 0.742 |
| seed1 @2000 | 0.031 (0.007) | 0.089 | 0.659 | 0.882 |
| seed2 @6000 | −0.053 (0.005) | 0.693 | 0.716 | 0.909 |

This tests **new configurations of familiar goods** — not unseen goods (OOD-50
remains that test) — and may not be used for training/reward/selection. Results:
`results/frozen_unused_results.json`; provenance (dataset SHA, evaluator commit,
adapter hashes, base model, PBS job IDs): `results/frozen_unused_evaluation_manifest.json`;
construction: `data/FROZEN_UNUSED_TEST.md` + `data/frozen_unused_test_goods.manifest.json`.

## Current Work

- **Replication CONFIRMED: 2/2 seeds PASS.** ID selection (seed 1 → step 2,000,
  seed 2 → step 6,000) plus one OOD-50 and one GSM8K per selected checkpoint are
  complete, and the mechanical pre-registration verdict
  (`results/seed_replication_report.json`) passes every gate for both seeds:
  seed 1 λ_OOD=0.259, seed 2 λ_OOD=0.064 (both ≤0.5), consistency 0.50 / 0.60,
  GSM8K paired-CI lower bounds −1.14pp / −1.44pp (both ≥ −3pp). seed 42 is
  supporting exploratory evidence.
- **Prospective unused-configuration evaluation DONE.** The matched base and
  two frozen seed selections were evaluated exactly once; the set remains
  closed to training, checkpoint selection, and method revision.
- **Full framing evaluation DONE.** It is a non-gating specificity result:
  exploratory step 8,000 is more framing-susceptible than the matched base
  (hard flip rate 0.505 → 0.689; absolute probability gap 0.490 → 0.679).
- Training-process + structural-trajectory figures are posted under
  `results/training_dynamics/` (GRPO reward/loss/KL/entropy/filtering and the
  λ/η/d/α/β/utility trajectory vs the exact local base).
- Diagnose training reward/loss, KL and DAPO filtering, and separately inspect
  NLS starting/final objective, multi-start stability, alpha/beta drift, and
  fitted utility preservation. Step 600 may be used only as an exploratory
  early-trajectory point; frozen selection remains 2k–30k @ 2k.
- Archive the Qwen-own-delta checkpoint sweep and OOD prediction/estimation
  artifacts.
- **Next experimental priorities:** matched SFT and sign-only GRPO, frozen
  prompt-semantic counterbalancing, IFEval, and one compact
  GSM-Symbolic-500 capability extension. Confidence calibration is explicitly
  lower priority. See `RESEARCH_ROADMAP.md` for the frozen-design requirements
  and ordering.
- Add robust pair/good-aware structural inference and estimator-recovery
  checks.

See `PAPER_READINESS.md` for the authoritative blocker list,
`RESEARCH_ROADMAP.md` for the next experimental program,
`PROJECT_OVERVIEW.md` for the research narrative, and `HISTORY.md` for the
commit-by-commit project history.
