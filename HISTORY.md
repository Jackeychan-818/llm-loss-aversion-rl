# Commit History — lambda-zero

This file combines the recent git commits into a project-level history, so the research state is readable without replaying `git log`.

## Chronological Summary

| Date | Commit | What changed | Research state |
|---|---|---|---|
| Apr 14, 2026 | `923a71e` | Initial experiment runner with Qwen-7B baseline support. | Qwen could be evaluated in the inherited endowment-effect pipeline. |
| Apr 27, 2026 | `40f4b0e` | Added GRPO training code, prompt builder, utility reward functions, NSCC setup, baseline Qwen-7B outputs, and frontier-consensus delta files. | Phase 1 baseline established: `lambda_before = 11.75`, `eta = 1.52`. Phase 2 reward design moved from open question to implemented utility-weighted reward. |
| Apr 27, 2026 | `c7088c7` | Updated agent context to reflect the chosen consensus-delta reward and training hyperparameters. | Phase 3 became the active focus: sanity/full training on NSCC with `G=16`, `temperature=1.5`, `beta=0.04`, LoRA rank 16. |
| Jun 18, 2026 | `d26ac54` | Fixed NSCC GRPO resume behavior and vLLM training path; added `submit_train_glong.pbs`. | Full training became resumable after walltime limits. |
| Jun 18, 2026 | `e938efa` | Restored NSCC-validated PyTorch module, gradient accumulation, and local Qwen eval registry entries after a Mac-side merge. | Stabilized NSCC environment around `pytorch/2.6.0-py3-cu11.8` and `gradient_accumulation_steps=16`. |
| Jun 18, 2026 | `dce0550` | Fixed `glong` submission by routing through the normal queue. | Long training jobs could route correctly on PBS. |
| Jun 23, 2026 | `b9c2041` | Disabled vLLM after compatible versions failed on NSCC; fell back to plain HF generation. | Training no longer depended on a broken vLLM import chain. |
| Jun 25, 2026 | `f4a2c9a` | Added vLLM-free local Qwen evaluation path: `eval/run_qwen_local.py` and `eval/estimate_qwen_grpo.py`. | Post-training evaluation became possible without a local vLLM server. |
| Jul 3, 2026 | `b511e53` | Added full held-out Phase 4 evaluation results and PBS eval scripts. | Phase 4 result: `lambda_after = 0.177` with SE `0.005`, a 98.5% reduction from `11.75`. |
| Jul 7, 2026 | `c643617` | Added debias and forced treatment results, Qwen-delta ablation data/config, and training docs. | Treatment robustness looks strong: baseline `0.177`, debias `0.205`, forced `0.173`. Qwen-own-delta reward became the active ablation. |
| Jul 9, 2026 | `f2a8710` | Added `monitor.sh`, `plot_training.py`, and log append fixes in PBS scripts. | Training-health monitoring became easier and log history is preserved across job restarts. |
| Jul 15, 2026 | Working-tree paper decision | Made Qwen-own-delta checkpoint 8,000 the primary paper model and the frontier-consensus model the reward-source ablation; documented selection-on-test, reward leakage, matched-base, seed, inference, capability, and reproducibility risks. | `test_goods` is validation for the primary model; the separately frozen evaluation is intended to carry final claims after artifacts are archived. |

## Current Research Snapshot

- Primary paper model: Qwen-own-delta GRPO, checkpoint 8,000.
- Primary ID validation result: `lambda = 0.111`, `eta = 0.504`. It is not an
  untouched final-test estimate because `test_goods` was used for checkpoint
  selection.
- Primary OOD result reported in the draft: `lambda = 0.226`, `eta = 0.790`,
  consistency `49.46%`; raw outputs and estimator artifacts must be archived.
- Reward-source ablation: consensus-delta GRPO, ID `lambda = 0.1765`,
  `eta = -0.0477`; reported OOD `lambda = 0.395`, `eta = 0.502`, consistency
  `64.89%`.
- Historical Together-hosted base: `lambda = 11.7519`, `eta = 1.5178`; a
  matched exact-local-base evaluation remains required.
- Held-out evaluation used 9,890 attribute configurations from `test_goods.json`; there are zero repeated prompts/configurations across train and test, while the 100 goods and 4,945 goods pairs are shared.
- Held-out attribute profiles are jointly significant under a leave-one-good-out jackknife (χ²(8) = 756.05, p < 10^-150), explain 48.8% of within-pair/perspective variation, and flip at least one perspective's answer for 42.18% of goods pairs.
- The split supports configuration generalization, but `test_goods` is
  validation for Qwen-own delta because it informed reward construction and
  checkpoint selection.
- Human and frontier-superiority claims are paused pending comparable tasks,
  estimands, model endpoints, samples, scorers, and estimators.

## Remaining Work

- Rerun the exact local base checkpoint with the adapter evaluation pipeline.
- Archive all Qwen-own checkpoint-sweep and OOD raw/model/estimator artifacts.
- Validate Qwen-own pseudo-utility against ownership-free Qwen preferences.
- Freeze a joint lambda/eta/consistency checkpoint-selection rule and run at
  least three seeds.
- Add matched SFT and sign-only GRPO baselines, robust inference and estimator
  recovery, plus GSM8K/IFEval capability checks.
- Use consensus delta as the reward-source ablation and rewrite the paper around
  Qwen-own delta. Full ordering and claim restrictions are in
  `PAPER_READINESS.md`.
