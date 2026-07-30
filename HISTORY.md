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
| Jul 15–16, 2026 | `ee42141`–`bbc1ae2` | Corrected the Model-A link-scale language, made Qwen-own-delta checkpoint 8,000 the exploratory primary model, validated the Qwen-derived reward against ownership-free anchors, ran the exactly matched local base, froze checkpoint selection and seed criteria, and fixed missing seed propagation. | The historical 11.75 estimate was retired as the causal baseline; `test_goods` became validation-only; seeds 1 and 2 became genuinely independent confirmatory runs. |
| Jul 16, 2026 | `171d283`, `0b964e4` | Added pseudo-utility alignment `W` as a descriptive third outcome alongside lambda and eta. | Preference alignment became measurable, but direct agreement with frozen ownership-free choices remains the stronger preservation test. |
| Jul 21–22, 2026 | `c7ae293`–`aade3ee` | Added frozen checkpoint selection, full-grid structural diagnostics, multi-start checks, Jacobian conditioning, and training/utility trajectories. | The selection procedure became auditable and estimator instability became visible rather than being hidden by a single fit. |
| Jul 23, 2026 | `2cd72cc`, `255687f` | Completed the two confirmatory seeds and their one-shot OOD-50 plus GSM8K evaluations. | Replication passed 2/2 under the frozen rule; seed 1 selected step 2,000 and seed 2 selected step 6,000. |
| Jul 23, 2026 | `ec9d84b` | Completed the framing-specificity evaluation. | The intervention must not be sold as general debiasing: the exploratory step-8,000 model was more framing-susceptible than the matched base. |
| Jul 23–25, 2026 | `6e0f480`–`9edd76d` | Froze, evaluated once, and documented the prospective unused-configuration suite. | On 49,450 new configurations of familiar pairs, matched-base lambda 5.946 fell to 0.031 and −0.053 for the two selected seeds. This is configuration generalization, not unseen-goods OOD. |
| Jul 25, 2026 | `2ecf68d` | Added a venue-independent research roadmap. | Causal baselines, prompt-semantic robustness, broader capability checks, and stronger inference became the ordered next program. |
| Jul 26–27, 2026 | `909cb15`–`c1ab3da` | Added and hardened matched SFT and sign-only GRPO infrastructure, then recorded a seed-1, 6k-step pilot. | Both pilots sharply reduce lambda on validation; SFT is especially strong at 4k–6k, but no method winner is established without the frozen full runs and a new untouched suite. |
| Jul 28, 2026 | `6c31e81` plus status reconciliation | Committed the manuscript source tree, result registry, paper scripts, and Qwen-utility delta builder; the follow-up reconciliation reduced the redundant root TeX manuscript to a deprecated pointer. | The earlier untracked/canonical-source problem is substantially resolved. Generated outputs, duplicate local copies, and the overlapping utility-delta builders still require classification. |
| Jul 30, 2026 | NSCC full SFT runs; manifests pending repository sync | Completed matched SFT seeds 1 and 2 through 30k with all 15 checkpoints per seed; preserved the seed-1 pilot separately. Runtime was approximately 5,422 seconds per seed and total charged cost approximately 317 SU. | SFT training is complete, but full-grid evaluation and selection have not run. The reported balance is approximately 3,312 SU; no full-SFT behavioral or method-superiority claim is yet supported. |

## Current Research Snapshot

- Primary paper model: Qwen-own-delta GRPO, checkpoint 8,000.
- Primary ID validation result: `lambda = 0.111`, `eta = 0.504`. It is not an
  untouched final-test estimate because `test_goods` was used for checkpoint
  selection.
- Confirmatory replication passed for both fresh seeds under the frozen
  procedure. Seed 1 selected step 2,000 and seed 2 selected step 6,000; both
  passed OOD-50 and GSM8K gates.
- The prospective unused-configuration result is complete: matched-base
  `lambda = 5.946` versus `0.031` and `-0.053` for the two selected seeds.
  This is strong within-benchmark configuration evidence, not new-goods
  generalization.
- Reward-source ablation: consensus-delta GRPO, ID `lambda = 0.1765`,
  `eta = -0.0477`; reported OOD `lambda = 0.395`, `eta = 0.502`, consistency
  `64.89%`.
- Exactly matched local base: `lambda = 7.637`, `eta = 1.007`; use this for
  before/after claims. The historical Together-hosted estimate of 11.7519 is
  retained only as provenance for the pipeline-mismatch lesson.
- Held-out evaluation used 9,890 attribute configurations from `test_goods.json`; there are zero repeated prompts/configurations across train and test, while the 100 goods and 4,945 goods pairs are shared.
- Held-out attribute profiles are jointly significant under a leave-one-good-out jackknife (χ²(8) = 756.05, p < 10^-150), explain 48.8% of within-pair/perspective variation, and flip at least one perspective's answer for 42.18% of goods pairs.
- The exploratory causal-baseline pilot is complete on `test_goods`: SFT and
  sign-only GRPO both reduce lambda, but the one-seed, three-checkpoint pilot
  is hypothesis-generating only.
- Full matched SFT training is complete for both predeclared seeds through
  30,000 steps, with the complete checkpoint grids. Full-grid behavioral
  evaluation and frozen selection remain pending.
- The framing result is adverse for any broad “general debiasing” claim.
- Human and frontier-superiority claims are paused pending comparable tasks,
  estimands, model endpoints, samples, scorers, and estimators.

## Remaining Work

- Freeze a new untouched comparison suite and method-comparison protocol before
  opening the full SFT trajectories. The already-opened prospective suite
  cannot be reused for method development.
- Evaluate and select the complete two-seed SFT grids only after that freeze;
  complete the two-seed sign-only/scale-matched program when budget permits.
- Finish raw prediction, adapter, environment, and reproduction archival for
  every claim-carrying result.
- Add pair/good-aware inference, estimator-recovery simulations, and explicit
  sensitivity to weakly identified goods and extreme fitted utilities.
- Directly test preservation of frozen ownership-free preferences, not only
  the in-sample pseudo-utility alignment score `W`.
- Run prompt-semantic counterbalancing and at least one broader capability
  benchmark such as IFEval; retain the adverse framing result.
- Build a venue-agnostic manuscript around the causal mechanism and its
  boundary conditions. Do not describe `draft/aaai27/` as the active target.
  Full ordering and claim restrictions are in `PAPER_READINESS.md` and
  `RESEARCH_ROADMAP.md`.
