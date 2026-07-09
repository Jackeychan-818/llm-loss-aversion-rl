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

## Current Research Snapshot

- Baseline Qwen2.5-7B-Instruct: `lambda_before = 11.7519` (SE `1.2219`), `eta = 1.5178` (SE `0.0852`).
- GRPO Qwen2.5-7B-Instruct LoRA: `lambda_after = 0.1765` (SE `0.0050`), `eta = -0.0477` (SE `0.0240`).
- Held-out evaluation used `test_goods.json` with 9,890 cases.
- Reduction in loss aversion: approximately 98.5%.
- The result is below the human benchmark usually cited around `lambda ~= 2.25`.
- Debias and forced treatments are also below the human benchmark: `0.205` and `0.173`.
- Current ablation: retraining with Qwen-7B's own NLS delta file, `data/deltas/delta_qwen_base.json`.

## Remaining Work

- Cross-model comparison against the frontier-model baseline.
- Ablations for reward source, beta/KL strength, learning rate, checkpoint selection, and generalization.
- Paper figures and writeup.
- Training-health audit: recent convergence plots suggest high DAPO filtering and weak reward improvement despite the strong final structural result, so checkpoint selection and held-out validation should be treated carefully.
