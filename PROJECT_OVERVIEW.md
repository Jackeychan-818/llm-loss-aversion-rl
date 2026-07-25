# Project Overview — lambda-zero
*Last updated: July 25, 2026*

---

## What This Project Is

**lambda-zero** is a reinforcement learning research project testing whether a
small open-source model can be trained to reduce ownership-dependent choice
behaviour while preserving its own estimated preference ordering.

The primary paper model fine-tunes **Qwen2.5-7B-Instruct** with **GRPO** using
Qwen's own estimated utility differences. Checkpoint 8,000 is the current
selected candidate. Its `test_goods` estimate is **λ̂ = 0.111**, **η̂ = 0.504**;
because that set was used for checkpoint selection, this is a validation
result. The separately frozen OOD evaluation reported in the current draft is
**λ̂ = 0.226**, **η̂ = 0.790**, with **49.46%** choice consistency, pending
archival of the underlying raw artifacts. The frontier-consensus-delta model
is now the reward-source ablation rather than the main output.

See `PAPER_READINESS.md` for the authoritative distinction between completed,
validation-only, claimed-but-unarchived, and still-required evidence.

---

## Experimental Paradigm

Each LLM is shown an endowment-effect task:

> "You receive good X (with attributes i, j). Would you trade for good Y (with attributes k, l)? Answer Yes or No."

Each case is run from both perspectives:
- **X-perspective:** endowed with X, offered Y.
- **Y-perspective:** endowed with Y, offered X.

A rational agent chooses based on which good is better. A loss-averse agent keeps whichever good it currently owns. The asymmetry is summarized by λ.

The structural model estimates λ and η with:
```text
z = (1 + λ) · exp(U_X) - exp(U_Y) + η
```

where λ is loss aversion, η is status-quo bias, and U is estimated from item/attribute fixed effects.

---

## Current Results

### Baseline Qwen2.5-7B-Instruct

Estimated from the baseline treatment with Model A NLS, `T=1`. This is a
historical Together-hosted baseline. It used a different endpoint, sample, and
probability scorer from the local adapter evaluation, so the exact local base
checkpoint must be rerun before this is used as the causal before/after value.

| Parameter | Estimate | Std. Error |
|---|---:|---:|
| λ, loss aversion | **11.7519** | 1.2219 |
| η, status-quo bias | **1.5178** | 0.0852 |

Raw choice fractions show the base model almost never trades:

| X \\ Y | Yes | tie | No |
|---|---:|---:|---:|
| Yes | 0.00 | 0.00 | 0.48 |
| tie | 0.00 | 0.00 | 0.00 |
| No | 0.45 | 0.00 | 99.07 |

### Primary: Qwen-Own-Delta GRPO

The economic motivation is agent-specific utility: the main experiment asks
whether post-training can remove the effect of ownership while retaining
Qwen's own estimated ranking, rather than imposing a ranking aggregated from
other models.

| Evaluation role | λ̂ | η̂ | Consistency | Interpretation |
|---|---:|---:|---:|---|
| ID `test_goods` | **0.111** | **0.504** | pending archived table | Validation/checkpoint-selection result for step 8,000 |
| New-goods OOD | **0.226** | **0.790** | **49.46%** | Intended final evaluation; raw artifacts still need archival |

The current Qwen-own delta was fitted from endowed base responses that include
`test_goods` and come from a model that says No approximately 99% of the time.
The paper must therefore validate this pseudo-utility against frozen
ownership-free Qwen preferences and treat `test_goods` as validation.

### Ablation: Frontier-Consensus-Delta GRPO

Evaluation on held-out attribute configurations in `test_goods.json` using `eval/run_qwen_local.py` and the Model A estimators.

| Treatment | λ̂ | SE(λ̂) | η̂ | SE(η̂) |
|---|---:|---:|---:|---:|
| baseline | **0.1765** | 0.0050 | -0.0477 | 0.0240 |
| debias | **0.205** | | 0.090 | |
| forced | **0.173** | | -0.379 | |

Baseline-treatment raw choice fractions after GRPO:

| X \\ Y | Yes | tie | No |
|---|---:|---:|---:|
| Yes | 1.30 | 0.03 | 43.31 |
| tie | 0.02 | 0.00 | 0.07 |
| No | 47.34 | 0.09 | 7.84 |

Interpretation: the consensus-delta model is no longer dominated by the
`(No, No)` status-quo pattern. It is a completed reward-source reference, not
the primary paper output. Human and frontier-superiority comparisons are not
currently treated as established because the estimands and evaluation
pipelines have not been harmonized.

### Held-Out Split and Attribute Sensitivity

Training and evaluation contain **zero repeated prompts and zero repeated exact
attribute configurations**. Both splits use the same 100 goods and the same
4,945 unordered goods pairs. For each pair, training uses 10 configurations and
evaluation reserves two disjoint configurations. The result therefore measures
generalization to held-out questions/configurations for familiar goods and
pairs, not transfer to unseen goods.

A paired within-pair analysis confirms that the held-out attribute changes are
both statistically and practically important:

| Diagnostic | Result |
|---|---:|
| Pair-clustered joint attribute test | χ²(8) = **5,097.22**, p < 10^-1000 |
| Leave-one-good-out joint attribute test | χ²(8) = **756.05**, p < 10^-150 |
| Within-pair/perspective R² | **0.488** |
| Attribute effect range on P(No) | **0.177–0.630** |
| At least one perspective flips across test configurations | **42.18%** of pairs |
| Pair-grouped CV error reduction from adding attributes | **47.43%** |

These diagnostics show that the learned policy remains sensitive to ordinal
attribute quality rather than collapsing to a constant response. The effects
refer to generic 3×3 ordinal profiles, not separate named semantic attributes.
The current split is sufficient for the narrow within-benchmark configuration
claim. For the primary Qwen-own-delta run, however, `test_goods` is validation
because its responses informed reward construction and its estimates informed
checkpoint selection. A separately frozen evaluation is therefore important
for the full-paper claim. Full methods and reproducibility details are in
`ATTRIBUTE_EFFECTS.md`, `eval/analyze_attribute_effects.py`, and
`PAPER_READINESS.md`.

### Frozen prospective unused-configuration test

After all training, frozen checkpoint selection, and confirmatory OOD/GSM8K
evaluation were complete, a second configuration test was constructed from
questions never used by those processes. For each of the 4,945 familiar goods
pairs, the 12 existing validation/training codes were removed, the remaining
69 were deterministically hash-shuffled with seed `20260723`, and 10 were
selected. The committed result contains 49,450 cases (98,900 paired-perspective
prompts per model), has zero overlap with both original splits, and passes
frozen joint-code and marginal-level balance gates.

It was opened **once** on July 24, 2026 for the matched local base and the
already-selected seed 1 step-2,000 and seed 2 step-6,000 checkpoints (only). It
cannot be used for checkpoint selection, reward construction, training, or
method changes. **λ collapses from the matched base 5.946 → ≈0 on both seeds**
(seed1 0.031, seed2 −0.053), with SEs ~3× tighter than `test_goods` and
η/consistency/W matching the ID pattern (W within 0.001 of ID per model). This
remains a familiar-goods, familiar-pairs **configuration** test; OOD-50 is the
distinct new-goods test. Results: `results/frozen_unused_results.json`;
provenance (dataset SHA, evaluator commit, adapter hashes, base model, PBS job
IDs): `results/frozen_unused_evaluation_manifest.json`. Construction, hashes,
and freeze policy: `data/FROZEN_UNUSED_TEST.md`,
`data/frozen_unused_test_goods.manifest.json`.

---

## Phase Status

| Phase | Status | Deliverable |
|---|---|---|
| 0. Frontier-model baseline | ✅ Done in `loss_aversion/` | λ̂ for 9 frontier models |
| 1. Small-model baseline | ✅ Matched | Exact local base: λ̂ = 7.637, η̂ = 1.007 on the local scorer |
| 2. Qwen-own reward design | 🟡 Partly validated | Ownership-free validation complete; leakage-clean reward construction remains open |
| 3. Qwen-own GRPO training | ✅ Replication confirmed | Seeds 1 and 2 completed the frozen grid and passed ID selection, OOD-50, and GSM8K gates |
| 4. Primary-model evaluation | 🟡 Evaluations complete; archival pending | Confirmatory OOD/GSM8K and the prospective unused-configuration test are complete; raw claim-carrying artifacts still need durable publication |
| 5. Reward-source ablation | ✅ Consensus run complete | Consensus ID results committed; reported OOD artifacts need archival |
| 6. Baselines/robustness | ⏳ Next | Matched SFT, sign-only GRPO, prompt-semantic counterbalancing, IFEval, GSM-Symbolic-500 |
| 7. Paper writeup | Deferred | Resume after the next experimental program clarifies the final scope |

---

## Recent Commit History

- `40f4b0e` — added GRPO training code, baseline outputs, consensus δ̃ files, and NSCC setup.
- `c7088c7` — updated project context after the reward decision and corrected hyperparameters.
- `d26ac54` — fixed NSCC resume/vLLM training path.
- `e938efa` — restored NSCC-validated PyTorch module and gradient accumulation.
- `dce0550` — fixed PBS `glong` queue routing.
- `b9c2041` — disabled broken vLLM dependency on NSCC and used plain HF generation.
- `f4a2c9a` — added vLLM-free local Qwen evaluation.
- `b511e53` — added held-out Phase 4 baseline result: λ̂_after = 0.177.
- `c643617` — added debias/forced treatment results, Qwen-delta ablation config/data, and training docs.
- `f2a8710` — added `monitor.sh`, `plot_training.py`, and log append fixes.

See `HISTORY.md` for the chronological project history.

---

## Reward Function and Paper Roles

The primary model uses Qwen's own estimated utility differences from
`data/deltas/delta_qwen_base.json`. The consensus file from six frontier models
uses the same reward form as a reward-source ablation.

```python
def reward(response, perspective, delta):
    # delta = model-specific or consensus pseudo-utility U_X - U_Y
    if perspective == "X":
        rational = "No" if delta > 0 else "Yes"
    else:
        rational = "Yes" if delta > 0 else "No"
    return abs(delta) if response == rational else -abs(delta)
```

The pure same-token symmetry reward was rejected because the meaning of Yes/No
reverses with the endowed good. A model that always says No keeps whichever
good it owns and is therefore ownership-dependent.

The economic reason for making Qwen-own delta primary is that utility is
agent-specific. This advantage is conditional on demonstrating that Qwen's
utility estimate is identified and agrees with ownership-free Qwen choices.
Neither reward source should be called objective cardinal ground truth.

## Optimization and Utility Diagnostics

Training and structural estimation are distinct. GRPO optimizes a DAPO policy
objective using fixed Qwen-own `delta` rewards, clipping `epsilon=0.2`, and KL
coefficient `beta_GRPO=0.04`. The downstream Model-A estimator separately fits
lambda, eta, item alpha, and attribute-profile beta by NLS, with
`U=exp(alpha+beta)`. Structural beta is not the GRPO KL coefficient.

The behavioral trajectory uses the matched local base as step 0 and the frozen
2k–30k @ 2k grid for confirmatory checkpoint selection. Step 600 may be
evaluated as an explicitly post-hoc early-training diagnostic because saves are
every 200 steps; it cannot enter selection or receive OOD candidate testing.

Diagnostics will report GRPO reward/loss, KL, entropy, gradient norm, learning
rate, reward dispersion and DAPO filtering, plus NLS starting/final RSS,
conditioning, multi-start stability, alpha/beta drift, fitted-utility
distributions, and preference-rank preservation. The default NLS starts at
`lambda=eta=0` with pooled-OLS alpha/beta values. The current estimator's
`curve_fit` callback is ignored, so its nominal iteration-history CSV is not a
valid optimization trace; convergence must be captured by a separate
non-destructive diagnostic wrapper.

---

## Training And Evaluation Infrastructure

- Base model: `Qwen2.5-7B-Instruct`.
- Fine-tuning: LoRA rank 16 with TRL `GRPOTrainer`.
- Main training data: `data/remaining_goods.json`.
- Evaluation data: `data/test_goods.json`.
- Primary reward data: `data/deltas/delta_qwen_base.json`.
- Reward-source ablation data: `data/deltas/delta_consensus_v3.json`.
- Primary Qwen-own-delta config: `train/configs/qwen25_7b_qwen_delta.yaml`.
- Consensus-delta ablation config: `train/configs/qwen25_7b.yaml`.
- vLLM-free eval script: `eval/run_qwen_local.py`.
- Monitoring: `monitor.sh`.
- Plotting: `plot_training.py`.

Key main-run hyperparameters:

| Parameter | Value |
|---|---:|
| Group size G | 16 |
| Sampling temperature | 1.5 |
| KL coefficient β | 0.04 |
| Learning rate | 1e-6 |
| LoRA rank | 16 |
| Gradient accumulation | 16 |

---

## Remaining Work

1. ID checkpoint evaluation and the frozen mechanical selection are **done** for
   both replication seeds (seed 1 → step 2,000, seed 2 → step 6,000). Training
   dynamics + structural trajectory posted under `results/training_dynamics/`.
2. DONE — one OOD-50 + one GSM8K per selected checkpoint (seed1@2000,
   seed2@6000) complete; mechanical verdict is **2/2 seeds PASS** (method
   replicates). See `results/seed_replication_report.json`.
3. DONE — the prospective unused-configuration evaluation and the full
   120-scenario framing specificity evaluation are complete. The latter is an
   adverse-transfer boundary result rather than evidence of general debiasing.
4. Add the post-hoc optimization, multi-start, alpha/beta, and utility-trajectory
   diagnostics without changing the frozen selection grid.
5. Archive and reproduce all seed, OOD, framing, and capability artifacts.
6. **Next:** add matched SFT and sign-only GRPO baselines; keep consensus as
   the reward-source ablation.
7. Freeze and run a prompt-semantic robustness suite covering response tokens,
   item labels, display order, attribute order, and paraphrases.
8. Complete IFEval, then add one compact GSM-Symbolic-500 math robustness
   check. Treat confidence calibration as a lower-priority side-effect test.
9. Add pair/good-aware structural robustness and estimator recovery.
10. Retain frontier or human comparisons only if their protocols and estimands
   are made comparable.
11. Defer the full paper rewrite until these experiments clarify the final
    scope.

The design requirements, priority ordering, and safeguards against reusing
opened test sets are recorded in `RESEARCH_ROADMAP.md`.

---

## Critical Conventions

1. Do not modify `eval/core_exp_refactored.py` casually; it is the shared structural estimation engine.
2. Do not delete `loss_aversion_X.json`, `loss_aversion_Y.json`, or `completed_index.json`; runners use checkpoint/resume logic.
3. Use `T=1` for Model A NLS when logprobs are available. `T=0` can silently produce degenerate zero-Jacobian results.
4. Keep train prompts aligned with eval prompts.
5. Prefer LoRA over full fine-tuning because the memory budget is a single A100.
6. vLLM is not required for the working NSCC path; plain HF generation/evaluation is the stable path.
7. Treat `test_goods.json` as validation for Qwen-own-delta reporting: it
   informed both the pseudo-utility construction and checkpoint selection.
8. Report lambda and eta jointly with direct consistency/keep-both/trade-both
   outcomes; do not select or rank models on lambda alone.
