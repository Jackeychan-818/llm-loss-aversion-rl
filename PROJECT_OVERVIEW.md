# Project Overview — lambda-zero
*Last updated: July 15, 2026*

---

## What This Project Is

**lambda-zero** is a reinforcement learning research project testing whether a small open-source model can be trained to reduce loss aversion below frontier-model and human benchmarks.

The project fine-tunes **Qwen2.5-7B-Instruct** with **GRPO** using a rationality-based reward. The main held-out attribute-configuration result is large: structural estimation moved Qwen from **λ̂_before = 11.75** to **λ̂_after = 0.177**, a **98.5% reduction**.

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

Estimated from the baseline treatment with Model A NLS, `T=1`.

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

### GRPO Fine-Tuned Qwen2.5-7B-Instruct

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

Interpretation: the fine-tuned model is no longer dominated by the `(No, No)` status-quo pattern. The post-training λ estimates are below the usual human benchmark around λ ≈ 2-2.5.

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
The current split is sufficient for the core within-benchmark paper claim. The
existing 50-good OOD suite is separate external-validity evidence rather than a
prerequisite for that claim; no additional OOD run is required by this
analysis. Full methods and reproducibility details are in
`ATTRIBUTE_EFFECTS.md` and `eval/analyze_attribute_effects.py`.

---

## Phase Status

| Phase | Status | Deliverable |
|---|---|---|
| 0. Frontier-model baseline | ✅ Done in `loss_aversion/` | λ̂ for 9 frontier models |
| 1. Small-model baseline | ✅ Done | Qwen λ̂_before = 11.75 |
| 2. Reward design | ✅ Done | Utility-weighted reward using consensus δ̃ v3 |
| 3. GRPO training | ✅ Done | Main LoRA checkpoint and NSCC training pipeline |
| 4. Post-training evaluation | ✅ Done | Baseline/debias/forced estimates and held-out attribute-sensitivity analysis |
| 5. Cross-model comparison | ⏳ Next | Qwen before/after vs frontier models |
| 6. Ablations | 🟡 Current | Qwen-delta reward run, checkpoint selection, training-health audit |
| 7. Paper writeup | ⏳ | NeurIPS workshop / ICML full-paper path |

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

## Reward Function

The main run uses a utility-weighted rationality reward from `data/deltas/delta_consensus_v3.json`, a consensus δ̃ from six frontier models.

```python
def reward(response, perspective, delta):
    # delta = U_X - U_Y from frontier consensus
    if perspective == "X":
        rational = "No" if delta > 0 else "Yes"
    else:
        rational = "Yes" if delta > 0 else "No"
    return abs(delta) if response == rational else -abs(delta)
```

The pure symmetry reward was rejected because a model that always says "No" from both perspectives can appear symmetric while still being economically irrational.

The current ablation retrains from scratch with Qwen-7B's own NLS utility differences from `data/deltas/delta_qwen_base.json`.

---

## Training And Evaluation Infrastructure

- Base model: `Qwen2.5-7B-Instruct`.
- Fine-tuning: LoRA rank 16 with TRL `GRPOTrainer`.
- Main training data: `data/remaining_goods.json`.
- Evaluation data: `data/test_goods.json`.
- Main reward data: `data/deltas/delta_consensus_v3.json`.
- Ablation reward data: `data/deltas/delta_qwen_base.json`.
- Main training config: `train/configs/qwen25_7b.yaml`.
- Qwen-delta config: `train/configs/qwen25_7b_qwen_delta.yaml`.
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

- Compare λ̂_after against the 9 frontier-model baselines from `loss_aversion/`.
- Finish and evaluate the Qwen-delta ablation.
- Select and justify final checkpoint(s).
- Audit training-health plots: recent plots show high zero-std/DAPO filtering and weak reward trend, so checkpoint choice should not rely on training reward alone.
- Run ablations for reward source, KL β, learning rate, and LoRA rank.
- Optionally report the existing 50-good OOD suite as separate external-validity evidence; no additional OOD run is required for the core within-benchmark claim.
- Build paper figures and write the workshop submission.

---

## Critical Conventions

1. Do not modify `eval/core_exp_refactored.py` casually; it is the shared structural estimation engine.
2. Do not delete `loss_aversion_X.json`, `loss_aversion_Y.json`, or `completed_index.json`; runners use checkpoint/resume logic.
3. Use `T=1` for Model A NLS when logprobs are available. `T=0` can silently produce degenerate zero-Jacobian results.
4. Keep train prompts aligned with eval prompts.
5. Prefer LoRA over full fine-tuning because the memory budget is a single A100.
6. vLLM is not required for the working NSCC path; plain HF generation/evaluation is the stable path.
