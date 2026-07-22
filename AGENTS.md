# AGENTS.md

This file gives Codex context about this project so it can help effectively without re-reading everything from scratch.

---

## Project Overview

**lambda-zero** (`loss_aversion_rl`) fine-tunes Qwen2.5-7B-Instruct with **GRPO** to reduce measured loss aversion in an endowment-effect task.

The project builds on a separate behavioral-economics baseline repo, `loss_aversion/`, which estimates **λ** (loss aversion coefficient) across frontier LLMs. This repo asks whether targeted RL can make a 7B model less loss-averse than much larger models.

### Authoritative paper direction

The primary paper model is now **Qwen-own-delta GRPO**, checkpoint 8,000. The
frontier-consensus-delta model is the reward-source ablation. The 8k
`test_goods` estimate (`lambda = 0.111`, `eta = 0.504`) is validation-only
because this set informed reward construction and checkpoint selection. See
`PAPER_READINESS.md` before making paper claims; it overrides older role and
readiness descriptions elsewhere in this file.

### Thesis in one sentence
A 7B model can be post-trained to reduce ownership-dependent choice while
preserving its own estimated preference ordering.

> **Status (July 22, 2026):** Exploratory checkpoint 8,000 remains the primary
> candidate. Confirmatory seeds 1 and 2 both reached the frozen 30,000-step
> endpoint with the complete 15-checkpoint grid; their frozen ID selection is
> **done** (seed 1 → step 2,000, seed 2 → step 6,000), with one-shot OOD and
> GSM8K per selected checkpoint still pending (replication verdict PENDING).
> Matched-base and ownership-free pseudo-utility validation are complete.
> Matched baselines, stronger inference, and complete raw-artifact archival
> remain required for a full paper.

---

## Core Results

| Model / treatment | λ̂ | η̂ | Notes |
|---|---:|---:|---|
| Historical Together-hosted Qwen base | 11.75 | 1.52 | Not yet matched to the local post-training evaluator |
| Qwen-own-delta GRPO, step 8k ID | 0.111 | 0.504 | Primary-model validation; selected on `test_goods` |
| Qwen-own-delta GRPO, step 8k OOD | 0.226 | 0.790 | Intended final evidence; raw artifacts pending |
| Consensus-delta GRPO, baseline | 0.177 | -0.048 | Reward-source ablation/reference |
| Qwen-7B-GRPO, debias | 0.205 | 0.090 | Treatment robustness |
| Qwen-7B-GRPO, forced | 0.173 | -0.379 | Treatment robustness |

Do not use the historical human or frontier comparisons as headline claims
until tasks, estimands, endpoints, samples, scorers, and estimators are
harmonized. Report lambda and eta jointly with direct choice outcomes.

### Held-Out Attribute-Configuration Result

Training and evaluation have zero repeated prompts and zero repeated exact
configurations. They share the same 100 goods and 4,945 unordered goods pairs;
training uses 10 configurations per pair and evaluation reserves two disjoint
configurations. This is a within-benchmark configuration holdout, not an
unseen-goods split.

On the 9,890 held-out configurations:

- Pair-clustered joint attribute test: χ²(8) = 5,097.22, p < 10^-1000.
- Leave-one-good-out joint test: χ²(8) = 756.05, p < 10^-150.
- Attribute profiles explain 48.8% of within-pair/perspective variation.
- At least one perspective's binary answer flips for 42.18% of goods pairs.
- Adding attributes reduces pair-grouped cross-validation MSE by 47.43%.

These are effects of generic 3×3 ordinal attribute profiles, not separate
semantic attributes such as flavor or scent. The current split is sufficient
for the core within-benchmark paper claim. A separate 50-good OOD suite exists
for external-validity analysis; no additional OOD run is required by the
attribute result. Do not treat the compositional split itself as unseen-goods
evidence. See `ATTRIBUTE_EFFECTS.md` and `eval/analyze_attribute_effects.py`.

---

## Experimental Paradigm

Each LLM is prompted:

> You receive good X (with attributes i, j). Would you trade for good Y (with attributes k, l)? Answer Yes or No.

Each case is run from both perspectives:
- X-perspective: endowed with X, offered Y.
- Y-perspective: endowed with Y, offered X.

A rational agent chooses based on which good is better, not which good it owns. A loss-averse agent keeps whichever good it is endowed with. This asymmetry is what λ measures.

Structural model:
```text
z = (1 + λ) · exp(U_X) - exp(U_Y) + η
```

where U = α + β item/attribute fixed effects, λ is loss aversion, and η is status-quo bias.

---

## Phase Status

| Phase | Status | Deliverable |
|---|---|---|
| 0. Frontier-model baseline | ✅ Done in `loss_aversion/` | λ̂ for 9 frontier models |
| 1. Small-model baseline | ✅ Matched | Exact local base λ̂ = 7.637, η̂ = 1.007 |
| 2. Qwen-own reward design | 🟡 Partly validated | Ownership-free validation complete; leakage-clean construction pending |
| 3. Qwen-own GRPO training | 🟡 Replication training complete | Seeds 1 and 2 reached 30k; frozen evaluations pending |
| 4. Primary-model evaluation | 🟡 Partially complete | ID validation and reported OOD table exist; raw artifacts pending |
| 5. Reward-source ablation | ✅ Consensus run complete | Consensus ID result committed |
| 6. Baselines/robustness | ⏳ Next | SFT, sign-only GRPO, robust inference, capability retention |
| 7. Paper writeup | ⏳ | Full paper after Priority-0 items in `PAPER_READINESS.md` |

---

## Repository Structure

```text
lambda-zero/
├── data/
│   ├── trial_goods.json
│   ├── test_goods.json
│   ├── remaining_goods.json
│   └── deltas/
│       ├── delta_consensus_v3.json       # Frontier-consensus ablation reward
│       ├── delta_qwen_base.json          # Primary Qwen-own pseudo-utility reward
│       └── build_delta_qwen_base.py
├── eval/
│   ├── run_all_models.py
│   ├── run_qwen_local.py                 # vLLM-free local/HF evaluator
│   ├── estimate_qwen_base.py
│   ├── estimate_qwen_grpo.py
│   ├── estimate_qwen_grpo_debias.py
│   ├── estimate_qwen_grpo_forced.py
│   ├── analyze_attribute_effects.py      # Held-out attribute-sensitivity diagnostics
│   └── core_exp_refactored.py            # Shared structural estimator
├── train/
│   ├── grpo_train.py
│   ├── reward_functions.py
│   ├── prompt_builder.py
│   ├── configs/qwen25_7b.yaml
│   ├── configs/qwen25_7b_qwen_delta.yaml
│   ├── submit_train_glong.pbs
│   ├── submit_train_glong_qwen_delta.pbs
│   └── submit_eval_treatments.pbs
├── baseline/Qwen-7B/
├── baseline/Qwen-7B-GRPO/
├── debias/Qwen-7B-GRPO/
├── forced/Qwen-7B-GRPO/
├── monitor.sh
├── plot_training.py
├── ATTRIBUTE_EFFECTS.md
├── HISTORY.md
├── PROJECT_OVERVIEW.md
├── TIMELINE.md
└── README.md
```

---

## Reward Function and Paper Roles

The primary run uses Qwen's own pseudo-utility from
`data/deltas/delta_qwen_base.json`. The completed consensus-delta run uses the
same reward form as a reward-source ablation.

```python
if perspective == "X":
    rational = "No" if delta > 0 else "Yes"
else:
    rational = "Yes" if delta > 0 else "No"
return abs(delta) if response == rational else -abs(delta)
```

The pure symmetry reward was rejected because an always-`No` model can be
symmetric while still ownership-dependent. Neither delta source should be
described as objective cardinal ground truth.

---

## Common Tasks

### Train consensus-delta ablation
```bash
python train/grpo_train.py --config train/configs/qwen25_7b.yaml --mode train
```

### Train primary Qwen-own-delta GRPO
```bash
qsub train/submit_train_glong_qwen_delta.pbs
```

### Evaluate GRPO checkpoint
```bash
python eval/run_qwen_local.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path checkpoints/grpo_20260427_1724/final \
  --model_name Qwen-7B-GRPO \
  --data_file data/test_goods.json \
  --treatment baseline \
  --batch_size 8 \
  --yes

python eval/estimate_qwen_grpo.py
```

### Monitor / plot training
```bash
./monitor.sh
python plot_training.py logs/<training-log>.out
```

---

## Critical Conventions

1. **Do not casually edit `eval/core_exp_refactored.py`**; it is the shared structural estimation engine.
2. **Do not delete `loss_aversion_X.json`, `loss_aversion_Y.json`, or `completed_index.json`**; runners use checkpoint/resume logic.
3. **Use Model A with `T=1` for logprob evaluations.** `T=0` caused a silent zero-Jacobian failure before.
4. **Keep train prompts aligned with eval prompts.**
5. **Prefer LoRA over full fine-tuning** because the memory budget is a single A100.
6. **vLLM is optional, not the working dependency.** NSCC-compatible vLLM failed; the working path uses plain HF generation/evaluation.
7. **Do not select checkpoints from training reward alone.** Recent plots show high zero-std/DAPO filtering, weak reward trend, and nontrivial KL drift; validate with held-out structural λ̂.
8. **Describe `test_goods.json` precisely.** It holds out exact attribute configurations/questions, while goods and goods-pair identities are shared with training. The separate 50-good OOD suite is external-validity evidence, not part of this split.
9. **Keep optimization layers distinct.** GRPO's `beta=0.04` is a KL coefficient; structural beta denotes attribute-profile effects. The matched local base is behavioral step 0, while the historical Qwen utility table is the fixed reward source.
10. **Do not add exploratory checkpoints to frozen selection.** Step 600 may be used to diagnose early training because saves occur every 200 steps, but confirmatory selection remains the 15 checkpoints from 2k to 30k at 2k intervals.

---

*Last updated: July 21, 2026*
