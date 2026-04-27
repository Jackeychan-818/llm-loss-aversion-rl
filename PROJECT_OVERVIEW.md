# Project Overview — lambda-zero
*Last updated: April 16, 2026*

---

## What This Project Is

**lambda-zero** is a reinforcement learning research project that asks one question: can a small 7B model, fine-tuned with GRPO, achieve lower loss aversion than frontier models 10–100× its size?

The foundation is a behavioral economics concept called the **endowment effect** — when people (or LLMs) are given an object, they irrationally overvalue it just because they own it. This is driven by **loss aversion**: the psychological pain of giving something up outweighs the pleasure of gaining something equivalent. Kahneman & Tversky (1992) measured this in humans as λ ≈ 2.25. Frontier LLMs show measurable loss aversion too.

This project fine-tunes **Qwen2.5-7B-Instruct** using **GRPO** (Group Relative Policy Optimization) with a rationality-based reward signal to reduce λ toward 0 — making the model behave more like a rational economic agent than a loss-averse one.

---

## The Experimental Paradigm

Each LLM is shown an endowment effect task:

> "You receive good X (with attributes i, j). Would you trade for good Y (with attributes k, l)? Answer Yes or No."

The same case is run from **both perspectives**:
- **X-perspective**: model is endowed with X, offered Y
- **Y-perspective**: model is endowed with Y, offered X

A rational agent makes **symmetric** choices — the decision depends only on which good is genuinely better. A loss-averse agent keeps whichever good it currently holds. The degree of asymmetry is what λ measures.

The structural model (from the econ baseline project) estimates λ and η using:
```
z = (1 + λ) · exp(U_X) − exp(U_Y) + η
```
where λ = loss aversion, η = status-quo bias, and U is estimated via item/attribute fixed effects.

Three treatments exist: `baseline` (plain question), `debias` (instructs model to ignore framing), and `forced` (50% random swap probability).

---

## What Has Been Done

### Phase 0 — Frontier Model Baseline ✅
Completed in the separate `loss_aversion/` repo. λ̂ has been estimated for 9 frontier models (GPT-4o, Claude, Gemini, Llama-70B, DeepSeek-R1, etc.) using the endowment effect paradigm. These serve as the benchmark Qwen-7B needs to beat after GRPO training.

### Phase 1 — Qwen-7B Baseline ✅
**Completed April 14, 2026.**

Qwen2.5-7B-Instruct was added to the experiment runner (`eval/run_all_models.py`) as a **Style A** model — meaning it returns logprobs, enabling the strongest estimator (Model A, NLS with T=1).

API calls were made via Together AI (`Qwen/Qwen2.5-7B-Instruct-Turbo`) across all three data files:
- `trial_goods.json` → sanity check
- `test_goods.json` → evaluation set
- `remaining_goods.json` → training set (largest)

Results (baseline treatment, Model A NLS, T=1):

| Parameter | Estimate | Std. Error |
|---|---|---|
| λ (loss aversion) | **11.75** | 1.22 |
| η (status-quo bias) | 1.52 | 0.09 |

Raw choice data: 99.07% (No, No) — the model refuses to trade from both perspectives almost always. The "No Bias" counterfactual shows a ~53/47 split, confirming the underlying utilities are balanced — it is loss aversion and status-quo bias driving the refusal, not the goods themselves.

**Key estimation note discovered:** T must be set to 1 (not 0) for Style A models with logprobs. T=0 uses a step function → zero Jacobian → NLS cannot converge. This caused a silent failure initially (λ̂ = 0 with all-zero standard errors) which was diagnosed and fixed.

**Interpretation:** λ̂ = 11.75 is roughly 5× the human benchmark (λ ≈ 2.25) and expected to be substantially above frontier models. This provides a strong training signal for GRPO — there is a large gap to close.

---

## What Needs to Be Decided (Current Blocker)

### Reward Function Design — Phase 2 🔴

This is the only thing blocking training. The reward function has not been finalized. Key candidates:

**Candidate A — Symmetry reward:**
+1 if choice(X→Y) is consistent with choice(Y→X), −1 if model keeps whichever good it holds.

*Problem:* A model that says No from both perspectives is perfectly symmetric (λ=0) and would receive +1. But this is driven by status-quo bias (η), not rationality. The model would learn to say No consistently — correct on the loss aversion metric but still irrational.

**Candidate B — Utility-based reward:**
Reward the model for choosing the objectively better good based on δ̃ (counterfactual utility gap from the structural model). Penalizes both λ and η.

*Problem:* Depends on structural estimates being reliable, and requires computing δ̃ per case during training.

**Candidate C — Hybrid / logprob-based:**
Use the logprob gap between Yes and No as a continuous reward signal. Reward the model when logprob(Yes | X-perspective) ≈ logprob(Yes | Y-perspective) for the same pair, weighted by utility difference. Directly targets λ→0 at the individual case level.

**Key insight from Kahneman:** λ (loss aversion) and η (status-quo bias) are distinct. The 99% No/No pattern is driven by both. A reward targeting only symmetry fixes λ but ignores η. A complete reward should push toward rational trading — choosing the better good — not just symmetric trading.

---

## What Needs to Be Prepared

### 1. Training infrastructure
- GPU access: A100 40GB (fits Qwen-7B with LoRA rank 16 + GRPO rollouts)
- vLLM for fast generation during rollouts
- TRL's `GRPOTrainer` as the training library
- Cost estimate: ~$70–120 total if renting cloud GPU (Vast.ai ~$0.67/hr), <$30 if own cluster

### 2. Dynamic sampling (DAPO technique — critical)
Standard GRPO will fail on this dataset because 99% of outputs are No → all G outputs in a batch get the same reward → zero advantage → zero gradient. **Dynamic sampling must be implemented**: discard batches where all G outputs are identical, keep sampling until the batch contains both rewarded and unrewarded responses. This is the single most important engineering decision before training.

### 3. Reward function implementation
Once the design decision is made (see above), implement `train/reward_functions.py`. The reward must be computable per-pair (X-perspective + Y-perspective together), not per single response.

### 4. Prompt builder
`train/prompt_builder.py` needs to generate paired (X, Y) perspective prompts from `remaining_goods.json` for training, maintaining case_id alignment.

### 5. Training config
`train/configs/qwen25_7b_symmetry.yaml` needs hyperparameters confirmed:
- Group size G = 8
- Sampling temperature = 0.7 (needed for output diversity — model defaults to No at temp=0)
- Clip ε = 0.2 (or decoupled ε_low/ε_high if adopting DAPO)
- KL coefficient β = 0.04
- Learning rate = 1e-6
- LoRA rank = 16

---

## What Needs to Be Done (Remaining Phases)

### Phase 3 — GRPO Training ⏳
Run `train/grpo_train.py` with the finalized reward function. Start small: sanity check on `trial_goods.json` (100 steps) to verify reward fires correctly and loss moves in the right direction before committing to full training on `remaining_goods.json`.

### Phase 4 — Post-Training Evaluation ⏳
Re-run the experiment runner on `test_goods.json` with the fine-tuned checkpoint. Run `eval/estimate_qwen_grpo.py` to compute **λ̂_after**. Compare against λ̂_before = 11.75. The question: did GRPO reduce λ?

### Phase 5 — Cross-Model Comparison ⏳
Pull frontier model λ̂ values from the `loss_aversion/` repo. Place Qwen-7B (before and after) in context. This is the core result of the paper: a 7B model vs GPT-4o, Claude, etc.

### Phase 6 — Ablations ⏳
- Different reward functions (symmetry vs utility vs hybrid)
- Different KL coefficients and learning rates
- Generalization: does lower λ persist on out-of-distribution goods?

### Phase 7 — Paper Writeup ⏳
Target: NeurIPS 2026 workshop (deadline ~August 2026) for early results, then ICML 2027 main track for the full paper. Workshop papers are non-archival — submitting to a workshop does not block the ICML submission.

---

## Key Papers to Read

| Paper | Why It Matters |
|---|---|
| DeepSeekMath (Shao et al., 2024) | Original GRPO formulation |
| DeepSeek-R1 (2025, arxiv 2501.12948) | GRPO at scale, reward design patterns |
| DAPO (ByteDance, 2025, arxiv 2503.14476) | Dynamic sampling — critical for your 99% No/No problem |
| Dr. GRPO (Sea AI Lab, 2025) | Length-bias correction for single-token outputs |
| λ-GRPO (2025, arxiv 2510.06870) | Learnable token preferences, unified GRPO framework |
| Kahneman & Tversky (1979) | Prospect theory, defines λ |
| Tversky & Kahneman (1992) | λ = 2.25 human benchmark, α = β = 0.88 |
| List (2004) | Endowment effect diminishes with market experience — motivates "RL = experience" framing |

---

## Repository Structure

```
loss_aversion_rl/
├── data/
│   ├── trial_goods.json          # Sanity check / validation
│   ├── test_goods.json           # Evaluation set
│   └── remaining_goods.json      # Training set (largest)
├── eval/
│   ├── run_all_models.py         # Experiment runner
│   ├── core_exp_refactored.py    # Structural estimation engine (do not modify carelessly)
│   ├── estimate_qwen_base.py     # λ̂_before (done)
│   └── estimate_qwen_grpo.py     # λ̂_after (not yet run)
├── train/                        # To be built
│   ├── grpo_train.py
│   ├── reward_functions.py       # ← reward design decision goes here
│   ├── prompt_builder.py
│   └── configs/qwen25_7b.yaml
├── baseline/Qwen-7B/             # λ̂_before results
├── results/                      # λ̂_after results (post training)
├── notebooks/compare_lambdas.ipynb
├── CLAUDE.md
├── TIMELINE.md
└── PROJECT_OVERVIEW.md           # ← this file
```

---

## Current Status Summary

| What | Status |
|---|---|
| Experimental paradigm | ✅ Inherited from loss_aversion/ repo |
| Frontier model baselines (Phase 0) | ✅ Done |
| Qwen-7B API setup (Style A, logprobs) | ✅ Done |
| λ̂_before = 11.75 | ✅ Done |
| Reward function decision | 🔴 Blocker |
| Dynamic sampling implementation | 🔴 Must do before training |
| GRPO training | ⏳ Blocked |
| λ̂_after | ⏳ Blocked |
| Cross-model comparison | ⏳ Blocked |
| Ablations | ⏳ Blocked |
| Workshop paper submission | ⏳ ~August 2026 |
| ICML 2027 submission | ⏳ ~January 2027 |
