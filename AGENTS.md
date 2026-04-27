# AGENTS.md

This file gives Codex context about this project so it can help effectively without re-reading everything from scratch.

---

## Project Overview

**`loss_aversion_rl`** — a reinforcement learning project that fine-tunes a small open-source LLM (Qwen2.5-7B-Instruct) using **GRPO (Group Relative Policy Optimization)** to reduce loss aversion below frontier models.

This project builds on top of an existing behavioral economics study (separate repo, `loss_aversion/`) that measures **λ** (loss aversion coefficient) across 9 LLMs using an endowment-effect paradigm. That study established that frontier LLMs show measurable loss aversion. This project asks whether targeted RL can correct it.

### Thesis in one sentence
A 7B model, fine-tuned with GRPO using a rationality-based reward, can achieve lower loss aversion (λ → 0) than frontier models 10–100× its size.

> **Note (April 2026):** The specific reward function and training direction are still under active discussion. Multiple reward designs are being considered (see Key Design Decisions). The immediate priority is establishing **λ̂_before** for Qwen2.5-7B-Instruct before any fine-tuning begins.

---

## Background — The Experimental Paradigm

### Endowment effect task
Each LLM is prompted:

> You receive good X (with attributes i, j). Would you trade for good Y (with attributes k, l)? Answer Yes or No.

Each case is run from **both perspectives**:
- X-perspective: endowed with X, offered Y
- Y-perspective: endowed with Y, offered X

A *rational* agent makes symmetric choices: the decision depends only on which good is genuinely better, not on which is endowed. A *loss-averse* agent keeps whichever good it's endowed with. This asymmetry is what λ measures.

### Three treatments
- `baseline` — plain trade question
- `debias` — instructs the model to ignore status quo / gain-loss framing
- `forced` — introduces a 50% random swap probability

### Structural model
Utility specification:
```
z = (1 + λ) · exp(U_X) − exp(U_Y) + η
```
where U = α + β (item + attribute fixed effects), λ is loss aversion, η is status-quo bias.

Estimation methods (in `core_exp_refactored.py`):
- **Model A (NLS)** — Nonlinear Least Squares, for initialization
- **Model B (SMS Bootstrap)** — Smoothed Maximum Score + 500 bootstrap replications, main estimator for binary (T=0) data
- **Model C (Logit MLE)** — for models with logprob access

---

## Project Phases

| Phase | Status | Deliverable |
|---|---|---|
| 0. Econ baseline (9 frontier models) | ✅ Done in `loss_aversion/` | λ̂ for GPT-5, GPT-4o, GPT-3.5, Codex, Gemini, Llama-70B, DeepSeek-R1, Apertus-70B, GPT-OSS-120B |
| 1. Small-model baseline | 🔴 **Current focus** | **λ̂_before** for Qwen2.5-7B — must complete before any training |
| 2. Reward function design | 🔴 **Under discussion** | `reward_functions.py` — training direction not yet decided |
| 3. GRPO training | ⏳ Blocked on 1 & 2 | LoRA checkpoints |
| 4. Post-training evaluation | ⏳ | **λ̂_after** |
| 5. Cross-model comparison | ⏳ | Final paper figures |
| 6. Ablations | ⏳ | Reward type, model size, generalization |
| 7. Paper writeup | ⏳ | NeurIPS / ICML submission |

---

## Repository Structure

```
loss_aversion_rl/
├── data/
│   ├── trial_goods.json          # Validation prompts
│   ├── test_goods.json           # Evaluation prompts
│   └── remaining_goods.json      # Training prompts (largest set)
│
├── eval/                         # Reused from econ project
│   ├── run_all_models.py         # Experiment runner (copy from loss_aversion/)
│   ├── core_exp_refactored.py    # Structural estimation engine
│   ├── estimate_qwen_base.py     # λ̂_before
│   └── estimate_qwen_grpo.py     # λ̂_after
│
├── train/                        # NEW — RL code
│   ├── grpo_train.py             # Main GRPO training loop
│   ├── reward_functions.py       # Symmetry + utility rewards
│   ├── prompt_builder.py         # Generates X/Y perspective pairs
│   └── configs/
│       └── qwen25_7b_symmetry.yaml
│
├── results/
│   ├── baseline/                 # λ̂_before outputs
│   ├── grpo_checkpoints/         # LoRA weights per training step
│   └── post_training/            # λ̂_after outputs
│
├── notebooks/
│   └── compare_lambdas.ipynb     # Final comparison figure
│
├── requirements.txt
├── AGENTS.md                     # ← this file
└── README.md
```

---

## Key Design Decisions

### Base model
**Qwen2.5-7B-Instruct** — chosen because:
- Well-supported by TRL, Unsloth, LLaMA-Factory for fine-tuning
- Not quantized (clean for later GRPO + LoRA)
- Represents a model family not yet in the econ baseline (fair addition)
- Fits on a single A100 40GB for GRPO training

### RL algorithm
**GRPO** (from DeepSeekMath / DeepSeek-R1), not PPO. Reason: our reward should be a clean verifiable signal, so we don't need a learned value network. Considering **Dr. GRPO** variant because our outputs are single-token (Yes/No), which is the regime where Dr. GRPO's length-bias fix matters most.

### Reward function — ⚠️ Under Discussion
The specific reward design has **not been finalized**. Candidates being considered:

**Candidate A — Symmetry reward:**
```
+1  if choice(X → Y) is consistent with choice(Y → X)
−1  if model keeps whichever good is endowed (loss-averse)
```
Self-contained, no dependence on external utility estimates.

**Candidate B — Utility-based reward:**
Using δ̃ (counterfactual utility gap) from the econ structural model as ground truth.

**Other directions** may also be explored. Do not treat any reward design as final until explicitly confirmed.

### Hyperparameters
| Param | Value | Rationale |
|---|---|---|
| Group size G | 8 | Binary reward, low variance — don't need G=16 |
| Sampling temp | 0.7 | Diversity without chaos |
| Clip ε | 0.2 | Standard |
| KL coefficient β | 0.04 | Tighter than DeepSeek-R1's 0.001 — we want to stay near base |
| Learning rate | 1e-6 | Conservative |
| LoRA rank | 16 | Standard for 7B models |

---

## Critical Conventions

### Data format (from the econ project)
Each JSON entry has this shape:
```json
{
  "case_id": 1,
  "X_num": 0,
  "Y_num": 1,
  "attr": [0, 1, 2, 1],       // i, j, k, l
  "Yes / No prob": [0.72, 0.28],
  "output": "full prompt + response"
}
```
`X` and `Y` JSON files are paired — same `case_id` in each represents the two perspectives of the same pair.

### Model registry pattern
New models are added to `MODEL_REGISTRY` in `run_all_models.py`. Three styles:
- **Style A:** logprobs + temp=0 (most flexible, best for MLE estimation)
- **Style B:** no logprobs, temp=0 (binary responses, use SMS Bootstrap)
- **Style C:** no temp=0 (reasoning models)

For Qwen via Together AI: Style B initially, upgrade to Style A when we host locally with vLLM.

### Case ID continuity
Case IDs continue across files: `trial_goods` → `test_goods` → `remaining_goods`. The runner reads `completed_index.json` to know the offset.

---

## Common Tasks

### Run baseline for a new model
```bash
python eval/run_all_models.py Qwen-7B trial_goods.json     baseline
python eval/run_all_models.py Qwen-7B test_goods.json      baseline
python eval/run_all_models.py Qwen-7B remaining_goods.json baseline
```

### Estimate λ̂
```bash
python eval/estimate_qwen_base.py
# Reads loss_aversion_X.json + loss_aversion_Y.json
# Writes results to baseline/Qwen-7B/Model_1/
```

### GRPO training (planned)
```bash
python train/grpo_train.py --config train/configs/qwen25_7b_symmetry.yaml
```

---

## Papers the Project Builds On

- **DeepSeekMath (Shao et al., 2024)** — original GRPO formulation
- **DeepSeek-R1 (2025)** — GRPO at scale, reward design patterns
- **Dr. GRPO (Sea AI Lab, 2025)** — length-bias correction, relevant because our outputs are 1 token
- **DAPO (ByteDance, 2025)** — decoupled clip + dynamic sampling (fallback if GRPO underperforms)
- **Kahneman & Tversky (1979)** — prospect theory, defines λ
- **List (2004)** — endowment effect diminishes with market experience (motivates "RL = experience" framing)

---

## Things Codex Should Know

1. **Don't touch `core_exp_refactored.py` without care** — it's the shared econ estimation engine. Bugs propagate into both my results and my coauthor's.
2. **The `run_all_models.py` has heavy checkpoint logic** — don't casually delete `loss_aversion_X.json` / `loss_aversion_Y.json` or the `completed_index.json`.
3. **Prefer LoRA over full fine-tuning** — memory budget is a single A100.
4. **The reward function and training direction are not yet finalized** — do not hard-code any specific reward design. Symmetry is one candidate but not confirmed.
5. **TRL's `GRPOTrainer` is the intended training library.** vLLM for generation during rollouts.

---

*Last updated: April 13, 2026*
