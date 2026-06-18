# CLAUDE.md

This file gives Claude Code context about this project so it can help effectively without re-reading everything from scratch.

---

## Project Overview

**lambda-zero** (`loss_aversion_rl`) — a reinforcement learning project that fine-tunes Qwen2.5-7B-Instruct using **GRPO (Group Relative Policy Optimization)** to reduce loss aversion below frontier models.

This project builds on an existing behavioral economics study (separate repo, `loss_aversion/`) that measures **λ** (loss aversion coefficient) across 9 LLMs using an endowment-effect paradigm. That study established that frontier LLMs show measurable loss aversion. This project asks whether targeted RL can correct it.

### Thesis in one sentence
A 7B model, fine-tuned with GRPO using a rationality-based reward, can achieve lower loss aversion (λ → 0) than frontier models 10–100× its size.

---

## Current Status (April 27, 2026)

**Phase 3 — GRPO Training** is the current focus. All prerequisite work is done:
- λ̂_before = 11.75 established (Phase 1)
- Reward function decided: utility-weighted using consensus δ̃ (Phase 2)
- Training code written: `grpo_train.py`, `reward_functions.py`, `prompt_builder.py`
- First NSCC full run hit the 24h walltime at step 15,265 / 98,900

**Immediate next step:** Resume from `checkpoints/grpo_20260427_1724/checkpoint-15200` with vLLM enabled, preferably via `train/submit_train_glong.pbs` if the full epoch still cannot fit inside g1's 24h walltime.

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
| 0. Econ baseline (9 frontier models) | ✅ Done in `loss_aversion/` | λ̂ for GPT-5, GPT-4o, GPT-3.5, Claude, Gemini, Llama-70B, DeepSeek-R1, Apertus-70B, GPT-OSS-120B |
| 1. Small-model baseline | ✅ Done | **λ̂_before = 11.75** (SE = 1.22), η̂ = 1.52 (SE = 0.09) for Qwen2.5-7B — baseline treatment, Model A (NLS), T=1 |
| 2. Reward function design | ✅ Done | Utility-weighted reward using consensus δ̃ from 6 frontier models (v3) |
| 3. GRPO training | 🟡 **Code written, not yet run** | LoRA checkpoints |
| 4. Post-training evaluation | ⏳ | **λ̂_after** |
| 5. Cross-model comparison | ⏳ | Final paper figures |
| 6. Ablations | ⏳ | Reward type, model size, generalization |
| 7. Paper writeup | ⏳ | NeurIPS / ICML submission |

---

## Repository Structure

```
lambda-zero/
├── data/
│   ├── trial_goods.json              # Validation prompts (990 cases)
│   ├── test_goods.json               # Evaluation prompts (8,910 cases)
│   ├── remaining_goods.json          # Training prompts (49,500 cases — largest set)
│   └── deltas/
│       ├── delta_consensus_v3.json   # ★ Ground truth δ̃ from 6 frontier models (59,400 entries)
│       ├── delta_deepseek.json       # DeepSeek-R1 standalone δ̃ (for ablation)
│       └── _summary.json             # Survey of all 9 models' Yes rates & estimator info
│
├── eval/                             # Reused from econ project
│   ├── run_all_models.py             # Experiment runner (copy from loss_aversion/)
│   ├── core_exp_refactored.py        # Structural estimation engine — DO NOT MODIFY CARELESSLY
│   ├── estimate_qwen_base.py         # λ̂_before
│   └── estimate_qwen_grpo.py         # λ̂_after (not yet run)
│
├── train/                            # RL training code
│   ├── grpo_train.py                 # ✅ Main GRPO training loop using TRL's GRPOTrainer
│   ├── reward_functions.py           # ✅ Utility-weighted reward using frontier δ̃
│   ├── prompt_builder.py             # ✅ Generates X/Y perspective pairs from goods data
│   ├── configs/
│   │   └── qwen25_7b.yaml            # ✅ All hyperparameters
│   ├── setup_nscc.sh                 # One-time NSCC environment setup
│   ├── submit_sanity.pbs             # PBS job: sanity check on trial_goods (gdev, 1.5h)
│   ├── submit_train.pbs              # PBS job: full training on remaining_goods (g1, 24h)
│   └── NSCC_GUIDE.md                 # NSCC ASPIRE 2A reference guide
│
├── baseline/Qwen-7B/                 # λ̂_before results
├── results/                          # λ̂_after results (post training)
├── notebooks/
│   └── compare_lambdas.ipynb         # Final comparison figure
│
├── everyday_goods_full.json          # Master goods reference (item names, attributes, values)
├── CLAUDE.md                         # ← this file
├── KNOWN_ISSUES.md                   # Running log of problems and risks
├── PROJECT_OVERVIEW.md               # Detailed project narrative
└── TIMELINE.md                       # Publication schedule
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
**GRPO** (from DeepSeekMath / DeepSeek-R1), not PPO. Reason: our reward is a clean verifiable signal, so we don't need a learned value network. Considering **Dr. GRPO** variant because our outputs are single-token (Yes/No), which is the regime where Dr. GRPO's length-bias fix matters most.

### Reward function — ✅ Decided (April 2026)
**Utility-weighted reward (Version 2)** using consensus δ̃ from 6 frontier models.

**Ground truth source:** `data/deltas/delta_consensus_v3.json` — mean δ̃ across GPT-4o, GPT-5, GPT-5.2, Gemini, Llama-70B, DeepSeek-R1 (excluded Claude, Apertus-70B, GPT-3.5 due to <5% Yes rates making their utility estimates unreliable).

**Why not Qwen-7B's own δ̃?** Qwen says No 99% of the time → NLS can't learn meaningful α/β → δ̃ values reflect reference category geometry, not real preferences. Frontier models that actually trade produce reliable utility estimates.

**Reward formula:**
```python
def reward(response, perspective, delta):
    # delta = δ̃ = U_X - U_Y from consensus
    if perspective == "X":
        # Endowed with X. δ̃ > 0 means X is better → keep X → rational = "No"
        rational = "No" if delta > 0 else "Yes"
    else:
        # Endowed with Y. δ̃ > 0 means X is better → trade for X → rational = "Yes"
        rational = "Yes" if delta > 0 else "No"
    return abs(delta) if response == rational else -abs(delta)
```

**Coverage:** 64% of cases have unanimous sign agreement across all 6 models. Remaining 36% use the mean δ̃ — the |δ̃| weighting naturally downweights ambiguous cases.

**Rejected alternative:** Symmetry reward (+1 for consistent X/Y choices) was rejected because a model saying No from both perspectives scores +1 while remaining deeply irrational (high η).

### Hyperparameters
| Param | Value | Rationale |
|---|---|---|
| Group size G | 16 | Increased from 8 — needed for output diversity given 99% No rate |
| Sampling temp | 1.5 | Increased from 0.7 — temp<1 makes peaked distributions MORE peaked |
| Clip ε | 0.2 | Standard |
| KL coefficient β | 0.04 | Tighter than DeepSeek-R1's 0.001 — we want to stay near base |
| Learning rate | 1e-6 | Conservative |
| LoRA rank | 16 | Standard for 7B models |
| max_completion_length | 4 | Only need "Yes" or "No" (1-2 tokens) |
| batch_size | 1 | Per-device, with gradient_accumulation_steps=4 |

---

## Training Infrastructure — NSCC ASPIRE 2A

### System details
- **Host:** `aspire2a.nus.edu.sg` (SSH login)
- **GPU:** NVIDIA A100-40GB SXM (4 per node)
- **Scheduler:** PBS Pro (NOT SLURM — use `qsub`, `qstat`, `qdel`)
- **Project ID:** `personal-jackeyc0`
- **Scratch:** `$HOME/scratch/lambda-zero/`
- **PyTorch module:** `pytorch/2.10.0-py3-cu12.6` (Python 3.13, CUDA 12.6)
- **Venv:** `$HOME/scratch/lambda-zero/venv/` (TRL, PEFT, vLLM, transformers)
- **Model weights:** `$HOME/scratch/lambda-zero/models/Qwen2.5-7B-Instruct/`
- **SU remaining:** ~33,389 (enough for ~139 full 24h training runs)

### Queues
| Queue | Max GPUs | Walltime | Use for |
|-------|----------|----------|---------|
| `gdev` | 4 | 2 hours | Quick tests, sanity checks |
| `g1` | 1 | 2–24 hours | Full training runs |
| `glong` | 1 | 24+ hours | Extended training |

Submit via `normal` routing queue: `qsub -q normal train/submit_train.pbs`

### Key commands
```bash
# Environment setup
module purge && module load pytorch/2.10.0-py3-cu12.6
source $HOME/scratch/lambda-zero/venv/bin/activate

# Submit jobs
qsub train/submit_sanity.pbs    # Sanity check (gdev, 1.5h, 100 steps)
qsub train/submit_train.pbs     # Full training (g1, 24h)

# Monitor
qstat -u jackeyc0
tail -f $HOME/scratch/lambda-zero/logs/*.out

# Interactive GPU session (for debugging)
qsub -I -q normal -l select=1:ncpus=8:ngpus=1:mem=80gb -l walltime=01:00:00 -P personal-jackeyc0
```

---

## Training Code Architecture

### `train/grpo_train.py` — Main training script
- Loads Qwen2.5-7B-Instruct, applies LoRA (rank 16), builds dataset, runs GRPOTrainer
- Two modes: `--mode sanity` (100 steps on trial_goods) and `--mode train` (full run)
- Config loaded from `train/configs/qwen25_7b.yaml`
- GRPOTrainer from TRL handles: sampling G completions, computing advantages, PPO-style updates

### `train/reward_functions.py` — Reward computation
- `compute_reward(response, perspective, delta)` → +|δ̃| or -|δ̃|
- `RewardFunction` class wraps this for TRL compatibility
- Unparseable responses (not "Yes"/"No") get -|δ̃| penalty

### `train/prompt_builder.py` — Dataset construction
- Reads `everyday_goods_full.json` (master goods reference) + goods file + delta file
- `build_training_examples()` → paired (X, Y) perspective examples with δ̃
- `build_grpo_dataset()` → flattened list for GRPOTrainer (one prompt per row)
- Prompt format matches `eval/run_all_models.py` exactly

### Data flow
```
remaining_goods.json → prompt_builder.py → paired prompts with δ̃
                                              ↓
everyday_goods_full.json ─────────────────→ item names, attributes, values
                                              ↓
delta_consensus_v3.json ──────────────────→ reward signal per case
                                              ↓
                                        GRPOTrainer
                                        (G=16 samples per prompt)
                                              ↓
                                        reward_functions.py scores each sample
                                              ↓
                                        advantage normalization → gradient update
                                              ↓
                                        LoRA checkpoint saved
```

### Case ID offsets (verified from baseline/Qwen-7B/completed_index.json)
- `trial_goods.json`: 60 cases, offset = 0 → IDs 1–60
- `test_goods.json`: 9,890 cases, offset = 60 → IDs 61–9,950
- `remaining_goods.json`: 49,450 cases, offset = 9,950 → IDs 9,951–59,400
- Delta file keys are global case IDs (1–59,400) as strings

---

## Critical Things to Know

1. **Don't touch `core_exp_refactored.py` without care** — it's the shared econ estimation engine. Bugs propagate into both my results and my coauthor's.
2. **The `run_all_models.py` has heavy checkpoint logic** — don't casually delete `loss_aversion_X.json` / `loss_aversion_Y.json` or the `completed_index.json`.
3. **Prefer LoRA over full fine-tuning** — memory budget is a single A100 40GB.
4. **The reward function uses consensus δ̃ from 6 frontier models** (`data/deltas/delta_consensus_v3.json`). Do NOT use Qwen-7B's own δ̃ — its utility estimates are unreliable due to 99% No rate.
5. **TRL's `GRPOTrainer` is the training library.** Check TRL docs for the exact API — it evolves fast.
6. **Dynamic sampling is critical** — 99% of outputs are "No", so most G=16 batches may be all-identical → zero advantage → zero gradient. If training produces zero gradients, implement DAPO-style filtering: discard batches where all G outputs match. See KNOWN_ISSUES.md item #2.
7. **Temperature must be ≥ 1.0** — temp < 1 makes the 99% No distribution even more peaked. Config uses 1.5.
8. **T=1 for NLS estimation** — T=0 causes zero Jacobian → NLS can't converge → returns degenerate λ̂=0 with zero SEs. Always use T=1 for Style A models with logprobs.
9. **The prompts in train/ must match eval/ exactly** — `prompt_builder.py` replicates `generate_prompt()` from `run_all_models.py`. If you change one, change both.
10. **On NSCC, always `module purge` first** — stale modules cause conflicts. Load `pytorch/2.10.0-py3-cu12.6` fresh.
11. **`everyday_goods_full.json` must be a real file on NSCC** — it's a symlink locally pointing to `../Loss_Aversion/`. Copy the actual file when deploying to NSCC.

---

## Common Tasks

### Run GRPO training
```bash
# On NSCC:
cd $HOME/scratch/lambda-zero
qsub train/submit_sanity.pbs    # sanity check first
qsub train/submit_train.pbs     # then full training

# Or locally for debugging:
python train/grpo_train.py --config train/configs/qwen25_7b.yaml --mode sanity --max_steps 10
```

### Evaluate after training
```bash
# Re-run experiment with fine-tuned model on test_goods.json
python eval/run_all_models.py Qwen-7B-GRPO test_goods.json baseline
python eval/estimate_qwen_grpo.py
# Compare λ̂_after vs λ̂_before = 11.75
```

### Run baseline for a new model
```bash
python eval/run_all_models.py Qwen-7B trial_goods.json     baseline
python eval/run_all_models.py Qwen-7B test_goods.json      baseline
python eval/run_all_models.py Qwen-7B remaining_goods.json  baseline
```

### Estimate λ̂
```bash
python eval/estimate_qwen_base.py
# Reads loss_aversion_X.json + loss_aversion_Y.json
# Writes results to baseline/Qwen-7B/Model_1/
```

---

## Baseline Results — Qwen2.5-7B-Instruct (λ̂_before)

Estimated April 14, 2026. Treatment: baseline. Estimator: Model A (NLS), T=1.

| Parameter | Estimate | Std. Error |
|---|---|---|
| λ (loss aversion) | **11.75** | 1.22 |
| η (status-quo bias) | 1.52 | 0.09 |

**Raw choice data** (9,950 cases):
- X-perspective: Yes = 48, No = 9,902 (keeps endowed good 99.5% of the time)
- Y-perspective: Yes = 45, No = 9,905 (same pattern)
- Cross-tab: 99.07% are (No, No) — model almost never trades regardless of perspective

**Interpretation:** Qwen-7B is massively loss-averse. λ̂ = 11.75 is roughly 5× the human benchmark (Kahneman & Tversky: λ ≈ 2–2.5) and expected to be well above frontier models from Phase 0. The "No Bias" counterfactual shows a ~53/47 split, confirming underlying utilities are balanced — the refusal to trade is driven by loss aversion and status-quo bias, not by the goods themselves. This provides a strong training signal for GRPO.

---

## Papers the Project Builds On

- **DeepSeekMath (Shao et al., 2024)** — original GRPO formulation
- **DeepSeek-R1 (2025)** — GRPO at scale, reward design patterns
- **Dr. GRPO (Sea AI Lab, 2025)** — length-bias correction, relevant because our outputs are 1 token
- **DAPO (ByteDance, 2025)** — decoupled clip + dynamic sampling (must adopt dynamic sampling for this project)
- **Kahneman & Tversky (1979)** — prospect theory, defines λ
- **Tversky & Kahneman (1992)** — human benchmark λ = 2.25
- **List (2004)** — endowment effect diminishes with market experience (motivates "RL = experience" framing)

---

## Publication Timeline

- **NeurIPS 2026 workshop** (~August 2026 deadline): Preliminary results, 4-6 pages, non-archival
- **ICML 2027 main track** (~January 2027 deadline): Full paper with ablations
- Workshop submission does NOT block ICML submission

---

*Last updated: April 27, 2026*
