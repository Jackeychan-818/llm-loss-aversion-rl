# CLAUDE.md

This file gives Claude Code context about this project so it can help effectively without re-reading everything from scratch.

> **Current paper-direction override (July 15, 2026):** The primary model is
> Qwen-own-delta GRPO checkpoint 8,000; frontier-consensus delta is the
> reward-source ablation. Its `test_goods` estimate (`lambda = 0.111`,
> `eta = 0.504`) is validation-only because this set informed reward
> construction and checkpoint selection. Read `PAPER_READINESS.md` before
> relying on older “main run,” “ablation,” or paper-readiness descriptions
> below.
>
> **Canonical paper source:** `training_overview.tex` (see `PAPER_SOURCES.md`).
> PDFs/slides are derived — regenerate, never hand-edit.

---

## Project Overview

**lambda-zero** (`loss_aversion_rl`) — a reinforcement learning project that fine-tunes Qwen2.5-7B-Instruct using **GRPO (Group Relative Policy Optimization)** to reduce loss aversion below frontier models.

This project builds on an existing behavioral economics study (separate repo, `loss_aversion/`) that measures **λ** (loss aversion coefficient) across 9 LLMs using an endowment-effect paradigm. That study established that frontier LLMs show measurable loss aversion. This project asks whether targeted RL can correct it.

### Thesis in one sentence
A 7B model, fine-tuned with GRPO using a rationality-based reward, can achieve lower loss aversion (λ → 0) than frontier models 10–100× its size.

---

## Current Status (July 21, 2026)

**The exploratory primary run and two confirmatory training runs are complete;
confirmatory evaluation remains open.**

### Completed
- Qwen-own-delta checkpoint 8,000 selected as the primary candidate.
- Primary ID validation: λ̂ = **0.111**, η̂ = **0.504**.
- Reported primary OOD result: λ̂ = **0.226**, η̂ = **0.790**,
  consistency = **49.46%**. Derived results + manifests committed; **raw X/Y
  predictions are NOT in git** (see `PAPER_READINESS.md` #6).
- Consensus reward-source ablation: λ̂ = **0.177**, η̂ = **-0.048** ID.
- Historical treatment comparison for the consensus run:
  - baseline: λ̂ = 0.177, η̂ = −0.048
  - debias:   λ̂ = 0.205, η̂ = +0.090
  - forced:   λ̂ = 0.173, η̂ = −0.379

### Active
- **Replication CONFIRMED: 2/2 seeds PASS.** ID selection (seed 1 → step 2,000,
  seed 2 → step 6,000) + one OOD-50 + one GSM8K per selected checkpoint are done;
  mechanical verdict passes all gates (seed 1 λ_OOD 0.259, seed 2 λ_OOD 0.064;
  GSM8K CI lower −1.14pp / −1.44pp). See `results/seed_replication_report.json`.
- Non-gating GRPO optimization and structural alpha/beta/utility diagnostics;
  step 600 is exploratory only and cannot enter frozen selection.
- Raw-prediction archival (derived results + manifests ARE committed; the raw
  X/Y predictions, ~167 MB, are not — numbers are traceable but not
  independently re-derivable from the repo alone).
- Matched SFT/sign-only baselines, robust inference, IFEval.

The complete blocker ordering and claim restrictions are in
`PAPER_READINESS.md`.

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
| 1. Small-model baseline | ✅ Matched | **λ̂ = 7.637** (SE 0.627), η̂ = 1.007 — exact local base, 9,890 rows, teacher-forced. Historical 11.75 (Together Turbo) superseded: inflated ~54% by pipeline mismatch |
| 2. Qwen-own reward design | ✅ Validated | δ̃ vs frozen ownership-free anchor: 71.1% test / 70.8% train sign agreement (85.5% at \|δ̃\|>1.0); α 91.9% / β 100% significant. `eval/validate_qwen_delta_anchor.py` |
| 3. Qwen-own GRPO training | 🟡 Replication training complete | Seeds 1 and 2 reached 30k with complete frozen grids; evaluations pending |
| 4. Primary evaluation | 🟡 Partially complete | ID validation + OOD table exist; derived results & manifests committed, raw X/Y predictions not (PAPER_READINESS #6) |
| 5. Reward-source ablation | ✅ Consensus run complete | Consensus ID result committed |
| 6. Baselines/robustness | ⏳ | SFT, sign-only GRPO, robust inference, capability retention |
| 7. Paper writeup | ⏳ | Full paper after Priority-0 items in `PAPER_READINESS.md` |

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
│   ├── run_qwen_local.py             # ✅ vLLM-free local evaluator (teacher-forced log-probs)
│   ├── estimate_qwen_base.py         # λ̂_before
│   ├── estimate_qwen_grpo.py         # λ̂_after (baseline treatment) → 0.177
│   ├── estimate_qwen_grpo_debias.py  # λ̂_after (debias treatment)   → 0.205
│   └── estimate_qwen_grpo_forced.py  # λ̂_after (forced treatment)   → 0.173
│
├── train/                            # RL training code
│   ├── grpo_train.py                 # ✅ Main GRPO training loop using TRL's GRPOTrainer
│   ├── reward_functions.py           # ✅ Utility-weighted reward using frontier δ̃
│   ├── prompt_builder.py             # ✅ Generates X/Y perspective pairs from goods data
│   ├── configs/
│   │   ├── qwen25_7b.yaml            # ✅ Frontier-consensus reward ablation
│   │   └── qwen25_7b_qwen_delta.yaml # ✅ Primary Qwen-own-delta reward
│   ├── setup_nscc.sh                 # One-time NSCC environment setup
│   ├── submit_sanity.pbs             # PBS job: sanity check (gdev, 1.5h)
│   ├── submit_train.pbs              # PBS job: full training (g1, 24h)
│   ├── submit_train_glong.pbs        # PBS job: consensus reference run
│   ├── submit_train_glong_qwen_delta.pbs  # PBS job: primary Qwen-own-delta run
│   ├── submit_eval_full.pbs          # PBS job: full held-out eval (g1)
│   ├── submit_eval_smoke.pbs         # PBS job: 20-case smoke test (gdev)
│   ├── submit_eval_treatments.pbs    # PBS job: debias + forced eval (g1)
│   └── NSCC_GUIDE.md                 # NSCC ASPIRE 2A reference guide
│
├── data/deltas/
│   ├── delta_consensus_v3.json       # Frontier consensus δ̃ — reward-source ablation
│   ├── delta_qwen_base.json          # Primary Qwen-own pseudo-utility reward
│   └── build_delta_qwen_base.py      # Script that generated delta_qwen_base.json
│
├── baseline/Qwen-7B/                 # λ̂_before results
├── baseline/Qwen-7B-GRPO/            # λ̂_after results (baseline treatment)
├── debias/Qwen-7B-GRPO/              # λ̂_after results (debias treatment)
├── forced/Qwen-7B-GRPO/              # λ̂_after results (forced treatment)
├── checkpoints/grpo_20260427_1724/   # Consensus-delta reference checkpoints
├── checkpoints/grpo_qwen_delta/      # Primary Qwen-own-delta checkpoints
│  (notebooks/compare_lambdas.ipynb: referenced historically, does NOT exist)
│
├── everyday_goods_full.json          # Master goods reference (item names, attributes, values)
├── CLAUDE.md                         # ← this file
├── PAPER_READINESS.md                # Authoritative full-paper blockers
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

**Why Qwen-7B's own δ̃ IS usable (corrected July 15, 2026).** An earlier version of
this file claimed Qwen's own δ̃ was unreliable — "NLS can't learn meaningful α/β →
δ̃ reflects reference category geometry, not real preferences." **That claim is
empirically false and has been retracted.** It also contradicted the current paper
direction, in which Qwen-own δ̃ is the PRIMARY reward (see `PAPER_READINESS.md`).

Measured on `baseline/Qwen-7B/Model_1/` and `data/deltas/delta_qwen_base.json`:

| check | value |
|---|---|
| α (item FE) identified | 99/99 non-zero SE, median \|t\| = 9.6, **91.9% significant** |
| β (attr FE) identified | 8/8 non-zero SE, median \|t\| = 27.5, **100% significant** |
| δ̃ spread | sd = 0.94, range [−4.06, +3.91], only **1.6%** near-zero, 46.9% positive |
| δ̃ sign vs frozen ownership-free anchor | **71.1%** (test) / **70.8%** (remaining), chance = 50% |
| … restricted to \|δ̃\| > 1.0 | **85.5%** / 85.3% |

**Why the 99% No rate does NOT destroy identification:** utilities are estimated
from **logprobs at link scale T=1**, not binary choices. P(Yes)=0.08 vs 0.28 both
argmax to "No" but carry rich continuous information about relative utility. Under
a T=0 / binary encoding the 99% No rate genuinely *would* flatten δ̃ into noise —
so the old objection is only valid for the estimator we don't use (see note #8).

**Reward calibration bonus:** agreement with the independent ownership-free anchor
rises with |δ̃| (85% at |δ̃|>1.0 vs 64% at |δ̃|≤0.5), and the reward weights by |δ̃| —
so it is loudest exactly where the signal is most trustworthy.

Reproduce with: `python eval/validate_qwen_delta_anchor.py`
(→ `results/qwen_delta_anchor_validation.json`).

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
| batch_size | 1 | Per-device, with gradient_accumulation_steps=16 |

---

## Training Infrastructure — NSCC ASPIRE 2A

### System details
- **Host:** `aspire2a.nus.edu.sg` (SSH login)
- **GPU:** NVIDIA A100-40GB SXM (4 per node)
- **Scheduler:** PBS Pro (NOT SLURM — use `qsub`, `qstat`, `qdel`)
- **Project ID:** `personal-jackeyc0`
- **Scratch:** `$HOME/scratch/lambda-zero/`
- **PyTorch module:** `pytorch/2.6.0-py3-cu11.8` (Python 3.x, CUDA 11.8 — gdev/g1 nodes have CUDA 12.4 driver, cu12.6 is too new)
- **Venv:** `$HOME/scratch/lambda-zero/venv/` (TRL, PEFT, transformers; vLLM disabled on NSCC after compatibility failures)
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
module purge && module load pytorch/2.6.0-py3-cu11.8
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
4. **The PRIMARY reward is Qwen's own δ̃** (`data/deltas/delta_qwen_base.json`);
   frontier-consensus δ̃ (`data/deltas/delta_consensus_v3.json`) is the
   **reward-source ablation**. The old instruction here ("Do NOT use Qwen-7B's own
   δ̃ — unreliable due to 99% No rate") is **retracted**: α/β are well identified
   (91.9% / 100% significant) and δ̃ agrees with Qwen's frozen ownership-free
   preferences 71% of the time (85% when |δ̃|>1.0). See §Reward function and
   `PAPER_READINESS.md`. Do not claim Qwen-own δ̃ dominates consensus on every
   axis — it has lower λ̂ but consensus has lower η̂ and higher OOD consistency.
5. **TRL's `GRPOTrainer` is the training library.** Check TRL docs for the exact API — it evolves fast.
6. **Dynamic sampling is critical and NOT implemented** — 99% of base outputs are "No", so most G=16 groups are all-identical and become **zero task-reward advantage groups** (~80% of steps). Say it that way; never "skipped batches" or "zero-update batches". Three facts to keep straight (details in KNOWN_ISSUES.md #4):
   - `make_reward_fn()` returns all-zero task rewards, but **nothing is skipped** — stock TRL `GRPOTrainer` still generates, backprops, and steps.
   - `scale_rewards: "none"` still **mean-centres**: `advantages = rewards − group mean`. Advantages are never raw rewards. (It is also not a NaN guard — TRL divides by `std + 1e-4`.)
   - With `beta: 0.04`, zero task advantage still leaves a **KL-only update** toward the reference policy.

   Because zeroed groups are logged as reward 0.0 rather than their true ±|δ̃|, treat training reward as a diagnostic and select checkpoints by held-out structural λ̂. See KNOWN_ISSUES.md items #3-4.
7. **Temperature must be ≥ 1.0** — temp < 1 makes the 99% No distribution even more peaked. Config uses 1.5.
8. **T is the structural LINK SCALE — not a sampling temperature.** Two different
   things are easily conflated; keep them apart:
   - **LLM decoding**: deterministic greedy (`do_sample=False` in `run_qwen_local.py`);
     the Yes/No choice is read from teacher-forced logprobs. Nothing is sampled.
   - **Estimator T**: the scale in the structural link `P(Yes) = 1/(1+exp(-z/T))`
     (`_calculate_y_hat` in `core_exp_refactored.py`). A property of the econometric
     model only.

   **Model A (NLS) requires T > 0 — use T=1** for Style A models with logprobs. At T=0
   the link becomes the step `1(z>0)` → zero Jacobian → NLS can't converge → degenerate
   λ̂=0 with zero SEs. The T=0 / binary maximum-score regime belongs to **Model B (SMS,
   `--robust_model 2`)**. `estimate_qwen_checkpoint.py` now hard-errors on Model A + T≤0.

   **CLI**: the option is `--link-scale`; `--temperature` is a DEPRECATED alias kept so
   existing NSCC scripts keep working (the old name wrongly implied sampling).

   **Approved paper wording:**
   > "Evaluation used deterministic greedy decoding (`do_sample=False`), equivalent to
   > the zero-temperature decoding limit. Model A used structural link scale T=1."

   (Decoding temperature is *inactive* under greedy decoding, not numerically set to 0 —
   do not write "temperature=0 was used" or "Model A NLS at T=0".)
9. **The prompts in train/ must match eval/ exactly** — `prompt_builder.py` replicates `generate_prompt()` from `run_all_models.py`. If you change one, change both.
10. **On NSCC, always `module purge` first** — stale modules cause conflicts. Load `pytorch/2.6.0-py3-cu11.8` fresh (cu12.6 is too new for the gdev/g1 driver).
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
# Re-run experiment with fine-tuned model on test_goods.json, no vLLM required
python eval/run_qwen_local.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path checkpoints/grpo_20260427_1724/final \
  --model_name Qwen-7B-GRPO \
  --data_file data/test_goods.json \
  --treatment baseline \
  --batch_size 8 \
  --yes

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

*Last updated: July 9, 2026*
