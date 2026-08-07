# Causal-Baseline Protocol — matched SFT and sign-only GRPO

*Frozen 2026-07-25, before any baseline training run. Implements Priority 1 of
`RESEARCH_ROADMAP.md`. This document is the pre-registration; the numbers and
rules below are fixed in advance and applied mechanically.*

**Freeze-time statement:** no baseline result existed when this protocol was
frozen. This file freezes the *design only*; the append-only execution record at
the end documents later runs without rewriting the rules.

## Purpose (two different questions)

| Baseline | Question it answers |
|---|---|
| **Matched SFT** (Qwen-own-delta targets) | Is **RL necessary**, or does ordinary supervised learning on the same rational-choice targets already remove the endowment effect? |
| **Sign-only GRPO** (reward ±1) | Does the reward **magnitude** `|δ̃|` contribute anything beyond the **sign** of the preferred good? |

These are distinct from the completed **frontier-consensus-delta** run, which is
a reward-**source** ablation (whose δ̃, not what shape of reward). All three are
separate experiments and must not be conflated.

## Shared, matched configuration

Held identical to the confirmatory Qwen-own-delta GRPO run
(`train/configs/qwen25_7b_qwen_delta.yaml`):

- **Base model:** `models/Qwen2.5-7B-Instruct` (exact local weights).
- **Training data:** `data/remaining_goods.json` with frozen Qwen-own δ̃
  (`data/deltas/delta_qwen_base.json`, sha256 `569bf72e…`). **No** `test_goods`,
  frozen-unused, or OOD-50 rows ever enter training.
- **Prompt construction:** the same `train/prompt_builder.py`
  (`generate_prompt`, baseline treatment), same X/Y perspective pairing.
- **LoRA:** r=16, α=32, dropout=0.05, target modules
  {q,k,v,o,gate,up,down}_proj, `bias=none`, `CAUSAL_LM`.
- **Optimizer/schedule:** AdamW, lr 1e-6, cosine, warmup_ratio 0.05,
  max_grad_norm 0.1, bf16.
- **Decoding at eval:** deterministic greedy (`do_sample=False`), structural
  link scale T=1 (unchanged evaluation pipeline, `eval/run_qwen_local.py` +
  Model A NLS).
- **Independent seeds:** 1 and 2 (matching the confirmatory seeds). Seed is
  threaded through the trainer RNG, data order, and (GRPO) rollout sampling.
- **Per-seed output directories** (never shared, never overwriting a confirmatory
  run):
  - SFT: `checkpoints/sft_qwen_delta_seed{N}`
  - sign-only GRPO: `checkpoints/grpo_qwen_delta_sign_seed{N}`

## Training data and target construction

Dataset = 49,450 goods-pair configurations × 2 perspectives = **98,900
examples**. Every case has a non-zero δ̃ (0 dropped). SFT targets are the
**rational (frozen-preferred) answer**, identical to the GRPO reward's notion of
"rational":

- **X-perspective** (endowed X, offered Y): target **No** if δ̃ > 0, else **Yes**.
- **Y-perspective** (endowed Y, offered X): target **Yes** if δ̃ > 0, else **No**.

Both perspectives encode the **same preferred good** (δ̃>0 ⇒ X; δ̃<0 ⇒ Y). Verified
label distribution: exactly **49,450 "Yes" and 49,450 "No"** (balanced, because
each case contributes one keep and one trade). The single source of truth is
`rational_choice(perspective, delta)` in `train/reward_functions.py`, reused by
both the SFT target builder and the GRPO reward.

**SFT loss is completion-only:** only the assistant answer tokens (the Yes/No
turn) contribute to the loss; prompt tokens are masked with label −100. This
mirrors GRPO, whose policy gradient applies only to generated completion tokens.
Rationale: with a one-token target, training on prompt tokens would swamp the
signal. (If a future stack cannot mask, that is documented at run time in the
manifest and the run is labelled accordingly.)

## Data-exposure calculation (the matching)

**Do not equate "30,000 SFT steps" with "30,000 GRPO steps."** They are only
comparable once translated into a common exposure quantity.

### GRPO, one optimizer step (from `qwen25_7b_qwen_delta.yaml`)

- `num_generations` G = 16
- `per_device_train_batch_size` = 1, `gradient_accumulation_steps` = 16
- ⇒ generation batch = 1×16 = 16 completions = **1 unique prompt / step**
  (16 completions of that one prompt form the GRPO group)
- ⇒ **1 optimizer update / step**, **16 completions generated / step**

At the frozen endpoint **MAX_STEPS = 30,000**:

| quantity | GRPO @ 30k |
|---|---|
| unique prompts seen | **30,000** (of 98,900 ⇒ **0.303 epoch**) |
| optimizer updates | **30,000** |
| completions generated | 30,000 × 16 = **480,000** |
| completion tokens trained (loss) | ≈ 480,000 × (1–2), ≤ 4/comp |
| forward tokens processed | ≈ 480,000 × (≈150 prompt + ≤4) ≈ **73 M** |

(The generation-batch / step identities are TRL-config specific to this file:
`generation_batch_size = per_device_bs × grad_accum = 16 = G`. If a TRL upgrade
changes `steps_per_generation`, re-derive before trusting the step↔prompt map.)

### SFT, matched (primary rule = **unique prompt/data exposure**)

Chosen SFT budget: **per_device_train_batch_size = 1, gradient_accumulation
_steps = 1** ⇒ **1 unique prompt = 1 optimizer update / step**. Then setting
SFT `MAX_STEPS = 30,000` matches GRPO on **both** primary quantities at once:

| quantity | SFT @ 30k (batch 1) | matches GRPO? |
|---|---|---|
| unique prompts seen | **30,000** (0.303 epoch) | ✅ primary match |
| optimizer updates | **30,000** | ✅ (secondary, also equal here) |
| completions generated | **0** (supervised) | ✗ by construction (RL-only) |
| completion tokens trained | ≈ 30,000 × (1–2) | ✗ ≈16× fewer (reported) |

**Primary matching rule:** equal **unique prompt/data exposure** (30,000
examples, 0.303 epoch). Optimizer updates also happen to match under batch 1.
Completions-generated and forward-token counts are **fundamentally different**
between RL and SFT and are **reported, not matched** — that asymmetry is the
whole point of the "is RL necessary?" question. **Every budget is configurable**
(`--max_steps`, batch, grad-accum, `MAX_STEPS` env); the SFT baseline is only
called "matched" on the unique-prompt axis, which the calculation above
supports.

Sign-only GRPO shares GRPO's mechanics exactly, so its exposure table is
identical to the GRPO column above at the same MAX_STEPS.

## Checkpoint grid, eligibility, and selection

- **Grid (frozen):** every 2,000 up to 30,000 (2k…30k, 15 checkpoints) — the
  same as the confirmatory grid. This is valid because, under batch 1 (SFT) and
  the GRPO config, **1 step = 1 unique prompt** for all three methods, so a
  step-indexed grid *is* a data-exposure-indexed grid. `save_steps = 2000`.
- **Selection rule (frozen):** the existing `eval/select_checkpoint.py`,
  unchanged — eligibility `consistency ≥ 0.50` + finite non-zero SEs, primary
  `min d = √(λ̂²+η̂²)`, tie-breaks as in `CHECKPOINT_PROTOCOL.md`. It is applied
  per baseline with a baseline-specific `--pattern`; it does **not** touch the
  frozen seed1→2000 / seed2→6000 confirmatory selections (different eval dirs).
  **No new or modified selector.** If a future need arises it will be a separate
  file.
- Selection uses **`test_goods.json` as VALIDATION only.**
- **Selection-manifest provenance (required for the baselines).** When the frozen
  selector is eventually run on a baseline full grid, it MUST be invoked with
  `--provenance_note` recording that **the previous final suites (OOD-50 and the
  frozen-unused configuration test) are already opened**, so the baseline
  selection is **validation-based** and any cross-method comparison
  (SFT vs GRPO, sign vs magnitude) is **post-hoc** unless a *new* untouched suite
  is frozen in advance. Suggested text:
  `"OOD-50 and frozen-unused final suites already opened (before these baselines);
  selection is validation-only on test_goods; SFT-vs-GRPO / sign-vs-magnitude
  comparisons are post-hoc until a new suite is frozen."`
  (The pilot does **not** run the selector at all — its 3-point grid is
  incomplete and the selector hard-fails on an incomplete grid.)

## Primary and secondary outcomes

- **Primary:** structural λ̂ (and η̂ reported jointly) on the selected checkpoint,
  vs the matched local base (λ̂ = 7.637). Report d = √(λ̂²+η̂²).
- **Secondary:** consistency, keep-both, trade-both, W (pseudo-utility
  alignment), and capability retention (GSM8K/IFEval per the roadmap).
- Report **every seed separately**, never the best alone.

## Evaluation datasets and their roles

| dataset | role for these baselines |
|---|---|
| `test_goods.json` | **VALIDATION** — checkpoint selection + reported λ̂ (validation, not untouched final test). |
| OOD-50, frozen-unused, framing | **CLOSED.** Already opened; may **not** be used to tune, select, or compare these baselines. |
| a NEW frozen suite | **Required for a confirmatory method-comparison claim** (see below). |

**Confirmatory-comparison caveat.** Because OOD-50 and the frozen-unused set are
already opened, any *confirmatory* claim that (SFT vs GRPO) or (sign vs
magnitude) differ would require a **new, untouched evaluation frozen in
advance**. Until then, baseline results on `test_goods` are **validation /
exploratory** and any cross-method comparison is labelled **post-hoc**.

## Stopping, failure, and restart

- **Stopping:** fixed endpoint MAX_STEPS (default 30,000); no early stopping on
  any metric. Pilot uses a smaller MAX_STEPS (see below).
- **Restart:** PBS scripts are resumable (`--resume_from_checkpoint auto`, latest
  checkpoint under the per-seed dir). Resume can **only** re-enter the same
  seed's own directory; treatment/seed are encoded in the directory name, so a
  resume can never overwrite another treatment or seed (enforced + unit-tested).
- On crash: resubmit the same job; it continues from the last saved checkpoint.

## GPU-hour estimate (rough, single A100-40GB, HF generate, no vLLM)

Anchored to the observed GRPO throughput (~6.1 s/step in the current config).

Two very different per-step costs: **GRPO/sign-only ≈ 6.1 s/step** (16 HF
generations + update, from the confirmatory run) vs **SFT ≈ 0.2–0.4 s/step**
(one supervised forward/backward, no generation — an estimate the smoke test
**measures and replaces**).

| run | steps | est. wall |
|---|---:|---:|
| SFT smoke | 10 | ~3–5 min (model load dominates) |
| sign-only GRPO smoke | 10 | ~3–5 min (model load dominates) |
| **Pilot** SFT seed 1 → 6,000 | 6,000 | ≈ 0.3 s/step ⇒ **~0.5–1 h** |
| **Pilot** sign-only GRPO seed 1 → 6,000 | 6,000 | ≈ 6.1 s/step ⇒ **~10 h** |
| Full SFT, 2 seeds × 30,000 | 60,000 | ≈ **2–3.5 h/seed ⇒ ~4–7 h** total |
| Full sign-only GRPO, per seed × 30,000 | 30,000 | ≈ 6.1 s/step ⇒ **~51 h/seed — fits ONE 72 h job** (no resume needed); 2 seeds = 2 jobs |
| Baseline checkpoint eval (per ckpt, test_goods 9,890) | — | ~0.5–1 h each (`submit_eval_baseline_ckpt.pbs`) |

**Full-experiment evaluation cost:** the complete comparison evaluates the full
grid for both methods and both seeds — 2 methods × 2 seeds × 15 checkpoints =
**60 checkpoint evaluations ≈ 30–60 additional GPU-hours** (≈ 0.5–1 h each) on
top of training. (The pilot evaluates only 2 methods × 1 seed × 3 checkpoints =
6 evals.)

The SFT per-step figure is the least certain; the smoke test calibrates it before
committing to the full runs. Sign-only inherits the confirmatory GRPO throughput.
*Measured in the smoke tests: SFT ≈ 0.28 s/step (full 30k ≈ 1.4 h/seed);
sign-only ≈ 5.57 s/step (full 30k ≈ 47 h/seed, one 72 h job).*

## Confirmatory vs post-hoc (status of results)

- The **confirmatory** Qwen-own-delta GRPO replication (2/2 seeds) is unchanged
  and untouched by this work.
- **All baseline results produced under this protocol are, at first,
  validation/exploratory.** A confirmatory method-comparison requires the new
  frozen suite above and independent replication seeds. Full paper rewriting
  remains deferred.

## Pilot (first stage, low-cost)

1 SFT seed (seed 1) + 1 sign-only GRPO seed (seed 1), MAX_STEPS = 6,000, a small
pre-declared checkpoint subset {2000, 4000, 6000}. **No OOD-50 or frozen-unused
evaluation.** The pilot calibrates throughput and sanity, and does **not** enter
any frozen selection or confirmatory claim.

## Append-only execution record

### July 27, 2026 — seed-1 pilot completed

Commit `c1ab3da` records the derived pilot summaries:

- `results/causal_baseline_pilot/pilot_core.json`
- `results/causal_baseline_pilot/pilot_table.md`

The pilot evaluated the predeclared steps 2,000, 4,000, and 6,000 for one SFT
seed and one sign-only GRPO seed on `test_goods` validation. It did not run the
frozen checkpoint selector and did not open OOD-50 or the frozen-unused suite.
Both methods sharply reduced lambda from the matched base. SFT reached
`d = sqrt(lambda^2 + eta^2)` of 0.060 and 0.052 at steps 4k and 6k;
sign-only GRPO was more variable across the three checkpoints. These are
pilot observations, not a method winner.

Observed 6k training times on one A100 were approximately 17.5 minutes for SFT
and 8.7 hours for sign-only GRPO.

**Provenance limitation:** the commit contains derived summaries only. It does
not yet provide raw prediction files, structural-fit CSVs, checkpoint/data/code
hash manifests, exact commands, or a deterministic generator for all table
cells. W, Jacobian condition numbers, and runtimes appear in the Markdown table
but not in `pilot_core.json`. Close `PILOT-001` in `KNOWN_ISSUES.md` before
citing the pilot beyond exploratory project status.

**Confirmatory status:** the two-seed, 30k runs and a newly frozen untouched
method-comparison suite remain pending. The already-opened OOD-50 and
frozen-unused suites cannot be used to select or revise the baseline methods.

### July 30, 2026 — full SFT training completed on NSCC

Both predeclared matched SFT seeds reached the fixed 30,000-step endpoint.
Each run contains all 15 scheduled checkpoints from step 2,000 through step
30,000. The earlier seed-1 pilot remains preserved separately, and provenance
manifests are present on NSCC.

Measured training runtime was 5,422 seconds for seed 1 and 4,668 seconds for
seed 2 — a mean of approximately 5,045 seconds (1.40 hours, 0.168 seconds per
step). An earlier version of this section quoted 5,422 seconds "per seed",
applying seed 1's runtime to both and understating SFT throughput; corrected on
August 5, 2026 from the per-run Trainer end-of-training summaries recorded in
`results/training_dynamics/sft/sft_training_manifest.json`. The two full runs
charged approximately 317 SU in total, reducing the reported allocation balance
from 3,629 SU to approximately 3,312 SU.

At the July 30 training milestone, no full-run checkpoint had yet been
evaluated on `test_goods`, and the frozen selector had not been run. Therefore,
at that time:

- this entry records training completion, not SFT behavioral performance;
- the seed-1 pilot remains the only evaluated SFT evidence and is exploratory;
- no claim that SFT beats or matches GRPO is licensed;
- the new untouched method-comparison suite and protocol must be frozen before
  the complete SFT trajectory is opened; and
- the complete 15-point grid must be retained because the frozen selector
  rejects incomplete grids.

The next authorized phase was the CPU-only work package in
`NSCC_CPU_WORK_PROMPT.md`. The later evaluation outcome is recorded below.

### July 30, 2026 — CPU-only paper-gate package completed (branch codex/cpu-paper-gates)

The six CPU-only tasks were built, tested, and committed without submitting any
GPU job, running any inference, evaluating any checkpoint, or opening any frozen
suite. This entry records infrastructure/analysis freezes only — no full-SFT
behavioral result and no method winner is created.

- **New untouched method-comparison suite frozen** (`data/method_comparison/`):
  4 previously-unused attribute codes per pair (19,780 cases; 39,560 X/Y
  prompts) drawn from the 59 codes never used by test/train/frozen-unused; zero
  (pair,code) overlap asserted; frozen semantic-counterbalancing subset (160
  cases, 48 forms); 31 integrity tests pass. UNEVALUATED.
- **Method-comparison protocol frozen** (`METHOD_COMPARISON_PROTOCOL.md`) before
  any full-grid SFT checkpoint was opened. Uses the unchanged selector; all
  per-seed selections happen on `test_goods` validation before the untouched
  suite is opened once.
- **Scale-matched reward control specified** (ABLATION-001): `scale_matched`
  weighting (±c, c=0.685029 mean_abs) added without changing the default
  magnitude path; ENV-001 hard-fail on dropped algorithm-defining keys; 29 tests.
  Run-pending (GPU).
- **Robustness inference added** (`eval/robust_inference.py`,
  `estimator_recovery.py`): strict ID join (PAIR-001), pair-clustered bootstrap
  and recovery for (λ,η); CPU-only PBS jobs. The frozen headline estimator is
  untouched; headline INFER-001 closure still needs per-model utilities fed into
  the bootstrap.
- **Full-behavior aggregation** (`results/full_behavior/`) and **GRPO efficiency
  analysis** (`results/grpo_efficiency/`) from committed results only,
  deterministic (`--check`), with an explicit "small λ is not success" guard.

**Provenance finding (open):** the full-SFT and pilot `sft_dataset_manifest.json`
record `git_commit: "unknown"` (the `git rev-parse` subprocess failed on the
compute node). Source data/delta/goods SHAs match this protocol, so the runs are
traceable, but the exact training commit is not captured. Recorded in
`KNOWN_ISSUES.md`; cannot be reconstructed retroactively with certainty.

### August 7, 2026 — full SFT validation grid and selection completed

All 30 predeclared SFT checkpoints (2 seeds × 15 steps) were evaluated on the
9,890-case `test_goods` validation set with the plain baseline prompt and Model
A NLS at T=1. The recorded grid contains zero parse failures. The unchanged
selector chose:

- seed 1 → step 4,000: lambda=-0.034, eta=0.027, d=0.043,
  consistency=0.730;
- seed 2 → step 6,000: lambda=-0.089, eta=-0.143, d=0.168,
  consistency=0.767.

These are validation-selected results and may benefit from selection. They do
not establish SFT-versus-GRPO superiority; the untouched method-comparison
suite remains unopened until the sign-only and scale-matched families are
trained and selected. The historical evaluator wrote no eval-time manifest,
so `results/sft_grid_verification.json` establishes expected-adapter and
evaluation-artifact consistency, not cryptographic adapter-to-prediction
binding. Raw SFT-grid prediction files remain untracked.
