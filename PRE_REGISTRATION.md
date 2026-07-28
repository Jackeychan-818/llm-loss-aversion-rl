# Pre-Registration — Qwen-own-delta Seed Replication

**STATUS: FROZEN — July 15, 2026, before any seed was submitted.**
No seed had been launched and no seed result existed when this was frozen. Do not
change any threshold or bar below after seeing a seed result; if something must
change, record it as an amendment with a date and reason, and report the analysis
as post-hoc.

Confirmatory seeds at freeze time: **none run**. Exploratory `seed=42` results
were known and are explicitly excluded from the confirmatory denominator below.

**Status update (July 21, 2026; does not alter the frozen design):** both new
training runs (`SEED=1,2`) reached the frozen `MAX_STEPS=30000` endpoint and
saved the complete 15-checkpoint selection grid. ID evaluation, mechanical
selection, one-shot OOD evaluation, and per-seed GSM8K remain to be completed.

## Purpose

The completed run (seed=42) shows **one adapter worked**. Replication tests
whether **the method works reliably**. Seeds are NOT required to reproduce
λ̂ = 0.111.

## Run design (frozen)

| item | value |
|---|---|
| Seeds | **2 new (first stage)**: `SEED=1,2`. `seed=42` = exploratory, NOT confirmatory. Confirmatory n=2, below `PAPER_READINESS` #5's "at least three" — a deliberate staged choice; a third seed may follow. |
| Training endpoint | **`MAX_STEPS=30000`** — fixed for every seed, so all seeds expose the same checkpoint grid |
| Checkpoint saves | every 200 steps |
| Selection grid | every **2,000** steps, **2k–30k** (15 checkpoints/seed; ~15 evals x 2 seeds ~= 30 GPU-h) |
| Selection rule | `CHECKPOINT_PROTOCOL.md`, applied mechanically per seed |
| Everything else | identical to the exploratory run (config `qwen25_7b_qwen_delta.yaml`) |

Why a fixed endpoint: without `--max_steps` a run goes to a full epoch or dies at
the 72h wall (the v2 run stopped at 20,600), so seeds would stop at different
steps and expose different grids — not comparable.

## Suite pairing rule (MANDATORY)

**Always compare a model to the base measured on the SAME suite.** Cross-suite
comparison inflates the apparent effect and is not permitted:

| suite | base comparator | never compare against |
|---|---|---|
| ID / validation (`test_goods`) | **λ̂ = 7.637**, η̂ = 1.007 | the OOD base |
| OOD-50 (unseen goods) | **λ̂ = 2.209**, η̂ = 5.358 | the ID base (7.637) |

An OOD result (e.g. λ̂ = 0.226) must be read against **2.209**, not 7.637.

## Success criteria

**Confirmatory denominator: the NEW seeds only.** `seed=42` is **exploratory** and
is NOT counted. Its results are already known and helped design the selection
rule, so including it would be doubly permissive: a "2 of 3" bar counting seed 42
would require only **one of two** new seeds to succeed.

**Bar:**
- **Both new seeds pass → report 2/2 confirmatory**, with seed 42 as *supporting
  exploratory* evidence (clearly labelled as such, never as a third confirmation).
- **The new seeds disagree (1/2) → run a third NEW seed and require 2 of 3 FRESH
  seeds.** Do not fall back on seed 42 to break the tie.
- **Both fail → report that the method did not replicate.**

**Report BOTH**, always:
1. the confirmatory verdict on the new seeds, *and*
2. **every individual seed's** numbers (λ̂ ID and OOD, η̂, consistency,
   keep/trade-both, selected step, KL) — never the best seed alone.

Judged **per seed**. With few seeds no summary statistic is meaningful: report all
values and the range, not a median.

**S1 — Eligible checkpoint exists.** Required in **every new seed**. A seed with no
checkpoint passing `CHECKPOINT_PROTOCOL` eligibility (consistency ≥ 0.50, finite
non-zero SEs) is a **method failure**, not a discarded seed.

**S2 — Loss-aversion reduction (primary). FROZEN.**

Judged on the **OOD-50 suite**, at the checkpoint chosen by the frozen selection
rule:

> **|λ̂_OOD| ≤ 0.5**

**Why OOD and not ID:** `test_goods` is used for checkpoint selection *and*
informed the Qwen-own reward construction (PAPER_READINESS #3/#4), so an ID
threshold would be graded on contaminated data. OOD-50 is the untouched
comparator. Its base is **λ̂ = 2.209**, so ≤ 0.5 is a ≥77% reduction *on the same
suite*. Reference: exploratory seed 42 gives λ̂_OOD = 0.226 → would pass.

The OOD suite is opened **once per seed**, on the **already-selected** checkpoint
only. Evaluating several checkpoints on OOD and choosing among them would
re-introduce the selection bias this protocol exists to remove.

**S3 — No degenerate choice behaviour.** Per seed: consistency ≥ 0.50 **and**
keep-both ≤ 0.50 **and** trade-both ≤ 0.50. This blocks a checkpoint from
"achieving" low λ̂ by collapsing to a constant policy.

**S4 — η̂ reported, not gated.** η̂ is reported jointly with λ̂ for every seed but
is **not** a pass/fail gate: the exploratory run shows the intervention removes
the asymmetry (λ) far more effectively than the level of rigidity (η̂ = 0.504 at
the selected checkpoint). Pre-declaring an η̂ threshold we already expect to fail
would be dishonest. Instead we pre-declare the **claim limit**: we will not claim
η is eliminated.

**S5 — Capability non-inferiority (GSM8K). FROZEN.**

Per seed, vs the **exact local base** (86.88%), paired over all 1,319 items:

> lower bound of the paired 95% bootstrap CI on Δ (tuned − base) **≥ −3 pp**

Reference: exploratory seed 42 gives Δ = −0.99 pp, CI [−2.12, +0.15],
McNemar p = 0.111 → would pass.

**S6 — IFEval.** Same non-inferiority form (**≥ −5 pp**) once the harness exists.
If IFEval is not ready before the seeds finish, report GSM8K alone and state that
the capability evidence is single-benchmark.

## What we will report regardless of outcome

- Every seed's λ̂, η̂, consistency, keep-both, trade-both, selected step, KL.
- Median + range across seeds; never the best seed alone.
- Ownership-free preference agreement per seed.
- GSM8K (and IFEval if available) per seed.
- If seeds disagree materially, that IS the finding: report the variance rather
  than the best run.

## Frozen decisions (July 15, 2026, pre-launch)

| item | decision |
|---|---|
| Confirmatory seeds | **2 new** (`SEED=1,2`). `seed=42` = exploratory, **excluded from the denominator** |
| Bar | **2/2** new seeds; if they disagree → third NEW seed, require **2/3 fresh** |
| Training endpoint | **MAX_STEPS=30000**, fixed for every seed |
| Selection grid | **2k–30k @ 2k** (15 checkpoints/seed), ID/validation suite |
| Selection rule | `CHECKPOINT_PROTOCOL.md` min-d, mechanical |
| **S2 primary threshold** | **\|λ̂_OOD\| ≤ 0.5** (OOD base 2.209; ID is contaminated by selection) |
| **S5 capability margin** | paired 95% CI lower bound **≥ −3 pp** on GSM8K |
| S4 η̂ | reported jointly, **not gated**; we will not claim η is eliminated |

Eval budget: 15 ID selection evals + 1 OOD + 1 GSM8K per new seed ≈ **34 evals**.
(seed 42 needs no grid top-up now that it is out of the denominator.)

## Amendments

### July 28, 2026 — terminology correction (no design change)

The phrase "zero-reward/DAPO filtering" used in the July 21 amendment below is
imprecise and is superseded by **"zero task-reward advantage fraction"**
(the logged `frac_reward_zero_std`). The frozen text is left unedited; only the
name of the diagnostic changes. Nothing is filtered or skipped: zero-diversity
groups still generate, backprop, and take an optimizer step; `scale_rewards:
"none"` still mean-centres advantages; and `beta=0.04` leaves a KL-only update at
zero task advantage. No threshold, outcome, selection rule, or confirmatory
checkpoint is affected. See `KNOWN_ISSUES.md` #4.

### July 21, 2026 — non-gating optimization and utility diagnostics

We will inspect GRPO training dynamics and structural-estimator stability as a
post-hoc diagnostic. This does **not** change any frozen threshold, outcome,
selection rule, or confirmatory checkpoint. The diagnostic may include the
matched local base as training step 0 and an early saved checkpoint such as
step 600 (the nearest saved checkpoint after 500 because adapters were saved
every 200 steps). Any checkpoint outside the frozen 2k–30k @ 2k grid is
exploratory only: it cannot be selected, cannot affect the seed verdict, and
must not be evaluated on OOD for checkpoint choice.

The diagnostic will report (i) GRPO reward/loss, KL, entropy, gradient norm,
learning rate, reward dispersion, and zero-reward/DAPO filtering; (ii) the NLS
starting and final objective under multiple numerical starts; and (iii) fitted
alpha, beta, and utility drift relative to the appropriate base. Results will
be labelled post-hoc and non-gating.
