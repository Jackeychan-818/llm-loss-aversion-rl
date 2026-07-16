# Pre-Registration — Qwen-own-delta Seed Replication

**STATUS: DRAFT — REQUIRES SIGN-OFF BEFORE THE FIRST SEED IS SUBMITTED.**
Numbers below are proposals. Once agreed, mark this FROZEN, commit it, and do not
change it after seeing any seed result. If it is not frozen before launch, the
read-out is a post-hoc judgement and the replication does not do its job.

*Drafted July 15, 2026.*

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

**Seed set for the formal criterion: n = 3** — `seed=42` (exploratory) plus the
two new confirmatory seeds. Counting seed 42 is legitimate *because* the design
uses the 2k–30k grid + the frozen `min d` rule rather than a fixed 8k stop: seed
42's checkpoint is re-selected **mechanically**, exactly like the new seeds.
**Residual caveat to state in the paper:** the rule's design (min d, consistency
floor) was informed by seed 42's curve, so seed 42 is not fully independent
evidence.

**Required for a like-for-like 2-of-3:** seed 42 must be selected over the SAME
2k–30k grid as the new seeds. All 15 of its grid checkpoints exist on disk, but
only 5 grid points are currently evaluated (8k, 10k, 12k, 20k, 30k); its original
selection used a 7-point grid that included 5k/15k — which are not on the 2k grid
— and never evaluated 2k, 4k or 6k. **10 further evals of seed 42 are required**
(2k, 4k, 6k, 14k, 16k, 18k, 22k, 24k, 26k, 28k) before its selection is
comparable. Until then its `d = 0.516` at step 8,000 is a 7-point-grid result,
and an earlier checkpoint could in principle win on the full grid.

Total selection-eval budget: 10 (seed 42 top-up) + 15 + 15 = **40 evals ≈ 40
GPU-hours**.

**Formal bar: 2 of 3.** Requiring 3/3 is too brittle — one unlucky training run
would make the entire method "fail", which measures luck rather than reliability.

**Report BOTH**, always:
1. the formal **2-of-3** verdict, *and*
2. **every individual seed's** numbers (λ̂, η̂, consistency, keep/trade-both,
   selected step, KL) — never the best seed alone — *and*
3. whether **3 of 3** would also have passed, so the reader sees how much the
   verdict depends on the 2/3 bar.

**S1 — Eligible checkpoint exists.** Required in **2 of 3** seeds. A seed with no
checkpoint passing `CHECKPOINT_PROTOCOL` eligibility (consistency ≥ 0.50, finite
non-zero SEs) is a **method failure**, not a discarded seed.

**S2 — Loss-aversion reduction (primary). THRESHOLD DEFERRED BY DECISION.**

The proposed bar was λ̂ ≤ 1.0 on the ID/validation suite (multiplier (1+λ) < 2.0;
≥87% below the matched base λ̂ = 7.637). **The threshold has been deferred until
after training.**

**Consequence, recorded before the fact:** because the bar is not fixed in
advance, S2 is **NOT a pre-registered test**. The paper must report the seed
λ̂ values **descriptively** — every seed, plus median and range — and must NOT
claim a pre-specified success criterion, "confirmed our hypothesis", or any
threshold-based pass/fail framing chosen after seeing the numbers. Monitoring
training is unaffected; only the success bar is at issue.

If a threshold is agreed BEFORE any seed's λ̂ is inspected, replace this section
and mark it pre-registered.

**S3 — No degenerate choice behaviour.** Per seed: consistency ≥ 0.50 **and**
keep-both ≤ 0.50 **and** trade-both ≤ 0.50. This blocks a checkpoint from
"achieving" low λ̂ by collapsing to a constant policy.

**S4 — η̂ reported, not gated.** η̂ is reported jointly with λ̂ for every seed but
is **not** a pass/fail gate: the exploratory run shows the intervention removes
the asymmetry (λ) far more effectively than the level of rigidity (η̂ = 0.504 at
the selected checkpoint). Pre-declaring an η̂ threshold we already expect to fail
would be dishonest. Instead we pre-declare the **claim limit**: we will not claim
η is eliminated.

**S5 — Capability non-inferiority (GSM8K). MARGIN DEFERRED BY DECISION.**

The proposed margin was: lower bound of the paired 95% bootstrap CI on
Δ (tuned − base, vs the exact local base 86.88%) ≥ −5 pp.
**Deferred until after training.**

**Consequence:** capability results will be reported descriptively (Δ with paired
CI and McNemar p per seed) and must NOT be framed as a pre-registered
non-inferiority test. Reference: the exploratory run gives Δ = −0.99 pp,
CI [−2.12, +0.15], McNemar p = 0.111.

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

## Decisions taken (July 15, 2026)

| item | decision |
|---|---|
| Seeds | **2 new** (`SEED=1,2`) as a first stage + `seed=42` exploratory |
| Grid | **2k–30k at 2k intervals** (15 checkpoints/seed) |
| Training endpoint | **MAX_STEPS=30000**, fixed for every seed |
| Formal pass bar | **2 of 3** seeds (n=3 incl. exploratory seed 42) — 3/3 judged too brittle |
| S2 λ̂ threshold | **deferred** → S2 is descriptive, not a pre-registered test |
| S5 capability margin | **deferred** → descriptive, not a non-inferiority test |

## Open — settle before launch

1. Whether to fix the S2 λ̂ threshold before inspecting any seed's λ̂ (which would
   recover a genuine pre-registered test) or accept descriptive-only reporting.
   The 2-of-3 bar is agreed; what "pass" means numerically is not.
