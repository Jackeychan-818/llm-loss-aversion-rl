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
| Seeds | **3 new**: `SEED=1,2,3` (`seed=42` is the exploratory run, not confirmatory) |
| Training endpoint | **`MAX_STEPS=30000`** — fixed for every seed, so all seeds expose the same checkpoint grid |
| Checkpoint saves | every 200 steps |
| Selection grid | every **1,000** steps, 1k–30k |
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

## Success criteria (PROPOSED — confirm before freezing)

Judged **per seed**. Reported as **median across seeds, with the full range and
every individual seed** — never the best seed alone. The median is used rather
than the mean because the checkpoint curve is strongly non-monotonic and a single
outlier would dominate a 3-seed mean.

**S1 — Eligible checkpoint exists.** Required in **3 of 3** seeds. A seed with no
checkpoint passing `CHECKPOINT_PROTOCOL` eligibility (consistency ≥ 0.50, finite
non-zero SEs) is a **method failure**, not a discarded seed.

**S2 — Loss-aversion reduction (primary).** Per seed, at the selected checkpoint,
on the **ID/validation** suite:

> **λ̂ ≤ 1.0** — i.e. the endowed-good multiplier (1 + λ) < 2.0, less than double
> weighting of the held good; a ≥87% reduction from the matched base λ̂ = 7.637.

Required in **3 of 3** seeds for "the method reliably reduces loss aversion".
**2 of 3** = partial/qualified support and must be reported as such.

**S3 — No degenerate choice behaviour.** Per seed: consistency ≥ 0.50 **and**
keep-both ≤ 0.50 **and** trade-both ≤ 0.50. This blocks a checkpoint from
"achieving" low λ̂ by collapsing to a constant policy.

**S4 — η̂ reported, not gated.** η̂ is reported jointly with λ̂ for every seed but
is **not** a pass/fail gate: the exploratory run shows the intervention removes
the asymmetry (λ) far more effectively than the level of rigidity (η̂ = 0.504 at
the selected checkpoint). Pre-declaring an η̂ threshold we already expect to fail
would be dishonest. Instead we pre-declare the **claim limit**: we will not claim
η is eliminated.

**S5 — Capability non-inferiority (GSM8K).** Per seed, vs the **exact local base**
(86.88%), paired over all 1,319 items:

> lower bound of the paired 95% bootstrap CI on Δ (tuned − base) **≥ −5 pp**.

Reference: the exploratory run gives Δ = −0.99 pp, CI [−2.12, +0.15] → passes.
Required in **3 of 3** seeds.

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

## Open items to settle before freezing

1. Confirm or change the **λ̂ ≤ 1.0** threshold (S2).
2. Confirm the **−5 pp** capability margin (S5/S6) — 3 pp is defensible and
   stricter; the observed CI passes either.
3. Confirm **3 of 3** vs **2 of 3** for S2.
4. Confirm **3 new seeds** vs 2 (2 new + exploratory = only 2 confirmatory runs,
   which is below the "at least three" in `PAPER_READINESS.md` #5).
5. Evaluating a 1k–30k grid at 1k spacing is **30 evals/seed ≈ 90 GPU-hours**
   across 3 seeds. Confirm the grid spacing, or coarsen it (e.g. 2k) to halve it.
