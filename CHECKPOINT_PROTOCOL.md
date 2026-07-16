# Frozen Checkpoint-Selection Protocol

*Frozen July 15, 2026 — before the Qwen-own-delta seed runs.*

This protocol answers `PAPER_READINESS.md` **#4** ("Freeze a checkpoint rule
before new seeds, apply it mechanically... The rule must consider lambda, eta,
and consistency jointly"). It is declared **in advance** of the seed runs and is
applied by `eval/select_checkpoint.py` with **no human judgement at selection
time**.

## Why this exists

The original step-8,000 choice was made *post hoc*: seven checkpoints were
inspected on `test_goods`, and 8,000 was preferred over 20,000 (which has the
smaller |λ̂|) because 20,000 has a much worse η̂. That reasoning is defensible but
was **not written down in advance**, which is exactly the degree of freedom a
reviewer will probe. From now on the rule is fixed first and applied
mechanically.

## Data roles (fixed)

| Set | Role | May be used for |
|---|---|---|
| `test_goods.json` | **VALIDATION** | checkpoint selection ONLY |
| frozen OOD suite | **FINAL** | opened once per seed, AFTER selection, on the selected checkpoint only — this is where the frozen success threshold (\|λ̂_OOD\| ≤ 0.5, `PRE_REGISTRATION.md` S2) is judged |

This rule selects a checkpoint; it does **not** decide success. Selection happens
on validation, the verdict is read on OOD. Never evaluate several checkpoints on
OOD and choose among them — that re-opens the bias this protocol removes.

`test_goods` cannot serve as a final test for Qwen-own-delta runs: it informed
both the reward construction (PAPER_READINESS #3) and checkpoint selection (#4).
Any λ̂ reported on it is a **validation** number.

## The rule

Applied per seed, independently, to a checkpoint grid declared in advance:
**every 2,000 steps from 2k to 30k** (15 checkpoints; see `PRE_REGISTRATION.md`).

**Step 1 — Eligibility.** Discard a checkpoint unless all hold:
- `consistency >= 0.50` — the pair makes a symmetric choice at least half the
  time. This rules out degenerate keep-both / trade-both collapse, where λ̂ and η̂
  are not meaningfully identified.
- NLS converged with finite, non-zero standard errors for both λ̂ and η̂.

**Step 2 — Primary criterion.** Among eligible checkpoints, minimise the joint
distance from the rational ideal (λ, η) = (0, 0):

```
d = sqrt(lambda_hat^2 + eta_hat^2)
```

λ and η are both latent-utility-scale deviations from ownership neutrality and
both target 0, so an unweighted Euclidean distance is the natural joint
objective. **λ alone is not the criterion**: a checkpoint can buy a small λ̂ by
loading the rigidity onto η̂ (step 20,000 is exactly this — smallest |λ̂| = 0.083,
worst η̂ = 1.263, worst d = 1.266).

**Step 3 — Tie-break 1.** If two checkpoints' `d` differ by less than
`D_TOL = 0.05`, prefer the higher `consistency`.

**Step 4 — Tie-break 2.** If consistency then differs by less than
`CONS_TOL = 0.02`, prefer the **earliest** checkpoint (less drift from base).

**Step 5 — No-pass.** If no checkpoint is eligible, report that the seed
produced no admissible checkpoint. **Do not relax the thresholds post hoc.**

## Thresholds (frozen)

| Constant | Value |
|---|---|
| `CONSISTENCY_FLOOR` | 0.50 |
| `D_TOL` | 0.05 |
| `CONS_TOL` | 0.02 |
| checkpoint grid | every 2,000 steps, 2k–30k |

## Verification on the existing run (post hoc, illustrative only)

Applying the rule mechanically to the completed Qwen-own-delta run reproduces
the original choice:

| ckpt | λ̂ | η̂ | consistency | d |
|---|---|---|---|---|
| 5,000 | +0.226 | +0.566 | 0.610 | 0.609 |
| **8,000** | **+0.111** | **+0.504** | 0.697 | **0.516** ← selected |
| 10,000 | +0.131 | +0.650 | 0.711 | 0.663 |
| 12,000 | +0.430 | +0.717 | 0.657 | 0.836 |
| 15,000 | +0.673 | +0.576 | 0.651 | 0.886 |
| 20,000 | −0.083 | +1.263 | 0.769 | 1.266 |
| 30,000 | +0.593 | +0.815 | 0.728 | 1.008 |

The rule selects **step 8,000** by a clear margin, and demotes step 20,000 to
last. This is reassuring but **does not retroactively make 8,000 a
pre-registered choice** for the existing run, and it does not remove the
selection-on-validation bias in its λ̂ = 0.111. It means the frozen rule is
consistent with the judgement already made, and the seeds will be selected
without judgement.

## Honest reading of the current numbers

At the selected checkpoint η̂ = 0.504 (SE 0.029) — **substantial residual
status-quo bias**, and keep-both is still 24.2% of pairs. The intervention drives
the *asymmetry* λ̂ toward 0 far more effectively than it removes the *level* of
choice rigidity η̂. Report λ and η jointly; a λ-only headline would misrepresent
the result.

## Usage

```bash
# selection (validation data only)
python eval/select_checkpoint.py --feature baseline --pattern 'Qwen-7B-GRPO-qd-ckpt*'

# writes results/checkpoint_selection/<run>.json recording the chosen checkpoint,
# the full grid, the rule constants, and the code commit.
```
