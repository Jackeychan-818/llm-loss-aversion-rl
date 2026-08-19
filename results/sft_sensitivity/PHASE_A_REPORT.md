# Phase A — SFT effective-batch sensitivity (endpoint evaluation only)

*Exploratory / post-hoc optimization ablation, 3 seeds. Protocol:
`SFT_BATCH_LR_SENSITIVITY_PROTOCOL.md` + Amendment 1. Code commit `0f5dafb`.*

**Does not revise the frozen matched-SFT selection and licenses no SFT-vs-GRPO
claim.** `test_goods` is the already-open validation set; every number here is a
validation estimate, not final-test performance.

## Execution

12 cells: effective batch {1, 16, 32, 64} × seeds {1, 2, 3}, LR 1e-6, exactly
6,016 unique prompts each, endpoint (6,016) evaluation only. All 12 trained and
evaluated successfully; **0 non-finite records** anywhere; every trainer history
complete at its expected step count.

| | GPU-hours | SU |
|---|---:|---:|
| training (12 cells) | 3.631 | 232.4 |
| endpoint evaluation (12) | 3.882 | 248.4 |
| **total** | **7.513** | **480.8** |
| authorized ceiling | 8.2 | 525 |

SU balance 1,933.353 → **1,452.123** (Δ 481.23, matching the summed per-job
charge). **Within authorization.**

## Frozen selection rule — applied exactly

Lowest mean endpoint validation cross-entropy across seeds 1–3 among the large
batches; exact tie → smaller batch.

| effective batch | mean validation CE | sd | selected |
|---:|---:|---:|:--:|
| 16 | **0.594123** | 0.001923 | **✓** |
| 32 | 0.665124 | 0.005038 | |
| 64 | 2.064153 | 0.009740 | |

**Selected: effective batch 16.** No tie-break needed. Behavioural metrics did
not enter this decision.

(Batch 1 is not a candidate — the rule selects among large batches — but for
reference its mean validation CE is 1.5228 ± 0.0584.)

## Per-cell results

| eb | seed | val CE | λ | η | d | consistency | keep-both | trade-both |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.4573 | −0.0952 | +0.0383 | 0.1026 | 0.7167 | 0.1276 | 0.1557 |
| 1 | 2 | 1.5420 | +0.2560 | −0.2379 | 0.3494 | 0.7201 | 0.1755 | 0.1044 |
| 1 | 3 | 1.5691 | +0.1199 | −0.0534 | 0.1312 | 0.7237 | 0.1445 | 0.1318 |
| 16 | 1 | 0.5963 | −0.1698 | +0.0884 | 0.1914 | 0.5575 | 0.1836 | 0.2589 |
| 16 | 2 | 0.5930 | −0.0392 | +0.0816 | 0.0905 | 0.5745 | 0.1966 | 0.2289 |
| 16 | 3 | 0.5930 | −0.1015 | +0.0762 | 0.1269 | 0.5684 | 0.1892 | 0.2424 |
| 32 | 1 | 0.6619 | −0.0795 | +0.6678 | 0.6725 | 0.5124 | 0.0996 | 0.3880 |
| 32 | 2 | 0.6625 | −0.0572 | +0.6479 | 0.6504 | 0.5106 | 0.1000 | 0.3894 |
| 32 | 3 | 0.6709 | −0.0484 | +0.6806 | 0.6823 | 0.4980 | 0.0949 | 0.4071 |
| 64 | 1 | 2.0529 | +2.0206 | +1.5052 | 2.5195 | 0.0234 | 0.9758 | 0.0008 |
| 64 | 2 | 2.0694 | +2.1912 | +1.4486 | 2.6265 | 0.0234 | 0.9759 | 0.0007 |
| 64 | 3 | 2.0701 | +2.0621 | +1.3602 | 2.4706 | 0.0223 | 0.9770 | 0.0007 |

### Aggregate (mean ± sd across seeds)

| eb | val CE | λ | η | consistency | seed sd of λ |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.5228 ± 0.0584 | +0.094 ± 0.177 | −0.084 ± 0.141 | 0.7202 ± 0.0035 | 0.177 |
| 16 | 0.5941 ± 0.0019 | −0.104 ± 0.065 | +0.082 ± 0.006 | 0.5668 ± 0.0086 | 0.065 |
| 32 | 0.6651 ± 0.0050 | −0.062 ± 0.016 | +0.665 ± 0.016 | 0.5070 ± 0.0079 | 0.016 |
| 64 | 2.0642 ± 0.0097 | +2.091 ± 0.089 | +1.438 ± 0.073 | 0.0230 ± 0.0006 | 0.089 |

## Optimization diagnostics (mean across seeds)

| eb | updates | grad-norm median | P90 | P99 | max | clipped | loss median | loss roughness |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,016 | 0.914 | 89.91 | 131.06 | 309.7 | 61.0% | 0.0042 | 0.7817 |
| 16 | 376 | 2.616 | 9.88 | 13.13 | 14.7 | 100% | 0.2192 | 0.1044 |
| 32 | 188 | 6.420 | 10.44 | 11.99 | 13.4 | 100% | 0.3692 | 0.1196 |
| 64 | 94 | 9.083 | 10.61 | 11.37 | 12.3 | 100% | 0.7750 | 0.1260 |

Learning-rate schedules verified: every cell peaks at 1e-6 after its 5% warmup
and decays to ~1e-10 or below at its own horizon.

Loss roughness is **secondary/descriptive and not comparable across batches**: a
batch-16 cell takes 16× fewer optimizer updates over the same prompts, and each
of its logged losses is a 16-example mean rather than a single example.

## Findings

**1. Larger effective batches sharply reduce gradient-norm tails.** The P99
falls from **131 to 11–13** (~10×) and the maximum from **310 to 12–15** (~25×).
The *median* rises (0.91 → 9.08) because averaging removes the many near-zero
single-example gradients as well as the extreme ones. This is textbook variance
reduction, and it is the clearest result here: **yes, batch 1 produces
heavy-tailed gradients, and batching removes the tail.**

**2. Structural estimates become more reproducible across seeds.** The seed
standard deviation of λ falls from **0.177 (batch 1) to 0.065 (16) to 0.016
(32)**, and of validation CE from 0.058 to 0.002. Larger batches give a
noticeably more repeatable measurement.

**3. But this does NOT translate into better validation behaviour at fixed LR.**
Paired consistency falls monotonically with batch size: **0.720 → 0.567 → 0.507
→ 0.023**. Batch 1 has by far the best consistency despite the worst validation
CE among {1, 16, 32}. Batch 32 additionally develops a large status-quo bias
(η ≈ +0.67) absent at batch 1 and 16.

**4. Batch and learning progress are confounded at fixed LR — the central
caveat.** At 6,016 prompts and LR 1e-6, batch 64 takes only 94 optimizer steps
and barely trains: λ = 2.09, consistency 0.023, keep-both 0.976, i.e. close to
the untrained base (λ = 7.637, consistency 0.009). Its poor score is an
artefact of taking 64× fewer steps, **not** evidence that batch 64 is
intrinsically worse. Phase A alone therefore cannot answer "is batch 1 bad?" —
disentangling batch from effective learning progress is exactly what the Phase-B
learning-rate sweep is for, and Phase B is **not authorized**.

**5. Clipping becomes universal at every large batch — a second confound.**
With `max_grad_norm = 0.1` unchanged, batch 1 clips 61% of updates while
batches 16/32/64 clip **100%**. Since the accumulated norm concentrates at
2.6–9.1, every large-batch update is rescaled to the same fixed length. Those
cells are therefore not "the same optimizer with a bigger batch" — they are
effectively taking **normalized-gradient steps of constant size**. Any read of
the batch effect at these settings is entangled with that.

### Vocabulary

Reported as distinct quantities, per protocol §8: pre-clipping norm (above);
clipping frequency (61% → 100%); numerical failure (**zero** non-finite records
in all 12 cells); optimization noise (loss roughness, descriptive); behavioural
drift (not measured — intermediate checkpoints retained but unevaluated).

**No gradient explosion occurred.** Large pre-clipping norms at batch 1 are
heavy-tailed single-example gradients, not divergence: no run produced a
non-finite loss, gradient or parameter.

## Limits

Three seeds. Exploratory and post-hoc. One learning rate, so the batch effect is
confounded with both the number of optimizer steps and universal clipping.
`test_goods` is validation data that informed reward construction and prior
selection. Nothing here revises the frozen matched-SFT selection, and no
SFT-versus-GRPO comparison is expressed or implied.

## Retained, not evaluated

The 2,048- and 4,096-prompt-exposure checkpoints of all 12 cells are preserved
under `checkpoints/sft_sensitivity/`, unevaluated, per Amendment 1.

Evaluating batch 1 and the selected batch 16 at both intermediate exposures —
2 batches × 3 seeds × 2 checkpoints = **12 evaluations** — would cost
**≈3.88 GPU-hours ≈ 248 SU** (17% of the remaining 1,452 SU), based on the
measured 0.3235 GPU-h per endpoint evaluation. **Not submitted.**
