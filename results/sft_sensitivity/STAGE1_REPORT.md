# Phase B, Stage 1 — two-seed LR screening (STOPPED at a frozen boundary)

*Exploratory / post-hoc. Protocol: `SFT_BATCH_LR_SENSITIVITY_PROTOCOL.md` +
Amendments 1–2, frozen at `23df937` before any Phase-B GPU job.*

## Outcome: the frozen stop rule fired

**1e-5 — the top of the grid — wins for all three batches.** Amendment 2 states:
*"1e-5 wins → stop for a cost and safety review before considering 3e-5."*
Work has stopped. No Stage-2 confirmation, no 3e-5, no seed-3 runs.

## Endpoint validation cross-entropy (seeds 1–2)

| batch | 1e-6 *(reused)* | 3e-6 | 1e-5 | winner |
|---:|---:|---:|---:|:--|
| 16 | 0.594687 | 0.548666 | **0.441274** | 1e-5 |
| 32 | 0.662219 | 0.569970 | **0.498594** | 1e-5 |
| 64 | 2.061167 | 0.597263 | **0.542131** | 1e-5 |

Monotone in LR for every batch. No interior optimum exists inside the tested
grid: the objective is still falling at the boundary.

## Seed-disagreement rule — did not fire

| batch | gap (top two) | noise floor | seed-3 required? |
|---:|---:|---:|:--|
| 16 | 0.107392 | 0.0019 | no (57× floor) |
| 32 | 0.071376 | 0.0050 | no (14× floor) |
| 64 | 0.055132 | 0.0097 | no (6× floor) |

Both seeds ranked the top two LRs identically in every batch, and every gap
clears its floor by a wide margin. Seed agreement is unusually tight.

## Pre-registered prediction — falsified

The non-binding path-length heuristic predicted **eb16→1e-6, eb32→3e-6,
eb64→3e-6**, with the three winners landing within 0.010 of one another.

Actual: **all three chose 1e-5**, and the winners span 0.441–0.542 — a maximum
pairwise difference of **0.101, ten times the pre-declared tolerance**. Path-length
matching does not explain the pattern. The prediction had no bearing on the
selection rule and did not receive one.

The simpler reading is that the entire grid was mis-centred: every batch,
including 16, was under-trained at 1e-6, so this is not a story about batch size
compensating for step count.

## Behavioural diagnostics (reported, not used for selection)

| batch | LR | λ | η | consistency | keep-both | trade-both |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1e-6 | −0.105 | +0.085 | 0.566 | 0.153 | 0.281 |
| 16 | 3e-6 | +0.098 | −0.149 | 0.649 | 0.145 | 0.206 |
| 16 | **1e-5** | +0.122 | −0.281 | **0.771** | 0.085 | 0.145 |
| 32 | 1e-6 | −0.068 | +0.658 | 0.512 | 0.386 | 0.103 |
| 32 | 3e-6 | +0.072 | −0.081 | 0.609 | 0.169 | 0.223 |
| 32 | **1e-5** | +0.122 | −0.218 | **0.724** | 0.117 | 0.159 |
| 64 | 1e-6 | +2.106 | +1.477 | 0.023 | 0.977 | 0.000 |
| 64 | 3e-6 | −0.088 | +0.235 | 0.575 | 0.214 | 0.211 |
| 64 | **1e-5** | +0.120 | −0.186 | **0.667** | 0.144 | 0.188 |

**Unlike Phase A, the behavioural metrics agree with the CE selection.**
Consistency rises with LR in every batch, and the CE-selected cell is the most
consistent in each. In Phase A the two disagreed sharply — batch 1 had the best
consistency and the worst CE. Here they point the same way, which makes the
selection less fragile. They still did not enter the decision.

**The Phase-A eb64 pathology is explained.** eb64 at 1e-6 (λ 2.11, consistency
0.023, keep-both 0.977) was near-untrained. At 3e-6 it recovers to consistency
0.575 and λ −0.088. That confirms the Phase-A diagnosis: eb64 was not
intrinsically bad, it had taken only 94 updates at too small a step.

## Optimization diagnostics

| batch | LR | grad median | grad max | clipped | clip coef (median) | ‖Δθ‖ median | relative | zero updates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 3e-6 | 1.84 | 16.3 | 100% | 0.0544 | 0.00245 | 7.6e-05 | 1 |
| 16 | 1e-5 | 1.67 | 15.8 | 100% | 0.0600 | 0.00789 | 2.4e-04 | 1 |
| 32 | 3e-6 | 1.50 | 12.5 | 100% | 0.0666 | 0.00280 | 8.7e-05 | 1 |
| 32 | 1e-5 | 1.12 | 12.7 | 100% | 0.0891 | 0.00887 | 2.7e-04 | 1 |
| 64 | 3e-6 | 1.87 | 11.0 | 100% | 0.0538 | 0.00365 | 1.3e-04 | 1 |
| 64 | 1e-5 | 0.89 | 11.1 | 100% | 0.1119 | 0.00958 | 3.1e-04 | 1 |

**The update-norm diagnostic answers the clipping question.** Realised
parameter movement scales essentially **linearly with the learning rate** —
eb16 goes 0.00245 → 0.00789 for a 3.33× LR increase (3.2×); eb32 0.00280 →
0.00887 (3.2×); eb64 0.00365 → 0.00958 (2.6×). So even with 100% of updates
clipped, the learning rate still controls step size directly. Clipping is not
neutralising the LR, which vindicates the Amendment-2 correction: under
`adamw_torch_fused` a constant gradient rescaling largely cancels in
`m̂/(√v̂+ε)`, so universal clipping is far milder than a naive reading of the
clipping fraction suggests.

Each Phase-B run records exactly **one zero update**, consistent with the final
optimizer step, where cosine decay drives the LR to ~1e-11 and the bf16
parameter delta underflows. Reported rather than suppressed, as the test
required.

Clipping remains **100% at every batch and every LR**, so the batch comparison
is still made under universal clipping and inherits that caveat.

## Cost

| | SU |
|---|---:|
| Stage-1 training (12 cells, measured) | 231.0 |
| Stage-1 evaluation (12 cells) | 243.9 |
| **Stage-1 total** | **474.9** |
| screening authorization | 500 |
| program cumulative (Phase A 480.8 + Stage 1 474.9) | 955.7 |
| balance remaining | **977.2** |

Stage 1 came in under its 500 SU gate. Note the program cumulative already
exceeds the 850 SU Phase-B ceiling **if Phase A is counted against it**; the 850
figure was set for Phase B alone, of which 474.9 is spent, leaving 375 SU inside
that ceiling.

## What this does and does not establish

It establishes that, on a 6,016-prompt cosine schedule with clipping 0.1, **the
tested LR grid was too low for every batch**, and that 1e-5 is the best of
{1e-6, 3e-6, 1e-5} for all three.

It does **not** identify any batch's optimal LR, because every optimum lies at or
beyond the grid boundary. The cross-batch ordering at 1e-5 (eb16 0.441 < eb32
0.499 < eb64 0.542) is a comparison **at a shared constraint, not at each
batch's own optimum**, and must not be read as "batch 16 is best". It is equally
consistent with all three still being under-trained by different amounts.

No frozen or untouched suite was opened. Phase-A intermediate checkpoints were
not evaluated. The 16 adapter-only snapshots per Phase-B run are preserved and
unevaluated. Batch 1 remains abandoned, so nothing here speaks to whether
batching beats batch 1.
