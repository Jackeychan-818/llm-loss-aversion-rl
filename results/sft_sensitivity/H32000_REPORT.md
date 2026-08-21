# Extended horizon — eb32 @ 1e-4, 32,000 prompts (single exploratory run)

*Amendment 4. **One seed, off-grid probe LR. It cannot select or promote
anything** — Amendment 3 bars a probe LR from promoting, Amendment 2 bars
promotion on one seed.* Code `fa8f4c5`.

## Result

| | 6,016-prompt probe | **32,000-prompt run** |
|---|---:|---:|
| optimizer updates | 188 | 1,000 |
| **validation CE** | 0.2206 | **0.0807** |
| λ (SE) | +0.0823 (0.0050) | **+0.000093 (0.000095)** |
| η (SE) | −0.3176 (0.0161) | **−0.0452 (0.0161)** |
| d = √(λ²+η²) | 0.3281 | **0.0452** |
| consistency | 0.8930 | **0.9671** |
| keep-both | 0.0335 | 0.0149 |
| trade-both | 0.0735 | 0.0180 |

Validation cross-entropy falls **63%**. λ lands at 9.3e-05 with SE 9.5e-05 —
statistically indistinguishable from zero. η shrinks sevenfold. Paired
consistency reaches **96.7%**.

The fit is not degenerate: keep-both 1.5% and trade-both 1.8% are both low, so
the model is making genuine case-by-case choices rather than collapsing onto one
answer. Compare the matched local base at λ = 7.637, consistency 0.9%.

## Optimization diagnostics

| | 6,016 probe | 32,000 run |
|---|---:|---:|
| train loss, median / last-10% | 0.0985 / 0.0682 | 0.0468 / 0.0196 |
| grad median / P90 / P99 / max | 0.775 / 1.73 / 8.47 / 11.63 | 0.588 / 1.18 / 6.21 / 11.94 |
| **clipped fraction** | **100.0%** | **98.3%** |
| ‖Δθ‖ median / max | 0.0750 / 0.2976 | 0.0674 / 0.1959 |
| relative update norm | 2.29e-03 | 2.00e-03 |
| non-finite records | 0 | 0 |
| zero updates | 1 | 1 |

**Clipping stopped being universal.** This is the first run in the entire
experiment below 100%: 1.7% of updates (17 of 1,000) had a pre-clipping gradient
norm at or under 0.1. Together with the falling P90/P99, that is the signature of
a model actually converging rather than being held at a fixed step length.

The single zero update is the final step, where cosine decay drives the LR into
bf16 underflow — the same benign artefact seen in every Phase-B run, reported
rather than suppressed.

## Cost

Training 1h31m (1,000/1,000 updates), evaluation ~20 min. Measured
**1.565 GPU-h = 100.2 SU** against a 126 SU estimate. Balance 892.7 → **792.5 SU**.
Phase-B cumulative ≈654 of the 850 ceiling.

## What this does and does not establish

**Establishes:** at effective batch 32 with peak LR 1e-4, extending the schedule
from 6,016 to 32,000 prompts improves every measured quantity, and does so
without instability — zero non-finite records across 1,000 updates at a learning
rate 100× the project's original setting.

**Does not establish** any of the following:

- *That 1e-4 is optimal.* It remains an unbracketed boundary winner; the search
  upward was stopped deliberately, not because a turnover was found.
- *That this is a selected configuration.* One seed, probe LR. Both amendments
  bar promotion. Confirming it needs seeds 2 and 3 (~200 SU at this horizon).
- *That more data caused the improvement.* **1,000 updates at the same peak LR is
  ~5.3× the probe's total optimization path length.** Exposure and optimization
  moved together, so this run tests "1e-4 at 32,000 prompts" and cannot separate
  the two. Isolating exposure would need a 32,000-prompt run at a
  path-length-matched (lower) LR.
- *Anything about batch size.* eb16 was not run at this horizon.

## Context — the number worth noticing

The frozen matched-SFT baseline trains on 30,000 prompts at lr 1e-6 and reaches
consistency **0.867** at its endpoint; its frozen selection (seed 1, step 4,000)
sits at **0.7298**. This run reaches **0.9671** at comparable data exposure,
differing essentially in the learning rate.

That does not invalidate the matched baseline, which is *deliberately* matched to
GRPO's hyper-parameters. It sharpens what that baseline is — a
**matched-hyper-parameter** SFT baseline, not a best-effort one — and any
"RL is necessary" claim has to say which of the two it means. That question is
governed by `METHOD_COMPARISON_PROTOCOL.md` and is recorded here, not acted on.

## Scope

`test_goods` used only as the already-open validation set; it informed reward
construction and prior checkpoint selection, so these are validation estimates,
not final-test performance. No method-comparison, frozen-unused, OOD-50, GSM8K,
semantic or framing suite was read or evaluated. The 17 intermediate snapshots
are preserved and unevaluated — `submit_eval_sft_sensitivity.pbs` refuses any
exposure other than 6,016 and 32,000. Nothing here revises the frozen
matched-SFT selection.
