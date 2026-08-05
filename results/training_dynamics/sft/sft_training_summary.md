# SFT training metrics — descriptive summary

Matched supervised fine-tuning baseline (Qwen-own-delta rational-choice targets), completion-only cross-entropy. **Optimization diagnostics only; not behavioral results.**

> Training loss measures fit to the supervised target tokens, whereas lambda, eta, consistency, and preference preservation are behavioral estimands computed after inference. A lower SFT loss is therefore not itself evidence of lower ownership dependence.

Smoothing in the figures: causal trailing rolling mean, window = 50 observations (~500 training steps), identical for every run; only current and preceding observations are used. All unsmoothed values are retained in the CSVs.

## Per-run descriptive statistics

| quantity | sft_full_seed1 | sft_full_seed2 | sft_pilot6k_seed1 |
|---|---|---|---|
| run role | full | full | pilot |
| seed | 1 | 2 | 1 |
| max_steps (cosine endpoint) | 30,000 | 30,000 | 6,000 |
| plotted observations | 3,000 | 3,000 | 600 |
| logging stride (steps) | 10 | 10 | 10 |
| first logged loss | 0.6636 | 0.7634 | 0.6636 |
| final logged loss | 0.5174 | 0.3209 | 0.5666 |
| minimum logged loss | 3.84e-06 | 1.77e-06 | 0.000795 |
| step of minimum loss | 29,480 | 13,340 | 5,910 |
| median loss, first 10% | 0.5030 | 0.4567 | 0.6189 |
| median loss, last 10% | 0.2018 | 0.1873 | 0.4627 |
| absolute change (early→late) | -0.3012 | -0.2694 | -0.1562 |
| relative change (early→late) | -59.9% | -59.0% | -25.2% |
| largest one-step loss increase | +2.2538 (2,990→3,000) | +1.6403 (18,640→18,650) | +2.1625 (2,990→3,000) |
| Trainer aggregate `train_loss` | 0.3374 | 0.3301 | 0.4925 |
| median gradient norm | 0.00743 | 0.00394 | 0.881 |
| maximum gradient norm | 513 | 543 | 182 |
| logged obs above clip threshold (0.1) | 38.2% | 37.3% | 61.3% |
| non-finite records | 0 | 0 | 0 |
| peak learning rate (step) | 1e-06 (1,510) | 1e-06 (1,510) | 1e-06 (310) |
| expected warmup boundary | 1,500 | 1,500 | 300 |
| LR matches configured schedule | yes | yes | yes |
| max relative LR deviation | < 1e-12 | < 1e-12 | < 1e-12 |
| final learning rate | 3.04e-15 | 3.04e-15 | 7.59e-14 |
| train runtime (s) | 5,422 | 4,668 | 1,051 |
| train steps / second | 5.53 | 6.43 | 5.71 |

### Reading the loss column

Each logged `loss` is the Trainer's mean over one logging interval (10 optimizer steps at batch 1 / accumulation 1), so the **final logged loss is one interval mean, not the run aggregate**. The Trainer's `train_loss` is the mean over all intervals; both are reported above and they are not interchangeable. With one training example per step the per-interval loss is extremely noisy, so early vs late medians — not any single point — carry the signal, and no convergence claim is made from a single log point.

## What these curves do and do not license

**May be read from these curves:** whether logged training loss generally decreased; whether the two seeds trace similar trajectories; whether either seed shows large spikes or numerical instability; how the gradient norm evolved; whether clipping was frequently active; whether the recorded learning-rate schedule matches the configured one; runtime and throughput.

**May NOT be read from these curves:** that full SFT reduced λ; that full SFT beats GRPO; that step 6,000 is the best full-run checkpoint; that the full seed-1 result equals the pilot result; that the model generalizes; that preferences are preserved; that training loss selects a checkpoint.

## Observations (descriptive)

- `sft_full_seed1`: median loss fell from 0.5030 (first 10% of logged training) to 0.2018 (last 10%), a relative change of -59.9%; the raw trace is highly dispersed (single logged points span 3.84e-06–2.76).
- `sft_full_seed2`: median loss fell from 0.4567 (first 10% of logged training) to 0.1873 (last 10%), a relative change of -59.0%; the raw trace is highly dispersed (single logged points span 1.77e-06–1.71).
- `sft_pilot6k_seed1`: median loss fell from 0.6189 (first 10% of logged training) to 0.4627 (last 10%), a relative change of -25.2%; the raw trace is highly dispersed (single logged points span 0.000795–2.65).
- The two full seeds track each other closely on the smoothed loss summaries (late medians 0.2018 vs 0.1873); this is a similarity of optimization trajectories only, not of behavior.
- `sft_full_seed1`: gradient norm is strongly bimodal (median 0.00743, max 513); 38.2% of LOGGED steps exceed the configured clip threshold 0.1, so clipping was frequently active. Expected at batch size 1: each step's norm comes from a single example, and an already-correct Yes/No target yields a near-zero gradient.
- `sft_full_seed1`: no non-finite loss / gradient-norm / learning-rate record.
- `sft_full_seed1`: the recorded learning rate reproduces the configured linear-warmup + cosine schedule to a maximum relative deviation below the 1e-12 machine-precision floor; peak at step 1,510, consistent with the 1,500-step warmup boundary.
- `sft_full_seed1`: 5,422 s wall clock (1.51 h) at 5.53 steps/s.
- `sft_full_seed2`: gradient norm is strongly bimodal (median 0.00394, max 543); 37.3% of LOGGED steps exceed the configured clip threshold 0.1, so clipping was frequently active. Expected at batch size 1: each step's norm comes from a single example, and an already-correct Yes/No target yields a near-zero gradient.
- `sft_full_seed2`: no non-finite loss / gradient-norm / learning-rate record.
- `sft_full_seed2`: the recorded learning rate reproduces the configured linear-warmup + cosine schedule to a maximum relative deviation below the 1e-12 machine-precision floor; peak at step 1,510, consistent with the 1,500-step warmup boundary.
- `sft_full_seed2`: 4,668 s wall clock (1.30 h) at 6.43 steps/s.
- `sft_pilot6k_seed1`: gradient norm is strongly bimodal (median 0.881, max 182); 61.3% of LOGGED steps exceed the configured clip threshold 0.1, so clipping was frequently active. Expected at batch size 1: each step's norm comes from a single example, and an already-correct Yes/No target yields a near-zero gradient.
- `sft_pilot6k_seed1`: no non-finite loss / gradient-norm / learning-rate record.
- `sft_pilot6k_seed1`: the recorded learning rate reproduces the configured linear-warmup + cosine schedule to a maximum relative deviation below the 1e-12 machine-precision floor; peak at step 310, consistent with the 300-step warmup boundary.
- `sft_pilot6k_seed1`: 1,051 s wall clock (0.29 h) at 5.71 steps/s.

## Behavioral / structural trajectory

The two full SFT runs have completed training, but their behavioral checkpoint grid has not been evaluated. Therefore, no full-run lambda/eta trajectory is available.

*Exploratory seed-1 pilot; test_goods validation; incomplete three-point grid; no frozen selector; not a full-run result.*

| pilot step | λ (SE) | η (SE) | d | consistency | keep-both | trade-both | W |
|---:|---|---|---:|---:|---:|---:|---:|
| 2,000 | -0.160 (0.015) | -0.466 (0.038) | 0.493 | 0.623 | 0.056 | 0.321 | 0.875 |
| 4,000 | +0.059 (0.014) | +0.012 (0.033) | 0.060 | 0.718 | 0.156 | 0.126 | 0.900 |
| 6,000 | +0.047 (0.014) | +0.023 (0.033) | 0.052 | 0.725 | 0.150 | 0.124 | 0.901 |

The July-27 evaluation logs record the adapter path as checkpoints/sft_qwen_delta_seed1/checkpoint-N. The pilot trained into that path on July 26 and was renamed to *_pilot6k before the full run reused the name on July 29, so those logs refer to the PILOT adapters. Directory mtimes and the pilot's own manifest (max_steps=6000) confirm the identity.

The pilot used a cosine schedule ending at 6,000. The full run used a cosine schedule ending at 30,000. Their step-6,000 weights are not interchangeable.

CPU-only. No GPU/PBS job, no inference, no checkpoint evaluation, no frozen selector run, and no frozen/untouched suite was opened to produce these artifacts.
