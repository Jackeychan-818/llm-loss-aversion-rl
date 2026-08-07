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

A complete 2-seed x 15-checkpoint SFT grid was located and every cell passed per-checkpoint identity and completeness verification; it is plotted read-only from the recorded verification manifest.

*Full 2-seed x 15-checkpoint SFT grid, every cell verified; test_goods VALIDATION estimates used for checkpoint selection — not final-test performance.*

Identity and completeness verified per checkpoint by `eval/verify_sft_grid.py` → `results/sft_grid_verification.json`: adapter hash, both perspectives, N = 9,890 paired cases, zero parse failures, a T=1 Model-A CSV, and finite estimates with positive standard errors. A `Model_1/` directory alone is not a completion test.

Comparator — matched local base under the **same plain `baseline` prompt** (same rows, scorer and estimator; only training differs): λ = 7.637 (SE 0.627), η = 1.007 (SE 0.120), consistency = 0.009, keep-both = 0.991, trade-both = 0.000, W = 0.744.

| seed | step | λ (SE) | η (SE) | consistency | keep-both | trade-both | W |
|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 2,000 | -0.237 (0.015) | -0.701 (0.040) | 0.538 | 0.032 | 0.431 | 0.860 |
| 1 | 4,000 | -0.034 (0.012) | +0.027 (0.031) | 0.730 | 0.122 | 0.148 | 0.904 |
| 1 | 6,000 | +0.274 (0.015) | -0.043 (0.031) | 0.749 | 0.182 | 0.069 | 0.921 |
| 1 | 8,000 | +0.361 (0.014) | -0.137 (0.030) | 0.774 | 0.176 | 0.051 | 0.934 |
| 1 | 10,000 | +0.102 (0.009) | -0.546 (0.027) | 0.820 | 0.055 | 0.125 | 0.948 |
| 1 | 12,000 | +0.042 (0.009) | -0.719 (0.028) | 0.813 | 0.032 | 0.156 | 0.955 |
| 1 | 14,000 | +0.172 (0.009) | -0.455 (0.027) | 0.848 | 0.070 | 0.083 | 0.962 |
| 1 | 16,000 | +0.232 (0.010) | -0.489 (0.027) | 0.853 | 0.078 | 0.069 | 0.966 |
| 1 | 18,000 | +0.093 (0.008) | -0.674 (0.028) | 0.848 | 0.038 | 0.114 | 0.968 |
| 1 | 20,000 | +0.026 (0.007) | -0.574 (0.028) | 0.849 | 0.033 | 0.118 | 0.970 |
| 1 | 22,000 | +0.142 (0.008) | -0.628 (0.029) | 0.862 | 0.048 | 0.090 | 0.972 |
| 1 | 24,000 | +0.314 (0.009) | -0.592 (0.028) | 0.864 | 0.082 | 0.054 | 0.973 |
| 1 | 26,000 | +0.233 (0.009) | -0.582 (0.028) | 0.866 | 0.067 | 0.067 | 0.973 |
| 1 | 28,000 | +0.219 (0.009) | -0.597 (0.028) | 0.866 | 0.063 | 0.072 | 0.973 |
| 1 | 30,000 | +0.206 (0.008) | -0.596 (0.028) | 0.867 | 0.061 | 0.072 | 0.973 |
| 2 | 2,000 | +0.098 (0.017) | +0.280 (0.035) | 0.654 | 0.241 | 0.106 | 0.883 |
| 2 | 4,000 | +0.054 (0.014) | +0.312 (0.032) | 0.717 | 0.196 | 0.087 | 0.901 |
| 2 | 6,000 | -0.089 (0.011) | -0.143 (0.030) | 0.767 | 0.068 | 0.166 | 0.917 |
| 2 | 8,000 | -0.061 (0.010) | -0.355 (0.028) | 0.791 | 0.045 | 0.165 | 0.933 |
| 2 | 10,000 | +0.251 (0.011) | -0.319 (0.027) | 0.823 | 0.108 | 0.069 | 0.947 |
| 2 | 12,000 | +0.244 (0.010) | -0.471 (0.027) | 0.842 | 0.083 | 0.075 | 0.956 |
| 2 | 14,000 | +0.279 (0.011) | -0.310 (0.027) | 0.842 | 0.105 | 0.052 | 0.961 |
| 2 | 16,000 | +0.215 (0.009) | -0.377 (0.027) | 0.856 | 0.084 | 0.061 | 0.967 |
| 2 | 18,000 | +0.288 (0.010) | -0.373 (0.027) | 0.860 | 0.094 | 0.046 | 0.969 |
| 2 | 20,000 | +0.109 (0.007) | -0.436 (0.026) | 0.868 | 0.054 | 0.078 | 0.972 |
| 2 | 22,000 | +0.151 (0.008) | -0.443 (0.026) | 0.872 | 0.061 | 0.067 | 0.973 |
| 2 | 24,000 | +0.089 (0.007) | -0.519 (0.026) | 0.868 | 0.045 | 0.087 | 0.973 |
| 2 | 26,000 | +0.176 (0.008) | -0.391 (0.026) | 0.872 | 0.070 | 0.057 | 0.974 |
| 2 | 28,000 | +0.167 (0.008) | -0.385 (0.026) | 0.875 | 0.068 | 0.058 | 0.974 |
| 2 | 30,000 | +0.139 (0.007) | -0.406 (0.026) | 0.874 | 0.061 | 0.064 | 0.974 |

Frozen checkpoint selection: seed 1 → step 4,000, seed 2 → step 6,000. This table reports the trajectory only; nothing here selects a checkpoint.

No SFT-versus-GRPO winner is declared or plotted. This figure shows SFT against its own matched plain-prompt base only; GRPO outputs are not read by this script (see module docstring), and a cross-method claim would additionally require the untouched method-comparison suite under METHOD_COMPARISON_PROTOCOL.md.

The pilot used a cosine schedule ending at 6,000. The full run used a cosine schedule ending at 30,000. Their step-6,000 weights are not interchangeable.

CPU-only rendering. This script ran no GPU/PBS job, no inference and no checkpoint evaluation, and opened no frozen or untouched suite. It also selects nothing: where a frozen checkpoint selection is shown it is READ from manifests written earlier by eval/select_checkpoint.py, and this script neither re-ranks nor breaks a tie. The underlying checkpoint evaluations were produced by separate GPU jobs (train/submit_eval_baseline_ckpt.pbs).
