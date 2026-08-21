# Phase-A training dynamics — optimization diagnostics

*Figures for the SFT effective-batch sensitivity experiment (`SFT_BATCH_LR_SENSITIVITY_PROTOCOL.md` + Amendment 1). CPU-only: no GPU job, no inference, no checkpoint evaluation.*

> **These are optimization diagnostics, not behavioural results.** Loss, gradient norm and learning rate describe the optimizer. λ, η, `d`, consistency and W are behavioural estimands computed after inference and live in `PHASE_A_REPORT.md`. A smoother loss curve is not evidence of less ownership dependence.

## Reading the axes

The x-axis is **prompts seen**, never optimizer steps. At the fixed 6,016-prompt exposure a batch-64 cell takes 94 updates and a batch-1 cell takes 6,016; a step axis would stretch the comparison by 64× and make the batches look like runs of different length.

Smoothing is a **causal trailing window of 512 prompts** — the same amount of *data* for every batch, which is 512 obs at eb1, 32 obs at eb16, 16 obs at eb32, 8 obs at eb64. Only current and preceding observations enter each point. Raw observations are drawn faintly beneath; the CSV keeps every unsmoothed value.

The gradient-norm panels smooth with a **median**, not a mean. That distribution is heavy-tailed — batch 1 has a median of 0.91 against a maximum of 310 — so a rolling mean plots batch 1 at ~25 and describes its tail rather than its typical update. Loss and learning rate use the mean.

## Gradient norm and clipping

| effective batch | updates/seed | median | P90 | P99 | max | clipped |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,016 | 0.900 | 90.07 | 131.59 | 381.5 | 61.0% |
| 16 | 376 | 2.618 | 9.90 | 13.33 | 15.9 | 100.0% |
| 32 | 188 | 6.433 | 10.51 | 12.21 | 14.3 | 100.0% |
| 64 | 94 | 9.101 | 10.57 | 11.80 | 12.9 | 100.0% |

**The tail collapses as the batch grows.** P99 falls from 132 at batch 1 to 13–12 at the large batches (~10×), and the maximum from 382 to ~16 (~25×). The **median rises** (0.9 → 9.1) because averaging removes the many near-zero single-example gradients as well as the extreme ones. That is variance reduction, and it is the clearest result of the experiment.

**Clipping becomes universal.** Batch 1 clips 61% of updates; batches 16, 32 and 64 clip **100%** — every single update. With `max_grad_norm = 0.1` and accumulated norms concentrated at 2.6–9.1, every large-batch update is rescaled to the same fixed length. Those cells are therefore **not "the same optimizer with a bigger batch"** — they take normalized-gradient steps of constant size. Any batch effect read from these figures is entangled with that.

A large pre-clipping norm is **not gradient explosion**. Batch 1's tail is heavy-tailed single-example gradients; **no run produced a non-finite loss, gradient or parameter** in any of the 12 cells.

## Loss and learning rate

The loss panels are not comparable point-for-point across batches: a batch-16 logged loss is a 16-example mean while a batch-1 logged loss is a single example, so batch 1 is dispersed by construction. Judge the smoothed level and trend, not the scatter.

Each cell runs its **own** cosine schedule over its own horizon (6,016 / 16 / 32 / 64 = 6,016 / 376 / 188 / 94 updates), with 5% warmup. On a prompts-seen axis the four schedules therefore trace the same shape, which is the point: the cells are matched on data, not on updates.

## Files

| file | contents |
|---|---|
| `dynamics_metrics.csv` | tidy per-update rows, unsmoothed |
| `dynamics_by_seed.png` | 3 panels, 12 cells, seed by linestyle |
| `dynamics_by_batch.png` | 3 panels, across-seed mean + min–max band |
| `dynamics_grad_norm_logscale.png` | uncropped log-scale tails |
| `dynamics_manifest.json` | sources, hashes, versions, clip stats |

## The extended-horizon figure

`dynamics_h32000_horizon.png` puts the 32,000-prompt run beside the 6,016-prompt probe it extends. Both are eb32 @ 1e-4, seed 1, on the same prompt ordering, so on a prompts-seen axis this is a like-for-like comparison of **horizon**.

The learning-rate panel is the one to read first: the probe's cosine peaks near 1,600 prompts and is back at zero by 6,016, while the 32,000 run is still near its peak at 6,016 and decays for another 26,000 prompts. That is the concrete form of the protocol's rule that runs with different horizons are not interchangeable — at 6,016 prompts these two models have had entirely different learning-rate histories, so neither is a checkpoint of the other.

The loss panels overlap closely through the first ~6,000 prompts, as they should given identical data and hyper-parameters, then diverge: the longer run keeps descending to ~0.02 while the probe has already stopped. The 32,000 run is also the first in the experiment whose gradient curve reaches the 0.1 clipping line at all — 1.7% of its updates fall at or below it, against 100% clipped everywhere else.

Phase-A and Phase-B figures are deliberately restricted to the 6,016-prompt window. Letting the 32,000 run into them would compress every other curve into the left fifth of the axis.

## Reproduce

```bash
python eval/plot_sft_sensitivity_dynamics.py --refresh   # trainer states -> CSV
python eval/plot_sft_sensitivity_dynamics.py             # CSV -> figures
python eval/plot_sft_sensitivity_dynamics.py --check     # verify agreement
```

Rendering and checking work from the tracked CSV alone; `--refresh` needs the gitignored `checkpoints/sft_sensitivity/` trainer states.
