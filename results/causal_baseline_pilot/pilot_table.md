# Causal-baseline PILOT — exploratory results (seed 1, test_goods VALIDATION)

*Generated 2026-07 from the 6k pilots. EXPLORATORY: 1 seed, incomplete 3-point
grid, `test_goods` = validation. NO frozen selector run, NO winner declared.
Confirmatory claims require the full 2-seed × 30k runs and a new untouched suite
(see CAUSAL_BASELINE_PROTOCOL.md).*

Reference — matched local base: λ=7.637 (SE 0.627), η=1.007, d=7.70,
consistency=0.008, W=0.744.  N=9,890 test_goods cases per checkpoint.

| method | ckpt | λ (SE) | η (SE) | d | consist | keep | trade | W | cond(J) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| SFT  | 2000 | −0.160 (0.015) | −0.466 (0.038) | 0.493 | 0.623 | 0.056 | 0.321 | 0.875 | 5.2e2 |
| SFT  | 4000 | +0.059 (0.014) | +0.012 (0.033) | 0.060 | 0.718 | 0.156 | 0.126 | 0.900 | 1.1e2 |
| SFT  | 6000 | +0.047 (0.014) | +0.023 (0.033) | 0.052 | 0.725 | 0.150 | 0.124 | 0.901 | 1.2e2 |
| sign | 2000 | +0.147 (0.018) | +0.062 (0.036) | 0.160 | 0.664 | 0.217 | 0.119 | 0.885 | 7.4e1 |
| sign | 4000 | +0.558 (0.024) | −0.064 (0.041) | 0.562 | 0.633 | 0.314 | 0.053 | 0.887 | 5.7e1 |
| sign | 6000 | +0.388 (0.021) | +0.039 (0.038) | 0.390 | 0.660 | 0.275 | 0.064 | 0.891 | 6.4e1 |

Runtime (6k steps, 1×A100): **SFT ≈ 17.5 min** (~0.18 s/step); **sign-only ≈ 8.7 h**
(~5.24 s/step). Baseline checkpoint eval ≈ 0.5–1 h each on test_goods.

Both methods collapse λ from the base 7.637 to near 0 with non-degenerate
choice behaviour (no keep-both/trade-both collapse) and well-conditioned NLS
fits (cond(J) 57–522). SFT reaches d≈0.05 by 4k–6k; sign-only is noisier and
higher-λ at matched steps. These are hypotheses for the full runs, not results.
