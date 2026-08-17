# SFT LoRA parameter localization atlas — Phase 1 (weight-space screening)

> **This is a screening analysis, not a causal result.** Everything here measures
> the *size and direction of the LoRA weight update* in parameter space. Nothing
> here shows that any layer, module or low-rank direction **causes** the measured
> SFT behavioral change. A large `||dW||` can sit in a direction the task inputs
> never excite; a small one can be decisive. Causal claims require the Phase-2
> necessity/sufficiency ablation, which has **not** been run.

## What this answers

The hypothesis under test is that the rapid SFT behavioral change is concentrated
in a small number of transformer layers, LoRA modules, or low-rank directions.
Phase 1 asks only the *weight-space* half of that question:

1. Is update magnitude concentrated in a few layers or module families?
2. Is the effective update rank substantially below the LoRA rank of 16?
3. Do the same modules dominate in both full SFT seeds?
4. How does the update change from 2k → selected checkpoint → 30k?
5. Is there a small, cross-seed-stable candidate set for a later causal ablation?

Read `sft_lora_summary.md` for the descriptive answers and the proposed (not
executed) Phase-2 shortlist.

## Scope — exactly six full-run adapters

| role | seed | step | path |
|---|---|---|---|
| first saved | 1 | 2,000 | `checkpoints/sft_qwen_delta_seed1/checkpoint-2000` |
| **frozen selected** | 1 | 4,000 | `checkpoints/sft_qwen_delta_seed1/checkpoint-4000` |
| endpoint | 1 | 30,000 | `checkpoints/sft_qwen_delta_seed1/checkpoint-30000` |
| first saved | 2 | 2,000 | `checkpoints/sft_qwen_delta_seed2/checkpoint-2000` |
| **frozen selected** | 2 | 6,000 | `checkpoints/sft_qwen_delta_seed2/checkpoint-6000` |
| endpoint | 2 | 30,000 | `checkpoints/sft_qwen_delta_seed2/checkpoint-30000` |

The historical pilot (`checkpoints/sft_qwen_delta_seed1_pilot6k`) and any `final`
adapter are **hard-rejected by path**. Every adapter's `adapter_model.safetensors`
and `adapter_config.json` SHA-256 is checked against the expected hash recorded in
`results/sft_grid_verification.json`; any mismatch in hash, path, seed, step, rank,
alpha, bias or target-module inventory aborts the refresh.

**No adapter exists before step 2,000** — the full runs saved nothing earlier — so
nothing in this directory says anything about the first 2,000 steps.

## LoRA mathematics

The effective update per adapted projection is

```
dW = (alpha / rank) * B @ A          alpha = 32, rank = 16, scale = 2
A: [16, in]      B: [out, 16]        dense dW: [out, in]
```

The scale is folded into `B` **exactly once**. Raw `||A||` and `||B||` are never
compared or ranked: the factorization has a gauge freedom (`A -> R A`,
`B -> B R^-1` for invertible `R`) under which the factor norms change freely while
`dW` does not. Only composite quantities are reported.

The dense `dW` is **never materialized** for any Qwen tensor. All of it comes from
at-most-16x16 algebra in float64:

| quantity | identity |
|---|---|
| `\|\|dW\|\|_F^2` | `trace[(B_s^T B_s)(A A^T)]`, `B_s = scale*B` |
| nonzero `sv(dW)` | `sv(R_B R_A^T)` with `B_s = Q_B R_B`, `A^T = Q_A R_A` |
| `<dW_1, dW_2>_F` | `trace[(B_1^T B_2)(A_2 A_1^T)]`, both scales included |
| `\|\|dW_1 - dW_2\|\|_F` | `sqrt(\|\|dW_1\|\|^2 + \|\|dW_2\|\|^2 - 2<dW_1,dW_2>)` |

Frozen base weights (`models/Qwen2.5-7B-Instruct/model.safetensors`, BF16) are read
tensor-by-tensor at their safetensors byte offsets and widened losslessly to
float32 — the model is **never instantiated** through Transformers. They are used
only as the denominator of relative-update metrics. Base `||W||_2` comes from a
deterministic randomized subspace iteration (seed 20260817, block 8, rel. tol
1e-7); `dW` singular values are exact.

## Reported metrics

**Magnitude** — `||dW||_F`, `||dW||_2`, `RMS = ||dW||_F/sqrt(out*in)`, base
`||W||_F` and `||W||_2`, relative Frobenius and spectral update, energy
(`||dW||_F^2`) share, cumulative energy rank.

**Rank** — all nonzero singular values, stable rank `||dW||_F^2/||dW||_2^2`,
entropy effective rank `exp(-sum p_j log p_j)` with `p_j = sv_j^2/sum sv^2`, energy
retained at rank 1/2/4/8/16, minimum rank for 90%/95%/99% of energy.

**Concentration** — top-1/5/10-module and top-5%/10%/25% energy shares, Gini over
the 196 module energies, shares by layer / module type / attention-vs-MLP.

Dimension-normalized metrics (`RMS`, relative Frobenius, and a `RMS^2`-weighted
analogue of every energy share) are reported **alongside** the raw norms
throughout, so the large MLP matrices cannot appear important purely because they
are large.

**Cross-checkpoint comparison** — composite-update cosine, difference norm,
relative norm change, change in effective rank and in energy share. Within each
seed: `2k → selected`, `selected → 30k`, `2k → 30k`. Across seeds, matched by
role: `s1@2k vs s2@2k`, `s1@4k vs s2@6k` (both selected), `s1@30k vs s2@30k`.
Aggregate cosines are reported both unweighted and energy-weighted
(`w_m = ||dW_a,m||_F * ||dW_b,m||_F`), so a high cosine on a negligible module
cannot drive the cross-seed conclusion.

## Files

| file | contents |
|---|---|
| `sft_lora_localization_snapshot.json` | canonical tracked snapshot — everything needed to regenerate the CSVs, Markdown and figures with no checkpoint or model directory present |
| `sft_lora_localization_manifest.json` | provenance: git commit, library versions, every adapter path/size/SHA-256 and its expected verification hash, base config/header/shard hashes, formula and metric definitions, dtype, and the explicit limitations |
| `sft_lora_module_stats.csv` | 6 x 196 rows — one per (checkpoint, layer, module), all magnitude/rank statistics plus the 16 singular values |
| `sft_lora_module_comparisons.csv` | 9 x 196 rows — one per (comparison, layer, module) |
| `sft_lora_summary.md` | descriptive answers to the nine questions + the proposed Phase-2 shortlist |
| `sft_lora_parameter_atlas.png` | layer x module heatmaps (relative Frobenius and dimension-normalized RMS), cumulative energy curves, depth profile, attention-vs-MLP and module-family shares, 2k/selected/30k comparison |
| `sft_lora_singular_energy.png` | singular spectra, within-module energy retention, effective rank by depth and module type, rank needed for 90/95/99% |
| `sft_lora_cross_seed_similarity.png` | cross-seed composite-update cosine heatmaps at selected and endpoint, unweighted vs energy-weighted aggregates, per-module energy-share scatter |

## Reproducing

```bash
# Snapshot-only: no checkpoints, no base model weights needed.
python eval/analyze_sft_lora_localization.py            # re-render CSV/MD/figures
python eval/analyze_sft_lora_localization.py --check    # verify deterministic rendering
python eval/test_sft_lora_localization.py               # synthetic unit tests

# Where the untracked adapters and base weights live (CPU-only, ~10 min):
python eval/analyze_sft_lora_localization.py --refresh
```

`--check` passes from a clean `git archive HEAD` extraction, which contains neither
`checkpoints/` nor `models/`.

## Compute

CPU only. No GPU, no PBS submission, no inference, no generation, no training, no
activation collection, no ablation. The frozen estimator
(`eval/core_exp_refactored.py`) is untouched, and none of the method-comparison,
frozen-unused, OOD or semantic-counterbalancing suites were read.

## Limitations

- **Weight-space screening only** — parameter-space magnitude is not causal
  importance, and this phase produces no necessity or sufficiency result.
- **No activation weighting.** An update direction that the actual task inputs
  never excite is counted the same as one that dominates the forward pass.
- **No behavioral evaluation in this phase.** No lambda/eta, no accuracy, no suite.
- **No sub-2k localization is possible** — the full runs saved no adapter before
  step 2,000, so the earliest behavioral movement cannot be localized here.
- **No method comparison.** Nothing here compares SFT to GRPO or to sign-only, and
  no method is declared a winner.
- **Rank 16 caps every effective rank by construction.** A low effective rank is
  low relative to that cap, not relative to the full weight-matrix rank.
- Base `||W||_2` is a deterministic iterative estimate, not an exact SVD.
- **Adapter-to-historical-prediction binding remains non-cryptographic.** The
  historical evaluator emitted no run manifest, so these adapters are tied to the
  published behavioral results by path/step convention, not by hash.
