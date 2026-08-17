# SFT LoRA parameter localization — weight-space screening (Phase 1)

> **Weight-space screening only. No causal importance is implied.** Full-run SFT adapters only — the historical pilot is excluded. No inference, generation, training, activation collection or ablation was run to produce this document, and no frozen or untouched evaluation suite was opened.

Update definition: `dW = (alpha/rank) * B @ A` with alpha=32, rank=16, scale=2. The dense `dW` is never formed; norms and singular values come from 16x16 Gram/QR/SVD identities. Raw `A`/`B` norms are never used as importance statistics (gauge freedom `A->RA, B->BR^-1`).

## Headline reading

The screening does **not** support a simple 'a few modules / a few directions' picture. Four descriptive findings, each developed below:

1. **Update energy is broadly spread, not concentrated.** At the selected checkpoints the single largest of 196 modules holds 2.0% / 1.7% of energy and the top 10 hold 17.4% / 15.4%. The raw Gini (0.55 / 0.55) is driven largely by matrix SIZE: once updates are divided by sqrt(out*in) the Gini falls to 0.22 / 0.19, i.e. close to an even per-element update across the whole adapted network.
2. **The effective rank is NOT far below 16.** Median entropy effective rank is 5.6 of the rank-16 cap, and the median module needs rank 7 for 90% of its energy. Only a small minority of modules are near rank-1.
3. **The two seeds agree on WHERE but not on WHICH DIRECTION.** The per-module energy ordering is almost identical across seeds (Spearman rho 0.98), yet the matched-role composite-update cosine is only +0.014. Same places, essentially unrelated directions.
4. **Depth shifts with training, and rank structure is depth-structured.** At 2k the update is concentrated late (layers 19-27 hold ~49% of energy); by the selected checkpoint and 30k it has spread toward a roughly flat depth profile. Independently, mean effective rank falls monotonically with depth — 7.7 in layers 0-9 down to 2.8 in layers 24-27, in both seeds — the one clearly reproducible low-rank structure the screening finds.

Read together, these argue that a Phase-2 ablation should be organized around coarse, pre-registered partitions (module family, depth block, rank truncation) with parameter-matched controls — **not** around a handful of individually 'important' modules, which this screening does not identify.

## 1. How many exact trainable LoRA values are present?

**40,370,176 trainable LoRA values per adapter**, identical across all six checkpoints: 28 layers x 7 modules = 196 adapted projections, each contributing `r*in + out*r` values at r=16.

| module | A shape | B shape | dense dW shape | trainable values | x28 layers |
|---|---|---|---|---|---|
| `q_proj` | [16, 3584] | [3584, 16] | [3584, 3584] | 114,688 | 3,211,264 |
| `k_proj` | [16, 3584] | [512, 16] | [512, 3584] | 65,536 | 1,835,008 |
| `v_proj` | [16, 3584] | [512, 16] | [512, 3584] | 65,536 | 1,835,008 |
| `o_proj` | [16, 3584] | [3584, 16] | [3584, 3584] | 114,688 | 3,211,264 |
| `gate_proj` | [16, 3584] | [18944, 16] | [18944, 3584] | 360,448 | 10,092,544 |
| `up_proj` | [16, 3584] | [18944, 16] | [18944, 3584] | 360,448 | 10,092,544 |
| `down_proj` | [16, 18944] | [3584, 16] | [3584, 18944] | 360,448 | 10,092,544 |
| **per layer** | | | | **1,441,792** | **40,370,176** |

## 2. How concentrated is update energy?

Energy = `||dW||_F^2`, shares over the 196 adapted modules of one adapter.

| checkpoint | role | total `||dW||_F` | top-1 | top-5 | top-10 | top-5% | top-10% | top-25% | Gini |
|---|---|---|---|---|---|---|---|---|---|
| seed 1 @ 2000 | first_saved | 0.5814 | 0.0272 | 0.1216 | 0.2234 | 0.2234 | 0.3956 | 0.6861 | 0.5926 |
| seed 1 @ 4000 | selected | 0.6674 | 0.0201 | 0.0934 | 0.1744 | 0.1744 | 0.3210 | 0.6731 | 0.5546 |
| seed 1 @ 30000 | endpoint | 1.3822 | 0.0205 | 0.0940 | 0.1777 | 0.1777 | 0.3330 | 0.6992 | 0.5747 |
| seed 2 @ 2000 | first_saved | 0.5421 | 0.0249 | 0.1178 | 0.2188 | 0.2188 | 0.3935 | 0.6840 | 0.5894 |
| seed 2 @ 6000 | selected | 0.8174 | 0.0173 | 0.0809 | 0.1543 | 0.1543 | 0.2979 | 0.6771 | 0.5468 |
| seed 2 @ 30000 | endpoint | 1.3864 | 0.0222 | 0.0935 | 0.1769 | 0.1769 | 0.3309 | 0.6984 | 0.5744 |

Dimension-normalized counterpart (energy of `delta_rms = ||dW||_F/sqrt(out*in)`), which removes the mechanical advantage of the large MLP matrices:

| checkpoint | norm top-1 | norm top-5 | norm top-10 | norm top-10% | norm Gini |
|---|---|---|---|---|---|
| seed 1 @ 2000 | 0.0142 | 0.0609 | 0.1162 | 0.2168 | 0.3219 |
| seed 1 @ 4000 | 0.0106 | 0.0481 | 0.0899 | 0.1670 | 0.2213 |
| seed 1 @ 30000 | 0.0201 | 0.0590 | 0.1021 | 0.1780 | 0.2210 |
| seed 2 @ 2000 | 0.0126 | 0.0572 | 0.1094 | 0.2057 | 0.3074 |
| seed 2 @ 6000 | 0.0157 | 0.0533 | 0.0903 | 0.1593 | 0.1916 |
| seed 2 @ 30000 | 0.0264 | 0.0693 | 0.1102 | 0.1846 | 0.2260 |

## 3. Which module families carry the largest raw and normalized updates?

Raw energy share by module type (each row sums to 1 across the seven modules):

| checkpoint | `q` | `k` | `v` | `o` | `gate` | `up` | `down` | attention | MLP |
|---|---|---|---|---|---|---|---|---|---|
| seed 1 @ 2000 | 0.0655 | 0.0079 | 0.0102 | 0.0825 | 0.3285 | 0.3871 | 0.1182 | 0.1662 | 0.8338 |
| seed 1 @ 4000 | 0.0699 | 0.0085 | 0.0100 | 0.0794 | 0.3424 | 0.3817 | 0.1080 | 0.1678 | 0.8322 |
| seed 1 @ 30000 | 0.0733 | 0.0091 | 0.0086 | 0.0630 | 0.4136 | 0.3528 | 0.0797 | 0.1539 | 0.8461 |
| seed 2 @ 2000 | 0.0662 | 0.0080 | 0.0101 | 0.0807 | 0.3304 | 0.3856 | 0.1191 | 0.1649 | 0.8351 |
| seed 2 @ 6000 | 0.0743 | 0.0089 | 0.0096 | 0.0727 | 0.3612 | 0.3745 | 0.0988 | 0.1655 | 0.8345 |
| seed 2 @ 30000 | 0.0778 | 0.0091 | 0.0085 | 0.0616 | 0.4109 | 0.3516 | 0.0805 | 0.1570 | 0.8430 |

Median dimension-normalized update `delta_rms` (x1e-4) by module type:

| checkpoint | `q` | `k` | `v` | `o` | `gate` | `up` | `down` |
|---|---|---|---|---|---|---|---|
| seed 1 @ 2000 | 0.071 | 0.066 | 0.079 | 0.084 | 0.073 | 0.078 | 0.045 |
| seed 1 @ 4000 | 0.092 | 0.086 | 0.095 | 0.102 | 0.091 | 0.096 | 0.052 |
| seed 1 @ 30000 | 0.181 | 0.178 | 0.179 | 0.187 | 0.206 | 0.190 | 0.092 |
| seed 2 @ 2000 | 0.066 | 0.062 | 0.074 | 0.079 | 0.068 | 0.074 | 0.041 |
| seed 2 @ 6000 | 0.113 | 0.106 | 0.112 | 0.117 | 0.114 | 0.117 | 0.061 |
| seed 2 @ 30000 | 0.184 | 0.177 | 0.177 | 0.184 | 0.206 | 0.188 | 0.094 |

Median relative Frobenius update `||dW||_F/||W||_F` (x1e-4) by module type:

| checkpoint | `q` | `k` | `v` | `o` | `gate` | `up` | `down` |
|---|---|---|---|---|---|---|---|
| seed 1 @ 2000 | 3.969 | 3.290 | 4.691 | 5.399 | 4.363 | 4.775 | 2.749 |
| seed 1 @ 4000 | 5.314 | 4.410 | 5.621 | 6.166 | 5.417 | 5.792 | 3.156 |
| seed 1 @ 30000 | 10.664 | 9.577 | 10.452 | 11.445 | 11.907 | 11.961 | 5.809 |
| seed 2 @ 2000 | 3.779 | 3.151 | 4.369 | 4.921 | 4.075 | 4.440 | 2.548 |
| seed 2 @ 6000 | 6.516 | 5.796 | 6.809 | 7.452 | 6.858 | 7.129 | 3.708 |
| seed 2 @ 30000 | 10.928 | 9.359 | 10.249 | 11.393 | 11.896 | 11.993 | 5.702 |

## 4. Are selected-checkpoint updates early, middle or late in depth?

Energy share by depth third (layers 0-9 / 10-18 / 19-27), and the three highest-energy individual layers:

| checkpoint | early 0-9 | middle 10-18 | late 19-27 | top layers (raw energy) |
|---|---|---|---|---|
| seed 1 @ 2000 | 0.1828 | 0.3255 | 0.4917 | L25 (0.066), L24 (0.066), L26 (0.060) |
| seed 1 @ 4000 | 0.2741 | 0.3472 | 0.3787 | L24 (0.051), L25 (0.049), L23 (0.046) |
| seed 1 @ 30000 | 0.3618 | 0.2862 | 0.3520 | L23 (0.044), L21 (0.044), L20 (0.044) |
| seed 2 @ 2000 | 0.1911 | 0.3162 | 0.4927 | L25 (0.062), L23 (0.061), L24 (0.060) |
| seed 2 @ 6000 | 0.3222 | 0.3368 | 0.3410 | L23 (0.044), L19 (0.042), L24 (0.041) |
| seed 2 @ 30000 | 0.3622 | 0.2855 | 0.3523 | L23 (0.046), L20 (0.042), L21 (0.042) |

Same split under the dimension-normalized `delta_rms^2` weighting:

| checkpoint | early 0-9 | middle 10-18 | late 19-27 |
|---|---|---|---|
| seed 1 @ 2000 | 0.1871 | 0.3232 | 0.4898 |
| seed 1 @ 4000 | 0.2800 | 0.3492 | 0.3707 |
| seed 1 @ 30000 | 0.3481 | 0.2948 | 0.3571 |
| seed 2 @ 2000 | 0.2013 | 0.3133 | 0.4854 |
| seed 2 @ 6000 | 0.3291 | 0.3386 | 0.3323 |
| seed 2 @ 30000 | 0.3536 | 0.2890 | 0.3574 |

Top-10 modules by raw energy share at each selected checkpoint:

- **seed 1 @ 4000**: L25.up (0.0201), L24.gate (0.0188), L24.up (0.0187), L19.up (0.0181), L26.up (0.0177), L23.up (0.0175), L18.up (0.0165), L23.gate (0.0159), L20.up (0.0156), L26.gate (0.0154)
- **seed 2 @ 6000**: L23.gate (0.0173), L23.up (0.0171), L19.up (0.0161), L24.gate (0.0153), L20.gate (0.0150), L24.up (0.0149), L18.up (0.0147), L25.up (0.0147), L26.up (0.0147), L25.gate (0.0146)

Top-10 modules by dimension-normalized `delta_rms` at each selected checkpoint:

- **seed 1 @ 4000**: L0.q (1.267e-05), L25.q (1.234e-05), L18.q (1.192e-05), L0.k (1.190e-05), L25.up (1.149e-05), L19.o (1.135e-05), L23.o (1.131e-05), L25.k (1.123e-05), L24.o (1.122e-05), L24.gate (1.110e-05)
- **seed 2 @ 6000**: L0.q (1.884e-05), L0.k (1.565e-05), L27.q (1.528e-05), L18.q (1.382e-05), L25.q (1.344e-05), L25.k (1.317e-05), L23.gate (1.307e-05), L23.up (1.298e-05), L20.q (1.281e-05), L19.o (1.265e-05)

## 5. What ranks retain 90% / 95% / 99% of update energy?

Per-module singular spectra of `dW` (exact; capped at rank 16 by construction). Values are medians across the 196 modules, with the [min, max] range.

| checkpoint | eff. rank (entropy) | stable rank | rank@90% | rank@95% | rank@99% | top-1 energy | top-4 energy |
|---|---|---|---|---|---|---|---|
| seed 1 @ 2000 | 5.49 [1.00452–12.0293] | 2.00 | 7 [1, 11] | 10 [1, 14] | 14 [1, 16] | 0.5012 | 0.8301 |
| seed 1 @ 4000 | 5.57 [1.03094–12.7375] | 1.93 | 7 [1, 12] | 10 [1, 14] | 14 [1, 16] | 0.5175 | 0.8191 |
| seed 1 @ 30000 | 4.61 [1.3596–12.4486] | 1.69 | 7 [1, 12] | 10 [2, 14] | 14 [4, 16] | 0.5918 | 0.8347 |
| seed 2 @ 2000 | 5.65 [1.00528–12.7664] | 1.97 | 7 [1, 12] | 10 [1, 14] | 14 [1, 16] | 0.5075 | 0.8138 |
| seed 2 @ 6000 | 5.59 [1.09613–13.253] | 1.89 | 7 [1, 12] | 10 [1, 14] | 14 [2, 16] | 0.5295 | 0.8097 |
| seed 2 @ 30000 | 4.95 [1.53978–13.0043] | 1.76 | 7 [1, 12] | 10 [2, 14] | 14 [4, 16] | 0.5681 | 0.8395 |

Fraction of modules whose 90% energy fits in rank <= k:

| checkpoint | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| seed 1 @ 2000 | 0.082 | 0.158 | 0.306 | 0.719 | 1.000 |
| seed 1 @ 4000 | 0.066 | 0.158 | 0.286 | 0.658 | 1.000 |
| seed 1 @ 30000 | 0.031 | 0.117 | 0.357 | 0.760 | 1.000 |
| seed 2 @ 2000 | 0.041 | 0.153 | 0.296 | 0.689 | 1.000 |
| seed 2 @ 6000 | 0.031 | 0.163 | 0.301 | 0.668 | 1.000 |
| seed 2 @ 30000 | 0.026 | 0.138 | 0.332 | 0.750 | 1.000 |

**Effective rank falls sharply with depth.** Averaging effective rank over the seven modules of each layer at the selected checkpoints:

| layer band | seed 1 mean eff. rank | seed 2 mean eff. rank |
|---|---|---|
| 0-9 (early) | 7.69 | 7.44 |
| 10-17 (middle) | 6.29 | 6.18 |
| 18-23 (late) | 3.59 | 3.45 |
| 24-27 (final) | 2.84 | 2.92 |

Both seeds show the same monotone decline: mean effective rank falls from 7.4-7.7 in layers 0-9 to 2.8-2.9 in layers 24-27. So the update is closest to genuinely low-rank exactly where it is *not* largest at the selected checkpoint. This is the clearest depth-structured signal in the screening and it is what makes the Tier-4 rank-truncation arm worth running: a rank-2 truncation should be nearly lossless late and lossy early, which is a testable prediction rather than a magnitude ranking.

Effective rank also differs by module type — `k_proj` is the highest (mean 8.6) and `down_proj` the lowest (4.3) — but note that `k_proj` is one of the smallest matrices and one of the smallest energy contributors, so this is a statement about the shape of its update, not its importance. Full ordering: `k` 8.6, `q` 6.1, `v` 6.0, `gate` 5.4, `o` 4.7, `up` 4.4, `down` 4.3.

## 6. Which patterns reproduce across both seeds?

Composite-update cosine between matched roles. `w_m = ||dW_a,m||_F * ||dW_b,m||_F` keeps a high cosine on a negligible module from dominating.

| comparison | mean cosine (unweighted) | mean cosine (energy-weighted) | median | min | max | attention (w) | MLP (w) |
|---|---|---|---|---|---|---|---|
| cross-seed first_saved (s1@2k vs s2@2k) | +0.0095 | +0.0125 | +0.0077 | -0.0248 | +0.0474 | +0.0108 | +0.0128 |
| cross-seed selected (s1@4k vs s2@6k) | +0.0125 | +0.0142 | +0.0106 | -0.0192 | +0.0545 | +0.0126 | +0.0146 |
| cross-seed endpoint (s1@30k vs s2@30k) | +0.0409 | +0.0563 | +0.0299 | -0.0027 | +0.2293 | +0.0323 | +0.0607 |

Reference scale for those cosines — the composite cosine between two **independent** random rank-16 updates of the same dense shape (200 draws per shape, fixed seed). This is a scale, not a significance test:

| dense shape | modules | null mean cosine | null sd | null p95 \|cos\| | null max \|cos\| |
|---|---|---|---|---|---|
| 18944x3584 | `gate`, `up` | +0.00000 | 0.00012 | 0.00022 | 0.00033 |
| 3584x18944 | `down` | +0.00001 | 0.00012 | 0.00023 | 0.00034 |
| 3584x3584 | `o`, `q` | -0.00002 | 0.00027 | 0.00063 | 0.00075 |
| 512x3584 | `k`, `v` | +0.00001 | 0.00073 | 0.00147 | 0.00192 |

The observed cross-seed cosines (+0.0142 at selection, +0.0563 at 30k) are **above** this null (largest null p95 |cos| = 0.0015) and consistently positive rather than sign-random, so the agreement is real — but it is small in absolute terms. Two seeds trained on the same objective end up with update directions that are, per module, close to orthogonal.

Rank correlation of the per-module energy ordering between seeds at the selected checkpoints, and the overlap of their top-k module sets:

- Spearman rho (raw energy share, seed1@4000 vs seed2@6000): **0.9822**
- Spearman rho (dimension-normalized `delta_rms`): **0.8874**
- top-5 raw-energy module overlap: **2/5**
- top-10 raw-energy module overlap: **8/10**
- top-20 raw-energy module overlap: **17/20**
- top-49 raw-energy module overlap: **48/49**

**Reconciling the two numbers.** These say different things and both matter. The near-perfect rank correlation means the seeds allocate update *magnitude* to the same modules — that pattern is a reproducible property of the task and architecture. The near-zero cosine means they do so along *different directions* within each module. So 'the same modules dominate both seeds' is true only in the magnitude sense. A Phase-2 group chosen from this table is therefore reproducible as a **set of locations**; nothing here licenses treating the two seeds' updates as the same learned solution, and any ablation must be run per seed rather than pooled.

## 7. What is already present at 2k, and 8. what keeps changing to 30k?

| comparison | mean cosine (unweighted) | mean cosine (energy-weighted) | total `||dW_a - dW_b||_F` | mean relative norm change | mean d(eff. rank) |
|---|---|---|---|---|---|
| seed1 2k->selected(4000) | +0.9061 | +0.9202 | 0.2772 | +0.2153 | +0.143 |
| seed1 selected(4000)->30k | +0.7016 | +0.6903 | 1.0487 | +1.0379 | -0.519 |
| seed1 2k->30k | +0.5148 | +0.4999 | 1.2145 | +1.5102 | -0.376 |
| seed2 2k->selected(6000) | +0.8420 | +0.8513 | 0.4720 | +0.6032 | -0.017 |
| seed2 selected(6000)->30k | +0.8013 | +0.8000 | 0.8881 | +0.6558 | -0.371 |
| seed2 2k->30k | +0.5182 | +0.5038 | 1.2186 | +1.6781 | -0.387 |

Total update magnitude by checkpoint (how much of the endpoint's norm and energy geometry already exists at 2k):

| seed | 2k `||dW||_F` | selected `||dW||_F` | 30k `||dW||_F` | 2k/30k norm ratio | 2k/selected norm ratio |
|---|---|---|---|---|---|
| 1 | 0.5814 | 0.6674 | 1.3822 | 0.4207 | 0.8713 |
| 2 | 0.5421 | 0.8174 | 1.3864 | 0.3910 | 0.6632 |

Modules whose direction moves most between the selected checkpoint and 30k (lowest composite cosine among the 25 highest-energy modules at selection):

- **seed 1**: L25.gate (cos +0.582), L23.gate (cos +0.598), L22.up (cos +0.604), L20.gate (cos +0.608), L21.up (cos +0.609), L26.gate (cos +0.610)
- **seed 2**: L22.up (cos +0.714), L21.up (cos +0.741), L24.gate (cos +0.744), L20.up (cos +0.744), L24.up (cos +0.747), L19.up (cos +0.754)

## 9. Does the evidence justify proceeding to causal ablation?

Descriptively, at the two frozen-selected checkpoints:

- Raw update energy is **spread**: the top 10 of 196 modules hold 17.4% (seed 1) and 15.4% (seed 2); the top 10% hold 32.1% / 29.8%; Gini 0.555 / 0.547.
- Under dimension normalization the picture is different: normalized top-10% share 16.7% / 15.9% (Gini 0.221 / 0.192).
- Median per-module entropy effective rank is **5.58 of the rank-16 cap**, so the update is not collapsing to a single direction inside each module.
- Matched-role cross-seed agreement at selection: energy-weighted mean composite cosine **+0.0142** (unweighted +0.0125).

**Verdict on the original hypothesis.** The hypothesis was that the rapid SFT behavioral change is concentrated in a small number of layers, modules or low-rank directions. In weight space, the screening finds **no such concentration**: energy is spread across depth and across the seven module types roughly in proportion to matrix size, and the within-module effective rank sits near the middle of the rank-16 cap rather than near 1. That is a genuine negative for the weight-space form of the hypothesis.

**What this does and does not license.** These are magnitudes and directions in weight space. They say where the optimizer moved, not which movement matters for the measured behavioral change. Fast behavioral movement early in training does not imply that few parameters carry it, and a large `||dW||` does not imply causal importance — an update in a direction the task inputs never excite contributes nothing. The converse also holds: a diffuse weight update is **not** evidence that the behavioral change is diffuse, because a small number of directions could still carry all of the behaviorally relevant effect. Only Phase 2 can separate these. The screening is informative enough to *define* a Phase-2 ablation design with a small number of pre-registered groups; it cannot substitute for one.

## Proposed Phase-2 shortlist (NOT executed)

Hierarchical, cross-seed-stable groups for a later necessity (zero-out) / sufficiency (keep-only) ablation. Nothing below has been run.

**Tier 1 — module-family partitions** (coarse, unambiguous, no selection on per-module magnitude):

1. `attention-only` — keep LoRA on q/k/v/o, zero gate/up/down.
2. `mlp-only` — keep LoRA on gate/up/down, zero q/k/v/o.
3. `qkv-only` vs `o_proj-only` — splits attention into what the block reads versus how it writes back.

**Tier 2 — depth blocks** (early 0-9 / middle 10-18 / late 19-27), each kept-only and each zeroed, so necessity and sufficiency are separated.

**Tier 3 — cross-seed-stable module group.** Modules ranked by the *worse* of the two seeds' standing under BOTH raw energy share and dimension-normalized `delta_rms`, so nothing enters because it is large in one seed only. Note the final column: these locations are stable in magnitude but NOT in direction, so this group must be ablated **per seed**, never pooled, and it is the weakest tier here precisely because the concentration it presumes is not present in the data:

| # | layer | module | family | worse energy rank | worse normalized rank | min energy share | min `delta_rms` | cross-seed cosine |
|---|---|---|---|---|---|---|---|---|
| 1 | 19 | `up_proj` | mlp | 4 | 13 | 0.0161 | 1.090e-05 | +0.020 |
| 2 | 23 | `up_proj` | mlp | 6 | 18 | 0.0171 | 1.071e-05 | +0.010 |
| 3 | 24 | `gate_proj` | mlp | 4 | 22 | 0.0153 | 1.110e-05 | +0.031 |
| 4 | 24 | `up_proj` | mlp | 6 | 24 | 0.0149 | 1.109e-05 | +0.027 |
| 5 | 18 | `up_proj` | mlp | 7 | 29 | 0.0147 | 1.040e-05 | +0.002 |
| 6 | 25 | `up_proj` | mlp | 8 | 29 | 0.0147 | 1.149e-05 | +0.022 |
| 7 | 26 | `up_proj` | mlp | 9 | 30 | 0.0147 | 1.077e-05 | +0.003 |
| 8 | 23 | `gate_proj` | mlp | 8 | 33 | 0.0159 | 1.022e-05 | +0.028 |
| 9 | 20 | `up_proj` | mlp | 12 | 39 | 0.0145 | 1.013e-05 | +0.002 |
| 10 | 21 | `up_proj` | mlp | 17 | 45 | 0.0143 | 1.006e-05 | +0.004 |
| 11 | 14 | `up_proj` | mlp | 15 | 50 | 0.0144 | 9.929e-06 | +0.006 |
| 12 | 13 | `up_proj` | mlp | 20 | 50 | 0.0141 | 9.931e-06 | +0.004 |
| 13 | 20 | `gate_proj` | mlp | 14 | 51 | 0.0149 | 9.900e-06 | +0.018 |
| 14 | 25 | `gate_proj` | mlp | 15 | 52 | 0.0146 | 9.897e-06 | +0.008 |
| 15 | 19 | `gate_proj` | mlp | 17 | 58 | 0.0143 | 9.669e-06 | +0.001 |

**Tier 4 — controls that must accompany every group above:**

- A **random module group matched on trainable-parameter count** (and separately on dense-element count), resampled over several draws — without it, any effect of a chosen group is confounded with simply removing that many parameters.
- **Rank truncation** of every retained module to rank 1 / 2 / 4 / 8 (project `dW` onto its top-k singular directions), which tests the low-rank claim directly rather than by proxy. Given the depth-dependent effective rank above, run this both uniformly and split by depth band — the pre-registered prediction is that a rank-2 truncation costs little in layers 24-27 and much more in layers 0-17.
- A **full-adapter** arm and a **zero-adapter** (base) arm to bracket the range.

Each arm must be run for **both seeds**, evaluated on the same behavioral instrument, and read against the base and full-adapter brackets. Group selection must be frozen before any behavioral number is looked at.

## Limitations

- weight-space screening only — no causal necessity or sufficiency result
- no activation weighting; unexcited directions are counted the same as excited ones
- no behavioral evaluation in this phase; no suite was opened
- no sub-2k localization — the full runs saved no adapter before step 2000
- no method comparison (SFT vs GRPO vs sign-only) is made or implied
- rank 16 caps every effective rank by construction
- base `||W||_2` is a deterministic iterative estimate; `dW` singular values are exact
- adapter-to-historical-prediction binding remains non-cryptographic
