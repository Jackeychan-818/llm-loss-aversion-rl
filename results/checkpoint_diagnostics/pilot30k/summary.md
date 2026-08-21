# Late-checkpoint (30,000-step) diagnostic pilot

*EXPLORATORY / POST-HOC. This does **not** change the frozen checkpoint selections (seed 1 → step 2,000; seed 2 → step 6,000), and the 30k adapters are **not** selected models. No frozen/untouched, neutral-preference, OOD-50, GSM8K, or IFEval suite was evaluated.*

Git commit: `abbd128eb48cda7979bc5769cd3fb53d2cac721b` · generated 2026-08-21T07:40:42.628683+00:00

## Surface-form stress (96 cases × 2 perspectives × 48 equivalent forms)

| model | role | invariance | fidelity | worst-form fid | flip rate | prob spread | keep-both | λ | η | d | ρ(utility) | ρ(α) | ρ(β) | log₁₀cond(J) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Base | matched local base (context) | 0.094 | 0.663 | 0.500 | 0.290 | 0.879 | 0.594 | 7.637 | 1.007 | 7.703 | 1.000 | 1.000 | 1.000 | 2.1 |
| GRPO-qd-seed1-ckpt2000 | seed1 frozen-selected checkpoint | 0.089 | 0.732 | 0.615 | 0.266 | 0.887 | 0.250 | 0.032 | 0.091 | 0.096 | 0.792 | 0.782 | 0.976 | 2.0 |
| GRPO-qd-seed2-ckpt6000 | seed2 frozen-selected checkpoint | 0.125 | 0.748 | 0.615 | 0.247 | 0.843 | 0.250 | -0.042 | 0.659 | 0.660 | 0.830 | 0.827 | 0.976 | 16.5 |
| GRPO-qd-seed1-ckpt30000 | seed1 late endpoint (exploratory, NOT selected) | 0.208 | 0.788 | 0.625 | 0.225 | 0.797 | 0.208 | 0.090 | 0.663 | 0.669 | 0.913 | 0.941 | 0.929 | 17.1 |
| GRPO-qd-seed2-ckpt30000 | seed2 late endpoint (exploratory, NOT selected) | 0.198 | 0.776 | 0.594 | 0.229 | 0.793 | 0.250 | 0.000 | 0.783 | 0.783 | 0.917 | 0.949 | 0.929 | 17.4 |

Read: high invariance **and** high fidelity = an ownership-invariant rule; high invariance with low fidelity = a constant/shortcut answer; low invariance = surface-form fragility.

### Base — fidelity by transformation axis
- answer_style: ab_kt=0.783, ab_tk=0.611, keep_trade=0.736, yes_no=0.520
- display_order: endowed_first=0.670, offered_first=0.655
- attr_order: normal=0.663, reversed=0.663
- paraphrase: p0=0.656, p1=0.689, p2=0.642

### GRPO-qd-seed1-ckpt2000 — fidelity by transformation axis
- answer_style: ab_kt=0.805, ab_tk=0.668, keep_trade=0.744, yes_no=0.714
- display_order: endowed_first=0.748, offered_first=0.717
- attr_order: normal=0.736, reversed=0.729
- paraphrase: p0=0.750, p1=0.747, p2=0.700

### GRPO-qd-seed2-ckpt6000 — fidelity by transformation axis
- answer_style: ab_kt=0.826, ab_tk=0.674, keep_trade=0.756, yes_no=0.738
- display_order: endowed_first=0.770, offered_first=0.727
- attr_order: normal=0.749, reversed=0.747
- paraphrase: p0=0.774, p1=0.765, p2=0.706

### GRPO-qd-seed1-ckpt30000 — fidelity by transformation axis
- answer_style: ab_kt=0.861, ab_tk=0.722, keep_trade=0.776, yes_no=0.794
- display_order: endowed_first=0.809, offered_first=0.768
- attr_order: normal=0.791, reversed=0.786
- paraphrase: p0=0.819, p1=0.809, p2=0.738

### GRPO-qd-seed2-ckpt30000 — fidelity by transformation axis
- answer_style: ab_kt=0.855, ab_tk=0.705, keep_trade=0.766, yes_no=0.776
- display_order: endowed_first=0.802, offered_first=0.749
- attr_order: normal=0.779, reversed=0.772
- paraphrase: p0=0.813, p1=0.800, p2=0.715

## Surface-form paired differences (30k − selected)

Cluster bootstrap over `(case_id, perspective)` units (`ownership_keep_both_rate` over `case_id`); prompt forms are never treated as independent samples.

### GRPO-qd-seed1-ckpt30000 − GRPO-qd-seed1-ckpt2000

| metric | 30k | selected | Δ | 95% CI | flagged |
|---|---|---|---|---|---|
| semantic_invariance_rate | 0.208 | 0.089 | 0.120 | [0.073, 0.177] | **YES** |
| fidelity_mean | 0.788 | 0.732 | 0.056 | [0.040, 0.071] | **YES** |
| worst_form_fidelity | 0.625 | 0.615 | 0.010 | [-0.010, 0.057] | no |
| mean_pairwise_flip_rate | 0.225 | 0.266 | -0.041 | [-0.056, -0.027] | **YES** |
| mean_prob_spread_across_forms | 0.797 | 0.887 | -0.091 | [-0.124, -0.061] | **YES** |
| ownership_keep_both_rate | 0.208 | 0.250 | -0.042 | [-0.125, 0.042] | no |

### GRPO-qd-seed2-ckpt30000 − GRPO-qd-seed2-ckpt6000

| metric | 30k | selected | Δ | 95% CI | flagged |
|---|---|---|---|---|---|
| semantic_invariance_rate | 0.198 | 0.125 | 0.073 | [0.031, 0.120] | **YES** |
| fidelity_mean | 0.776 | 0.748 | 0.027 | [0.016, 0.038] | **YES** |
| worst_form_fidelity | 0.594 | 0.615 | -0.021 | [-0.047, 0.016] | no |
| mean_pairwise_flip_rate | 0.229 | 0.247 | -0.019 | [-0.030, -0.007] | **YES** |
| mean_prob_spread_across_forms | 0.793 | 0.843 | -0.051 | [-0.075, -0.028] | **YES** |
| ownership_keep_both_rate | 0.250 | 0.250 | 0.000 | [-0.073, 0.073] | no |

## Adverse framing (120 scenarios × 23 probabilities × 2 frames = 5,520)

No framing predictions exist for the frozen-selected seed1@2000 / seed2@6000 checkpoints. The only earlier GRPO framing comparator is the EXPLORATORY seed42 step-8000 run, so the 30k-vs-earlier framing contrast is NOT a clean within-seed checkpoint comparison.

| model | role | hard-flip | mean abs prob gap | prob monotonicity viol. | hard monotonicity viol. | λ | η | d | ρ(utility) | log₁₀cond(J) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen-7B-Base | matched local base (comparator) | 0.505 | 0.490 | 0.245 | 0.0055 | — | — | — | — | — |
| Qwen-7B-GRPO-step8000 | seed42 step-8000 (EXPLORATORY; not a selected checkpoint) | 0.689 | 0.679 | 0.258 | 0.0061 | 0.111 | 0.504 | 0.516 | 0.839 | 16.7 |
| GRPO-qd-seed1-ckpt30000 | seed1 late endpoint (exploratory, NOT selected) | 0.682 | 0.655 | 0.257 | 0.0078 | 0.090 | 0.663 | 0.669 | 0.913 | 17.1 |
| GRPO-qd-seed2-ckpt30000 | seed2 late endpoint (exploratory, NOT selected) | 0.687 | 0.669 | 0.254 | 0.0100 | 0.000 | 0.783 | 0.783 | 0.917 | 17.4 |

## Framing paired differences

Cluster bootstrap over scenarios.

### GRPO-qd-seed1-ckpt30000 − Qwen-7B-Base

| metric | 30k | reference | Δ | 95% CI | flagged |
|---|---|---|---|---|---|
| hard_flip_rate | 0.682 | 0.505 | 0.177 | [0.139, 0.213] | **YES** |
| mean_absolute_probability_gap | 0.655 | 0.490 | 0.165 | [0.135, 0.193] | **YES** |
| mean_probability_gap_negative_minus_positive | -0.651 | -0.489 | -0.162 | [-0.191, -0.133] | **YES** |
| classical_flip_rate | 0.680 | 0.504 | 0.176 | [0.139, 0.212] | **YES** |
| probability_monotonicity_violation_rate | 0.257 | 0.245 | 0.013 | [0.001, 0.025] | **YES** |
| hard_choice_monotonicity_violation_rate | 0.008 | 0.005 | 0.002 | [-0.002, 0.006] | no |

### GRPO-qd-seed2-ckpt30000 − Qwen-7B-Base

| metric | 30k | reference | Δ | 95% CI | flagged |
|---|---|---|---|---|---|
| hard_flip_rate | 0.687 | 0.505 | 0.182 | [0.142, 0.221] | **YES** |
| mean_absolute_probability_gap | 0.669 | 0.490 | 0.179 | [0.146, 0.210] | **YES** |
| mean_probability_gap_negative_minus_positive | -0.665 | -0.489 | -0.176 | [-0.207, -0.143] | **YES** |
| classical_flip_rate | 0.685 | 0.504 | 0.180 | [0.141, 0.220] | **YES** |
| probability_monotonicity_violation_rate | 0.254 | 0.245 | 0.009 | [-0.002, 0.020] | no |
| hard_choice_monotonicity_violation_rate | 0.010 | 0.005 | 0.005 | [0.001, 0.008] | **YES** |

### GRPO-qd-seed1-ckpt30000 − Qwen-7B-GRPO-step8000

| metric | 30k | reference | Δ | 95% CI | flagged |
|---|---|---|---|---|---|
| hard_flip_rate | 0.682 | 0.689 | -0.007 | [-0.024, 0.009] | no |
| mean_absolute_probability_gap | 0.655 | 0.679 | -0.023 | [-0.035, -0.011] | **YES** |
| mean_probability_gap_negative_minus_positive | -0.651 | -0.676 | 0.024 | [0.012, 0.036] | **YES** |
| classical_flip_rate | 0.680 | 0.689 | -0.008 | [-0.026, 0.008] | no |
| probability_monotonicity_violation_rate | 0.257 | 0.258 | -0.001 | [-0.012, 0.009] | no |
| hard_choice_monotonicity_violation_rate | 0.008 | 0.006 | 0.002 | [-0.001, 0.005] | no |

### GRPO-qd-seed2-ckpt30000 − Qwen-7B-GRPO-step8000

| metric | 30k | reference | Δ | 95% CI | flagged |
|---|---|---|---|---|---|
| hard_flip_rate | 0.687 | 0.689 | -0.003 | [-0.020, 0.013] | no |
| mean_absolute_probability_gap | 0.669 | 0.679 | -0.009 | [-0.020, 0.001] | no |
| mean_probability_gap_negative_minus_positive | -0.665 | -0.676 | 0.011 | [-0.000, 0.022] | no |
| classical_flip_rate | 0.685 | 0.689 | -0.004 | [-0.021, 0.013] | no |
| probability_monotonicity_violation_rate | 0.254 | 0.258 | -0.005 | [-0.014, 0.004] | no |
| hard_choice_monotonicity_violation_rate | 0.010 | 0.006 | 0.004 | [0.002, 0.006] | **YES** |

## Interpretation

Which of the five predeclared readings the numbers support (selected mechanically from the paired results above, not by hand):

- **Possibility 1 — SUPPORTED.** Utility correlation rises and semantic invariance improves: consistent with later learning of a more transferable decision policy.
- **Possibility 3 — SUPPORTED.** Utility correlation rises while adverse-framing susceptibility is worse than the matched base: later preference recovery does not imply general robustness.
- Possibility 2 — not supported. Utility correlation rises but invariance is unchanged.
- Possibility 4 — not supported. Both invariance and framing improve.
- Possibility 5 — not supported. Results disagree across seeds.

Seed agreement: **both seeds move the same way on every axis**.

### What the fitted-utility correlation does and does not show

The rise in Spearman correlation between a checkpoint's fitted structural utilities and the base's is a property of an **estimated econometric model fitted to choices**. It is **not** direct evidence about hidden neural representations, and nothing here measures what the network internally encodes.

It is also weakly identified at these checkpoints. Both 30,000-step endpoints have `log10 cond(J)` ≈ 17 — a **near-singular** Jacobian — so individual late-checkpoint alpha and utility values are poorly pinned down even though the rank correlation is high. Treat the late structural utility estimates as suggestive, not precise.

## Predeclared exploratory decision rule

Flag a potentially meaningful change when the paired 95% interval excludes zero **or** |Δ| ≥ 0.05. This decides whether to run the complete checkpoint trajectory; it is **not** a confirmatory hypothesis test.

**Any metric flagged:** True

**Recommendation:** Run the complete checkpoint trajectory: at least one paired comparison met the predeclared exploratory rule.

## Caveats

- Exploratory: cannot and does not change the frozen checkpoint selections (seed1 -> 2000, seed2 -> 6000).
- The 30k adapters are NOT selected models.
- No frozen/untouched, neutral-preference, OOD-50, GSM8K, or IFEval suite was evaluated.
- Fitted-utility correlation is a property of an estimated structural model, not direct evidence about hidden neural representations.
- Late-checkpoint structural utilities are near-singular (log10 cond(J) ~ 17), so individual alpha/utility values there are weakly identified.
