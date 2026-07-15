# Held-Out Attribute-Configuration Analysis

*Last updated: July 15, 2026*

## Project Decision

The existing `remaining_goods.json` / `test_goods.json` split is sufficient for
the paper's core **within-benchmark** claim. A separate 50-good OOD suite exists
for external-validity analysis, but no additional OOD run is required to report
the current held-out result. The paper must not interpret the compositional
split itself as evidence of zero-shot transfer to unseen goods or categories.

## What Is Held Out

- Training contains 49,450 configurations and 98,900 X/Y-perspective prompts.
- Evaluation contains 9,890 configurations and 19,780 X/Y-perspective prompts.
- There are **zero repeated configurations and zero repeated prompt strings**
  across training and evaluation.
- Both splits use the same 100 goods and the same 4,945 unordered goods pairs.
- For every goods pair, training uses 10 configurations and evaluation reserves
  two disjoint configurations.

The appropriate description is therefore **held-out attribute-configuration
generalization for familiar goods and goods pairs**.

## Primary Attribute-Sensitivity Test

For unordered goods pair `g`, perspective `p`, held-out configuration `c`, and
non-reference ordinal profile `h`, define

```text
d[g,p,c,h] = 1(endowed profile = h) - 1(offered profile = h)
```

The two held-out configurations are first-differenced within the same goods
pair and perspective:

```text
delta P(No)[g,p] = sum_h theta[h] * delta d[g,p,h] + error[g,p]
```

This removes pair-by-perspective fixed factors. The eight `theta` coefficients
are diagnostic probability effects for the generic 3x3 ordinal profiles; they
are not the structural NLS `beta` utilities and do not represent separate
semantic attributes such as flavor, freshness, or scent.

## Results

| Diagnostic | Result |
|---|---:|
| Differenced observations | 9,890 |
| Unordered pair clusters | 4,945 |
| Pair-clustered joint Wald test | chi-square(8) = **5,097.22** |
| Pair-clustered p-value | less than 10^-1000 |
| Leave-one-good-out joint Wald test | chi-square(8) = **756.05** |
| Leave-one-good-out p-value | less than 10^-150 |
| Smallest leave-one-good-out absolute z-statistic | **11.23** |
| Within R-squared | **0.488** |
| Attribute effect range on P(No) | **0.177 to 0.630** |

Estimated probability effects relative to the `(1,1)` reference profile:

| Attribute profile | Estimate | Pair-clustered SE | Leave-one-good-out SE |
|---|---:|---:|---:|
| `(1,2)` | 0.177 | 0.011 | 0.016 |
| `(1,3)` | 0.298 | 0.011 | 0.016 |
| `(2,1)` | 0.228 | 0.011 | 0.017 |
| `(2,2)` | 0.446 | 0.011 | 0.021 |
| `(2,3)` | 0.546 | 0.011 | 0.022 |
| `(3,1)` | 0.355 | 0.011 | 0.020 |
| `(3,2)` | 0.566 | 0.011 | 0.025 |
| `(3,3)` | 0.630 | 0.011 | 0.024 |

The estimates are monotone in both quality dimensions. They show that choices
change systematically with attribute quality rather than collapsing to a
constant Yes/No response.

## Practical Significance

Comparing the two held-out configurations for the same goods pair:

- X-perspective answers flip in 1,685 of 4,945 pairs (**34.07%**).
- Y-perspective answers flip in 1,693 of 4,945 pairs (**34.24%**).
- At least one perspective flips in 2,086 pairs (**42.18%**).
- Both perspectives flip in 1,292 pairs (**26.13%**).
- The mean absolute change in `P(No)` is **0.340**.

In a complementary levels regression with 99 item-difference fixed effects,
adding the eight attribute profiles raises R-squared from **0.320** to **0.642**.
The attribute partial R-squared is **0.475**, and five-fold cross-validation
grouped by unordered goods pair shows a **47.43%** reduction in mean squared
prediction error relative to the item-only model.

## Interpretation for the Paper

Recommended wording:

> For each of 4,945 item pairs, GRPO used ten attribute configurations and
> evaluation reserved two non-overlapping configurations. No exact question
> appeared in both sets. Attribute-level combinations strongly affected
> held-out choices: we reject the joint null of no attribute differentiation
> under both pair-clustered and leave-one-good-out inference. Attribute profiles
> explain 48.8% of within-pair, within-perspective variation, and changing the
> reserved configuration changes at least one binary response for 42.2% of item
> pairs. These results demonstrate within-benchmark generalization to unseen
> configurations of familiar goods and pairs; they do not establish transfer
> to unseen goods or categories.

## Reproduction

Run from the project root:

```bash
python eval/analyze_attribute_effects.py
```

The script reads the held-out baseline-treatment responses from
`baseline/Qwen-7B-GRPO/loss_aversion_X.json` and
`baseline/Qwen-7B-GRPO/loss_aversion_Y.json` and prints the complete results as
JSON. It uses deterministic teacher-forced Yes/No probabilities; the reported
uncertainty does not capture variation across training seeds or stochastic
model samples.
