# Reward Design v2: Training Without a Cardinal Ground-Truth Delta

## Motivation

The current main reward uses `data/deltas/delta_consensus_v3.json`. For each
case, six frontier models provide an estimated utility difference, and their
mean determines both the preferred good and the magnitude of the reward. This
is useful as a pseudo-utility oracle, but it is not economic ground truth:

- preferences between heterogeneous consumption goods are subjective;
- the six models disagree on the sign for a substantial fraction of cases;
- averaging model estimates does not make the result a true cardinal utility
  difference; and
- weighting by `abs(delta)` adds a cardinal-strength assumption that the task
  itself does not identify.

The consensus-delta experiment remains valuable as the main completed run or
as an ablation. A more defensible follow-up should train primarily from
counterfactual consistency and ordinal relations that can be established from
the experimental design.

## Proposed reward

For a training unit containing matched X-endowed and Y-endowed prompts, use:

```text
R = w_pair * R_pair
  + w_neutral * R_neutral
  + w_dom * R_dominance
  + w_mono * R_monotonicity
  + R_format
```

Every active component should be normalized to approximately `[-1, 1]`. A
reasonable initial configuration is:

```text
w_pair = 1.0
w_neutral = 0.5
w_dom = 1.0
w_mono = 0.5
format reward = 0.1
```

These are hyperparameters to ablate, not theoretically privileged constants.
If a component is undefined for a case, omit it and renormalize by the sum of
the active weights so that different case types have comparable reward scale.

## 1. Counterfactual ownership consistency (`R_pair`)

The same goods configuration is presented from both ownership perspectives:

- X perspective: endowed with X and offered Y;
- Y perspective: endowed with Y and offered X.

The Yes/No strings must be converted to the final good selected:

| X-endowed response | Y-endowed response | Final choices | Interpretation | `R_pair` |
|---|---|---|---|---:|
| No | Yes | X, X | consistently chooses X | +1 |
| Yes | No | Y, Y | consistently chooses Y | +1 |
| No | No | X, Y | keeps both endowments | -1 |
| Yes | Yes | Y, X | trades away both endowments | -1 |

Therefore:

```text
R_pair = +1 if response_X != response_Y else -1
```

This is not the rejected, incorrectly specified "same answer under both
perspectives" rule. Always-No receives `-1`, because No/No selects different
goods when ownership reverses. The component directly targets endowment
dependence without requiring a ranking of X and Y.

### Limitation

Ownership consistency alone does not identify a sensible preference. A model
could always choose the first alphabetic good and still score well. The other
components anchor the direction of choice and test whether the learned policy
respects meaningful ordinal relations.

## 2. Ownership-free preference anchor (`R_neutral`)

For subjective pairs with no objective ordering, first ask an ownership-free
question such as:

> Which option would you prefer for yourself, X or Y?

The anchor should be produced before RL and frozen. To reduce prompt and order
artifacts, collect it under multiple paraphrases and both X/Y display orders.
Use only cases whose selected good is stable and whose probability margin
exceeds a preregistered threshold. Ambiguous cases receive no neutral-anchor
reward.

If the frozen preference is X, the endowed answers should be No from the X
perspective and Yes from the Y perspective. If it is Y, the answers reverse.
Give `+1` when the response selects the anchored good and `-1` otherwise.

This signal must be described as a **frozen neutral-preference pseudo-label**,
not ground truth. Possible sources should be reported separately:

1. human preference judgments, if available;
2. the frozen base model's ownership-free preferences; or
3. a preregistered multi-model consensus.

Human labels improve external validity, but none of these sources creates a
universal true preference over heterogeneous goods.

## 3. Pareto dominance (`R_dominance`)

The data use three ordered attribute levels, with level 1 < level 2 < level 3.
This supports objective ordinal supervision only when X and Y are expressed on
the same comparable dimensions and higher is unambiguously better.

For shared dimensions `k`, X Pareto-dominates Y when:

```text
x_k >= y_k for every k, and x_k > y_k for at least one k.
```

If X dominates Y, the correct responses are No when endowed with X and Yes
when endowed with Y. If Y dominates X, they reverse. Give `+1` for selecting
the dominant configuration and `-1` otherwise.

### Critical restriction

The numeric level labels are not comparable across unrelated attributes. For
example, level-3 camera battery and level-2 jacket warmth do not imply that the
camera dominates the jacket. Genuine Pareto labels should therefore be used
only for:

- two configurations of the same good;
- alternatives sharing the same attributes; or
- attributes for which a common comparison has been explicitly justified.

Cases involving different goods with different attribute dimensions are not
Pareto cases, even if all of X's level numbers exceed all of Y's.

## 4. Ordinal attribute monotonicity (`R_monotonicity`)

The three levels remain useful when heterogeneous goods cannot be Pareto
ranked. Construct controlled counterfactuals that hold everything fixed while
changing one attribute by one level.

If an attribute of X improves while Y is unchanged, the probability of
selecting X should not decrease:

```text
P(select X | improved X, Y) >= P(select X | original X, Y).
```

Likewise, improving Y while holding X fixed should not increase the
probability of selecting X. A pairwise implementation can reward the expected
ordering of sampled choice rates or log probabilities, with a small tolerance
to avoid punishing numerical noise.

This is **attribute monotonicity**, not Pareto dominance and not a cardinal
utility difference. It supplies an ordinal constraint without asserting how
much a one-level improvement is worth or how improvements across different
attributes trade off.

## Training-unit requirement

`R_pair` cannot be implemented correctly by a reward function that sees only
one prompt at a time. The trainer must preserve a joint identifier for the
matched X/Y perspectives and score them together.

For GRPO with `G` completions per perspective:

1. generate `G` completions for the X-endowed prompt;
2. generate `G` completions for the matched Y-endowed prompt;
3. pair completion `g` from X with completion `g` from Y;
4. compute the joint `R_pair`;
5. assign the same joint reward contribution to both members; and
6. add any perspective-specific neutral, dominance, monotonicity, and format
   contributions.

The batching and distributed sampler must keep matched records together. Unit
tests should cover all four Yes/No cross-tab cells before training.

## Evaluation and ablations

Do not evaluate the method only with the same constraints used for training.
Report:

- the full X-perspective by Y-perspective choice cross-tab;
- keep-both, trade-both, and consistent-choice rates;
- structural lambda and eta, with identification diagnostics;
- performance on the existing in-distribution compositional test;
- performance on `data/ood_new_goods_50_test.json`;
- neutral-preference agreement on frozen cases;
- Pareto accuracy and monotonicity violation rate; and
- KL divergence and general capability checks.

Recommended ablations are:

1. consensus delta (completed reference);
2. Qwen-own delta (completed/current ablation);
3. `R_pair` only;
4. `R_pair + R_neutral`;
5. `R_pair + R_dominance + R_monotonicity`;
6. the complete proposed reward; and
7. sensitivity to component weights and neutral-anchor confidence thresholds.

## Recommended claim

Avoid saying that the model is trained against the true utility difference for
subjective goods. A defensible description is:

> We train with counterfactual ownership consistency, frozen ownership-free
> preference anchors, Pareto dominance where alternatives share comparable
> attributes, and ordinal attribute monotonicity. No universal cardinal
> ground-truth utility difference between heterogeneous goods is assumed.

The existing six-model delta should be called a **consensus pseudo-utility
oracle**. Its results remain informative, but they should not be presented as
validation against economic ground truth.
