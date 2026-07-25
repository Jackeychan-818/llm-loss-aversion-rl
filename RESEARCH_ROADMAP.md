# Research Roadmap

*Decision recorded: July 25, 2026*

This document records the next experimental priorities after the confirmatory
seed replication and the prospective unused-configuration evaluation. It is a
research plan, not a paper-writing schedule. A full manuscript rewrite is
deliberately deferred until the next experiments clarify the final scope.

## Current evidence

- Confirmatory Qwen-own-delta GRPO seeds both passed the frozen ID-selection,
  one-shot OOD-50, and GSM8K gates:
  - seed 1 selected at step 2,000;
  - seed 2 selected at step 6,000.
- The prospective unused-configuration test is complete. On 49,450 new
  configurations of familiar goods and goods pairs, lambda changes from 5.946
  for the matched base to 0.031 and -0.053 for the two selected seeds.
- The full framing benchmark is complete and is a specificity result, not
  evidence of general debiasing: the exploratory step-8,000 model is more
  framing-susceptible than the matched base.
- GSM8K shows no material capability loss for either confirmatory seed under
  the frozen non-inferiority rule.

The next work should therefore test why the intervention works, whether it is
robust to prompt semantics, and whether capability retention extends beyond
the existing GSM8K result.

## Priority 1: Matched causal baselines

### 1A. Matched supervised fine-tuning

Train a Qwen-own-delta SFT baseline using:

- the same base weights;
- the same training prompts and binary targets;
- the same LoRA parameterization;
- a matched data and optimization budget;
- independent seeds and a checkpoint rule frozen before training.

This tests whether ordinary supervised learning is sufficient to produce the
observed behavioral change.

### 1B. Sign-only GRPO

Train a GRPO baseline that keeps every other design choice fixed but replaces
the magnitude-weighted reward with:

```text
+1 for selecting the frozen preferred good
-1 otherwise
```

This tests whether `abs(delta)` contributes beyond the sign of the preferred
good. The existing frontier-consensus-delta run remains the reward-source
ablation and should not be conflated with this reward-magnitude ablation.

### Required discipline

- Freeze seeds, checkpoints, eligibility rules, metrics, and inference before
  launching.
- Report lambda and eta jointly with consistency, keep-both, trade-both,
  pseudo-utility alignment, and capability retention.
- Use the exact matched local base and identical evaluators.
- Do not use the already-opened prospective unused-configuration results for
  training, checkpoint selection, or method revision. Any comparison added
  after seeing those results must be labelled post-hoc, or a new untouched
  evaluation must be frozen in advance.
- If the paper claims that one training method outperforms another, replicate
  the relevant baseline across independent seeds rather than relying on a
  single favorable run.

## Priority 2: Prompt-semantic robustness

Test whether the learned behavior survives changes that preserve the decision
but alter its surface form. Freeze a moderate, stratified subset before model
evaluation and include:

- `Yes`/`No` versus explicit `keep`/`trade` responses;
- `X`/`Y` versus `A`/`B` labels;
- reversed display order;
- reversed attribute order;
- a small fixed set of prompt paraphrases.

Use the exact local base, seed 1 at step 2,000, and seed 2 at step 6,000 as the
primary model set. The primary outcome should be whether the same final good is
selected across semantically equivalent forms. Also report response-token
rates and the paired structural outcomes where estimable.

Freeze template text, subset IDs, hashes, decoding, and the analysis before
opening trained-model results. This experiment distinguishes an
ownership-invariant choice policy from a narrow `Yes`/`No`, label, order, or
template heuristic.

## Priority 3: Broader capability retention

### 3A. IFEval first

Complete IFEval before building a large math benchmark battery. It tests
objectively verifiable instruction-following behavior that GSM8K does not
cover.

- Paper: <https://arxiv.org/abs/2311.07911>
- Official evaluator:
  <https://github.com/google-research/google-research/tree/master/instruction_following_eval>

### 3B. One compact math robustness test

The preferred next math extension is **GSM-Symbolic-500**:

- use 100 source templates and freeze five generated numerical variants per
  template;
- record the generator version, seed, selected IDs, and dataset SHA-256;
- run the exact local base, seed 1 at step 2,000, and seed 2 at step 6,000;
- use identical prompting, greedy decoding, parsing, and answer scoring;
- cluster uncertainty by source template because variants from the same
  template are dependent.

GSM-Symbolic tests whether the GSM8K retention result survives controlled
changes to numerical values and problem structure. It should be described as
contamination-reduced rather than guaranteed unseen because its templates are
derived from GSM8K.

- Paper: <https://arxiv.org/abs/2410.05229>
- Official repository: <https://github.com/apple/ml-gsm-symbolic>

For every capability test:

- freeze a non-inferiority margin before evaluation;
- report each selected seed separately against the same base responses;
- use paired confidence intervals and discordant-case counts;
- count parse failures as incorrect rather than dropping them;
- archive complete generations and manifests;
- do not use capability results for checkpoint selection.

MATH-500 may be added later as a harder secondary check, but it is not required
for the current roadmap and should not be described as clean OOD.

## Priority 4: Confidence calibration — low priority

Overconfidence is an interesting side-effect test, but it should not delay the
causal baselines, semantic robustness, IFEval, or the compact math extension.
If run later, call it **answer-confidence calibration** rather than generic
overconfidence.

A minimal study would:

1. generate a fixed answer;
2. show the question and fixed answer back to the same model;
3. estimate normalized `P(True)` versus `P(False)` at `T=1`;
4. compare accuracy, mean confidence minus accuracy, Brier score, reliability,
   confidence on incorrect answers, and risk-coverage behavior.

Use the matched local base and both frozen seed selections. Freeze the dataset,
prompts, hashes, metrics, and any non-inferiority rule before evaluation.
Calibration improvement is not predicted by the current reward: unchanged
calibration would support behavioral specificity, while worse calibration
would reveal a post-training side effect.

The P(True) design follows:
<https://arxiv.org/abs/2207.05221>.

## Supporting analysis

The following work can proceed alongside the experimental priorities:

- pair- and good-aware bootstrap inference;
- estimator-recovery simulations with known lambda and eta;
- direct paired-logit diagnostics for ownership and preference components;
- transparent include/exclude/regularized sensitivity for problematic goods
  37 and 51;
- durable publication of selected adapters and raw claim-carrying predictions.

These analyses strengthen identification and reproducibility. They do not
change the frozen confirmatory selections or authorize reuse of opened test
sets for further method selection.
