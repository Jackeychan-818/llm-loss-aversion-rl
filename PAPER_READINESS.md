# Full-Paper Readiness and Current Methodological Risks

*Last updated: July 30, 2026*

This document is the authoritative summary of the current paper direction,
which results are final versus validation-only, and which methodological
problems must be addressed before making full-paper claims.

The project is currently venue-agnostic. `draft/aaai27/` is a historical
submission workspace, not the active target or schedule.

## Current Paper Decision

- **Primary model:** Qwen2.5-7B trained with Qwen's own estimated utility
  differences (`data/deltas/delta_qwen_base.json`), with checkpoint 8,000 as
  the current selected candidate.
- **Reward-source ablation:** the completed frontier-consensus-delta model
  (`data/deltas/delta_consensus_v3.json`).
- **Economic motivation:** utility is agent-specific. The main experiment asks
  whether post-training can remove ownership dependence while preserving
  Qwen's own estimated preference ordering, rather than imposing a preference
  ordering aggregated from other models.
- **Terminology:** both Qwen-own delta and frontier-consensus delta are
  pseudo-utility signals, not objective cardinal ground truth over
  heterogeneous goods.

Reward V2 remains a follow-up design unless and until it has completed,
replicated results. It should not compete with the completed Qwen-own-delta
story in the main paper without such evidence.

## Current Result Roles

| Model/result | Role in paper | Current interpretation |
|---|---|---|
| Qwen-own delta, step 8,000: ID lambda = 0.111, eta = 0.504, W = 0.909 | Primary-model validation result | `test_goods.json` was used for checkpoint selection; 0.111 is not an untouched final-test estimate. |
| Qwen-own delta, step 8,000: OOD lambda = 0.226, eta = 0.790, consistency = 49.46% | Intended final generalization result | May carry the final claim only after raw outputs, estimator artifacts, checkpoint identity, and reproduction commands are archived and the suite's frozen status is documented. |
| Consensus-delta: ID lambda = 0.177, eta = -0.048, W = 0.890 | Reward-source ablation/reference | Completed and committed. Use the existing exact matched local base—not the historical 11.75 estimate—for any before/after comparison. |
| Consensus-delta: OOD lambda = 0.395, eta = 0.502, consistency = 64.89% | Reward-source ablation on OOD | Shows a trade-off: higher lambda than Qwen-own delta, but lower eta and stronger consistency. Raw OOD artifacts must also be archived. |
| **Matched local base: lambda = 7.637 (SE 0.627), eta = 1.007, W = 0.744** | **Causal baseline — use this** | Exact local weights, same 9,890 rows, same teacher-forced scorer, same estimator. Behaviour: keep-both 99.13%, consistency 0.87%. |
| Historical base Qwen: lambda = 11.75, eta = 1.52 | Superseded — do not use for before/after | Together Turbo endpoint, 9,950 cases, top-5 first-token probs. Inflated ~54% vs the matched base by pipeline mismatch, not training. |

The paper must not claim that Qwen-own delta dominates consensus on every
dimension. It currently has lower estimated lambda, whereas consensus has
lower eta and higher OOD consistency.

### Frozen prospective configuration test — EVALUATED (July 24, 2026)

On July 23, 2026, after the confirmatory seed selections and OOD-50/GSM8K
verdict were complete, `data/frozen_unused_test_goods.json` was constructed and
frozen: 10 previously unused joint attribute configurations for each of the
4,945 existing goods pairs (49,450 cases, 98,900 paired-perspective prompts per
model), excluding all 12 codes already in `test_goods.json`/`remaining_goods.json`.

It was opened **once** on July 24, 2026 for the three policy-allowed models
(Model A NLS, T=1, N=49,450):

| model | λ (SE) | η (SE) | consistency | keep-both | W |
|---|---|---|---|---|---|
| matched base | 5.946 (0.192) | 1.458 (0.046) | 0.008 | 0.992 | 0.742 |
| seed1 @2000 | 0.031 (0.007) | 0.089 (0.014) | 0.659 | 0.183 | 0.882 |
| seed2 @6000 | −0.053 (0.005) | 0.693 (0.013) | 0.716 | 0.211 | 0.909 |

**λ collapses base 5.946 → ≈0 on both confirmatory seeds**, with SEs ~3× tighter
than `test_goods` (5× the data) and η/consistency/W matching the ID pattern (W
within 0.001 of ID for every model). This is a within-benchmark **configuration**
generalization result, **not** new-goods OOD — OOD-50 remains that test. Still
prohibited for training, reward construction, checkpoint selection, and method
changes. Results: `results/frozen_unused_results.json`. Full provenance (dataset
SHA `793c4721…`, evaluator commit, base-model/adapter hashes, PBS job IDs):
`results/frozen_unused_evaluation_manifest.json`. Construction (immutable):
`data/FROZEN_UNUSED_TEST.md`, `data/frozen_unused_test_goods.manifest.json`.

## Priority 0: Must Resolve for the Main Claims

The live implementation and reproducibility defects are tracked by stable ID
in `KNOWN_ISSUES.md`. Before launching or interpreting the full causal
baselines, close `PAIR-001`, `PROV-001`, `SELECT-001`, `RESUME-001`,
`ABLATION-001`, and `ENV-001`. Before final paper claims, also close
`INFER-001`, `BUILD-001`, `PILOT-001`, and `ARTIFACT-001`.

### 1. Matched local base evaluation

The original `lambda_before = 11.75` and post-training estimates were not a
strictly matched before/after comparison. The former used Together's
`Qwen/Qwen2.5-7B-Instruct-Turbo`, 9,950 cases, and top-five first-token
probabilities. The post-training evaluator uses a local
`models/Qwen2.5-7B-Instruct` checkpoint, 9,890 cases, and exact normalized
teacher-forced Yes/No sequence scores.

**Status (July 15, 2026): RESOLVED.** The exact local base was evaluated with no
adapter on the same 9,890 `test_goods.json` rows via `eval/run_qwen_local.py`
and estimated with the same structural code (Model A NLS, `--link-scale 1`).

| base estimate | lambda | SE | eta | SE | W |
|---|---|---|---|---|---|
| Historical (Together Turbo, 9,950, top-5 first-token) | 11.75 | 1.22 | 1.52 | 0.085 | -- |
| **Matched local (local weights, 9,890, teacher-forced)** | **7.637** | 0.627 | **1.007** | 0.120 | **0.744** |

**The historical 11.75 was inflated by ~54% by the pipeline mismatch, not by
training.** The matched base is well identified (|t| = 12.2) and behaviourally
extreme: consistency 0.87%, keep-both 99.13%, trade-both 0.00%.

**Use the matched base for every before/after claim.** Against the primary model
(Qwen-own delta, step 8,000): lambda 7.637 -> 0.111 (-98.5%), eta 1.007 -> 0.504
(-50%), keep-both 99.13% -> 24.2%, consistency 0.87% -> 69.7%. This comparison is
now single-pipeline (same weights, rows, scorer, estimator) and may be described
as causal. Note lambda collapses while eta only halves.

Artifacts: `baseline/Qwen-7B-Base-Local/`. Reproduce:
`qsub train/submit_eval_base_matched.pbs`.

### 2. Reliability of Qwen's own pseudo-utility

The agent-specific utility motivation is economically defensible, but base
Qwen says No in approximately 99% of endowed prompts. This limited choice
variation may weakly identify its item and attribute utilities.

**Status (July 15, 2026): largely RESOLVED — evidence below.**

*Identification.* The low trade rate does not flatten the utilities because
scoring uses log-probabilities at link scale `T=1`, not binary choices:
`P(Yes)=0.08` and `P(Yes)=0.28` both decode to "No" but differ substantially.
(Under a `T=0`/binary encoding the objection would be valid — see CLAUDE.md #8.)
Measured on `baseline/Qwen-7B/Model_1/`:

| check | value |
|---|---|
| α (item FE) | 99/99 non-zero SE, median \|t\| = 9.6, 91.9% significant |
| β (attr FE) | 8/8 non-zero SE, median \|t\| = 27.5, 100% significant |
| δ̃ spread | sd = 0.94, range [−4.06, +3.91], 1.6% near-zero, 46.9% positive |

*Ownership-free validation (the required action).* `sign(δ̃)` vs Qwen's frozen
ownership-free preferences (multiple paraphrases, both display orders):

| split | n compared | sign agreement |
|---|---|---|
| test_goods | 6,184 | **71.1%** |
| remaining_goods | 30,757 | **70.8%** |

By magnitude (test / remaining): |δ̃|>1.0 → **85.5% / 85.3%**;
0.5<|δ̃|≤1.0 → 67.9% / 70.0%; |δ̃|≤0.5 → 63.6% / 62.3%. Chance = 50%.

Agreement rises with |δ̃|, and the reward is weighted by |δ̃| — so it is loudest
exactly where the signal is most trustworthy. Retained coverage: stable anchors
exist for 6,184/9,890 test and 30,757/49,450 training configurations (~62%).

Reproduce: `python eval/validate_qwen_delta_anchor.py`
→ `results/qwen_delta_anchor_validation.json`.

**Remaining action:** human validation on a stratified subset would further
strengthen the claim. Note the anchor and δ̃ are both Qwen-derived, so this is
convergent validity across two elicitation framings, not external ground truth.

### 3. Indirect leakage in Qwen-own delta construction

The Qwen utility table used to build `delta_qwen_base.json` was estimated from
9,950 base cases containing the 60 trial and 9,890 `test_goods` cases. Those
fitted utilities then generated rewards for `remaining_goods.json`. Exact test
prompts were not used in GRPO, but test responses informed the training reward
oracle.

**Consequence:** `test_goods.json` is not a fully untouched final test for the
current Qwen-own-delta run.

**Required action for the current run:** treat `test_goods` as validation and
use a separately frozen evaluation suite for final claims. **Required action
for a clean rerun:** construct the Qwen-own reward from training-only base
responses or training-only frozen ownership-free preferences.

### 4. Checkpoint selection on `test_goods`

Seven Qwen-own-delta checkpoints were evaluated on `test_goods`, and step 8,000
was selected after inspecting lambda and eta. Its `lambda = 0.111` is therefore
a validation result and may be optimistically selected.

The choice was also post-hoc: step 20,000 has the smaller absolute lambda
(`|-0.083| < 0.111`) but a much worse `eta = 1.263`. Rejecting it is reasonable,
but the joint rule was not written in advance.

**Status (July 15, 2026): rule FROZEN — see `CHECKPOINT_PROTOCOL.md`.**

The rule is now fixed in advance and applied mechanically by
`eval/select_checkpoint.py` (no judgement at selection time): eligibility
requires `consistency >= 0.50` and finite non-zero SEs; the primary criterion is
`min d = sqrt(lambda^2 + eta^2)` (the joint distance from the rational ideal, so
lambda alone cannot be bought by loading rigidity onto eta); tie-breaks are
higher consistency, then the earliest checkpoint.

Applied post hoc to the completed run it reproduces the original choice —
step 8,000 wins with `d = 0.516`, and step 20,000 (smallest |lambda| = 0.083 but
`eta = 1.263`) is demoted to *last* with `d = 1.266`. This shows the frozen rule
agrees with the judgement already made, but does NOT retroactively make 8,000 a
pre-registered choice, and does not remove the selection-on-validation bias in
its `lambda = 0.111`.

Artifact: `results/checkpoint_selection/qwen_delta_seed0.json`.

**Remaining action:** describe `test_goods` as validation, not final test
(done in `draft/training_overview.tex`), and apply the frozen rule to each new seed,
opening the frozen final suite only after selection.

### 5. Training-seed replication

The Qwen-own-delta checkpoint curve is highly non-monotonic. Conditional NLS
standard errors do not measure variation across training seeds.

**Status (July 23, 2026): replication CONFIRMED — 2/2 confirmatory seeds PASS.**
Both new seeds (`SEED=1,2`) finished cleanly at `MAX_STEPS=30000` with the full
15/15 grid. Frozen mechanical selection: seed 1 → **step 2,000**
(λ̂_ID +0.032, η̂ 0.091, d 0.096), seed 2 → **step 6,000**
(λ̂_ID −0.042, η̂ 0.659, d 0.660; step 2,000 excluded, consistency 0.419 < 0.50).
The one-shot OOD-50 and GSM8K per selected checkpoint are now done and the
mechanical `seed_replication_report.json` verdict is **2/2 PASS**:
- seed 1: S2 |λ_OOD|=0.259 ≤0.5; S3 consistency 0.50, keep 0.39, trade 0.11;
  S5 GSM8K paired-CI lower −1.14pp ≥ −3 (Δ−0.15pp). η_OOD +0.323 (not gated).
- seed 2: S2 |λ_OOD|=0.064 ≤0.5; S3 consistency 0.60, keep 0.31, trade 0.09;
  S5 GSM8K paired-CI lower −1.44pp ≥ −3 (Δ−0.30pp). η_OOD +0.593 (not gated).
Report every seed jointly; note seed 1's S3 consistency sits exactly at the 0.50
floor and both λ_OOD are small-but-positive — a clean pass, not a blowout.
Raw OOD/GSM8K predictions (73.1 MB) are checksummed in
`results/ood_gsm8k_raw_manifest.json` but not committed (see #6).

**BLOCKER FOUND AND FIXED (July 15, 2026) — read before launching seeds.**
`grpo_train.py` had no seed handling at all: no `--seed`, no config key, nothing
passed to `GRPOConfig`. `TrainingArguments.seed` defaults to **42**, so *every*
run silently shared one RNG stream (same init, same data order, same rollout
sampling). Launching "seed 2" and "seed 3" as the code stood would have produced
near-identical trajectories, suspiciously consistent lambdas, and a **fake
replication measuring GPU non-determinism rather than training variance** — a
result that looks like strong evidence and is worthless.

Fixed: `--seed` (CLI, overrides config) is now threaded into
`GRPOConfig(seed=..., data_seed=...)`, `seed: 42` is declared in
`train/configs/qwen25_7b_qwen_delta.yaml`, and
`train/submit_train_glong_qwen_delta.pbs` takes `-v SEED=...`, writes to a
per-seed output dir (`checkpoints/grpo_qwen_delta_seed<N>`) and a per-seed log.
Each run logs `Training seed: <N>` so the seed is auditable afterwards.

**The original completed run used seed=42.** It remains exploratory and is not
part of the confirmatory denominator. The two new runs use seeds 1 and 2.

**Completed action:** the frozen evaluation pipeline and decision rule in
`PRE_REGISTRATION.md` were applied to seeds 1 and 2. Both passed, so the
pre-registered rule does not require a third fresh seed. Continue to report
seed-level lambda, eta, consistency, final-evaluation performance, and
capability-retention metrics jointly.

**Pre-register the success criterion BEFORE launching** (otherwise the read-out
is another post-hoc judgement): each seed must reduce lambda far below the
matched base (lambda = 7.637) without degenerate choice behaviour or material
capability loss. Seeds are NOT required to reproduce lambda = 0.111 exactly.
Report every seed, not only the best.

#### Optimization path and structural utility diagnostics (post-hoc)

Two optimization layers must be reported separately. GRPO uses the fixed
Qwen-own pseudo-utility difference to assign `+|delta|` or `-|delta|`, then
optimizes a DAPO policy objective with clipping (`epsilon=0.2`) and a KL penalty
(`beta_GRPO=0.04`). It does not directly optimize the structural alpha or beta
parameters. Training-path diagnostics should plot reward, reward dispersion,
loss, KL, entropy, gradient norm, learning rate, and the zero-reward/DAPO
filtering rate for every seed.

Model A subsequently estimates behavior by NLS, minimizing squared error
between teacher-forced `P(No)` (the probability of keeping the endowed good)
and the logistic link of
`z=(1+lambda)U_X-U_Y+eta`, with
`U=exp(alpha_item+beta_attribute_profile)`. Here `alpha_1=0` and
`beta_1,1=0` are reference levels. The default `--starting ols` uses
`lambda_0=eta_0=0` and pooled-OLS starts for the remaining alpha/beta values.
The structural beta parameters are unrelated to the GRPO KL coefficient.

The current `curve_fit` implementation returns final parameters and covariance
but does not expose a genuine per-iteration callback; its nominal convergence
history is therefore not an optimization trace. A diagnostic wrapper should
record starting/final RSS, convergence status, Jacobian conditioning, and
agreement across OLS, zero, and perturbed starts without changing
`eval/core_exp_refactored.py` or overwriting the claim-carrying estimates.

Current fitted ranges motivate this check:

| estimate source | alpha range | beta range | fitted utility range |
|---|---:|---:|---:|
| Historical Qwen used for the reward | [-2.632, 0.409] | [0.451, 1.019] | [0.072, 4.173] |
| Matched local base (behavioral step 0) | [-1.746, 0.624] | [0.358, 0.869] | [0.174, 4.450] |
| Exploratory Qwen-own-delta step 8,000 | [-1075.111, 1.884] | [0.435, 1.026] | [0, 18.375] |

The extreme step-8,000 alpha and zero implied utilities are a warning about
weak identification or quasi-separation for some goods. Low lambda alone does
not establish stable utility recovery. Report alpha/beta stability, utility
distribution and rank preservation, and multi-start sensitivity. The matched
local base is the training step-0 comparator; the historical Together-hosted
utility is the fixed reward source. They must not be conflated.

An early saved adapter such as step 600 may be added to the ID trajectory as a
post-hoc diagnostic because checkpoints were saved every 200 steps. It is not
part of the frozen 2k–30k selection grid, cannot affect selection, and must not
be opened on OOD as a candidate.

### 6. Archive the claimed checkpoint and OOD evidence

**Status (July 17, 2026): PARTIALLY resolved. Be precise about which half.**

**Committed (derived results — every reported number is traceable):**
- `results/manifests/*.json` — per result: the parsed estimate, SHA-256 of every
  file in the eval dir, adapter identity + checksum, exact repro command, code
  commit, and a per-file `tracked_in_git` flag. Covers: `matched_local_base`,
  `primary_qwen_delta_step8000`, `ood50_primary_qwen_delta_step8000`,
  `ood50_base`, `ood50_consensus_ablation`, `gsm8k_capability_retention`.
- The estimation outputs behind each number (NLS CSVs, choice-prob and raw-count
  CSVs), the GSM8K summaries + `comparison.json`, `ood50_summary.html`,
  `results/qwen_delta_anchor_validation.json`,
  `results/checkpoint_selection/*.json`.
- Raw ID `loss_aversion_X/Y.json` predictions for the exact matched local base
  and selected Qwen-own-delta checkpoint, plus the corresponding consensus and
  prompt-treatment ID predictions.

**NOT committed (the complete reported result set is therefore not reproducible
from the repository alone):**
- Raw OOD `loss_aversion_X/Y.json` predictions for the base, selected
  Qwen-own-delta checkpoint, and consensus ablation. A SHA-256 verifies a file
  someone **already has**; it does **not** make the file downloadable or the
  result independently re-derivable.
- Per-item GSM8K prediction records needed to reproduce the paired comparison
  from model outputs.
- Adapter weights for the selected checkpoints (checksummed in the manifests
  only).

**Required action (decision pending):** commit the missing OOD and per-item
capability records, or publish them to an anonymous submission archive and
record stable identifiers plus checksums here. Package the selected adapters as
supplementary artifacts. Until then, the honest status is: **ID structural raw
predictions and derived results are archived; OOD/capability raw evidence and
adapter weights are incomplete.** Do not describe artifact archival as complete.

## Priority 1: Needed for a Competitive Full Paper

### 7. Matched optimization baselines and reward ablations

Because every training prompt has a binary target, reviewers will ask whether
ordinary supervised LoRA fine-tuning is sufficient.

**Required comparisons:**

1. exact local base Qwen;
2. Qwen-own-delta supervised fine-tuning on the same prompts/targets;
3. Qwen-own-delta GRPO;
4. frontier-consensus-delta GRPO as the reward-source ablation; and
5. preferably sign-only flat-reward GRPO to test whether delta magnitude adds
   value.

**Status (July 30, 2026): infrastructure complete; exploratory pilot complete;
full SFT training complete; confirmatory comparison still open.** Both SFT
seeds reached 30,000 steps on NSCC and contain the complete 15-checkpoint grid.
The full grid has not been evaluated and the selector has not run, so there is
no full-SFT behavioral result. The seed-1 pilot evaluated steps
2k/4k/6k on `test_goods` validation. Both methods reduce lambda sharply; SFT
reaches `d = sqrt(lambda^2 + eta^2)` of 0.060 and 0.052 at 4k and 6k, while
sign-only GRPO is less stable across the three checkpoints. These results do
not show that SFT wins: the pilot has one seed, no frozen selector, an
incomplete grid, and no untouched method-comparison suite. See
`results/causal_baseline_pilot/pilot_table.md`.

**Required next action:** before opening the full SFT checkpoint trajectories,
freeze a new untouched comparison suite and the full method-comparison
protocol. **DONE (2026-07-30, branch `codex/cpu-paper-gates`):** the untouched
suite is frozen (`data/method_comparison/`, 19,780 cases + semantic
counterbalancing, unevaluated) and the protocol is frozen
(`METHOD_COMPARISON_PROTOCOL.md`); the scale-matched control is specified+tested
(ABLATION-001, run-pending). **Still pending (GPU/budget-gated):** evaluate the
complete SFT grid, apply the unchanged selector, complete the matched
sign-only/scale-matched runs, then open the untouched suite once. If matched SFT
confirms the effect, narrow the central claim from “GRPO is necessary” to
“targeted post-training can remove ownership dependence,” with GRPO versus SFT
treated as a mechanism and efficiency comparison (see
`results/grpo_efficiency/`).

### 8. Predeclare the primary outcome and report the full behavior

The paper targets lambda, but lambda alone can be misleading when eta absorbs
choice rigidity. The current Qwen-own and consensus models illustrate a real
trade-off.

**Recommended hierarchy:**

1. primary outcome: lambda;
2. constraint/secondary structural outcome: eta;
3. direct outcomes: consistent choice, keep-both, trade-both, and target
   agreement; and
4. preference-preservation outcome: agreement with frozen ownership-free Qwen
   preferences.

### 9. Stronger structural inference

The current Model A standard errors come from SciPy `curve_fit` and do not
fully reflect repeated perspectives, goods pairs, shared goods, or training
randomness.

**Required action:** add pair-aware resampling, leave-one-good-out robustness,
intervals across training seeds, multi-start/objective diagnostics, and
Jacobian/conditioning checks. Run estimator-recovery simulations with known
lambda and eta, and always interpret both parameters jointly.

### 10. General-capability retention

**Status: GSM8K DONE (matched); IFEval outstanding.**

The GSM8K half of this item is complete and uses the *exact local base*
(`models/Qwen2.5-7B-Instruct`, no adapter) vs the selected Qwen-own-delta
checkpoint 8,000, scored by one harness on all 1,319 test problems, 8-shot CoT,
greedy decoding, 100% answer-parse rate on both sides:

| model | accuracy |
|---|---|
| exact local base | **86.88%** (1146/1319) |
| Qwen-own-delta step 8,000 | **85.90%** (1133/1319) |

Paired: Δ = **−0.99 pp**, bootstrap 95% CI **[−2.12, +0.15] pp**, McNemar exact
**p = 0.111**, discordant 35 base-only vs 22 tuned-only → `no_clear_change`.
Fair statement: no statistically significant degradation; at most a ~1–2 pp
effect indistinguishable from noise. Do NOT write "identical" — the point
estimate is a small negative drift and the CI leans negative.

Artifacts: `results/gsm8k/{base,qwen_delta_8000}/summary.json`,
`results/gsm8k/comparison.json`. Reproduce:
`qsub -v GSM8K_DATA_FILE=data/gsm8k_full.csv train/submit_eval_gsm8k.pbs`.

**Required action:** add at least one instruction-following/general benchmark
such as IFEval to complete the pair.

### 11. Cross-model and human claims

Do not currently claim that the model is lower than every frontier model or
below humans. Frontier results have not been harmonized across model endpoints,
samples, probability scorers, and estimators. The commonly cited human value
around 2.25 also comes from a different prospect-theory task, while this
project parameterizes the endowed-good multiplier as `1 + lambda`.

**Required action:** either run a harmonized frontier comparison and matched
human experiment, or narrow the paper to reductions relative to an exactly
matched local base and the ownership-neutral target `lambda = 0`.

## Reported quantities

Alongside lambda (loss aversion) and eta (status-quo bias) we report **W, the mean
pseudo-utility alignment** (`eval/pseudo_utility_alignment.py`):

    w_q = u_chosen / max(u_1, u_2);  W = mean(w_q)

The frozen Model-A reference utilities are strictly positive, so W lies in
(0,1]. A higher-utility choice receives 1; a lower-utility choice receives its
utility as a fraction of the best available utility. W is **descriptive**,
reported with lambda/eta; it is **not** a seed success gate (S2 remains
|lambda_OOD| <= 0.5) and was added AFTER the seed pre-registration was frozen,
so it must not be framed as a pre-registered criterion. The shared reference is
`baseline/Qwen-7B/Model_1/Qwen-7B_utility_of_each_goods_Model_A.csv`, the same
utility table used to construct `delta_qwen_base.json`, so W is comparable
across models.

The matched ID results are:

| model | lambda (SE) | eta (SE) | W | rational-choice rate |
|---|---:|---:|---:|---:|
| Matched local base | 7.637 (0.627) | 1.007 (0.120) | **0.744** | 0.504 |
| Qwen-own delta step 8,000 (primary) | 0.111 (0.014) | 0.504 (0.029) | **0.909** | 0.753 |
| Consensus-delta GRPO (ablation) | 0.177 (0.005) | -0.048 (0.024) | **0.890** | 0.725 |

Machine-readable combined results are under
`results/pseudo_utility_alignment/`.

Per-choice w distribution:

| w range | Matched local base | Qwen-own delta step 8,000 | Consensus-delta GRPO |
|---|---:|---:|---:|
| 0.0 <= w < 0.2 | 8.43% | 0.83% | 1.17% |
| 0.2 <= w < 0.4 | 13.40% | 4.01% | 5.67% |
| 0.4 <= w < 0.6 | 10.71% | 5.82% | 6.75% |
| 0.6 <= w < 0.8 | 9.18% | 6.69% | 6.70% |
| 0.8 <= w < 1.0 | 7.85% | 7.35% | 7.22% |
| **w = 1** | **50.42%** | **75.29%** | **72.48%** |
| **Total** | **100%** | **100%** | **100%** |

**Open:** OOD-50 uses new goods with no entries in the frozen Qwen-base utility
table, so W is currently an ID/test_goods quantity only. Computing W on OOD
needs a shared positive utility reference for the new goods (or falling back to
each model's own fitted utilities, which would not be comparable across
models). Decide before reporting OOD W.

## What Is Not Currently a Major Problem

- Training and `test_goods` contain zero repeated exact prompts and zero
  repeated exact attribute configurations.
- The shared 100 goods and 4,945 pairs make `test_goods` a valid
  within-benchmark configuration holdout when described precisely.
- Attribute configurations have strong joint effects, substantial explanatory
  power, and change at least one hard response for 42.18% of pairs.
- The learned policy is not a constant Yes/No rule.

These facts support configuration generalization. They do not remove the need
for a separately frozen final evaluation because the current Qwen-own reward
and checkpoint selection used information from `test_goods`.

## Recommended Paper Claim

> We estimate a frozen, model-specific preference ordering for Qwen and use it
> as a pseudo-utility signal for post-training. The selected model substantially
> reduces measured endowment dependence on held-out attribute configurations
> and retains low loss aversion on goods excluded from fine-tuning. We report
> loss aversion, status-quo bias, choice consistency, and preference preservation
> jointly.

This wording is supported by the matched-base, replication, OOD-50, and
prospective-configuration results. A top-conference claim about mechanism or
method superiority remains conditional on completing the matched baselines,
stronger inference, direct preference-preservation analysis, and
reproducibility work above.

## Execution Order

1. DONE — rerun the exact local base with the local post-training evaluator.
2. DONE — validate Qwen-own delta against ownership-free Qwen preferences.
3. DONE — freeze the checkpoint-selection rule and primary/secondary outcomes.
4. DONE — frozen ID-selection + OOD + GSM8K complete for seeds 1 and 2; verdict
   is 2/2 PASS, so no third seed is required by the pre-registered rule.
5. DONE — open the frozen unused-configuration suite once on only the matched
   base and two frozen seed selections; preserve it as an opened final result.
6. IN PROGRESS — archive and reproduce all Qwen-own-delta checkpoint and OOD
   artifacts. Derived results are traceable; complete raw predictions,
   adapters, environment, and durable release remain incomplete.
7. IN PROGRESS — matched SFT seeds 1 and 2 have completed 30k training with the
   full checkpoint grids, but full-grid evaluation/selection has not started.
   Sign-only remains a one-seed 6k pilot. Freeze the new untouched
   method-comparison suite and protocol before opening the full SFT
   trajectories; then complete matched evaluations and sign-only training when
   budget permits.
8. Add robust structural inference, multi-start optimization/utility
   diagnostics, and estimator-recovery checks.
9. Add direct frozen preference-preservation and prompt-semantic robustness
   tests.
10. DONE for GSM8K; run IFEval or an equivalent complementary capability test.
11. Rewrite a venue-agnostic paper with Qwen-own delta as primary and consensus
    delta as the reward-source ablation.
