# Known Issues & Open Questions — lambda-zero

A running log of problems encountered, potential risks, and things to verify
before they become bugs. `PAPER_READINESS.md` is the authoritative prioritized
summary for the current full-paper plan.

## Current Paper Direction — July 28, 2026

- **Primary model:** Qwen-own-delta GRPO, with checkpoint 8,000 as the current
  exploratory selected candidate.
- **Reward-source ablation:** frontier-consensus-delta GRPO.
- **Primary validation estimate:** Qwen-own-delta step 8,000 has
  `lambda = 0.111`, `eta = 0.504` on `test_goods.json`. Because this dataset
  was used to select the checkpoint, 0.111 is a validation result, not
  untouched final-test performance.
- **Confirmatory replication:** 2/2 fresh seeds passed the frozen selection,
  OOD-50, and GSM8K gates; seed 1 selected step 2,000 and seed 2 selected step
  6,000.
- **Prospective configuration result:** the frozen unused-configuration suite
  was opened once on the matched base and two selected seeds. Lambda was 5.946
  for base, 0.031 for seed 1, and −0.053 for seed 2. This is new
  configurations of familiar goods/pairs, not unseen-goods OOD.
- **Causal baselines:** infrastructure and a one-seed 6k pilot now exist. Both
  SFT and sign-only GRPO reduce lambda on validation, but the pilot cannot
  establish a winner or a confirmatory method comparison.
- **Economic rationale:** use Qwen's own estimated preference ordering rather
  than imposing the preferences of frontier models. Both reward sources remain
  pseudo-utility signals, not objective cardinal ground truth.
- **Venue status:** the project is no longer targeting AAAI-27;
  `draft/aaai27/` is a historical submission workspace.

---

## Active audit ledger — July 28, 2026

This is the canonical live problem ledger. `PAPER_READINESS.md` decides what
the paper may claim; `RESEARCH_ROADMAP.md` orders the work; `HISTORY.md` records
what already happened.

| ID | Severity | Status | Problem and evidence | Closure condition |
|---|---|---|---|---|
| `INFER-001` | P0 scientific | Open | Model-A SEs come from iid `scipy.curve_fit` covariance in `eval/core_exp_refactored.py`, despite paired perspectives, repeated pairs, and shared goods. | Pair-clustered bootstrap plus good-aware/leave-one-good-out inference and estimator-recovery coverage are implemented and reported. |
| `PAIR-001` | P0 correctness | Open | `eval/core_exp_refactored.py` pairs X/Y rows by list index, without joining or asserting `case_id`, attributes, lengths, or order. Current tracked pairs align, but a partial/asymmetric run can silently corrupt a fit. | Join by stable ID; assert one X and one Y row, matching goods/attributes, no duplicates, and expected sample size before estimation. |
| `PROV-001` | P0 provenance | Open | `eval/select_checkpoint.py` always writes “frozen final suite not opened,” while baseline provenance must state that OOD-50 and frozen-unused are already opened. A future manifest can contradict its own note. | Make suite state a required structured field and derive console/manifest text from it; regression-test both states. |
| `SELECT-001` | P0 reporting | Open | `eval/seed_replication_report.py` compares every manifest step with the 15-step grid. Adding an allowed off-grid diagnostic such as step 600 can falsely turn a complete run into `PENDING`. | Filter to `_on_grid` checkpoints before completeness checks and add an off-grid regression test. |
| `BUILD-001` | P0 reproducibility | Open | `draft/aaai27/main.tex` requires `figures/task_overview.pdf`, but the figure is untracked. The submission checker passes only because local untracked figures mask the clean-checkout failure. | Track the required vector figures or make the default documented build generate them before checking; verify from `git archive`. |
| `RESUME-001` | P1 correctness | Open | Explicit resume paths in `train/grpo_train.py` and `train/sft_train.py` are not required to belong to the current output directory and are not checked against immutable seed/config/data manifests. | Validate ancestry and the prior manifest before loading; never overwrite provenance before the check passes. |
| `ABLATION-001` | P0 identification | Open | Sign-only GRPO changes `±|delta|` to `±1` while reward-scale normalization is disabled. It therefore changes both relative case weighting and global gradient scale. | Describe the current contrast precisely and add a scale-matched sign control, or otherwise match effective reward/gradient scale. |
| `PILOT-001` | P0 provenance | Open | `results/causal_baseline_pilot/` contains only a summary JSON and hand-facing table. W, Jacobian condition numbers, and runtimes are not in the JSON; raw predictions, fit CSVs, hashes, commands, and a deterministic generator are absent. | Add immutable run/eval manifests, machine-readable inputs for every table cell, and a deterministic aggregation command before citation beyond exploratory status. |
| `ENV-001` | P1 reproducibility | Open | `train/grpo_train.py` warns and silently drops unsupported GRPO fields on library-version mismatch, including scientifically defining options. There is no repository-wide environment lock. | Hard-fail for critical keys; record exact Torch/Transformers/TRL/PEFT versions and model revision/hash in immutable run manifests. |
| `EVAL-001` | P1 correctness | Open | `eval/run_qwen_local.py` resumes an output directory keyed mainly by model name without validating adapter, base, dataset, treatment, or hashes. Different evaluations can be silently mixed. | Create and validate an immutable evaluation manifest before resume; refuse mismatches and concurrent directory reuse. |
| `PBS-001` | P1 operations | Open | Several original PBS scripts pipe Python through `tee` without `set -o pipefail`, so a failed job can appear successful and leave stale/partial inputs for estimation. | Add failure propagation consistently and test representative failing commands. |
| `ARTIFACT-001` | P0 reproducibility | Partial | Derived manifests make headline numbers traceable, but raw OOD/GSM8K generations, selected adapters, and a durable environment/package are not all available from a clean clone. | Publish or archive every claim-carrying raw artifact and adapter with hashes, licenses, exact commands, and environment metadata. |
| `DOC-001` | P1 governance | In progress | Project status has drifted across `AGENTS.md`, `CLAUDE.md`, `PROJECT_OVERVIEW.md`, the canonical manuscript, and the historical AAAI workboard. | All mirrors point to this ledger/readiness/roadmap and contain no contradictory current statuses or prohibited claims. |
| `REPO-001` | P2 hygiene | Open | Generated outputs, numbered duplicate copies, redundant delta builders/data, stale stashes/worktrees, and large loose Git objects obscure source state. | Classify or remove each item safely, update scoped ignores/build rules, then verify a clean worktree and clean-clone build. |

The full two-seed causal-baseline launch should not begin until at least
`PAIR-001`, `PROV-001`, `SELECT-001`, `RESUME-001`, `ABLATION-001`, and
`ENV-001` are resolved or explicitly frozen as documented design limitations.

---

## Critical — Must resolve before paper claims

### 1. Are δ̃ values reliable?
**Status:** ✅ Largely resolved (July 15, 2026) — validated; see evidence below

**Original concern:** Qwen-7B says No in approximately 99% of endowed prompts,
so its item and attribute utilities may be weakly identified.

**Resolution — identification survives the low trade rate.** Utilities are
estimated from **logprobs at link scale T=1**, not binary choices: P(Yes)=0.08
and P(Yes)=0.28 both decode to "No" but differ substantially. (Under a T=0 /
binary encoding the concern *would* be valid — see CLAUDE.md note #8.) Measured
on `baseline/Qwen-7B/Model_1/`:

| check | value |
|---|---|
| α (item FE) | 99/99 non-zero SE, median \|t\| = 9.6, 91.9% significant |
| β (attr FE) | 8/8 non-zero SE, median \|t\| = 27.5, 100% significant |
| δ̃ spread | sd = 0.94, range [−4.06, +3.91], 1.6% near-zero, 46.9% positive |

**Resolution — ownership-free validation (the requested action).** `sign(δ̃)` vs
Qwen's frozen ownership-free preferences (multiple paraphrases, both display
orders), chance = 50%:

| split | n compared | sign agreement |
|---|---|---|
| test_goods | 6,184 | **71.1%** |
| remaining_goods | 30,757 | **70.8%** |

By magnitude (test/remaining): |δ̃|>1.0 → **85.5% / 85.3%**; 0.5<|δ̃|≤1.0 →
67.9% / 70.0%; |δ̃|≤0.5 → 63.6% / 62.3%. Retained coverage ≈62% (stable anchors:
6,184/9,890 test; 30,757/49,450 training). Agreement rises with |δ̃| and the
reward weights by |δ̃|, so the signal is loudest where it is most trustworthy.

Reproduce: `python eval/validate_qwen_delta_anchor.py` →
`results/qwen_delta_anchor_validation.json` (committed).

**Remaining caveat:** both signals are Qwen-derived, so this is convergent
validity across two elicitation framings, not external ground truth. Human
validation on a stratified subset would strengthen it further.
See PAPER_READINESS.md item #2.

### 2. Which δ̃ source to use?
**Status:** ✅ Paper-role decision updated (July 15, 2026)

**Decision:** Use `delta_qwen_base.json` for the primary model because economic
utility is agent-specific. Use `delta_consensus_v3.json` as the reward-source
ablation/reference.

This role decision does not resolve issue 1 or the indirect leakage in issue 9.
The paper must report the trade-off honestly: Qwen-own delta currently has
lower estimated lambda, while consensus has lower eta and higher OOD
consistency.

### 2A. Full-paper blockers not captured by the original issue numbering

**Status:** Reconciled July 28; see `PAPER_READINESS.md` for full specifications.

**Resolved:**

- exact matched local base evaluation;
- validation-only treatment of `test_goods`;
- frozen joint checkpoint-selection rule;
- genuinely independent seed propagation;
- 2/2 confirmatory seed result under the pre-registered rule;
- ownership-free reward-anchor validation; and
- one-shot prospective unused-configuration evaluation with provenance.

**Partially resolved:**

- derived result manifests are committed, but the full raw prediction,
  adapter, environment, and durable-release package is incomplete;
- matched SFT and sign-only GRPO code is hardened and a pilot exists, but the
  confirmatory two-seed 30k comparison and new untouched suite do not; and
- multi-start/conditioning diagnostics exist, but pair/good-aware inference
  and estimator-recovery validation remain incomplete.

**Open:**

- direct preservation of frozen ownership-free preferences after training;
- prompt-semantic counterbalancing;
- IFEval or another complementary capability benchmark; and
- a leakage-clean reward-source rerun if the paper requires a fully clean
  Qwen-own construction rather than transparent validation/final-test
  separation.

### 3. Training-health interpretation
**Status:** Open after full run and new convergence plots (July 2026)
**Risk:** Medium-high — structural result is strong, but training metrics are not clean convergence evidence.

Recent plots show high zero-std/DAPO filtering, flat-ish positive reward, and nontrivial KL drift. This does not invalidate the held-out λ̂_after result, but it means checkpoint selection and ablation claims need care.

**Action needed:**
- Evaluate several checkpoints, not only the final checkpoint.
- Log Yes/No rates by perspective for candidate checkpoints.
- Compare structural λ̂, η̂, raw choice counts, reward, KL, entropy, and zero-std fraction.
- Treat training reward as a diagnostic, not the final model-selection metric.

### 4. Dynamic sampling implementation
**Status:** Partially handled during training; still important for ablations.
**Risk:** Medium — generation-level filtering still missing.

The base model produces "No" almost all the time. When all completions in a
prompt group receive the same task reward, the normalized **task-reward
advantage** is zero.

**What's done:** `train/reward_functions.py` `make_reward_fn()` returns
all-zero task rewards for any group where all G completions are identical
(detected after parsing). This prevents that group from supplying a misleading
task-reward direction.

**Important qualification:** this is not necessarily a zero **total** update.
The GRPO objective also uses `beta = 0.04` for KL regularization, so an
all-identical/zero-advantage group may still contribute a KL-only gradient.
Documentation and diagnostics should say “zero task advantage,” not “zero
gradient.”

**What's still missing:** Generation-level filtering. The current implementation still spends generation compute on all-identical batches before discarding them. A proper fix subclasses `GRPOTrainer` and overrides `_generate_completions` or `compute_loss` to skip batches entirely.

**Empirical result:** The April 27 full run logged `frac_reward_zero_std: 0.8`
at step 10, meaning roughly 80% of prompt groups supplied no differentiated
task-reward signal.

**Action needed:**
- If rerunning training, consider generation-level filtering via a GRPOTrainer subclass.
- If zero-std fraction stays near 80%, test higher temperature, larger G, or a more targeted prompt/reward setup.
- Report the high filtering rate transparently if it remains part of the final training recipe.

### 5. T=0 vs T=1 estimation bug
**Status:** ✅ Resolved (April 14, 2026)
**What happened:** Running NLS with T=0 for a Style A model (with logprobs) caused the Jacobian to be all zeros. NLS couldn't converge, silently returned λ̂ = 0 with zero standard errors. Looked like a valid result but was completely meaningless.
**Fix:** Set T=1 for Style A models. T=0 is only for Style B (binary, no logprobs).
**Lesson:** Always check that standard errors are non-zero after estimation. All-zero SEs = degenerate solution.

---

## Medium — Should address in ablations

### 6. Edge cases where |δ̃| ≈ 0
**Status:** Open
**Risk:** Medium — noisy reward signal for ambiguous cases.

When δ̃ ≈ 0, both goods are nearly equivalent and either choice is rational. The reward function (Version 2) naturally handles this since reward = ±|δ̃| ≈ 0 for these cases. But they still consume training compute without contributing useful signal.

**Possible actions:**
- Filter out cases where |δ̃| < some threshold from training data
- Or keep them — the near-zero reward means they contribute almost no gradient anyway
- Monitor whether these cases cause instability in advantage normalization

### 7. Temperature adequacy for output diversity
**Status:** Empirically strained at temp=1.5, G=16
**Risk:** High — observed `frac_reward_zero_std: 0.8` means most generation compute does not update the model.

Previously noted as a risk with G=8, temp=0.7. Both values have since been increased:
- G: 8 → 16 (more chances to see a "Yes" in each group)
- Temperature: 0.7 → 1.5 (flattens the distribution, unlike temp < 1 which sharpens it)

**Action needed:**
- If rerunning training, consider temp 1.8-2.0, G=32, or generation-level filtering.
- Log `sum(reward) / len(reward)` per batch during training to check signal quality.
- Compare λ̂ across checkpoints so the final result is not selected from noisy training reward alone.

### 8. LoRA rank 16 — is it sufficient?
**Status:** Assumed adequate
**Risk:** Low-medium — insufficient capacity could limit how far λ can be reduced.

Rank 16 is standard for 7B models, but this task may require very specific behavioral changes. If training plateaus at a λ̂_after that's still high, insufficient LoRA capacity could be a factor.

**Action needed:**
- Start with rank 16
- If results plateau, try rank 32 or 64 as ablation
- Monitor training loss — if it stops decreasing but λ̂ is still high, capacity may be the bottleneck

---

## Low — Monitor during evaluation

### 9. Generalization beyond training goods
**Status:** ✅ Resolved for the core within-benchmark claim; confirmatory
configuration and OOD-50 evidence now exist. `test_goods.json` remains
validation, not a novel-goods holdout.
**Risk:** Low for the narrow configuration-generalization claim; broader
external validity and method-comparison validity remain separate questions.

An exact split audit found no duplicated `(goods pair, attribute configuration)` cases between `remaining_goods.json` and `test_goods.json`. However, both files use exactly the same 100 goods, the same 4,945 goods pairs, and the same global set of 81 attribute codes. For every shared pair, training contains 10 configurations and test contains 2 different configurations. The current test therefore measures interpolation to unseen configurations of familiar goods pairs, not generalization to new goods or attributes.

The held-out attribute changes are substantively meaningful. A paired
within-pair analysis rejects the joint null of no attribute differentiation
under both pair-clustered inference (χ²(8) = 5,097.22) and a leave-one-good-out
jackknife (χ²(8) = 756.05). Attribute profiles explain 48.8% of within-pair,
within-perspective variation, and at least one perspective's binary answer
changes for 42.18% of goods pairs. Full details are in
`ATTRIBUTE_EFFECTS.md`.

The Qwen-delta ablation has an additional indirect leakage path: Qwen's base utility table was fitted from 9,950 baseline cases (60 trial + 9,890 test), then used to construct rewards for `remaining_goods.json`. The adapter never sees the exact test prompts during GRPO, but test responses informed the utility model that generated its training reward. Results on `test_goods.json` should therefore be described as checkpoint-selection/compositional results rather than a final untouched test.

**Decision:**
- Use `test_goods.json` as validation for the primary Qwen-own-delta model.
- Describe it as within-benchmark configuration generalization, not
  unseen-goods transfer.
- The separately frozen unused-configuration suite has now been evaluated
  once. Preserve it as opened evidence and never use it for training,
  checkpoint selection, or method revision.
- OOD-50 provides the unseen-goods evidence for the two frozen selected seeds;
  do not reuse it for method selection.
- A claim comparing SFT, sign-only GRPO, and magnitude-weighted GRPO requires a
  newly frozen untouched suite.

### 10. KL penalty β = 0.04 — calibration
**Status:** Assumed reasonable
**Risk:** Low — but wrong β could cause undertrained (too high) or degenerate (too low) model.

β = 0.04 is tighter than DeepSeek-R1's 0.001, which makes sense for staying close to the base model. But it hasn't been validated for this specific task.

**Action needed:**
- Monitor KL divergence during training — is it staying in a reasonable range?
- If model barely changes: β may be too high, try 0.01
- If model degenerates (starts outputting garbage): β may be too low, try 0.1
- Include β sweep in Phase 6 ablations

### 11. Cross-model comparability
**Status:** Matched before/after comparison resolved; frontier/human comparison
still open.
**Risk:** Medium for broad superiority claims; low for the exact-local-base
causal comparison.

The paper now uses the exactly matched local base (`lambda = 7.637`,
`eta = 1.007`) for before/after claims. The historical Qwen value 11.75 and
frontier values may use different endpoints, samples, probability scorers, or
estimators and are not headline comparators.

**Action needed:**
- Check what estimator and T value was used for each frontier model in loss_aversion/
- If broad comparison is retained, re-estimate with consistent methods;
  otherwise omit the superiority claim rather than relying on a footnote.

### 12. Reward function assumes single rational answer per case
**Status:** Acknowledged
**Risk:** Low — but worth noting for the paper.

The utility-based reward assumes δ̃ definitively determines which good is better. In reality, preferences are subjective — there may not be a single "rational" answer. The structural model estimates average utility, not ground truth. This is a modeling assumption worth discussing in the paper's limitations section.

### 13. Structural outcome semantics are easy to mislabel
**Status:** Documentation correction in progress
**Risk:** Medium — a Yes/No label reversal can invert the verbal
interpretation without changing a numerical fit.

`eval/core_exp_refactored.py` assigns the dependent variable to the second
entry of `[P(Yes), P(No)]`. The logistic structural equation therefore models
`P(No)`, i.e. the probability of keeping the currently endowed good. Some
comments and older documents call the modeled quantity `P(Yes)`.

**Action needed:**

- use `P(No)` consistently in the manuscript and protocol documents;
- add an estimator semantic/unit test with hand-constructed X/Y cases; and
- avoid changing the shared estimator until the test fixes the intended
  convention explicitly.

### 14. Repository source/output hygiene
**Status:** Source-of-truth problem substantially resolved July 28; local
cleanup remains.
**Risk:** Low scientifically, medium for reproducibility and safe merging.

Commit `6c31e81` tracks the canonical manuscript source tree. The July 28
follow-up reconciliation reduces the redundant root `training_overview.tex` to
a deprecated pointer. The remaining untracked files are mostly generated
TeX/PDF outputs, local visual assets, and duplicate files whose names end in
` 2`.

**Action needed:**

- decide which generated figures are required to build the paper from a clean
  clone and either track them or make generation part of the build;
- keep logs, PDFs, scratch outputs, and duplicate ` 2` copies out of commits;
- review `draft/archive/` and `draft/assets/` before adding them; and
- reconcile `train/build_utility_delta_file.py` with
  `data/deltas/build_delta_qwen_base.py`; they overlap in purpose but use
  different paths/schema, and the former is currently unreferenced; and
- push local main only after reviewing the documentation changes and the
  untracked list.

---

## Resolved

### T=0 estimation bug
See item 5 above. ✅ Fixed by setting T=1.

### δ̃ reliability
The old resolution by consensus is retained as historical context. It is
superseded for the primary Qwen-own-delta paper direction; see items 1 and 2.

### vLLM dependency on NSCC
vLLM was disabled after compatible versions failed on NSCC. The working path uses plain HF generation and `eval/run_qwen_local.py`; keep vLLM as optional future acceleration only.

---

*Last updated: July 28, 2026*
