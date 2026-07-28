# Known Issues & Open Questions — lambda-zero

A running log of problems encountered, potential risks, and things to verify
before they become bugs. `PAPER_READINESS.md` is the authoritative prioritized
summary for the current full-paper plan.

## Current Paper Direction — July 15, 2026

- **Primary model:** Qwen-own-delta GRPO, with checkpoint 8,000 as the current
  selected candidate.
- **Reward-source ablation:** frontier-consensus-delta GRPO.
- **Primary validation estimate:** Qwen-own-delta step 8,000 has
  `lambda = 0.111`, `eta = 0.504` on `test_goods.json`. Because this dataset
  was used to select the checkpoint, 0.111 is a validation result, not
  untouched final-test performance.
- **Intended final result:** the separately frozen new-goods evaluation. The
  current draft reports Qwen-own-delta OOD `lambda = 0.226`, `eta = 0.790`,
  and consistency `49.46%`, but the raw outputs and estimator artifacts must
  be archived before this is paper-ready.
- **Economic rationale:** use Qwen's own estimated preference ordering rather
  than imposing the preferences of frontier models. Both reward sources remain
  pseudo-utility signals, not objective cardinal ground truth.

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

**Status:** Open; see `PAPER_READINESS.md` for full specifications.

- Rerun the exact local base with the same 9,890 rows and local scorer as the
  adapters.
- Treat `test_goods` as validation because it informed Qwen-own reward
  construction and checkpoint selection.
- Freeze a joint lambda/eta/consistency checkpoint-selection rule.
- Run at least three Qwen-own-delta training seeds.
- Archive raw checkpoint and OOD outputs, estimator artifacts, checkpoint
  identities/checksums, and reproduction commands.
- Add matched SFT and preferably sign-only GRPO baselines.

### 3. Training-health interpretation
**Status:** Open after full run and new convergence plots (July 2026)
**Risk:** Medium-high — structural result is strong, but training metrics are not clean convergence evidence.

Recent plots show a high zero task-reward advantage fraction (`frac_reward_zero_std`), flat-ish positive reward, and nontrivial KL drift. This does not invalidate the held-out λ̂_after result, but it means checkpoint selection and ablation claims need care.

**Action needed:**
- Evaluate several checkpoints, not only the final checkpoint.
- Log Yes/No rates by perspective for candidate checkpoints.
- Compare structural λ̂, η̂, raw choice counts, reward, KL, entropy, and zero-std fraction.
- Treat training reward as a diagnostic, not the final model-selection metric.

### 4. Dynamic sampling implementation
**Status:** NOT implemented. Correct terminology audited and fixed July 28, 2026.
**Risk:** Medium — generation-level filtering is absent, and earlier docs overstated what exists.

99% of model outputs are "No", so most G=16 groups are all-identical and produce **zero task-reward advantage**. Use that phrase. Earlier wording here ("skip batches", "prevents degenerate gradient updates") was wrong and has been corrected — nothing is skipped and no update is prevented.

**What actually happens** (stock TRL `GRPOTrainer`, v1.3.0, no subclass — see `train/grpo_train.py`):
- `make_reward_fn()` returns all-zero task rewards for a zero-diversity group.
- TRL computes `advantages = rewards - mean_grouped_rewards` (`grpo_trainer.py`). With `scale_rewards: "none"` the std is computed **for logging only** and never divided out — advantages are mean-centred, **not** raw rewards.
- An all-identical group already has a constant reward vector, so its advantage is 0 **with or without** our zero-return branch. The branch is a numerical no-op on the gradient; what it changes is the **logged mean reward** (reported as 0.0 instead of the group's true ±|δ̃|). This is an extra reason the training-reward curve is diagnostic only — it is a conditional statistic over diverse groups.
- With `beta: 0.04` the loss is `per_token_loss + beta * per_token_kl`. At zero advantage the policy term vanishes but the KL term does not, so the group still applies a **KL-only gradient** toward the reference policy. Since `generation_batch_size = 1 × 16 = G`, one group is one optimizer step: these are **not** zero-update steps.

**What's still missing:** Generation-level filtering (true DAPO dynamic sampling — resample until a group has reward variance). Both generation and gradient compute are still spent on zero-diversity groups. A proper fix subclasses `GRPOTrainer` and overrides `_generate_completions`.

**Empirical result:** The April 27 full run logged `frac_reward_zero_std: 0.8` at step 10, meaning ~80% of prompt groups were all-identical and carried no policy-gradient signal. This metric is computed by TRL from the rewards themselves (`is_std_zero`), so our zero-return branch does not distort it.

**Action needed:**
- If rerunning training, consider generation-level filtering via a GRPOTrainer subclass.
- If zero-std fraction stays near 80%, test higher temperature, larger G, or a more targeted prompt/reward setup.
- Report the high zero-advantage rate transparently if it remains part of the final training recipe.
- In any writeup, say "zero task-reward advantage groups" — never "skipped batches" or "zero-update batches".

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
**Status:** ✅ Resolved for the core within-benchmark claim (July 15, 2026). `test_goods.json` is an in-distribution compositional holdout, not a novel-goods holdout. A separate 50-good OOD suite is available for optional external-validity analysis in `data/ood_new_goods_50.json` and `data/ood_new_goods_50_test.json`.
**Risk:** Low for the core configuration-generalization claim; external validity to unseen goods remains a separate question.

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
- Use a separately frozen suite for final primary-model claims.
- Do not use the new-goods OOD suite for checkpoint selection, and archive its
  raw outputs before presenting it as paper-ready evidence.

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
**Status:** Next major paper step
**Risk:** Low — but important for the paper narrative.

Qwen-7B λ̂_before = 11.75 was estimated with Model A (NLS), T=1. The frontier model λ̂ values from Phase 0 may have used different estimators (Model B or C) or different T values. If estimation methods differ, the λ̂ values aren't directly comparable.

**Action needed:**
- Check what estimator and T value was used for each frontier model in loss_aversion/
- If methods differ, either re-estimate with consistent methods or note the caveat in the paper

### 12. Reward function assumes single rational answer per case
**Status:** Acknowledged
**Risk:** Low — but worth noting for the paper.

The utility-based reward assumes δ̃ definitively determines which good is better. In reality, preferences are subjective — there may not be a single "rational" answer. The structural model estimates average utility, not ground truth. This is a modeling assumption worth discussing in the paper's limitations section.

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

*Last updated: July 15, 2026*
