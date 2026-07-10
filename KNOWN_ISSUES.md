# Known Issues & Open Questions — lambda-zero

A running log of problems encountered, potential risks, and things to verify before they become bugs.

---

## Critical — Must resolve before paper claims

### 1. Are δ̃ values reliable?
**Status:** ✅ Resolved (April 2026)

**Finding:** Qwen-7B's own δ̃ values are unreliable — α/β estimates are all near zero because the model says No 99% of the time, giving NLS no variation to learn from. δ̃ values reflected reference category geometry, not real preferences.

**Solution:** Use consensus δ̃ from 6 frontier models (v3: GPT-4o, GPT-5, GPT-5.2, Gemini, Llama-70B, DeepSeek-R1). Excluded Claude (0.05% Yes rate, degenerate δ̃ in thousands), Apertus-70B (0.5% Yes rate), and GPT-3.5 (2.8% Yes rate). 64% of cases have unanimous sign agreement across all 6 models. File: `data/deltas/delta_consensus_v3.json`.

### 2. Which δ̃ source to use?
**Status:** ✅ Resolved (April 2026)

**Decision:** Use `delta_consensus_v3.json` (mean δ̃ from 6 strong frontier models) with Option A — include all cases, even those where models disagree on sign. The |δ̃| weighting in the reward function naturally downweights ambiguous cases (small |mean δ̃| → small reward magnitude → small gradient).

DeepSeek-R1 standalone δ̃ also saved as `data/deltas/delta_deepseek.json` for potential ablation. Qwen's own NLS delta is now saved as `data/deltas/delta_qwen_base.json` for the active ablation.

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

99% of model outputs are "No." Without DAPO-style dynamic sampling, almost every training batch has zero advantage and zero gradient.

**What's done:** `train/reward_functions.py` `make_reward_fn()` returns all-zero rewards for any group where all G completions are identical (detected after parsing). This prevents degenerate gradient updates.

**What's still missing:** Generation-level filtering. The current implementation still spends generation compute on all-identical batches before discarding them. A proper fix subclasses `GRPOTrainer` and overrides `_generate_completions` or `compute_loss` to skip batches entirely.

**Empirical result:** The April 27 full run logged `frac_reward_zero_std: 0.8` at step 10, meaning ~80% of prompt groups were all-identical and produced zero useful GRPO signal.

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
**Status:** `test_goods.json` is an in-distribution compositional holdout, not a novel-goods holdout. A separate 50-good OOD suite is now available in `data/ood_new_goods_50.json` and `data/ood_new_goods_50_test.json`.
**Risk:** Medium — the model may learn item-, pair-, or attribute-specific policies rather than a general reduction in loss aversion.

An exact split audit found no duplicated `(goods pair, attribute configuration)` cases between `remaining_goods.json` and `test_goods.json`. However, both files use exactly the same 100 goods, the same 4,945 goods pairs, and the same global set of 81 attribute codes. For every shared pair, training contains 10 configurations and test contains 2 different configurations. The current test therefore measures interpolation to unseen configurations of familiar goods pairs, not generalization to new goods or attributes.

The Qwen-delta ablation has an additional indirect leakage path: Qwen's base utility table was fitted from 9,950 baseline cases (60 trial + 9,890 test), then used to construct rewards for `remaining_goods.json`. The adapter never sees the exact test prompts during GRPO, but test responses informed the utility model that generated its training reward. Results on `test_goods.json` should therefore be described as checkpoint-selection/compositional results rather than a final untouched test.

**Action needed:**
- Keep `test_goods.json` as the ID/compositional checkpoint-selection set.
- Freeze checkpoint choice before running the OOD suite; never use OOD results for tuning or checkpoint selection.
- Evaluate base Qwen, consensus-GRPO, and the selected Qwen-delta checkpoint on the same 50-good OOD suite.
- Report ID/compositional and OOD estimates separately, including both λ̂ and η̂.

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
See items 1 and 2 above. ✅ Resolved by using frontier model consensus δ̃ (v3).

### vLLM dependency on NSCC
vLLM was disabled after compatible versions failed on NSCC. The working path uses plain HF generation and `eval/run_qwen_local.py`; keep vLLM as optional future acceleration only.

---

*Last updated: July 9, 2026*
