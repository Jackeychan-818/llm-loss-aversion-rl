# Scale-matched reward ablation (Task 3 / ABLATION-001)

*Specified 2026-07-30. No training run; no scale tuned on evaluation results.*

## Problem (ABLATION-001)

The sign-only condition replaces `±|δ̃|` with `±1`. That changes **two** things at
once versus the magnitude reward:

1. **per-case weighting** — magnitude gives each case a different weight `|δ̃|`;
   sign-only gives every case weight 1;
2. **global scale** — the average reward magnitude *increases* from
   `E[|δ̃|] ≈ 0.685` to 1 (since the constant 1 exceeds the mean absolute δ̃).
   Scale-matched `±0.685` instead holds this chosen first absolute moment fixed.

So a sign-only-vs-magnitude difference cannot be attributed to the informational
value of per-case magnitude alone.

## Fix — a scale-matched control

Add a third weighting, `scale_matched`, that keeps sign-only's **per-case-uniform
magnitude** but sets the constant `c` so the **global reward scale matches**
`±|δ̃|`:

| weighting | per-case weight | isolates |
|---|---|---|
| `magnitude` | `|δ̃|` | — (confirmatory default, unchanged) |
| `sign_only` | `1` | sign only (changes weighting **and** scale) |
| `scale_matched` | `c` (constant) | **per-case weighting**, holding global scale fixed |

- **magnitude vs scale_matched** → isolates the value of per-case magnitude
  weighting at matched global scale.
- **sign_only vs scale_matched** → isolates the pure global-scale effect.

## Scale-matching rule and moments

`c` is characterised from the **frozen training deltas** (`delta_qwen_base.json`,
case IDs 9,951–59,400; non-zero only), by `train/build_scale_matched_spec.py`:

| rule | `c` | matches |
|---|---:|---|
| `mean_abs` (default) | **0.685029** | `E[|δ̃|]` — first absolute moment / mean reward magnitude |
| `rms` | **0.947166** | `sqrt(E[δ̃²])` — second moment / RMS effective advantage scale |

The default is `mean_abs` (most direct "same average reward scale"). `rms` is
recorded as the second-moment alternative: within a GRPO group the advantage
magnitude is ∝ the reward weight, so RMS matching equalises the root-mean-square
effective advantage across prompts.

## Documented limitations

GRPO mean-centres advantages by the **group mean** (`scale_rewards: "none"` still
mean-centres) and ~80% of groups carry **zero task-reward advantage**. Therefore
**no constant `c` reproduces the realized per-step gradient** of the magnitude
reward — `c` matches only the chosen moment of the reward-magnitude distribution.
`scale_matched` removes per-case weighting while matching global scale;
`sign_only` removes both. This control narrows, but does not fully eliminate, the
confound, and that is stated in the spec JSON.

## Implementation (default path unchanged)

- `train/reward_functions.py`: `compute_reward(..., weighting, scale_constant)`,
  `make_reward_fn(weighting, scale_constant)`, `characterize_deltas`,
  `compute_scale_constant`. The `magnitude` and `sign_only` paths are byte-for-byte
  behaviourally unchanged (regression-tested).
- `train/grpo_train.py`: `--reward_weighting scale_matched` + `--scale_constant`
  (or YAML `scale_constant`); refuses to run scale_matched without a positive
  constant; records `scale_constant` in the run manifest.
- **ENV-001 hard-fail**: `filter_supported_config_kwargs` now hard-fails if the
  installed TRL lacks any **algorithm-defining** key (`beta`, `epsilon`,
  `loss_type`, `scale_rewards`, `num_generations`, `temperature`,
  `mask_truncated_completions`, `max_completion_length`) instead of silently
  dropping it; non-critical keys still warn.
- `train/configs/qwen25_7b_qwen_delta_scale_matched.yaml`: identical to the
  sign-only config except `reward_weighting: scale_matched` + `scale_constant`.

## Files

| file | contents |
|---|---|
| `train/build_scale_matched_spec.py` | deterministic `c` computation (`--check`) |
| `results/scale_matched_reward/scale_matched_reward_spec.json` | characterization + `c`, delta SHA |
| `train/configs/qwen25_7b_qwen_delta_scale_matched.yaml` | frozen config |
| `train/test_scale_matched_reward.py` | 29 tests (signs, scale calc, parsing, default-path equivalence, deterministic manifest, ENV-001 hard-fail) |

## ABLATION-001 status

**Not yet closable.** The scale-matched *specification, code, and tests* exist,
but closing ABLATION-001 requires the scale-matched (and sign-only) runs to be
trained and compared under `METHOD_COMPARISON_PROTOCOL.md`. This is a GPU-phase,
budget-gated step. The ledger entry is updated to reflect spec-complete /
run-pending.
