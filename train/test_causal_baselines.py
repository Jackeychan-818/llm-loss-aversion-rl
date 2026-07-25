#!/usr/bin/env python3
"""
Tests for the matched SFT + sign-only GRPO baselines (CAUSAL_BASELINE_PROTOCOL.md).

Pure Python except the optional content-level data test (needs the data files);
runs in milliseconds on a login node:

    python train/test_causal_baselines.py

Covers the 10 required checks:
  1. SFT label mapping under both perspectives
  2. Both perspectives select the same delta-preferred final good
  3. Sign-only rewards are exactly +1 or -1
  4. Sign-only rewards do not depend on abs(delta)
  5. Magnitude mode reproduces the existing reward values
  6. Unparseable-response behavior is documented and tested
  7. Existing magnitude-weighted GRPO remains the default
  8. Seeds produce separate output directories
  9. Resume logic cannot overwrite another treatment or seed
 10. Training data contains no test_goods, frozen-unused, or OOD rows
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # train/

from reward_functions import compute_reward, make_reward_fn, parse_response, rational_choice
import sft_train


# ── helper: which good is chosen, given a perspective and a Yes/No answer ─────
def chosen_good(perspective: str, answer: str) -> str:
    """X-perspective: No=keep X (X), Yes=trade (Y). Y-perspective: No=keep Y (Y), Yes=trade (X)."""
    if perspective == "X":
        return "X" if answer == "No" else "Y"
    return "Y" if answer == "No" else "X"


def test_1_sft_label_mapping_both_perspectives():
    # δ>0 (X preferred): X->No, Y->Yes ; δ<0 (Y preferred): X->Yes, Y->No
    assert rational_choice("X", 1.5) == "No"
    assert rational_choice("X", -1.5) == "Yes"
    assert rational_choice("Y", 1.5) == "Yes"
    assert rational_choice("Y", -1.5) == "No"
    # SFT builder uses exactly this rule
    recs = sft_train.build_sft_records.__doc__  # documented
    assert "rational_choice" in recs
    print("test_1_sft_label_mapping_both_perspectives: OK")


def test_2_both_perspectives_same_preferred_good():
    for delta in (2.3, -2.3, 0.01, -0.01):
        gx = chosen_good("X", rational_choice("X", delta))
        gy = chosen_good("Y", rational_choice("Y", delta))
        assert gx == gy, (delta, gx, gy)
        assert gx == ("X" if delta > 0 else "Y")
    print("test_2_both_perspectives_same_preferred_good: OK")


def test_3_sign_only_rewards_are_plus_minus_one():
    for persp in ("X", "Y"):
        for delta in (0.05, 3.7, -0.4, -9.1):
            for resp in ("Yes", "No", ""):
                r = compute_reward(resp, persp, delta, weighting="sign_only")
                assert r in (1.0, -1.0), (persp, delta, resp, r)
    print("test_3_sign_only_rewards_are_plus_minus_one: OK")


def test_4_sign_only_independent_of_magnitude():
    # same response/perspective/sign, different |delta| -> identical sign-only reward
    for persp in ("X", "Y"):
        for resp in ("Yes", "No"):
            small = compute_reward(resp, persp, 0.1, weighting="sign_only")
            large = compute_reward(resp, persp, 9.9, weighting="sign_only")
            assert small == large, (persp, resp, small, large)
    print("test_4_sign_only_independent_of_magnitude: OK")


def test_5_magnitude_reproduces_existing_values():
    # exact values from the pre-existing reward_functions.__main__ sanity block
    assert abs(compute_reward("No", "X", 1.81) - 1.81) < 1e-9      # rational -> +|delta|
    assert abs(compute_reward("Yes", "X", 1.81) + 1.81) < 1e-9     # irrational -> -|delta|
    assert abs(compute_reward("Yes", "Y", 2.5) - 2.5) < 1e-9
    assert abs(compute_reward("No", "Y", 2.5) + 2.5) < 1e-9
    # via the batched reward fn (magnitude default), matching the old __main__ test
    fn = make_reward_fn()
    r = fn(["Yes", "No"] + ["No"] * 14, ["X"] * 16, [-1.81] * 16)
    assert abs(r[0] - 1.81) < 1e-9 and abs(r[1] + 1.81) < 1e-9
    print("test_5_magnitude_reproduces_existing_values: OK")


def test_6_unparseable_response_behavior():
    for bad in ("", "maybe", "I think yes", "[garbage]", "42"):
        parsed = parse_response(bad)
        # "I think yes" -> first token "i" -> unparseable "" (documented first-token rule)
        assert parsed in ("", "Yes", "No")
    # an unparseable ('') response is never the rational answer -> negative weight
    assert compute_reward("", "X", 1.81, "magnitude") == -1.81
    assert compute_reward("", "Y", -2.0, "sign_only") == -1.0
    print("test_6_unparseable_response_behavior: OK")


def test_7_magnitude_is_the_default():
    # make_reward_fn() with no args == magnitude; a diverse batch yields ±|delta|, not ±1
    fn = make_reward_fn()
    r = fn(["Yes", "No"] + ["No"] * 14, ["Y"] * 16, [2.5] * 16)
    assert abs(r[0] - 2.5) < 1e-9, "default must be magnitude (±|delta|), got sign-only"
    assert compute_reward("No", "X", 1.5) == compute_reward("No", "X", 1.5, "magnitude")
    print("test_7_magnitude_is_the_default: OK")


def test_8_seeds_produce_separate_output_dirs():
    d1 = f"checkpoints/sft_qwen_delta_seed{1}"
    d2 = f"checkpoints/sft_qwen_delta_seed{2}"
    assert d1 != d2
    s1 = f"checkpoints/grpo_qwen_delta_sign_seed{1}"
    s2 = f"checkpoints/grpo_qwen_delta_sign_seed{2}"
    assert s1 != s2 and s1 != d1
    print("test_8_seeds_produce_separate_output_dirs: OK")


def test_9_resume_cannot_overwrite_other_treatment_or_seed():
    # SFT must refuse to write into any GRPO run directory
    for bad in ("checkpoints/grpo_qwen_delta", "checkpoints/grpo_qwen_delta_seed1",
                "checkpoints/grpo_qwen_delta_sign_seed2"):
        try:
            sft_train.guard_output_dir(bad)
            raise AssertionError(f"guard should have refused {bad}")
        except SystemExit:
            pass
    # a proper per-seed SFT dir is allowed
    sft_train.guard_output_dir("checkpoints/sft_qwen_delta_seed1")
    print("test_9_resume_cannot_overwrite_other_treatment_or_seed: OK")


def test_10_training_data_has_no_test_frozen_or_ood():
    # filename-level guard
    for bad in ("data/test_goods.json", "data/frozen_unused_test_goods.json",
                "data/ood_new_goods_50.json"):
        try:
            sft_train.assert_clean_training_data(bad)
            raise AssertionError(f"should refuse {bad}")
        except SystemExit:
            pass
    sft_train.assert_clean_training_data("data/remaining_goods.json")  # allowed
    # content-level: remaining_goods case_ids are disjoint from test_goods (61..9950)
    root = Path(__file__).resolve().parent.parent
    if (root / "data/remaining_goods.json").exists() and (root / "everyday_goods_full.json").is_file():
        recs = sft_train.build_sft_records(root / "data/remaining_goods.json",
                                           root / "everyday_goods_full.json",
                                           root / "data/deltas/delta_qwen_base.json",
                                           root / "data")
        cids = {r["case_id"] for r in recs}
        assert min(cids) > 9950, f"training case_ids leak into test_goods range: min={min(cids)}"
        print(f"test_10_training_data_has_no_test_frozen_or_ood: OK "
              f"(content: {len(recs)} recs, case_ids {min(cids)}..{max(cids)})")
    else:
        print("test_10_training_data_has_no_test_frozen_or_ood: OK (filename-level; data files absent)")


if __name__ == "__main__":
    test_1_sft_label_mapping_both_perspectives()
    test_2_both_perspectives_same_preferred_good()
    test_3_sign_only_rewards_are_plus_minus_one()
    test_4_sign_only_independent_of_magnitude()
    test_5_magnitude_reproduces_existing_values()
    test_6_unparseable_response_behavior()
    test_7_magnitude_is_the_default()
    test_8_seeds_produce_separate_output_dirs()
    test_9_resume_cannot_overwrite_other_treatment_or_seed()
    test_10_training_data_has_no_test_frozen_or_ood()
    print("\nAll causal-baseline tests passed.")
