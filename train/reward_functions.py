#!/usr/bin/env python3
"""
Reward functions for lambda-zero GRPO training.

Implements utility-weighted reward using consensus δ̃ from 6 frontier models.
Reward formula: +|δ̃| for rational choice, −|δ̃| for irrational/unparseable.

See CLAUDE.md §Key Design Decisions → Reward function for full rationale.
Reward formula verbatim from CLAUDE.md:
    if perspective == "X":
        rational = "No" if delta > 0 else "Yes"
    else:
        rational = "Yes" if delta > 0 else "No"
    return abs(delta) if response == rational else -abs(delta)
"""

import re


def parse_response(completion: str) -> str:
    """Extract Yes/No from a model completion. Returns 'Yes', 'No', or '' if unparseable."""
    text = completion.strip()
    # Strip reasoning traces from thinking models (e.g. DeepSeek-R1 format)
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>"):].strip()
    tokens = text.split()
    if not tokens:
        return ""
    first = re.sub(r"[^a-zA-Z]", "", tokens[0]).lower()
    if first in ("yes", "y"):
        return "Yes"
    if first in ("no", "n"):
        return "No"
    return ""


def compute_reward(response: str, perspective: str, delta: float) -> float:
    """
    Utility-weighted reward.

    delta = δ̃ = U_X - U_Y  (mean across 6 frontier models, from delta_consensus_v3.json)
    perspective: "X" (endowed with X, offered Y) or "Y" (endowed with Y, offered X)

    Rational choices:
      X-perspective: δ̃ > 0 → X better → keep X → "No"
                     δ̃ < 0 → Y better → trade   → "Yes"
      Y-perspective: δ̃ > 0 → X better → trade for X → "Yes"
                     δ̃ < 0 → Y better → keep Y      → "No"

    Unparseable responses ('') are treated as irrational → −|δ̃|.
    δ̃ == 0.0 cases are filtered out in prompt_builder.py; guard here too.
    """
    if delta == 0.0:
        return 0.0

    if perspective == "X":
        rational = "No" if delta > 0 else "Yes"
    else:
        rational = "Yes" if delta > 0 else "No"

    magnitude = abs(delta)
    return magnitude if response == rational else -magnitude


def make_reward_fn():
    """
    Return a reward function compatible with TRL's GRPOTrainer.

    TRL calls this as:
        reward_fn(completions, perspective=[...], delta=[...], **kwargs)

    All arguments are lists of length (batch_size × num_generations).
    With batch_size=1 and num_generations=16, this is 16 items per call:
      - completions: 16 generated strings (the group)
      - perspective: ["X"]*16 or ["Y"]*16  (same value repeated G times)
      - delta: [float]*16                  (same value repeated G times)

    DAPO-style dynamic sampling (KNOWN_ISSUES.md #2):
    If all G completions in a group are identical (all "No" at initialization,
    likely), standard GRPO gets zero advantage and zero gradient. This wastes
    a gradient step. We return all-zero rewards explicitly to avoid numerical
    noise from normalizing a constant reward vector.

    Note: this saves gradient compute but not generation compute. For generation-
    level filtering, subclass GRPOTrainer and override _generate_completions.
    """
    def reward_fn(completions, perspective, delta, **kwargs):
        parsed = [parse_response(c) for c in completions]

        # DAPO-style: skip batches with zero diversity (all outputs identical)
        if len(set(parsed)) == 1:
            return [0.0] * len(completions)

        return [
            compute_reward(p, persp, float(d))
            for p, persp, d in zip(parsed, perspective, delta)
        ]

    return reward_fn


if __name__ == "__main__":
    # Quick sanity check
    fn = make_reward_fn()

    # All "No" → DAPO filter → all zeros
    completions_all_no = ["No"] * 16
    r = fn(completions_all_no, ["X"] * 16, [-1.81] * 16)
    assert all(x == 0.0 for x in r), "DAPO filter should zero out all-identical batches"
    print("DAPO filter (all No): OK — all zeros")

    # Mixed batch with X-perspective, delta < 0 → rational = "Yes"
    completions_mixed = ["Yes", "No"] + ["No"] * 14
    r = fn(completions_mixed, ["X"] * 16, [-1.81] * 16)
    print(f"Mixed batch (X-perspective, delta=-1.81, rational=Yes):")
    print(f"  Yes reward: {r[0]:.4f}  (expected +1.81)")
    print(f"  No  reward: {r[1]:.4f}  (expected -1.81)")
    assert abs(r[0] - 1.81) < 1e-6
    assert abs(r[1] + 1.81) < 1e-6

    # Y-perspective, delta > 0 → rational = "Yes"
    r = fn(["Yes", "No"] + ["No"] * 14, ["Y"] * 16, [2.5] * 16)
    print(f"Y-perspective, delta=+2.5, rational=Yes:")
    print(f"  Yes reward: {r[0]:.4f}  (expected +2.5)")
    print(f"  No  reward: {r[1]:.4f}  (expected -2.5)")
    assert abs(r[0] - 2.5) < 1e-6
    assert abs(r[1] + 2.5) < 1e-6

    print("All checks passed.")
