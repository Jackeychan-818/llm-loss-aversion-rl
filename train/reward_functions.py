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
from typing import Any


def completion_to_text(completion: Any) -> str:
    """Normalize TRL plain or conversational completions to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        if not completion:
            return ""
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(completion)


def parse_response(completion: Any) -> str:
    """Extract Yes/No from a model completion. Returns 'Yes', 'No', or '' if unparseable."""
    text = completion_to_text(completion).strip()
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


def rational_choice(perspective: str, delta: float) -> str:
    """The rational (frozen-preferred) Yes/No answer for a perspective and δ̃.

    SINGLE SOURCE OF TRUTH for "which answer is rational", reused by both the
    GRPO reward (compute_reward) and the SFT target builder (train/sft_train.py)
    so the two treatments cannot silently diverge.

      X-perspective (endowed X, offered Y): δ̃>0 → X better → keep  → "No"
                                            δ̃<0 → Y better → trade → "Yes"
      Y-perspective (endowed Y, offered X): δ̃>0 → X better → trade → "Yes"
                                            δ̃<0 → Y better → keep  → "No"

    Both perspectives therefore encode the SAME preferred good (δ̃>0 ⇒ X, δ̃<0 ⇒ Y).
    Returns "" for δ̃ == 0 (no preferred good; such cases are filtered upstream).
    """
    if delta == 0.0:
        return ""
    if perspective == "X":
        return "No" if delta > 0 else "Yes"
    return "Yes" if delta > 0 else "No"


def compute_reward(response: str, perspective: str, delta: float,
                   weighting: str = "magnitude") -> float:
    """
    Reward for a Yes/No response.

    delta = δ̃ = U_X - U_Y  (Qwen-own or consensus, from the configured delta file)
    perspective: "X" (endowed with X, offered Y) or "Y" (endowed with Y, offered X)
    weighting:
      - "magnitude" (DEFAULT, the confirmatory behavior): ±|δ̃|.
      - "sign_only" (roadmap Priority-1 ablation):        ±1, independent of |δ̃|.

    In both modes the SIGN is +weight for the rational choice, -weight otherwise;
    only the magnitude of the weight differs. Unparseable responses ('') are
    irrational → -weight. δ̃ == 0.0 cases are filtered in prompt_builder.py;
    guarded here too (return 0.0).
    """
    if delta == 0.0:
        return 0.0
    if weighting not in ("magnitude", "sign_only"):
        raise ValueError(f"unknown reward weighting {weighting!r}; "
                         "expected 'magnitude' or 'sign_only'")

    rational = rational_choice(perspective, delta)
    weight = abs(delta) if weighting == "magnitude" else 1.0
    return weight if response == rational else -weight


def make_reward_fn(weighting: str = "magnitude"):
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
    if weighting not in ("magnitude", "sign_only"):
        raise ValueError(f"unknown reward weighting {weighting!r}; "
                         "expected 'magnitude' or 'sign_only'")

    def reward_fn(completions, perspective, delta, **kwargs):
        parsed = [parse_response(c) for c in completions]

        # DAPO-style: skip batches with zero diversity (all outputs identical).
        # Identical for both weightings — the diversity filter is orthogonal to
        # the reward magnitude, so the sign-only contrast is purely the weight.
        if len(set(parsed)) == 1:
            return [0.0] * len(completions)

        return [
            compute_reward(p, persp, float(d), weighting=weighting)
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

    # Conversational TRL format → parse assistant content
    structured = [[{"role": "assistant", "content": "Yes"}], [{"role": "assistant", "content": "No"}]]
    r = fn(structured, ["Y", "Y"], [2.5, 2.5])
    assert abs(r[0] - 2.5) < 1e-6
    assert abs(r[1] + 2.5) < 1e-6

    print("All checks passed.")
