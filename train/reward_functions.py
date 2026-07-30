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


VALID_WEIGHTINGS = ("magnitude", "sign_only", "scale_matched")


def _weight_for(weighting: str, delta: float, scale_constant: float | None) -> float:
    """The non-negative reward magnitude for a case under a weighting scheme.

      - "magnitude":     |δ̃|                (per-case, varies across prompts)
      - "sign_only":     1.0                (uniform; changes both weighting AND scale)
      - "scale_matched": scale_constant c   (uniform like sign_only, but c chosen
                         so the GLOBAL reward scale matches magnitude — isolates
                         per-case weighting from global scale; see ABLATION-001)
    """
    if weighting == "magnitude":
        return abs(delta)
    if weighting == "sign_only":
        return 1.0
    if weighting == "scale_matched":
        if scale_constant is None or not (scale_constant > 0):
            raise ValueError("scale_matched weighting requires a positive scale_constant "
                             f"(got {scale_constant!r})")
        return float(scale_constant)
    raise ValueError(f"unknown reward weighting {weighting!r}; expected one of {VALID_WEIGHTINGS}")


def compute_reward(response: str, perspective: str, delta: float,
                   weighting: str = "magnitude",
                   scale_constant: float | None = None) -> float:
    """
    Reward for a Yes/No response.

    delta = δ̃ = U_X - U_Y  (Qwen-own or consensus, from the configured delta file)
    perspective: "X" (endowed with X, offered Y) or "Y" (endowed with Y, offered X)
    weighting:
      - "magnitude" (DEFAULT, the confirmatory behavior): ±|δ̃|.
      - "sign_only" (roadmap Priority-1 ablation):        ±1, independent of |δ̃|.
      - "scale_matched" (ABLATION-001 control): ±c with a FIXED constant c (see
        compute_scale_constant). Uniform per case like sign_only, but scaled so
        the global reward scale matches ±|δ̃|, so magnitude-vs-scale_matched
        isolates per-case magnitude information from global scale.

    In every mode the SIGN is +weight for the rational choice, -weight otherwise;
    only the magnitude of the weight differs. Unparseable responses ('') are
    irrational → -weight. δ̃ == 0.0 cases are filtered in prompt_builder.py;
    guarded here too (return 0.0).
    """
    if delta == 0.0:
        return 0.0
    if weighting not in VALID_WEIGHTINGS:
        raise ValueError(f"unknown reward weighting {weighting!r}; "
                         f"expected one of {VALID_WEIGHTINGS}")

    rational = rational_choice(perspective, delta)
    weight = _weight_for(weighting, delta, scale_constant)
    return weight if response == rational else -weight


def characterize_deltas(deltas) -> dict:
    """Summarise the magnitude-reward distribution |δ̃| from an iterable of δ̃.

    Deterministic; the reported moments justify the scale-matching rules in
    compute_scale_constant. Non-zero δ̃ only (δ̃==0 cases are filtered upstream).
    """
    vals = [abs(float(d)) for d in deltas if float(d) != 0.0]
    if not vals:
        raise ValueError("no non-zero deltas to characterize")
    n = len(vals)
    mean_abs = sum(vals) / n
    rms = (sum(v * v for v in vals) / n) ** 0.5
    ordered = sorted(vals)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    var = sum((v - mean_abs) ** 2 for v in vals) / n
    return {
        "n_nonzero": n,
        "mean_abs": mean_abs,     # E[|δ̃|]        → matches mean reward magnitude
        "rms": rms,               # sqrt(E[δ̃²])   → matches per-case gradient 2nd moment
        "median_abs": median,
        "std_abs": var ** 0.5,
        "min_abs": ordered[0],
        "max_abs": ordered[-1],
    }


def compute_scale_constant(deltas, rule: str = "mean_abs") -> float:
    """Scale-matching constant c for scale_matched weighting.

    rule:
      - "mean_abs" (DEFAULT): c = E[|δ̃|]. Matches the FIRST absolute moment, i.e.
        the mean reward magnitude / mean absolute advantage scale. Most direct
        "same average reward scale" match.
      - "rms": c = sqrt(E[δ̃²]). Matches the SECOND moment, i.e. the RMS reward
        magnitude. Within a GRPO group the advantage magnitude is ∝ the reward
        weight, so RMS matching equalises the root-mean-square effective
        advantage/gradient scale across prompts.

    Limitation (documented, not fixable by choice of c): GRPO normalises
    advantages by the GROUP mean (scale_rewards="none" still mean-centres), and
    ~80% of groups carry zero task-reward advantage. So no constant c reproduces
    the realized per-step gradient of the magnitude reward; c only matches the
    chosen moment of the reward-magnitude distribution.
    """
    stats = characterize_deltas(deltas)
    if rule == "mean_abs":
        return stats["mean_abs"]
    if rule == "rms":
        return stats["rms"]
    raise ValueError(f"unknown scale rule {rule!r}; expected 'mean_abs' or 'rms'")


def make_reward_fn(weighting: str = "magnitude", scale_constant: float | None = None):
    """
    Return a reward function compatible with TRL's GRPOTrainer.

    TRL calls this as:
        reward_fn(completions, perspective=[...], delta=[...], **kwargs)

    All arguments are lists of length (batch_size × num_generations).
    With batch_size=1 and num_generations=16, this is 16 items per call:
      - completions: 16 generated strings (the group)
      - perspective: ["X"]*16 or ["Y"]*16  (same value repeated G times)
      - delta: [float]*16                  (same value repeated G times)

    Zero-diversity groups (KNOWN_ISSUES.md #4):
    If all G completions in a group are identical (all "No" at initialization,
    likely), we return all-zero task rewards. This does NOT skip the batch —
    nothing here can. Be precise about what actually happens downstream in the
    stock TRL GRPOTrainer (trl 1.3.0, `grpo_trainer.py`):

      - Advantages are mean-centred, NOT raw rewards, even with
        `scale_rewards: "none"`:  `advantages = rewards - mean_grouped_rewards`
        (with "none" the std is computed for logging only, never divided out).
      - An all-identical group already has a constant reward vector, so its
        advantage is 0 with or without this branch. Numerically the branch is a
        no-op on the gradient; what it changes is the LOGGED mean reward, which
        is reported as 0.0 instead of the group's true ±|δ̃|.
      - Zero task advantage is not a zero update. With `beta = 0.04` the loss is
        `per_token_loss + beta * per_token_kl`, so the group still contributes a
        KL-only gradient toward the reference policy. Since
        generation_batch_size = 1 x 16 = G, one group IS one optimizer step.

    Call these "zero task-reward advantage groups" — not "skipped batches" and
    not "zero-update batches". Neither generation nor gradient compute is saved.
    True generation-level filtering (DAPO dynamic sampling) requires subclassing
    GRPOTrainer and overriding _generate_completions; it is not implemented.
    """
    if weighting not in VALID_WEIGHTINGS:
        raise ValueError(f"unknown reward weighting {weighting!r}; "
                         f"expected one of {VALID_WEIGHTINGS}")
    if weighting == "scale_matched" and not (scale_constant and scale_constant > 0):
        raise ValueError("scale_matched weighting requires a positive scale_constant")

    def reward_fn(completions, perspective, delta, **kwargs):
        parsed = [parse_response(c) for c in completions]

        # Zero-diversity group (all outputs identical) → all-zero task rewards.
        # The batch is NOT skipped; see the docstring above for what TRL does
        # with these. Identical for all weightings — the diversity branch is
        # orthogonal to reward magnitude, so the sign-only / scale-matched
        # contrast is purely the weight.
        if len(set(parsed)) == 1:
            return [0.0] * len(completions)

        return [
            compute_reward(p, persp, float(d), weighting=weighting,
                           scale_constant=scale_constant)
            for p, persp, d in zip(parsed, perspective, delta)
        ]

    return reward_fn


if __name__ == "__main__":
    # Quick sanity check
    fn = make_reward_fn()

    # All "No" → zero-diversity group → all-zero task rewards (batch still runs)
    completions_all_no = ["No"] * 16
    r = fn(completions_all_no, ["X"] * 16, [-1.81] * 16)
    assert all(x == 0.0 for x in r), "all-identical group should give zero task rewards"
    print("zero-diversity group (all No): OK — all-zero task rewards")

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
