#!/usr/bin/env python3
"""
Reward Design v2 (core) for lambda-zero GRPO training.

Implements the two components that run on the EXISTING data (see
REWARD_DESIGN_V2.md and the coverage audit): counterfactual ownership
consistency (R_pair) and the frozen ownership-free preference anchor
(R_neutral). R_dominance / R_monotonicity are deferred to v2.1 (they need
constructed counterfactual data) and are not implemented here.

Key structural difference from reward_functions.py
---------------------------------------------------
R_pair is a *paired* signal: it cannot be computed from a single prompt. The
same goods pair must be scored from BOTH ownership perspectives together, and
the g-th X-endowed completion is paired with the g-th Y-endowed completion
(REWARD_DESIGN_V2.md §"Training-unit requirement"). The trainer must therefore
sample matched X/Y prompts in the same step and call `score_paired_group`.

    ┌──────────────── one case_id ────────────────┐
    X-endowed prompt ──► G completions  x_0..x_{G-1}
    Y-endowed prompt ──► G completions  y_0..y_{G-1}
    pair (x_g, y_g) ──► R_pair_g (shared by both members)
    + per-member R_neutral (needs only the frozen anchor for this case)

R_neutral alone IS computable per single completion, so a non-paired fallback
is provided for ablation "R_neutral only" runs.
"""

from __future__ import annotations

from typing import Any, Optional

# Reuse the battle-tested parser from v1 so Yes/No extraction stays identical.
from reward_functions import parse_response  # noqa: F401


# ── Default weights (hyperparameters to ablate, per REWARD_DESIGN_V2.md) ──────
W_PAIR_DEFAULT = 1.0
W_NEUTRAL_DEFAULT = 0.5


# ── Component 1: counterfactual ownership consistency ────────────────────────
def r_pair(resp_x: str, resp_y: str) -> float:
    """
    +1 when the two endowment perspectives select the SAME good (off-diagonal),
    -1 when they keep-both (No/No) or trade-both (Yes/Yes).

    resp_x = parsed answer to the X-endowed prompt ("Yes"/"No"/""),
    resp_y = parsed answer to the Y-endowed prompt.

    An unparseable answer on either side cannot establish consistency, so it is
    treated as irrational (-1); a dedicated R_format term (not here) should
    additionally penalize malformed output.
    """
    if resp_x not in ("Yes", "No") or resp_y not in ("Yes", "No"):
        return -1.0
    return 1.0 if resp_x != resp_y else -1.0


# ── Component 2: frozen ownership-free preference anchor ──────────────────────
def rational_answer_for_anchor(perspective: str, anchor_pref: str) -> str:
    """
    Given the frozen ownership-free preferred good, the endowment-consistent
    answer for each perspective (REWARD_DESIGN_V2.md §2):
      pref X → X-perspective "No" (keep X), Y-perspective "Yes" (trade for X)
      pref Y → X-perspective "Yes" (trade for Y), Y-perspective "No" (keep Y)
    """
    if anchor_pref == "X":
        return "No" if perspective == "X" else "Yes"
    if anchor_pref == "Y":
        return "Yes" if perspective == "X" else "No"
    raise ValueError(f"anchor_pref must be 'X' or 'Y', got {anchor_pref!r}")


def r_neutral(response: str, perspective: str, anchor_pref: Optional[str]) -> Optional[float]:
    """
    +1 if the response selects the anchored good, -1 otherwise. Returns None
    when the case has no stable anchor (ambiguous) — the caller must then omit
    this component and renormalize by the remaining active weights.
    """
    if anchor_pref is None:
        return None
    if response not in ("Yes", "No"):
        return -1.0
    return 1.0 if response == rational_answer_for_anchor(perspective, anchor_pref) else -1.0


# ── GATED paired reward (v2-core production reward) ──────────────────────────
# Single scalar per (X,Y) pair, shared by both members. Replaces the additive
# w_pair*r_pair + w_neutral*r_neutral combination: the anchor-direction credit is
# GATED behind the ownership-consistency gate, giving a monotone ladder
#   keep-both / trade-both / malformed  = -1
#   consistent but wrong preferred good =  0
#   consistent and picks anchored good  = +1
# See REWARD_DESIGN_V2.md §"Gated reward". anchor_pref must be "X" or "Y" for the
# graded ladder; the training set is filtered to stable anchors so this holds.
# If anchor_pref is None (no stable anchor) we fall back to the consistency gate
# alone (+1 consistent / -1 not) — this path is dead on the filtered set and the
# trainer asserts anchor coverage == 100%, but it keeps the function total.
GATED_UNPARSEABLE = -1.0
GATED_INCONSISTENT = -1.0
GATED_WRONG_PREF = 0.0
GATED_CORRECT = 1.0


GATED_MODES = ("shaped", "hard")


def r_gated(resp_x: str, resp_y: str, anchor_pref: Optional[str],
            mode: str = "shaped") -> float:
    """Gated paired reward for one aligned (X-endowed, Y-endowed) completion pair.

    resp_x / resp_y are parsed answers ("Yes"/"No"/other). Returns a single float
    shared by both members of the pair (pair-level advantage).

    mode:
      "shaped" — three-level ladder: inconsistent/malformed = -1, consistent but
                 wrong preferred good = 0, consistent + anchored good = +1. The
                 consistency gate is the primary escape from the penalty region
                 (this is what reduces λ); direction is the refinement (0 → +1).
                 Three distinct levels also give richer within-group variance,
                 which reduces zero-std advantage collapse (KNOWN_ISSUES #2).
      "hard"   — binary: +1 iff consistent AND picks the anchored good, else -1.
                 Directly optimizes the rational outcome; no intermediate credit.
    """
    if mode not in GATED_MODES:
        raise ValueError(f"mode must be one of {GATED_MODES}, got {mode!r}")
    parseable = resp_x in ("Yes", "No") and resp_y in ("Yes", "No")
    consistent = parseable and resp_x != resp_y
    # Consistent: the pair selected the same good from both perspectives.
    #   (No, Yes) → X-persp keeps X, Y-persp trades for X → picked "X"
    #   (Yes, No) → X-persp trades for Y, Y-persp keeps Y → picked "Y"
    correct = consistent and (anchor_pref is None or ("X" if resp_x == "No" else "Y") == anchor_pref)

    if mode == "hard":
        return GATED_CORRECT if correct else GATED_INCONSISTENT
    # shaped
    if not parseable or not consistent:
        return GATED_INCONSISTENT              # malformed / keep-both / trade-both
    return GATED_CORRECT if correct else GATED_WRONG_PREF


# ── Weighted combination with active-weight renormalization ──────────────────
def _combine(components: dict[str, tuple[float, Optional[float]]]) -> float:
    """
    components: name -> (weight, value_or_None). Undefined (None) components are
    dropped and the result is renormalized by the sum of ACTIVE weights, so cases
    with different active components stay on a comparable ~[-1, 1] scale.
    """
    num = 0.0
    den = 0.0
    for _name, (w, v) in components.items():
        if v is None:
            continue
        num += w * v
        den += w
    return num / den if den > 0 else 0.0


def score_paired_group(
    x_responses: list[str],
    y_responses: list[str],
    anchor_pref: Optional[str],
    w_pair: float = W_PAIR_DEFAULT,
    w_neutral: float = W_NEUTRAL_DEFAULT,
) -> tuple[list[float], list[float]]:
    """
    Core v2 scorer for ONE case_id with G aligned completions per perspective.

    Returns (x_rewards, y_rewards), each length G. The shared R_pair_g is added
    to both members of pair g; R_neutral is per-member.

    x_responses / y_responses are already parsed to "Yes"/"No"/"".
    """
    if len(x_responses) != len(y_responses):
        raise ValueError("paired group requires equal-length X and Y completion lists")

    x_rewards, y_rewards = [], []
    for rx, ry in zip(x_responses, y_responses):
        rp = r_pair(rx, ry)                       # shared, always defined
        rn_x = r_neutral(rx, "X", anchor_pref)     # may be None
        rn_y = r_neutral(ry, "Y", anchor_pref)
        x_rewards.append(_combine({"pair": (w_pair, rp), "neutral": (w_neutral, rn_x)}))
        y_rewards.append(_combine({"pair": (w_pair, rp), "neutral": (w_neutral, rn_y)}))
    return x_rewards, y_rewards


# ── Non-paired fallback: R_neutral only (for the ablation ladder) ────────────
def make_neutral_only_reward_fn(anchor_by_case: dict[int, str]):
    """
    TRL-compatible reward for the "R_neutral only" ablation (no pairing needed).
    Signature mirrors reward_functions.make_reward_fn: the trainer passes
    `perspective` and `case_id` arrays aligned with `completions`.
    """
    def reward_fn(completions, perspective, case_id, **kwargs):
        out = []
        for c, persp, cid in zip(completions, perspective, case_id):
            pref = anchor_by_case.get(int(cid))
            v = r_neutral(parse_response(c), persp, pref)
            out.append(0.0 if v is None else v)   # ambiguous case → no signal
        return out
    return reward_fn


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER INTEGRATION (TODO — not wired into grpo_train.py yet)
#
# TRL's GRPOTrainer scores one prompt's group at a time, so R_pair needs a
# custom path. Sketch:
#   1. Build the dataset so each row carries a `pair_id = case_id` and both
#      perspectives are present (prompt_builder already emits matched X/Y rows
#      sharing case_id).
#   2. Subclass GRPOTrainer; ensure the sampler keeps the matched X and Y rows
#      for a case_id in the same optimization step with the same G.
#   3. After generation, group completions by (case_id, perspective), parse,
#      align X_g with Y_g, call score_paired_group(...), and scatter the
#      per-completion rewards back into TRL's expected flat reward vector.
#   4. Keep monitor.sh's DAPO panel in view — consistency rewards can collapse
#      to zero within-group variance (KNOWN_ISSUES.md #2).
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # ---- R_pair: all four cross-tab cells (REWARD_DESIGN_V2.md §1 table) ----
    assert r_pair("No", "Yes") == 1.0    # X,X consistent
    assert r_pair("Yes", "No") == 1.0    # Y,Y consistent
    assert r_pair("No", "No") == -1.0    # keep-both
    assert r_pair("Yes", "Yes") == -1.0  # trade-both
    assert r_pair("", "No") == -1.0      # unparseable → irrational
    print("R_pair cross-tab: OK")

    # ---- R_neutral: both anchor directions ----
    assert r_neutral("No", "X", "X") == 1.0    # pref X, X-persp keep → correct
    assert r_neutral("Yes", "Y", "X") == 1.0   # pref X, Y-persp trade → correct
    assert r_neutral("Yes", "X", "X") == -1.0  # pref X, X-persp trade → wrong
    assert r_neutral("No", "Y", "Y") == 1.0    # pref Y, Y-persp keep → correct
    assert r_neutral("No", "X", None) is None  # ambiguous case → no signal
    print("R_neutral directions: OK")

    # ---- Paired group: pref X, mix of consistent and keep-both ----
    xr, yr = score_paired_group(
        x_responses=["No", "No", "Yes"],   # keep, keep, trade
        y_responses=["Yes", "No", "Yes"],  # trade, keep, trade
        anchor_pref="X",
        w_pair=1.0, w_neutral=0.5,
    )
    # pair 0: (No,Yes) consistent-X → R_pair +1; neutral X: No→+1, Y: Yes→+1
    #   x0 = (1*1 + 0.5*1)/1.5 = 1.0 ; y0 = 1.0
    assert abs(xr[0] - 1.0) < 1e-9 and abs(yr[0] - 1.0) < 1e-9
    # pair 1: (No,No) keep-both → R_pair -1; neutral X: No→+1, Y: No→-1
    #   x1 = (1*-1 + 0.5*1)/1.5 = -1/3 ; y1 = (1*-1 + 0.5*-1)/1.5 = -1
    assert abs(xr[1] - (-1/3)) < 1e-9 and abs(yr[1] - (-1.0)) < 1e-9
    print(f"Paired group (pref X): x={[round(v,3) for v in xr]}  y={[round(v,3) for v in yr]}")

    # ---- Ambiguous anchor → R_pair carries alone, renormalized ----
    xr, yr = score_paired_group(["No"], ["Yes"], anchor_pref=None)
    assert abs(xr[0] - 1.0) < 1e-9   # only R_pair active → +1
    print("Ambiguous-anchor renormalization: OK")

    # ---- GATED reward: SHAPED (three-level) table ----
    # anchor pref X: consistent-picks-X = +1, consistent-picks-Y = 0
    assert r_gated("No", "Yes", "X") == 1.0     # picked X, anchor X → correct
    assert r_gated("Yes", "No", "X") == 0.0     # picked Y, anchor X → wrong pref
    # anchor pref Y: mirror
    assert r_gated("Yes", "No", "Y") == 1.0     # picked Y, anchor Y → correct
    assert r_gated("No", "Yes", "Y") == 0.0     # picked X, anchor Y → wrong pref
    # consistency gate fails regardless of anchor
    assert r_gated("No", "No", "X") == -1.0     # keep-both
    assert r_gated("Yes", "Yes", "X") == -1.0   # trade-both
    assert r_gated("No", "No", "Y") == -1.0
    # unparseable on either side
    assert r_gated("", "Yes", "X") == -1.0
    assert r_gated("No", "maybe", "Y") == -1.0
    # dead-path fallback: no anchor → consistency gate only
    assert r_gated("No", "Yes", None) == 1.0    # consistent → +1
    assert r_gated("No", "No", None) == -1.0    # inconsistent → -1
    # ladder is monotone: inconsistent < wrong-pref < correct
    assert GATED_INCONSISTENT < GATED_WRONG_PREF < GATED_CORRECT
    print("Gated reward (shaped) table: OK")

    # ---- GATED reward: HARD (binary) table ----
    assert r_gated("No", "Yes", "X", mode="hard") == 1.0    # consistent + correct → +1
    assert r_gated("Yes", "No", "X", mode="hard") == -1.0   # consistent but wrong → -1 (no 0)
    assert r_gated("No", "No", "X", mode="hard") == -1.0    # keep-both → -1
    assert r_gated("Yes", "Yes", "Y", mode="hard") == -1.0  # trade-both → -1
    assert r_gated("", "Yes", "X", mode="hard") == -1.0     # malformed → -1
    assert r_gated("Yes", "No", "Y", mode="hard") == 1.0    # consistent + correct → +1
    assert r_gated("No", "Yes", None, mode="hard") == 1.0   # no anchor → gate only
    # hard mode collapses the 0 level onto -1
    assert set(r_gated(a, b, "X", mode="hard") for a in ("Yes","No") for b in ("Yes","No")) == {-1.0, 1.0}
    try:
        r_gated("No", "Yes", "X", mode="bogus"); assert False
    except ValueError:
        pass
    print("Gated reward (hard) table: OK")

    print("All v2-core reward checks passed.")
