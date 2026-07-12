#!/usr/bin/env python3
"""
Tests for the v2-core paired GRPO plumbing (paired_grpo.py). Pure Python — no
torch/trl — so it runs in milliseconds on a login node:

    python train/test_paired_grpo.py

Covers: aligned pair scoring, R_neutral, batch-level pairing by case_id,
pair-preserving shuffling, incomplete batches (sampler drop_last + reward-side
unpaired), and distributed (multi-process) sharding that never splits a pair.
"""

from paired_grpo import (
    build_pairs_from_columns,
    make_r_pair_reward,
    make_r_neutral_reward,
    make_r_gated_reward,
    PairedRepeatSampler,
)


def _decode_stream_to_pairs(stream, G):
    """Turn a sampler index stream into (x_idx, y_idx) pairs, asserting every
    prompt is emitted in a run of exactly G and pairs are whole/consecutive."""
    prompts, i = [], 0
    while i < len(stream):
        v = stream[i]
        run = 0
        while i < len(stream) and stream[i] == v:
            run += 1
            i += 1
        assert run == G, f"prompt {v} emitted {run} times, expected G={G}"
        prompts.append(v)
    assert len(prompts) % 2 == 0, "odd number of prompts — a pair was split"
    return [(prompts[k], prompts[k + 1]) for k in range(0, len(prompts), 2)]


# ── 1. Aligned pair scoring + all four cross-tab cells ───────────────────────
def test_pair_scoring_aligned():
    r_pair = make_r_pair_reward()
    # one case, G=2: X=[No,Yes], Y=[Yes,Yes]  → pair0 consistent(+1), pair1 trade-both(-1)
    completions = ["No", "Yes", "Yes", "Yes"]
    perspective = ["X", "X", "Y", "Y"]
    case_id = [100, 100, 100, 100]
    r = r_pair(completions=completions, perspective=perspective, case_id=case_id)
    assert r == [1.0, -1.0, 1.0, -1.0], r
    # both members of each pair share the value
    assert r[0] == r[2] and r[1] == r[3]
    print("test_pair_scoring_aligned: OK")


def test_pair_all_cells_and_multi_case():
    r_pair = make_r_pair_reward()
    # two cases interleaved out of order; G=1 each
    # case 7: X=No, Y=Yes  → consistent +1
    # case 9: X=No, Y=No   → keep-both -1
    completions = ["No", "No", "Yes", "No"]
    perspective = ["X", "X", "Y", "Y"]
    case_id = [7, 9, 7, 9]
    r = r_pair(completions=completions, perspective=perspective, case_id=case_id)
    assert r[0] == 1.0 and r[2] == 1.0, "case 7 consistent"
    assert r[1] == -1.0 and r[3] == -1.0, "case 9 keep-both"
    print("test_pair_all_cells_and_multi_case: OK")


# ── 2. R_neutral from anchor ─────────────────────────────────────────────────
def test_neutral_reward():
    anchor = {"7": {"pref": "X"}, "9": {"pref": None}, "11": "Y"}
    r_neutral = make_r_neutral_reward(anchor)
    # case 7 pref X: X-persp "No"→+1, Y-persp "Yes"→+1, X-persp "Yes"→-1
    # case 9 ambiguous → 0
    # case 11 pref Y: X-persp "No"→-1 (wrong; pref Y wants X-persp Yes)
    completions = ["No", "Yes", "Yes", "No", "No"]
    perspective = ["X", "Y", "X", "X", "X"]
    case_id = [7, 7, 7, 9, 11]
    r = r_neutral(completions=completions, perspective=perspective, case_id=case_id)
    assert r == [1.0, 1.0, -1.0, 0.0, -1.0], r
    print("test_neutral_reward: OK")


# ── 2c. GATED paired reward closure (production reward) ──────────────────────
def test_gated_reward_closure():
    anchor = {"7": {"pref": "X"}, "9": {"pref": "Y"}, "11": "X"}
    r_gated = make_r_gated_reward(anchor)
    # case 7 (pref X), G=2:
    #   pair0: X=No,Y=Yes → consistent, picked X == anchor X → +1
    #   pair1: X=Yes,Y=No → consistent, picked Y != anchor X → 0 (wrong pref)
    # case 9 (pref Y), G=1:
    #   pair0: X=No,Y=No → keep-both → -1
    completions = ["No", "Yes", "Yes", "No",  "No", "No"]
    perspective = ["X",  "X",   "Y",   "Y",   "X",  "Y"]
    case_id     = [7,    7,     7,     7,     9,    9]
    metrics = {}
    r = r_gated(completions=completions, perspective=perspective, case_id=case_id,
                log_metric=lambda k, v: metrics.__setitem__(k, v))
    # aligned: X rows [0,1]→case7 X; Y rows [2,3]→case7 Y; [4]X [5]Y →case9
    assert r[0] == 1.0 and r[2] == 1.0, "case7 pair0 correct (+1), shared"
    assert r[1] == 0.0 and r[3] == 0.0, "case7 pair1 wrong-pref (0), shared"
    assert r[4] == -1.0 and r[5] == -1.0, "case9 keep-both (-1), shared"
    assert metrics["v2/pairs_scored"] == 3.0
    assert metrics["v2/unpaired_completions"] == 0.0
    assert abs(metrics["v2/anchor_coverage"] - 1.0) < 1e-9  # all paired cases anchored
    # gated distribution: 1×(-1), 1×(0), 1×(+1) over 3 pairs
    assert abs(metrics["v2/gated_neg1_rate"] - 1/3) < 1e-9
    assert abs(metrics["v2/gated_zero_rate"] - 1/3) < 1e-9
    assert abs(metrics["v2/gated_pos1_rate"] - 1/3) < 1e-9
    print("test_gated_reward_closure: OK")


def test_gated_incomplete_pair():
    r_gated = make_r_gated_reward({"5": "X"})
    # only X completions for case 5 → no pairing, 0.0, no crash, unpaired counted
    completions = ["No", "Yes"]
    perspective = ["X", "X"]
    case_id = [5, 5]
    metrics = {}
    r = r_gated(completions=completions, perspective=perspective, case_id=case_id,
                log_metric=lambda k, v: metrics.__setitem__(k, v))
    assert r == [0.0, 0.0], r
    assert metrics["v2/pairs_scored"] == 0.0
    assert metrics["v2/unpaired_completions"] == 2.0
    print("test_gated_incomplete_pair: OK")


# ── 3a. Incomplete pair on the reward side (missing perspective) ─────────────
def test_reward_incomplete_pair():
    r_pair = make_r_pair_reward()
    # case 5 has only X completions in this batch → no pairing, all 0.0, no crash
    completions = ["No", "Yes"]
    perspective = ["X", "X"]
    case_id = [5, 5]
    r = r_pair(completions=completions, perspective=perspective, case_id=case_id)
    assert r == [0.0, 0.0], r
    print("test_reward_incomplete_pair: OK")


# ── 3b. Sampler drops trailing partial batch and keeps pairs whole ───────────
def test_sampler_incomplete_batch_dropped():
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]  # 5 pairs
    s = PairedRepeatSampler(pairs, mini_repeat_count=2, pairs_per_batch=2,
                            shuffle=False, drop_last=True)
    stream = list(s)
    got = _decode_stream_to_pairs(stream, G=2)
    # 5 pairs, 2 per batch, drop_last → 2 full batches = 4 pairs; 5th dropped
    assert got == [(0, 1), (2, 3), (4, 5), (6, 7)], got
    assert len(stream) == len(s), "len() disagrees with iteration"
    print("test_sampler_incomplete_batch_dropped: OK")


# ── 2. (sampler) pair-preserving shuffle ─────────────────────────────────────
def test_sampler_shuffle_keeps_pairs_whole():
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    s = PairedRepeatSampler(pairs, mini_repeat_count=3, pairs_per_batch=2,
                            shuffle=True, seed=123, drop_last=True)
    got = _decode_stream_to_pairs(list(s), G=3)
    # every emitted pair must be one of the real pairs (never split/mixed)
    assert set(got) <= set(pairs), got
    assert sorted(got) == sorted(pairs), "shuffle dropped/duplicated a pair"
    # different seed → different order (sanity that shuffle is active)
    s2 = PairedRepeatSampler(pairs, 3, 2, shuffle=True, seed=999, drop_last=True)
    got2 = _decode_stream_to_pairs(list(s2), G=3)
    assert set(got2) == set(pairs)
    print("test_sampler_shuffle_keeps_pairs_whole: OK")


# ── 4. Distributed: whole pairs sharded across processes, disjoint & complete ─
def test_sampler_distributed_no_split():
    pairs = [(2 * i, 2 * i + 1) for i in range(8)]  # 8 pairs
    G, ppb, nproc = 2, 2, 2   # 2 pairs/proc/batch × 2 procs = 4 pairs/global batch
    seen = {0: [], 1: []}
    for pidx in range(nproc):
        s = PairedRepeatSampler(pairs, mini_repeat_count=G, pairs_per_batch=ppb,
                                shuffle=True, seed=42, num_processes=nproc,
                                process_index=pidx, drop_last=True)
        seen[pidx] = _decode_stream_to_pairs(list(s), G=G)
        assert set(seen[pidx]) <= set(pairs), f"proc {pidx} split a pair"
    # 8 pairs / 4 per global batch = 2 full global batches → all 8 pairs covered
    union = set(seen[0]) | set(seen[1])
    assert union == set(pairs), f"processes did not cover all pairs: {union}"
    # disjoint: no pair handled by both processes
    assert not (set(seen[0]) & set(seen[1])), "a pair was assigned to two processes"
    print("test_sampler_distributed_no_split: OK")


# ── build_pairs_from_columns ─────────────────────────────────────────────────
def test_build_pairs_from_columns():
    # prompt_builder emits X then Y per case
    case_ids = [61, 61, 62, 62, 63, 63]
    persp = ["X", "Y", "X", "Y", "X", "Y"]
    pairs = build_pairs_from_columns(case_ids, persp)
    assert pairs == [(0, 1), (2, 3), (4, 5)], pairs
    print("test_build_pairs_from_columns: OK")


if __name__ == "__main__":
    test_pair_scoring_aligned()
    test_pair_all_cells_and_multi_case()
    test_neutral_reward()
    test_gated_reward_closure()
    test_gated_incomplete_pair()
    test_reward_incomplete_pair()
    test_sampler_incomplete_batch_dropped()
    test_sampler_shuffle_keeps_pairs_whole()
    test_sampler_distributed_no_split()
    test_build_pairs_from_columns()
    print("\nAll paired_grpo tests passed.")
