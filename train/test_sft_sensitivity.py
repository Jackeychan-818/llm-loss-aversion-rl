#!/usr/bin/env python3
"""CPU-only tests for the SFT batch/LR sensitivity experiment.

Covers exactly the things that, if wrong, would invalidate the comparison
silently rather than loudly: exposure arithmetic, deterministic ordering,
manifest correctness, and output-directory collision protection.

    python train/test_sft_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

import sft_sensitivity_plan as P  # noqa: E402

_fail: list[str] = []
_pass = 0


def check(cond, msg):
    global _pass
    if cond:
        _pass += 1
    else:
        _fail.append(msg)


def raises(fn, msg):
    try:
        fn()
    except P.ExposureError:
        check(True, msg)
        return
    except Exception as exc:                                  # pragma: no cover
        check(False, f"{msg} (wrong exception: {exc!r})")
        return
    check(False, f"{msg} (no exception raised)")


def fake_records(n: int) -> list[dict]:
    # two perspectives per case, emitted in a deliberately unhelpful order so a
    # permutation that forgot to canonicalise would be detectable
    recs = []
    for cid in range(n // 2, 0, -1):
        for p in ("Y", "X"):
            recs.append({"case_id": cid, "perspective": p, "prompt": f"p{cid}{p}",
                         "target": "Yes"})
    return recs


# ── exposure arithmetic ─────────────────────────────────────────────────────
def t_exposure_matches_protocol():
    check(P.optimizer_steps(6016, 1) == 6016, "6,016 prompts @ batch 1 -> 6,016 steps")
    check(P.optimizer_steps(6016, 16) == 376, "6,016 @ 16 -> 376")
    check(P.optimizer_steps(6016, 32) == 188, "6,016 @ 32 -> 188")
    check(P.optimizer_steps(6016, 64) == 94, "6,016 @ 64 -> 94")
    check(P.optimizer_steps(30016, 16) == 1876, "30,016 @ 16 -> 1,876")
    check(P.optimizer_steps(30016, 32) == 938, "30,016 @ 32 -> 938")
    check(P.optimizer_steps(30016, 64) == 469, "30,016 @ 64 -> 469")


def t_exposure_is_identical_across_batches():
    for exposure in (P.PILOT_EXPOSURE, P.FULL_EXPOSURE):
        seen = {b * P.optimizer_steps(exposure, b) for b in P.LARGE_BATCHES}
        check(seen == {exposure},
              f"steps x batch reconstructs the exposure exactly at {exposure:,}")


def t_inexact_division_hard_fails():
    raises(lambda: P.optimizer_steps(6000, 64), "inexact exposure/batch hard-fails")
    raises(lambda: P.optimizer_steps(100, 0), "zero batch hard-fails")
    raises(lambda: P.optimizer_steps(0, 16), "zero exposure hard-fails")
    raises(lambda: P.optimizer_steps(-64, 16), "negative exposure hard-fails")


def t_checkpoints_are_exposure_aligned():
    for b in P.BATCHES:
        m = P.checkpoint_steps(P.PILOT_EXPOSURE, b)
        check(tuple(m) == P.PILOT_CKPT_EXPOSURES,
              f"pilot checkpoints are the protocol exposures at batch {b}")
        check(all(e == s * b for e, s in m.items()),
              f"pilot checkpoint steps reconstruct their exposure at batch {b}")
        check(P.PILOT_EXPOSURE in m, f"final exposure always saved at batch {b}")
    for b in P.LARGE_BATCHES:
        m = P.checkpoint_steps(P.FULL_EXPOSURE, b)
        check(all(e == s * b for e, s in m.items()),
              f"full checkpoint steps reconstruct their exposure at batch {b}")
        check(max(m) == P.FULL_EXPOSURE, f"full run saves the endpoint at batch {b}")
        check(len(m) == 15, f"full run has 15 checkpoints at batch {b}")
    # every batch must produce checkpoints at the SAME exposures - that is the
    # entire point of an exposure-aligned grid
    grids = {tuple(P.checkpoint_steps(P.FULL_EXPOSURE, b)) for b in P.LARGE_BATCHES}
    check(len(grids) == 1, "all batches checkpoint at identical exposures")


def t_save_steps_lands_on_stride():
    for b in P.LARGE_BATCHES:
        ss = P.save_steps_for(P.FULL_EXPOSURE, b)
        check(ss * b == P.FULL_CKPT_STRIDE, f"save_steps is one 2,048 stride at batch {b}")


# ── deterministic ordering ──────────────────────────────────────────────────
def t_order_is_deterministic_and_batch_independent():
    recs = fake_records(400)
    o1 = P.deterministic_order(recs, 1)
    o2 = P.deterministic_order(recs, 1)
    check(o1 == o2, "ordering is reproducible for a fixed seed")
    check(P.order_hash(recs, o1) == P.order_hash(recs, o2),
          "order hash is stable across calls")
    # the prefix property: every cell of a seed consumes a prefix of ONE list
    h_small = P.order_hash(recs, o1, 64)
    h_big = P.order_hash(recs, o1, 128)
    check(h_small != h_big, "different exposures give different hashes")
    check(P.order_hash(recs, o1[:128], 64) == h_small,
          "a longer run's first 64 examples are the short run's 64 examples")


def t_order_differs_across_seeds():
    recs = fake_records(400)
    hashes = {s: P.order_hash(recs, P.deterministic_order(recs, s)) for s in P.SEEDS}
    check(len(set(hashes.values())) == len(P.SEEDS), "each seed has its own order")


def t_order_is_a_true_permutation():
    recs = fake_records(400)
    o = P.deterministic_order(recs, 2)
    check(sorted(o) == list(range(len(recs))), "ordering is a permutation, nothing lost")


def t_order_ignores_input_order():
    """Canonicalisation must make the permutation independent of how the record
    builder happened to emit rows; otherwise the 'same order' guarantee silently
    depends on an unrelated implementation detail."""
    recs = fake_records(400)
    shuffled = list(reversed(recs))
    h1 = P.order_hash(recs, P.deterministic_order(recs, 1))
    h2 = P.order_hash(shuffled, P.deterministic_order(shuffled, 1))
    check(h1 == h2, "order hash is invariant to the builder's emission order")


def t_no_duplicate_exposure():
    recs = fake_records(400)
    o = P.deterministic_order(recs, 1)
    P.assert_no_duplicate_exposure(recs, o, 400)
    check(True, "a full pass over distinct examples is accepted")
    raises(lambda: P.assert_no_duplicate_exposure(recs, o, 401),
           "exposure beyond the pool hard-fails (no prompt may repeat)")
    dupes = recs + recs
    od = P.deterministic_order(dupes, 1)
    raises(lambda: P.assert_no_duplicate_exposure(dupes, od, len(dupes)),
           "a duplicated example pool is detected")


# ── cell identity / collision protection ────────────────────────────────────
def t_cell_names_are_unique_and_informative():
    cells = P.phase_a_cells() + P.phase_b_cells(32) + P.full_cells(32, 3e-6)
    P.assert_unique_output_dirs(cells)
    check(True, "the whole planned cell set has no output-directory collision")
    names = [c.name for c in cells]
    check(len(set(names)) == len(names), "cell names are unique")
    for c in cells:
        check(f"eb{c.effective_batch}" in c.name, "name encodes effective batch")
        check(f"seed{c.seed}" in c.name, "name encodes seed")
        check(c.horizon_tag in c.name, "name encodes schedule horizon")


def t_pilot_and_full_never_collide():
    """Same batch/LR/seed at pilot and full horizons must not share a directory:
    their cosine horizons differ, so their weights are not interchangeable."""
    pilot = P.Cell("A", 32, 1e-6, 1, P.PILOT_EXPOSURE)
    full = P.Cell("full", 32, 1e-6, 1, P.FULL_EXPOSURE)
    check(pilot.output_dir != full.output_dir,
          "pilot and full horizons get separate directories")
    check(pilot.horizon_tag != full.horizon_tag, "horizon tags differ")


def t_collision_is_detected():
    dup = [P.Cell("A", 32, 1e-6, 1, P.PILOT_EXPOSURE)] * 2
    raises(lambda: P.assert_unique_output_dirs(dup), "a real collision hard-fails")


def t_outputs_stay_out_of_frozen_dirs():
    for c in P.phase_a_cells() + P.full_cells(32, 3e-6):
        s = str(c.output_dir)
        check("sft_sensitivity" in s, "outputs live under checkpoints/sft_sensitivity")
        for frozen in ("sft_qwen_delta_seed1", "sft_qwen_delta_seed2",
                       "sft_qwen_delta_seed1_pilot6k", "grpo_qwen_delta"):
            check(not s.rstrip("/").endswith(frozen),
                  f"output dir never equals the frozen run dir {frozen}")


def t_phase_b_reuses_the_1e6_cells():
    reused = [c for c in P.phase_b_cells(32) if P.phase_b_reuses_phase_a(c)]
    check(len(reused) == len(P.SEEDS),
          "the lr=1e-6 Phase-B column reuses Phase A (one cell per seed)")
    c = P.Cell("B", 32, 1e-6, 1, P.PILOT_EXPOSURE)
    src = P.phase_b_reuses_phase_a(c)
    check(src is not None and src.optimizer_steps == c.optimizer_steps,
          "the reused cell is identical in exposure and steps")
    check(P.phase_b_reuses_phase_a(P.Cell("B", 32, 3e-6, 1, P.PILOT_EXPOSURE)) is None,
          "a genuinely new LR is not treated as a reuse")


def t_phase_b_rejects_batch_one():
    raises(lambda: P.phase_b_cells(1), "Phase B must use a large batch, not 1")


# ── manifest correctness ────────────────────────────────────────────────────
def t_h32k_exposure_arithmetic():
    """Amendment 4: 32,000 prompts at effective batch 32."""
    check(P.optimizer_steps(P.HORIZON_32K, 32) == 1000,
          "32,000 prompts @ batch 32 -> exactly 1,000 optimizer updates")
    check(P.HORIZON_32K == 1000 * 32,
          "exposure reconstructs from updates x effective batch")
    # 32,000 divides 16/32/64 exactly (2,000/1,000/500), so pick a batch that
    # does NOT: 32,000 = 48*666 + 32.
    raises(lambda: P.optimizer_steps(P.HORIZON_32K, 48),
           "a batch that does not divide the exposure hard-fails")
    check(P.optimizer_steps(P.HORIZON_32K, 64) == 500,
          "32,000 also divides batch 64 exactly (500 updates)")
    check(P.HORIZON_32K < 98_900,
          "the horizon fits inside the pool, so no prompt is repeated")


def t_h32k_checkpoint_grid():
    m = P.h32k_checkpoint_steps(32)
    early, coarse = P.h32k_early_and_coarse(32)
    check(len(m) == 18, f"18 save points, got {len(m)}")
    check(len(early) == 10, f"10 dense early snapshots, got {len(early)}")
    check(sorted(early) == list(range(128, 1281, 128)),
          "early window is 128..1,280 in 128-prompt steps")
    check(sorted(early.values()) == list(range(4, 41, 4)),
          "early window is updates 4, 8, ... 40")
    check(sorted(coarse)[:2] == [4096, 8192], "coarse block starts at 4,096")
    check(P.HORIZON_32K in m and m[P.HORIZON_32K] == 1000,
          "endpoint 32,000 present at update 1,000")
    check(all(e % 32 == 0 for e in m),
          "every checkpoint exposure divides the effective batch exactly")
    check(all(e == s * 32 for e, s in m.items()),
          "every checkpoint step reconstructs its exposure")
    check(not (set(early) & set(coarse)), "early and coarse blocks are disjoint")
    check(max(early) < min(coarse), "the two blocks do not interleave")
    check(tuple(P.checkpoint_exposures(P.HORIZON_32K)) != tuple(m),
          "the 2,048-stride fallback is NOT what this horizon uses")


def t_h32k_cell_identity():
    c = P.Cell("full", 32, 1.0e-4, 1, P.HORIZON_32K)
    probe = P.Cell("B", 32, 1.0e-4, 1, P.PILOT_EXPOSURE)
    check(c.optimizer_steps == 1000, "cell reports 1,000 updates")
    check("h32000" in c.name, "name encodes the 32,000 horizon")
    check(c.output_dir != probe.output_dir,
          "the 32,000 run cannot collide with the h6016 probe sharing its "
          "batch, LR and seed")
    check(c.horizon_tag != probe.horizon_tag, "horizon tags differ")
    P.assert_unique_output_dirs([c, probe])
    check(True, "no collision across the two horizons")
    d = c.describe()
    check(d["prompt_exposure"] == d["optimizer_steps"] *
          d["gradient_accumulation_steps"] * d["per_device_train_batch_size"],
          "manifest exposure identity holds at the 32,000 horizon")


def t_manifest_describes_effective_batch():
    c = P.Cell("A", 32, 1e-6, 2, P.PILOT_EXPOSURE)
    d = c.describe()
    check(d["per_device_train_batch_size"] == 1, "per-device batch pinned to 1")
    check(d["gradient_accumulation_steps"] == 32,
          "effective batch is realised by accumulation")
    check(d["effective_batch"] == 32, "manifest states the effective batch")
    check(d["prompt_exposure"] == 6016, "manifest states prompt exposure")
    check(d["optimizer_steps"] == 188, "manifest states optimizer steps")
    check(d["prompt_exposure"] ==
          d["optimizer_steps"] * d["gradient_accumulation_steps"] *
          d["per_device_train_batch_size"],
          "manifest exposure arithmetic is internally consistent")
    check(d["schedule_horizon_steps"] == d["optimizer_steps"],
          "cosine horizon equals this run's own step count")
    check(d["max_grad_norm"] == 0.1, "clipping threshold unchanged")
    check(d["logging_steps"] == 1, "logging every optimizer update")
    check("batch 1 / accum 1" not in str(d),
          "no hard-coded 'batch 1 / accum 1' language survives in the manifest")


def t_manifest_is_json_serialisable():
    import json
    for c in P.phase_a_cells()[:3]:
        json.dumps(c.describe())
    check(True, "cell descriptions serialise to JSON")


def main() -> int:
    for fn in sorted((f for n, f in globals().items() if n.startswith("t_")),
                     key=lambda f: f.__code__.co_firstlineno):
        fn()
    if _fail:
        print(f"FAILED {len(_fail)} / {_pass + len(_fail)} checks:")
        for f in _fail:
            print(f"  x {f}")
        return 1
    print(f"OK - {_pass} checks passed (SFT batch/LR sensitivity plan).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
