#!/usr/bin/env python3
"""Exposure arithmetic, deterministic ordering, and cell identity for the
post-hoc SFT batch/learning-rate sensitivity experiment.

Pure and CPU-only: no torch, no model, no data-file reads beyond the training
record list. Every number the protocol quotes comes from here, so the protocol,
the configs, the PBS scripts and the tests cannot drift apart.

Central invariant
-----------------
A *cell* is identified by (phase, effective_batch, learning_rate, seed,
horizon). Cells differ ONLY in how gradients are accumulated and how large the
step is — never in which examples are seen or in what order. Prompt exposure,
not optimizer steps, is the comparison axis:

    optimizer_steps = prompt_exposure / effective_batch      (must be exact)

`per_device_train_batch_size` is pinned to 1 for every cell and the effective
batch is realised purely through `gradient_accumulation_steps`, so the forward/
backward arithmetic per prompt is bit-for-bit the same across cells and only the
optimizer update boundary moves.

Scientific status: EXPLORATORY / POST-HOC optimization ablation. It does not
revise, replace or reselect the frozen matched-SFT baseline.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── frozen experiment design ────────────────────────────────────────────────
PILOT_EXPOSURE = 6_016          # unique prompts per Phase-A / Phase-B run
FULL_EXPOSURE = 30_016          # unique prompts per full sensitivity run
PILOT_CKPT_EXPOSURES = (2_048, 4_096, 6_016)
# Phase B: dense ADAPTER-ONLY snapshots over the early window, preserved but not
# evaluated. 128 divides 16/32/64 exactly (8/4/2 optimizer steps), so the
# snapshot grid is exposure-aligned across batches just like the checkpoint grid.
EARLY_SNAPSHOT_STRIDE = 128
EARLY_SNAPSHOT_MAX = 2_048
EARLY_SNAPSHOT_PROMPTS = tuple(range(EARLY_SNAPSHOT_STRIDE,
                                     EARLY_SNAPSHOT_MAX + 1, EARLY_SNAPSHOT_STRIDE))
# Phase-A across-seed sd of endpoint validation CE, per batch. Frozen here as the
# NOISE FLOOR of the seed-disagreement rule so the threshold cannot be chosen
# after seeing Phase-B data.
PHASE_A_CE_NOISE_FLOOR = {16: 0.0019, 32: 0.0050, 64: 0.0097}
# "Within noise" for the non-binding path-length prediction: the largest floor
# above, rounded up. Applies to three-seed mean validation CE. Non-binding.
PREDICTION_WITHIN_NOISE_TOL = 0.010
FULL_CKPT_STRIDE = 2_048
BATCHES = (1, 16, 32, 64)
LARGE_BATCHES = (16, 32, 64)
PHASE_A_LR = 1.0e-6
PHASE_B_LRS = (3.0e-7, 1.0e-6, 3.0e-6, 1.0e-5)
# Off-grid bracketing probe (Amendment 3). Kept SEPARATE from PHASE_B_LRS so it
# cannot silently enter the frozen Stage-1/Stage-2 decision tree: a probe LR may
# bracket the optimum but may never promote a setting.
PROBE_LRS = (1.0e-4,)
SEEDS = (1, 2, 3)

# Held fixed across every cell (deliberately NOT swept in this experiment).
FIXED = {
    "per_device_train_batch_size": 1,
    "max_grad_norm": 0.1,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "bf16": True,
    "logging_steps": 1,          # every optimizer update
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "loss": "completion-only cross-entropy (prompt tokens masked with -100)",
}

RESULT_ROOT = PROJECT_ROOT / "results" / "sft_sensitivity"
CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "sft_sensitivity"


class ExposureError(ValueError):
    """Raised when an exposure is not exactly divisible by the effective batch."""


def optimizer_steps(exposure: int, effective_batch: int) -> int:
    """Optimizer updates for a prompt exposure at an effective batch.

    Hard-fails on inexact division: a cell that cannot consume its exposure in
    whole optimizer steps would either truncate or over-run the comparison axis,
    which silently breaks the matching the whole experiment rests on.
    """
    if effective_batch <= 0:
        raise ExposureError(f"effective_batch must be positive, got {effective_batch}")
    if exposure <= 0:
        raise ExposureError(f"exposure must be positive, got {exposure}")
    if exposure % effective_batch:
        raise ExposureError(
            f"exposure {exposure:,} is not divisible by effective batch "
            f"{effective_batch} (remainder {exposure % effective_batch}); "
            "choose an exposure that is an exact multiple for every batch in "
            f"{BATCHES}")
    return exposure // effective_batch


def checkpoint_exposures(total_exposure: int) -> tuple[int, ...]:
    """Exposure points at which a checkpoint is saved, always including the end."""
    if total_exposure == PILOT_EXPOSURE:
        pts = list(PILOT_CKPT_EXPOSURES)
    else:
        pts = list(range(FULL_CKPT_STRIDE, total_exposure, FULL_CKPT_STRIDE))
    if total_exposure not in pts:
        pts.append(total_exposure)
    return tuple(sorted(set(pts)))


def checkpoint_steps(total_exposure: int, effective_batch: int) -> dict[int, int]:
    """Map each checkpoint exposure to its optimizer step for this batch."""
    return {e: optimizer_steps(e, effective_batch)
            for e in checkpoint_exposures(total_exposure)}


def early_snapshot_steps(effective_batch: int) -> dict[int, int]:
    """Adapter-only snapshot exposures -> optimizer step, for Phase-B runs."""
    return {e: optimizer_steps(e, effective_batch) for e in EARLY_SNAPSHOT_PROMPTS}


def seed3_required(batch: int, mean_ce_by_lr: dict[float, float]) -> dict:
    """The FROZEN seed-disagreement rule (Phase-B Amendment 2).

    Seed 3 is required for BOTH top candidates — not only the provisional
    winner — because adding a third seed to one side alone would compare a
    three-seed mean against a two-seed mean.
    """
    ranked = sorted(mean_ce_by_lr.items(), key=lambda kv: kv[1])
    if len(ranked) < 2:
        return {"required": False, "reason": "fewer than two candidates"}
    (lr1, ce1), (lr2, ce2) = ranked[0], ranked[1]
    gap = ce2 - ce1
    floor = PHASE_A_CE_NOISE_FLOOR[batch]
    close = gap < floor
    return {"required": bool(close), "top_two": [lr1, lr2], "gap": gap,
            "noise_floor": floor, "reason":
            (f"gap {gap:.6f} < Phase-A noise floor {floor} for eb{batch}" if close
             else f"gap {gap:.6f} >= noise floor {floor}"),
            "both_candidates_required": True}


def save_steps_for(total_exposure: int, effective_batch: int) -> int:
    """HF `save_steps` that lands on the intermediate checkpoint exposures.

    The endpoint is saved explicitly by the trainer afterwards, because the final
    exposure is generally not a multiple of the stride (6,016 and 30,016 are
    not multiples of 2,048).
    """
    return optimizer_steps(FULL_CKPT_STRIDE, effective_batch)


# ── deterministic ordering ──────────────────────────────────────────────────
ORDER_ALGORITHM = (
    "numpy.random.Generator(PCG64(seed)).permutation over record indices sorted "
    "by (case_id, perspective); the first N of that single permutation is the "
    "exposure prefix for EVERY cell of that seed"
)


def example_id(rec: dict) -> str:
    """Stable identifier for one training example (one prompt/target pair)."""
    return f"{int(rec['case_id'])}:{rec['perspective']}"


def deterministic_order(records: list[dict], seed: int) -> list[int]:
    """One permutation per seed, independent of batch, LR and horizon.

    Records are first sorted into a canonical (case_id, perspective) order so the
    permutation cannot inherit whatever order the builder happened to emit; then
    a single PCG64 draw fixes the sequence. Every cell for a seed takes a PREFIX
    of this one list, which is what makes "same examples, same order" checkable
    rather than merely intended.
    """
    import numpy as np
    canon = sorted(range(len(records)),
                   key=lambda i: (int(records[i]["case_id"]),
                                  str(records[i]["perspective"])))
    perm = np.random.Generator(np.random.PCG64(int(seed))).permutation(len(canon))
    return [canon[int(p)] for p in perm]


def order_hash(records: list[dict], order: list[int], n: int | None = None) -> str:
    """SHA-256 over the ordered example identifiers (newline-joined, UTF-8)."""
    take = order if n is None else order[:n]
    h = hashlib.sha256()
    for i in take:
        h.update(example_id(records[i]).encode())
        h.update(b"\n")
    return h.hexdigest()


def assert_no_duplicate_exposure(records: list[dict], order: list[int],
                                 n: int) -> None:
    """No example may be consumed twice within one run."""
    if n > len(order):
        raise ExposureError(
            f"exposure {n:,} exceeds the {len(order):,} available examples; "
            "a run must not repeat a prompt")
    ids = [example_id(records[i]) for i in order[:n]]
    if len(set(ids)) != len(ids):
        dup = len(ids) - len(set(ids))
        raise ExposureError(f"{dup} duplicated prompt exposure(s) in the first "
                            f"{n:,} examples")


# ── cell identity ───────────────────────────────────────────────────────────
def _lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("-0", "-").replace("+0", "")


@dataclass(frozen=True)
class Cell:
    phase: str                  # "A" | "B" | "full"
    effective_batch: int
    learning_rate: float
    seed: int
    exposure: int

    @property
    def optimizer_steps(self) -> int:
        return optimizer_steps(self.exposure, self.effective_batch)

    @property
    def horizon_tag(self) -> str:
        """Schedule horizon, in the name: pilot and full weights are NOT
        interchangeable at equal exposure because the cosine horizons differ."""
        return f"h{self.exposure}"

    @property
    def name(self) -> str:
        return (f"sft_sens_ph{self.phase}_eb{self.effective_batch}"
                f"_lr{_lr_tag(self.learning_rate)}_seed{self.seed}"
                f"_{self.horizon_tag}")

    @property
    def output_dir(self) -> Path:
        return CKPT_ROOT / self.name

    def describe(self) -> dict:
        return {**asdict(self),
                "name": self.name,
                "optimizer_steps": self.optimizer_steps,
                "gradient_accumulation_steps": self.effective_batch,
                "per_device_train_batch_size": 1,
                "prompt_exposure": self.exposure,
                "schedule_horizon_steps": self.optimizer_steps,
                "checkpoint_exposure_to_step": checkpoint_steps(
                    self.exposure, self.effective_batch),
                "output_dir": str(self.output_dir.relative_to(PROJECT_ROOT)),
                **FIXED}


def phase_a_cells() -> list[Cell]:
    return [Cell("A", b, PHASE_A_LR, s, PILOT_EXPOSURE)
            for b in BATCHES for s in SEEDS]


def phase_b_cells(selected_batch: int, include_probe: bool = False) -> list[Cell]:
    """Phase-B cells for one batch. Probe LRs are opt-in and never part of the
    frozen grid used by the selection rules."""
    if selected_batch not in LARGE_BATCHES:
        raise ExposureError(f"Phase B batch must be one of {LARGE_BATCHES}")
    lrs = PHASE_B_LRS + (PROBE_LRS if include_probe else ())
    return [Cell("B", selected_batch, lr, s, PILOT_EXPOSURE)
            for lr in lrs for s in SEEDS]


def is_probe(cell: Cell) -> bool:
    """True for an off-grid bracketing cell, which may not promote a setting."""
    return cell.learning_rate in PROBE_LRS


def full_cells(batch: int, lr: float) -> list[Cell]:
    return [Cell("full", batch, lr, s, FULL_EXPOSURE) for s in SEEDS]


def phase_b_reuses_phase_a(cell: Cell) -> Cell | None:
    """The lr=1e-6 Phase-B cells ARE the Phase-A cells at that batch: identical
    exposure, order, schedule horizon and hyper-parameters. Re-running them would
    burn GPU hours to reproduce bit-identical work."""
    if cell.phase == "B" and cell.learning_rate == PHASE_A_LR:
        return Cell("A", cell.effective_batch, PHASE_A_LR, cell.seed, cell.exposure)
    return None


def assert_unique_output_dirs(cells: list[Cell]) -> None:
    """No two cells may share an output directory."""
    seen: dict[str, Cell] = {}
    for c in cells:
        if c.name in seen:
            raise ExposureError(
                f"output-directory collision: {c.name!r} produced by both "
                f"{seen[c.name].describe()} and {c.describe()}")
        seen[c.name] = c


if __name__ == "__main__":
    a = phase_a_cells()
    assert_unique_output_dirs(a)
    print(json.dumps({
        "phase_a_cells": len(a),
        "pilot_exposure": PILOT_EXPOSURE,
        "steps_by_batch": {b: optimizer_steps(PILOT_EXPOSURE, b) for b in BATCHES},
        "pilot_checkpoints": {b: checkpoint_steps(PILOT_EXPOSURE, b) for b in BATCHES},
        "full_steps_by_batch": {b: optimizer_steps(FULL_EXPOSURE, b)
                                for b in LARGE_BATCHES},
        "full_n_checkpoints": {b: len(checkpoint_steps(FULL_EXPOSURE, b))
                               for b in LARGE_BATCHES},
    }, indent=2))
