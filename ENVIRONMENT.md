# Environment — declared minimums (NOT a complete lock)

*Added 2026-07-30 for the CPU-only paper-gate package (branch
`codex/cpu-paper-gates`). This records the interpreter and library minimums the
CPU-phase code needs. It is **not** a full environment lock — no pinned lockfile,
no Torch/Transformers/TRL/PEFT freeze, no container image is produced here. A
complete reproducible lock remains an open reproducibility item
(`ARTIFACT-001`, `ENV-001` in `KNOWN_ISSUES.md`).*

## Minimum Python

**Python >= 3.10.** The reward code uses PEP 604 union annotations
(`float | None`) evaluated at definition time in `train/reward_functions.py`
(no `from __future__ import annotations`), which require 3.10+. Other modules add
`from __future__ import annotations` but the 3.10 floor stands.

## Required libraries (CPU-phase scripts)

| library | used by | minimum |
|---|---|---|
| **SciPy** | `eval/robust_inference.py`, `eval/estimator_recovery.py` (`scipy.optimize.least_squares`) | any modern SciPy (>= 1.7 is safe; `least_squares` exists since 0.17) |
| **NumPy** | robustness layer, recovery, analyses | >= 1.20 |

The suite generators (`data/method_comparison/`), the scale-matched spec
(`train/build_scale_matched_spec.py`), the full-behavior aggregation
(`eval/aggregate_full_behavior.py`), and the GRPO-efficiency analysis
(`eval/analyze_grpo_efficiency.py`) use only the Python standard library.

## Tested with

Python **3.13.3**, SciPy **1.15.2**, NumPy **2.2.4** (NSCC venv). All CPU-phase
unit/integrity/regeneration checks pass under this combination.

## Not covered here

Training/evaluation dependencies (PyTorch, Transformers, TRL, PEFT) and their
exact revisions are **not** declared or locked in this file; that is tracked
separately as a Priority-0 reproducibility task.
