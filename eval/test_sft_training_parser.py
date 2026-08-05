#!/usr/bin/env python3
"""Synthetic tests for the SFT training-log parser.

Covers the failure modes that must never be silently repaired or fabricated:
missing optional fields, duplicate steps, out-of-order entries, non-finite
values, log entries with no loss, and the pilot/full run separation. Also
checks that the smoothing is strictly causal and that the learning-rate
reference curve is evaluated on the Trainer's logging clock.

Run from the repository root:

    python eval/plot_sft_training_dynamics.py --refresh   # (optional, needs NSCC)
    python eval/test_sft_training_parser.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from plot_sft_training_dynamics import (  # noqa: E402
    RUNS, causal_rolling, parse_run_summary, parse_trainer_state, theoretical_lr,
)

_fail: list[str] = []
_pass = 0


def check(cond: bool, label: str) -> None:
    global _pass
    if cond:
        _pass += 1
    else:
        _fail.append(label)


def write_state(tmp: Path, name: str, *, max_steps: int, history: list[dict],
                logging_steps: int = 10) -> Path:
    p = tmp / name
    p.write_text(json.dumps({
        "max_steps": max_steps, "global_step": max_steps,
        "logging_steps": logging_steps, "save_steps": 2000,
        "epoch": 0.1, "log_history": history,
    }, allow_nan=True))
    return p


META = {"run_role": "full", "seed": 1, "expected_max_steps": 30000,
        "ckpt_dir": "/nonexistent-checkpoint-dir"}


def t_clean_history(tmp: Path) -> None:
    hist = [{"step": s, "epoch": s / 1000, "loss": 0.5, "grad_norm": 1.0,
             "learning_rate": 1e-6} for s in range(10, 101, 10)]
    df, a = parse_trainer_state(write_state(tmp, "clean.json", max_steps=100,
                                            history=hist), "r", META)
    check(len(df) == 10, "clean: 10 rows parsed")
    check(a["steps_monotonically_increasing"] is True, "clean: monotonic")
    check(a["duplicated_steps"] == [], "clean: no duplicates")
    check(a["n_missing_expected_logging_intervals"] == 0, "clean: no gaps")
    check(a["n_entries_with_loss"] == 10, "clean: loss count")
    check(a["n_malformed_or_nonfinite_records"] == 0, "clean: nothing malformed")
    check(list(df.columns)[:4] == ["run_id", "run_role", "seed", "max_steps"],
          "clean: identity columns are carried on every row")


def t_missing_optional_fields(tmp: Path) -> None:
    hist = [{"step": 10, "loss": 0.5},                       # no grad_norm / lr
            {"step": 20, "loss": 0.4, "grad_norm": 2.0},     # no lr
            {"step": 30, "loss": 0.3, "grad_norm": 1.0, "learning_rate": 1e-6}]
    df, a = parse_trainer_state(write_state(tmp, "missing.json", max_steps=30,
                                            history=hist), "r", META)
    check(df["grad_norm"].isna().sum() == 1, "missing: absent grad_norm -> NaN")
    check(df["learning_rate"].isna().sum() == 2, "missing: absent lr -> NaN")
    check(df["epoch"].isna().sum() == 3, "missing: absent epoch -> NaN")
    check(a["n_missing_by_field"]["learning_rate"] == 2,
          "missing: manifest records the missing-field status")
    check(df.loc[df["step"] == 10, "grad_norm"].isna().all(),
          "missing: absent field stays explicitly missing, not zero")


def t_entries_without_loss(tmp: Path) -> None:
    hist = [{"step": 10, "loss": 0.5, "grad_norm": 1.0, "learning_rate": 1e-6},
            {"step": 20, "grad_norm": 3.0, "learning_rate": 9e-7},   # eval-style
            {"step": 30, "loss": 0.3, "grad_norm": 1.0, "learning_rate": 8e-7}]
    df, a = parse_trainer_state(write_state(tmp, "noloss.json", max_steps=30,
                                            history=hist), "r", META)
    check(len(df) == 3, "no-loss: row retained")
    check(a["n_log_history_entries"] == 3, "no-loss: history size reported")
    check(a["n_entries_with_loss"] == 2, "no-loss: loss-bearing entries counted")
    check(bool(df["loss"].isna().iloc[1]), "no-loss: loss is NaN, not filled")


def t_duplicates_and_disorder(tmp: Path) -> None:
    hist = [{"step": 10, "loss": 0.5, "grad_norm": 1.0, "learning_rate": 1e-6},
            {"step": 30, "loss": 0.4, "grad_norm": 1.0, "learning_rate": 9e-7},
            {"step": 20, "loss": 0.3, "grad_norm": 1.0, "learning_rate": 8e-7},
            {"step": 30, "loss": 0.2, "grad_norm": 1.0, "learning_rate": 7e-7}]
    df, a = parse_trainer_state(write_state(tmp, "dup.json", max_steps=40,
                                            history=hist), "r", META)
    check(a["steps_monotonically_increasing"] is False, "disorder: detected")
    check(a["duplicated_steps"] == [30], "duplicates: reported")
    check(len(df) == 4, "duplicates: rows kept, not silently de-duplicated")
    check(40 in a["missing_expected_logging_intervals"],
          "gaps: missing expected logging interval reported")
    check(a["n_missing_expected_logging_intervals"] == 1, "gaps: counted")


def t_nonfinite(tmp: Path) -> None:
    hist = [{"step": 10, "loss": 0.5, "grad_norm": 1.0, "learning_rate": 1e-6},
            {"step": 20, "loss": float("nan"), "grad_norm": float("inf"),
             "learning_rate": 9e-7},
            {"step": 30, "loss": 0.3, "grad_norm": 1.0, "learning_rate": 8e-7},
            {"step": 40, "loss": "not-a-number", "grad_norm": 1.0,
             "learning_rate": 7e-7}]
    df, a = parse_trainer_state(write_state(tmp, "nonfinite.json", max_steps=40,
                                            history=hist), "r", META)
    check(a["n_malformed_or_nonfinite_records"] == 3,
          "non-finite: NaN + inf + non-numeric all flagged")
    check(math.isinf(df.loc[df["step"] == 20, "grad_norm"].iloc[0]),
          "non-finite: value preserved verbatim, not clipped")
    check(bool(df.loc[df["step"] == 40, "loss"].isna().iloc[0]),
          "non-finite: unparseable value -> NaN, never coerced to a number")
    reasons = " ".join(r["reason"] for r in a["malformed_or_nonfinite_records"])
    check("non-numeric loss" in reasons, "non-finite: reason recorded")


def t_malformed_entries(tmp: Path) -> None:
    hist = [{"loss": 0.5}, {"step": "abc", "loss": 0.4},
            {"step": 10, "loss": 0.3, "grad_norm": 1.0, "learning_rate": 1e-6}]
    df, a = parse_trainer_state(write_state(tmp, "malformed.json", max_steps=10,
                                            history=hist), "r", META)
    check(len(df) == 1, "malformed: only well-formed rows enter the table")
    check(a["n_malformed_or_nonfinite_records"] == 2, "malformed: both flagged")


def t_identity_and_separation(tmp: Path) -> None:
    """Pilot and full runs must stay separable, never merged into one series."""
    full = write_state(tmp, "full.json", max_steps=30000,
                       history=[{"step": s, "loss": 0.5, "grad_norm": 1.0,
                                 "learning_rate": 1e-6}
                                for s in range(10, 30001, 10)])
    pilot = write_state(tmp, "pilot.json", max_steps=6000,
                        history=[{"step": s, "loss": 0.5, "grad_norm": 1.0,
                                  "learning_rate": 1e-6}
                                 for s in range(10, 6001, 10)])
    dfa, aa = parse_trainer_state(full, "sft_full_seed1", META)
    pmeta = dict(META, run_role="pilot", expected_max_steps=6000)
    dfb, ab = parse_trainer_state(pilot, "sft_pilot6k_seed1", pmeta)
    check(aa["max_steps_matches_expected_identity"], "identity: full is 30k")
    check(ab["max_steps_matches_expected_identity"], "identity: pilot is 6k")
    check(set(dfa["run_id"]) == {"sft_full_seed1"}, "separation: full tagged")
    check(set(dfb["run_id"]) == {"sft_pilot6k_seed1"}, "separation: pilot tagged")
    check(dfa["max_steps"].iloc[0] != dfb["max_steps"].iloc[0],
          "separation: different cosine endpoints preserved per row")
    both = pd.concat([dfa, dfb])
    overlap = both[both["step"] == 6000]
    check(len(overlap) == 2 and overlap["run_id"].nunique() == 2,
          "separation: step 6,000 exists twice, one row per run, never merged")

    # a mislabelled source must be caught, not accepted
    _, bad = parse_trainer_state(pilot, "sft_full_seed1", META)
    check(bad["max_steps_matches_expected_identity"] is False,
          "identity: a 6k log presented as the 30k full run is flagged")


def t_causal_smoothing() -> None:
    s = pd.Series([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
    sm = causal_rolling(s, 3)
    check(sm.iloc[2] == 0.0,
          "smoothing: a point BEFORE a spike is unaffected (no future leakage)")
    check(sm.iloc[3] > 0, "smoothing: the spike enters at its own index")
    check(sm.iloc[0] == s.iloc[0], "smoothing: min_periods=1 at the left edge")
    check(len(sm) == len(s), "smoothing: length preserved")


def t_theoretical_lr() -> None:
    steps = np.array([1, 750, 1501, 15000, 30000])
    lr = theoretical_lr(steps, 30000, 1e-6, 0.05)
    check(abs(lr[0]) < 1e-18, "lr: step 1 is the start of warmup (clock offset -1)")
    check(abs(lr[1] - 1e-6 * 749 / 1500) < 1e-18, "lr: linear warmup segment")
    check(lr[2] > 0.999e-6, "lr: peak just after the 1,500-step boundary")
    check(lr[4] < 1e-12, "lr: cosine decays to ~0 at max_steps")
    mid = theoretical_lr(np.array([1 + 1500 + (30000 - 1500) // 2]), 30000, 1e-6, 0.05)
    check(abs(mid[0] - 0.5e-6) < 1e-12, "lr: cosine half-way point is half the peak")


def t_run_summary(tmp: Path) -> None:
    log = tmp / "run.log"
    log.write_text(
        "{'loss': '0.5', 'grad_norm': '1.0'}\n"
        "{'train_runtime': '1051', 'train_samples_per_second': '5.71', "
        "'train_steps_per_second': '5.71', 'train_loss': '0.4925', 'epoch': '0.06067'}\n"
        "{'train_runtime': '5422', 'train_samples_per_second': '5.533', "
        "'train_steps_per_second': '5.533', 'train_loss': '0.3374', 'epoch': '0.3033'}\n")
    full = parse_run_summary(log, 0.3033)
    check(full["found"] and full["train_runtime_s"] == 5422,
          "summary: the full run is matched by its final epoch")
    pilot = parse_run_summary(log, 0.06067)
    check(pilot["found"] and pilot["train_loss"] == 0.4925,
          "summary: the pilot is matched separately in the SAME log file")
    check(full["train_loss"] != pilot["train_loss"],
          "summary: pilot and full summaries are never conflated")
    none = parse_run_summary(log, 0.9999)
    check(not none["found"] and none["train_runtime_s"] is None,
          "summary: an unmatched epoch yields no value rather than a guess")
    absent = parse_run_summary(tmp / "does-not-exist.log", 0.3)
    check(not absent["found"], "summary: absent log is reported, not fabricated")

    amb = tmp / "amb.log"
    amb.write_text(
        "{'train_runtime': '1', 'train_loss': '0.1', 'epoch': '0.3033'}\n"
        "{'train_runtime': '2', 'train_loss': '0.2', 'epoch': '0.3033'}\n")
    a = parse_run_summary(amb, 0.3033)
    check(a["ambiguous"] and not a["found"],
          "summary: two summaries at the same epoch are ambiguous, not guessed")


def t_registry_sanity() -> None:
    roles = {r: m["run_role"] for r, m in RUNS.items()}
    check(sum(v == "full" for v in roles.values()) == 2, "registry: two full runs")
    check(sum(v == "pilot" for v in roles.values()) == 1, "registry: one pilot")
    check(RUNS["sft_pilot6k_seed1"]["expected_max_steps"] == 6000,
          "registry: pilot cosine endpoint is 6,000")
    check(all(m["expected_max_steps"] == 30000
              for m in RUNS.values() if m["run_role"] == "full"),
          "registry: full cosine endpoint is 30,000")
    check(RUNS["sft_full_seed1"]["state"] != RUNS["sft_pilot6k_seed1"]["state"],
          "registry: pilot and full seed 1 read different sources")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t_clean_history(tmp)
        t_missing_optional_fields(tmp)
        t_entries_without_loss(tmp)
        t_duplicates_and_disorder(tmp)
        t_nonfinite(tmp)
        t_malformed_entries(tmp)
        t_identity_and_separation(tmp)
        t_causal_smoothing()
        t_theoretical_lr()
        t_run_summary(tmp)
        t_registry_sanity()
    if _fail:
        print(f"FAILED {len(_fail)} / {_pass + len(_fail)} checks:")
        for f in _fail:
            print(f"  ✗ {f}")
        return 1
    print(f"OK — {_pass} checks passed (synthetic SFT trainer_state parsing).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
