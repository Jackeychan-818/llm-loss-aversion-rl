#!/usr/bin/env python3
"""CPU-only tests for the Phase-B instrumentation (Amendment 2).

Verifies exactly the four properties the protocol requires of the
parameter-update-norm diagnostic, plus that an adapter-only snapshot is
loadable by the evaluation path before any GPU job is submitted.

    python train/test_sft_sensitivity_callbacks.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from sft_sensitivity_callbacks import (EarlyAdapterSnapshotCallback,  # noqa: E402
                                       UpdateNormCallback, trainable_params)

_fail: list[str] = []
_pass = 0


def check(cond, msg):
    global _pass
    if cond:
        _pass += 1
    else:
        _fail.append(msg)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)
        self.frozen = nn.Linear(4, 4, bias=False)
        self.frozen.weight.requires_grad_(False)   # stands in for the frozen base


class FakeState:
    def __init__(self, step=1):
        self.global_step = step


def step_with_delta(cb, model, delta: float, step: int = 1):
    """Simulate one optimizer step that moves every trainable param by `delta`."""
    cb.on_pre_optimizer_step(None, FakeState(step), None)
    with torch.no_grad():
        for _, p in trainable_params(model):
            p.add_(delta)
    cb.on_optimizer_step(None, FakeState(step), None)


# ── 1. a known synthetic step produces the expected nonzero norm ────────────
def t_known_step_expected_norm():
    m = Toy()
    cb = UpdateNormCallback(m)
    n_trainable = sum(p.numel() for _, p in trainable_params(m))
    delta = 0.01
    step_with_delta(cb, m, delta)
    # every trainable element moved by exactly `delta`, so the L2 norm is
    # sqrt(n * delta^2) = delta * sqrt(n)
    expected = delta * (n_trainable ** 0.5)
    got = cb.records[0]["update_norm_l2"]
    check(len(cb.records) == 1, "one optimizer step -> exactly one record")
    check(abs(got - expected) < 1e-5,
          f"norm matches the analytic value: got {got:.6f}, expected {expected:.6f}")
    check(got > 0, "a real step yields a NON-zero norm")
    rel = cb.records[0]["relative_update_norm"]
    check(rel is not None and rel > 0, "relative update norm is recorded and positive")
    check(abs(rel - got / cb.records[0]["param_norm_l2"]) < 1e-9,
          "relative norm equals ||dtheta|| / ||theta||")


# ── 2. no optimizer step produces zero ──────────────────────────────────────
def t_no_zero_updates():
    m = Toy()
    cb = UpdateNormCallback(m)
    for i in range(1, 6):
        step_with_delta(cb, m, 0.001 * i, step=i)
    s = cb.summary()
    check(s["n_optimizer_updates_recorded"] == 5, "five steps recorded")
    check(s["n_zero_updates"] == 0, "no optimizer step recorded a zero norm")
    check(all(r["update_norm_l2"] > 0 for r in cb.records),
          "every recorded update moved the parameters")
    # a genuinely zero step must still be REPORTED, not hidden
    cb2 = UpdateNormCallback(Toy())
    step_with_delta(cb2, cb2.model, 0.0)
    check(cb2.summary()["n_zero_updates"] == 1,
          "a genuinely zero update is reported rather than silently dropped")


# ── 3. accumulation microsteps are not counted as optimizer updates ─────────
def t_substeps_not_counted():
    m = Toy()
    cb = UpdateNormCallback(m)
    accum = 64                       # effective batch 64
    cb.on_pre_optimizer_step(None, FakeState(1), None)
    for _ in range(accum):
        cb.on_substep_end(None, FakeState(1), None)
    with torch.no_grad():
        for _, p in trainable_params(m):
            p.add_(0.01)
    cb.on_optimizer_step(None, FakeState(1), None)
    check(len(cb.records) == 1,
          f"{accum} accumulation microsteps produce ONE optimizer record, not {accum}")
    s = cb.summary()
    check(s["n_accumulation_substeps_seen"] == accum,
          "microsteps are counted separately for audit")
    check(s["n_optimizer_updates_recorded"] == 1,
          "the summary counts optimizer updates, not microsteps")


# ── 4. only trainable LoRA parameters are included ──────────────────────────
def t_only_trainable_params():
    m = Toy()
    names = [n for n, _ in trainable_params(m)]
    check("lin.weight" in names, "trainable parameter included")
    check("frozen.weight" not in names, "frozen base parameter EXCLUDED")

    cb = UpdateNormCallback(m)
    cb.on_pre_optimizer_step(None, FakeState(1), None)
    with torch.no_grad():
        m.frozen.weight.add_(1000.0)      # huge move in a FROZEN parameter
        m.lin.weight.add_(0.01)
    cb.on_optimizer_step(None, FakeState(1), None)
    n = sum(p.numel() for _, p in trainable_params(m))
    check(abs(cb.records[0]["update_norm_l2"] - 0.01 * (n ** 0.5)) < 1e-5,
          "a huge move in a frozen parameter does NOT enter the norm")


# ── 5. adapter-only snapshots are loadable by the evaluation path ───────────
def t_adapter_only_snapshot_loads():
    from peft import LoraConfig, PeftModel, get_peft_model
    base = Toy()
    peft_model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, target_modules=["lin"], lora_dropout=0.0, bias="none"))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        cb = EarlyAdapterSnapshotCallback(
            peft_model, out, {2: 128, 4: 256},
            {"cell": "unit-test", "git_commit": "deadbeef"})
        cb.on_step_end(None, FakeState(2), None)
        cb.on_step_end(None, FakeState(3), None)   # not on the grid
        cb.on_step_end(None, FakeState(4), None)

        check(len(cb.saved) == 2, "snapshots taken only at the frozen exposures")
        d = out / "early-exposure-128"
        check((d / "adapter_config.json").is_file(), "adapter_config.json written")
        weights = list(d.glob("adapter_model.*"))
        check(bool(weights), "adapter weight file written")
        check(not (d / "optimizer.pt").exists(),
              "no optimizer state — the snapshot is adapter-only by construction")

        meta = json.loads((d / "snapshot_metadata.json").read_text())
        check(meta["resume_supported"] is False, "snapshot marked resume_supported: false")
        check(meta["prompt_exposure"] == 128 and meta["optimizer_step"] == 2,
              "snapshot records its exposure and optimizer step")
        check(meta["git_commit"] == "deadbeef", "provenance carried into the snapshot")

        # the real load path used by eval/run_qwen_local.py --adapter_path
        reloaded = PeftModel.from_pretrained(Toy(), str(d))
        check(reloaded is not None,
              "PeftModel.from_pretrained loads the adapter-only snapshot "
              "(the exact call eval/run_qwen_local.py makes)")


def main() -> int:
    for fn in sorted((f for n, f in globals().items() if n.startswith("t_")),
                     key=lambda f: f.__code__.co_firstlineno):
        fn()
    if _fail:
        print(f"FAILED {len(_fail)} / {_pass + len(_fail)} checks:")
        for f in _fail:
            print(f"  x {f}")
        return 1
    print(f"OK - {_pass} checks passed (Phase-B instrumentation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
