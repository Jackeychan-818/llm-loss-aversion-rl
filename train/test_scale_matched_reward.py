#!/usr/bin/env python3
"""Tests for the scale-matched reward ablation (Task 3 / ABLATION-001).

Covers reward signs, scale calculation, config parsing / validation,
default-path equivalence (magnitude & sign_only unchanged), the deterministic
spec manifest, and the ENV-001 hard-fail on dropped algorithm-defining keys.

Run from the repository root:
    python3 train/test_scale_matched_reward.py
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

from reward_functions import (  # noqa: E402
    compute_reward, make_reward_fn, characterize_deltas, compute_scale_constant,
    VALID_WEIGHTINGS,
)
import grpo_train  # noqa: E402

_fail: list[str] = []
_pass = 0


def check(name, cond, detail=""):
    global _pass
    if cond:
        _pass += 1
        print(f"  PASS  {name}")
    else:
        _fail.append(f"{name}: {detail}")
        print(f"  FAIL  {name}: {detail}")


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


# --- 1. reward signs across weightings -------------------------------------
# X-perspective, delta=-1.81 -> rational = "Yes"
check("magnitude +|delta| rational",
      approx(compute_reward("Yes", "X", -1.81, "magnitude"), 1.81), "")
check("magnitude -|delta| irrational",
      approx(compute_reward("No", "X", -1.81, "magnitude"), -1.81), "")
check("sign_only +1 rational",
      approx(compute_reward("Yes", "X", -1.81, "sign_only"), 1.0), "")
check("sign_only -1 irrational",
      approx(compute_reward("No", "X", -1.81, "sign_only"), -1.0), "")
check("scale_matched +c rational",
      approx(compute_reward("Yes", "X", -1.81, "scale_matched", scale_constant=0.7), 0.7), "")
check("scale_matched -c irrational",
      approx(compute_reward("No", "X", -1.81, "scale_matched", scale_constant=0.7), -0.7), "")
# scale_matched magnitude is independent of |delta|
check("scale_matched independent of |delta|",
      approx(compute_reward("Yes", "Y", 3.5, "scale_matched", scale_constant=0.7),
             compute_reward("Yes", "X", -0.2, "scale_matched", scale_constant=0.7)), "")
# delta==0 always 0
check("delta==0 -> 0 (scale_matched)",
      approx(compute_reward("Yes", "X", 0.0, "scale_matched", scale_constant=0.7), 0.0), "")

# --- 2. scale calculation ---------------------------------------------------
sample = [3.0, -4.0, 0.0, 1.0, -1.0]  # non-zero abs: 3,4,1,1 -> mean 2.25, rms=sqrt(27/4)
stats = characterize_deltas(sample)
check("characterize n_nonzero", stats["n_nonzero"] == 4, str(stats["n_nonzero"]))
check("characterize mean_abs", approx(stats["mean_abs"], 2.25), str(stats["mean_abs"]))
check("characterize rms", approx(stats["rms"], (27 / 4) ** 0.5), str(stats["rms"]))
check("compute_scale_constant mean_abs", approx(compute_scale_constant(sample, "mean_abs"), 2.25), "")
check("compute_scale_constant rms", approx(compute_scale_constant(sample, "rms"), (27/4)**0.5), "")
try:
    compute_scale_constant(sample, "bogus")
    check("unknown scale rule raises", False, "no raise")
except ValueError:
    check("unknown scale rule raises", True)

# --- 3. config parsing / validation ----------------------------------------
check("VALID_WEIGHTINGS has scale_matched", "scale_matched" in VALID_WEIGHTINGS, str(VALID_WEIGHTINGS))
for bad in (None, 0.0, -1.0):
    try:
        compute_reward("Yes", "X", -1.0, "scale_matched", scale_constant=bad)
        check(f"scale_matched rejects c={bad}", False, "no raise")
    except ValueError:
        check(f"scale_matched rejects c={bad}", True)
try:
    make_reward_fn("scale_matched")  # missing constant
    check("make_reward_fn scale_matched needs c", False, "no raise")
except ValueError:
    check("make_reward_fn scale_matched needs c", True)
try:
    compute_reward("Yes", "X", -1.0, "nonsense")
    check("unknown weighting raises", False, "no raise")
except ValueError:
    check("unknown weighting raises", True)

# --- 4. default-path equivalence (magnitude & sign_only UNCHANGED) ----------
fn_mag = make_reward_fn()  # default magnitude
fn_mag_explicit = make_reward_fn("magnitude")
fn_sign = make_reward_fn("sign_only")
fn_scale = make_reward_fn("scale_matched", scale_constant=0.685029)
comps = ["Yes", "No"] + ["No"] * 14
r_mag = fn_mag(comps, ["X"] * 16, [-1.81] * 16)
r_mag2 = fn_mag_explicit(comps, ["X"] * 16, [-1.81] * 16)
r_sign = fn_sign(comps, ["X"] * 16, [-1.81] * 16)
r_scale = fn_scale(comps, ["X"] * 16, [-1.81] * 16)
check("magnitude default path unchanged", r_mag == r_mag2 and approx(r_mag[0], 1.81) and approx(r_mag[1], -1.81), str(r_mag[:2]))
check("sign_only path unchanged", approx(r_sign[0], 1.0) and approx(r_sign[1], -1.0), str(r_sign[:2]))
check("scale_matched group", approx(r_scale[0], 0.685029) and approx(r_scale[1], -0.685029), str(r_scale[:2]))
# zero-diversity group -> all zeros for every weighting
allno = ["No"] * 16
check("zero-diversity zeros (magnitude)", all(x == 0.0 for x in fn_mag(allno, ["X"]*16, [-1.81]*16)), "")
check("zero-diversity zeros (scale_matched)", all(x == 0.0 for x in fn_scale(allno, ["X"]*16, [-1.81]*16)), "")

# --- 5. deterministic spec manifest ----------------------------------------
r = subprocess.run([sys.executable, str(ROOT / "train" / "build_scale_matched_spec.py"), "--check"],
                   capture_output=True, text=True)
check("scale-matched spec --check byte-identical", r.returncode == 0, (r.stdout + r.stderr)[-300:])

# --- 6. ENV-001 hard-fail on dropped algorithm-defining keys ----------------
@dataclasses.dataclass
class FakeCfgFull:
    beta: float = 0.0
    epsilon: float = 0.0
    loss_type: str = ""
    scale_rewards: str = ""
    num_generations: int = 0
    temperature: float = 0.0
    mask_truncated_completions: bool = False
    max_completion_length: int = 0
    logging_steps: int = 0

@dataclasses.dataclass
class FakeCfgMissingCritical:
    logging_steps: int = 0          # supports a non-critical key only

full_kwargs = {"beta": 0.04, "epsilon": 0.2, "loss_type": "dapo",
               "scale_rewards": "none", "num_generations": 16, "temperature": 1.5,
               "mask_truncated_completions": True, "max_completion_length": 4,
               "logging_steps": 10}
# all critical keys supported -> no raise, returns them
out = grpo_train.filter_supported_config_kwargs(FakeCfgFull, full_kwargs)
check("critical keys pass when supported", out.get("beta") == 0.04 and "loss_type" in out, str(sorted(out)))
# critical keys unsupported -> hard-fail
try:
    grpo_train.filter_supported_config_kwargs(FakeCfgMissingCritical, full_kwargs)
    check("hard-fail on dropped critical key", False, "no SystemExit")
except SystemExit:
    check("hard-fail on dropped critical key", True)
# only a non-critical key unsupported -> warn, no raise
try:
    out2 = grpo_train.filter_supported_config_kwargs(FakeCfgFull, {**full_kwargs, "report_to": "none"})
    check("non-critical dropped key warns (no raise)", "report_to" not in out2, "report_to leaked")
except SystemExit:
    check("non-critical dropped key warns (no raise)", False, "raised on non-critical")

print(f"\n{_pass} passed, {len(_fail)} failed")
if _fail:
    for f in _fail:
        print("  -", f)
    raise SystemExit(1)
print("ALL SCALE-MATCHED REWARD TESTS PASSED")
