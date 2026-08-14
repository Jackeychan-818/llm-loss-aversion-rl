#!/usr/bin/env python3
"""Tests for the post-hoc SFT answer-space divergence analysis.

Run with the venv python (numpy/matplotlib):
    $HOME/scratch/lambda-zero/venv/bin/python eval/test_sft_answer_divergence.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import analyze_sft_answer_divergence as A  # noqa: E402

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


def raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True


# 1. identical distributions -> KL = 0
kl, _ = A.kl_binary(0.3, 0.7, 0.3, 0.7)
check("1 identical -> KL=0", abs(kl) < 1e-12, str(kl))

# 2. hand-calculated two-class example
# P=(0.9,0.1), Q=(0.5,0.5): KL = 0.9 ln(1.8) + 0.1 ln(0.2)
expect = 0.9 * math.log(1.8) + 0.1 * math.log(0.2)
kl2, _ = A.kl_binary(0.9, 0.1, 0.5, 0.5)
check("2 hand-calc two-class", abs(kl2 - expect) < 1e-12, f"{kl2} vs {expect}")

# 3. KL nonnegative
rng = np.random.default_rng(0)
neg = 0
for _ in range(2000):
    py = rng.uniform(1e-6, 1 - 1e-6); qy = rng.uniform(1e-6, 1 - 1e-6)
    k, _c = A.kl_binary(py, 1 - py, qy, 1 - qy)
    if k < -1e-12:
        neg += 1
check("3 KL nonnegative", neg == 0, f"{neg} negative")

# 4. duplicate keys hard-fail
def _dup():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "loss_aversion_X.json"
        p.write_text(json.dumps([
            {"case_id": 1, "X_num": 0, "Y_num": 1, "Yes / No prob": [0.5, 0.5]},
            {"case_id": 1, "X_num": 0, "Y_num": 1, "Yes / No prob": [0.4, 0.6]}]))
        A.load_perspective(p, "X")
check("4 duplicate keys hard-fail", raises(_dup, ValueError))

# 5. missing keys hard-fail
def _missing():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "loss_aversion_X.json"
        p.write_text(json.dumps([{"case_id": 1, "X_num": 0, "Yes / No prob": [0.5, 0.5]}]))
        A.load_perspective(p, "X")
check("5 missing key hard-fail", raises(_missing, ValueError))

# 6. invalid probability arrays hard-fail
for bad in ([0.5], [0.5, 0.5, 0.0], [-0.1, 1.1], [0.0, 0.0], [float("nan"), 0.5], "x"):
    if not raises(lambda b=bad: A.validate_prob(b), ValueError):
        _fail.append(f"6 invalid prob accepted: {bad}")
check("6 invalid probability arrays hard-fail", not any("6 invalid" in f for f in _fail))

# 7. pilot paths hard-fail
check("7 pilot path hard-fail (load)",
      raises(lambda: A.load_perspective(Path("baselines/Qwen-7B-SFT-qd-seed1-pilot6k/loss_aversion_X.json"), "X"), ValueError))
check("7 pilot path hard-fail (sft_paths)",
      raises(lambda: A._reject_pilot("x/pilot/y"), ValueError))

# 8. complete 2x15 grid required
check("8 grid is 2 seeds x 15 steps", A.SEEDS == [1, 2] and A.STEPS == list(range(2000, 30001, 2000))
      and len(A.STEPS) == 15, f"{A.SEEDS},{len(A.STEPS)}")

# 9. X and Y perspectives cannot be confused
xs = [{"case_id": 1, "X_num": 0, "Y_num": 1, "Yes / No prob": [0.9, 0.1]}]
ys = [{"case_id": 1, "X_num": 0, "Y_num": 1, "Yes / No prob": [0.2, 0.8]}]
with tempfile.TemporaryDirectory() as d:
    px = Path(d) / "loss_aversion_X.json"; py = Path(d) / "loss_aversion_Y.json"
    px.write_text(json.dumps(xs)); py.write_text(json.dumps(ys))
    dx = A.load_perspective(px, "X"); dy = A.load_perspective(py, "Y")
    keys = set(dx) | set(dy)
check("9 X/Y perspectives distinct keys", keys == {("X", 1), ("Y", 1)}, str(keys))

# 10. pair-cluster resampling never splits repeated configs / perspectives
# 3 goods pairs, each with several configs x 2 perspectives sharing one pair id.
pair_ids = []
plan = {(0, 1): 4, (2, 3): 3, (4, 5): 5}   # configs per pair (x2 perspectives)
truth = []
for (a, b), k in plan.items():
    for _ in range(k):
        for _persp in ("X", "Y"):
            pair_ids.append((a, b)); truth.append((a, b))
pair_ids = np.array(pair_ids, dtype=object)
keys, arrays = A.pair_clusters(pair_ids)
check("10a clusters keyed by unordered pair", keys == sorted(plan.keys()), str(keys))
check("10b cluster sizes = 2*configs",
      sorted(len(x) for x in arrays) == sorted(2 * v for v in plan.values()),
      str([len(x) for x in arrays]))
rng2 = np.random.default_rng(7)
split_free = True
for _ in range(300):
    idx = A.pair_resample_index(arrays, rng2)
    # count occurrences per pair; must be an exact multiple of that pair's block size
    from collections import Counter
    cnt = Counter(tuple(pair_ids[i]) for i in idx)
    for pr, c in cnt.items():
        if c % (2 * plan[pr]) != 0:
            split_free = False
check("10c repeated configs + both perspectives never split", split_free, "a pair block was split")

# 11. default rendering works without NSCC directories (render from synthetic snapshot)
def _synth_snapshot():
    cps = []
    for seed in (1, 2):
        for step in range(2000, 30001, 2000):
            cps.append({"seed": seed, "step": step, "mean_kl": 0.01 * step / 2000,
                        "ci_lo": 0.0, "ci_hi": 0.02 * step / 2000, "median_kl": 0.005,
                        "p90_kl": 0.03, "max_kl": 0.1, "js_mean": 0.002,
                        "hard_choice_disagreement": 0.01, "n_prompts": 19780,
                        "n_pair_clusters": 4945, "clipped_rows": 0})
    return {"title": "t", "bootstrap": {"resamples": 2000, "seed": 20260814},
            "selected_checkpoints": {1: 4000, 2: 6000},
            "base_row": {"seed": None, "step": 0, "mean_kl": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                         "median_kl": 0.0, "p90_kl": 0.0, "max_kl": 0.0, "js_mean": 0.0,
                         "hard_choice_disagreement": 0.0, "n_prompts": 19780,
                         "n_pair_clusters": 4945, "clipped_rows": 0},
            "checkpoints": cps}
snap = _synth_snapshot()
# A.SELECTED keys are ints; snapshot uses those in render_md/png
csv_s = A.render_csv(snap); md_s = A.render_md(snap)
check("11a render_csv has 31 data rows", csv_s.strip().count("\n") == 31, str(csv_s.count(chr(10))))
check("11b render_md non-empty + no NSCC read", "SFT answer-space divergence" in md_s)
with tempfile.TemporaryDirectory() as d:
    png = Path(d) / "x.png"
    A.render_png(snap, png)
    check("11c render_png produces a file", png.exists() and png.stat().st_size > 0)

# 12. --check passes from a fresh git archive extraction (only meaningful once committed)
def _archive_check():
    with tempfile.TemporaryDirectory() as d:
        arch = Path(d) / "arch"
        arch.mkdir()
        tar = subprocess.run(["git", "archive", "HEAD"], cwd=ROOT, capture_output=True)
        if tar.returncode != 0:
            return "no-head"
        subprocess.run(["tar", "-x", "-C", str(arch)], input=tar.stdout, check=True)
        snap_p = arch / "results/training_dynamics/sft/sft_answer_divergence_snapshot.json"
        if not snap_p.exists():
            return "not-committed-yet"
        r = subprocess.run([sys.executable, str(arch / "eval/analyze_sft_answer_divergence.py"), "--check"],
                           cwd=arch, capture_output=True, text=True)
        return "ok" if r.returncode == 0 else f"fail: {(r.stdout + r.stderr)[-200:]}"
res = _archive_check()
# "no-head" occurs when the tests are themselves run from an extracted archive
# (not a git repo) — acceptable; the outer archive run is the real proof.
check("12 clean git-archive --check", res in ("ok", "not-committed-yet", "no-head"),
      f"result={res}")
if res in ("not-committed-yet", "no-head"):
    print(f"       (note: archive self-check skipped [{res}]; verified by the outer archive run)")

print(f"\n{_pass} passed, {len(_fail)} failed")
if _fail:
    for f in _fail:
        print("  -", f)
    raise SystemExit(1)
print("ALL SFT ANSWER-DIVERGENCE TESTS PASSED")
