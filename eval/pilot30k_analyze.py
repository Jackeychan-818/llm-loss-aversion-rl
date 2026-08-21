#!/usr/bin/env python3
"""Exploratory 30k-checkpoint diagnostic — analysis (CPU only).

Compares the two LATE magnitude-GRPO endpoints (seed1@30000, seed2@30000)
against the previously evaluated frozen-selected checkpoints (seed1@2000,
seed2@6000) and the matched local base, on two already-available exploratory
axes:

  * surface-form stress  (data/surface_form_stress/, 96 cases x 2 persp x 48 forms)
  * adverse framing      (data/framing_effects_23prob.json, 120 x 23 x 2 = 5,520)

EXPLORATORY / POST-HOC. Nothing here can change the frozen checkpoint
selections, and the 30k adapters are NOT "selected" models.

Paired 30k-minus-selected differences use a cluster bootstrap:
  * surface form -> clusters are (case_id, perspective) units, except the
    ownership keep-both rate, which is inherently case-level and is therefore
    clustered by case_id (recorded per metric in the JSON).
  * framing      -> clusters are scenarios.
Individual prompt forms are NEVER treated as independent samples.

Predeclared exploratory flag rule: a change is "potentially meaningful" when the
paired 95% interval excludes zero OR |Delta| >= 0.05. This is a decision rule
for whether to run the complete checkpoint trajectory, not a confirmatory test.

    python eval/pilot30k_analyze.py            # full run
    python eval/pilot30k_analyze.py --bootstrap_reps 200   # quick
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from run_framing_local import (  # noqa: E402  (identical metric definitions)
    build_complete_pairs,
    monotonicity_summary,
    pair_metric_summary,
)

SUBSET_PATH = ROOT / "data" / "surface_form_stress" / "surface_form_subset.json"
FRAMING_DATA = ROOT / "data" / "framing_effects_23prob.json"
STRUCTURAL_CSV = ROOT / "results" / "training_dynamics" / "structural_trajectory.csv"

EXPECTED_SUBSET_SHA = "78525395ea208e4e4f2ef3e4266a5386259175cdcabeb29d8fb887cb4b272262"
EXPECTED_FRAMING_SHA = "dd58443d5d4a014f540e0406fc4d29166f5d5193c525c42851a595742c2cc15c"
EXPECTED_SURFACE_ROWS = 9216
EXPECTED_FRAMING_ROWS = 5520

AXES = ["answer_style", "display_order", "attr_order", "paraphrase"]

# model_key -> (which result root, adapter path, expected adapter sha256, role)
SURFACE_MODELS = {
    "Base": (
        "baseline", None, None,
        "matched local base (context)"),
    "GRPO-qd-seed1-ckpt2000": (
        "baseline", "checkpoints/grpo_qwen_delta_seed1/checkpoint-2000",
        "223b80bd16e7383b09c73287b60e3dde4c26608c64ce9365557d5a88199bc190",
        "seed1 frozen-selected checkpoint"),
    "GRPO-qd-seed2-ckpt6000": (
        "baseline", "checkpoints/grpo_qwen_delta_seed2/checkpoint-6000",
        "e6bd31a865d71aeb33d5db08ec86c4c9211aba55f1b181ca28026621686df52c",
        "seed2 frozen-selected checkpoint"),
    "GRPO-qd-seed1-ckpt30000": (
        "pilot", "checkpoints/grpo_qwen_delta_seed1/checkpoint-30000",
        "6b9580a4ed24fabf910cfd654f555a867149e2beeb9f277650aa3d52324ffd0d",
        "seed1 late endpoint (exploratory, NOT selected)"),
    "GRPO-qd-seed2-ckpt30000": (
        "pilot", "checkpoints/grpo_qwen_delta_seed2/checkpoint-30000",
        "3d02042db2349e5c732210f243922a983e9506872711c1fbd68ceed00a4e107e",
        "seed2 late endpoint (exploratory, NOT selected)"),
}

SURFACE_PAIRS = [
    ("GRPO-qd-seed1-ckpt30000", "GRPO-qd-seed1-ckpt2000"),
    ("GRPO-qd-seed2-ckpt30000", "GRPO-qd-seed2-ckpt6000"),
]

# framing: model_key -> (root kind, subdir, adapter, expected sha, role)
FRAMING_MODELS = {
    "Qwen-7B-Base": (
        "existing", "framing/full_qd8k_120x23/Qwen-7B-Base/single_word", None, None,
        "matched local base (comparator)"),
    "Qwen-7B-GRPO-step8000": (
        "existing", "framing/full_qd8k_120x23/Qwen-7B-GRPO-step8000/single_word",
        "checkpoints/grpo_qwen_delta/checkpoint-8000", None,
        "seed42 step-8000 (EXPLORATORY; not a selected checkpoint)"),
    "GRPO-qd-seed1-ckpt30000": (
        "pilot", "GRPO-qd-seed1-ckpt30000/single_word",
        "checkpoints/grpo_qwen_delta_seed1/checkpoint-30000",
        "6b9580a4ed24fabf910cfd654f555a867149e2beeb9f277650aa3d52324ffd0d",
        "seed1 late endpoint (exploratory, NOT selected)"),
    "GRPO-qd-seed2-ckpt30000": (
        "pilot", "GRPO-qd-seed2-ckpt30000/single_word",
        "checkpoints/grpo_qwen_delta_seed2/checkpoint-30000",
        "3d02042db2349e5c732210f243922a983e9506872711c1fbd68ceed00a4e107e",
        "seed2 late endpoint (exploratory, NOT selected)"),
}

FRAMING_PAIRS = [
    ("GRPO-qd-seed1-ckpt30000", "Qwen-7B-Base"),
    ("GRPO-qd-seed2-ckpt30000", "Qwen-7B-Base"),
    ("GRPO-qd-seed1-ckpt30000", "Qwen-7B-GRPO-step8000"),
    ("GRPO-qd-seed2-ckpt30000", "Qwen-7B-GRPO-step8000"),
]

# structural_trajectory.csv model_name -> our model key
STRUCTURAL_KEY = {
    "Qwen-7B-Base-Local": "Base",
    "Qwen-7B-GRPO-qd-seed1-ckpt2000": "GRPO-qd-seed1-ckpt2000",
    "Qwen-7B-GRPO-qd-seed2-ckpt6000": "GRPO-qd-seed2-ckpt6000",
    "Qwen-7B-GRPO-qd-seed1-ckpt30000": "GRPO-qd-seed1-ckpt30000",
    "Qwen-7B-GRPO-qd-seed2-ckpt30000": "GRPO-qd-seed2-ckpt30000",
    "Qwen-7B-GRPO-qd-ckpt8000": "Qwen-7B-GRPO-step8000",
}

FLAG_ABS = 0.05


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rel(path: Path) -> str:
    """Repo-relative path where possible; absolute otherwise (temp dirs in tests)."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def ci95(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return [None, None]
    return [percentile(vals, 0.025), percentile(vals, 0.975)]


def simpson_flip(vals):
    """Pairwise disagreement probability over a unit's final goods."""
    n = len(vals)
    if n < 2:
        return 0.0
    counts = Counter(vals)
    same = sum(c * (c - 1) for c in counts.values())
    return 1.0 - same / (n * (n - 1))


def flag(delta, lo, hi):
    if delta is None:
        return False, "no paired estimate"
    excl = lo is not None and hi is not None and (lo > 0 or hi < 0)
    big = abs(delta) >= FLAG_ABS
    if excl and big:
        reason = "CI excludes 0 and |delta| >= 0.05"
    elif excl:
        reason = "CI excludes 0"
    elif big:
        reason = "|delta| >= 0.05"
    else:
        reason = "neither criterion met"
    return (excl or big), reason


# --------------------------------------------------------------------------- #
# surface form
# --------------------------------------------------------------------------- #
def load_surface(model_key, surface_root, baseline_root, strict=True):
    kind, adapter, exp_sha, role = SURFACE_MODELS[model_key]
    base_dir = (surface_root if kind == "pilot" else baseline_root) / model_key
    pred = base_dir / "form_predictions.jsonl"
    meta_path = base_dir / "run_metadata.json"
    prov = {"model": model_key, "role": role, "result_dir": _rel(base_dir),
            "adapter_path": adapter, "expected_adapter_sha256": exp_sha,
            "reused_existing": kind == "baseline"}
    if not pred.exists():
        prov["status"] = "MISSING"
        return None, prov
    rows = [json.loads(l) for l in open(pred) if l.strip()]
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    prov.update({
        "observed_rows": len(rows), "expected_rows": EXPECTED_SURFACE_ROWS,
        "observed_adapter_sha256": meta.get("adapter_sha256"),
        "observed_subset_sha256": meta.get("subset_sha256"),
        "expected_subset_sha256": EXPECTED_SUBSET_SHA,
        "predictions_sha256": sha256_file(pred),
    })
    problems = []
    if len(rows) != EXPECTED_SURFACE_ROWS:
        problems.append(f"row count {len(rows)} != {EXPECTED_SURFACE_ROWS}")
    if meta.get("subset_sha256") != EXPECTED_SUBSET_SHA:
        problems.append("subset sha mismatch")
    if exp_sha is not None and meta.get("adapter_sha256") != exp_sha:
        problems.append("adapter sha mismatch")
    if adapter is None and meta.get("adapter_sha256") is not None:
        problems.append("expected no adapter but one is recorded")
    prov["integrity_problems"] = problems
    prov["status"] = "OK" if not problems else "FAILED_INTEGRITY"
    if problems and strict:
        raise SystemExit(f"Integrity failure for {model_key}: {problems}")
    return rows, prov


def surface_units(rows):
    """(case_id, perspective) -> list of 48 form rows."""
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[(r["case_id"], r["perspective"])].append(r)
    return by_unit


def surface_metrics(by_unit, cases, unit_keys=None):
    """Compute the surface-form metrics over the given unit keys (with
    repetition allowed, for the cluster bootstrap)."""
    keys = list(by_unit.keys()) if unit_keys is None else list(unit_keys)
    pref = {cid: (c["X_num"] if c["delta"] > 0 else c["Y_num"]) for cid, c in cases.items()}

    inv, flips, spreads, fid = [], [], [], []
    per_axis = {ax: defaultdict(lambda: [0, 0]) for ax in AXES}
    per_form = defaultdict(lambda: [0, 0])
    modal = {}
    strat = defaultdict(lambda: {"inv": [], "fid": [0, 0]})

    for k in keys:
        frs = by_unit[k]
        cid = k[0]
        finals = [r["final_good"] for r in frs]
        modal[k] = Counter(finals).most_common(1)[0][0]
        is_inv = len(set(finals)) == 1
        inv.append(is_inv)
        flips.append(simpson_flip(finals))
        pts = [r["p_trade"] for r in frs]
        spreads.append(max(pts) - min(pts))
        p = pref[cid]
        sk = "%s|%s" % (cases[cid]["delta_sign"], cases[cid]["delta_bin"])
        strat[sk]["inv"].append(is_inv)
        for r in frs:
            hit = int(r["final_good"] == p)
            fid.append(hit)
            strat[sk]["fid"][0] += hit
            strat[sk]["fid"][1] += 1
            for ax in AXES:
                per_axis[ax][r[ax]][0] += hit
                per_axis[ax][r[ax]][1] += 1
            fk = (r["answer_style"], r["display_order"], r["attr_order"], r["paraphrase"])
            per_form[fk][0] += hit
            per_form[fk][1] += 1
    worst = min(((h / n, k) for k, (h, n) in per_form.items()), default=(None, None))
    return {
        "n_units": len(keys),
        "n_form_rows": sum(len(by_unit[k]) for k in keys),
        "parse_failures": 0,
        "parse_failure_note": "teacher-forced scoring: the choice is always defined",
        "semantic_invariance_rate": sum(inv) / len(inv) if inv else None,
        "fidelity_mean": sum(fid) / len(fid) if fid else None,
        "worst_form_fidelity": worst[0],
        "worst_form": ("%s/%s/%s/%s" % worst[1]) if worst[1] else None,
        "mean_pairwise_flip_rate": sum(flips) / len(flips) if flips else None,
        "mean_prob_spread_across_forms": sum(spreads) / len(spreads) if spreads else None,
        "fidelity_by_axis": {ax: {v: h / n for v, (h, n) in sorted(d.items())}
                             for ax, d in per_axis.items()},
        "by_stratum": {s: {"invariance": sum(d["inv"]) / len(d["inv"]),
                           "fidelity": d["fid"][0] / d["fid"][1]}
                       for s, d in sorted(strat.items())},
        "_modal": modal,
    }


def keep_both_rate(by_unit, cases, case_keys=None):
    """Case-level ownership metric: modal choice keeps the endowed good in BOTH
    perspectives."""
    keys = list(cases.keys()) if case_keys is None else list(case_keys)
    hit = tot = 0
    for cid in keys:
        ux, uy = by_unit.get((cid, "X")), by_unit.get((cid, "Y"))
        if not ux or not uy:
            continue
        mx = Counter(r["final_good"] for r in ux).most_common(1)[0][0]
        my = Counter(r["final_good"] for r in uy).most_common(1)[0][0]
        tot += 1
        hit += int(mx == cases[cid]["X_num"] and my == cases[cid]["Y_num"])
    return hit / tot if tot else None


SURFACE_UNIT_METRICS = ["semantic_invariance_rate", "fidelity_mean", "worst_form_fidelity",
                        "mean_pairwise_flip_rate", "mean_prob_spread_across_forms"]

FORM_KEYS = None  # populated on first use; fixed ordering of the 48 form combos


def surface_sufficient_stats(by_unit, cases):
    """Per-cluster sufficient statistics, so the bootstrap never re-scans rows.

    Returns numpy arrays indexed by a FIXED unit ordering:
      inv[U]        unit is strictly invariant across its 48 forms
      fid[U, 48]    per-form hit (chosen final good == frozen preferred good)
      flip[U]       pairwise disagreement probability across forms
      spread[U]     max-min p_trade across forms
    plus keep[C] over a fixed case ordering (both perspectives keep the endowed
    good under the modal choice).

    Every metric in SURFACE_UNIT_METRICS is a plain mean (or the column-min of
    column means) over these, so bootstrap values are exact, not approximated.
    """
    global FORM_KEYS
    import numpy as np

    pref = {cid: (c["X_num"] if c["delta"] > 0 else c["Y_num"]) for cid, c in cases.items()}
    unit_keys = sorted(by_unit.keys())
    if FORM_KEYS is None:
        FORM_KEYS = sorted({(r["answer_style"], r["display_order"], r["attr_order"],
                             r["paraphrase"]) for k in unit_keys for r in by_unit[k]})
    form_idx = {k: i for i, k in enumerate(FORM_KEYS)}

    n_u, n_f = len(unit_keys), len(FORM_KEYS)
    inv = np.zeros(n_u, dtype=float)
    fid = np.zeros((n_u, n_f), dtype=float)
    flip = np.zeros(n_u, dtype=float)
    spread = np.zeros(n_u, dtype=float)
    modal = {}
    for i, k in enumerate(unit_keys):
        frs = by_unit[k]
        finals = [r["final_good"] for r in frs]
        modal[k] = Counter(finals).most_common(1)[0][0]
        inv[i] = float(len(set(finals)) == 1)
        flip[i] = simpson_flip(finals)
        pts = [r["p_trade"] for r in frs]
        spread[i] = max(pts) - min(pts)
        p = pref[k[0]]
        for r in frs:
            j = form_idx[(r["answer_style"], r["display_order"],
                          r["attr_order"], r["paraphrase"])]
            fid[i, j] = float(r["final_good"] == p)

    case_keys = sorted(cases.keys())
    keep = np.zeros(len(case_keys), dtype=float)
    for j, cid in enumerate(case_keys):
        mx, my = modal.get((cid, "X")), modal.get((cid, "Y"))
        keep[j] = (float(mx == cases[cid]["X_num"] and my == cases[cid]["Y_num"])
                   if mx is not None and my is not None else 0.0)
    return {"unit_keys": unit_keys, "case_keys": case_keys,
            "inv": inv, "fid": fid, "flip": flip, "spread": spread, "keep": keep}


def surface_stats_metrics(st, unit_idx=None, case_idx=None):
    """The five unit-level metrics + keep-both, from sufficient statistics."""
    import numpy as np
    ui = np.arange(len(st["unit_keys"])) if unit_idx is None else unit_idx
    ci = np.arange(len(st["case_keys"])) if case_idx is None else case_idx
    f = st["fid"][ui]
    return {
        "semantic_invariance_rate": float(st["inv"][ui].mean()),
        "fidelity_mean": float(f.mean()),
        "worst_form_fidelity": float(f.mean(axis=0).min()),
        "mean_pairwise_flip_rate": float(st["flip"][ui].mean()),
        "mean_prob_spread_across_forms": float(st["spread"][ui].mean()),
        "ownership_keep_both_rate": float(st["keep"][ci].mean()),
    }


def surface_paired_bootstrap(st_a, st_b, reps, seed):
    """Paired (a - b) cluster bootstrap. Unit-level metrics resample
    (case_id, perspective) clusters; keep-both resamples case_id clusters,
    because keeping both is defined across a case's two perspectives."""
    import numpy as np
    assert st_a["unit_keys"] == st_b["unit_keys"], "unit ordering must be aligned"
    assert st_a["case_keys"] == st_b["case_keys"], "case ordering must be aligned"
    rng = np.random.default_rng(seed)
    n_u, n_c = len(st_a["unit_keys"]), len(st_a["case_keys"])
    names = SURFACE_UNIT_METRICS + ["ownership_keep_both_rate"]
    draws = {m: [] for m in names}
    for _ in range(reps):
        ui = rng.integers(0, n_u, n_u)
        ci = rng.integers(0, n_c, n_c)
        ma = surface_stats_metrics(st_a, ui, ci)
        mb = surface_stats_metrics(st_b, ui, ci)
        for m in names:
            draws[m].append(ma[m] - mb[m])
    return draws


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #
FRAMING_METRICS = {
    "hard_flip_rate": ("pair", "hard_flip_rate"),
    "mean_absolute_probability_gap": ("pair", "mean_absolute_probability_gap"),
    "mean_probability_gap_negative_minus_positive":
        ("pair", "mean_probability_gap_negative_minus_positive"),
    "classical_flip_rate": ("pair", "classical_flip_rate"),
    "probability_monotonicity_violation_rate": ("mono", "probability_violation_rate"),
    "hard_choice_monotonicity_violation_rate": ("mono", "hard_choice_violation_rate"),
}


def load_framing(model_key, framing_root, strict=True):
    kind, sub, adapter, exp_sha, role = FRAMING_MODELS[model_key]
    d = (framing_root / sub) if kind == "pilot" else (ROOT / sub)
    pred_path = d / "predictions.json"
    man_path = d / "manifest.json"
    prov = {"model": model_key, "role": role, "result_dir": _rel(d),
            "adapter_path": adapter, "expected_adapter_sha256": exp_sha,
            "reused_existing": kind == "existing"}
    if not pred_path.exists():
        prov["status"] = "MISSING"
        return None, prov
    rows = json.loads(pred_path.read_text())
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    prov.update({
        "observed_rows": len(rows), "expected_rows": EXPECTED_FRAMING_ROWS,
        "observed_benchmark_sha256": man.get("benchmark_sha256"),
        "expected_benchmark_sha256": EXPECTED_FRAMING_SHA,
        "manifest_adapter_path": man.get("adapter_path"),
        "prompt_style": man.get("prompt_style"),
        "predictions_sha256": sha256_file(pred_path),
        "predictions_bytes": pred_path.stat().st_size,
    })
    problems = []
    if len(rows) != EXPECTED_FRAMING_ROWS:
        problems.append(f"row count {len(rows)} != {EXPECTED_FRAMING_ROWS}")
    if man.get("benchmark_sha256") != EXPECTED_FRAMING_SHA:
        problems.append("benchmark sha mismatch")
    if man.get("prompt_style") != "single_word":
        problems.append("prompt style is not single_word")
    if adapter and man.get("adapter_path") and not str(man["adapter_path"]).endswith(adapter):
        problems.append("manifest adapter path mismatch")
    if exp_sha is not None:
        p = ROOT / adapter / "adapter_model.safetensors"
        got = sha256_file(p) if p.exists() else None
        prov["observed_adapter_sha256"] = got
        if got != exp_sha:
            problems.append("adapter sha mismatch")
    prov["integrity_problems"] = problems
    prov["status"] = "OK" if not problems else "FAILED_INTEGRITY"
    if problems and strict:
        raise SystemExit(f"Integrity failure for framing {model_key}: {problems}")
    return rows, prov


def framing_by_scenario(rows):
    d = defaultdict(list)
    for r in rows:
        d[int(r["scenario_id"])].append(r)
    return d


def framing_metrics(by_scen, scen_keys=None):
    keys = list(by_scen.keys()) if scen_keys is None else list(scen_keys)
    rows = []
    for i, k in enumerate(keys):
        for r in by_scen[k]:
            # unique scenario id per draw so repeated clusters stay distinct
            rr = dict(r)
            rr["scenario_id"] = i
            rows.append(rr)
    pairs = build_complete_pairs(rows)
    pair = pair_metric_summary(pairs)
    mono = monotonicity_summary(rows)
    out = {"pair_metrics": pair, "monotonicity": mono}
    flat = {}
    for name, (src, key) in FRAMING_METRICS.items():
        flat[name] = (pair if src == "pair" else mono).get(key)
    out["headline"] = flat
    return out


def framing_domain_metrics(rows):
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)
    out = {}
    for dom, drows in sorted(by_dom.items()):
        pairs = build_complete_pairs(drows)
        p = pair_metric_summary(pairs)
        m = monotonicity_summary(drows)
        out[dom] = {
            "complete_pairs": p["complete_pairs"],
            "hard_flip_rate": p["hard_flip_rate"],
            "mean_absolute_probability_gap": p["mean_absolute_probability_gap"],
            "probability_monotonicity_violation_rate": m["probability_violation_rate"],
            "hard_choice_monotonicity_violation_rate": m["hard_choice_violation_rate"],
        }
    return out


def framing_sufficient_stats(by_scen):
    """Per-scenario sufficient statistics for the framing headline metrics.

    Every headline metric is a ratio of sums (pair means over complete
    positive/negative pairs; monotonicity violations over adjacent transitions),
    and both numerator and denominator decompose by scenario, so a scenario
    cluster bootstrap over these sums is exact.
    """
    import numpy as np
    keys = sorted(by_scen.keys())
    cols = ["n_pairs", "hard_flip", "abs_gap", "signed_gap", "classical_flip",
            "transitions", "prob_viol", "hard_viol"]
    m = {c: np.zeros(len(keys), dtype=float) for c in cols}
    for i, k in enumerate(keys):
        rows = by_scen[k]
        pairs = build_complete_pairs(rows)
        m["n_pairs"][i] = len(pairs)
        m["hard_flip"][i] = sum(float(p["hard_flip"]) for p in pairs)
        m["abs_gap"][i] = sum(p["absolute_probability_gap"] for p in pairs)
        m["signed_gap"][i] = sum(
            p["probability_gap_negative_minus_positive"] for p in pairs)
        m["classical_flip"][i] = sum(float(p["classical_flip"]) for p in pairs)
        mono = monotonicity_summary(rows)
        m["transitions"][i] = mono["adjacent_transitions"]
        m["prob_viol"][i] = mono["probability_violation_count"]
        m["hard_viol"][i] = mono["hard_choice_violation_count"]
    m["scen_keys"] = keys
    return m


def framing_stats_metrics(st, idx=None):
    import numpy as np
    i = np.arange(len(st["scen_keys"])) if idx is None else idx
    npairs = st["n_pairs"][i].sum()
    ntrans = st["transitions"][i].sum()

    def r(num, den):
        return float(num / den) if den else None

    return {
        "hard_flip_rate": r(st["hard_flip"][i].sum(), npairs),
        "mean_absolute_probability_gap": r(st["abs_gap"][i].sum(), npairs),
        "mean_probability_gap_negative_minus_positive":
            r(st["signed_gap"][i].sum(), npairs),
        "classical_flip_rate": r(st["classical_flip"][i].sum(), npairs),
        "probability_monotonicity_violation_rate": r(st["prob_viol"][i].sum(), ntrans),
        "hard_choice_monotonicity_violation_rate": r(st["hard_viol"][i].sum(), ntrans),
    }


def framing_paired_bootstrap(st_a, st_b, reps, seed):
    """Paired (a - b) cluster bootstrap over scenarios."""
    import numpy as np
    assert st_a["scen_keys"] == st_b["scen_keys"], "scenario ordering must be aligned"
    rng = np.random.default_rng(seed)
    n = len(st_a["scen_keys"])
    draws = {m: [] for m in FRAMING_METRICS}
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        ha = framing_stats_metrics(st_a, idx)
        hb = framing_stats_metrics(st_b, idx)
        for m in FRAMING_METRICS:
            draws[m].append(None if ha[m] is None or hb[m] is None else ha[m] - hb[m])
    return draws


# --------------------------------------------------------------------------- #
# structural join
# --------------------------------------------------------------------------- #
def load_structural():
    out = {}
    if not STRUCTURAL_CSV.exists():
        return out
    with open(STRUCTURAL_CSV) as fh:
        for row in csv.DictReader(fh):
            key = STRUCTURAL_KEY.get(row["model_name"])
            if key is None:
                continue
            out[key] = {
                "structural_model_name": row["model_name"],
                "seed": row["seed"], "step": int(row["step"]),
                "lambda": float(row["lambda"]), "eta": float(row["eta"]),
                "d": float(row["d"]),
                "spearman_utility": float(row["spearman_utility"]),
                "spearman_alpha": float(row["spearman_alpha"]),
                "spearman_beta": float(row["spearman_beta"]),
                "log10_cond_jacobian": float(row["log10_cond_jacobian"]),
                "on_selection_grid": row["on_selection_grid"] == "True",
                "frozen_selected": row["selected"] == "True",
            }
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface_root",
                    default="results/checkpoint_diagnostics/pilot30k/surface")
    ap.add_argument("--baseline_surface_root", default="results/surface_form_stress")
    ap.add_argument("--framing_root",
                    default="results/checkpoint_diagnostics/pilot30k/framing")
    ap.add_argument("--out_dir", default="results/checkpoint_diagnostics/pilot30k")
    ap.add_argument("--bootstrap_reps", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=20260821)
    ap.add_argument("--allow_missing", action="store_true",
                    help="report MISSING/FAILED models instead of aborting")
    ap.add_argument("--no_figure", action="store_true")
    args = ap.parse_args()

    surface_root = ROOT / args.surface_root
    baseline_root = ROOT / args.baseline_surface_root
    framing_root = ROOT / args.framing_root
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    strict = not args.allow_missing

    subset = json.loads(SUBSET_PATH.read_text())
    cases = {c["case_id"]: c for c in subset["cases"]}
    subset_sha = sha256_file(SUBSET_PATH)
    framing_sha = sha256_file(FRAMING_DATA) if FRAMING_DATA.exists() else None
    if subset_sha != EXPECTED_SUBSET_SHA:
        raise SystemExit(f"Frozen subset SHA mismatch: {subset_sha}")
    if framing_sha and framing_sha != EXPECTED_FRAMING_SHA:
        raise SystemExit(f"Framing benchmark SHA mismatch: {framing_sha}")

    structural = load_structural()

    # ---------------- surface ---------------- #
    surf_rows, surf_prov, surf_units_by_model, surf_results = {}, [], {}, {}
    for key in SURFACE_MODELS:
        rows, prov = load_surface(key, surface_root, baseline_root, strict)
        surf_prov.append(prov)
        if rows is None:
            continue
        surf_rows[key] = rows
        u = surface_units(rows)
        surf_units_by_model[key] = surface_sufficient_stats(u, cases)
        m = surface_metrics(u, cases)
        m.pop("_modal", None)
        m["ownership_keep_both_rate"] = keep_both_rate(u, cases)
        # the bootstrap path must reproduce the direct point estimates exactly
        chk = surface_stats_metrics(surf_units_by_model[key])
        for _k, _v in chk.items():
            assert abs(_v - m[_k]) < 1e-12, f"{key}: {_k} sufficient-stat mismatch"
        m["model"] = key
        m["role"] = SURFACE_MODELS[key][3]
        m["structural"] = structural.get(key)
        surf_results[key] = m

    surf_pair_out = []
    for a, b in SURFACE_PAIRS:
        entry = {"late": a, "reference": b,
                 "cluster_unit": "(case_id, perspective); "
                                 "ownership_keep_both_rate clustered by case_id",
                 "bootstrap_reps": args.bootstrap_reps,
                 "bootstrap_seed": args.bootstrap_seed}
        if a not in surf_units_by_model or b not in surf_units_by_model:
            entry["status"] = "SKIPPED_MISSING_MODEL"
            surf_pair_out.append(entry)
            continue
        draws = surface_paired_bootstrap(surf_units_by_model[a], surf_units_by_model[b],
                                         args.bootstrap_reps, args.bootstrap_seed)
        metrics = {}
        for m in SURFACE_UNIT_METRICS + ["ownership_keep_both_rate"]:
            va, vb = surf_results[a].get(m), surf_results[b].get(m)
            delta = None if va is None or vb is None else va - vb
            lo, hi = ci95(draws[m])
            fl, why = flag(delta, lo, hi)
            metrics[m] = {"late": va, "reference": vb, "paired_delta": delta,
                          "ci95_lower": lo, "ci95_upper": hi,
                          "flagged": fl, "flag_reason": why}
        entry["status"] = "OK"
        entry["metrics"] = metrics
        surf_pair_out.append(entry)

    # ---------------- framing ---------------- #
    fram_prov, fram_scen, fram_results = [], {}, {}
    for key in FRAMING_MODELS:
        rows, prov = load_framing(key, framing_root, strict)
        fram_prov.append(prov)
        if rows is None:
            continue
        by_scen = framing_by_scenario(rows)
        fram_scen[key] = framing_sufficient_stats(by_scen)
        m = framing_metrics(by_scen)
        chk = framing_stats_metrics(fram_scen[key])
        for _k, _v in chk.items():
            _d = m["headline"][_k]
            assert (_v is None and _d is None) or abs(_v - _d) < 1e-9, \
                f"{key}: {_k} sufficient-stat mismatch ({_v} vs {_d})"
        m["model"] = key
        m["role"] = FRAMING_MODELS[key][4]
        m["by_domain"] = framing_domain_metrics(rows)
        m["structural"] = structural.get(key)
        fram_results[key] = m

    fram_pair_out = []
    for a, b in FRAMING_PAIRS:
        entry = {"late": a, "reference": b, "cluster_unit": "scenario_id",
                 "bootstrap_reps": args.bootstrap_reps,
                 "bootstrap_seed": args.bootstrap_seed}
        if a not in fram_scen or b not in fram_scen:
            entry["status"] = "SKIPPED_MISSING_MODEL"
            fram_pair_out.append(entry)
            continue
        draws = framing_paired_bootstrap(fram_scen[a], fram_scen[b],
                                         args.bootstrap_reps, args.bootstrap_seed)
        metrics = {}
        for m in FRAMING_METRICS:
            va = fram_results[a]["headline"][m]
            vb = fram_results[b]["headline"][m]
            delta = None if va is None or vb is None else va - vb
            lo, hi = ci95(draws[m])
            fl, why = flag(delta, lo, hi)
            metrics[m] = {"late": va, "reference": vb, "paired_delta": delta,
                          "ci95_lower": lo, "ci95_upper": hi,
                          "flagged": fl, "flag_reason": why}
        entry["status"] = "OK"
        entry["metrics"] = metrics
        fram_pair_out.append(entry)

    # ---------------- assemble ---------------- #
    any_flag = any(mm["flagged"] for e in surf_pair_out + fram_pair_out
                   if e.get("status") == "OK" for mm in e["metrics"].values())
    summary = {
        "status": "EXPLORATORY_POST_HOC",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "purpose": "Do the late 30,000-step magnitude-GRPO endpoints behave "
                   "differently from the frozen-selected checkpoints on "
                   "surface-form equivalence and adverse framing?",
        "constraints": [
            "Exploratory: cannot and does not change the frozen checkpoint "
            "selections (seed1 -> 2000, seed2 -> 6000).",
            "The 30k adapters are NOT selected models.",
            "No frozen/untouched, neutral-preference, OOD-50, GSM8K, or IFEval "
            "suite was evaluated.",
            "Fitted-utility correlation is a property of an estimated structural "
            "model, not direct evidence about hidden neural representations.",
            "Late-checkpoint structural utilities are near-singular "
            "(log10 cond(J) ~ 17), so individual alpha/utility values there are "
            "weakly identified.",
        ],
        "decision_rule": {
            "description": "Predeclared exploratory rule for whether to run the "
                           "complete checkpoint trajectory.",
            "flag_if": "paired 95% CI excludes zero OR |paired delta| >= 0.05",
            "abs_threshold": FLAG_ABS,
            "not_a_confirmatory_test": True,
        },
        "inputs": {
            "surface_form_subset": {"path": "data/surface_form_stress/surface_form_subset.json",
                                    "sha256": subset_sha},
            "framing_benchmark": {"path": "data/framing_effects_23prob.json",
                                  "sha256": framing_sha},
            "structural_trajectory": {
                "path": "results/training_dynamics/structural_trajectory.csv",
                "sha256": sha256_file(STRUCTURAL_CSV) if STRUCTURAL_CSV.exists() else None},
        },
        "surface_form": {"provenance": surf_prov,
                         "per_model": list(surf_results.values()),
                         "paired_comparisons": surf_pair_out},
        "framing": {"provenance": fram_prov,
                    "per_model": [{k: v for k, v in r.items() if k != "by_domain"}
                                  for r in fram_results.values()],
                    "by_domain": {k: r["by_domain"] for k, r in fram_results.items()},
                    "paired_comparisons": fram_pair_out,
                    "comparator_note": (
                        "No framing predictions exist for the frozen-selected "
                        "seed1@2000 / seed2@6000 checkpoints. The only earlier GRPO "
                        "framing comparator is the EXPLORATORY seed42 step-8000 run, "
                        "so the 30k-vs-earlier framing contrast is NOT a clean "
                        "within-seed checkpoint comparison.")},
        "any_metric_flagged": any_flag,
        "recommendation": (
            "Run the complete checkpoint trajectory: at least one paired "
            "comparison met the predeclared exploratory rule."
            if any_flag else
            "Do not prioritise the complete checkpoint trajectory on this "
            "evidence: no paired comparison met the predeclared exploratory rule."),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    write_csv(out_dir / "summary.csv", surf_results, surf_pair_out,
              fram_results, fram_pair_out)
    (out_dir / "summary.md").write_text(render_md(summary, surf_results, surf_pair_out,
                                                  fram_results, fram_pair_out))
    write_raw_manifest(out_dir, surf_prov, fram_prov, subset_sha, framing_sha)
    if not args.no_figure:
        try:
            make_figure(out_dir, surf_results, surf_pair_out, fram_results, fram_pair_out)
        except Exception as exc:  # figure is a nicety, never a gate
            print(f"WARNING: figure not produced: {exc}")

    print(f"Wrote {out_dir/'summary.json'}")
    print(f"any_metric_flagged = {any_flag}")


def write_csv(path, surf, surf_pairs, fram, fram_pairs):
    rows = []
    for key, m in surf.items():
        st = m.get("structural") or {}
        rows.append({
            "axis": "surface_form", "kind": "level", "model": key, "role": m["role"],
            "metric": "", "value": "", "reference_model": "", "reference_value": "",
            "paired_delta": "", "ci95_lower": "", "ci95_upper": "", "flagged": "",
            "semantic_invariance_rate": m["semantic_invariance_rate"],
            "fidelity_mean": m["fidelity_mean"],
            "worst_form_fidelity": m["worst_form_fidelity"],
            "worst_form": m["worst_form"],
            "mean_pairwise_flip_rate": m["mean_pairwise_flip_rate"],
            "mean_prob_spread_across_forms": m["mean_prob_spread_across_forms"],
            "ownership_keep_both_rate": m["ownership_keep_both_rate"],
            "lambda": st.get("lambda"), "eta": st.get("eta"), "d": st.get("d"),
            "spearman_utility": st.get("spearman_utility"),
            "spearman_alpha": st.get("spearman_alpha"),
            "spearman_beta": st.get("spearman_beta"),
            "log10_cond_jacobian": st.get("log10_cond_jacobian"),
        })
    for key, m in fram.items():
        st = m.get("structural") or {}
        h = m["headline"]
        rows.append({
            "axis": "framing", "kind": "level", "model": key, "role": m["role"],
            "metric": "", "value": "", "reference_model": "", "reference_value": "",
            "paired_delta": "", "ci95_lower": "", "ci95_upper": "", "flagged": "",
            "hard_flip_rate": h["hard_flip_rate"],
            "mean_absolute_probability_gap": h["mean_absolute_probability_gap"],
            "mean_probability_gap_negative_minus_positive":
                h["mean_probability_gap_negative_minus_positive"],
            "classical_flip_rate": h["classical_flip_rate"],
            "probability_monotonicity_violation_rate":
                h["probability_monotonicity_violation_rate"],
            "hard_choice_monotonicity_violation_rate":
                h["hard_choice_monotonicity_violation_rate"],
            "lambda": st.get("lambda"), "eta": st.get("eta"), "d": st.get("d"),
            "spearman_utility": st.get("spearman_utility"),
            "spearman_alpha": st.get("spearman_alpha"),
            "spearman_beta": st.get("spearman_beta"),
            "log10_cond_jacobian": st.get("log10_cond_jacobian"),
        })
    for axis, pairs in (("surface_form", surf_pairs), ("framing", fram_pairs)):
        for e in pairs:
            if e.get("status") != "OK":
                continue
            for name, d in e["metrics"].items():
                rows.append({
                    "axis": axis, "kind": "paired_delta", "model": e["late"],
                    "role": "", "metric": name, "value": d["late"],
                    "reference_model": e["reference"], "reference_value": d["reference"],
                    "paired_delta": d["paired_delta"], "ci95_lower": d["ci95_lower"],
                    "ci95_upper": d["ci95_upper"], "flagged": d["flagged"],
                })
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _f(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"


def render_md(summary, surf, surf_pairs, fram, fram_pairs):
    L = ["# Late-checkpoint (30,000-step) diagnostic pilot",
         "",
         "*EXPLORATORY / POST-HOC. This does **not** change the frozen checkpoint "
         "selections (seed 1 → step 2,000; seed 2 → step 6,000), and the 30k "
         "adapters are **not** selected models. No frozen/untouched, "
         "neutral-preference, OOD-50, GSM8K, or IFEval suite was evaluated.*",
         "",
         f"Git commit: `{summary['git_commit']}` · generated "
         f"{summary['generated_at_utc']}",
         "",
         "## Surface-form stress (96 cases × 2 perspectives × 48 equivalent forms)",
         "",
         "| model | role | invariance | fidelity | worst-form fid | flip rate | "
         "prob spread | keep-both | λ | η | d | ρ(utility) | ρ(α) | ρ(β) | log₁₀cond(J) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, m in surf.items():
        st = m.get("structural") or {}
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            key, m["role"], _f(m["semantic_invariance_rate"]), _f(m["fidelity_mean"]),
            _f(m["worst_form_fidelity"]), _f(m["mean_pairwise_flip_rate"]),
            _f(m["mean_prob_spread_across_forms"]), _f(m["ownership_keep_both_rate"]),
            _f(st.get("lambda")), _f(st.get("eta")), _f(st.get("d")),
            _f(st.get("spearman_utility")), _f(st.get("spearman_alpha")),
            _f(st.get("spearman_beta")), _f(st.get("log10_cond_jacobian"), 1)))
    L += ["", "Read: high invariance **and** high fidelity = an ownership-invariant "
          "rule; high invariance with low fidelity = a constant/shortcut answer; "
          "low invariance = surface-form fragility.", ""]

    for key, m in surf.items():
        L.append(f"### {key} — fidelity by transformation axis")
        for ax, vals in m["fidelity_by_axis"].items():
            L.append(f"- {ax}: " + ", ".join(f"{v}={x:.3f}" for v, x in vals.items()))
        L.append("")

    L += ["## Surface-form paired differences (30k − selected)", "",
          "Cluster bootstrap over `(case_id, perspective)` units "
          "(`ownership_keep_both_rate` over `case_id`); prompt forms are never "
          "treated as independent samples.", ""]
    for e in surf_pairs:
        L.append(f"### {e['late']} − {e['reference']}")
        if e.get("status") != "OK":
            L += [f"*{e['status']}*", ""]
            continue
        L += ["", "| metric | 30k | selected | Δ | 95% CI | flagged |",
              "|---|---|---|---|---|---|"]
        for name, d in e["metrics"].items():
            L.append(f"| {name} | {_f(d['late'])} | {_f(d['reference'])} | "
                     f"{_f(d['paired_delta'])} | [{_f(d['ci95_lower'])}, "
                     f"{_f(d['ci95_upper'])}] | {'**YES**' if d['flagged'] else 'no'} |")
        L.append("")

    L += ["## Adverse framing (120 scenarios × 23 probabilities × 2 frames = 5,520)", "",
          summary["framing"]["comparator_note"], "",
          "| model | role | hard-flip | mean abs prob gap | prob monotonicity viol. | "
          "hard monotonicity viol. | λ | η | d | ρ(utility) | log₁₀cond(J) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, m in fram.items():
        h, st = m["headline"], (m.get("structural") or {})
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            key, m["role"], _f(h["hard_flip_rate"]),
            _f(h["mean_absolute_probability_gap"]),
            _f(h["probability_monotonicity_violation_rate"]),
            _f(h["hard_choice_monotonicity_violation_rate"], 4),
            _f(st.get("lambda")), _f(st.get("eta")), _f(st.get("d")),
            _f(st.get("spearman_utility")), _f(st.get("log10_cond_jacobian"), 1)))
    L += ["", "## Framing paired differences", "",
          "Cluster bootstrap over scenarios.", ""]
    for e in fram_pairs:
        L.append(f"### {e['late']} − {e['reference']}")
        if e.get("status") != "OK":
            L += [f"*{e['status']}*", ""]
            continue
        L += ["", "| metric | 30k | reference | Δ | 95% CI | flagged |",
              "|---|---|---|---|---|---|"]
        for name, d in e["metrics"].items():
            L.append(f"| {name} | {_f(d['late'])} | {_f(d['reference'])} | "
                     f"{_f(d['paired_delta'])} | [{_f(d['ci95_lower'])}, "
                     f"{_f(d['ci95_upper'])}] | {'**YES**' if d['flagged'] else 'no'} |")
        L.append("")

    L += ["## Predeclared exploratory decision rule", "",
          "Flag a potentially meaningful change when the paired 95% interval "
          "excludes zero **or** |Δ| ≥ 0.05. This decides whether to run the "
          "complete checkpoint trajectory; it is **not** a confirmatory "
          "hypothesis test.", "",
          f"**Any metric flagged:** {summary['any_metric_flagged']}", "",
          f"**Recommendation:** {summary['recommendation']}", "",
          "## Caveats", ""]
    for c in summary["constraints"]:
        L.append(f"- {c}")
    L.append("")
    return "\n".join(L)


def write_raw_manifest(out_dir, surf_prov, fram_prov, subset_sha, framing_sha):
    entries = []
    total = 0
    for p in surf_prov + fram_prov:
        if p.get("status") == "MISSING":
            continue
        d = Path(p["result_dir"])
        if not d.is_absolute():
            d = ROOT / d
        for f in sorted(d.glob("*")):
            if not f.is_file():
                continue
            total += f.stat().st_size
            entries.append({"path": _rel(f), "bytes": f.stat().st_size,
                            "sha256": sha256_file(f),
                            "committed": f.name not in
                            ("form_predictions.jsonl", "predictions.json")})
    manifest = {
        "note": "SHA-256 of the pilot30k raw predictions. Raw prediction files are "
                "NOT committed; derived summaries and metadata ARE. A checksum "
                "verifies a file someone already has — it does not make the file "
                "downloadable.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "inputs": {
            "surface_form_subset_sha256": subset_sha,
            "framing_benchmark_sha256": framing_sha,
            "base_model_path": "models/Qwen2.5-7B-Instruct",
        },
        "adapters": {
            path: {"sha256": (sha256_file(ROOT / path / "adapter_model.safetensors")
                              if (ROOT / path / "adapter_model.safetensors").exists()
                              else None)}
            for path in sorted({p["adapter_path"] for p in surf_prov + fram_prov
                                if p.get("adapter_path")})
        },
        "n_files": len(entries), "total_MB": round(total / 1e6, 2),
        "files": entries,
    }
    (out_dir / "raw_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def make_figure(out_dir, surf, surf_pairs, fram, fram_pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = {"Base": "base", "GRPO-qd-seed1-ckpt2000": "s1@2k",
             "GRPO-qd-seed2-ckpt6000": "s2@6k",
             "GRPO-qd-seed1-ckpt30000": "s1@30k",
             "GRPO-qd-seed2-ckpt30000": "s2@30k",
             "Qwen-7B-Base": "base", "Qwen-7B-GRPO-step8000": "seed42@8k"}
    colors = {"base": "#777777", "s1@2k": "#1f77b4", "s2@6k": "#17becf",
              "s1@30k": "#d62728", "s2@30k": "#ff7f0e", "seed42@8k": "#9467bd"}

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))

    # panel 1 — surface-form levels
    ax = axes[0]
    names = [short[k] for k in surf]
    metrics = [("invariance", "semantic_invariance_rate"),
               ("fidelity", "fidelity_mean"),
               ("worst-form fid", "worst_form_fidelity")]
    w = 0.8 / len(metrics)
    for i, (lab, key) in enumerate(metrics):
        vals = [surf[k][key] or 0 for k in surf]
        ax.bar([x + i * w for x in range(len(names))], vals, w, label=lab)
    ax.set_xticks([x + w for x in range(len(names))])
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1); ax.set_title("Surface form: invariance & fidelity", fontsize=10)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)

    # panel 2 — surface-form paired deltas with CIs
    ax = axes[1]
    labels, deltas, los, his, flags = [], [], [], [], []
    for e in surf_pairs:
        if e.get("status") != "OK":
            continue
        for name, d in e["metrics"].items():
            if d["paired_delta"] is None:
                continue
            labels.append(f"{short[e['late']]}−{short[e['reference']]}\n{name[:22]}")
            deltas.append(d["paired_delta"])
            los.append(d["paired_delta"] - (d["ci95_lower"] or d["paired_delta"]))
            his.append((d["ci95_upper"] or d["paired_delta"]) - d["paired_delta"])
            flags.append(d["flagged"])
    y = range(len(labels))
    ax.errorbar(deltas, list(y), xerr=[los, his], fmt="o", ms=4, lw=1,
                color="#333333", ecolor="#999999", capsize=2)
    for i, (dv, fl) in enumerate(zip(deltas, flags)):
        if fl:
            ax.plot([dv], [i], "o", ms=8, mfc="none", mec="#d62728", mew=1.6)
    ax.axvline(0, color="k", lw=.8)
    ax.axvspan(-0.05, 0.05, color="#cccccc", alpha=.35, zorder=0)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Surface form: paired Δ (30k − selected)\n95% CI, clustered", fontsize=10)
    ax.grid(axis="x", alpha=.3)

    # panel 3 — framing levels
    ax = axes[2]
    fnames = [short[k] for k in fram]
    fmetrics = [("hard-flip", "hard_flip_rate"),
                ("|prob gap|", "mean_absolute_probability_gap"),
                ("prob monot. viol.", "probability_monotonicity_violation_rate")]
    w = 0.8 / len(fmetrics)
    for i, (lab, key) in enumerate(fmetrics):
        vals = [fram[k]["headline"][key] or 0 for k in fram]
        ax.bar([x + i * w for x in range(len(fnames))], vals, w, label=lab)
    ax.set_xticks([x + w for x in range(len(fnames))])
    ax.set_xticklabels(fnames, rotation=30, ha="right", fontsize=8)
    ax.set_title("Adverse framing", fontsize=10)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)

    # panel 4 — structural utility Spearman vs step
    ax = axes[3]
    for k, m in surf.items():
        st = m.get("structural")
        if not st:
            continue
        ax.scatter([st["step"]], [st["spearman_utility"]], s=60,
                   color=colors.get(short[k], "#333333"), label=short[k], zorder=3)
    ax.set_xlabel("training step", fontsize=8)
    ax.set_ylabel("Spearman(utility) vs base", fontsize=8)
    ax.set_title("Fitted-utility rank preservation\n(estimated model, not "
                 "representations)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    fig.suptitle("Late 30,000-step magnitude-GRPO endpoints — EXPLORATORY diagnostic "
                 "(frozen selections unchanged)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "pilot30k_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
