#!/usr/bin/env python3
"""Tests for the exploratory 30k-checkpoint diagnostic analyzer (CPU, no GPU).

Run before submitting the GPU jobs:

    python eval/test_pilot30k_analyze.py

Covers: the flag rule, the cluster-bootstrap sufficient statistics (which must
reproduce the direct metric definitions exactly), framing metric agreement with
run_framing_local's own definitions on the real base predictions, and a
full end-to-end pass where the "30k" inputs are copies of an already evaluated
model — in which case every paired delta and CI bound must be exactly zero.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import pilot30k_analyze as P  # noqa: E402
from run_framing_local import (  # noqa: E402
    build_complete_pairs,
    monotonicity_summary,
    pair_metric_summary,
)

FAILURES = []


def check(cond, label):
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        FAILURES.append(label)


# --------------------------------------------------------------------------- #
def test_flag_rule():
    print("test_flag_rule")
    check(P.flag(0.06, -0.10, 0.20)[0] is True, "|delta| >= 0.05 alone flags")
    check(P.flag(0.01, 0.002, 0.02)[0] is True, "CI excluding zero alone flags")
    check(P.flag(0.01, -0.02, 0.04)[0] is False, "small delta with CI over zero does not flag")
    check(P.flag(-0.05, -0.30, 0.20)[0] is True, "negative delta at the threshold flags")
    check(P.flag(0.049, -0.30, 0.20)[0] is False, "just under the threshold does not flag")
    check(P.flag(None, None, None)[0] is False, "missing estimate does not flag")
    check(P.FLAG_ABS == 0.05, "predeclared absolute threshold is 0.05")


def test_simpson_flip():
    print("test_simpson_flip")
    check(P.simpson_flip([1, 1, 1, 1]) == 0.0, "constant answers give zero flip rate")
    check(abs(P.simpson_flip([1, 2]) - 1.0) < 1e-12, "two distinct answers give flip rate 1")
    check(abs(P.simpson_flip([1, 1, 2, 2]) - (8 / 12)) < 1e-12, "balanced split flip rate")
    check(P.simpson_flip([1]) == 0.0, "single observation is degenerate, not an error")


def test_ci95():
    print("test_ci95")
    vals = list(range(101))
    lo, hi = P.ci95(vals)
    check(abs(lo - 2.5) < 1e-9 and abs(hi - 97.5) < 1e-9, "percentile interval endpoints")
    check(P.ci95([]) == [None, None], "empty draws give no interval")


def test_surface_sufficient_stats_match_direct():
    """The bootstrap statistics must reproduce the row-scanning definitions."""
    print("test_surface_sufficient_stats_match_direct")
    subset = json.loads(P.SUBSET_PATH.read_text())
    cases = {c["case_id"]: c for c in subset["cases"]}
    path = ROOT / "results" / "surface_form_stress" / "Base" / "form_predictions.jsonl"
    if not path.exists():
        print("  SKIP  no Base surface predictions available")
        return
    rows = [json.loads(l) for l in open(path) if l.strip()]
    units = P.surface_units(rows)
    direct = P.surface_metrics(units, cases)
    direct["ownership_keep_both_rate"] = P.keep_both_rate(units, cases)
    st = P.surface_sufficient_stats(units, cases)
    fast = P.surface_stats_metrics(st)
    for k, v in fast.items():
        check(abs(v - direct[k]) < 1e-12, f"{k} matches the direct definition")
    check(len(st["unit_keys"]) == 192, "192 (case, perspective) clusters")
    check(st["fid"].shape == (192, 48), "48 equivalent forms per cluster")
    check(len(st["case_keys"]) == 96, "96 case clusters for keep-both")


def test_framing_sufficient_stats_match_upstream():
    """Framing headline metrics must equal run_framing_local's own definitions."""
    print("test_framing_sufficient_stats_match_upstream")
    path = (ROOT / "framing" / "full_qd8k_120x23" / "Qwen-7B-Base" /
            "single_word" / "predictions.json")
    if not path.exists():
        print("  SKIP  no base framing predictions available")
        return
    rows = json.loads(path.read_text())
    pairs = build_complete_pairs(rows)
    up_pair = pair_metric_summary(pairs)
    up_mono = monotonicity_summary(rows)
    st = P.framing_sufficient_stats(P.framing_by_scenario(rows))
    fast = P.framing_stats_metrics(st)
    check(abs(fast["hard_flip_rate"] - up_pair["hard_flip_rate"]) < 1e-12,
          "hard-flip rate matches run_framing_local")
    check(abs(fast["mean_absolute_probability_gap"]
              - up_pair["mean_absolute_probability_gap"]) < 1e-12,
          "mean absolute probability gap matches run_framing_local")
    check(abs(fast["probability_monotonicity_violation_rate"]
              - up_mono["probability_violation_rate"]) < 1e-12,
          "probability monotonicity violation rate matches run_framing_local")
    check(abs(fast["hard_choice_monotonicity_violation_rate"]
              - up_mono["hard_choice_violation_rate"]) < 1e-12,
          "hard-choice monotonicity violation rate matches run_framing_local")
    check(len(st["scen_keys"]) == 120, "120 scenario clusters")


def test_integrity_rejects_wrong_hash(tmp: Path):
    """A wrong adapter hash or short file must FAIL, never be silently accepted."""
    print("test_integrity_rejects_wrong_hash")
    src = ROOT / "results" / "surface_form_stress" / "GRPO-qd-seed1-ckpt2000"
    if not src.exists():
        print("  SKIP  no selected-checkpoint surface predictions available")
        return
    root = tmp / "surface_bad"
    dst = root / "GRPO-qd-seed1-ckpt30000"
    dst.mkdir(parents=True)
    shutil.copy(src / "form_predictions.jsonl", dst / "form_predictions.jsonl")
    meta = json.loads((src / "run_metadata.json").read_text())
    meta["adapter_path"] = "checkpoints/grpo_qwen_delta_seed1/checkpoint-30000"
    (dst / "run_metadata.json").write_text(json.dumps(meta))  # WRONG adapter sha
    _, prov = P.load_surface("GRPO-qd-seed1-ckpt30000", root, root, strict=False)
    check(prov["status"] == "FAILED_INTEGRITY", "wrong adapter hash fails integrity")
    check(any("adapter sha" in p for p in prov["integrity_problems"]),
          "the adapter hash mismatch is named")

    # truncated predictions must also fail
    lines = (src / "form_predictions.jsonl").read_text().splitlines()[:100]
    (dst / "form_predictions.jsonl").write_text("\n".join(lines) + "\n")
    _, prov = P.load_surface("GRPO-qd-seed1-ckpt30000", root, root, strict=False)
    check(any("row count" in p for p in prov["integrity_problems"]),
          "a short prediction file fails the row-count check")


def test_partial_file_is_skipped_not_summarised(tmp: Path):
    """A half-written prediction file (e.g. a job still running) must be
    reported as failing integrity and EXCLUDED, never summarised."""
    print("test_partial_file_is_skipped_not_summarised")
    s_src = ROOT / "results" / "surface_form_stress"
    f_src = ROOT / "framing" / "full_qd8k_120x23"
    if not (s_src / "GRPO-qd-seed1-ckpt2000").exists() or not f_src.exists():
        print("  SKIP  prerequisite predictions unavailable")
        return
    surface_root = tmp / "partial_surface"
    framing_root = tmp / "partial_framing"
    out_dir = tmp / "partial_out"
    for late, ref in (("GRPO-qd-seed1-ckpt30000", "GRPO-qd-seed1-ckpt2000"),
                      ("GRPO-qd-seed2-ckpt30000", "GRPO-qd-seed2-ckpt6000")):
        d = surface_root / late
        d.mkdir(parents=True)
        lines = (s_src / ref / "form_predictions.jsonl").read_text().splitlines()[:4000]
        (d / "form_predictions.jsonl").write_text("\n".join(lines) + "\n")
        shutil.copy(s_src / ref / "run_metadata.json", d / "run_metadata.json")
    for late in ("GRPO-qd-seed1-ckpt30000", "GRPO-qd-seed2-ckpt30000"):
        d = framing_root / late / "single_word"
        d.mkdir(parents=True)
        rows = json.loads((f_src / "Qwen-7B-Base" / "single_word"
                           / "predictions.json").read_text())[:1200]
        (d / "predictions.json").write_text(json.dumps(rows))
        shutil.copy(f_src / "Qwen-7B-Base" / "single_word" / "manifest.json",
                    d / "manifest.json")

    saved_s, saved_f = dict(P.SURFACE_MODELS), dict(P.FRAMING_MODELS)
    try:
        for k in ("GRPO-qd-seed1-ckpt30000", "GRPO-qd-seed2-ckpt30000"):
            kind, adapter, _sha, role = P.SURFACE_MODELS[k]
            P.SURFACE_MODELS[k] = (kind, adapter, None, role)
            kind, sub, adapter, _sha, role = P.FRAMING_MODELS[k]
            P.FRAMING_MODELS[k] = (kind, sub, adapter, None, role)
        sys.argv = ["pilot30k_analyze",
                    "--surface_root", str(surface_root),
                    "--baseline_surface_root", str(s_src),
                    "--framing_root", str(framing_root),
                    "--out_dir", str(out_dir),
                    "--bootstrap_reps", "20", "--allow_missing", "--no_figure"]
        P.main()   # must not raise
    finally:
        P.SURFACE_MODELS.clear(); P.SURFACE_MODELS.update(saved_s)
        P.FRAMING_MODELS.clear(); P.FRAMING_MODELS.update(saved_f)

    summary = json.loads((out_dir / "summary.json").read_text())
    analysed = {m["model"] for m in summary["surface_form"]["per_model"]}
    check("GRPO-qd-seed1-ckpt30000" not in analysed,
          "a partial surface file is excluded from the per-model summary")
    statuses = {p["model"]: p["status"] for p in summary["surface_form"]["provenance"]}
    check(statuses["GRPO-qd-seed1-ckpt30000"] == "FAILED_INTEGRITY",
          "the partial surface file is reported as failing integrity")
    check(all(e["status"] == "SKIPPED_MISSING_OR_FAILED_MODEL"
              for e in summary["surface_form"]["paired_comparisons"]),
          "no surface paired comparison is computed from a partial file")
    check(all(e["status"] == "SKIPPED_MISSING_OR_FAILED_MODEL"
              for e in summary["framing"]["paired_comparisons"]),
          "no framing paired comparison is computed from a partial file")


def test_end_to_end_self_comparison(tmp: Path):
    """Copy an evaluated model in as the '30k' input: every paired delta must be
    exactly zero, which exercises the whole pipeline without a GPU."""
    print("test_end_to_end_self_comparison")
    s_src = ROOT / "results" / "surface_form_stress"
    f_src = ROOT / "framing" / "full_qd8k_120x23"
    if not (s_src / "GRPO-qd-seed1-ckpt2000").exists() or not f_src.exists():
        print("  SKIP  prerequisite predictions unavailable")
        return

    surface_root = tmp / "surface"
    framing_root = tmp / "framing"
    out_dir = tmp / "out"
    for late, ref in (("GRPO-qd-seed1-ckpt30000", "GRPO-qd-seed1-ckpt2000"),
                      ("GRPO-qd-seed2-ckpt30000", "GRPO-qd-seed2-ckpt6000")):
        d = surface_root / late
        d.mkdir(parents=True)
        shutil.copy(s_src / ref / "form_predictions.jsonl", d / "form_predictions.jsonl")
        shutil.copy(s_src / ref / "run_metadata.json", d / "run_metadata.json")
    for late in ("GRPO-qd-seed1-ckpt30000", "GRPO-qd-seed2-ckpt30000"):
        d = framing_root / late / "single_word"
        d.mkdir(parents=True)
        for f in ("predictions.json", "manifest.json"):
            shutil.copy(f_src / "Qwen-7B-Base" / "single_word" / f, d / f)

    # relax only the checkpoint-identity expectations, since these are copies
    saved_s = dict(P.SURFACE_MODELS)
    saved_f = dict(P.FRAMING_MODELS)
    try:
        for k in ("GRPO-qd-seed1-ckpt30000", "GRPO-qd-seed2-ckpt30000"):
            kind, adapter, _sha, role = P.SURFACE_MODELS[k]
            P.SURFACE_MODELS[k] = (kind, adapter, None, role)
            kind, sub, adapter, _sha, role = P.FRAMING_MODELS[k]
            P.FRAMING_MODELS[k] = (kind, sub, adapter, None, role)
        sys.argv = ["pilot30k_analyze",
                    "--surface_root", str(surface_root),
                    "--baseline_surface_root", str(s_src),
                    "--framing_root", str(framing_root),
                    "--out_dir", str(out_dir),
                    "--bootstrap_reps", "50"]
        P.main()
    finally:
        P.SURFACE_MODELS.clear(); P.SURFACE_MODELS.update(saved_s)
        P.FRAMING_MODELS.clear(); P.FRAMING_MODELS.update(saved_f)

    summary = json.loads((out_dir / "summary.json").read_text())
    for f in ("summary.json", "summary.csv", "summary.md",
              "raw_manifest.json", "pilot30k_comparison.png"):
        check((out_dir / f).exists(), f"{f} written")

    zero_ok = True
    n_checked = 0
    for e in summary["surface_form"]["paired_comparisons"]:
        if e["status"] != "OK":
            zero_ok = False
            continue
        for name, d in e["metrics"].items():
            n_checked += 1
            if abs(d["paired_delta"]) > 1e-12 or abs(d["ci95_lower"]) > 1e-12 \
               or abs(d["ci95_upper"]) > 1e-12 or d["flagged"]:
                zero_ok = False
    check(n_checked == 12, "both surface pairs produced 6 metrics each")
    check(zero_ok, "self-comparison gives exactly zero surface deltas and CIs")

    fz, fn = True, 0
    for e in summary["framing"]["paired_comparisons"]:
        if e["status"] != "OK":
            fz = False
            continue
        if e["reference"] != "Qwen-7B-Base":
            continue   # base-vs-seed42 is a genuine, nonzero contrast
        for name, d in e["metrics"].items():
            fn += 1
            if abs(d["paired_delta"]) > 1e-12 or d["flagged"]:
                fz = False
    check(fn == 12, "both framing-vs-base pairs produced 6 metrics each")
    check(fz, "self-comparison gives exactly zero framing deltas")
    check(summary["any_metric_flagged"] is True,
          "the genuine seed42-vs-copy contrast still flags, so the rule is live")
    check(summary["status"] == "EXPLORATORY_POST_HOC", "status is exploratory/post-hoc")

    manifest = json.loads((out_dir / "raw_manifest.json").read_text())
    committed = {Path(f["path"]).name for f in manifest["files"] if f["committed"]}
    check("form_predictions.jsonl" not in committed and "predictions.json" not in committed,
          "raw prediction files are marked NOT committed")


def test_pbs_scripts_are_valid():
    print("test_pbs_scripts_are_valid")
    for name in ("submit_eval_pilot30k_surface.pbs", "submit_eval_pilot30k_framing.pbs"):
        p = ROOT / "train" / name
        check(p.exists(), f"{name} exists")
        if not p.exists():
            continue
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True)
        check(r.returncode == 0, f"{name} parses as bash")
        text = p.read_text()
        # The prohibited suites may appear only inside a refusal guard, never as
        # a DATA_FILE / OUTPUT_ROOT assignment.
        for line in text.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("DATA_FILE=")
                    or stripped.startswith("OUTPUT_ROOT=")
                    or stripped.startswith("SUBSET=")):
                continue
            for banned in ("method_comparison", "semantic_counterbalancing",
                           "frozen_unused", "neutral", "ood", "gsm8k", "ifeval"):
                check(banned not in stripped.lower(),
                      f"{name}: '{banned}' is not an input path ({stripped[:40]})")
        check("results/checkpoint_diagnostics/pilot30k" in text,
              f"{name} writes under the dedicated pilot30k output root")
        check("checkpoint-30000" in text, f"{name} targets the 30,000-step adapters")
        check("adapter_model.safetensors" in text,
              f"{name} verifies the adapter file before running")


def main():
    print("=" * 72)
    print("pilot30k analyzer tests")
    print("=" * 72)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_flag_rule()
        test_simpson_flip()
        test_ci95()
        test_surface_sufficient_stats_match_direct()
        test_framing_sufficient_stats_match_upstream()
        test_integrity_rejects_wrong_hash(tmp)
        test_pbs_scripts_are_valid()
        test_partial_file_is_skipped_not_summarised(tmp)
        test_end_to_end_self_comparison(tmp)
    print("=" * 72)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
