#!/usr/bin/env python3
"""
Per-checkpoint ADAPTER/ARTIFACT CONSISTENCY + COMPLETENESS verification for the
full SFT grid.

Why this exists
---------------
`eval/plot_sft_training_dynamics.py::scan_full_behavioural_grid` decides whether
a grid exists by testing `<root>/<prefix><step>/Model_1` for directory
existence. That is not a completion test: `estimate_qwen_checkpoint.py` creates
`Model_1/` and writes its PNGs BEFORE the NLS CSV lands, so a checkpoint whose
estimation is still running — or died halfway — presents an indistinguishable
`Model_1/` directory. Selecting or plotting off that scan would silently mix
finished and unfinished checkpoints.

This module is the explicit verification the plotting script's
`build_behavioural` asks for before a complete grid may be plotted. It checks
the expected adapter and evaluation artifacts for every grid cell and
HARD-FAILS on anything missing, duplicated, mismatched or partial. The old
wording said this "binds" predictions to the adapter. That was too strong:
the historical evaluator did not write an eval-time manifest, so the verifier
cannot prove retrospectively which adapter produced a prediction file.

Per checkpoint it checks:
  * adapter directory exists at the path the eval job used, and carries a
    LoRA `adapter_config.json` (hashed) + weight file (size + sha256).
  * evaluation output dir resolves under exactly ONE scanned root (duplicate
    evaluations of the same seed/step are a hard failure).
  * both perspectives present: loss_aversion_X.json and _Y.json.
  * both raw prediction files recorded with size + sha256.
  * N == EXPECTED_N paired case_ids, X and Y covering the same case_id set.
  * no parse failures: every row carries a finite [P(Yes), P(No)] pair.
  * NLS CSV exists, is named for structural link scale T=1, and yields finite
    lambda/eta with strictly positive standard errors.
  * behaviour (consistency / keep-both / trade-both) recomputed from raw rows
    with the SAME helpers the frozen selector uses.

It runs read-only. It selects nothing, plots nothing, and never touches OOD-50,
the frozen unused-configuration suite, or the method-comparison suite.

    python eval/verify_sft_grid.py                    # verify + write manifest
    python eval/verify_sft_grid.py --quiet            # manifest only
"""
from __future__ import annotations

import argparse
import csv as _csv
import glob
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from sweep_partition_estimate import load_rows, choice_from_row  # noqa: E402
from pseudo_utility_alignment import (  # noqa: E402
    DEFAULT_DATA_DIR, DEFAULT_UTILITY_FILE, compute_W, load_case_utilities)

# ── frozen expectations ──────────────────────────────────────────────────────
# Mirrors eval/select_checkpoint.py::FROZEN_GRID and the submit script
# train/submit_eval_baseline_ckpt.pbs. Kept as literals so a drift in either
# file surfaces as a verification failure rather than being silently adopted.
GRID_STEPS = tuple(range(2000, 30001, 2000))          # 15 checkpoints
SEEDS = (1, 2)
EVAL_ROOTS = ("baselines", "baseline")
MODEL_PREFIX = "Qwen-7B-SFT-qd-seed{seed}-ckpt"
ADAPTER_TMPL = "checkpoints/sft_qwen_delta_seed{seed}/checkpoint-{step}"
EXPECTED_TREATMENT = "baseline"
EXPECTED_DATA = "data/test_goods.json"
EXPECTED_N = 9890
EXPECTED_LINK_SCALE = 1
MANIFEST = PROJECT_ROOT / "results" / "sft_grid_verification.json"

# Evaluation output roots that must never be reached from here.
FORBIDDEN_ROOTS = ("ood", "frozen_unused", "method_comparison", "semantic")


def sha256_of(path: Path, cap: int | None = None) -> str:
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            if cap is not None and read + len(chunk) > cap:
                h.update(chunk[: cap - read])
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def verify_adapter(seed: int, step: int) -> dict:
    """Identity of the trained weights the evaluation was supposed to load."""
    rel = ADAPTER_TMPL.format(seed=seed, step=step)
    d = PROJECT_ROOT / rel
    out: dict = {"adapter_path": rel, "exists": d.is_dir(), "problems": []}
    if not out["exists"]:
        out["problems"].append(f"adapter directory absent: {rel}")
        return out

    cfg = d / "adapter_config.json"
    if not cfg.is_file():
        out["problems"].append(f"adapter_config.json absent under {rel}")
    else:
        out["adapter_config_sha256"] = sha256_of(cfg)
        try:
            c = json.loads(cfg.read_text())
            out["peft_type"] = c.get("peft_type")
            out["lora_r"] = c.get("r")
            out["lora_alpha"] = c.get("lora_alpha")
            base = c.get("base_model_name_or_path") or ""
            out["base_model_name_or_path"] = base
            if "Qwen2.5-7B-Instruct" not in str(base):
                out["problems"].append(
                    f"adapter base model is {base!r}, expected Qwen2.5-7B-Instruct")
        except Exception as exc:                       # pragma: no cover
            out["problems"].append(f"adapter_config.json unreadable: {exc}")

    weights = sorted(glob.glob(str(d / "adapter_model.*")))
    if not weights:
        out["problems"].append(f"no adapter_model.* weight file under {rel}")
    else:
        w = Path(weights[0])
        out["adapter_weight_file"] = w.name
        out["adapter_weight_bytes"] = w.stat().st_size
        out["adapter_weight_sha256"] = sha256_of(w)
    return out


def verify_evaluation(seed: int, step: int, case_utilities: dict | None = None) -> dict:
    """Locate and validate the evaluation output for one grid cell."""
    model_name = MODEL_PREFIX.format(seed=seed) + str(step)
    out: dict = {"model_name": model_name, "problems": []}

    hits = [r for r in EVAL_ROOTS if (PROJECT_ROOT / r / model_name).is_dir()]
    out["roots_containing_output"] = hits
    if not hits:
        out["problems"].append(
            f"no evaluation output for {model_name} under {list(EVAL_ROOTS)}")
        return out
    if len(hits) > 1:
        # Two roots holding the same seed/step is ambiguous provenance, not a
        # convenience: refuse rather than pick one.
        out["problems"].append(
            f"DUPLICATE evaluation of {model_name} in roots {hits}; "
            "resolve before selection")
        return out

    root = hits[0]
    if any(f in root for f in FORBIDDEN_ROOTS):
        out["problems"].append(f"refusing forbidden evaluation root {root!r}")
        return out
    d = PROJECT_ROOT / root / model_name
    out["eval_root"] = root
    out["eval_dir"] = str(d.relative_to(PROJECT_ROOT))
    out["treatment"] = EXPECTED_TREATMENT
    out["evaluation_data"] = EXPECTED_DATA
    out["dataset_role"] = "VALIDATION (test_goods)"

    # ── both perspectives, N, and parse integrity ────────────────────────────
    xp, yp = d / "loss_aversion_X.json", d / "loss_aversion_Y.json"
    for p in (xp, yp):
        if not p.is_file():
            out["problems"].append(f"missing perspective file {p.name}")
    if out["problems"]:
        return out
    out["raw_prediction_files"] = {
        "X": {"path": str(xp.relative_to(PROJECT_ROOT)),
              "size_bytes": xp.stat().st_size, "sha256": sha256_of(xp)},
        "Y": {"path": str(yp.relative_to(PROJECT_ROOT)),
              "size_bytes": yp.stat().st_size, "sha256": sha256_of(yp)},
    }
    try:
        xr, yr = load_rows(xp), load_rows(yp)
    except Exception as exc:
        out["problems"].append(f"raw rows unreadable: {exc}")
        return out

    out["n_x_rows"], out["n_y_rows"] = len(xr), len(yr)
    common = sorted(set(xr) & set(yr))
    out["n_paired"] = len(common)
    if len(xr) != EXPECTED_N or len(yr) != EXPECTED_N:
        out["problems"].append(
            f"expected {EXPECTED_N} rows per perspective, got X={len(xr)} Y={len(yr)}")
    if len(common) != EXPECTED_N:
        # Distinguish "still writing" from "X and Y disagree about which cases
        # were run" — only the second is a corruption rather than a partial run.
        detail = ("evaluation incomplete" if len(xr) == len(yr) == len(common)
                  else "X and Y cover different case_id sets")
        out["problems"].append(
            f"expected {EXPECTED_N} paired case_ids, got {len(common)} ({detail})")

    bad = 0
    for src in (xr, yr):
        for r in src.values():
            p = r.get("Yes / No prob")
            if isinstance(p, str):
                try:
                    p = json.loads(p)
                except Exception:
                    bad += 1
                    continue
            try:
                a, b = float(p[0]), float(p[1])
            except Exception:
                bad += 1
                continue
            if not (math.isfinite(a) and math.isfinite(b)) or a < 0 or b < 0:
                bad += 1
    out["parse_failures"] = bad
    if bad:
        out["problems"].append(f"{bad} unparseable/non-finite probability rows")

    # ── structural estimate ──────────────────────────────────────────────────
    csvs = sorted(glob.glob(str(d / "Model_1" / "*NLS_estimation*.csv")))
    out["model_1_dir_exists"] = (d / "Model_1").is_dir()
    if not csvs:
        # The exact partial-completion case the directory test cannot see.
        out["problems"].append(
            "no NLS estimation CSV under Model_1/ — estimation incomplete "
            "(a Model_1 directory alone does NOT prove completion)")
        return out
    if len(csvs) > 1:
        out["problems"].append(f"{len(csvs)} NLS CSVs under Model_1/: {csvs}")
        return out

    c = Path(csvs[0])
    out["nls_csv"] = str(c.relative_to(PROJECT_ROOT))
    out["nls_csv_sha256"] = sha256_of(c)
    if f"T{EXPECTED_LINK_SCALE}" not in c.name:
        out["problems"].append(
            f"NLS CSV {c.name!r} does not record structural link scale "
            f"T={EXPECTED_LINK_SCALE}")
    out["estimator"] = f"Model A NLS, structural link scale T={EXPECTED_LINK_SCALE}"
    out["decoding"] = "deterministic greedy (do_sample=False), teacher-forced logprobs"

    rows = list(_csv.DictReader(open(c)))
    try:
        lam, lse = float(rows[0]["Estimate"]), float(rows[0]["Std. Err."])
        eta, ese = float(rows[1]["Estimate"]), float(rows[1]["Std. Err."])
    except Exception as exc:
        out["problems"].append(f"NLS CSV unreadable: {exc}")
        return out
    if rows[0]["Parameter"] != "lambda" or rows[1]["Parameter"] != "eta":
        out["problems"].append(
            f"NLS CSV row order unexpected: {rows[0]['Parameter']}, {rows[1]['Parameter']}")
    out.update({"lambda": lam, "lambda_se": lse, "eta": eta, "eta_se": ese,
                "d": math.hypot(lam, eta)})
    for k, v in (("lambda", lam), ("eta", eta)):
        if not math.isfinite(v):
            out["problems"].append(f"{k} is not finite")
    for k, v in (("lambda_se", lse), ("eta_se", ese)):
        if not math.isfinite(v) or v <= 0:
            out["problems"].append(f"{k}={v} not finite/positive (NLS did not converge)")

    # ── behaviour, via the selector's own helpers ────────────────────────────
    con = keep = trade = 0
    for cid in common:
        a, b = choice_from_row(xr[cid]), choice_from_row(yr[cid])
        if a != b:
            con += 1
        elif a == "No":
            keep += 1
        else:
            trade += 1
    n = len(common) or 1
    out.update({"consistency": con / n, "keep_both_rate": keep / n,
                "trade_both_rate": trade / n, "n_cases": len(common)})

    # ── W: pseudo-utility alignment (descriptive, shared reference table) ────
    if case_utilities is not None:
        try:
            w = compute_W(d, case_utilities)
            out.update({"W": w["W"], "W_n_choices": w["N"],
                        "rational_choice_rate": w["rational_choice_rate"],
                        "W_cases_skipped_missing_utility":
                            w["cases_skipped_missing_utility"]})
        except Exception as exc:
            out["problems"].append(f"W could not be computed: {exc}")
    return out


def verify_grid() -> dict:
    # One shared, positive Qwen-base utility table for every checkpoint, so W is
    # comparable across the grid (see eval/pseudo_utility_alignment.py).
    try:
        case_utilities = load_case_utilities(DEFAULT_DATA_DIR, DEFAULT_UTILITY_FILE)
        w_reference = str(DEFAULT_UTILITY_FILE.relative_to(PROJECT_ROOT))
    except Exception as exc:
        case_utilities, w_reference = None, f"unavailable: {exc}"

    cells, problems = [], []
    for seed in SEEDS:
        for step in GRID_STEPS:
            ad = verify_adapter(seed, step)
            ev = verify_evaluation(seed, step, case_utilities)
            probs = list(ad["problems"]) + list(ev["problems"])
            cell = {"seed": seed, "step": step, "adapter": ad, "evaluation": ev,
                    "status": "VERIFIED" if not probs else "FAILED",
                    "problems": probs}
            for p in probs:
                problems.append(f"seed {seed} step {step}: {p}")
            cells.append(cell)

    verified = [c for c in cells if c["status"] == "VERIFIED"]
    # Duplicate detection across the whole grid, not just per cell.
    seen: dict[tuple[int, int], int] = {}
    for c in verified:
        k = (c["seed"], c["step"])
        seen[k] = seen.get(k, 0) + 1
    dupes = [k for k, v in seen.items() if v > 1]
    if dupes:
        problems.append(f"duplicate verified grid cells: {dupes}")

    n_required = len(SEEDS) * len(GRID_STEPS)
    return {
        "title": "SFT grid verification — adapter/artifact consistency and completeness",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_commit(),
        "protocol": {
            "seeds": list(SEEDS),
            "grid_steps": list(GRID_STEPS),
            "n_required": n_required,
            "evaluation_data": EXPECTED_DATA,
            "dataset_role": (
                "VALIDATION — test_goods informed reward construction and "
                "checkpoint selection; not a final-test estimate"),
            "expected_n_per_checkpoint": EXPECTED_N,
            "treatment": EXPECTED_TREATMENT,
            "estimator": f"Model A NLS, structural link scale T={EXPECTED_LINK_SCALE}",
            "decoding": "deterministic greedy (do_sample=False)",
            "W_reference_utility_table": w_reference,
            "comparator": (
                "matched local base under the SAME plain `baseline` prompt "
                "(baseline/Qwen-7B-Base-Local, lambda=7.637, eta=1.007) — the "
                "grid used --treatment baseline, so the debias-treated base is "
                "NOT the matched comparator"),
            "verification_scope": {
                "adapter_artifact_present_and_hashed": True,
                "raw_prediction_files_present_and_hashed": True,
                "structural_output_present_and_hashed": True,
                "adapter_to_predictions_cryptographically_bound": False,
                "limitation": (
                    "The historical evaluator did not emit an evaluation-time "
                    "manifest binding its resolved adapter path/hash and data/code "
                    "hashes to the prediction files. This verifier establishes "
                    "expected-path consistency and artifact completeness at "
                    "verification time, not cryptographic proof that the expected "
                    "adapter produced those predictions."),
            },
            "suites_not_opened": [
                "data/ood_new_goods*", "data/frozen_unused_test_goods.json",
                "data/method_comparison/"],
        },
        "n_required": n_required,
        "n_verified": len(verified),
        "n_failed": n_required - len(verified),
        "complete": len(verified) == n_required,
        "adapter_and_artifact_consistency_verified": len(verified) == n_required,
        "problems": problems,
        "checkpoints": cells,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default=str(MANIFEST))
    args = ap.parse_args()

    man = verify_grid()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=2, default=str))

    if not args.quiet:
        print(f"\nSFT grid verification — {man['n_verified']}/{man['n_required']} verified")
        print(f"  evaluation data : {man['protocol']['evaluation_data']} "
              f"(N={EXPECTED_N}, VALIDATION)")
        print(f"  estimator       : {man['protocol']['estimator']}\n")
        hdr = (f"{'seed':>4} | {'step':>6} | {'lambda':>9} | {'eta':>9} | "
               f"{'d':>7} | {'consist':>7} | {'N':>5} | status")
        print(hdr); print("-" * len(hdr))
        for c in man["checkpoints"]:
            e = c["evaluation"]
            fmt = lambda k, w, p: (f"{e[k]:>+{w}.{p}f}" if isinstance(e.get(k), float)
                                   else f"{'-':>{w}}")
            print(f"{c['seed']:>4} | {c['step']:>6} | {fmt('lambda',9,4)} | "
                  f"{fmt('eta',9,4)} | {fmt('d',7,4)} | {fmt('consistency',7,4)} | "
                  f"{e.get('n_cases','-'):>5} | {c['status']}")
        if man["problems"]:
            print(f"\nHARD FAILURE: {len(man['problems'])} problem(s)")
            for p in man["problems"][:40]:
                print(f"  - {p}")
            if len(man["problems"]) > 40:
                print(f"  ... and {len(man['problems']) - 40} more")
        print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")

    return 0 if man["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
