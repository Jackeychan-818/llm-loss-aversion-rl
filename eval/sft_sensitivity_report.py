#!/usr/bin/env python3
"""Phase-A analysis for the SFT batch/LR sensitivity experiment.

Computes, per cell:
  * completion-only VALIDATION cross-entropy on test_goods against the frozen
    rational Yes/No target, with pair-clustered (by case_id) uncertainty —
    recovered on CPU from the teacher-forced probabilities the standard
    evaluation already wrote, so it costs no GPU time;
  * training-side diagnostics from trainer_state.json: loss distribution,
    pre-clipping gradient-norm median/P90/P99/max, clipping fraction,
    non-finite counts, learning-rate schedule verification, loss roughness;
  * structural λ, η, d and the direct choice rates from the endpoint evaluation.

Then applies the FROZEN selection rule of §6 exactly: lowest mean endpoint
validation cross-entropy across seeds 1-3 among the large batches, smaller batch
on an exact tie. Behavioural metrics are reported but never allowed to override.

    python eval/sft_sensitivity_report.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "eval"))
import sft_sensitivity_plan as P                       # noqa: E402
from reward_functions import rational_choice           # noqa: E402
from sweep_partition_estimate import choice_from_row, load_rows   # noqa: E402

EVAL_ROOT = ROOT / "sft_sensitivity_eval"
CKPT_ROOT = P.CKPT_ROOT
OUT = P.RESULT_ROOT
DELTA = ROOT / "data" / "deltas" / "delta_qwen_base.json"
CLIP = P.FIXED["max_grad_norm"]


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return math.nan
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def validation_ce(model_name: str, deltas: dict) -> dict:
    """Completion-only CE against the frozen rational target, pair-clustered.

    Uses the SAME target rule as training (reward_functions.rational_choice), so
    the number is comparable across cells and to the training objective.
    """
    d = EVAL_ROOT / model_name
    xp, yp = d / "loss_aversion_X.json", d / "loss_aversion_Y.json"
    if not (xp.is_file() and yp.is_file()):
        return {"available": False, "reason": f"missing raw rows under {d}"}
    xr, yr = load_rows(xp), load_rows(yp)
    common = sorted(set(xr) & set(yr))

    per_case, all_ce, bad = [], [], 0
    for cid in common:
        delta = deltas.get(str(cid))
        if delta is None:
            continue
        pair = []
        for persp, rows in (("X", xr), ("Y", yr)):
            p = rows[cid].get("Yes / No prob")
            if isinstance(p, str):
                p = json.loads(p)
            p_yes, p_no = float(p[0]), float(p[1])
            tot = p_yes + p_no
            if not math.isfinite(tot) or tot <= 0:
                bad += 1
                continue
            target = rational_choice(persp, delta)
            p_t = (p_yes if target == "Yes" else p_no) / tot
            ce = -math.log(max(p_t, 1e-12))
            pair.append(ce)
            all_ce.append(ce)
        if pair:
            per_case.append(sum(pair) / len(pair))

    if not all_ce:
        return {"available": False, "reason": "no usable rows"}
    mean = sum(all_ce) / len(all_ce)
    # cluster-robust: the two perspectives of a case are not independent
    se = (st.pstdev(per_case) / math.sqrt(len(per_case))) if len(per_case) > 1 else math.nan
    return {"available": True, "validation_ce_mean": mean,
            "validation_ce_se_pair_clustered": se,
            "n_prompts": len(all_ce), "n_case_clusters": len(per_case),
            "n_unusable_rows": bad,
            "target_rule": "reward_functions.rational_choice(perspective, delta)"}


def training_diagnostics(cell: P.Cell) -> dict:
    # Sort by the NUMERIC step, not lexicographically: "checkpoint-64" sorts
    # after "checkpoint-188" as a string, which would silently read a truncated
    # history for every large-batch cell. The endpoint adapter written by
    # save_model() carries no trainer_state.json, so the highest-numbered
    # checkpoint is the authoritative full history.
    cands = []
    for c in glob.glob(str(cell.output_dir / "checkpoint-*")):
        tail = Path(c).name.rsplit("-", 1)[-1]
        if tail.isdigit() and (Path(c) / "trainer_state.json").is_file():
            cands.append((int(tail), Path(c) / "trainer_state.json"))
    if not cands:
        return {"available": False, "reason": "no trainer_state.json"}
    last_step, st_path = max(cands)
    if last_step != cell.optimizer_steps:
        return {"available": False,
                "reason": (f"highest trainer_state is step {last_step}, expected "
                           f"{cell.optimizer_steps} — incomplete run")}
    hist = json.loads(st_path.read_text()).get("log_history", [])
    loss = [float(e["loss"]) for e in hist if "loss" in e and e["loss"] is not None]
    gn = [float(e["grad_norm"]) for e in hist
          if e.get("grad_norm") is not None and math.isfinite(float(e["grad_norm"]))]
    lr = [float(e["learning_rate"]) for e in hist if "learning_rate" in e]
    nonfinite = sum(1 for e in hist for k in ("loss", "grad_norm", "learning_rate")
                    if e.get(k) is not None and not math.isfinite(float(e[k])))
    rough = (st.mean(abs(b - a) for a, b in zip(loss, loss[1:])) if len(loss) > 1
             else math.nan)
    return {
        "available": True,
        "source": str(st_path.relative_to(ROOT)),
        "n_logged_updates": len(hist),
        "final_global_step": last_step,
        "expected_optimizer_steps": cell.optimizer_steps,
        "history_complete": last_step == cell.optimizer_steps,
        "loss_first": loss[0] if loss else None,
        "loss_last": loss[-1] if loss else None,
        "loss_median": st.median(loss) if loss else None,
        "loss_p90": pct(loss, 0.90), "loss_max": max(loss) if loss else None,
        # roughness: mean |Δloss| between consecutive OPTIMIZER UPDATES.
        # Secondary/descriptive. It is NOT comparable across batches without
        # care: a batch-64 cell has 64x fewer updates over the same prompts.
        "loss_roughness_mean_abs_step_delta": rough,
        "grad_norm_median": st.median(gn) if gn else None,
        "grad_norm_p90": pct(gn, 0.90), "grad_norm_p99": pct(gn, 0.99),
        "grad_norm_max": max(gn) if gn else None,
        "clipping_fraction_above_0p1": (sum(1 for g in gn if g > CLIP) / len(gn)
                                        if gn else None),
        "n_nonfinite_records": nonfinite,
        "lr_max": max(lr) if lr else None,
        "lr_final": lr[-1] if lr else None,
    }


def structural(model_name: str) -> dict:
    d = EVAL_ROOT / model_name
    csvs = sorted(glob.glob(str(d / "Model_1" / "*NLS_estimation*.csv")))
    if not csvs:
        return {"available": False, "reason": "no NLS CSV"}
    rows = {r["Parameter"]: r for r in csv.DictReader(open(csvs[0]))}
    lam, eta = float(rows["lambda"]["Estimate"]), float(rows["eta"]["Estimate"])
    xr, yr = load_rows(d / "loss_aversion_X.json"), load_rows(d / "loss_aversion_Y.json")
    common = sorted(set(xr) & set(yr))
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
    return {"available": True, "lambda": lam, "eta": eta,
            "lambda_se": float(rows["lambda"]["Std. Err."]),
            "eta_se": float(rows["eta"]["Std. Err."]),
            "d": math.hypot(lam, eta), "consistency": con / n,
            "keep_both": keep / n, "trade_both": trade / n, "n_cases": n,
            "estimator": "Model A NLS, structural link scale T=1"}


def apply_frozen_rule(cells: dict) -> dict:
    """§6 exactly: lowest MEAN endpoint validation CE across seeds among the
    LARGE batches; exact tie at stored precision -> smaller batch."""
    means = {}
    for b in P.LARGE_BATCHES:
        vals = [c["validation"]["validation_ce_mean"] for c in cells.values()
                if c["effective_batch"] == b and c["validation"].get("available")]
        if len(vals) == len(P.SEEDS):
            means[b] = {"mean": sum(vals) / len(vals),
                        "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
                        "per_seed": vals}
    if not means:
        return {"selected": None, "reason": "no large batch has all 3 seeds complete"}
    best = min(m["mean"] for m in means.values())
    tied = sorted(b for b, m in means.items() if m["mean"] == best)
    return {"rule": "lowest mean endpoint validation CE across seeds 1-3; "
                    "exact tie at stored precision -> smaller batch",
            "candidates": means, "tied_at_best": tied,
            "selected_batch": tied[0],
            "tie_break_used": len(tied) > 1,
            "note": "Behavioural and stability metrics are secondary and did not "
                    "enter this selection."}


def main() -> int:
    deltas_raw = json.loads(DELTA.read_text())
    deltas = {k: float(v["mean_delta"]) for k, v in deltas_raw.items()}

    cells: dict[str, dict] = {}
    for cell in P.phase_a_cells():
        cells[cell.name] = {
            "cell": cell.name, "effective_batch": cell.effective_batch,
            "seed": cell.seed, "learning_rate": cell.learning_rate,
            "prompt_exposure": cell.exposure,
            "optimizer_updates": cell.optimizer_steps,
            "training": training_diagnostics(cell),
            "validation": validation_ce(cell.name, deltas),
            "structural": structural(cell.name),
        }

    doc = {
        "title": "SFT batch sensitivity — Phase A (endpoint evaluation only)",
        "protocol": "SFT_BATCH_LR_SENSITIVITY_PROTOCOL.md + Amendment 1",
        "scientific_status": ("EXPLORATORY / POST-HOC optimization ablation, "
                              "3 seeds. Does not revise the frozen matched-SFT "
                              "selection and licenses no SFT-vs-GRPO claim."),
        "evaluation": "test_goods VALIDATION, endpoint exposure 6,016 only; the "
                      "2,048 and 4,096 checkpoints are retained, not evaluated",
        "cells": cells,
        "selection": apply_frozen_rule(cells),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase_a_results.json").write_text(json.dumps(doc, indent=2) + "\n")

    with open(OUT / "phase_a_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "effective_batch", "seed", "prompt_exposure",
                    "optimizer_updates", "validation_ce", "validation_ce_se",
                    "lambda", "eta", "d", "consistency", "keep_both", "trade_both",
                    "grad_norm_median", "grad_norm_p90", "grad_norm_p99",
                    "grad_norm_max", "clipping_fraction", "loss_median",
                    "loss_roughness", "n_nonfinite"])
        for c in cells.values():
            v, t, s = c["validation"], c["training"], c["structural"]
            w.writerow([c["cell"], c["effective_batch"], c["seed"],
                        c["prompt_exposure"], c["optimizer_updates"],
                        v.get("validation_ce_mean"), v.get("validation_ce_se_pair_clustered"),
                        s.get("lambda"), s.get("eta"), s.get("d"), s.get("consistency"),
                        s.get("keep_both"), s.get("trade_both"),
                        t.get("grad_norm_median"), t.get("grad_norm_p90"),
                        t.get("grad_norm_p99"), t.get("grad_norm_max"),
                        t.get("clipping_fraction_above_0p1"), t.get("loss_median"),
                        t.get("loss_roughness_mean_abs_step_delta"),
                        t.get("n_nonfinite_records")])

    ready = sum(1 for c in cells.values() if c["validation"].get("available"))
    print(f"cells with endpoint evaluation: {ready}/{len(cells)}")
    hdr = (f"{'eb':>3} {'seed':>4} {'val CE':>9} {'lambda':>8} {'eta':>8} "
           f"{'consist':>8} {'gn med':>9} {'gn P99':>9} {'clip':>7}")
    print(hdr); print("-" * len(hdr))
    for c in sorted(cells.values(), key=lambda x: (x["effective_batch"], x["seed"])):
        v, t, s = c["validation"], c["training"], c["structural"]
        f = lambda x, w, p: (f"{x:>{w}.{p}f}" if isinstance(x, float) and math.isfinite(x)
                             else f"{'-':>{w}}")
        print(f"{c['effective_batch']:>3} {c['seed']:>4} "
              f"{f(v.get('validation_ce_mean'), 9, 4)} {f(s.get('lambda'), 8, 4)} "
              f"{f(s.get('eta'), 8, 4)} {f(s.get('consistency'), 8, 4)} "
              f"{f(t.get('grad_norm_median'), 9, 4)} {f(t.get('grad_norm_p99'), 9, 2)} "
              f"{f(t.get('clipping_fraction_above_0p1'), 7, 3)}")
    print("\nselection:", json.dumps(doc["selection"].get("selected_batch")))
    print(f"wrote {(OUT / 'phase_a_results.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
