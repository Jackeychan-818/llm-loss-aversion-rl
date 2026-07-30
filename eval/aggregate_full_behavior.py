#!/usr/bin/env python3
"""Deterministic full-behavior aggregation (Task 5).

Reads ONLY existing committed results/predictions (no model inference, no
checkpoint or frozen-suite evaluation) and produces one machine-readable table
of the full behavioral picture per model/seed/step, then renders Markdown and
LaTeX FROM that table (no hand-entered numbers).

Reproducibility model: a tracked canonical snapshot
(`results/full_behavior/full_behavior_snapshot.json`) holds every row's primitive
statistics + provenance (paths/hashes). All JSON/CSV/MD/TEX outputs render from
that snapshot, so `--check` passes from a clean `git archive`. NSCC directory
discovery (which reads untracked prediction dirs) is behind `--refresh` only.

`--refresh` auto-discovers every `<root>/<model>/Model_1/*_NLS_estimation_T1(Model
A).csv` under `baseline/`. Models without a committed NLS CSV (e.g. the
un-evaluated full SFT grid) are simply absent — never fabricated.

Guard: a small lambda is NOT reported as success when eta, inconsistency, parse
failures, or choice collapse contradict it; each row carries an explicit
`caveats` list and a boolean `clean_reduction`.

    python3 eval/aggregate_full_behavior.py            # render outputs from snapshot
    python3 eval/aggregate_full_behavior.py --check    # verify byte-identical (clean clone)
    python3 eval/aggregate_full_behavior.py --refresh  # re-read NSCC dirs, rewrite snapshot
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_ROOT = PROJECT_ROOT / "baseline"
DELTA_FILE = PROJECT_ROOT / "data" / "deltas" / "delta_qwen_base.json"
W_TRAJ = PROJECT_ROOT / "results" / "pseudo_utility_alignment_trajectory.json"
OUT_DIR = PROJECT_ROOT / "results" / "full_behavior"
OUT_JSON = OUT_DIR / "full_behavior.json"
OUT_CSV = OUT_DIR / "full_behavior.csv"
OUT_MD = OUT_DIR / "full_behavior.md"
OUT_TEX = OUT_DIR / "full_behavior.tex"
# Canonical TRACKED input snapshot. All rendered outputs derive from this; it is
# the only artifact that must be committed for a clean-clone `--check`. NSCC
# directory discovery is behind `--refresh` and rewrites this snapshot.
SNAPSHOT = OUT_DIR / "full_behavior_snapshot.json"

# Success-guard thresholds (documented; conservative).
ETA_MAX = 1.0            # |eta| above this is a status-quo-bias red flag
CONS_MIN = 0.50          # consistency floor (matches selector eligibility)
PARSE_FAIL_MAX = 0.01    # >1% unparsable rows is a red flag
COLLAPSE_MIN = 0.02      # keep_both or trade_both < 2% => a degenerate/collapsed side


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def parse_model_id(model_name: str):
    """(seed, step) from a model dir name. seed 'base' for the matched base."""
    if "Base-Local" in model_name:
        return "base", 0
    m = re.search(r"seed(\d+)-ckpt(\d+)", model_name)
    if m:
        return f"seed{m.group(1)}", int(m.group(2))
    m = re.search(r"ckpt(\d+)", model_name)
    if m:
        return "exploratory", int(m.group(1))
    return "other", -1


def read_nls(csv_path: Path):
    lam = eta = lam_se = eta_se = None
    variances = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            name = row["Parameter"].strip()
            est = float(row["Estimate"]); se = float(row["Std. Err."])
            var = float(row.get("Variance", "nan") or "nan")
            if math.isfinite(var):
                variances.append(var)
            if name == "lambda":
                lam, lam_se = est, se
            elif name == "eta":
                eta, eta_se = est, se
    # variance-ratio conditioning proxy (not the true Jacobian condition number)
    pos = [v for v in variances if v > 0]
    cond_proxy = (max(pos) / min(pos)) if pos else float("nan")
    return lam, eta, lam_se, eta_se, cond_proxy


def choice(entry):
    """argmax choice 'Yes'/'No' from teacher-forced probs, or '' if unparsable."""
    if "Yes / No prob" not in entry:
        return ""
    try:
        yes, no = entry["Yes / No prob"]
    except (ValueError, TypeError):
        return ""
    return "Yes" if float(yes) > float(no) else "No"


def rational(perspective: str, delta: float) -> str:
    if delta == 0.0:
        return ""
    if perspective == "X":
        return "No" if delta > 0 else "Yes"
    return "Yes" if delta > 0 else "No"


def behavior(model_dir: Path, deltas: dict):
    xp = model_dir / "Model_1" / "loss_aversion_X_for_Model_A.json"
    yp = model_dir / "Model_1" / "loss_aversion_Y_for_Model_A.json"
    if not (xp.exists() and yp.exists()):
        return None
    with open(xp) as fh:
        xs = {int(e["case_id"]): e for e in json.load(fh)}
    with open(yp) as fh:
        ys = {int(e["case_id"]): e for e in json.load(fh)}
    ids = sorted(set(xs) & set(ys))
    n = keep = trade = consistent = parse_fail = 0
    tgt_ok = tgt_tot = 0
    for cid in ids:
        cx, cy = choice(xs[cid]), choice(ys[cid])
        if cx == "" or cy == "":
            parse_fail += 1
            continue
        n += 1
        if cx == "No" and cy == "No":
            keep += 1
        elif cx == "Yes" and cy == "Yes":
            trade += 1
        else:
            consistent += 1
        d = deltas.get(str(cid))
        if d is not None:
            dv = float(d["mean_delta"])
            for persp, ch in (("X", cx), ("Y", cy)):
                r = rational(persp, dv)
                if r:
                    tgt_tot += 1
                    tgt_ok += (ch == r)
    if n == 0:
        return None
    return {
        "n": n,
        "consistency": consistent / n,
        "keep_both_rate": keep / n,
        "trade_both_rate": trade / n,
        "hard_choice_yes_rate": sum(1 for cid in ids if choice(xs[cid]) == "Yes") / max(len(ids), 1),
        "parse_failure_rate": parse_fail / len(ids),
        "target_agreement": (tgt_ok / tgt_tot) if tgt_tot else None,
        "x_sha256": sha256_file(xp),
        "y_sha256": sha256_file(yp),
    }


def caveats_for(lam, eta, beh) -> list[str]:
    c = []
    if abs(eta) > ETA_MAX:
        c.append(f"|eta|={abs(eta):.2f}>{ETA_MAX}: status-quo bias remains")
    if beh["consistency"] < CONS_MIN:
        c.append(f"consistency={beh['consistency']:.2f}<{CONS_MIN}")
    if beh["parse_failure_rate"] > PARSE_FAIL_MAX:
        c.append(f"parse_failures={beh['parse_failure_rate']:.3f}")
    if beh["keep_both_rate"] < COLLAPSE_MIN or beh["trade_both_rate"] < COLLAPSE_MIN:
        c.append("choice collapse (a keep/trade side <2%)")
    return c


def discover_rows():
    """Read the (untracked) NSCC baseline directories and compute every row's
    primitive statistics + provenance. Called ONLY by --refresh; the rendered
    outputs never touch NSCC directories."""
    with open(DELTA_FILE) as fh:
        deltas = json.load(fh)
    w_by_key = {}
    if W_TRAJ.exists():
        for r in json.load(open(W_TRAJ)).get("rows", []):
            w_by_key[(str(r.get("seed")), int(r.get("step", -1)))] = r.get("W")

    rows = []
    for csv_path in sorted(BASELINE_ROOT.glob("*/Model_1/*NLS_estimation_T1(Model A).csv")):
        model_dir = csv_path.parent.parent
        model_name = model_dir.name
        seed, step = parse_model_id(model_name)
        lam, eta, lam_se, eta_se, cond_proxy = read_nls(csv_path)
        if lam is None or eta is None:
            continue
        beh = behavior(model_dir, deltas)
        if beh is None:
            continue
        d = math.sqrt(lam * lam + eta * eta)
        cav = caveats_for(lam, eta, beh)
        rows.append({
            "model_name": model_name,
            "seed": seed,
            "step": step,
            "lambda": lam,
            "lambda_se": lam_se,
            "eta": eta,
            "eta_se": eta_se,
            "d": d,
            "consistency": beh["consistency"],
            "keep_both_rate": beh["keep_both_rate"],
            "trade_both_rate": beh["trade_both_rate"],
            "hard_choice_yes_rate": beh["hard_choice_yes_rate"],
            "target_agreement": beh["target_agreement"],
            "parse_failure_rate": beh["parse_failure_rate"],
            "W_pseudo_utility": w_by_key.get((seed, step)),
            "n": beh["n"],
            "estimator_cond_proxy": cond_proxy,
            "clean_reduction": (abs(lam) < 0.5 and not cav),
            "caveats": cav,
            "provenance": {
                "nls_csv": str(csv_path.relative_to(PROJECT_ROOT)),
                "x_sha256": beh["x_sha256"],
                "y_sha256": beh["y_sha256"],
            },
        })
    rows.sort(key=lambda r: (r["seed"], r["step"]))
    return rows


def render_md(rows) -> str:
    hdr = ("| model | seed | step | λ (SE) | η (SE) | d | cons | keep | trade | "
           "tgt | W | clean | caveats |")
    sep = "|" + "---|" * 13
    lines = [
        "# Full-behavior aggregation (Task 5)",
        "",
        "*Rendered by `eval/aggregate_full_behavior.py` from the tracked "
        "`full_behavior_snapshot.json` (NSCC discovery is behind `--refresh`). "
        "`test_goods` = validation. A small λ is not called success when caveats fire.*",
        "",
        hdr, sep,
    ]
    for r in rows:
        w = f"{r['W_pseudo_utility']:.3f}" if r["W_pseudo_utility"] is not None else "—"
        tgt = f"{r['target_agreement']:.3f}" if r["target_agreement"] is not None else "—"
        lines.append(
            f"| {r['model_name']} | {r['seed']} | {r['step']} | "
            f"{r['lambda']:.3f} ({r['lambda_se']:.3f}) | {r['eta']:.3f} ({r['eta_se']:.3f}) | "
            f"{r['d']:.3f} | {r['consistency']:.3f} | {r['keep_both_rate']:.3f} | "
            f"{r['trade_both_rate']:.3f} | {tgt} | {w} | "
            f"{'yes' if r['clean_reduction'] else 'NO'} | "
            f"{'; '.join(r['caveats']) if r['caveats'] else '—'} |")
    n_total = len(rows)
    n_clean = sum(1 for r in rows if r["clean_reduction"])
    n_small_caveat = sum(1 for r in rows if abs(r["lambda"]) < 0.5 and r["caveats"])
    lines += ["",
              f"Rows: {n_total}. **{n_total - n_clean}/{n_total} rows are "
              f"`clean=NO`.** Of those, **{n_small_caveat}** have `|λ|<0.5` *plus* a "
              "contradictory caveat (η, inconsistency, parse failures, or choice "
              "collapse) — the direct evidence that λ alone can mislead; the rest "
              "are `clean=NO` only because `|λ|≥0.5`.", ""]
    return "\n".join(lines)


def render_tex(rows) -> str:
    out = [r"% Generated by eval/aggregate_full_behavior.py -- do not hand-edit.",
           r"\begin{tabular}{llrrrrrrr}", r"\toprule",
           r"model & seed & step & $\lambda$ & $\eta$ & $d$ & cons & tgt & clean \\",
           r"\midrule"]
    for r in rows:
        tgt = f"{r['target_agreement']:.3f}" if r["target_agreement"] is not None else "--"
        out.append(f"{r['model_name'].replace('_','-')} & {r['seed']} & {r['step']} & "
                   f"{r['lambda']:.3f} & {r['eta']:.3f} & {r['d']:.3f} & "
                   f"{r['consistency']:.3f} & {tgt} & "
                   f"{'yes' if r['clean_reduction'] else 'no'} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def render_csv(rows) -> str:
    import io
    cols = ["model_name", "seed", "step", "lambda", "lambda_se", "eta", "eta_se", "d",
            "consistency", "keep_both_rate", "trade_both_rate", "hard_choice_yes_rate",
            "target_agreement", "parse_failure_rate", "W_pseudo_utility", "n",
            "estimator_cond_proxy", "clean_reduction"]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow([r[c] for c in cols])
    return buf.getvalue()


def load_snapshot() -> dict:
    if not SNAPSHOT.exists():
        raise SystemExit(f"Snapshot missing: {SNAPSHOT.relative_to(PROJECT_ROOT)}. "
                         "Run with --refresh on NSCC to regenerate it from the "
                         "baseline directories.")
    with open(SNAPSHOT) as fh:
        return json.load(fh)


def build_snapshot() -> dict:
    """Refresh the canonical snapshot by discovering NSCC baseline directories.
    This is the ONLY function that reads the (untracked) prediction directories.
    Stores primitive statistics + provenance (paths/hashes) so all rendered
    outputs — and the clean-clone `--check` — derive solely from the snapshot."""
    rows = discover_rows()
    return {
        "schema_version": 1,
        "role": "canonical tracked input snapshot for full-behavior rendering",
        "generated_by": "aggregate_full_behavior.py --refresh (reads NSCC baseline dirs)",
        "delta_file": str(DELTA_FILE.relative_to(PROJECT_ROOT)),
        "delta_file_sha256": sha256_file(DELTA_FILE),
        "n_rows": len(rows),
        "rows": rows,
    }


def render_doc(snapshot: dict) -> dict:
    """Pure function of the snapshot — no filesystem discovery."""
    rows = snapshot["rows"]
    n_total = len(rows)
    n_clean = sum(1 for r in rows if r["clean_reduction"])
    n_not_clean = n_total - n_clean
    n_small = sum(1 for r in rows if abs(r["lambda"]) < 0.5 and r["caveats"])
    n_large = sum(1 for r in rows if abs(r["lambda"]) >= 0.5)
    return {
        "purpose": "Deterministic full-behavior aggregation rendered from the tracked snapshot (Task 5).",
        "note": ("test_goods is validation. estimator_cond_proxy is max/min parameter "
                 "variance (a conditioning proxy, NOT the Jacobian condition number). "
                 "clean_reduction requires |lambda|<0.5 AND no caveat. Rendered from "
                 "full_behavior_snapshot.json; NSCC discovery is behind --refresh."),
        "delta_file_sha256": snapshot["delta_file_sha256"],
        "n_rows": n_total,
        "summary": {
            "n_clean_reduction": n_clean,
            "n_clean_no": n_not_clean,
            "n_small_lambda_with_caveat": n_small,
            "n_large_lambda_ge_0_5": n_large,
            "interpretation": (
                f"{n_not_clean}/{n_total} rows are clean=NO. Of those, {n_small} have "
                f"|lambda|<0.5 BUT a contradictory caveat (eta, inconsistency, parse "
                f"failures, or choice collapse) — the direct evidence that lambda alone "
                f"can mislead — while the remaining {n_not_clean - n_small} are clean=NO "
                f"simply because |lambda|>=0.5."
            ),
        },
        "rows": rows,
    }


def render_outputs(snapshot: dict) -> dict:
    rows = snapshot["rows"]
    doc = render_doc(snapshot)
    return {
        OUT_JSON: json.dumps(doc, indent=2) + "\n",
        OUT_CSV: render_csv(rows),
        OUT_MD: render_md(rows),
        OUT_TEX: render_tex(rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Render from the tracked snapshot and verify byte-identity "
                         "(works from a clean git archive).")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-discover NSCC baseline directories and rewrite the "
                         "snapshot (the only mode that reads untracked predictions).")
    args = ap.parse_args()

    if args.refresh:
        snapshot = build_snapshot()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(snapshot, indent=2) + "\n")
        print(f"Refreshed {SNAPSHOT.relative_to(PROJECT_ROOT)} "
              f"({snapshot['n_rows']} rows from NSCC directories)")
    else:
        snapshot = load_snapshot()

    outputs = render_outputs(snapshot)
    if args.check:
        bad = [p.name for p, s in outputs.items() if not p.exists() or p.read_text() != s]
        if not SNAPSHOT.exists():
            bad.append(SNAPSHOT.name)
        if bad:
            raise SystemExit("CHECK FAILED: drifted/missing " + ", ".join(bad))
        print("CHECK PASSED: snapshot present and all rendered outputs byte-identical.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, s in outputs.items():
        p.write_text(s)
        print(f"Wrote {p.relative_to(PROJECT_ROOT)}")
    rows = snapshot["rows"]
    clean = sum(1 for r in rows if r["clean_reduction"])
    print(f"{len(rows)} rows; {clean} flagged clean_reduction; "
          f"{len(rows) - clean} carry caveats.")


if __name__ == "__main__":
    main()
