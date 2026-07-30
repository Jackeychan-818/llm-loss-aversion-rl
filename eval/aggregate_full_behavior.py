#!/usr/bin/env python3
"""Deterministic full-behavior aggregation (Task 5).

Reads ONLY existing committed results/predictions (no model inference, no
checkpoint or frozen-suite evaluation) and produces one machine-readable table
of the full behavioral picture per model/seed/step, then renders Markdown and
LaTeX FROM that table (no hand-entered numbers).

Auto-discovers every `<root>/<model>/Model_1/*_NLS_estimation_T1(Model A).csv`
under `baseline/`. Models without a committed NLS CSV (e.g. the un-evaluated full
SFT grid) are simply absent — never fabricated.

Guard: a small lambda is NOT reported as success when eta, inconsistency, parse
failures, or choice collapse contradict it; each row carries an explicit
`caveats` list and a boolean `clean_reduction`.

    python3 eval/aggregate_full_behavior.py            # writes JSON+CSV+MD+TEX
    python3 eval/aggregate_full_behavior.py --check    # verify byte-identical
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


def build_rows():
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
        "*Generated by `eval/aggregate_full_behavior.py` from committed results only. "
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
    lines += ["", f"Rows: {len(rows)}. `clean=NO` means λ small but a caveat fired "
              "(η, inconsistency, parse failures, or choice collapse).", ""]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    rows = build_rows()
    doc = {
        "purpose": "Deterministic full-behavior aggregation from committed results (Task 5).",
        "note": ("test_goods is validation. estimator_cond_proxy is max/min parameter "
                 "variance (a conditioning proxy, NOT the Jacobian condition number). "
                 "clean_reduction requires |lambda|<0.5 AND no caveat."),
        "delta_file_sha256": sha256_file(DELTA_FILE),
        "n_rows": len(rows),
        "rows": rows,
    }
    outputs = {
        OUT_JSON: json.dumps(doc, indent=2) + "\n",
        OUT_CSV: render_csv(rows),
        OUT_MD: render_md(rows),
        OUT_TEX: render_tex(rows),
    }
    if args.check:
        bad = [p.name for p, s in outputs.items()
               if not p.exists() or p.read_text() != s]
        if bad:
            raise SystemExit("CHECK FAILED: drifted/missing " + ", ".join(bad))
        print("CHECK PASSED: all full-behavior outputs byte-identical.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, s in outputs.items():
        p.write_text(s)
        print(f"Wrote {p.relative_to(PROJECT_ROOT)}")
    clean = sum(1 for r in rows if r["clean_reduction"])
    print(f"{len(rows)} rows; {clean} flagged clean_reduction; "
          f"{len(rows) - clean} carry caveats.")


if __name__ == "__main__":
    main()
