#!/usr/bin/env python3
"""Survey 9 frontier models in Loss_Aversion repo, compute δ̃ for all goods pair cases.

Outputs:
  data/deltas/delta_{model}.json     — per-case δ̃ values
  data/deltas/_summary.json          — model survey table + cross-model summary stats
"""
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

REPO = Path("/Users/jackey/Desktop/Loss_Aversion")
OUT = Path("/Users/jackey/Desktop/grpo/data/deltas")
OUT.mkdir(parents=True, exist_ok=True)

# 9 frontier models in MODEL_REGISTRY (run_all_models.py).
# For each, we record which estimation file is the "primary" (used for δ̃).
MODELS = {
    "GPT-3.5":     {"estimator": "A", "est_file": "GPT-3.5_NLS_estimation_T1(Model A).csv"},
    "GPT-4o":      {"estimator": "A", "est_file": "GPT-4o_NLS_estimation_T1(Model A).csv"},
    "GPT-5":       {"estimator": "C", "est_file": "GPT-5_Logit_MLE_Results_Final(Model C).csv"},
    "GPT-5.2":     {"estimator": "A", "est_file": "GPT-5.2_NLS_estimation_T1(Model A).csv"},
    "Claude":      {"estimator": "B", "est_file": "Claude_sms_results_bootstrap(Model B).csv"},
    "Gemini":      {"estimator": "C", "est_file": "Gemini_Logit_MLE_Results_Final(Model C).csv"},
    "Apertus-70B": {"estimator": "A", "est_file": "Apertus-70B_NLS_estimation_T1(Model A).csv"},
    "Llama-70B":   {"estimator": "A", "est_file": "Llama-70B_NLS_estimation_T1(Model A).csv"},
    "DeepSeek-R1": {"estimator": "A", "est_file": "DeepSeek-R1_NLS_estimation_T1(Model A).csv"},
}

GOODS_FILES = ["trial_goods.json", "test_goods.json", "remaining_goods.json"]


# ── shared decoders (mirror run_all_models.py) ────────────────────────────
def decode_attr(c):
    i = c % 3; c //= 3
    j = c % 3; c //= 3
    k = c % 3; c //= 3
    l = c % 3
    return i, j, k, l


def load_estimates(path):
    """Return (alpha[1..100], beta[(i,j) for i,j in 1..3], se_nonzero, n_alphas, n_betas).
    α_1 fixed = 0; β_{1,1} fixed = 0."""
    alphas = {1: 0.0}
    betas = {(1, 1): 0.0}
    se_alphas = []
    se_betas = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            name = row[0].strip().strip('"')
            est = float(row[1])
            se = float(row[2])
            if name.startswith("alpha_"):
                idx = int(name.split("_")[1])
                alphas[idx] = est
                se_alphas.append(se)
            elif name.startswith("beta_"):
                # name like "beta_2,3"
                ij = name.split("_")[1].split(",")
                i, j = int(ij[0]), int(ij[1])
                betas[(i, j)] = est
                se_betas.append(se)
    nz_alpha = sum(1 for s in se_alphas if s > 0)
    nz_beta = sum(1 for s in se_betas if s > 0)
    return {
        "alphas": alphas,
        "betas": betas,
        "n_alphas": len(se_alphas),
        "n_alpha_se_nonzero": nz_alpha,
        "n_betas": len(se_betas),
        "n_beta_se_nonzero": nz_beta,
        "median_alpha_se": sorted(se_alphas)[len(se_alphas) // 2] if se_alphas else None,
        "median_beta_se": sorted(se_betas)[len(se_betas) // 2] if se_betas else None,
    }


def load_raw_counts(model):
    """Return (yes_x_rate, yes_y_rate, total) from raw count number CSV.
    Matrix rows = X-perspective answer; cols = Y-perspective answer."""
    base = REPO / "baseline" / model / "Model_1"
    p_num = base / f"{model}_raw_choice_counts_number.csv"
    p_combined = base / f"{model}_raw_choice_counts_combined.csv"

    if p_num.exists():
        with open(p_num) as f:
            rows = list(csv.reader(f))
        # rows[1..3] = Yes / tie / No rows;  cols 1..3 = Yes/tie/No
        yes_yes = int(rows[1][1]); yes_tie = int(rows[1][2]); yes_no = int(rows[1][3])
        tie_yes = int(rows[2][1]); tie_tie = int(rows[2][2]); tie_no = int(rows[2][3])
        no_yes = int(rows[3][1]); no_tie = int(rows[3][2]); no_no = int(rows[3][3])
        total = yes_yes + yes_tie + yes_no + tie_yes + tie_tie + tie_no + no_yes + no_tie + no_no
        yes_x = yes_yes + yes_tie + yes_no       # X said Yes
        yes_y = yes_yes + tie_yes + no_yes       # Y said Yes
        return {
            "yes_x_rate": yes_x / total if total else None,
            "yes_y_rate": yes_y / total if total else None,
            "total": total,
            "yes_yes": yes_yes,
            "yes_no": yes_no,
            "no_yes": no_yes,
            "no_no": no_no,
            "source": p_num.name,
        }
    elif p_combined.exists():
        # combined file has "mean (median)" cell strings, percentages (0..100)
        with open(p_combined) as f:
            rows = list(csv.reader(f))
        def pct(cell):
            return float(cell.split()[0]) / 100.0
        yes_yes = pct(rows[1][1]); yes_tie = pct(rows[1][2]); yes_no = pct(rows[1][3])
        tie_yes = pct(rows[2][1]); tie_tie = pct(rows[2][2]); tie_no = pct(rows[2][3])
        no_yes = pct(rows[3][1]); no_tie = pct(rows[3][2]); no_no = pct(rows[3][3])
        yes_x = yes_yes + yes_tie + yes_no
        yes_y = yes_yes + tie_yes + no_yes
        return {
            "yes_x_rate": yes_x,
            "yes_y_rate": yes_y,
            "total": None,    # only fractions stored
            "yes_yes": yes_yes,
            "yes_no": yes_no,
            "no_yes": no_yes,
            "no_no": no_no,
            "source": p_combined.name,
        }
    return None


# ── delta computation ─────────────────────────────────────────────────────
def compute_deltas_for_model(model, model_cfg, goods_cases):
    """goods_cases: list of (case_id, X_num, Y_num, attr_code) — global IDs.
    Returns dict {case_id: delta}."""
    est_path = REPO / "baseline" / model / "Model_1" / model_cfg["est_file"]
    if not est_path.exists():
        return None, f"missing est file: {est_path}"
    est = load_estimates(est_path)
    a = est["alphas"]
    b = est["betas"]

    out = {}
    missing = 0
    for cid, X_num, Y_num, attr_code in goods_cases:
        i, j, k, l = decode_attr(attr_code)
        # X_num/Y_num are 0-indexed in dataset; α is 1-indexed in CSV → +1
        aX = a.get(X_num + 1)
        aY = a.get(Y_num + 1)
        bX = b.get((i + 1, j + 1))
        bY = b.get((k + 1, l + 1))
        if aX is None or aY is None or bX is None or bY is None:
            missing += 1
            continue
        delta = math.exp(aX + bX) - math.exp(aY + bY)
        out[cid] = delta
    return out, f"computed {len(out)} cases, {missing} missing"


def build_global_cases():
    """Replicate run_all_models.py case-id assignment for trial → test → remaining."""
    all_cases = []  # list of (cid, X_num, Y_num, attr_code, source_file)
    cid = 0
    per_file = {}
    for fname in GOODS_FILES:
        ds = json.load(open(REPO / fname))
        start = cid + 1
        for entry in ds:
            X_num, Y_num, attr_list = entry[0], entry[1], entry[2]
            for code in attr_list:
                cid += 1
                all_cases.append((cid, X_num, Y_num, code, fname))
        per_file[fname] = (start, cid)
    return all_cases, per_file


# ── main ──────────────────────────────────────────────────────────────────
def main():
    print("Building global case index from trial→test→remaining...")
    all_cases, per_file = build_global_cases()
    print(f"  Total cases: {len(all_cases)}")
    for f, (s, e) in per_file.items():
        print(f"  {f}: case_id {s} – {e}")

    # ── Step 1: Survey ──
    print("\n=== Step 1: Model survey ===")
    survey = {}
    for model, cfg in MODELS.items():
        raw = load_raw_counts(model)
        est_path = REPO / "baseline" / model / "Model_1" / cfg["est_file"]
        if est_path.exists():
            est = load_estimates(est_path)
            valid = est["n_alpha_se_nonzero"] > 0 and est["n_beta_se_nonzero"] > 0
        else:
            est = None
            valid = False
        survey[model] = {
            "estimator": cfg["estimator"],
            "est_file": cfg["est_file"],
            "est_file_exists": est_path.exists(),
            "raw_yes_x_rate": raw["yes_x_rate"] if raw else None,
            "raw_yes_y_rate": raw["yes_y_rate"] if raw else None,
            "raw_total": raw["total"] if raw else None,
            "raw_source": raw["source"] if raw else None,
            "n_alphas": est["n_alphas"] if est else None,
            "n_alpha_se_nonzero": est["n_alpha_se_nonzero"] if est else None,
            "n_betas": est["n_betas"] if est else None,
            "n_beta_se_nonzero": est["n_beta_se_nonzero"] if est else None,
            "median_alpha_se": est["median_alpha_se"] if est else None,
            "median_beta_se": est["median_beta_se"] if est else None,
            "valid_for_delta": valid,
        }

    # Print survey table
    hdr = f"{'Model':<13} {'Est':<3} {'YesX':>6} {'YesY':>6} {'N':>7} {'α#':>4} {'α-nzSE':>7} {'β-nzSE':>7} {'medα-SE':>9} {'medβ-SE':>9} {'OK':>3}"
    print(hdr)
    print("-" * len(hdr))
    for m, s in survey.items():
        yx = f"{s['raw_yes_x_rate']:.3f}" if s['raw_yes_x_rate'] is not None else "—"
        yy = f"{s['raw_yes_y_rate']:.3f}" if s['raw_yes_y_rate'] is not None else "—"
        n = str(s['raw_total']) if s['raw_total'] else "—"
        na = str(s['n_alphas']) if s['n_alphas'] else "—"
        nzA = str(s['n_alpha_se_nonzero']) if s['n_alpha_se_nonzero'] is not None else "—"
        nzB = str(s['n_beta_se_nonzero']) if s['n_beta_se_nonzero'] is not None else "—"
        mA = f"{s['median_alpha_se']:.4f}" if s['median_alpha_se'] is not None else "—"
        mB = f"{s['median_beta_se']:.4f}" if s['median_beta_se'] is not None else "—"
        ok = "✓" if s["valid_for_delta"] else "✗"
        print(f"{m:<13} {s['estimator']:<3} {yx:>6} {yy:>6} {n:>7} {na:>4} {nzA:>7} {nzB:>7} {mA:>9} {mB:>9} {ok:>3}")

    # ── Step 2: Compute δ̃ ──
    print("\n=== Step 2: Compute δ̃ per model ===")
    cases_for_delta = [(cid, X, Y, code) for (cid, X, Y, code, _) in all_cases]
    deltas_by_model = {}
    for model, cfg in MODELS.items():
        if not survey[model]["valid_for_delta"]:
            print(f"  {model}: SKIP (no valid estimates)")
            continue
        deltas, msg = compute_deltas_for_model(model, cfg, cases_for_delta)
        print(f"  {model}: {msg}")
        if deltas:
            deltas_by_model[model] = deltas
            out_path = OUT / f"delta_{model}.json"
            with open(out_path, "w") as f:
                # write keyed by string case_id (JSON object keys)
                json.dump({str(cid): v for cid, v in sorted(deltas.items())}, f, indent=2)
            print(f"    → saved {out_path}")

    # ── Step 3: Cross-model summary ──
    print("\n=== Step 3: Cross-model summary ===")
    # Per case: collect signs from each valid model
    case_signs = defaultdict(dict)
    case_deltas = defaultdict(dict)
    for m, dd in deltas_by_model.items():
        for cid, v in dd.items():
            case_signs[cid][m] = (1 if v > 0 else (-1 if v < 0 else 0))
            case_deltas[cid][m] = v

    n_models = len(deltas_by_model)
    unanimous_pos = 0
    unanimous_neg = 0
    unanimous_any = 0
    split = 0
    only_one = 0
    case_disagreement = []  # cases where models disagree most

    for cid, signs in case_signs.items():
        if len(signs) < 2:
            only_one += 1
            continue
        s = set(signs.values())
        s.discard(0)
        if len(s) == 0:
            unanimous_any += 1
        elif len(s) == 1:
            unanimous_any += 1
            if 1 in s:
                unanimous_pos += 1
            else:
                unanimous_neg += 1
        else:
            split += 1
            # measure of disagreement: max - min δ across models
            vals = case_deltas[cid].values()
            spread = max(vals) - min(vals)
            case_disagreement.append((cid, spread, dict(case_deltas[cid])))

    # mean |δ̃| per case (across models with valid δ for that case)
    mean_abs = []
    for cid, dd in case_deltas.items():
        if not dd:
            continue
        mean_abs.append(sum(abs(v) for v in dd.values()) / len(dd))
    mean_abs.sort()

    def pct(p, arr):
        if not arr:
            return None
        idx = max(0, min(len(arr) - 1, int(p * len(arr))))
        return arr[idx]

    print(f"  Models with valid δ̃: {n_models}  → {sorted(deltas_by_model.keys())}")
    print(f"  Cases analyzed:      {len(case_signs)}")
    print(f"  Unanimous sign:      {unanimous_any}  ({unanimous_pos} all positive, {unanimous_neg} all negative)")
    print(f"  Split (disagreement): {split}")
    print(f"  Distribution of mean |δ̃| across cases:")
    print(f"    min={mean_abs[0]:.4f}  p25={pct(0.25, mean_abs):.4f}  median={pct(0.5, mean_abs):.4f}  "
          f"p75={pct(0.75, mean_abs):.4f}  p95={pct(0.95, mean_abs):.4f}  max={mean_abs[-1]:.4f}")

    # Top-10 disagreement cases
    case_disagreement.sort(key=lambda t: -t[1])
    top_disagree = []
    print(f"\n  Top-10 cases by max-min δ̃ spread:")
    for cid, spread, dd in case_disagreement[:10]:
        top_disagree.append({"case_id": cid, "spread": spread, "deltas": dd})
        per_model = ", ".join(f"{m}={v:+.3f}" for m, v in sorted(dd.items()))
        print(f"    case {cid}: spread={spread:.3f}  {per_model}")

    # ── Step 4: Consensus subsets (drop weak/degenerate models) ──
    # v2: drop Claude (boundary-corner SMS estimate, "always No" baseline)
    # v3: drop Claude + Apertus-70B + GPT-3.5 (all baseline Yes-rates < 5%)
    consensus_specs = [
        ("v2", [m for m in deltas_by_model if m != "Claude"]),
        ("v3", [m for m in deltas_by_model if m not in ("Claude", "Apertus-70B", "GPT-3.5")]),
    ]
    consensus_results = {}
    print("\n=== Step 4: Consensus subsets ===")
    for tag, models in consensus_specs:
        per_case = {}
        u_pos = u_neg = sp = 0
        m_abs = []
        for cid in sorted({c for m in models for c in deltas_by_model[m]}):
            vals = {m: deltas_by_model[m][cid] for m in models if cid in deltas_by_model[m]}
            if not vals:
                continue
            s = {1 if v > 0 else (-1 if v < 0 else 0) for v in vals.values()}
            s.discard(0)
            if len(s) <= 1:
                if s == {1}: u_pos += 1
                elif s == {-1}: u_neg += 1
            else:
                sp += 1
            mean_d = sum(vals.values()) / len(vals)
            m_abs.append(abs(mean_d))
            per_case[str(cid)] = {
                "mean_delta": mean_d,
                "per_model": vals,
                "unanimous_sign": len(s) <= 1,
            }

        out_path = OUT / f"delta_consensus_{tag}.json"
        with open(out_path, "w") as f:
            json.dump(per_case, f, indent=2)

        m_abs.sort()
        stats = {
            "tag": tag,
            "models": models,
            "n_models": len(models),
            "n_cases": len(per_case),
            "unanimous_sign": u_pos + u_neg,
            "unanimous_positive": u_pos,
            "unanimous_negative": u_neg,
            "split": sp,
            "mean_abs_delta_distribution": {
                "min": m_abs[0] if m_abs else None,
                "p25": pct(0.25, m_abs),
                "median": pct(0.5, m_abs),
                "p75": pct(0.75, m_abs),
                "p95": pct(0.95, m_abs),
                "max": m_abs[-1] if m_abs else None,
            },
        }
        consensus_results[tag] = stats
        print(f"  {tag}: {len(models)} models — {models}")
        print(f"    unanimous sign: {stats['unanimous_sign']}  (pos={u_pos}, neg={u_neg})")
        print(f"    split:          {sp}")
        d = stats["mean_abs_delta_distribution"]
        print(f"    mean |δ̃|:  min={d['min']:.4f}  p25={d['p25']:.4f}  median={d['median']:.4f}  "
              f"p75={d['p75']:.4f}  p95={d['p95']:.4f}  max={d['max']:.4f}")
        print(f"    → saved {out_path}")

    # DeepSeek-R1 standalone copy (for single-model use)
    if "DeepSeek-R1" in deltas_by_model:
        ds_path = OUT / "delta_deepseek.json"
        with open(ds_path, "w") as f:
            json.dump({str(cid): v for cid, v in sorted(deltas_by_model["DeepSeek-R1"].items())},
                      f, indent=2)
        print(f"  DeepSeek-R1 standalone → saved {ds_path}")

    # Save full summary
    summary = {
        "models_surveyed": survey,
        "delta_computation": {
            "total_cases": len(all_cases),
            "per_file_case_id_ranges": {f: {"start": s, "end": e} for f, (s, e) in per_file.items()},
            "n_models_with_valid_delta": n_models,
            "models_with_valid_delta": sorted(deltas_by_model.keys()),
        },
        "cross_model_all": {
            "n_cases": len(case_signs),
            "unanimous_sign": unanimous_any,
            "unanimous_positive": unanimous_pos,
            "unanimous_negative": unanimous_neg,
            "split": split,
            "mean_abs_delta_distribution": {
                "min": mean_abs[0] if mean_abs else None,
                "p25": pct(0.25, mean_abs),
                "median": pct(0.5, mean_abs),
                "p75": pct(0.75, mean_abs),
                "p95": pct(0.95, mean_abs),
                "max": mean_abs[-1] if mean_abs else None,
            },
            "top10_disagreement_cases": top_disagree,
        },
        "consensus": consensus_results,
    }
    with open(OUT / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → summary written to {OUT/'_summary.json'}")


if __name__ == "__main__":
    main()
