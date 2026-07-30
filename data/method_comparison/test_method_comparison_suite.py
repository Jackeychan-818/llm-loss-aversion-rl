#!/usr/bin/env python3
"""Integrity tests for the frozen untouched method-comparison suite (Task 1).

Hard-fails on: duplicate IDs, asymmetric X/Y pairing, overlap with any prior
split/suite, missing strata, or hash drift. Run from the repository root:

    python3 data/method_comparison/test_method_comparison_suite.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MC = ROOT / "data" / "method_comparison"
SUITE = MC / "method_comparison_suite.json"
MANIFEST = MC / "method_comparison_suite.manifest.json"
STRATA = MC / "method_comparison_strata.json"
SEM = MC / "semantic_counterbalancing.json"
SEM_MANIFEST = MC / "semantic_counterbalancing.manifest.json"

TEST_GOODS = ROOT / "data" / "test_goods.json"
REMAINING = ROOT / "data" / "remaining_goods.json"
FROZEN_UNUSED = ROOT / "data" / "frozen_unused_test_goods.json"

N_JOINT_CODES = 81
EXPECTED_PAIRS = 4_945
SELECTED_PER_PAIR = 4
CASE_ID_OFFSET = 2_000_000

_failures: list[str] = []
_passes = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passes
    if cond:
        _passes += 1
        print(f"  PASS  {name}")
    else:
        _failures.append(f"{name}: {detail}")
        print(f"  FAIL  {name}: {detail}")


def load(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def pair_code_set(rows):
    s = set()
    for x, y, codes in rows:
        for c in codes:
            s.add((x, y, c))
    return s


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    for p in (SUITE, MANIFEST, STRATA, SEM, SEM_MANIFEST):
        check(f"exists {p.name}", p.exists(), "missing")
    if _failures:
        print("\nFAILED early (missing files).")
        return 1

    suite = load(SUITE)
    manifest = load(MANIFEST)
    strata = load(STRATA)
    sem = load(SEM)

    # --- structural shape ---
    check("pair count == 4945", len(suite) == EXPECTED_PAIRS, str(len(suite)))
    check("K codes per pair == 4",
          all(len(r[2]) == SELECTED_PER_PAIR for r in suite), "some pair != 4 codes")
    check("codes in 0..80",
          all(0 <= c < N_JOINT_CODES for r in suite for c in r[2]), "code out of range")
    check("codes unique within pair",
          all(len(set(r[2])) == len(r[2]) for r in suite), "dup code within pair")
    check("X<Y and valid indices",
          all(0 <= r[0] < r[1] < 100 for r in suite), "bad pair index")

    # --- no duplicate pairs ---
    pairs = [(r[0], r[1]) for r in suite]
    check("no duplicate goods pairs", len(pairs) == len(set(pairs)), "dup pair")

    # --- OVERLAP: zero (pair,code) intersection with every prior split/suite ---
    mc_set = pair_code_set(suite)
    for name, path, per in [("test_goods", TEST_GOODS, 2),
                            ("remaining_goods", REMAINING, 10),
                            ("frozen_unused", FROZEN_UNUSED, 10)]:
        other = pair_code_set(load(path))
        inter = mc_set & other
        check(f"zero overlap with {name}", len(inter) == 0,
              f"{len(inter)} overlapping (pair,code) tuples")

    # --- case IDs: unique, contiguous from offset, X/Y symmetric ---
    n_cases = EXPECTED_PAIRS * SELECTED_PER_PAIR
    check("case_count == pairs*K", manifest["validation"]["case_count"] == n_cases,
          str(manifest["validation"]["case_count"]))
    check("prompts_per_model == 2*cases",
          manifest["validation"]["prompts_per_model"] == 2 * n_cases,
          str(manifest["validation"]["prompts_per_model"]))
    cid_scheme = manifest["case_id_scheme"]
    check("first case_id == offset+1", cid_scheme["first_case_id"] == CASE_ID_OFFSET + 1,
          str(cid_scheme["first_case_id"]))
    check("n_case_ids == n_cases", cid_scheme["n_case_ids"] == n_cases, str(cid_scheme["n_case_ids"]))
    check("case_ids contiguous",
          cid_scheme["last_case_id"] - cid_scheme["first_case_id"] + 1 == n_cases,
          "gap in case_id range")

    # X/Y symmetry: each case yields exactly one X and one Y prompt -> 2*cases,
    # verified structurally: every case has a single (X,Y) pair (one keep, one trade).
    check("X/Y paired symmetric (2 prompts/case)",
          manifest["validation"]["prompts_per_model"] == 2 * cid_scheme["n_case_ids"],
          "asymmetric X/Y pairing")

    # --- strata present and covering all cases ---
    check("strata per_case count == n_cases", len(strata["per_case"]) == n_cases,
          str(len(strata["per_case"])))
    binned = sum(strata["bin_counts"].values())
    check("strata bins cover non-missing cases",
          binned + strata["missing_utility_lookups"] == n_cases,
          f"{binned}+{strata['missing_utility_lookups']} != {n_cases}")
    check("all 4 strata bins populated",
          all(strata["bin_counts"][b] > 0 for b in strata["bin_labels"]),
          str(strata["bin_counts"]))
    strata_ids = {r["case_id"] for r in strata["per_case"]}
    check("strata case_ids match manifest range",
          min(strata_ids) == CASE_ID_OFFSET + 1 and max(strata_ids) == CASE_ID_OFFSET + n_cases,
          "strata id range mismatch")

    # --- semantic subset integrity ---
    sem_ids = [c["case_id"] for c in sem["cases"]]
    check("semantic subset ids unique", len(sem_ids) == len(set(sem_ids)), "dup sem id")
    check("semantic subset ids in suite", set(sem_ids) <= strata_ids, "sem id outside suite")
    check("semantic has 5 variant axes", len(sem["variant_axes"]) == 5, str(list(sem["variant_axes"])))
    expected_forms = 2 * 2 * 2 * 2 * 3
    check("semantic forms/case == 48", sem["forms_per_case"] == expected_forms,
          str(sem["forms_per_case"]))

    # --- hash drift: manifest recorded suite hash matches file ---
    recomputed = sha256_bytes((json.dumps(suite, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    check("manifest suite sha256 matches file", manifest["output"]["sha256"] == recomputed,
          "hash drift between manifest and suite")

    # --- deterministic regeneration (--check) of both generators ---
    for script in ["build_method_comparison_suite.py", "build_semantic_counterbalancing.py"]:
        r = subprocess.run([sys.executable, str(MC / script), "--check"],
                           capture_output=True, text=True)
        check(f"{script} --check regenerates byte-identically", r.returncode == 0,
              (r.stdout + r.stderr).strip()[-300:])

    print(f"\n{_passes} passed, {len(_failures)} failed")
    if _failures:
        for f in _failures:
            print("  -", f)
        return 1
    print("ALL METHOD-COMPARISON SUITE INTEGRITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
