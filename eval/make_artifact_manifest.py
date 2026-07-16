#!/usr/bin/env python3
"""
Build an artifact manifest for a reported result (PAPER_READINESS.md #6).

For each reported number the paper must be able to answer: which files produced
it, are they intact, and what command regenerates them. This records:

  * the estimate itself (lambda/eta + SEs) parsed from the NLS CSV,
  * SHA-256 of every raw/derived file in the eval directory (so a file can be
    verified even if it is too large to track in git),
  * the adapter/checkpoint identity and its SHA-256 where applicable,
  * the exact reproduction command,
  * the code commit.

Manifests are small and ALWAYS committed, so every cited number is traceable and
verifiable regardless of whether the multi-MB predictions are tracked.

    python eval/make_artifact_manifest.py \\
        --eval_dir baseline/Qwen-7B-Base-Local \\
        --name matched_local_base \\
        --repro "qsub train/submit_eval_base_matched.pbs"
"""
from __future__ import annotations

import argparse
import csv as _csv
import glob
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def read_estimate(eval_dir: Path) -> dict | None:
    csvs = glob.glob(str(eval_dir / "Model_1" / "*NLS_estimation*.csv"))
    if not csvs:
        return None
    rows = list(_csv.DictReader(open(csvs[0])))
    return {
        "source_csv": str(Path(csvs[0]).relative_to(PROJECT_ROOT)),
        "lambda": float(rows[0]["Estimate"]), "lambda_se": float(rows[0]["Std. Err."]),
        "eta": float(rows[1]["Estimate"]), "eta_se": float(rows[1]["Std. Err."]),
        "estimator": "Model A (NLS), structural link scale T=1",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="e.g. baseline/Qwen-7B-Base-Local")
    ap.add_argument("--name", required=True, help="manifest name, e.g. matched_local_base")
    ap.add_argument("--repro", default="", help="exact command that regenerates this")
    ap.add_argument("--adapter", default=None, help="adapter dir, if any (checksums adapter_model.safetensors)")
    ap.add_argument("--note", default="", help="free-text role of this result in the paper")
    args = ap.parse_args()

    d = (PROJECT_ROOT / args.eval_dir).resolve()
    if not d.exists():
        raise SystemExit(f"eval dir not found: {d}")

    # Batch the tracked-check: one `git ls-files` for the whole dir. (Per-file
    # --error-unmatch calls were slow, and running this BEFORE `git add` silently
    # reported everything as untracked — regenerate manifests AFTER staging.)
    tracked = set(subprocess.run(
        ["git", "ls-files", args.eval_dir], cwd=PROJECT_ROOT,
        capture_output=True, text=True).stdout.split())

    files = []
    for p in sorted(d.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(PROJECT_ROOT))
            files.append({
                "path": rel,
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
                "tracked_in_git": rel in tracked,
            })

    adapter = None
    if args.adapter:
        ap_dir = (PROJECT_ROOT / args.adapter).resolve()
        w = ap_dir / "adapter_model.safetensors"
        adapter = {"path": args.adapter, "exists": ap_dir.exists()}
        if w.exists():
            adapter["adapter_model.safetensors_sha256"] = sha256(w)
            adapter["bytes"] = w.stat().st_size

    manifest = {
        "name": args.name,
        "note": args.note,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "eval_dir": args.eval_dir,
        "reproduce": args.repro,
        "estimate": read_estimate(d),
        "adapter": adapter,
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }

    out = PROJECT_ROOT / "results" / "manifests" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out, "w"), indent=2)

    est = manifest["estimate"]
    print(f"{args.name}: {len(files)} files, {manifest['total_bytes']/1e6:.1f} MB, "
          f"{sum(f['tracked_in_git'] for f in files)} tracked in git")
    if est:
        print(f"  lambda = {est['lambda']:+.4f} (SE {est['lambda_se']:.4f})   "
              f"eta = {est['eta']:+.4f} (SE {est['eta_se']:.4f})")
    print(f"  repro: {args.repro or '(none given)'}")
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
