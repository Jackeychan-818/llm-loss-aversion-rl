#!/bin/bash
# Submit the 12 Phase-A cells of the SFT batch sensitivity experiment.
#
# Runs on the LOGIN node, where git exists. It verifies a clean tree once and
# passes the resolved commit into every job, because git is NOT available on the
# compute nodes and a manifest recording "unknown" is the provenance gap
# (SFTPROV-001) this experiment is meant not to repeat.
#
# Scope: Amendment 1 — Phase A only, endpoint evaluation only, ceiling
# 8.2 GPU-hours / 525 SU. This script does not submit Phase B, intermediate
# checkpoint evaluations, or the full 30,016-prompt runs.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! git diff --quiet HEAD -- train/ eval/; then
    echo "REFUSED: uncommitted changes under train/ or eval/. Commit first."
    git status --short train/ eval/
    exit 2
fi
COMMIT="$(git rev-parse HEAD)"
echo "submitting Phase A at commit $COMMIT"

for EB in 1 16 32 64; do
  for S in 1 2 3; do
    CELL="sft_sens_phA_eb${EB}_lr1e-6_seed${S}_h6016"
    ID=$(qsub -v "CELL=${CELL},GIT_COMMIT=${COMMIT}" train/submit_sft_sensitivity.pbs)
    printf "%-42s %s\n" "$CELL" "$ID"
  done
done
