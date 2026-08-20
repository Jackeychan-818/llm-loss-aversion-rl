#!/bin/bash
# Progress across every SFT-sensitivity run, at a glance.
#
# HF writes tqdm progress with carriage returns, not newlines, so a plain
# `tail` on the log shows one huge overwritten line. `tr '\r' '\n'` splits it
# back into the individual bar updates.
cd "$(dirname "$0")/.."
printf "%-40s %10s %7s %10s %s\n" CELL STEPS PCT ELAPSED STATE
printf '%.0s-' {1..88}; printf '\n'
for f in logs/sft_sensitivity/*.log; do
    [ -e "$f" ] || continue
    cell=$(basename "$f" .log)
    case "$cell" in *validate*|eval_*|estimate_*) continue ;; esac
    last=$(tr '\r' '\n' < "$f" | grep -E '\|.*\| *[0-9]+/[0-9]+' | tail -1)
    frac=$(sed -E 's/.*\| *([0-9]+\/[0-9]+).*/\1/' <<<"$last")
    pct=$(sed -E 's/^ *([0-9]+)%.*/\1%/' <<<"$last")
    el=$(sed -E 's/.*\[([0-9:]+)<.*/\1/' <<<"$last")
    if grep -q "Done:" "$f" 2>/dev/null; then state=done
    elif [ -n "$frac" ]; then state=running
    else state=starting; frac=-; pct=-; el=-; fi
    printf "%-40s %10s %7s %10s %s\n" "$cell" "${frac:--}" "${pct:--}" "${el:--}" "$state"
done
echo
echo "queue: $(qstat -u "$USER" 2>/dev/null | grep -cE ' R ') running, $(qstat -u "$USER" 2>/dev/null | grep -cE ' Q ') queued"
