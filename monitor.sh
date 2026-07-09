#!/bin/bash
# ─────────────────────────────────────────────────────────────
# monitor.sh — live training monitor for lambda-zero GRPO runs
#
# Usage:
#   bash monitor.sh                    # watches train_qd_run.log (ablation)
#   bash monitor.sh train_long_run     # watches a different log
#   bash monitor.sh --once             # print current status and exit
#
# Shows: job status, step progress, pace, reward, DAPO filter rate,
#        and the last few sample completions (Yes/No probabilities).
# ─────────────────────────────────────────────────────────────

ONCE=false
[[ "$1" == "--once" || "$2" == "--once" ]] && ONCE=true

if [[ "$1" == "--once" ]]; then
    LOG_STEM="train_qd_run"
else
    LOG_STEM="${1:-train_qd_run}"
fi
LOG="$HOME/scratch/lambda-zero/logs/${LOG_STEM}.log"

TOTAL_STEPS=98900

print_status() {
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  lambda-zero training monitor  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "══════════════════════════════════════════════════════════"

    # ── PBS job status ────────────────────────────────────────
    echo ""
    echo "── PBS Jobs ──────────────────────────────────────────────"
    qstat -u jackeyc0 2>/dev/null || echo "  (no jobs running)"

    # ── Log file check ────────────────────────────────────────
    if [[ ! -f "$LOG" ]]; then
        echo ""
        echo "  Log not found: $LOG"
        return
    fi

    # ── Step progress ─────────────────────────────────────────
    echo ""
    echo "── Progress ($LOG_STEM) ──────────────────────────────────"
    PROGRESS_LINE=$(grep -oP '\d+%\|[█▌▍▎▏ ]+\| \d+/\d+ \[[\d:]+<[\d:a-z]+,\s*[\d.]+s/it\]' "$LOG" 2>/dev/null | tail -1)
    if [[ -n "$PROGRESS_LINE" ]]; then
        STEP=$(echo "$PROGRESS_LINE" | grep -oP '\| \K\d+(?=/)')
        PCT=$(echo "$PROGRESS_LINE" | grep -oP '^\d+')
        ELAPSED=$(echo "$PROGRESS_LINE" | grep -oP '\[\K[^<]+')
        REMAINING=$(echo "$PROGRESS_LINE" | grep -oP '<\K[^,]+')
        PACE=$(echo "$PROGRESS_LINE" | grep -oP '[\d.]+(?=s/it)')
        echo "  Step:      $STEP / $TOTAL_STEPS  ($PCT%)"
        echo "  Elapsed:   $ELAPSED"
        echo "  Remaining: $REMAINING"
        echo "  Pace:      ${PACE}s/it"
    else
        echo "  (no step data yet — model may still be loading)"
    fi

    # ── Recent rewards ────────────────────────────────────────
    echo ""
    echo "── Recent Rewards & Metrics (last 5 logged steps) ────────"
    grep -oP "'reward': '[-\d.e+]+'" "$LOG" 2>/dev/null | grep -oP "[-\d.e+]+(?=')" | tail -5 | while read r; do
        python3 -c "
v=float('$r')
bar='█'*min(int(abs(v)*15),30)
sign='+' if v>=0 else ''
note='(positive ✓)' if v>0 else '(negative — irrational)'
print(f'  reward: {sign}{v:.4f}  {bar}  {note}')
" 2>/dev/null
    done

    # ── DAPO filter rate ──────────────────────────────────────
    echo ""
    echo "── DAPO Filter — frac_reward_zero_std ────────────────────"
    echo "  (1.0 = all G=16 outputs identical → zero gradient; want → 0)"
    grep -oP "'frac_reward_zero_std': '[\d.]+'" "$LOG" 2>/dev/null | grep -oP "[\d.]+(?=')" | tail -5 | while read v; do
        python3 -c "
v=float('$v')
n=int(v*30)
bar='█'*n+'░'*(30-n)
note='⚠ converging' if v<0.5 else ('✓ healthy diversity' if v<0.8 else '✗ mostly degenerate')
print(f'  {v:.3f}  [{bar}]  {note}')
" 2>/dev/null
    done

    # ── Yes-rate from sample completions ──────────────────────
    echo ""
    echo "── Sample Completions Yes-rate (from log table) ──────────"
    echo "  (rational target ~50%; 0% = model still refuses everything)"
    python3 -c "
import re, sys
log = open('$LOG').read()
# extract reward_fn column values from completion tables
vals = re.findall(r'│ [│├└]\s+\w.*?\s+│\s+([-\d.]+)\s+│', log)
if vals:
    recent = vals[-20:]
    pos = sum(1 for v in recent if float(v) > 0)
    print(f'  Last 20 completions: {pos}/20 rational ({pos/20*100:.0f}%)')
else:
    print('  (no completion data yet)')
" 2>/dev/null

    echo ""
    echo "══════════════════════════════════════════════════════════"
}

if $ONCE; then
    print_status
else
    echo "Monitoring $LOG  (refresh every 60s, Ctrl+C to stop)"
    while true; do
        clear
        print_status
        sleep 60
    done
fi
