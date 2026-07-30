# CPU-only paper-gates — audit report

*Branch `codex/cpu-paper-gates`, 2026-07-30. Executes the six CPU-only tasks in
`NSCC_CPU_WORK_PROMPT.md`. No GPU job submitted, no model inference, no checkpoint
evaluated, no frozen suite opened.*

## Read-only audit (pre-work)

- Branch at start: `main` @ `eac80ad`; created `codex/cpu-paper-gates`.
- Full SFT runs verified: `checkpoints/sft_qwen_delta_seed{1,2}`, **15
  checkpoints each** (2k–30k) + `final`; seeds 1/2; `max_steps=30000`;
  batch1/accum1; 98,900 examples balanced 49,450/49,450; 49,450 distinct case
  IDs. Source SHAs match the protocol (data `67e4e95e…`, delta `569bf72e…`, goods
  `7a28e91a…`). Pilot preserved at `…seed1_pilot6k` (distinguishable by
  `max_steps=6000`).
- **Provenance gap:** `git_commit:"unknown"` in all three SFT manifests
  (compute-node `git rev-parse` failure). Recorded as `SFTPROV-001`.

## Deliverables (commit → files)

| Task | Commit | Key files |
|---|---|---|
| 1 — untouched suite | `adc729b` | `data/method_comparison/{build_method_comparison_suite.py,method_comparison_suite.json,.manifest.json,method_comparison_strata.json,build_semantic_counterbalancing.py,semantic_counterbalancing.json,.manifest.json,METHOD_COMPARISON_SUITE.md,test_method_comparison_suite.py}` |
| 2 — comparison protocol | `cffe7e8` | `METHOD_COMPARISON_PROTOCOL.md` |
| 3 — scale-matched control | `a5dbcb3` | `train/{reward_functions.py,grpo_train.py,build_scale_matched_spec.py,test_scale_matched_reward.py,configs/qwen25_7b_qwen_delta_scale_matched.yaml}`, `results/scale_matched_reward/{scale_matched_reward_spec.json,SCALE_MATCHED_REWARD.md}` |
| 4 — robust inference | `65c6604` | `eval/{robust_inference.py,estimator_recovery.py,run_robust_bootstrap.py,test_robust_inference.py,ROBUST_INFERENCE.md}`, `train/{submit_cpu_recovery.pbs,submit_cpu_bootstrap.pbs}` |
| 5 — full-behavior aggregation | `42581f5` | `eval/aggregate_full_behavior.py`, `results/full_behavior/{full_behavior.json,.csv,.md,.tex,README.md}` |
| 6 — GRPO efficiency | `5554c9a` | `eval/analyze_grpo_efficiency.py`, `results/grpo_efficiency/{grpo_efficiency.json,.md}` |
| docs sync + audit | (this commit) | `RESEARCH_ROADMAP.md`, `PAPER_READINESS.md`, `KNOWN_ISSUES.md`, `CAUSAL_BASELINE_PROTOCOL.md`, `PROJECT_OVERVIEW.md`, `HISTORY.md`, `AGENTS.md`, `CLAUDE.md`, this file |

## Checks run (all pass)

- Task 1: `test_method_comparison_suite.py` → **31/31** (overlap, pairing, IDs,
  strata, hash drift, deterministic `--check`).
- Task 3: `test_scale_matched_reward.py` → **29/29**; existing
  `test_causal_baselines.py` → **10/10** (default path unchanged).
- Task 4: `test_robust_inference.py` → **16/16**; `estimator_recovery.py --quick`
  near-zero bias, no failures.
- Task 5/6: `--check` byte-identical regeneration.
- `eval/test_select_checkpoint.py` → pass (selector unchanged).
- `qstat -u jackeyc0`: no jobs (no GPU submission).

## Key findings

- **Untouched suite is genuinely disjoint:** 59 never-used codes/pair remain
  after excluding test/train/frozen-unused's 22; the suite uses 4/pair with zero
  `(pair,code)` overlap (asserted). OOD-50 uses a different goods population.
- **λ does not tell the whole story:** the full-behavior aggregation flags 15/40
  committed rows `clean_reduction=NO` — low λ contradicted by high η,
  inconsistency, or choice collapse (incl. the matched base itself).
- **Magnitude weighting concentrates updates:** high-|δ̃|(>1) cases carry 63.1% of
  total |δ̃| mass from 27.1% of cases; measured zero task-reward advantage ≈57%.
- **Efficiency ≠ effectiveness:** SFT is far cheaper (~1.4 h vs ~51 h/seed at 30k)
  for the same unique-prompt exposure, but the method winner is decided only on
  the untouched suite under the frozen protocol.

## Ledger movement (KNOWN_ISSUES.md)

- `ABLATION-001` → **spec-complete, run-pending**.
- `ENV-001` → **partial** (hard-fail on dropped algorithm-defining keys added).
- `PAIR-001`, `INFER-001` → **partial** (tested ID-join + pair-clustered
  bootstrap/recovery in a separate layer; frozen headline estimator untouched).
- New: `SFTPROV-001` (SFT manifests' `git_commit:"unknown"`).

## Unresolved blockers / commands still requiring GPU (next phase)

1. **Evaluate the full SFT 15×2 grid on `test_goods`** (`submit_eval_baseline_ckpt.pbs`),
   then run the unchanged selector **with `--provenance_note`** per the protocol.
   Budget-gated: ~30 evals may exceed the ~3,312 SU balance (subset or top-up
   decision still open).
2. **Train sign-only and scale-matched GRPO** (seeds 1&2) —
   `submit_train_glong_qwen_delta_sign.pbs` and a scale-matched submit using
   `qwen25_7b_qwen_delta_scale_matched.yaml` — then compare. Sign-only alone is
   ~21k SU (postponed).
3. **Open the untouched suite once** on the already-selected checkpoints + base,
   under `METHOD_COMPARISON_PROTOCOL.md`.
4. **CPU (no GPU) but PBS-sized:** `submit_cpu_recovery.pbs` (recovery grid) and
   `submit_cpu_bootstrap.pbs` (pair-clustered bootstrap on existing predictions).
5. **Provenance:** capture git commit robustly in future training manifests
   (`SFTPROV-001`); feed per-model frozen-estimator utilities into the bootstrap
   to close `INFER-001` on the headline numbers.

## Constraints honored

No GPU job submitted · no model inference · no checkpoint evaluated · no frozen
suite opened or adapted to · frozen headline estimator (`core_exp_refactored.py`)
unmodified · default magnitude GRPO path unchanged · large artifacts/checkpoints
left untracked · small reviewable commits per task · branch pushed, **not**
merged.
