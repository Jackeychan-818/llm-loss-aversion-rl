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

- Task 1: `test_method_comparison_suite.py` → **39/39** (overlap, pairing, IDs,
  strata, hash drift, deterministic `--check`, no-embedded-checkpoint-IDs guard,
  and the DRAFT-status / `--validate` / `--emit-ready` refusal guards). Up from
  31 → 35 (correction round) → 39 (governance guards).
- Task 3: `test_scale_matched_reward.py` → **29/29**; existing
  `test_causal_baselines.py` → **10/10** (default path unchanged).
- Task 4: `test_robust_inference.py` → **22/22** (incl. the pair-split-free
  proof and the no-fabricated-probability / hard-choice guards);
  `estimator_recovery.py --quick` near-zero bias, no failures.
- Task 5/6: `--check` byte-identical regeneration; the full-behavior `--check`
  additionally **passes from a clean `git archive`** (snapshot-only path).
- `eval/test_select_checkpoint.py` → pass (selector unchanged).
- `qstat -u jackeyc0`: no jobs (no GPU submission).

**Reproducibility & governance deliverables (correction rounds):**
- `results/full_behavior/full_behavior_snapshot.json` — tracked canonical row
  snapshot; JSON/CSV/MD/TEX render from it, `--refresh` re-reads NSCC dirs.
- `data/method_comparison/semantic_preopening_eval_manifest.json` (status
  `DRAFT_UNRESOLVED_DO_NOT_OPEN`, `all_families_resolved: false`) +
  `build_semantic_preopening_manifest.py` with a hard `--validate` / `--emit-ready`
  gate that refuses opening until every family is resolved with non-null selector
  hashes; the immutable `FROZEN_READY_TO_OPEN` manifest is emitted only then.
- `ENVIRONMENT.md` — declared Python/SciPy/NumPy/PyYAML/PyTorch minimums (not a
  full lock).

## Key findings

- **Untouched suite is genuinely disjoint:** 59 never-used codes/pair remain
  after excluding test/train/frozen-unused's 22; the suite uses 4/pair with zero
  `(pair,code)` overlap (asserted). OOD-50 uses a different goods population.
- **λ does not tell the whole story:** the full-behavior aggregation flags 15/40
  committed rows `clean_reduction=NO`; of those, **10 have `|λ|<0.5` plus a
  contradictory caveat** (high η, inconsistency, or choice collapse), and 5 are
  `NO` only because `|λ|≥0.5` — the matched base is one of those 5 (λ=7.637), not
  one of the 10.
- **Magnitude weighting concentrates updates:** high-|δ̃|(>1) cases carry 63.1% of
  total |δ̃| mass from 27.1% of cases; measured zero task-reward advantage ≈57%.
- **Efficiency ≠ effectiveness:** SFT is far cheaper (~1.51 h vs ~51 h/seed at 30k)
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
2. **Train sign-only and scale-matched GRPO** (seeds 1&2, distinct families) —
   `submit_train_glong_qwen_delta_sign.pbs` and a scale-matched submit using
   `qwen25_7b_qwen_delta_scale_matched.yaml`. See the combined budget below.
3. **Open the untouched suite once** on the already-selected checkpoints + base,
   under `METHOD_COMPARISON_PROTOCOL.md`.
4. **CPU (no GPU) but PBS-sized:** `submit_cpu_recovery.pbs` (recovery grid) and
   `submit_cpu_bootstrap.pbs` (pair-clustered bootstrap on existing predictions).
5. **Provenance:** capture git commit robustly in future training manifests
   (`SFTPROV-001`); feed per-model frozen-estimator utilities into the bootstrap
   to close `INFER-001` on the headline numbers.

## Combined GPU-phase budget estimate (both new families + evals)

Rough, single A100-40GB, HF generate (no vLLM). GPU-hours anchored to measured
throughput (GRPO ~5.57–6.1 s/step; test_goods eval ~0.5–1 h; the untouched suite
is ~2× test_goods so ~1–2 h/model). SU is shown as a range because the observed
rate spans ~106 SU/GPU-h (clean SFT measurement) to ~220 SU/GPU-h (older jobs).

| item | quantity | GPU-h | SU (≈106–220/GPU-h) |
|---|---|---:|---:|
| **Train sign-only GRPO** | 2 seeds × 30k @ ~46 h | ~93 | ~9.8k–20.5k |
| **Train scale-matched GRPO** | 2 seeds × 30k @ ~46 h | ~93 | ~9.8k–20.5k |
| **Eval SFT grid** (trained) | 15 × 2 seeds | ~15–30 | ~1.6k–6.6k |
| **Eval sign-only grid** | 15 × 2 seeds | ~15–30 | ~1.6k–6.6k |
| **Eval scale-matched grid** | 15 × 2 seeds | ~15–30 | ~1.6k–6.6k |
| **Final untouched-suite eval** | base + 2 seeds × 4 families = 9 models × ~1–2 h | ~9–18 | ~1.0k–4.0k |
| **Semantic-counterbalancing eval** | 9 models × 160 cases × 48 forms = 7,680 prompt-forms/model | ~3.5–7 | ~0.4k–1.5k |
| **Total (before capability contingency)** | | **~243–301 GPU-h** | **~26k–67k SU** |

**Conclusion:** the full program costs **~26k–67k SU** *before* any capability
contingency (IFEval / GSM-Symbolic add more), dominated by sign-only +
scale-matched *training* (~40k SU alone at the high rate). This vastly exceeds
the **~3,312 SU** balance, so it requires a substantial **SU top-up** (roughly
10–20× the current balance). Cheapest partial progress within budget: the SFT
eval grid only (~1.6k–6.6k SU) — still borderline — or a reduced-subset SFT
trajectory. The CPU-phase recovery/bootstrap PBS jobs consume **no GPU SU**.

## Pre-opening corrections (commit `e475699`) + origin/main merge

Applied transparently before any frozen suite was opened:

1. **True pair-clustered bootstrap** — resamples whole goods-pair clusters keyed
   by `(X_num, Y_num)`; a test proves a pair's repeated configurations cannot be
   split across clusters (`test_robust_inference.py`, now **22/22**). Recovery
   uses singleton pairs per simulated case.
2. **Semantic-counterbalancing amended** (recorded as an explicit amendment, not
   a silent rewrite): `permitted_models` broadened to base + every selected
   magnitude-GRPO/SFT/sign-only/scale-matched seed; primary invariance metric +
   secondary flip/token metrics defined.
3. **Sign-only and scale-matched are now distinct families** in
   `METHOD_COMPARISON_PROTOCOL.md` (separate grids, selections, comparisons).
4. **Efficiency reconciled to measured values:** ~56–57% mean zero-task-advantage
   across full seed-1/2/42 runs (seed means 0.569/0.558/0.573); 80% only as an
   explicitly labelled early logged observation; SFT **5,422 s/seed = 0.181
   s/step ≈ 1.51 h**.
5. **Scale wording corrected:** `±1` *increases* average magnitude vs
   `E|δ|=0.685`; scale-matched `±0.685` holds the first absolute moment fixed.
6. **Behavior summary corrected:** 15/40 rows `clean=NO`; of those **10** have
   `|λ|<0.5` plus a contradictory caveat, and 5 are `NO` only because `|λ|≥0.5`.
7. **`ENVIRONMENT.md`** declares Python ≥ 3.10 + SciPy/NumPy minimums (tested
   3.13.3 / scipy 1.15.2 / numpy 2.2.4); explicitly **not** a complete lock.

**Merge:** `origin/main` (`6306a15 Simplify framing specificity table`, touches
only `draft/project-overview.html`) merged into the branch with `--no-edit` (no
rebase of published commits); no conflicts. All checks rerun and pass post-merge.

## Constraints honored

No GPU job submitted · no model inference · no checkpoint evaluated · no frozen
suite opened or adapted to · frozen headline estimator (`core_exp_refactored.py`)
unmodified · default magnitude GRPO path unchanged · large artifacts/checkpoints
left untracked · small reviewable commits per task · branch pushed, **not**
merged.
