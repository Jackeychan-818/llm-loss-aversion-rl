# Structural non-identification at Qwen-own-delta step 8,000 — forensic record

*Recorded 2026-07-21. Non-gating diagnostic. This does NOT alter the committed
estimate; the committed `alpha_37 = -1075.111` remains authoritative.*

## Why this file exists

The diagnostic wrapper (`eval/estimator_diagnostics.py`) reported the most
negative item fixed effect as `alpha_51 = -685.587`, while the committed CSV /
`PAPER_READINESS.md` records `alpha_37 = -1075.111`. These are **two different
NLS solutions of the same fit**, present in the same repository:

| | value read | provenance |
|---|---|---|
| Committed (git HEAD blob, = Desktop copy) | `alpha_37 = -1075.111` (global min) | commit `5e574c2` |
| Working tree (uncommitted, mtime 2026-07-21 21:37) | `alpha_51 = -685.587` (global min) | regeneration provenance unknown |

## Exact artifacts

- **Model / eval dir:** `baseline/Qwen-7B-GRPO-qd-ckpt8000`
- **File (relative):** `baseline/Qwen-7B-GRPO-qd-ckpt8000/Model_1/Qwen-7B-GRPO-qd-ckpt8000_NLS_estimation_T1(Model A).csv`
- **Estimator:** Model A (NLS), structural link scale T=1
- **Starting method:** not stamped in the CSV. The pipeline default in
  `eval/estimate_qwen_checkpoint.py` is `--starting ols`; the regeneration that
  produced the working-tree copy is **not attributed** (do not assume a
  different start was used — the objective is flat enough that the same start
  can reach either point).

| version | SHA-256 | alpha_37 | alpha_51 | lambda | eta |
|---|---|---|---|---|---|
| committed (HEAD) | `ea4e1182391bb2e97ec9802a35e17726b2595e156b4a7f97e10fafdec8100a6d` | **-1075.111** | -367.303 | 0.11099538 | 0.50408026 |
| working tree | `ad4774892ee75640b2e572382123fd0515148057623034ddca477b5fca4ec0f0` | -267.017 | **-685.587** | 0.11099538 | 0.50408026 |

Preserved copy of the working-tree (alternative) solution:
`results/estimator_diagnostics/qd_ckpt8000_ALT_SOLUTION_worktree.csv` (same SHA
as the working tree above).

## What differs between the two solutions

- **lambda and eta are identical to ~1e-9** (0.11099538 / 0.50408026).
- **Only two parameters move materially: `alpha_37` and `alpha_51`.** The huge
  negative magnitude is redistributed between items 37 and 51; the other 97
  alphas, all 8 betas, lambda and eta agree to < 1e-6 (the remaining ~46
  line-level diffs are last-digit numerical noise).
- **Both movers carry `Std. Err. = 0.00000000` and `Variance = 0.00000000` in
  both versions** — the covariance has no curvature in those directions.

## Diagnostic corroboration (working-tree solution)

From `eval/estimator_diagnostics.py` on this checkpoint
(`results/estimator_diagnostics/qd_ckpt8000.json`):

- `cond(Jacobian) = 6.085e16` (numerically singular; ≈ 1/machine-epsilon)
- covariance not invertible → `cond(pcov) = nan`, some SEs non-finite
- utility grid range `[0, 18.375]` (utilities collapse to 0 for the flat items)
- **rank preservation vs the matched base is still Spearman 0.834 (alpha) /
  0.839 (utility)** — the pathology is *localized* to a couple of goods, not a
  collapse of the whole utility structure.

Contrast — a healthy checkpoint, `seed1-ckpt2000`:
`cond(Jacobian) = 97.3`, SEs finite & positive, alpha range `[-2.17, 1.48]`.

## Interpretation (for the paper)

Low lambda at step 8,000 remains **behaviorally meaningful** — the
cross-perspective asymmetry is pinned (lambda, eta are invariant across the two
optima). What is **not** identified is a small number of *individual structural
utilities*: items 37 and 51 sit on a flat direction of the NLS objective, so
re-optimization moves them by hundreds while nothing behavioral changes. This is
a clean, reportable instance of numerical non-identification, and both solutions
should be reported together as evidence rather than either being presented as
"the" utility of those goods.

## Actions

- **Do NOT replace `-1075.111`.** It is the committed, authoritative estimate.
- The working tree currently holds an **uncommitted modification** to this
  claim-carrying CSV. Decide explicitly whether to (a) `git checkout` it back to
  the committed version so the tracked file matches HEAD (the alternative
  solution is already preserved above), or (b) keep investigating its origin
  first. Do not commit the modified CSV silently.
