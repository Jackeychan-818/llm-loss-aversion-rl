# AAAI-27 Submission Workboard

*Prepared July 16, 2026 (Asia/Singapore).*

AAAI deadlines are Anywhere on Earth (UTC-12). In Singapore time:

| Milestone | Official deadline | Singapore equivalent |
|---|---|---|
| Abstract registration | July 21, 11:59 PM UTC-12 | July 22, 7:59 PM SGT |
| Full paper | July 28, 11:59 PM UTC-12 | July 29, 7:59 PM SGT |
| Supplement/code/data | July 31, 11:59 PM UTC-12 | August 1, 7:59 PM SGT |

Use internal deadlines at least 24 hours earlier.

## Submission-Critical Status

| Workstream | Status | Paper consequence |
|---|---|---|
| Anonymous author-kit manuscript | **Started** | `main.tex` now exists with title, abstract, methods, tables, limitations, and references |
| Complete OpenReview author list | **Blocked on authors** | Registration cannot safely be completed without the final author list |
| Exact matched local base | **Done on origin/main** | Use ID `lambda=7.637`, not historical `11.75` |
| Ownership-free pseudo-utility validation | **Done on origin/main** | 71.1% validation agreement; 85.5% at high magnitude |
| Frozen checkpoint rule | **Done on origin/main** | Joint `sqrt(lambda^2+eta^2)` rule with consistency eligibility |
| Exploratory seed 42 | **Partial** | Existing 8k selection uses seven evaluated checkpoints; ten grid top-up evaluations remain |
| Confirmatory seeds 1 and 2 | **Pending** | Required for seed-level claims and variation |
| Seed-aware 40-evaluation sweep | **Pending** | 10 seed-42 top-ups + 15 checkpoints for each new seed |
| Matched Qwen-own SFT | **Pending, P0** | Required to establish whether GRPO adds value over supervised binary targets |
| Pair bootstrap / leave-one-good-out / recovery | **Pending, P0** | Current structural errors are conditional and incomplete |
| GSM8K | **Done on origin/main** | Base 86.88%, tuned 85.90%, paired delta -0.99 pp |
| IFEval | **Pending, P1** | Needed for instruction-following retention |
| Derived result manifests | **Partial** | Derived estimates/checksums exist on origin/main |
| Raw predictions and adapter archive | **Pending, P0** | Current repository is not independently reproducible from raw evidence |
| Automated paper tables | **Started** | Registry and deterministic table writer now exist |
| Anonymous supplement ZIP | **Pending, P0** | Manifest created; actual staged package still needed |

## Day-by-Day Critical Path

### July 16--17

1. Reconcile the dirty/behind worktree with `origin/main` in a safe branch or
   clean worktree; preserve all current edits.
2. Confirm final title, complete author list, author order, and track.
3. Register the substantive OpenReview abstract.
4. Launch seed-42 checkpoint-grid top-up evaluations and confirm seed-aware
   output directories/manifests.
5. Launch the matched SFT baseline.
6. Start raw artifact collection into an anonymous staging directory.

### July 18--20

1. Launch or continue seeds 1 and 2 to the fixed 30k endpoint.
2. Implement and validate pair-cluster bootstrap, leave-one-good-out
   structural robustness, and estimator-recovery simulation.
3. Implement IFEval with the exact base/tuned paired harness.
4. Freeze paper figure definitions and table schema.
5. Populate environment/version and compute manifests.

### July 21

1. Final author/title/abstract check.
2. Submit the real OpenReview abstract before the internal cutoff.
3. Verify confirmation email.

### July 22--25

1. Complete checkpoint selection mechanically for every seed.
2. Open the frozen OOD suite only after selecting each checkpoint.
3. Evaluate SFT, every selected GRPO seed, and consensus on the same suites.
4. Fill all seed, SFT, preference-agreement, robust-inference, GSM8K, and
   IFEval table cells.
5. Regenerate tables and figures from tracked JSON artifacts.

### July 26--27

1. Compress to seven technical pages without formatting tricks.
2. Complete references, limitations, ethical statement if retained, and the
   reproducibility checklist.
3. Run anonymity, font, metadata, overflow, page-count, and source checks.
4. Have every author read the exact submission PDF.

### July 28

1. Upload the anonymous paper PDF and separate checklist.
2. Re-download and verify the portal copy.
3. Freeze the paper-facing result registry.

### July 29--31

1. Finish the anonymous code/data ZIP and optional technical supplement.
2. Remove identity-bearing paths, repository remotes, comments, logs, and
   metadata.
3. Verify every primary table from the staged package on a clean environment.
4. Upload by the internal cutoff, at least 24 hours before the official
   supplementary deadline.

## Minimum Competitive Experiment Matrix

| Model | ID validation | Frozen OOD | Robust inference | GSM8K | IFEval |
|---|---:|---:|---:|---:|---:|
| Exact local base | done | done | pending | done | pending |
| Qwen-own SFT | pending | pending | pending | pending | pending |
| Qwen-own GRPO seed 1 | pending | pending | pending | pending | pending |
| Qwen-own GRPO seed 2 | pending | pending | pending | pending | pending |
| Exploratory seed 42 | partial | done | pending | done | pending |
| Consensus GRPO ablation | done | done | pending | optional | optional |

Sign-only GRPO is optional only after the matched SFT, seeds, robust inference,
IFEval, and reproducibility package are secure.
