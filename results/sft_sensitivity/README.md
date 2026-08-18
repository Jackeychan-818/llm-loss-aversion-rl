# SFT batch / learning-rate sensitivity — prepared, validated, NOT submitted

**Status: no training job has been submitted.** Everything below is prepared and
CPU-validated. Submission is blocked on a budget decision, not on readiness.

Protocol: [`SFT_BATCH_LR_SENSITIVITY_PROTOCOL.md`](../../SFT_BATCH_LR_SENSITIVITY_PROTOCOL.md)
(frozen and committed at `4798e89`, before any job).

## Why nothing was submitted

The remaining allocation cannot cover the experiment as specified.

| | GPU-hours | SU | vs balance |
|---|---:|---:|---:|
| Phase A, endpoint evaluation only | 7.1–8.2 | 452–525 | 27% |
| Phase A, all 3 exposure checkpoints | 15.0–17.4 | 957–1,112 | 58% |
| Phase A + Phase B, endpoint only | 12.4–14.4 | 791–919 | 48% |
| Phase A + Phase B, all checkpoints | 26.2–30.4 | 1,675–1,947 | **101%** |
| **Everything, incl. full 30,016-prompt runs** | **44.9–52.1** | **2,871–3,337** | **173%** |

Balance at estimate: **1,933.4 SU = 30.2 GPU-hours** at **64 SU/GPU-hour**
(`myprojects`, 2026-08-18). Rates are measured from this repository's own
artefacts, not assumed — see `cost_estimate.json` for the source of each.

**The cost driver is evaluation, not training.** Training the entire program is
~10 GPU-hours; the 108 `test_goods` checkpoint evaluations are roughly three
times that. Validation cross-entropy, which the frozen selection rule needs, is
recoverable on CPU from the teacher-forced probabilities the standard evaluation
already writes, so it adds no GPU cost.

## What is ready

| file | role |
|---|---|
| `../../SFT_BATCH_LR_SENSITIVITY_PROTOCOL.md` | frozen pre-registration |
| `../../train/sft_sensitivity_plan.py` | exposure arithmetic, ordering, cell identity |
| `../../train/sft_sensitivity_train.py` | training driver (`--cell-list`, `--validate`) |
| `../../train/sft_sensitivity_cost.py` | measured GPU-hour / SU estimate |
| `../../train/configs/sft_sensitivity_base.yaml` | fixed settings, verified against the plan |
| `../../train/submit_sft_sensitivity.pbs` | one cell per job, collision-safe logs |
| `../../train/test_sft_sensitivity.py` | 224 CPU checks |
| `order_hashes.json` | frozen data ordering, seeds 1–3 |
| `cost_estimate.json` | machine-readable estimate |
| `EXECUTION_MANIFEST.md` | append-only; currently CPU validations only |

## CPU validation already performed

- **224 checks pass** — exposure arithmetic, deterministic ordering, manifest
  correctness, output-directory collision protection.
- **Order hashes computed on the real 98,900-example pool** and frozen for seeds
  1–3; the driver aborts on mismatch before touching a GPU.
- **Ordering verified identical across cells.** Every seed-1 cell at 6,016
  prompts — batch 1, 16, 64, Phase A and Phase B, every learning rate — reports
  the same order hash `3d1c9403…`, while the 30,016-prompt full run correctly
  extends the same permutation to `5852aa46…`.
- **End-to-end dry run** of six representative cells via `--validate`: dataset,
  ordering, exposure arithmetic and manifest all build with no GPU.

## Exposure grid

| effective batch | Phase A/B steps (6,016 prompts) | full-run steps (30,016 prompts) |
|---:|---:|---:|
| 1 | 6,016 | — |
| 16 | 376 | 1,876 |
| 32 | 188 | 938 |
| 64 | 94 | 469 |

Every batch checkpoints at the **same prompt exposures**, so trajectories are
comparable point-for-point rather than only at the endpoint.

## Reproduce

```bash
python train/test_sft_sensitivity.py                     # 224 CPU checks
python train/sft_sensitivity_plan.py                     # the frozen grid
python train/sft_sensitivity_cost.py                     # measured estimate
python train/sft_sensitivity_train.py --cell-list        # every planned cell
python train/sft_sensitivity_train.py \
    --cell sft_sens_phA_eb32_lr1e-6_seed1_h6016 --validate   # CPU dry run
```

## Audit — nothing frozen was touched

- No GPU job submitted; no inference run; no checkpoint evaluated.
- `data/method_comparison/`, `data/frozen_unused_test_goods.json`, OOD-50, the
  semantic and framing suites, and GSM8K were **not read, inspected or
  evaluated**. The only mention of them in the new code is the manifest's own
  `suites_not_read` declaration.
- `eval/core_exp_refactored.py` unmodified.
- `train/configs/qwen25_7b_sft_qwen_delta.yaml` unmodified — the frozen
  matched-SFT baseline config is untouched.
- `results/checkpoint_selection/` unmodified; the frozen matched-SFT selections
  (seed 1 → 4,000, seed 2 → 6,000) stand.
- No existing checkpoint, result or config was modified, reset or overwritten.
  All new outputs live under `checkpoints/sft_sensitivity/` and
  `results/sft_sensitivity/`.
