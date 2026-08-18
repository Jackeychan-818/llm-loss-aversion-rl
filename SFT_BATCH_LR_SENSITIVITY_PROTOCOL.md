# SFT batch / learning-rate sensitivity protocol

*Frozen 2026-08-18, before any training job of this experiment was submitted.
Branch `exp/sft-batch-lr-sensitivity`. This document is the pre-registration:
the design, the arithmetic and the selection rule below are fixed in advance and
applied mechanically.*

## 0. Scientific status — read first

This is an **EXPLORATORY, POST-HOC OPTIMIZATION ABLATION**. It asks a narrow
engineering question:

> Does the matched-SFT baseline's effective batch size of 1 produce noisy
> training loss, heavy-tailed pre-clipping gradient norms, and unstable
> behavioural trajectories — and does a larger effective batch reduce that?

It is **not** a method comparison and **not** a re-selection.

- It does **not** modify, replace or re-rank the frozen matched-SFT selections
  (seed 1 → step 4,000; seed 2 → step 6,000). Those stand exactly as recorded in
  `results/checkpoint_selection/` under `CAUSAL_BASELINE_PROTOCOL.md`.
- It does **not** license any SFT-versus-GRPO claim. `METHOD_COMPARISON_PROTOCOL.md`
  governs that, and a comparison there requires the untouched suite.
- If an optimized SFT configuration is later proposed for a confirmatory
  comparison, it needs a **newly frozen protocol and a newly frozen untouched
  evaluation set**. The existing `data/method_comparison/` suite may **not** be
  used for it: this experiment's selections are made on `test_goods`, so any
  later comparison against a suite chosen before would be post-hoc.

### Suites that stay closed

`data/method_comparison/`, `data/frozen_unused_test_goods.json`, the OOD-50 set,
the semantic-counterbalancing component, the framing suite, and GSM8K are **not
read, inspected or evaluated** anywhere in this experiment.

`data/test_goods.json` is used **only** as the already-open validation set. It
informed reward construction and prior checkpoint selection, so every number it
produces here is a validation estimate, never final-test performance.

`eval/core_exp_refactored.py` is not edited.

## 1. What is held fixed

Every cell shares these, so that the only things varying are the gradient
accumulation boundary and the step size:

| quantity | value |
|---|---|
| base model | `models/Qwen2.5-7B-Instruct` (exact local weights) |
| training data | `data/remaining_goods.json` + `data/deltas/delta_qwen_base.json` |
| targets | `reward_functions.rational_choice(perspective, delta)` |
| loss | completion-only cross-entropy; prompt tokens masked with `-100` |
| LoRA | r=16, α=32, dropout=0.05, {q,k,v,o,gate,up,down}_proj, bias=none |
| precision | BF16 |
| `per_device_train_batch_size` | **1**, always |
| effective batch | realised **only** through `gradient_accumulation_steps` |
| `max_grad_norm` | **0.1**, unchanged for this whole experiment |
| schedule | cosine, `warmup_ratio` 0.05, horizon = that run's own step count |
| logging | every optimizer update (`logging_steps: 1`) |
| GPU | one |

Pinning `per_device_train_batch_size` to 1 and moving the effective batch purely
into accumulation means the per-prompt forward/backward arithmetic is identical
across cells. A batch-64 cell performs the same 64 forward/backward passes as
sixty-four batch-1 steps; only the optimizer update boundary moves. Training
cost therefore tracks **prompt exposure**, not optimizer steps.

## 2. Comparison axis: prompt exposure, never optimizer steps

    optimizer_steps = prompt_exposure / effective_batch     (must divide exactly)

Inexact division is a hard failure (`train/sft_sensitivity_plan.py::optimizer_steps`),
because a cell that cannot consume its exposure in whole steps would truncate or
over-run the axis the whole experiment rests on.

The exposures are chosen so that all four batches divide exactly:

| effective batch | Phase A/B steps (6,016 prompts) | full-run steps (30,016 prompts) |
|---:|---:|---:|
| 1 | 6,016 | — |
| 16 | 376 | 1,876 |
| 32 | 188 | 938 |
| 64 | 94 | 469 |

All trajectories are plotted and tabulated against **prompts seen**, with a
common smoothing window measured in prompt exposures.

## 3. Controlled data ordering

For each seed, **one** deterministic permutation of the 98,900 training examples
is created, and every cell of that seed consumes a **prefix** of that single
ordering. Cells therefore see the same examples in the same order regardless of
batch, learning rate or horizon.

Records are first sorted into canonical `(case_id, perspective)` order, so the
permutation cannot inherit the record builder's emission order, then permuted
once by `numpy.random.Generator(PCG64(seed))`.

Example identifier: `"{case_id}:{perspective}"`. The order hash is SHA-256 over
the newline-joined identifiers of the consumed prefix.

**Frozen order hashes** (`results/sft_sensitivity/order_hashes.json`, computed
2026-08-18 on the real 98,900-example pool):

| seed | first 6,016 prompts (Phase A/B) | first 30,016 prompts (full runs) |
|---:|---|---|
| 1 | `3d1c9403…023acdc7` | `5852aa46…17f7b188` |
| 2 | `38061828…1139a567` | `043e0b5b…ada4fef4` |
| 3 | `d5c5488d…6c009e816b48` | `a7d81169…81a4f881` |

Every run recomputes its own order hash and **aborts** if it does not match the
value frozen here. No example may be consumed twice within a run; exposure never
exceeds the pool, so no run repeats a prompt.

## 4. Phase A — fixed-learning-rate batch sweep

- effective batches {1, 16, 32, 64} × seeds {1, 2, 3} = **12 cells**
- learning rate **1e-6** (the frozen baseline's value)
- exposure exactly **6,016 unique prompts**
- checkpoints at **2,048 / 4,096 / 6,016** prompts; the endpoint is always saved

## 5. Phase B — learning-rate sweep

One large batch from {16, 32, 64} is selected by the rule in §6, then seeds
{1, 2, 3} are run at **3e-7, 1e-6, 3e-6, 1e-5**, with the same exposure, prompt
sequence, logging, checkpoints and clipping.

The **1e-6 column reuses the corresponding Phase-A cells**: at the selected
batch they are identical in exposure, order, horizon and every hyper-parameter,
so re-running them would spend GPU hours reproducing bit-identical work. That
leaves **9 new cells**.

## 6. Frozen selection rule

**Selection is never made on training loss.**

For every complete cell, compute **completion-only cross-entropy on `test_goods`
validation against the frozen rational Yes/No target**. This is recoverable on
CPU from the teacher-forced Yes/No probabilities the standard evaluation already
writes, so it needs no extra GPU pass.

1. Select the large batch with the **lowest mean endpoint validation
   cross-entropy across seeds 1–3**.
2. If means are exactly tied at stored precision, select the **smaller** batch.
3. After the LR sweep, select the learning rate by the same rule.
4. Behavioural and stability metrics (λ, η, d, consistency, keep-both,
   trade-both, W, gradient-norm tails, loss roughness) are **mandatory secondary
   outcomes**. They are reported in full and they may **not** silently override
   this rule.

A cell is excluded **only** for a predeclared technical failure: non-finite
values, incomplete exposure, an incorrect data/order hash, missing artefacts, or
a manifest mismatch.

## 7. Full sensitivity runs

After Phase B, the selected batch/LR pair is rerun **from scratch — never from a
pilot checkpoint** — for seeds 1, 2, 3 at exactly **30,016 unique prompts**, with
a **fresh cosine schedule whose horizon is the full run**.

> Pilot and full-run weights at the same prompt exposure are **not
> interchangeable**, because their cosine horizons differ. A 6,016-prompt pilot
> decays to zero by 6,016; a full run at 6,016 prompts is one fifth of the way
> down a 30,016-prompt cosine. The schedule horizon is encoded in every output
> directory name (`h6016` / `h30016`) so the two can never be confused or mixed.

Checkpoints every 2,048 prompt exposures plus the endpoint (15 per run). All
outputs live under `checkpoints/sft_sensitivity/` and
`results/sft_sensitivity/`, entirely separate from the frozen matched-SFT
directories.

## 8. Required diagnostics

Per pilot cell and per full-run checkpoint:

- prompts seen and optimizer updates;
- training-loss distribution; held-out validation cross-entropy;
- pre-clipping gradient-norm **median, P90, P99, maximum, and fraction
  exceeding 0.1** — never the arithmetic mean of a heavy-tailed quantity;
- non-finite record count;
- learning-rate schedule verification against the configured cosine;
- λ and η **jointly**, Model A, structural link scale **T=1**;
- `d = sqrt(λ² + η²)`; consistency; keep-both; trade-both; W;
- per-seed values, mean and standard deviation across seeds;
- pair-clustered uncertainty for prompt-level validation metrics where applicable.

Loss roughness and checkpoint-to-checkpoint behavioural drift are reported and
labelled **secondary / descriptive**.

### Vocabulary discipline

A large pre-clipping gradient norm is **not** "gradient explosion". These five
are reported as distinct quantities and never conflated:

| term | what it means here |
|---|---|
| pre-clipping norm | the norm HF logs *before* `clip_grad_norm_` applies |
| clipping frequency | fraction of optimizer updates whose norm exceeded 0.1 |
| numerical failure | a non-finite loss, gradient or parameter |
| optimization noise | step-to-step variance of the logged training loss |
| behavioural drift | checkpoint-to-checkpoint movement of λ, η or the choice rates |

## 9. Provenance

Every run records: git commit; full command and resolved config; config, data and
order hashes; model and tokenizer identity; seed and data seed; GPU model; CUDA,
Python, PyTorch, Transformers, PEFT and Accelerate versions; PBS job ID;
effective batch, accumulation, optimizer updates and exact prompt exposure;
checkpoint and log hashes; start/end time and runtime.

Output and log directories encode phase, batch, LR, seed and schedule horizon:

    checkpoints/sft_sensitivity/sft_sens_ph{A|B|full}_eb{N}_lr{L}_seed{S}_h{EXPOSURE}

`${PBS_JOBID}` is **not** used in a `#PBS -o/-e` path — PBS does not expand it
there, so concurrent submissions would overwrite one literal filename. Scheduler
logs stay at their default job-specific paths and the job additionally tees a
per-cell log whose name is collision-free by construction.

`results/sft_sensitivity/EXECUTION_MANIFEST.md` is **append-only**: every
submitted job is recorded with its cell name, job ID and final status.

## 10. Safety and stopping

- A failed run is **not** continued with changed hyper-parameters.
- Abort only on technical failure: OOM, non-finite training, corrupted
  artefacts, incorrect exposure or order hash, or provenance mismatch.
- If resources are insufficient, everything is prepared and validated, a precise
  GPU-hour/SU estimate is produced, and **no job is submitted**.
- Phase A, Phase B and the full runs are reported **separately**.
- Conclusions are stated for three seeds and not overstated.

## 11. Reproduction

```bash
python train/test_sft_sensitivity.py          # CPU: arithmetic, ordering, manifest, collisions
python train/sft_sensitivity_plan.py          # the frozen exposure/step grid
python train/sft_sensitivity_cost.py          # measured GPU-hour / SU estimate
python train/sft_sensitivity_train.py --cell-list          # every planned cell
python train/sft_sensitivity_train.py --cell <name> --validate   # CPU dry run, no GPU
```
