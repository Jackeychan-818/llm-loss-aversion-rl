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

---

## Amendment 1 — Phase-A endpoint-only evaluation (2026-08-18)

**Recorded before any GPU job of this experiment was submitted. No experimental
outcome of any kind has been observed: no training cell has run, no checkpoint
exists, and no validation or behavioural number has been computed. This
amendment is therefore driven purely by cost, and cannot have been influenced by
results.**

### Status of the original text

Endpoint-only evaluation is arguably already permitted:

- **§4** requires checkpoints to be **saved** at 2,048 / 4,096 / 6,016 prompts.
  It does not require them to be evaluated. Saving is unchanged.
- **§6** defines the selection rule on **endpoint** validation cross-entropy per
  cell. Endpoint-only satisfies it exactly as written.
- **§8** names the unit as "per pilot **cell** and per full-run **checkpoint**".
  For pilots the unit is the cell, so endpoint diagnostics satisfy it literally.

However `results/sft_sensitivity/README.md` priced a "Phase A, all 3 exposure
checkpoints" scenario, which implies a broader intent than the text requires.
Rather than rely on the narrower reading, the scope is fixed here explicitly.

### Amendment

For **Phase A only**, the required behavioural diagnostics of §8 (λ, η, d,
consistency, keep-both, trade-both, W) are computed at the **endpoint exposure
(6,016 prompts) only**.

1. **Endpoint validation cross-entropy is sufficient for the frozen
   batch-selection rule.** §6 already selects on the endpoint mean across seeds
   1–3. Evaluating the 2,048- and 4,096-prompt checkpoints would add trajectory
   detail but could not change the selection, because the rule does not read
   them.
2. **The intermediate checkpoints are retained, not discarded.** Both the
   2,048- and 4,096-exposure checkpoints of every cell are saved and preserved
   exactly as §4 requires. They remain available for a later trajectory
   evaluation, which would be a separate, separately costed submission.
3. **Training-side diagnostics are unaffected.** Loss distribution, pre-clipping
   gradient-norm median/P90/P99/max, clipping fraction, non-finite counts,
   learning-rate schedule verification, runtime and loss roughness all come from
   the training logs and are reported in full for every cell. The central
   question — whether a larger effective batch reduces gradient noise — is
   answered from these, at no evaluation cost.

### Authorized scope

| | |
|---|---|
| cells | effective batch {1, 16, 32, 64} × seeds {1, 2, 3} = 12 |
| learning rate | 1e-6 |
| exposure | exactly 6,016 unique prompts per cell |
| evaluation | endpoint (6,016) only, `test_goods` validation, 12 evaluations |
| **ceiling** | **8.2 GPU-hours / 525 SU** |

**Not authorized under this amendment:** Phase B, intermediate-checkpoint
evaluations, and the full 30,016-prompt runs. Each requires a separate decision.

Work stops after Phase A and reports before any further submission.

All other provisions of this protocol — the ordering guarantee, the exposure
arithmetic, the selection rule and its tie-break, the vocabulary discipline of
§8, and the closed-suite list of §0 — are unchanged.

---

## Amendment 2 — Phase-B learning-rate search (2026-08-19)

**Committed before any Phase-B GPU job. Phase-A outcomes ARE known and are cited
below; this amendment is therefore not outcome-blind, and its purpose is to fix
the Phase-B decision tree mechanically so that no Phase-B result can influence
how Phase B is decided.**

Supersedes §5 of the original protocol for Phase B.

### Scope

Effective batches **16, 32, 64**. **Effective batch 1 is abandoned** as a
candidate — a decision taken on Phase-A evidence and recorded here. The
consequence is stated plainly: Phase B can conclude which of 16/32/64 is best at
6,016 prompts, and **cannot** conclude that batching beats batch 1, since batch 1
is not tuned.

Fixed for every cell: 6,016 prompts, the same deterministic per-seed ordering,
`per_device_train_batch_size = 1` with accumulation, `max_grad_norm = 0.1`,
cosine with 5% warmup over each cell's own horizon, endpoint evaluation at
6,016 only.

### Existing results are reused

Phase A already ran **lr 1e-6 at all three seeds for every batch**. Those cells
are reused as-is and are **not rerun**, including not rerun merely to obtain the
early snapshots introduced below. A batch whose winner turns out to be 1e-6 will
therefore have no 128-prompt snapshots; that is accepted.

### Stage 1 — two-seed screening (12 new cells)

lr **3e-6** and **1e-5**, seeds 1 and 2, for each of the three batches. Compare
1e-6 / 3e-6 / 1e-5 by mean endpoint validation CE across seeds 1–2.

Incumbent two-seed means from Phase A: eb16 0.59465, eb32 0.66220, eb64 2.06115.

### Boundary rules

- **3e-6 wins** → provisionally select it.
- **1e-6 wins** → run **3e-7** for that batch, seeds 1–2, then select again.
- **1e-5 wins** → **stop** for a cost and safety review before considering 3e-5.
- No setting is promoted on one seed alone.

### Seed-disagreement rule (frozen)

Add seed 3 **for both competing LRs** before deciding when either holds:

1. seeds 1 and 2 rank the top two LRs differently; or
2. the difference between their two-seed mean CEs is below the Phase-A noise
   floor for that batch — **eb16 0.0019, eb32 0.0050, eb64 0.0097** (the
   across-seed sd measured in Phase A, frozen here so the threshold cannot be
   chosen after seeing Phase-B data).

Seed 3 is run for **both** top candidates, never only the provisional winner:
adding a third seed to one side alone would compare a three-seed mean against a
two-seed mean. Where a candidate is 1e-6, its seed-3 cell already exists and is
reused.

### Stage 2 — confirmation

Seed 3 for each batch's provisional winner (reused if that winner is 1e-6, or if
the seed-disagreement rule already produced it). The global winner is then the
lowest mean validation CE across **all three seeds** among the three batch
winners — every candidate compared on equal seed counts.

### Checkpoints

Per new run: **adapter-only** snapshots at prompt exposures 128, 256, … 2,048
(16 of them; 8/4/2 optimizer steps at batch 16/32/64, exact), plus **one full
resumable checkpoint at 6,016**. Snapshots are taken by a dedicated callback,
not by enabling `save_only_model` globally, so the endpoint stays resumable.

Every snapshot records exposure, optimizer step, LR, batch, seed, base-model
identity, git commit, and `resume_supported: false`. They are preserved and
**not evaluated** unless separately authorized. Phase-A intermediate checkpoints
are likewise not evaluated.

### Diagnostics (reported, never used for selection)

λ, η, `d`, consistency, keep-both, trade-both, W; pre-clipping gradient
quantiles and clipping fraction; the **clipping coefficient** `min(1, 0.1/‖g‖)`;
the **LoRA parameter-update norm**

    ‖Δθ_t‖₂ = sqrt( Σ_j ‖θ_{j,t+1} − θ_{j,t}‖₂² )

over all trainable LoRA parameters, together with the **relative** update norm
`‖Δθ‖ / ‖θ‖`; loss dynamics and seed variation.

The update norm is measured between `on_pre_optimizer_step` and
`on_optimizer_step`, so accumulation microsteps cannot be counted as updates.
It is worth the instrumentation because universal clipping means the
pre-clipping gradient norm no longer reveals actual parameter movement — and
because `adamw_torch_fused` applies `lr·m̂/(√v̂+ε)`, which is already scale-free,
so a constant rescaling of the gradient largely cancels. Only the realised
movement settles what clipping is doing.

### Budget — staged, not a single ceiling

| gate | limit |
|---|---:|
| Stage-1 screening authorization | **500 SU** (projected 481) |
| stop and report after screening | mandatory |
| conditional refinements + confirmation | require a second decision |
| **absolute cumulative ceiling** | **850 SU** |

Cost is recalculated before each stage. If applying the seed-disagreement rule
would exceed 850 SU, the program **stops** rather than evaluating only one side
of a comparison.

Worst case fits: screening 481 + at most 120 SU per batch afterwards (3e-7 pair
at 80, plus a single new seed-3 cell at 40) = 841 SU.

### Non-binding prediction (pre-registered, no effect on selection)

Heuristic only. At fixed exposure the total optimization path length is
≈ ½·peak_lr·N with N = 6016/batch, so matching path length predicts optimal LR
scaling **linearly** with batch. Anchoring eb16 at 1e-6, that predicts eb32 near
**2e-6** and eb64 near **4e-6**; on the available grid it maps approximately to
**eb16 → 1e-6, eb32 → 3e-6, eb64 → 3e-6**.

If path length is the whole story, the three tuned winners should be **within
noise** of each other, defined in advance as **maximum pairwise difference in
three-seed mean validation CE ≤ 0.010** (the largest Phase-A across-seed sd,
eb64's 0.0097, rounded up).

This hypothesis is recorded so the sweep is a test rather than a fit. It has
**no bearing on the frozen selection rule** and may not be used to override it.

### Standing constraints

Post-hoc and exploratory. No frozen or untouched suite is opened. The result is
optimal **only** for the 6,016-prompt cosine schedule at clipping threshold 0.1
— not universally. Transferring the winner to the 30,016-prompt schedule
requires separate confirmation.

---

## Amendment 3 — off-grid 1e-4 bracketing probe (2026-08-19)

**Recorded before the GPU job. Stage-1 outcomes ARE known (1e-5 won every
batch), so this is not outcome-blind; it is the cost-and-safety review that
Amendment 2 requires when 1e-5 wins.**

### Why a deviation is needed

Amendment 2 froze the LR grid at {3e-7, 1e-6, 3e-6, 1e-5} and directed that a
1e-5 win triggers a stop "before considering **3e-5**". This probe goes to
**1e-4**, skipping 3e-5. That is a deviation from the frozen tree and is
recorded as one.

Rationale: Stage 1 found validation CE **monotone in LR at every batch**, so the
optimum lies at or beyond the boundary and its distance is unknown. Stepping to
3e-5 risks landing on the boundary again and spending ~240 SU without bracketing
anything. A full decade brackets the turnover instead — if 1e-4 is worse than
1e-5 the optimum is trapped between them and can be bisected; if it is still
better, the grid was off by more than an order of magnitude. 1e-4 is also a
conventional LoRA learning rate; the original 1e-6 was inherited from the GRPO
config rather than chosen for SFT.

### Scope

| | |
|---|---|
| cells | eb16 and eb32, **seed 1 only**, lr 1e-4 |
| exposure | 6,016 prompts, same deterministic ordering, clipping 0.1 |
| evaluation | endpoint only |
| cost | **79 SU** (measured: 18.1 min train + 19.1 min eval per cell) |
| Phase-B cumulative after this | 554 of the 850 ceiling |

eb64 is omitted: two cells are enough to locate a turnover, and eb64 was the
weakest batch at every LR tested.

### This probe CANNOT promote anything

Amendment 2 states that no setting is promoted on one seed alone, and that
constraint is unchanged. This run may only **bracket** — establish whether the
CE curve turns over between 1e-5 and 1e-4. It may **not** select 1e-4, change
the batch winner, or feed the global selection. Promoting any LR found here
requires seeds 2 and 3 for the competing candidates under the Amendment-2 rules
(+40 SU per cell per seed).

### Predeclared failure handling

A non-finite loss, gradient or parameter at 1e-4 is a **technical failure** under
§6 of the original protocol: the cell is excluded and is **not** retried with
altered hyper-parameters. Divergence is a plausible outcome at this learning rate
and is itself an informative bracket — it would place the usable ceiling below
1e-4.

All other standing constraints are unchanged: post-hoc, no frozen or untouched
suite opened, Phase-A and early snapshots not evaluated, and the result remains
conditional on the 6,016-prompt schedule and clipping threshold 0.1.
