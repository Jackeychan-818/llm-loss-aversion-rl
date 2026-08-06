# Evaluating the full SFT checkpoint grid (behavioral trajectory)

How to produce the λ/η trajectory for the two **completed full matched-SFT runs**
(seed 1 and seed 2, cosine → 30,000). Training is done; this is the behavioral
measurement that training completion does *not* provide.

Nothing here is submitted automatically. Copy the block you want and `qsub` it.

## What each job does

`train/submit_eval_baseline_ckpt.pbs` (unchanged, already frozen) loads one LoRA
checkpoint, scores all 9,890 `test_goods` cases from both perspectives with
deterministic greedy decoding, then fits Model A NLS at structural link scale
T=1. Output lands in `baselines/Qwen-7B-SFT-qd-seed{N}-ckpt{S}/`.

Measured cost per checkpoint, from the pilot logs: 1,237 batches at ~1.15 it/s
≈ **18 min inference + a few minutes NLS ≈ 20–25 min**. Walltime is 2 h, so each
job has ample headroom.

## Full grid — 2,000 stride (the frozen grid; 30 jobs, ~11–13 GPU-hours)

```bash
cd $HOME/scratch/lambda-zero
for SEED in 1 2; do
  for CKPT in 2000 4000 6000 8000 10000 12000 14000 16000 18000 \
              20000 22000 24000 26000 28000 30000; do
    qsub -v METHOD=sft,SEED=$SEED,CKPT=$CKPT train/submit_eval_baseline_ckpt.pbs
  done
done
```

## Reduced grid — 4,000 stride (16 jobs, ~6 GPU-hours)

Use this if the SU budget is tight; it gives the shape of the trajectory and can
be infilled later with the remaining steps.

```bash
cd $HOME/scratch/lambda-zero
for SEED in 1 2; do
  for CKPT in 2000 6000 10000 14000 18000 22000 26000 30000; do
    qsub -v METHOD=sft,SEED=$SEED,CKPT=$CKPT train/submit_eval_baseline_ckpt.pbs
  done
done
```

Note that a partial grid is **not** eligible for the frozen selector, which
requires the complete 2k–30k @ 2k grid. It is fine for the trajectory plot.

## Run this first — the base model under a debias prompt (2 jobs, ~45 min)

Cheapest high-value measurement available, and it is a **prerequisite for
interpreting the grid**, not a side quest.

The `debias` treatment instructs the model to ignore status-quo and gain–loss
framing — it is the no-training alternative to everything this project does. But
`debias/` and `forced/` currently contain **only** `Qwen-7B-GRPO`. There is no
base-model measurement under either treatment, so the repository cannot answer
"does prompting alone do what training does?" If a prompt collapsed λ from 7.637
on its own, the entire grid would be measuring something far less interesting.

```bash
cd $HOME/scratch/lambda-zero
qsub -v TREATMENT=debias train/submit_eval_base_treatments.pbs   # the informative one
qsub -v TREATMENT=forced train/submit_eval_base_treatments.pbs   # completes the triple
```

Same weights, same 9,890 rows, same scorer, same Model A NLS at T=1 as
`baseline/Qwen-7B-Base-Local` (λ = 7.637, η = 1.007) — **the prompt is the only
thing that differs**. Results land in `debias/Qwen-7B-Base-Local/` and
`forced/Qwen-7B-Base-Local/`.

Smoke-test one first if you want to confirm the path end to end:

```bash
qsub -v TREATMENT=debias,LIMIT=20 train/submit_eval_base_treatments.pbs
```

## Budget

The two full SFT *training* runs consumed 2.8 GPU-hours and were charged ≈317 SU,
which naively scales to roughly **1,000–1,400 SU for the 30-job grid** against a
reported balance of ≈3,312 SU. Confirm the actual charging rate before committing
to the full grid — this is a material fraction of the remaining allocation and
the estimate is an extrapolation, not a measurement.

The two base-treatment jobs above are ≈2 checkpoint-equivalents (~45 min), so
running them first costs almost nothing and tells you whether the grid is worth
its full price.

## Monitoring

```bash
qstat -u jackeyc0
ls -d baselines/Qwen-7B-SFT-qd-seed*-ckpt*/Model_1 | wc -l   # 30 when complete
tail -f logs/eval_baseline_*.out
```

## After the jobs land

```bash
python eval/plot_sft_training_dynamics.py --refresh   # rescans the grid
python eval/plot_sft_training_dynamics.py --check
```

`--refresh` rescans `baselines/` and records what it found in the manifest. The
behavioral figure then gains the two full-run trajectories alongside the pilot.

## Guard rails already in place

- **`test_goods` is VALIDATION**, never a final test. The PBS script hard-refuses
  OOD-50 and the frozen-unused suite, and the frozen method-comparison and
  semantic suites are untouched.
- **Output root is `baselines/`, not `baseline/`.** The confirmatory selector
  globs `baseline/*`, so these evaluations cannot be swept into the frozen
  seed1→2,000 / seed2→6,000 GRPO selection.
- **The pilot's evaluations are quarantined** in `baselines/pilot6k/` (moved
  2026-08-06; see its `PROVENANCE.md`). They previously occupied the exact flat
  names these jobs write to, and `eval/run_qwen_local.py` resumes from any
  existing `loss_aversion_X/Y.json` in the output directory — so three of the
  thirty jobs would have printed `Already complete: 9890 / Nothing to do` and
  returned **pilot** numbers under a full-run label. Verify the quarantine held
  before submitting:

  ```bash
  ls -d baselines/Qwen-7B-SFT-qd-seed*-ckpt* 2>/dev/null   # must be empty
  ```

- **No selector run.** Producing the trajectory does not select a checkpoint.
  Running `eval/select_checkpoint.py` on this grid is a separate, deliberate act
  governed by `CAUSAL_BASELINE_PROTOCOL.md`, and it requires the
  `--provenance_note` recording that OOD-50 and the frozen-unused suite are
  already opened, so any SFT-vs-GRPO comparison is post-hoc unless a new
  untouched suite is frozen first.

## What the resulting curve will and will not show

It will show how λ, η, consistency, keep-both, trade-both and W evolve across
training for both full seeds, on validation data.

It will **not** establish that SFT beats GRPO, that any particular checkpoint is
the right one, or that the behavior generalizes — those need the frozen untouched
method-comparison suite. And the pilot's step-6,000 point is still not comparable
to the full runs' step-6,000 point: different cosine endpoints, different weights.
