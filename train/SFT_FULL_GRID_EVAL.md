# Full SFT checkpoint-grid evaluation

## Status: complete (August 7, 2026)

Both completed matched-SFT runs were evaluated at every frozen checkpoint:

- seeds 1 and 2;
- steps 2,000 through 30,000 at a 2,000-step stride;
- 30/30 evaluations, each with 9,890 paired `test_goods` validation cases;
- deterministic teacher-forced Yes/No scoring and Model A NLS at structural
  link scale T=1;
- zero recorded parse failures.

The unchanged selector then chose seed 1 at step 4,000 and seed 2 at step
6,000. These are validation selections, not final-test estimates. Any
SFT-versus-GRPO winner claim still requires the untouched method-comparison
suite under `METHOD_COMPARISON_PROTOCOL.md`.

Results and checks:

- `results/sft_grid_verification.json`
- `results/checkpoint_selection/Qwen-7B-SFT-qd-seed{1,2}.json`
- `results/training_dynamics/sft/sft_behavioral_trajectory.{csv,png}`
- `results/training_dynamics/sft/sft_training_summary.{json,md}`

The historical evaluation did not emit an eval-time manifest. The verification
snapshot establishes expected-adapter hashes and complete evaluation artifacts,
but it cannot cryptographically prove which adapter generated each prediction
file. Raw predictions also remain untracked. Future verifier runs hash both raw
perspective files; future evaluation infrastructure should bind adapter, data,
code, and output hashes at inference time.

## Important separation from the pilot

The earlier seed-1 pilot used a cosine schedule ending at 6,000. The full runs
used a cosine schedule ending at 30,000, so their step-6,000 weights are not
interchangeable. Pilot evaluations remain quarantined in `baselines/pilot6k/`
and are excluded structurally from the flat full-grid scan.

## Rerunning a checkpoint (only if recovery is required)

`train/submit_eval_baseline_ckpt.pbs` evaluates one checkpoint. It is restricted
to `test_goods` validation and refuses the OOD/frozen suites.

```bash
cd "$HOME/scratch/lambda-zero"
qsub -v METHOD=sft,SEED=1,CKPT=4000 train/submit_eval_baseline_ckpt.pbs
```

Do not launch the full 30-job grid again: the outputs already exist and the
runner resumes existing prediction files. If recovery is genuinely needed,
first identify the exact failed cell and preserve the existing artifacts.

Monitor the stable per-model logs rather than a wildcard PBS directive path:

```bash
qstat -u jackeyc0
tail -f logs/eval_Qwen-7B-SFT-qd-seed1-ckpt4000.log
tail -f logs/estimate_Qwen-7B-SFT-qd-seed1-ckpt4000.log
```

PBS stdout/stderr is left at the scheduler's default job-specific path because
PBS does not expand `${PBS_JOBID}` inside `#PBS -o/-e` directives.

## Separate base-prompt treatment control

`train/submit_eval_base_treatments.pbs` is available for matched local-base
`debias` and `forced` prompt evaluations. No result artifact for those jobs is
part of this branch, so their status is **not established here**.

## Guard rails

- `test_goods` is validation/checkpoint-selection data, not an untouched test.
- Output root is `baselines/`, not the confirmatory GRPO `baseline/` root.
- The frozen unused-configuration suite and OOD-50 are never reopened here.
- The trajectory and selector do not establish an SFT-versus-GRPO winner.
