# Pre-result model-set clarification — 2026-08-03

Dated pre-result clarification (recorded **before** any surface-form / neutral /
framing / OOD result was interpreted).

## Correction

The central diagnostic question concerns the **main Qwen-own MAGNITUDE-reward
GRPO**, not sign-only GRPO. The core model set is:

1. exact matched local base;
2. **SFT seed 1 @ checkpoint 6000** (`checkpoints/sft_qwen_delta_seed1_pilot6k/checkpoint-6000`);
3. **magnitude-GRPO seed 1 @ frozen-selected step 2000** (`checkpoints/grpo_qwen_delta_seed1/checkpoint-2000`, sha `223b80bd…`);
4. **magnitude-GRPO seed 2 @ frozen-selected step 6000** (`checkpoints/grpo_qwen_delta_seed2/checkpoint-6000`, sha `e6bd31a8…`).

**Sign-only GRPO is demoted to a supplementary ablation** — its outputs are
preserved, not deleted, and it is removed from *new* surface-form and neutral
jobs.

## What is unchanged

The frozen surface-form **subset, transformations, metrics, manifest, and
analysis logic are NOT changed**. Only the evaluated model list is updated
(`surface_form_analyze.py` / `neutral_preference_analyze.py` MODELS; new PBS
model lists). The neutral-preference design/manifest are likewise unchanged.

## Handling of in-flight jobs (as of this clarification)

- **OOD-50 `15074831` — FINISHED.** Produced SFT s1@6k OOD (the needed new
  evaluation) and sign-only s1@6k OOD (supplementary). Both preserved. The
  magnitude-GRPO **seed1@2000 and seed2@6000 OOD results already exist and are
  reused** (not rerun). Compare λ against the OOD base (2.209).
- **Surface-form `15075536` — already RUNNING** (Base done at 9,216 rows; SFT /
  sign-only in progress). It was submitted with the obsolete base+SFT+sign-only
  set. Because it had already started (not queued), it is **allowed to finish**
  and its outputs preserved; sign-only is treated as supplementary. An
  **additional magnitude-GRPO job** (`submit_eval_surface_form_magnitude.pbs`)
  evaluates seed1@2000 + seed2@6000 on the **identical frozen forms**, and the
  analyzer aggregates all models consistently (core vs supplementary labelled).
- **Framing `15074751` — queued** (SFT + sign-only). The main magnitude-GRPO
  framing result already exists; **SFT framing is retained as the necessary new
  evaluation** and sign-only is optional. Left as-is.

## Exploratory status

Unchanged: this whole diagnostic is EXPLORATORY / POST-HOC and may not change any
checkpoint or method selection.
