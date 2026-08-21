# STATUS — exploratory 30,000-step checkpoint diagnostic pilot

**Status: COMPLETE — both checkpoints evaluated, all integrity checks passed**

*This file is updated in place as the pilot progresses. It is the execution
record; `summary.md` carries the results.*

## What this is

An exploratory pilot asking one question: **do the two late 30,000-step
magnitude-GRPO endpoints behave differently from the previously evaluated
frozen-selected checkpoints** on

1. equivalent prompt formats / surface-form stress, and
2. adverse framing?

## What this is NOT

- It **does not** change the frozen checkpoint selections
  (seed 1 → step 2,000; seed 2 → step 6,000).
- The 30k adapters are **not** newly selected models and the checkpoint
  selector was not modified or re-run.
- No prohibited suite was evaluated: not
  `data/method_comparison/method_comparison_suite.json`, not
  `data/method_comparison/semantic_counterbalancing.json`, not the frozen
  unused-configuration suite, not the neutral-preference suite, not OOD-50,
  GSM8K, or IFEval. Both PBS jobs hard-refuse those paths before loading a
  model.

## Allowed inputs actually used

| input | path | SHA-256 |
|---|---|---|
| surface-form subset (exploratory) | `data/surface_form_stress/surface_form_subset.json` | `78525395ea208e4e4f2ef3e4266a5386259175cdcabeb29d8fb887cb4b272262` |
| adverse-framing benchmark (already opened) | `data/framing_effects_23prob.json` | `dd58443d5d4a014f540e0406fc4d29166f5d5193c525c42851a595742c2cc15c` |
| goods reference | `everyday_goods_full.json` | `7a28e91adc6de95008b8f83d25415909bfb30eb2b93874fa2a284349335f3934` |

## Base model revision

Weights are a local directory, not a pinned Hub revision, so the identity is
recorded by content hash:

| file | SHA-256 |
|---|---|
| `models/Qwen2.5-7B-Instruct/model.safetensors` | `c8f5aa1b3293e01920382648b804df1d84e0655b3fce71619071fcd7fbecc918` |
| `models/Qwen2.5-7B-Instruct/config.json` | `619fcb509e6900453e6ff5498733e43b4c6c46eb5013978b81f327be8717de39` |
| `models/Qwen2.5-7B-Instruct/generation_config.json` | `87c51d61bd70a204993da36384296ccfeb56d46ee2cc7247929f917d0d8118f7` |
| `models/Qwen2.5-7B-Instruct/tokenizer.json` | `3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8` |

No upstream Hub commit is recorded in the local snapshot, so the revision
cannot be named beyond these content hashes.

## Adapters

Both 30k adapters were verified to contain `adapter_model.safetensors`. No
substitution was made or permitted — both PBS jobs abort if either file is
absent.

| model | adapter path | `adapter_model.safetensors` SHA-256 | role |
|---|---|---|---|
| `GRPO-qd-seed1-ckpt30000` | `checkpoints/grpo_qwen_delta_seed1/checkpoint-30000` | `6b9580a4ed24fabf910cfd654f555a867149e2beeb9f277650aa3d52324ffd0d` | late endpoint (exploratory) |
| `GRPO-qd-seed2-ckpt30000` | `checkpoints/grpo_qwen_delta_seed2/checkpoint-30000` | `3d02042db2349e5c732210f243922a983e9506872711c1fbd68ceed00a4e107e` | late endpoint (exploratory) |
| `GRPO-qd-seed1-ckpt2000` | `checkpoints/grpo_qwen_delta_seed1/checkpoint-2000` | `223b80bd16e7383b09c73287b60e3dde4c26608c64ce9365557d5a88199bc190` | seed 1 frozen selection (comparator) |
| `GRPO-qd-seed2-ckpt6000` | `checkpoints/grpo_qwen_delta_seed2/checkpoint-6000` | `e6bd31a865d71aeb33d5db08ec86c4c9211aba55f1b181ca28026621686df52c` | seed 2 frozen selection (comparator) |

## Reuse of earlier results

Earlier results are reused **only** after their provenance is re-verified by
`eval/pilot30k_analyze.py` (it aborts otherwise):

| reused result | check | outcome |
|---|---|---|
| `results/surface_form_stress/Base` | subset SHA, no adapter, 9,216 rows | verified |
| `results/surface_form_stress/GRPO-qd-seed1-ckpt2000` | subset SHA, adapter SHA `223b80bd…`, 9,216 rows | verified |
| `results/surface_form_stress/GRPO-qd-seed2-ckpt6000` | subset SHA, adapter SHA `e6bd31a8…`, 9,216 rows | verified |
| `framing/full_qd8k_120x23/Qwen-7B-Base/single_word` | benchmark SHA, `single_word`, 5,520 rows | verified |
| `framing/full_qd8k_120x23/Qwen-7B-GRPO-step8000/single_word` | benchmark SHA, `single_word`, 5,520 rows | verified |

Nothing in `results/surface_form_stress/` or `framing/` is overwritten: both
jobs write only under `results/checkpoint_diagnostics/pilot30k/` and refuse
otherwise.

## Framing comparator limitation

**No framing predictions exist for the frozen-selected seed1@2000 or
seed2@6000 checkpoints.** The only earlier GRPO framing comparator is the
**exploratory seed 42 step-8,000** run. The 30k-versus-earlier framing contrast
is therefore **not a clean within-seed checkpoint comparison**; the clean
contrast available on this axis is 30k versus the matched base.

## Output paths

| what | path |
|---|---|
| surface-form raw predictions | `results/checkpoint_diagnostics/pilot30k/surface/<model>/form_predictions.jsonl` |
| framing raw predictions | `results/checkpoint_diagnostics/pilot30k/framing/<model>/single_word/predictions.json` |
| summaries | `results/checkpoint_diagnostics/pilot30k/summary.{json,csv,md}` |
| checksums of raw files | `results/checkpoint_diagnostics/pilot30k/raw_manifest.json` |
| figure | `results/checkpoint_diagnostics/pilot30k/pilot30k_comparison.png` |

Raw prediction files are gitignored and checksummed only.

## Expected row counts

| job | model | expected |
|---|---|---|
| surface | `GRPO-qd-seed1-ckpt30000` | 9,216 (96 cases × 2 perspectives × 48 forms) |
| surface | `GRPO-qd-seed2-ckpt30000` | 9,216 |
| framing | `GRPO-qd-seed1-ckpt30000` | 5,520 (120 scenarios × 23 probabilities × 2 frames) |
| framing | `GRPO-qd-seed2-ckpt30000` | 5,520 |

Both PBS jobs assert these counts and exit non-zero on a mismatch; the analyzer
re-checks them independently.

## Resumability

- **Surface:** `eval/surface_form_infer.py` skips a model whose
  `form_predictions.jsonl` already holds all 9,216 rows, recomputes a partial
  file, and writes through a `.tmp` + atomic rename, so a walltime-killed job
  never leaves a mixed directory. Resubmit the same script.
- **Framing:** `eval/run_framing_local.py` checkpoints every 100 cases, resumes
  from the existing predictions file, and refuses to resume unless the stored
  manifest matches this run's model path, adapter, prompt style, benchmark
  SHA-256, and dtype. Resubmit the same script.

## Pre-GPU validation (all CPU)

| check | result |
|---|---|
| `data/surface_form_stress/build_surface_form_subset.py --check` | PASS — subset regenerates byte-identically |
| `eval/test_surface_form_transforms.py` | PASS |
| `eval/test_pilot30k_analyze.py` | PASS (all assertions, incl. end-to-end) |
| `bash -n` on both PBS scripts | PASS |
| analyzer dry run over existing results only | PASS |

`eval/test_pilot30k_analyze.py` includes a full end-to-end pass in which the
"30k" inputs are copies of an already evaluated model; every paired delta and
CI bound must then be exactly zero, which exercises the whole pipeline without
a GPU.

## Git

| item | value |
|---|---|
| branch | `codex/grpo-30k-diagnostic` |
| branch point | `fa8f4c5f0d9885086e627b8c05fcab1d6de975bf` |
| infrastructure commit | `e30c1e5d6a1531d8d54cfe5e56d87e0a49858709` (pushed) |
| results commit | *(pending)* |

## PBS job IDs

| job | script | job ID | status |
|---|---|---|---|
| surface-form | `train/submit_eval_pilot30k_surface.pbs` | `15221288.pbs101` | **COMPLETED** (queue `g1`) |
| adverse framing | `train/submit_eval_pilot30k_framing.pbs` | `15221289.pbs101` | **COMPLETED** (queue `g1`) |

Neither job was resubmitted; each finished in a single attempt, so the resume
paths were exercised only by the pre-GPU tests.

## Observed row counts

| job | model | expected | observed | verdict |
|---|---|---|---|---|
| surface | `GRPO-qd-seed1-ckpt30000` | 9,216 | **9,216** | PASS |
| surface | `GRPO-qd-seed2-ckpt30000` | 9,216 | **9,216** | PASS |
| framing | `GRPO-qd-seed1-ckpt30000` | 5,520 | **5,520** | PASS |
| framing | `GRPO-qd-seed2-ckpt30000` | 5,520 | **5,520** | PASS |

Counts were checked twice: once inside each PBS job (non-zero exit on mismatch)
and again independently by `eval/pilot30k_analyze.py`.

## Post-run integrity checks

| check | result |
|---|---|
| adapter SHA-256 recorded by the surface runner == expected | PASS both seeds |
| adapter SHA-256 recorded by the framing job == expected | PASS both seeds |
| surface subset SHA-256 in every `run_metadata.json` | PASS (`78525395…`) |
| framing benchmark SHA-256 in every `manifest.json` | PASS (`dd58443d…`) |
| framing prompt style | `single_word` for all four models |
| reused comparator results re-verified | PASS (3 surface, 2 framing) |
| existing `results/surface_form_stress/` and `framing/` modified | NO |
| prohibited suite touched | NO |

## Raw artifacts

Raw predictions are checksummed in `raw_manifest.json` and **not committed**
(25 files, 34.2 MB total across both new result trees and the re-verified comparators). A checksum verifies a file
someone already has; it does not make the file downloadable. Derived summaries,
metadata, manifests, and the figure are committed.

## Completion status

**COMPLETE.** Both 30,000-step checkpoints were evaluated on both axes, every
expected row is present, and every provenance check passed. Results are in
`summary.md` / `summary.json` / `summary.csv`, with the figure in
`pilot30k_comparison.png`.

The predeclared exploratory rule (paired 95% CI excludes zero, or |Δ| ≥ 0.05)
is met on multiple metrics for both seeds, so it **recommends running the
complete checkpoint trajectory**. That recommendation is a decision about what
to run next; it is not a confirmatory finding and it does not alter the frozen
checkpoint selections.
