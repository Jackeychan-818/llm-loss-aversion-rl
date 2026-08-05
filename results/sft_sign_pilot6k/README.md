# SFT vs sign-only pilot — side-effect eval (EXPLORATORY_POST_HOC)

Low-cost, **exploratory / post-hoc** comparison of two pilot adapters with the
same seed (1) and 6,000-step endpoint:

- **matched SFT** — `checkpoints/sft_qwen_delta_seed1_pilot6k/checkpoint-6000`
- **sign-only GRPO** — `checkpoints/grpo_qwen_delta_sign_seed1/checkpoint-6000`

on the already-opened **framing** benchmark and **GSM8K** capability retention.
Full provenance (adapter/base/dataset/evaluator SHA-256, decoding, output paths,
governance rules) is frozen in `execution_manifest.json`.

**This is NOT the frozen confirmatory method comparison and NOT evidence of
general method superiority.** The frozen two-seed untouched-suite comparison
(`METHOD_COMPARISON_PROTOCOL.md`) remains required for method-level claims. The
untouched method-comparison suite and semantic suite stay **unopened**. Results
here may **not** change any checkpoint selection.

## Checkpoint identity note

The SFT full run (`sft_qwen_delta_seed1/checkpoint-6000`, adapter `096be360…`)
and the preserved pilot (`sft_qwen_delta_seed1_pilot6k/checkpoint-6000`, adapter
`c814df36…`) have **different weights** at step 6000 (cosine-to-30k vs
cosine-to-6k schedule). The **pilot** adapter underlies the recorded
`results/causal_baseline_pilot/` result and is the one used here.

## Base reuse (no base re-evaluation)

- Framing base: `framing/full_qd8k_120x23/Qwen-7B-Base/single_word/predictions.json`
  (5,520 cases, sha `9c92ec1e…`).
- GSM8K base: `results/gsm8k/base/predictions.jsonl` (1,319 records, all unique,
  sha `bfe129cd…`, validated).

## Cost (estimate; measured rate ~105–220 SU/GPU-h; balance 3,312 SU)

| job | work | GPU-h | SU |
|---|---|---:|---:|
| framing (2 adapters) | 5,520 cases each, ~2 min compute + load | ~0.3 | ~30–70 |
| GSM8K (2 adapters) | 1,319 × 512-tok greedy each | ~2–3 | ~210–660 |
| **combined** | | **~2.3–3.3** | **~240–730** |

**Under the 1,000 SU GSM8K guard.** Actual SU/job IDs are recorded after runs.

## Run (self-submitted; smoke first, then full)

```bash
# Framing smoke (10 scenarios x 5 probs x 2 frames):
qsub -v LIMIT_SCENARIOS=10,PROBABILITY_SUBSET=0.10:0.30:0.50:0.70:0.90 \
  train/submit_eval_framing_sft_sign_pilot.pbs
# Framing full:
qsub train/submit_eval_framing_sft_sign_pilot.pbs

# GSM8K smoke (20 problems each):
qsub -v LIMIT=20 train/submit_eval_gsm8k_sft_sign_pilot.pbs
# GSM8K full:
qsub train/submit_eval_gsm8k_sft_sign_pilot.pbs
```

Outputs:
- `framing/sft_sign_pilot6k/{SFT,SignOnly}-seed1-step6000/single_word/`
- `results/gsm8k_sft_sign_pilot6k/{sft,sign}_seed1_step6000/` (+ `comparison_vs_base.json`)

After results exist: extract framing flip-rate / prob-gap / monotonicity and the
GSM8K paired accuracy / bootstrap CI / McNemar / discordant counts, write compact
summaries + a raw-artifact manifest (paths/sizes/hashes; raw predictions not
committed), and update `PROJECT_OVERVIEW.md` / `PAPER_READINESS.md` /
`RESEARCH_ROADMAP.md` / `HISTORY.md` (labelled exploratory).
