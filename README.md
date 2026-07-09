# llm-loss-aversion-rl

Fine-tuning Qwen2.5-7B-Instruct with GRPO to reduce measured loss aversion in an endowment-effect task.

## Headline Results

- Baseline Qwen2.5-7B: λ̂_before = **11.75** (SE = 1.22)
- GRPO LoRA, baseline treatment: λ̂_after = **0.177** (SE = 0.005)
- GRPO LoRA, debias treatment: λ̂ = **0.205**
- GRPO LoRA, forced treatment: λ̂ = **0.173**
- Main reduction: **98.5%** on held-out `test_goods.json`

## Current Work

- Qwen-delta reward ablation using `data/deltas/delta_qwen_base.json`
- Checkpoint/training-health audit using `monitor.sh` and `plot_training.py`
- Cross-model comparison against the frontier-model baseline

See `PROJECT_OVERVIEW.md` for the research narrative and `HISTORY.md` for the commit-by-commit project history.
