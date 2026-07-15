# llm-loss-aversion-rl

Fine-tuning Qwen2.5-7B-Instruct with GRPO to reduce measured loss aversion in an endowment-effect task.

## Headline Results

- Baseline Qwen2.5-7B: λ̂_before = **11.75** (SE = 1.22)
- GRPO LoRA, baseline treatment: λ̂_after = **0.177** (SE = 0.005)
- GRPO LoRA, debias treatment: λ̂ = **0.205**
- GRPO LoRA, forced treatment: λ̂ = **0.173**
- Main reduction: **98.5%** on held-out attribute configurations in `test_goods.json`
- Attribute profiles explain **48.8%** of within-pair/perspective variation
- Changing the held-out configuration flips at least one answer for **42.2%** of goods pairs

Training and compositional evaluation contain no repeated prompts or exact
configurations. They use the same 100 goods and 4,945 goods pairs, with 10
configurations per pair used for training and two disjoint configurations
reserved for evaluation. This supports within-benchmark configuration
generalization; it is not by itself an unseen-goods claim. See
`ATTRIBUTE_EFFECTS.md` for the complete analysis.

## Current Work

- Qwen-delta reward ablation using `data/deltas/delta_qwen_base.json`
- Checkpoint/training-health audit using `monitor.sh` and `plot_training.py`
- Cross-model comparison against the frontier-model baseline
- Follow-up reward design based on paired ownership consistency, neutral
  preference anchors, valid Pareto cases, and ordinal attribute monotonicity

See `PROJECT_OVERVIEW.md` for the research narrative,
`ATTRIBUTE_EFFECTS.md` for the held-out configuration analysis,
`REWARD_DESIGN_V2.md` for the proposed non-cardinal reward design, and
`HISTORY.md` for the commit-by-commit project history.
