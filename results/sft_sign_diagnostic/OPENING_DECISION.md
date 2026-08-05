# Frozen-suite opening decision — SFT vs sign-only diagnostic (2026-08-03)

**Decision (user-authorized, informed):** open the frozen suites for an
**exploratory / post-hoc** diagnostic of reward-hacking vs task-simplicity,
comparing **SFT seed1 step6000** vs **sign-only GRPO seed1 step6000**. This is
explicitly **not** a method-winner test and **cannot** change any checkpoint
selection.

## Irreversible consequences (acknowledged)

- Using the **untouched method-comparison suite** (`data/method_comparison/`) or
  the **semantic-counterbalancing suite** for this diagnostic **forfeits their
  status as the untouched confirmatory instrument**. Any later confirmatory
  two-seed cross-method claim would then require a *new* untouched suite frozen
  in advance.
- **OOD-50** and the **framing** benchmark are already opened; reusing them here
  for a method diagnostic is **post-hoc** and cannot support a confirmatory
  method claim.

## Scope of this diagnostic

| axis | data | compute | status |
|---|---|---|---|
| λ, η, direct ownership, consistency, target agreement | test_goods (validation) | CPU (existing preds) | DONE (`structural_ownership.json`) |
| adverse framing transfer | `data/framing_effects_23prob.json` (opened) | GPU | pending |
| genuinely new goods | OOD-50 (opened) | GPU | pending |
| answer-label swaps / keep-trade wording / order reversal / paraphrases | surface-form variants | GPU + new harness | pending |
| neutral preference preservation | ownership-free neutral prompts | GPU | pending |

## Interpretation rule (pre-declared)

- Both pass the stress tests → rapid convergence reflects an **easy-to-learn
  task**.
- Both fail the same stress tests → **shared shortcut learning**.
- Only sign-only GRPO fails → evidence consistent with **reward
  overoptimization**.

Governance note: `KNOWN_ISSUES.md` / `METHOD_COMPARISON_PROTOCOL.md` will be
updated to mark any frozen suite actually consumed once its GPU evaluation runs.
