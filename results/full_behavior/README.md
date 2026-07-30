# Full-behavior aggregation (Task 5) — does λ tell the full story?

Deterministic, CPU-only aggregation built by `eval/aggregate_full_behavior.py`
from **committed results only** (no model inference, no checkpoint or
frozen-suite evaluation). Machine-readable tables are the source of truth;
Markdown/LaTeX are rendered from them (no hand-entered numbers).

## Outputs

| file | role |
|---|---|
| `full_behavior.json` | canonical per-row record incl. provenance (NLS CSV path, X/Y SHA-256) |
| `full_behavior.csv` | flat table for spreadsheets |
| `full_behavior.md` | human-readable table |
| `full_behavior.tex` | paper table (generated; do not hand-edit) |

## Per row

λ, η (+ SEs), `d = √(λ²+η²)`, consistency, keep-both, trade-both, hard-choice
Yes-rate, target agreement (vs the frozen rational target), W (pseudo-utility
alignment where available), sample size, parse-failure rate, an estimator
conditioning proxy (max/min parameter variance — **not** the Jacobian condition
number), and provenance.

## The success guard

`clean_reduction = (|λ| < 0.5) AND no caveat`. Caveats fire when a small λ is
contradicted by:

- `|η| > 1.0` (status-quo bias remains),
- consistency `< 0.50`,
- parse-failure rate `> 1%`, or
- choice collapse (a keep-both or trade-both side `< 2%`).

This prevents reporting a low λ as success when the model still keeps everything,
has high η, or has degenerate choices. In the current committed set, **15/40 rows
are `clean=NO`**; of those, **10 have `|λ|<0.5` plus a contradictory caveat** (the
direct evidence that λ alone can mislead), and the remaining 5 are `clean=NO`
only because `|λ|≥0.5`.

## Scope / honesty notes

- `test_goods` is **validation**. These are not confirmatory method results.
- The **un-evaluated full SFT grid has no NLS CSVs**, so it is absent here — never
  fabricated. It enters only after the GPU-phase evaluation under
  `METHOD_COMPARISON_PROTOCOL.md`.
- **Direct ownership-free anchor agreement** is a dataset-level reward-validation
  quantity (`results/qwen_delta_anchor_validation.json`: 71.1% on test_goods),
  not a per-checkpoint number; the per-model analog reported here is
  `target_agreement`.

## Reproduce

```bash
python3 eval/aggregate_full_behavior.py          # regenerate
python3 eval/aggregate_full_behavior.py --check   # verify byte-identical
```
