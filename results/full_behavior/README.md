# Full-behavior aggregation (Task 5) — does λ tell the full story?

Deterministic, CPU-only aggregation by `eval/aggregate_full_behavior.py`. No model
inference, no checkpoint or frozen-suite evaluation.

## Snapshot architecture (reproducibility scope)

There are two clearly separated stages:

1. **Render (default, clean-clone reproducible).** All outputs render **only**
   from the tracked canonical **`full_behavior_snapshot.json`** (40 rows, each
   with primitive statistics + provenance). Because the snapshot and the four
   rendered files are committed, **table rendering and `--check` reproduce from a
   clean clone / `git archive`** with no NSCC access. Verified: `--check` passes
   from a fresh `git archive` extraction.
2. **Refresh (NOT clean-clone reproducible).** `--refresh` **recomputes** the
   snapshot rows by reading the **untracked/raw NSCC prediction directories**
   (`baseline/*/Model_1/loss_aversion_{X,Y}_for_Model_A.json`) and the NLS
   estimation CSVs. Those raw sources are **not** in git (see `ARTIFACT-001`), so
   **row recomputation is reproducible only where those NSCC artifacts exist** —
   it cannot be reproduced from the repository alone. The snapshot's per-row
   provenance records the source paths and X/Y SHA-256 so the numbers stay
   traceable even though the raw inputs are not committed.

In short: **rendering the tables is repo-reproducible; regenerating the underlying
rows is not** — it requires the raw NSCC predictions/NLS sources via `--refresh`.

## Outputs

| file | role |
|---|---|
| `full_behavior_snapshot.json` | **tracked canonical input** — 40 rows: primitives + provenance (NLS CSV path, X/Y SHA-256). The only source the render/`--check` paths read. |
| `full_behavior.json` | rendered per-row record + summary |
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
# Clean-clone reproducible (renders from the tracked snapshot; no NSCC needed):
python3 eval/aggregate_full_behavior.py          # render outputs from snapshot
python3 eval/aggregate_full_behavior.py --check   # verify byte-identical (git archive-safe)

# Row recomputation — requires the untracked/raw NSCC predictions + NLS sources:
python3 eval/aggregate_full_behavior.py --refresh # re-read NSCC dirs, rewrite the snapshot
```
