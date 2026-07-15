# Paper Sources — Canonical vs Derived

*Established July 15, 2026*

This policy exists because the repository already suffered **model-role drift**:
after the primary model changed to Qwen-own delta, `training_overview.tex` still
presented the *consensus* result (λ̂ = 0.177) as the headline and retained
superiority claims the evidence does not support. Two sources disagreed, and the
stale one was the one a reader would have believed.

## Canonical source

**`training_overview.tex` is the single source of truth for the paper.**

Everything else — PDFs, slide decks, exported figures, any `*_merged.tex`,
notebooks that render tables — is **derived**. Derived artifacts are
**regenerated** from the canonical source, never hand-edited.

| Artifact | Status |
|---|---|
| `training_overview.tex` | **CANONICAL** — tracked, reviewed, edited directly |
| PDFs / slides / figures | Derived — regenerate; build outputs are gitignored |
| `training_overview_merged.tex` | **MISSING — see Open Problem below** |
| `notebooks/compare_lambdas.ipynb` | Referenced historically; **does not exist** |

## Rules

1. **One number, one home.** A result cited anywhere (README, CLAUDE.md,
   KNOWN_ISSUES, PAPER_READINESS, slides) must trace to the canonical `.tex` and
   to a committed artifact under `results/`. If a number has no committed
   artifact, it is not yet reportable (PAPER_READINESS #6).
2. **Model roles are declared once**, in the canonical header, and must match
   `PAPER_READINESS.md`:
   - **PRIMARY** — Qwen-own δ̃, GRPO step 8,000: λ̂ = 0.111 (SE 0.014), η̂ = 0.504
   - **ABLATION** — frontier-consensus δ̃: λ̂ = 0.177 (SE 0.005), η̂ = −0.048
   - Both `test_goods` estimates are **validation-only**.
   - The 11.75 base row is **historical and unmatched** (Together Turbo, 9,950
     cases, first-token probs) until the matched local-base rerun lands.
3. **Never re-add retracted claims.** Superiority over the human λ ≈ 2.25
   benchmark (different task and parameterization — our multiplier is 1+λ),
   superiority over frontier models (unharmonized endpoints/samples/scorers), or
   a causal before/after reduction (base not yet matched). See the tex section
   "Claims We Do Not Make" and PAPER_READINESS #11.
4. **Stale artifact found?** Regenerate it or archive it under `archive/` with a
   dated note. Do not edit it in place — that is how two sources of truth start.

## Open problem: `training_overview_merged.tex`

`PAPER_READINESS.md` #6 states the **checkpoint curve and OOD tables** are
"currently recorded in `training_overview_merged.tex`". **That file does not
exist** — it is absent from the working tree and has never been committed on any
branch. So paper-critical evidence currently has **no canonical home**: it lives
only on a local machine, or the reference is stale.

**Required action (one of):**
- commit the file if it exists locally, then **fold its unique content into the
  canonical `training_overview.tex`** and archive the merged copy; or
- if it is stale, delete the reference and migrate the checkpoint-curve/OOD
  tables into the canonical source, generated from tracked artifacts.

Until then, treat any checkpoint-curve or OOD number sourced from that file as
**unverifiable**.
