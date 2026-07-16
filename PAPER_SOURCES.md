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
| `notebooks/compare_lambdas.ipynb` | Referenced historically; does not exist |

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
   - **BASE = matched local**: λ̂ = 7.637 (SE 0.627), η̂ = 1.007 — same weights,
     rows, scorer and estimator as the post-training rows, so before/after is
     causal. The old λ̂ = 11.75 (Together Turbo) is **superseded**: ~54% of it was
     pipeline mismatch, not the model. Never cite 11.75 as the baseline.
3. **Never re-add retracted claims.** Superiority over the human λ ≈ 2.25
   benchmark (different task and parameterization — our multiplier is 1+λ),
   superiority over frontier models (unharmonized endpoints/samples/scorers), or
   or the superseded 11.75 baseline. (A causal before/after reduction IS now
   supported, against the matched base only.) See the tex section "Claims We Do
   Not Make" and PAPER_READINESS #11.
4. **Stale artifact found?** Regenerate it or archive it under `archive/` with a
   dated note. Do not edit it in place — that is how two sources of truth start.
