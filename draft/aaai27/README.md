# AAAI-27 Paper Workspace

This directory is the anonymous AAAI-27 manuscript workspace for:

> **Reducing Loss-Averse Choice in LLMs with Reinforcement Learning**

## Important Repository-State Note

The numerical registry in `data/results_registry.json` reflects project
artifacts through `main@c1ab3da` plus the merged manuscript updates dated
July 28, 2026. Every completed row must continue to trace to its manifest or
result JSON.

The fuller project-record draft is `../training_overview.tex`; `main.tex` is
the concise AAAI-formatted version and intentionally omits some diagnostic
tables for space.

## Files

- `main.tex`: single-file anonymous AAAI manuscript.
- `references.bib`: project bibliography.
- `aaai2027.sty`, `aaai2027.bst`: unmodified files copied from AuthorKit27.
- `openreview_abstract.txt`: substantive registration abstract.
- `OPENREVIEW.md`: title, track, keywords, and registration checks.
- `reproducibility_checklist.tex`: standalone checklist with a truthful initial
  completion; update answers as missing experiments and packaging are finished.
- `data/results_registry.json`: machine-readable paper table registry.
- `scripts/generate_tables.py`: rewrites bounded result-table blocks inside the
  single `main.tex`.
- `scripts/make_figures.py`: generates paper-native vector figures.
- `scripts/check_submission.py`: checks common AAAI, anonymity, table, PDF, and
  font failures.
- `SUBMISSION_WORKBOARD.md`: dated execution plan through the deadlines.
- `supplement/MANIFEST.md`: anonymous code/data package inventory.

## Build

AAAI requires PDFLaTeX. A standard build is:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Then run:

```bash
python3 scripts/generate_tables.py --check
python3 scripts/check_submission.py
```

This workspace was QA-compiled with a temporary user-space BasicTeX
extraction; no system TeX installation was changed. Future builds require a
PDFLaTeX installation or Overleaf. Do not substitute XeLaTeX or LuaLaTeX: the
AAAI style rejects them.

## Table Updates

Edit only `data/results_registry.json`, then run:

```bash
python3 scripts/generate_tables.py --write
```

The script keeps the paper as a single `.tex` file, as required by the author
kit. Do not manually edit content between `BEGIN AUTO` and `END AUTO` markers.

## Claim Freeze

The paper may claim a matched reduction relative to the exact local base,
generalization to the frozen new-goods suite, and the frozen 2/2 confirmatory
seed verdict, subject to artifact completion and the remaining matched
baselines and robustness checks.

It must not claim:

- complete elimination of loss aversion;
- elimination of status-quo bias `eta`;
- human or universal frontier-model superiority;
- that `test_goods` is an untouched final test;
- that Qwen-own dominates the consensus reward on every metric;
- capability equivalence from a nonsignificant GSM8K comparison.

## AAAI-27 Rules Used

- Abstract deadline: July 21, 2026, 11:59 PM UTC-12.
- Full paper deadline: July 28, 2026, 11:59 PM UTC-12.
- Supplementary material/code deadline: July 31, 2026, 11:59 PM UTC-12.
- Seven pages of technical content; pages 8--9 may contain references only.
- Anonymous two-column submission.
- Reproducibility checklist uploaded separately with the paper.
- Reviewers are not required to consult supplementary material.

Official pages:

- https://aaai.org/conference/aaai/aaai-27/
- https://aaai.org/conference/aaai/aaai-27/submission-instructions/
- https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/
- https://aaai.org/conference/aaai/aaai-27/supplementary-material/
