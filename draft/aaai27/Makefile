PYTHON ?= python3

.PHONY: figures tables check paper checklist render

figures:
	MPLCONFIGDIR=/tmp/grpo-aaai-mpl $(PYTHON) scripts/make_figures.py

tables:
	$(PYTHON) scripts/generate_tables.py --write

paper: figures tables
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	mkdir -p output/pdf
	cp main.pdf output/pdf/main.pdf

checklist:
	pdflatex -interaction=nonstopmode -halt-on-error reproducibility_checklist.tex
	mkdir -p output/pdf
	cp reproducibility_checklist.pdf output/pdf/reproducibility_checklist.pdf

render:
	mkdir -p tmp/pdfs
	pdftoppm -png -r 150 output/pdf/main.pdf tmp/pdfs/main
	pdftoppm -png -r 150 output/pdf/reproducibility_checklist.pdf tmp/pdfs/checklist

check:
	$(PYTHON) scripts/generate_tables.py --check
	$(PYTHON) scripts/check_submission.py
