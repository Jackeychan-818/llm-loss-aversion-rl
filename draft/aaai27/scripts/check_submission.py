#!/usr/bin/env python3
"""Check common AAAI-27 formatting, anonymity, and provenance failures."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.tex"
CHECKLIST = ROOT / "reproducibility_checklist.tex"
BIB = ROOT / "references.bib"
FIGURES = [ROOT / "figures" / "task_overview.pdf"]
PDF_CANDIDATES = [ROOT / "output" / "pdf" / "main.pdf", ROOT / "main.pdf"]

BANNED_PACKAGES = {
    "authblk",
    "balance",
    "CJK",
    "float",
    "flushend",
    "fullpage",
    "geometry",
    "hyperref",
    "navigator",
    "indentfirst",
    "layout",
    "multicol",
    "nameref",
    "pgfplots",
    "savetrees",
    "setspace",
    "stfloats",
    "tabu",
    "titlesec",
    "tocbibind",
    "ulem",
    "wrapfig",
}

BANNED_COMMAND_PATTERNS = {
    r"\\nocopyright\b": r"\nocopyright",
    r"\\addtolength\b": r"\addtolength",
    r"\\balance\b": r"\balance",
    r"\\baselinestretch\b": r"\baselinestretch",
    r"\\clearpage\b": r"\clearpage",
    r"\\columnsep\b": r"\columnsep",
    r"\\newpage\b": r"\newpage",
    r"\\pagebreak\b": r"\pagebreak",
    r"\\pagestyle\b": r"\pagestyle",
    r"\\tiny\b": r"\tiny",
    r"\\vspace\s*\{\s*-": "negative vspace",
    r"\\vskip\s*-": "negative vskip",
    r"\\input\s*\{": r"\input (paper source must remain single-file)",
    r"\\includegraphics\s*\[[^\]]*(?:trim|clip|viewport)": "trim/clip/viewport in includegraphics",
    r"\\resizebox\b": r"\resizebox",
}

IDENTITY_PATTERNS = {
    r"(?i)\bjackey\b": "local username",
    r"/Users/": "absolute macOS home path",
    r"/scratch/users/": "absolute cluster user path",
    r"(?i)\bnational university of singapore\b": "institution name",
    r"(?i)\bnus\b": "institution acronym",
    r"(?i)github\.com/Jackeychan": "identifying repository URL",
}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.ok: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def pass_(self, message: str) -> None:
        self.ok.append(message)

    def print(self) -> None:
        for item in self.ok:
            print(f"PASS: {item}")
        for item in self.warnings:
            print(f"WARN: {item}")
        for item in self.errors:
            print(f"FAIL: {item}")
        print(
            f"\nSummary: {len(self.ok)} passed, "
            f"{len(self.warnings)} warnings, {len(self.errors)} failures."
        )


def command_path(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    bundled = (
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override"
        / name
    )
    return str(bundled) if bundled.exists() else None


def check_source(report: Report) -> None:
    text = MAIN.read_text(encoding="utf-8")

    if r"\documentclass[letterpaper]{article}" in text:
        report.pass_("US-letter article document class")
    else:
        report.error("main.tex must use \\documentclass[letterpaper]{article}.")

    if r"\usepackage[submission]{aaai2027}" in text:
        report.pass_("anonymous aaai2027 submission mode")
    else:
        report.error("main.tex is not in aaai2027 submission mode.")

    if r"\author{Anonymous Submission}" in text and r"\affiliations{}" in text:
        report.pass_("anonymous author and empty affiliation blocks")
    else:
        report.error("anonymous author/affiliation blocks are not exact.")

    packages = set(re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", text))
    flattened = {piece.strip() for item in packages for piece in item.split(",")}
    for package in sorted(flattened & BANNED_PACKAGES):
        report.error(f"banned package in main.tex: {package}")

    for pattern, label in BANNED_COMMAND_PATTERNS.items():
        if re.search(pattern, text):
            report.error(f"banned or noncompliant command in main.tex: {label}")

    for pattern, label in IDENTITY_PATTERNS.items():
        if re.search(pattern, text):
            report.error(f"possible de-anonymization in main.tex: {label}")

    if r"\bibliographystyle" in text:
        report.error("Do not set bibliographystyle; aaai2027 sets it.")
    else:
        report.pass_("bibliography style left to aaai2027")

    if "-- & -- & --" in text:
        report.warn("result tables intentionally contain unfinished cells")


def check_citations(report: Report) -> None:
    text = MAIN.read_text(encoding="utf-8")
    bib = BIB.read_text(encoding="utf-8")
    cited: set[str] = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", text):
        cited.update(key.strip() for key in group.split(","))
    keys = set(re.findall(r"@\w+\{\s*([^,\s]+)", bib))
    missing = sorted(cited - keys)
    if missing:
        report.error("citations missing from references.bib: " + ", ".join(missing))
    else:
        report.pass_(f"all {len(cited)} cited keys exist in references.bib")


def check_tables(report: Report) -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_tables.py"), "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        report.pass_("generated result-table blocks are current")
    else:
        report.error(proc.stdout.strip() or proc.stderr.strip())


def check_checklist(report: Report) -> None:
    text = CHECKLIST.read_text(encoding="utf-8")
    questions = text.split("% The questions start here", maxsplit=1)[-1]
    if "Type your response here" in questions:
        report.error("reproducibility checklist has unanswered questions")
    else:
        report.pass_("reproducibility checklist has an answer for every question")
    report.warn("checklist contains partial answers that must be revisited before submission")


def check_figures(report: Report) -> None:
    for figure in FIGURES:
        if not figure.exists():
            report.error(f"missing figure: {figure.relative_to(ROOT)}")
            continue
        payload = figure.read_bytes()
        if b"/Subtype /Type3" in payload or b"/FontType 3" in payload:
            report.error(f"Type 3 font marker found in {figure.relative_to(ROOT)}")
        else:
            report.pass_(f"no Type 3 marker in {figure.relative_to(ROOT)}")


def parse_pdfinfo(path: Path) -> dict[str, str]:
    pdfinfo = command_path("pdfinfo")
    if not pdfinfo:
        return {}
    proc = subprocess.run([pdfinfo, str(path)], text=True, capture_output=True)
    if proc.returncode != 0:
        return {}
    info: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", maxsplit=1)
            info[key.strip()] = value.strip()
    return info


def check_pdf(report: Report) -> None:
    pdf = next((path for path in PDF_CANDIDATES if path.exists()), None)
    if pdf is None:
        report.warn("compiled manuscript PDF not found; page/layout/font QA not yet possible")
        return

    info = parse_pdfinfo(pdf)
    if not info:
        report.warn("pdfinfo unavailable or failed")
        return

    pages = int(info.get("Pages", "0"))
    if pages <= 9:
        report.pass_(f"PDF has {pages} pages (maximum total is 9)")
    else:
        report.error(f"PDF has {pages} pages; AAAI total maximum is 9")

    page_size = info.get("Page size", "")
    if "612 x 792 pts" in page_size:
        report.pass_("PDF is US letter")
    else:
        report.error(f"unexpected PDF page size: {page_size}")

    version = info.get("PDF version", "0")
    try:
        version_ok = float(version) >= 1.5
    except ValueError:
        version_ok = False
    if version_ok:
        report.pass_(f"PDF version is {version}")
    else:
        report.error(f"PDF version must be at least 1.5; found {version}")

    if info.get("Encrypted", "").lower() == "no":
        report.pass_("PDF is not encrypted")
    else:
        report.error("PDF must not be encrypted")

    for key in ("Author", "Creator"):
        value = info.get(key, "")
        if re.search(r"(?i)jackey|nus|national university", value):
            report.error(f"identifying PDF metadata in {key}: {value}")

    pdffonts = command_path("pdffonts")
    if pdffonts:
        proc = subprocess.run([pdffonts, str(pdf)], text=True, capture_output=True)
        if "Type 3" in proc.stdout:
            report.error("compiled PDF contains Type 3 fonts")
        elif proc.returncode == 0:
            report.pass_("compiled PDF contains no Type 3 fonts")
    else:
        report.warn("pdffonts unavailable; full font embedding check skipped")


def main() -> int:
    report = Report()
    check_source(report)
    check_citations(report)
    check_tables(report)
    check_checklist(report)
    check_figures(report)
    check_pdf(report)
    report.print()
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
