#!/usr/bin/env python3
"""Regenerate the AAAI result tables inside the single-file manuscript.

AAAI's author kit requires one paper .tex file rather than section files joined
with \\input. This script therefore replaces bounded AUTO blocks in main.tex.

Usage:
    python scripts/generate_tables.py --check
    python scripts/generate_tables.py --write
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "results_registry.json"
MANUSCRIPT = ROOT / "main.tex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Fail if generated blocks are stale.")
    mode.add_argument("--write", action="store_true", help="Rewrite generated blocks in main.tex.")
    return parser.parse_args()


def tex_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    rendered = f"{abs(value):.{digits}f}"
    return f"$-{rendered}$" if value < 0 else rendered


def tex_percent(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}\\%"


def result_cells(result: dict | None, include_w: bool) -> list[str]:
    if not result:
        return ["--"] * (7 if include_w else 6)
    cells = [
        tex_number(result.get("lambda")),
        tex_number(result.get("eta")),
    ]
    if include_w:
        cells.append(tex_number(result.get("w")))
    cells.extend([
        tex_percent(result.get("consistency_pct")),
        tex_percent(result.get("keep_both_pct")),
        tex_percent(result.get("trade_both_pct")),
        tex_percent(result.get("preference_agreement_pct")),
    ])
    return cells


def result_table(models: list[dict], split: str) -> str:
    include_w = split == "id"
    rows = []
    for model in models:
        cells = result_cells(model.get(split), include_w)
        rows.append(f"{model['label']} & " + " & ".join(cells) + r" \\")

    if split == "id":
        caption = (
            "Configuration-validation results on \\texttt{test\\_goods}. Every row\n"
            "uses the same local base weights, 9,890 cases, scorer, and estimator. These are\n"
            "matched training contrasts, but the post-training estimates are\n"
            "validation-only because this split informed reward construction and checkpoint\n"
            "selection. ``Cons.'' selects the same final good under both endowments;\n"
            "``Keep'' and ``Trade'' denote keep-both and trade-both. $W$ is mean\n"
            "pseudo-utility alignment,\n"
            "$w_q=u_{\\mathrm{chosen}}/\\max(u_1,u_2)$, using one shared frozen Qwen-base\n"
            "utility reference. It is a descriptive quantity added after the seed\n"
            "pre-registration. Preference agreement is reserved for the trained policy's\n"
            "agreement with frozen ownership-free preferences, not the 71.1\\% reward-signal\n"
            "validation. Dashes mark required unfinished runs or metrics."
        )
        label = "tab:id-main"
    else:
        caption = (
            "Frozen 50-good OOD results. OOD models are compared with the OOD base\n"
            "on the same suite, never with the ID base. Both confirmatory seeds pass the\n"
            "frozen OOD and non-degeneracy gates. The reward-source rows illustrate a\n"
            "trade-off between $\\widehat\\lambda$, $\\widehat\\eta$, and consistency.\n"
            "Dashes mark required unfinished runs or metrics."
        )
        label = "tab:ood-main"

    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrrrrrr}" if include_w else "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            (
                "Model & $\\widehat\\lambda$ & $\\widehat\\eta$ & $W$ & Cons. & Keep & Trade & Pref. agr. \\\\"
                if include_w
                else "Model & $\\widehat\\lambda$ & $\\widehat\\eta$ & Cons. & Keep & Trade & Pref. agr. \\\\"
            ),
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table*}",
        ]
    )


def capability_table(registry: dict) -> str:
    rows = []
    for item in registry["capability"]:
        rows.append(
            f"{item['label']} & {tex_percent(item.get('gsm8k_pct'))} & "
            f"{tex_percent(item.get('ifeval_pct'))} \\\\"
        )
    paired = registry["gsm8k_paired"]
    low, high = paired["bootstrap_95_ci_percentage_points"]
    delta = paired["delta_percentage_points"]
    p_value = paired["mcnemar_exact_p"]
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "Model & GSM8K & IFEval \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            (
                "\\caption{Capability retention. GSM8K uses a paired "
                f"{paired['n']:,}-item evaluation:\n"
                f"$\\Delta={delta:.2f}$ percentage points, paired bootstrap 95\\% CI\n"
                f"$[{low:.2f},{high:+.2f}]$, and exact McNemar $p={p_value:.3f}$. "
                "IFEval is pending.}"
            ),
            "\\label{tab:capability}",
            "\\end{table}",
        ]
    )


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(
        rf"(% BEGIN AUTO {re.escape(name)}\n).*?(\n% END AUTO {re.escape(name)})",
        flags=re.DOTALL,
    )
    updated, count = pattern.subn(
        lambda match: match.group(1) + body + match.group(2),
        text,
    )
    if count != 1:
        raise RuntimeError(f"Expected one AUTO block named {name}, found {count}.")
    return updated


def render() -> tuple[str, str]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    current = MANUSCRIPT.read_text(encoding="utf-8")
    expected = replace_block(current, "ID_TABLE", result_table(registry["models"], "id"))
    expected = replace_block(expected, "OOD_TABLE", result_table(registry["models"], "ood"))
    expected = replace_block(expected, "CAPABILITY_TABLE", capability_table(registry))
    return current, expected


def main() -> int:
    args = parse_args()
    current, expected = render()
    if args.check:
        if current != expected:
            print("Generated table blocks are stale. Run with --write.")
            return 1
        print("Generated table blocks are current.")
        return 0

    MANUSCRIPT.write_text(expected, encoding="utf-8")
    print(f"Updated {MANUSCRIPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
