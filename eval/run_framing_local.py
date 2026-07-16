#!/usr/bin/env python3
"""
Evaluate a local Hugging Face causal LM, with an optional PEFT LoRA adapter,
on the paired positive/negative framing-effect benchmark.

The script has four useful modes:

1. Prepare a portable JSON benchmark from the original Excel workbook:

   python eval/run_framing_local.py \
     --prepare_xlsx "/path/to/framing effect test cases.xlsx" \
     --data_file data/framing_effects_23prob.json \
     --prepare_only

2. Smoke-test benchmark construction without loading a model:

   python eval/run_framing_local.py \
     --data_file data/framing_effects_23prob.json \
     --limit_scenarios 10 \
     --probability_subset 0.10,0.30,0.50,0.70,0.90 \
     --dry_run

3. Evaluate the exact local base:

   python eval/run_framing_local.py \
     --model_path models/Qwen2.5-7B-Instruct \
     --model_name Qwen-7B-Base \
     --data_file data/framing_effects_23prob.json \
     --batch_size 8 \
     --yes

4. Evaluate the selected GRPO adapter and compare it with the base:

   python eval/run_framing_local.py \
     --model_path models/Qwen2.5-7B-Instruct \
     --adapter_path checkpoints/grpo_qwen_delta/checkpoint-8000 \
     --model_name Qwen-7B-GRPO-step8000 \
     --data_file data/framing_effects_23prob.json \
     --compare_to framing/Qwen-7B-Base/single_word/predictions.json \
     --batch_size 8 \
     --yes

The primary prompt requests a single-word Yes/No answer, aligned with the GRPO
training and the local loss-aversion evaluator. ``--prompt_style original``
reproduces the older long-answer framing prompt as a secondary robustness
condition, while still scoring exact teacher-forced Yes/No continuations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER_RE = re.compile(r"X{1,2}%")
PROBABILITY_COLUMN_RE = re.compile(r"^P(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local base/LoRA models on paired framing effects"
    )

    preparation = parser.add_argument_group("benchmark preparation")
    preparation.add_argument(
        "--prepare_xlsx",
        help="Optional source Excel workbook to convert into portable benchmark JSON",
    )
    preparation.add_argument(
        "--prepare_only",
        action="store_true",
        help="Prepare --data_file and exit without loading a model",
    )
    preparation.add_argument(
        "--sheet_name",
        default=0,
        help="Excel sheet name or zero-based index used during preparation (default: 0)",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model_path",
        default="models/Qwen2.5-7B-Instruct",
        help="Base Hugging Face model path",
    )
    model.add_argument("--adapter_path", default=None, help="Optional PEFT LoRA adapter")
    model.add_argument("--model_name", default="Qwen-7B-GRPO-Framing")
    model.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    model.add_argument("--device_map", default="auto")
    model.add_argument("--batch_size", type=int, default=8)

    benchmark = parser.add_argument_group("benchmark")
    benchmark.add_argument(
        "--data_file",
        default="data/framing_effects_23prob.json",
        help="Prepared framing benchmark JSON",
    )
    benchmark.add_argument(
        "--prompt_style",
        choices=["single_word", "original"],
        default="single_word",
    )
    benchmark.add_argument(
        "--limit_scenarios",
        type=int,
        default=None,
        help="Use only the first N scenarios for a smoke test",
    )
    benchmark.add_argument(
        "--probability_subset",
        default=None,
        help=(
            "Comma- or colon-separated underlying probabilities, "
            "e.g. 0.1,0.3,0.5 or 0.1:0.3:0.5"
        ),
    )

    output = parser.add_argument_group("output and analysis")
    output.add_argument("--output_root", default="framing")
    output.add_argument(
        "--predictions_file",
        default=None,
        help="Override the default predictions path; useful with --analyze_only",
    )
    output.add_argument(
        "--compare_to",
        default=None,
        help="Predictions JSON from the matched local base model",
    )
    output.add_argument("--bootstrap_reps", type=int, default=2000)
    output.add_argument("--bootstrap_seed", type=int, default=20260716)
    output.add_argument("--save_every", type=int, default=100)
    output.add_argument(
        "--analyze_only",
        action="store_true",
        help="Recompute metrics from existing predictions without loading a model",
    )
    output.add_argument(
        "--dry_run",
        action="store_true",
        help="Build and summarize cases without loading a model",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace predictions in the selected output directory",
    )
    output.add_argument("--yes", action="store_true", help="Skip confirmation")

    return parser.parse_args()


def resolve(path: Union[str, Path]) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_excel_cell(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def prepare_benchmark_from_xlsx(
    source_path: Path, output_path: Path, sheet_name
) -> dict:
    """Convert the spreadsheet into 120 explicit paired scenario templates."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Preparing from Excel requires pandas and openpyxl. "
            "Evaluation from an existing JSON benchmark does not."
        ) from exc

    if isinstance(sheet_name, str) and sheet_name.isdigit():
        sheet_name = int(sheet_name)

    frame = pd.read_excel(source_path, sheet_name=sheet_name)
    required = {"Domains", "Framing", "Questions"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Workbook is missing required columns: {sorted(missing)}")

    probability_columns = sorted(
        (
            (int(match.group(1)), column)
            for column in frame.columns
            if (match := PROBABILITY_COLUMN_RE.match(str(column)))
        ),
        key=lambda item: item[0],
    )
    if not probability_columns:
        raise ValueError("No P1, P2, ... probability columns were found")
    probability_column_names = [column for _, column in probability_columns]

    if len(frame) % 2 != 0:
        raise ValueError(f"Expected an even number of framing rows, found {len(frame)}")

    first_positive = frame.iloc[0]
    probabilities = [float(first_positive[column]) for column in probability_column_names]
    scenarios = []
    mismatched_domains = []

    for pair_start in range(0, len(frame), 2):
        positive = frame.iloc[pair_start]
        negative = frame.iloc[pair_start + 1]
        positive_frame = str(positive["Framing"]).strip().lower()
        negative_frame = str(negative["Framing"]).strip().lower()
        if positive_frame != "positive" or negative_frame != "negative":
            raise ValueError(
                f"Rows {pair_start + 2}-{pair_start + 3} are not a positive/negative pair"
            )

        positive_template = str(positive["Questions"])
        negative_template = str(negative["Questions"])
        if not PLACEHOLDER_RE.search(positive_template):
            raise ValueError(f"Positive row {pair_start + 2} lacks an X% placeholder")
        if not PLACEHOLDER_RE.search(negative_template):
            raise ValueError(f"Negative row {pair_start + 3} lacks an X% placeholder")

        for column, underlying_probability in zip(
            probability_column_names, probabilities
        ):
            positive_probability = float(positive[column])
            negative_probability = float(negative[column])
            if not math.isclose(
                positive_probability, underlying_probability, abs_tol=1e-9
            ):
                raise ValueError(
                    f"Positive probability grid differs at row {pair_start + 2}, {column}"
                )
            if not math.isclose(
                positive_probability + negative_probability, 1.0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"Frame probabilities are not complementary at "
                    f"rows {pair_start + 2}-{pair_start + 3}, {column}"
                )

        positive_domain = clean_excel_cell(positive["Domains"])
        negative_domain = clean_excel_cell(negative["Domains"])
        domain_mismatch = bool(
            positive_domain
            and negative_domain
            and positive_domain != negative_domain
        )

        # Ten boundary pairs in the source workbook give the positive row the
        # preceding domain and the negative row the new/content-matching domain.
        # Preserve both source labels and use the negative label only when they
        # disagree, so the correction is explicit and auditable.
        domain = (
            negative_domain
            if domain_mismatch
            else positive_domain or negative_domain or "Unknown"
        )

        scenario_id = pair_start // 2 + 1
        if domain_mismatch:
            mismatched_domains.append(
                {
                    "scenario_id": scenario_id,
                    "positive_domain": positive_domain,
                    "negative_domain": negative_domain,
                }
            )

        scenarios.append(
            {
                "scenario_id": scenario_id,
                "domain": domain,
                "positive_template": positive_template,
                "negative_template": negative_template,
                "source_positive_domain": positive_domain,
                "source_negative_domain": negative_domain,
                "domain_label_mismatch": domain_mismatch,
                "source_excel_rows": [pair_start + 2, pair_start + 3],
            }
        )

    benchmark = {
        "benchmark_name": "Framing LLMs paired framing effects",
        "version": 1,
        "source_file": source_path.name,
        "source_sha256": sha256_file(source_path),
        "domain_resolution": (
            "When positive/negative source labels disagree, use the negative-row "
            "label and retain both original labels."
        ),
        "probabilities": probabilities,
        "scenario_count": len(scenarios),
        "expanded_prompt_count": len(scenarios) * len(probabilities) * 2,
        "domain_label_mismatches": mismatched_domains,
        "scenarios": scenarios,
    }
    dump_json(output_path, benchmark)
    return benchmark


def parse_probability_subset(
    raw: Optional[str], available: Sequence[float]
) -> List[float]:
    if not raw:
        return list(available)
    requested = [
        float(piece.strip())
        for piece in re.split(r"[:,]", raw)
        if piece.strip()
    ]
    selected = []
    for value in requested:
        match = next(
            (
                available_value
                for available_value in available
                if math.isclose(value, available_value, abs_tol=1e-9)
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"Requested probability {value} is not in benchmark grid {available}"
            )
        selected.append(match)
    return selected


def make_prompt(question: str, prompt_style: str) -> str:
    if prompt_style == "single_word":
        return (
            f"Please consider the following option: {question}\n"
            "Answer with the single word Yes or No."
        )
    if prompt_style == "original":
        return (
            f"Please consider the following option: {question}\n"
            "Your answer needs to have two parts:\n"
            "A yes / no recommendation. You need to pick one of the two options. "
            "Even if you are unsure about the answer, pick the one that you think "
            "is more likely good advice.\n\n"
            "Explain your reasoning process in detail. List the key facts used in "
            "your reasoning, presented as declarative sentences and enumerated."
        )
    raise ValueError(f"Unknown prompt style: {prompt_style}")


def instantiate_question(template: str, displayed_probability: float) -> str:
    percentage = int(round(displayed_probability * 100))
    return PLACEHOLDER_RE.sub(f"{percentage}%", template)


def build_cases(
    benchmark: dict,
    prompt_style: str,
    limit_scenarios: Optional[int],
    probability_subset: Optional[str],
) -> List[dict]:
    scenarios = list(benchmark["scenarios"])
    if limit_scenarios is not None:
        if limit_scenarios <= 0:
            raise ValueError("--limit_scenarios must be positive")
        scenarios = scenarios[:limit_scenarios]

    available_probabilities = [float(value) for value in benchmark["probabilities"]]
    selected_probabilities = parse_probability_subset(
        probability_subset, available_probabilities
    )
    probability_indices = {
        value: available_probabilities.index(value) + 1
        for value in selected_probabilities
    }

    cases = []
    for scenario in scenarios:
        scenario_id = int(scenario["scenario_id"])
        for underlying_probability in selected_probabilities:
            probability_index = probability_indices[underlying_probability]
            for frame in ("positive", "negative"):
                displayed_probability = (
                    underlying_probability
                    if frame == "positive"
                    else 1.0 - underlying_probability
                )
                template = scenario[f"{frame}_template"]
                question = instantiate_question(template, displayed_probability)
                prompt = make_prompt(question, prompt_style)
                cases.append(
                    {
                        "case_id": (
                            f"s{scenario_id:03d}_p{probability_index:02d}_{frame}"
                        ),
                        "scenario_id": scenario_id,
                        "probability_index": probability_index,
                        "domain": scenario["domain"],
                        "frame": frame,
                        "underlying_probability": underlying_probability,
                        "displayed_probability": displayed_probability,
                        "question": question,
                        "prompt": prompt,
                    }
                )
    return cases


def safe_logit(probability: float, epsilon: float = 1e-6) -> float:
    probability = min(max(probability, epsilon), 1.0 - epsilon)
    return math.log(probability / (1.0 - probability))


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.fmean(values) if values else None


def format_number(value: Optional[float], digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def percentile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_complete_pairs(rows: Sequence[dict]) -> List[dict]:
    indexed: Dict[Tuple[int, float], dict] = defaultdict(dict)
    for row in rows:
        key = (
            int(row["scenario_id"]),
            round(float(row["underlying_probability"]), 12),
        )
        indexed[key][row["frame"]] = row

    pairs = []
    for (scenario_id, probability), frames in sorted(indexed.items()):
        if "positive" not in frames or "negative" not in frames:
            continue
        positive = frames["positive"]
        negative = frames["negative"]
        positive_yes = float(positive["p_yes"])
        negative_yes = float(negative["p_yes"])
        positive_hard = str(positive["hard_choice"]).lower() == "yes"
        negative_hard = str(negative["hard_choice"]).lower() == "yes"
        pairs.append(
            {
                "scenario_id": scenario_id,
                "domain": positive["domain"],
                "underlying_probability": probability,
                "probability_gap_negative_minus_positive": (
                    negative_yes - positive_yes
                ),
                "absolute_probability_gap": abs(negative_yes - positive_yes),
                "absolute_log_odds_gap": abs(
                    safe_logit(negative_yes) - safe_logit(positive_yes)
                ),
                "hard_flip": positive_hard != negative_hard,
                "classical_flip": positive_hard and not negative_hard,
                "reverse_flip": (not positive_hard) and negative_hard,
                "same_yes": positive_hard and negative_hard,
                "same_no": (not positive_hard) and (not negative_hard),
            }
        )
    return pairs


def pair_metric_summary(pairs: Sequence[dict]) -> dict:
    if not pairs:
        return {
            "complete_pairs": 0,
            "mean_probability_gap_negative_minus_positive": None,
            "mean_absolute_probability_gap": None,
            "mean_absolute_log_odds_gap": None,
            "hard_flip_rate": None,
            "classical_flip_rate": None,
            "reverse_flip_rate": None,
            "same_yes_rate": None,
            "same_no_rate": None,
        }
    return {
        "complete_pairs": len(pairs),
        "mean_probability_gap_negative_minus_positive": mean(
            pair["probability_gap_negative_minus_positive"] for pair in pairs
        ),
        "mean_absolute_probability_gap": mean(
            pair["absolute_probability_gap"] for pair in pairs
        ),
        "mean_absolute_log_odds_gap": mean(
            pair["absolute_log_odds_gap"] for pair in pairs
        ),
        "hard_flip_rate": mean(float(pair["hard_flip"]) for pair in pairs),
        "classical_flip_rate": mean(
            float(pair["classical_flip"]) for pair in pairs
        ),
        "reverse_flip_rate": mean(float(pair["reverse_flip"]) for pair in pairs),
        "same_yes_rate": mean(float(pair["same_yes"]) for pair in pairs),
        "same_no_rate": mean(float(pair["same_no"]) for pair in pairs),
    }


def monotonicity_summary(rows: Sequence[dict]) -> dict:
    curves: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    for row in rows:
        curves[(int(row["scenario_id"]), row["frame"])].append(row)

    probability_violations = 0
    hard_violations = 0
    transitions = 0
    nonmonotonic_probability_curves = 0
    nonmonotonic_hard_curves = 0

    for curve_rows in curves.values():
        ordered = sorted(curve_rows, key=lambda row: row["underlying_probability"])
        curve_probability_violation = False
        curve_hard_violation = False
        for previous, current in zip(ordered, ordered[1:]):
            transitions += 1
            if float(current["p_yes"]) + 1e-12 < float(previous["p_yes"]):
                probability_violations += 1
                curve_probability_violation = True
            if (
                str(previous["hard_choice"]).lower() == "yes"
                and str(current["hard_choice"]).lower() == "no"
            ):
                hard_violations += 1
                curve_hard_violation = True
        nonmonotonic_probability_curves += int(curve_probability_violation)
        nonmonotonic_hard_curves += int(curve_hard_violation)

    curve_count = len(curves)
    return {
        "curve_count": curve_count,
        "adjacent_transitions": transitions,
        "probability_violation_count": probability_violations,
        "probability_violation_rate": (
            probability_violations / transitions if transitions else None
        ),
        "hard_choice_violation_count": hard_violations,
        "hard_choice_violation_rate": hard_violations / transitions if transitions else None,
        "nonmonotonic_probability_curve_rate": (
            nonmonotonic_probability_curves / curve_count if curve_count else None
        ),
        "nonmonotonic_hard_choice_curve_rate": (
            nonmonotonic_hard_curves / curve_count if curve_count else None
        ),
    }


def cluster_bootstrap(
    pairs: Sequence[dict], reps: int, seed: int
) -> Dict[str, List[Optional[float]]]:
    if reps <= 0 or not pairs:
        return {}

    metric_names = {
        "mean_probability_gap_negative_minus_positive": (
            "probability_gap_negative_minus_positive"
        ),
        "mean_absolute_probability_gap": "absolute_probability_gap",
        "mean_absolute_log_odds_gap": "absolute_log_odds_gap",
        "hard_flip_rate": "hard_flip",
        "classical_flip_rate": "classical_flip",
        "reverse_flip_rate": "reverse_flip",
    }
    by_scenario: Dict[int, List[dict]] = defaultdict(list)
    for pair in pairs:
        by_scenario[int(pair["scenario_id"])].append(pair)

    scenario_metric_values = {}
    for scenario_id, scenario_pairs in by_scenario.items():
        scenario_metric_values[scenario_id] = {
            output_name: mean(
                float(pair[source_name]) for pair in scenario_pairs
            )
            for output_name, source_name in metric_names.items()
        }

    scenario_ids = sorted(scenario_metric_values)
    random_generator = random.Random(seed)
    draws = {metric_name: [] for metric_name in metric_names}
    for _ in range(reps):
        sampled = [
            random_generator.choice(scenario_ids) for _ in range(len(scenario_ids))
        ]
        for metric_name in metric_names:
            draws[metric_name].append(
                statistics.fmean(
                    scenario_metric_values[scenario_id][metric_name]
                    for scenario_id in sampled
                )
            )

    return {
        metric_name: [
            percentile(metric_draws, 0.025),
            percentile(metric_draws, 0.975),
        ]
        for metric_name, metric_draws in draws.items()
    }


def summarize_predictions(
    rows: Sequence[dict], bootstrap_reps: int, bootstrap_seed: int
) -> dict:
    pairs = build_complete_pairs(rows)
    frames = {
        frame: [row for row in rows if row["frame"] == frame]
        for frame in ("positive", "negative")
    }
    domains = sorted({row["domain"] for row in rows})
    domain_metrics = {}
    for domain in domains:
        domain_pairs = [pair for pair in pairs if pair["domain"] == domain]
        domain_metrics[domain] = pair_metric_summary(domain_pairs)

    scenario_ids = {int(row["scenario_id"]) for row in rows}
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "scenario_count": len(scenario_ids),
        "frame_counts": {frame: len(frame_rows) for frame, frame_rows in frames.items()},
        "mean_yes_probability_by_frame": {
            frame: mean(float(row["p_yes"]) for row in frame_rows)
            for frame, frame_rows in frames.items()
        },
        "hard_yes_rate_by_frame": {
            frame: mean(
                float(str(row["hard_choice"]).lower() == "yes")
                for row in frame_rows
            )
            for frame, frame_rows in frames.items()
        },
        "paired_framing": pair_metric_summary(pairs),
        "monotonicity": monotonicity_summary(rows),
        "cluster_bootstrap_95_ci": cluster_bootstrap(
            pairs, bootstrap_reps, bootstrap_seed
        ),
        "domain_metrics": domain_metrics,
    }


def compare_metrics(current: dict, baseline: dict) -> dict:
    current_pair = current["paired_framing"]
    baseline_pair = baseline["paired_framing"]
    current_mono = current["monotonicity"]
    baseline_mono = baseline["monotonicity"]

    baseline_directional = baseline_pair[
        "mean_probability_gap_negative_minus_positive"
    ]
    current_directional = current_pair[
        "mean_probability_gap_negative_minus_positive"
    ]
    return {
        "interpretation": (
            "Positive reduction values favor the current model over the baseline."
        ),
        "baseline": {
            "paired_framing": baseline_pair,
            "monotonicity": baseline_mono,
        },
        "current": {
            "paired_framing": current_pair,
            "monotonicity": current_mono,
        },
        "reductions": {
            "absolute_directional_probability_gap": (
                abs(baseline_directional) - abs(current_directional)
            ),
            "mean_absolute_probability_gap": (
                baseline_pair["mean_absolute_probability_gap"]
                - current_pair["mean_absolute_probability_gap"]
            ),
            "mean_absolute_log_odds_gap": (
                baseline_pair["mean_absolute_log_odds_gap"]
                - current_pair["mean_absolute_log_odds_gap"]
            ),
            "hard_flip_rate": (
                baseline_pair["hard_flip_rate"] - current_pair["hard_flip_rate"]
            ),
            "probability_monotonicity_violation_rate": (
                baseline_mono["probability_violation_rate"]
                - current_mono["probability_violation_rate"]
            ),
            "hard_choice_monotonicity_violation_rate": (
                baseline_mono["hard_choice_violation_rate"]
                - current_mono["hard_choice_violation_rate"]
            ),
        },
    }


def dtype_from_name(torch, name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def candidate_token_ids(tokenizer, text: str) -> List[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Candidate {text!r} encoded to no tokens")
    return token_ids


def score_yes_no(model, tokenizer, prompt_texts: Sequence[str]) -> List[Tuple[float, float]]:
    """Score exact Yes and No continuation sequences and normalize over the pair."""
    import torch

    yes_ids = candidate_token_ids(tokenizer, "Yes")
    no_ids = candidate_token_ids(tokenizer, "No")
    encoded_prompts = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for prompt in prompt_texts
    ]

    full_ids = []
    metadata = []
    for prompt_index, prompt_ids in enumerate(encoded_prompts):
        for label, candidate_ids in (("yes", yes_ids), ("no", no_ids)):
            full_ids.append(prompt_ids + candidate_ids)
            metadata.append(
                (prompt_index, label, len(prompt_ids), len(candidate_ids))
            )

    pad_id = tokenizer.pad_token_id
    max_length = max(len(token_ids) for token_ids in full_ids)
    input_ids = torch.full(
        (len(full_ids), max_length), pad_id, dtype=torch.long
    )
    attention_mask = torch.zeros(
        (len(full_ids), max_length), dtype=torch.long
    )
    for row_index, token_ids in enumerate(full_ids):
        input_ids[row_index, : len(token_ids)] = torch.tensor(
            token_ids, dtype=torch.long
        )
        attention_mask[row_index, : len(token_ids)] = 1

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        log_probabilities = torch.log_softmax(logits, dim=-1)

    scores = [
        {"yes": float("-inf"), "no": float("-inf")}
        for _ in prompt_texts
    ]
    for row_index, (prompt_index, label, prompt_length, candidate_length) in enumerate(
        metadata
    ):
        score = 0.0
        for offset in range(candidate_length):
            token_position = prompt_length + offset
            token_id = int(input_ids[row_index, token_position])
            score += float(
                log_probabilities[row_index, token_position - 1, token_id]
            )
        scores[prompt_index][label] = score

    probabilities = []
    for score in scores:
        maximum = max(score["yes"], score["no"])
        yes = math.exp(score["yes"] - maximum)
        no = math.exp(score["no"] - maximum)
        normalizer = yes + no
        probabilities.append((yes / normalizer, no / normalizer))
    return probabilities


def prediction_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    output_root = resolve(args.output_root)
    output_dir = output_root / args.model_name / args.prompt_style
    predictions_path = (
        resolve(args.predictions_file)
        if args.predictions_file
        else output_dir / "predictions.json"
    )
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"
    comparison_path = output_dir / "comparison_to_base.json"
    return predictions_path, metrics_path, manifest_path, comparison_path


def run_analysis(
    args: argparse.Namespace,
    predictions_path: Path,
    metrics_path: Path,
    comparison_path: Path,
) -> dict:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")
    rows = load_json(predictions_path)
    metrics = summarize_predictions(
        rows, args.bootstrap_reps, args.bootstrap_seed
    )
    dump_json(metrics_path, metrics)

    paired = metrics["paired_framing"]
    print("\n=== Framing metrics ===")
    print(f"Rows:                              {metrics['row_count']}")
    print(f"Complete frame pairs:              {paired['complete_pairs']}")
    print(
        "Mean P(Yes) gap, negative-positive: "
        f"{format_number(paired['mean_probability_gap_negative_minus_positive'])}"
    )
    print(
        "Mean absolute probability gap:     "
        f"{format_number(paired['mean_absolute_probability_gap'])}"
    )
    print(
        "Mean absolute log-odds gap:        "
        f"{format_number(paired['mean_absolute_log_odds_gap'])}"
    )
    print(
        "Hard-choice flip rate:              "
        f"{format_percent(paired['hard_flip_rate'])}"
    )
    print(
        "Hard monotonicity violation rate:  "
        f"{format_percent(metrics['monotonicity']['hard_choice_violation_rate'])}"
    )
    print(f"Metrics saved to: {metrics_path}")

    if args.compare_to:
        baseline_path = resolve(args.compare_to)
        baseline_rows = load_json(baseline_path)
        current_ids = {row["case_id"] for row in rows}
        baseline_ids = {row["case_id"] for row in baseline_rows}
        if current_ids != baseline_ids:
            raise ValueError(
                "Current and baseline predictions do not contain identical case IDs "
                f"(current={len(current_ids)}, baseline={len(baseline_ids)})"
            )
        baseline_metrics = summarize_predictions(baseline_rows, 0, args.bootstrap_seed)
        comparison = compare_metrics(metrics, baseline_metrics)
        comparison["baseline_predictions"] = str(baseline_path)
        comparison["current_predictions"] = str(predictions_path)
        dump_json(comparison_path, comparison)
        print("\n=== Improvement over matched base ===")
        for name, value in comparison["reductions"].items():
            print(f"{name}: {value:+.6f}")
        print(f"Comparison saved to: {comparison_path}")

    return metrics


def main() -> None:
    args = parse_args()
    data_path = resolve(args.data_file)

    if args.prepare_xlsx:
        source_path = resolve(args.prepare_xlsx)
        benchmark = prepare_benchmark_from_xlsx(
            source_path, data_path, args.sheet_name
        )
        print(
            f"Prepared {benchmark['scenario_count']} scenarios and "
            f"{benchmark['expanded_prompt_count']} prompts at {data_path}"
        )
        print(
            "Domain-label mismatches retained and resolved: "
            f"{len(benchmark['domain_label_mismatches'])}"
        )
        if args.prepare_only:
            return
    elif args.prepare_only:
        raise ValueError("--prepare_only requires --prepare_xlsx")

    predictions_path, metrics_path, manifest_path, comparison_path = (
        prediction_paths(args)
    )

    if args.analyze_only:
        run_analysis(
            args, predictions_path, metrics_path, comparison_path
        )
        return

    if not data_path.exists():
        raise FileNotFoundError(
            f"Benchmark JSON not found: {data_path}. "
            "Use --prepare_xlsx ... --prepare_only first."
        )

    benchmark = load_json(data_path)
    cases = build_cases(
        benchmark,
        args.prompt_style,
        args.limit_scenarios,
        args.probability_subset,
    )
    print("=" * 72)
    print("LOCAL FRAMING-EFFECT EVALUATION")
    print("=" * 72)
    print(f"Benchmark:       {data_path}")
    print(f"Scenarios:       {len({case['scenario_id'] for case in cases})}")
    print(f"Prompts:         {len(cases)}")
    print(f"Prompt style:    {args.prompt_style}")
    print(f"Model path:      {resolve(args.model_path)}")
    print(
        f"Adapter path:    "
        f"{resolve(args.adapter_path) if args.adapter_path else '(none: local base)'}"
    )
    print(f"Predictions:     {predictions_path}")

    if args.dry_run:
        frame_counts = {
            frame: sum(case["frame"] == frame for case in cases)
            for frame in ("positive", "negative")
        }
        print(f"Frame counts:    {frame_counts}")
        print("First two cases:")
        for case in cases[:2]:
            print(json.dumps(case, indent=2, ensure_ascii=False))
        return

    model_path = resolve(args.model_path)
    adapter_path = resolve(args.adapter_path) if args.adapter_path else None
    data_hash = sha256_file(data_path)
    expected_manifest = {
        "model_path": str(model_path),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "model_name": args.model_name,
        "prompt_style": args.prompt_style,
        "benchmark_path": str(data_path),
        "benchmark_sha256": data_hash,
        "dtype": args.dtype,
    }

    existing_rows = []
    if args.overwrite:
        print("Overwrite requested; existing predictions will be replaced.")
        if predictions_path.exists():
            dump_json(predictions_path, [])
    elif predictions_path.exists():
        existing_rows = load_json(predictions_path)
        if not manifest_path.exists():
            raise ValueError(
                f"Predictions exist without a manifest: {predictions_path}. "
                "Use --overwrite only after confirming the file is disposable."
            )
        existing_manifest = load_json(manifest_path)
        mismatches = {
            key: (existing_manifest.get(key), value)
            for key, value in expected_manifest.items()
            if existing_manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Existing output manifest does not match this run. "
                f"Use a new --model_name/output path. Mismatches: {mismatches}"
            )

    completed_ids = {row["case_id"] for row in existing_rows}
    remaining_cases = [
        case for case in cases if case["case_id"] not in completed_ids
    ]
    print(f"Already complete: {len(completed_ids)}")
    print(f"Remaining:        {len(remaining_cases)}")

    if remaining_cases and not args.yes:
        confirmation = input("Proceed? Type 'yes' to continue: ").strip().lower()
        if confirmation != "yes":
            print("Aborted.")
            return

    manifest = dict(expected_manifest)
    manifest.update(
        {
            "created_or_resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_count_requested": len(cases),
            "limit_scenarios": args.limit_scenarios,
            "probability_subset": args.probability_subset,
            "predictions_path": str(predictions_path),
        }
    )
    dump_json(manifest_path, manifest)

    if remaining_cases:
        import torch
        from tqdm import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=dtype_from_name(torch, args.dtype),
            device_map=args.device_map,
            trust_remote_code=True,
        )
        if adapter_path:
            from peft import PeftModel

            print(f"Loading LoRA adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()

        output_by_id = {row["case_id"]: row for row in existing_rows}
        rows_since_save = 0
        for start in tqdm(
            range(0, len(remaining_cases), args.batch_size),
            desc="Scoring framing prompts",
        ):
            batch_cases = remaining_cases[start : start + args.batch_size]
            chat_prompts = [
                apply_chat_template(tokenizer, case["prompt"])
                for case in batch_cases
            ]
            probabilities = score_yes_no(model, tokenizer, chat_prompts)
            for case, (p_yes, p_no) in zip(batch_cases, probabilities):
                row = dict(case)
                row.update(
                    {
                        "p_yes": p_yes,
                        "p_no": p_no,
                        "yes_log_odds": safe_logit(p_yes),
                        "hard_choice": "Yes" if p_yes >= p_no else "No",
                    }
                )
                output_by_id[row["case_id"]] = row
                rows_since_save += 1

            if (
                rows_since_save >= args.save_every
                or start + args.batch_size >= len(remaining_cases)
            ):
                ordered_rows = sorted(
                    output_by_id.values(),
                    key=lambda row: (
                        row["scenario_id"],
                        row["probability_index"],
                        row["frame"],
                    ),
                )
                dump_json(predictions_path, ordered_rows)
                rows_since_save = 0

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump_json(manifest_path, manifest)
    run_analysis(args, predictions_path, metrics_path, comparison_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Completed batches remain saved.", file=sys.stderr)
        raise SystemExit(130)
