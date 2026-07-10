#!/usr/bin/env python3
"""Estimate structural loss-aversion parameters for any Qwen eval output.

Unlike the legacy per-treatment estimators, this entry point accepts the output
root (``feature``) and model directory name on the command line. This makes it
suitable for checkpoint comparisons and the OOD new-goods evaluation suite.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from core_exp_refactored import LossAversionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True, help="Directory name under --feature.")
    parser.add_argument("--feature", default="baseline", help="Eval output root, e.g. baseline or ood.")
    parser.add_argument("--robust_model", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--temperature", type=float, default=1.0, help="Structural estimator T; use 1 for logprobs.")
    parser.add_argument("--starting", default="ols", choices=["ols", "zero", "prior", "nls"])
    parser.add_argument("--input_x", default="loss_aversion_X.json")
    parser.add_argument("--input_y", default="loss_aversion_Y.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = PROJECT_ROOT / args.feature / args.model_name
    for filename in (args.input_x, args.input_y):
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluator output: {path}")

    model = LossAversionModel(
        Model_name=args.model_name,
        feature=args.feature,
        robust_model=args.robust_model,
        input_X=args.input_x,
        input_Y=args.input_y,
        T=args.temperature,
        starting=args.starting,
    )

    print("=" * 70)
    print(f"Initializing parameters: {args.feature}/{args.model_name}")
    print("=" * 70)
    model.initialize_parameters()

    print("=" * 70)
    print("Running Model A NLS estimation")
    print("=" * 70)
    model.ModelANLS()
    model.Present(which_model="A")

    print("=" * 70)
    print("Writing utilities, fitted choice probabilities, and raw counts")
    print("=" * 70)
    model.calculate_utility_of_each_goods(which_model="A")
    model.calculate_delta_and_delta_tilder(which_model="A")
    model.choice_prob_from_model(which_model="A")
    model.raw_choice_counts()

    print("=" * 70)
    print(f"Complete: {args.feature}/{args.model_name}/Model_{args.robust_model}")
    print("=" * 70)


if __name__ == "__main__":
    main()
