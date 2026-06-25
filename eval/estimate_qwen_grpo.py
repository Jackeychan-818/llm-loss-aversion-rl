#!/usr/bin/env python3
"""
Estimate lambda_after for Qwen-7B-GRPO from local/offline eval outputs.

Expected inputs:
    baseline/Qwen-7B-GRPO/loss_aversion_X.json
    baseline/Qwen-7B-GRPO/loss_aversion_Y.json
"""

import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_exp_refactored import LossAversionModel

MODEL_NAME = "Qwen-7B-GRPO"
FEATURE = "baseline"
ROBUST_MODEL = 1
T = 1


def main() -> None:
    model = LossAversionModel(
        Model_name=MODEL_NAME,
        feature=FEATURE,
        robust_model=ROBUST_MODEL,
        input_X="loss_aversion_X.json",
        input_Y="loss_aversion_Y.json",
        T=T,
    )

    print("=" * 60)
    print("Initializing Parameters...")
    print("=" * 60)
    model.initialize_parameters()

    print("=" * 60)
    print(f"Running NLS Estimation for {MODEL_NAME}")
    print("=" * 60)
    model.ModelANLS()

    print("\n" + "=" * 60)
    print("Results Presentation - Model A (NLS)")
    print("=" * 60)
    model.Present(which_model="A")

    print("\n" + "=" * 60)
    print("Calculating Utilities...")
    print("=" * 60)
    model.calculate_utility_of_each_goods(which_model="A")
    model.calculate_delta_and_delta_tilder(which_model="A")

    print("\n" + "=" * 60)
    print("Choice Probability from Model")
    print("=" * 60)
    model.choice_prob_from_model(which_model="A")

    print("\n" + "=" * 60)
    print("Raw Choice Counts")
    print("=" * 60)
    model.raw_choice_counts()

    print("\n" + "=" * 60)
    print(f"ESTIMATION COMPLETE FOR {MODEL_NAME}")
    print(f"Results saved to: {FEATURE}/{MODEL_NAME}/Model_{ROBUST_MODEL}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
