#!/usr/bin/env python3
"""Estimate lambda_after for Qwen-7B-GRPO under the FORCED treatment."""

import os, sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_exp_refactored import LossAversionModel

MODEL_NAME = "Qwen-7B-GRPO"
FEATURE    = "forced"
T          = 1


def main() -> None:
    model = LossAversionModel(
        Model_name=MODEL_NAME, feature=FEATURE,
        robust_model=1,
        input_X="loss_aversion_X.json",
        input_Y="loss_aversion_Y.json",
        T=T,
    )
    print("=" * 60); print("Initializing Parameters..."); print("=" * 60)
    model.initialize_parameters()
    print("=" * 60); print(f"Running NLS — {MODEL_NAME} / {FEATURE}"); print("=" * 60)
    model.ModelANLS()
    model.Present(which_model="A")
    model.calculate_utility_of_each_goods(which_model="A")
    model.calculate_delta_and_delta_tilder(which_model="A")
    model.choice_prob_from_model(which_model="A")
    model.raw_choice_counts()
    print(f"\nResults saved to: {FEATURE}/{MODEL_NAME}/Model_1/")


if __name__ == "__main__":
    main()
