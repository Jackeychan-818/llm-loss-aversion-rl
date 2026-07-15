#!/usr/bin/env python3
"""Estimate structural loss-aversion parameters for any Qwen eval output.

Unlike the legacy per-treatment estimators, this entry point accepts the output
root (``feature``) and model directory name on the command line. This makes it
suitable for checkpoint comparisons and the OOD new-goods evaluation suite.

Note on T (``--link-scale``): T is the scale of the structural link
P(Yes) = 1/(1+exp(-z/T)) — a property of the econometric model, NOT the LLM's
sampling temperature. Evaluation decoding is deterministic greedy
(``do_sample=False`` in run_qwen_local.py), i.e. the zero-temperature decoding
limit; the Yes/No choice is read from teacher-forced logprobs. Model A (NLS)
requires T > 0 (T=1 for logprob data); the T=0 / binary maximum-score regime is
Model B's (``--robust_model 2``).
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
    parser.add_argument(
        "--link-scale", "--link_scale", dest="link_scale", type=float, default=None,
        help="Structural link scale T in P(Yes) = 1/(1+exp(-z/T)) (default 1.0). This is "
             "a parameter of the ECONOMETRIC MODEL, NOT a sampling temperature — "
             "evaluation decoding is deterministic greedy (do_sample=False). Must be > 0 "
             "for Model A (NLS): at T=0 the link becomes a step function, the Jacobian is "
             "zero, and NLS returns a degenerate lambda_hat=0 with zero SEs. Use Model B "
             "(--robust_model 2) for the T=0 / binary maximum-score regime.",
    )
    parser.add_argument(
        "--temperature", dest="temperature", type=float, default=None,
        help="DEPRECATED alias for --link-scale. The name is misleading (it is not a "
             "sampling temperature); retained so existing NSCC scripts keep working.",
    )
    parser.add_argument("--starting", default="ols", choices=["ols", "zero", "prior", "nls"])
    parser.add_argument("--input_x", default="loss_aversion_X.json")
    parser.add_argument("--input_y", default="loss_aversion_Y.json")
    args = parser.parse_args()

    # Resolve --link-scale / deprecated --temperature.
    if args.link_scale is not None and args.temperature is not None:
        if args.link_scale != args.temperature:
            parser.error(
                f"--link-scale ({args.link_scale}) and deprecated --temperature "
                f"({args.temperature}) disagree; pass only --link-scale."
            )
    elif args.temperature is not None:
        print("[DEPRECATION] --temperature is a misleading name for the structural link "
              "scale and will be removed; use --link-scale instead. (It is NOT a sampling "
              "temperature: evaluation decoding is greedy/deterministic.)", file=sys.stderr)
        args.link_scale = args.temperature
    if args.link_scale is None:
        args.link_scale = 1.0

    # Guard the one incoherent combination: Model A (NLS) needs a differentiable link.
    if args.robust_model == 1 and args.link_scale <= 0:
        parser.error(
            f"--link-scale must be > 0 for Model A (NLS); got {args.link_scale}. At T=0 the "
            "link P=1(z>0) is a step function -> zero Jacobian -> NLS cannot converge and "
            "returns a degenerate lambda_hat=0 with zero SEs. For the T=0 / binary "
            "maximum-score regime use Model B: --robust_model 2 --link-scale 0."
        )
    return args


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
        T=args.link_scale,
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
