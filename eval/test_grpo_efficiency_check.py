#!/usr/bin/env python3
"""Regression checks for the portable GRPO-efficiency JSON comparison."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from analyze_grpo_efficiency import numeric_differences  # noqa: E402


def main() -> int:
    checks = [
        (not numeric_differences({"x": 1.0}, {"x": 1.0 + 1e-13}),
         "sub-tolerance float noise is accepted"),
        (not numeric_differences({"x": 0.0}, {"x": 1e-16}),
         "near-zero float noise is accepted"),
        (bool(numeric_differences({"x": 1.0}, {"x": 1.01})),
         "material numeric drift is rejected"),
        (bool(numeric_differences({"x": True}, {"x": 1})),
         "booleans are not treated as numbers"),
        (bool(numeric_differences({"x": [1]}, {"x": [1, 2]})),
         "list-length drift is rejected"),
        (bool(numeric_differences({"x": "a"}, {"x": "b"})),
         "string drift is rejected"),
    ]
    failed = [label for ok, label in checks if not ok]
    if failed:
        for label in failed:
            print(f"FAIL: {label}")
        return 1
    print(f"OK — {len(checks)} GRPO-efficiency comparison checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
