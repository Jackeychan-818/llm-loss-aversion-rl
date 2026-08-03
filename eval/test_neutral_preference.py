#!/usr/bin/env python3
"""Unit tests for neutral-preference forms + canonicalization (safeguard 9).

Hand-constructed cases with known goods; verify A/B canonicalization tracks the
physical good under order counterbalancing, no ownership framing leaks in, and
the manifest is deterministic.

    python3 eval/test_neutral_preference.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from neutral_preference import (  # noqa: E402
    Goods, build_neutral_forms, canonical_choice, ORDERS, PARAPHRASES,
)

_fail: list[str] = []
_pass = 0


def check(name, cond, detail=""):
    global _pass
    if cond:
        _pass += 1
        print(f"  PASS  {name}")
    else:
        _fail.append(f"{name}: {detail}")
        print(f"  FAIL  {name}: {detail}")


g = Goods()
X, Y, CODE = 0, 6, 16      # loaf(0) vs chocolate(6)
forms = build_neutral_forms(g, 61, X, Y, CODE)

check("forms/case == orders x paraphrases",
      len(forms) == len(ORDERS) * len(PARAPHRASES) == 4, str(len(forms)))

# order counterbalancing: in AX_BY, A=X; in AY_BX, A=Y
axby = [f for f in forms if f.order == "AX_BY"]
aybx = [f for f in forms if f.order == "AY_BX"]
check("AX_BY: A->X, B->Y",
      all(canonical_choice(f, "A") == X and canonical_choice(f, "B") == Y for f in axby), "")
check("AY_BX: A->Y, B->X",
      all(canonical_choice(f, "A") == Y and canonical_choice(f, "B") == X for f in aybx), "")
check("same token 'A' picks different physical good across orders",
      canonical_choice(axby[0], "A") != canonical_choice(aybx[0], "A"), "order swap ineffective")

# both goods named; NO ownership/endowment wording
for f in forms:
    p = f.prompt.lower()
    if "loaf of bread" not in p or "chocolate bar" not in p:
        _fail.append(f"{f.form_id}: missing a good name")
    for banned in ("keep", "trade", "you receive", "you currently hold", "endow"):
        if banned in p:
            _fail.append(f"{f.form_id}: ownership wording leaked ({banned!r})")
check("both goods named, no ownership wording", not _fail, "; ".join(_fail[:3]))

check("unknown token -> None", canonical_choice(forms[0], "maybe") is None, "")

# deterministic manifest
r = subprocess.run([sys.executable, str(ROOT / "data" / "neutral_preference" / "build_neutral_manifest.py"), "--check"],
                   capture_output=True, text=True)
check("neutral manifest --check byte-identical", r.returncode == 0, (r.stdout + r.stderr)[-200:])

print(f"\n{_pass} passed, {len(_fail)} failed")
if _fail:
    for f in _fail:
        print("  -", f)
    raise SystemExit(1)
print("ALL NEUTRAL-PREFERENCE TESTS PASSED")
