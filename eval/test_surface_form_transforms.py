#!/usr/bin/env python3
"""Unit tests for surface-form transforms + canonicalization (safeguard 4).

Hand-constructed cases where the correct final good is KNOWN. Tests every
transformation axis and verifies that answer-token remapping preserves the same
SEMANTIC choice (keep -> endowed good; trade -> offered good) across all 48
forms and both perspectives, so surface changes never silently flip the mapping.

    python3 eval/test_surface_form_transforms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from surface_form_transforms import (  # noqa: E402
    Goods, build_forms, canonical_final_good,
    ANSWER_STYLES, DISPLAY_ORDERS, ATTR_ORDERS, PARAPHRASES, decode_attr,
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


def _action(form, token):
    # replicate canonicalization's token normalisation lookup
    from surface_form_transforms import _norm
    return form.token_to_action.get(_norm(token))


def keep_token(form):
    return next(t for t in form.answer_tokens if _action(form, t) == "keep")


def trade_token(form):
    return next(t for t in form.answer_tokens if _action(form, t) == "trade")


g = Goods()
CASE_ID, X_NUM, Y_NUM, CODE = 61, 0, 6, 16   # loaf of bread (0) vs chocolate bar (6)

# --- 1. factorial completeness ---------------------------------------------
formsX = build_forms(g, CASE_ID, X_NUM, Y_NUM, CODE, "X")
formsY = build_forms(g, CASE_ID, X_NUM, Y_NUM, CODE, "Y")
check("48 forms per perspective", len(formsX) == 48 and len(formsY) == 48,
      f"{len(formsX)},{len(formsY)}")
combos = {(f.answer_style, f.display_order, f.attr_order, f.paraphrase) for f in formsX}
check("all 48 axis combos unique", len(combos) == 48, str(len(combos)))
check("axis coverage",
      len({f.answer_style for f in formsX}) == 4 and len({f.display_order for f in formsX}) == 2
      and len({f.attr_order for f in formsX}) == 2 and len({f.paraphrase for f in formsX}) == 3, "")

# --- 2. keep/trade tokens map to the KNOWN physical good in EVERY form ------
# X-perspective: endowed = loaf(0), offered = chocolate(6).
ok_keep = all(canonical_final_good(f, keep_token(f)) == X_NUM for f in formsX)
ok_trade = all(canonical_final_good(f, trade_token(f)) == Y_NUM for f in formsX)
check("X-persp keep -> endowed(loaf) for all 48 forms", ok_keep, "some form mis-mapped keep")
check("X-persp trade -> offered(chocolate) for all 48 forms", ok_trade, "some form mis-mapped trade")
# Y-perspective: endowed = chocolate(6), offered = loaf(0).
ok_keepY = all(canonical_final_good(f, keep_token(f)) == Y_NUM for f in formsY)
ok_tradeY = all(canonical_final_good(f, trade_token(f)) == X_NUM for f in formsY)
check("Y-persp keep -> endowed(chocolate) for all 48 forms", ok_keepY, "")
check("Y-persp trade -> offered(loaf) for all 48 forms", ok_tradeY, "")

# --- 3. semantic equivalence: 'keep' outcome good is invariant across forms -
keep_goods = {canonical_final_good(f, keep_token(f)) for f in formsX}
trade_goods = {canonical_final_good(f, trade_token(f)) for f in formsX}
check("keep-outcome good invariant across all forms", keep_goods == {X_NUM}, str(keep_goods))
check("trade-outcome good invariant across all forms", trade_goods == {Y_NUM}, str(trade_goods))

# --- 4. label swap tests a position shortcut: same 'A' -> different good ----
ab_kt = next(f for f in formsX if f.answer_style == "ab_kt")
ab_tk = next(f for f in formsX if f.answer_style == "ab_tk")
check("ab_kt 'A' -> keep(endowed loaf)", canonical_final_good(ab_kt, "A") == X_NUM, "")
check("ab_tk 'A' -> trade(offered chocolate)", canonical_final_good(ab_tk, "A") == Y_NUM, "")
check("label swap makes same token 'A' pick different goods",
      canonical_final_good(ab_kt, "A") != canonical_final_good(ab_tk, "A"), "swap not effective")

# --- 5. perspective is the ownership axis: keep picks different physical good
check("keep differs across perspective (ownership)",
      canonical_final_good(keepX := formsX[0], keep_token(keepX)) !=
      canonical_final_good(keepY := formsY[0], keep_token(keepY)), "")

# --- 6. surface axes never change the physical keep/trade goods -------------
for f in formsX:
    if not (f.keep_good == X_NUM and f.trade_good == Y_NUM):
        _fail.append(f"form {f.form_id} has wrong physical goods")
check("display/attr/paraphrase/style never change physical goods",
      all(f.keep_good == X_NUM and f.trade_good == Y_NUM for f in formsX), "")

# --- 7. prompts contain both goods; answer instruction present -------------
loaf, choc = g.name(X_NUM).lower(), g.name(Y_NUM).lower()
check("every prompt names both goods",
      all(loaf in f.prompt.lower() and choc in f.prompt.lower() for f in formsX), "")
check("attr-reversed changes attribute order in text",
      any(f.attr_order == "reversed" for f in formsX)
      and next(f for f in formsX if f.attr_order == "reversed").prompt
      != next(f for f in formsX if f.attr_order == "normal"
              and f.answer_style == "yes_no" and f.display_order == "endowed_first"
              and f.paraphrase == "p0").prompt, "")

# --- 8. parse failure: unknown token -> None -------------------------------
check("unknown token -> None (parse fail)", canonical_final_good(formsX[0], "maybe") is None, "")

# --- 9. decode_attr sanity (matches build_delta) ---------------------------
check("decode_attr(16)", decode_attr(16) == (1, 2, 1, 0), str(decode_attr(16)))

print(f"\n{_pass} passed, {len(_fail)} failed")
if _fail:
    for f in _fail:
        print("  -", f)
    raise SystemExit(1)
print("ALL SURFACE-FORM TRANSFORM TESTS PASSED")
