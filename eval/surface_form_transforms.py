#!/usr/bin/env python3
"""Surface-form transforms + canonicalization for the SFT-vs-sign-only diagnostic.

Generates, for one endowment case + perspective, the 48 semantically EQUIVALENT
surface forms and provides a form-driven canonicalization from a raw answer token
to the selected FINAL PHYSICAL GOOD. No torch; pure prompt/logic so it is unit
testable without a GPU.

Axes (4 x 2 x 2 x 3 = 48):
  answer_style   : yes_no | keep_trade | ab_kt (A=keep,B=trade) | ab_tk (A=trade,B=keep)
  display_order  : endowed_first | offered_first
  attr_order     : normal | reversed
  paraphrase     : p0 | p1 | p2

Canonicalization is form-driven: each Form carries its two valid answer tokens,
a token->action ({keep,trade}) map, and the physical keep_good / trade_good, so
`canonical_final_good(form, token)` returns the physical good index regardless of
labels, order, wording, or perspective.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ANSWER_STYLES = ["yes_no", "keep_trade", "ab_kt", "ab_tk"]
DISPLAY_ORDERS = ["endowed_first", "offered_first"]
ATTR_ORDERS = ["normal", "reversed"]
PARAPHRASES = ["p0", "p1", "p2"]


class Goods:
    """Minimal goods accessor over everyday_goods_full.json (flat index 0..99)."""

    def __init__(self, path: Path | None = None):
        data = json.load(open(path or ROOT / "everyday_goods_full.json"))
        self.names: list[str] = []
        self.attr: dict[str, list[str]] = {}
        self.vals: dict[str, list[list[str]]] = {}
        for _cat, items in data.items():
            for name, attrs in items.items():
                self.names.append(name)
                keys = list(attrs.keys())
                self.attr[name] = keys
                self.vals[name] = [attrs[k] for k in keys]

    def name(self, idx: int) -> str:
        return self.names[idx]

    def attribute(self, idx: int, num: int) -> str:      # num in {1,2}
        return self.attr[self.names[idx]][num - 1]

    def value(self, idx: int, num: int, level: int) -> str:  # level 0-indexed
        return self.vals[self.names[idx]][num - 1][level]


def decode_attr(code: int) -> tuple[int, int, int, int]:
    i = code % 3; code //= 3
    j = code % 3; code //= 3
    k = code % 3; code //= 3
    l = code % 3
    return i, j, k, l


@dataclass
class Form:
    case_id: int
    perspective: str            # "X" or "Y" (which physical good is endowed)
    keep_good: int              # physical index of the endowed good
    trade_good: int             # physical index of the offered good
    answer_style: str
    display_order: str
    attr_order: str
    paraphrase: str
    prompt: str
    answer_tokens: list         # the two valid answer tokens for this form
    token_to_action: dict       # token -> "keep" | "trade"
    form_id: str = ""
    axes: dict = field(default_factory=dict)


def canonical_final_good(form: Form, token: str) -> int | None:
    """Map a raw answer token to the selected FINAL PHYSICAL GOOD index.
    Returns None if the token is not one of the form's valid tokens (parse fail)."""
    action = form.token_to_action.get(_norm(token))
    if action is None:
        return None
    return form.keep_good if action == "keep" else form.trade_good


def _norm(tok: str) -> str:
    t = str(tok).strip().lower()
    # tolerate punctuation / word forms
    for a, b in (("yes", "yes"), ("no", "no"), ("keep", "keep"), ("trade", "trade")):
        if t.startswith(a):
            return b
    if t in ("a", "(a)", "a."):
        return "a"
    if t in ("b", "(b)", "b."):
        return "b"
    return t


def _describe(g: Goods, idx: int, a1_level: int, a2_level: int, attr_order: str) -> str:
    pairs = [(g.attribute(idx, 1), g.value(idx, 1, a1_level)),
             (g.attribute(idx, 2), g.value(idx, 2, a2_level))]
    if attr_order == "reversed":
        pairs = list(reversed(pairs))
    nm = g.name(idx).lower()
    return (f"In terms of {pairs[0][0]}, the {nm} is {pairs[0][1]}. "
            f"In terms of {pairs[1][0]}, it is {pairs[1][1]}.")


def _answer_clause(style: str, endowed: str, offered: str):
    """Return (question_text, answer_tokens, token_to_action)."""
    if style == "yes_no":
        return ("Would you trade it? Answer with the single word Yes or No.",
                ["Yes", "No"], {"yes": "trade", "no": "keep"})
    if style == "keep_trade":
        return ("Would you keep it or trade it? Answer with the single word keep or trade.",
                ["keep", "trade"], {"keep": "keep", "trade": "trade"})
    if style == "ab_kt":
        return (f"Choose one: (A) keep the {endowed}; (B) trade it for the {offered}. "
                f"Answer with the single letter A or B.",
                ["A", "B"], {"a": "keep", "b": "trade"})
    if style == "ab_tk":
        return (f"Choose one: (A) trade it for the {offered}; (B) keep the {endowed}. "
                f"Answer with the single letter A or B.",
                ["A", "B"], {"a": "trade", "b": "keep"})
    raise ValueError(f"unknown answer_style {style!r}")


_INTRO = {
    "p0": "Suppose that, by chance, you receive a {e} out of many possible goods that you could have received.",
    "p1": "Imagine you happen to end up owning a {e}.",
    "p2": "You have just been given a {e} at random.",
}
_OPTION = {
    "p0": "You can keep the {e} or trade it for a {o}.",
    "p1": "You may either hold on to the {e} or exchange it for a {o}.",
    "p2": "Your choice is to keep the {e} or swap it for a {o}.",
}


def build_forms(goods: Goods, case_id: int, x_num: int, y_num: int, code: int,
                perspective: str) -> list[Form]:
    i, j, k, l = decode_attr(code)
    # X carries attrs (i,j); Y carries attrs (k,l). Perspective sets endowment.
    if perspective == "X":
        endowed_idx, offered_idx = x_num, y_num
        endowed_lv, offered_lv = (i, j), (k, l)
    elif perspective == "Y":
        endowed_idx, offered_idx = y_num, x_num
        endowed_lv, offered_lv = (k, l), (i, j)
    else:
        raise ValueError("perspective must be 'X' or 'Y'")
    endowed, offered = goods.name(endowed_idx).lower(), goods.name(offered_idx).lower()

    forms: list[Form] = []
    for style in ANSWER_STYLES:
        for order in DISPLAY_ORDERS:
            for aorder in ATTR_ORDERS:
                for para in PARAPHRASES:
                    e_desc = _describe(goods, endowed_idx, endowed_lv[0], endowed_lv[1], aorder)
                    o_desc = _describe(goods, offered_idx, offered_lv[0], offered_lv[1], aorder)
                    intro = _INTRO[para].format(e=endowed)
                    option = _OPTION[para].format(e=endowed, o=offered)
                    q, tokens, t2a = _answer_clause(style, endowed, offered)
                    if order == "endowed_first":
                        body = f"{intro}\n{e_desc}\n{option}\n{o_desc}"
                    else:  # offered_first: describe the offered good's block first
                        body = f"{intro}\n{option}\n{o_desc}\nFor the {endowed} you currently hold: {e_desc}"
                    prompt = f"{body}\n{q}"
                    fid = f"{case_id}:{perspective}:{style}:{order}:{aorder}:{para}"
                    forms.append(Form(
                        case_id=case_id, perspective=perspective,
                        keep_good=endowed_idx, trade_good=offered_idx,
                        answer_style=style, display_order=order, attr_order=aorder,
                        paraphrase=para, prompt=prompt, answer_tokens=tokens,
                        token_to_action=t2a, form_id=fid,
                        axes={"answer_style": style, "display_order": order,
                              "attr_order": aorder, "paraphrase": para},
                    ))
    return forms


if __name__ == "__main__":
    g = Goods()
    fs = build_forms(g, 61, 0, 6, 16, "X")
    print(f"{len(fs)} forms for case 61 X-perspective")
    print("keep_good(endowed)=", g.name(fs[0].keep_good), " trade_good=", g.name(fs[0].trade_good))
    print("--- sample yes_no form ---\n", fs[0].prompt)
    print("--- sample ab_tk form ---")
    ab = next(f for f in fs if f.answer_style == "ab_tk")
    print(ab.prompt)
    print("canonical(ab_tk,'A') ->", g.name(canonical_final_good(ab, "A")),
          "(should be the OFFERED good)")
