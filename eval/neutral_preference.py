#!/usr/bin/env python3
"""Neutral-preference forms + canonicalization (safeguard 9).

A model can be surface-form-invariant yet still OVERWRITE the base model's
underlying preferences. This test elicits preference WITHOUT any ownership /
endowment framing and checks whether the tuned model preserves the base model's
neutral preference (and the frozen delta-preferred good).

Uses the SAME frozen 96-case subset as the surface-form test, so no case is
selected on any pending result. For each case, presents the two goods neutrally
as "Good A" / "Good B" with order counterbalancing (2) x paraphrase (2) = 4
forms; the answer (A/B) is canonicalized to the physical good.

No torch; unit-testable without a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from surface_form_transforms import Goods, decode_attr, _norm  # noqa: E402

ORDERS = ["AX_BY", "AY_BX"]        # which physical good is labelled A vs B
PARAPHRASES = ["p0", "p1"]

_TEMPLATE = {
    "p0": ("Consider two goods you could receive.\n"
           "Good A is a {na}. In terms of {a1a}, it is {a1va}. In terms of {a2a}, it is {a2va}.\n"
           "Good B is a {nb}. In terms of {a1b}, it is {a1vb}. In terms of {a2b}, it is {a2vb}.\n"
           "Which would you prefer to have? Answer with the single letter A or B."),
    "p1": ("Two goods are available.\n"
           "Option A: a {na} ({a1a}: {a1va}; {a2a}: {a2va}).\n"
           "Option B: a {nb} ({a1b}: {a1vb}; {a2b}: {a2vb}).\n"
           "Which do you prefer? Reply with a single letter, A or B."),
}


@dataclass
class NeutralForm:
    case_id: int
    order: str
    paraphrase: str
    good_for_A: int
    good_for_B: int
    prompt: str
    answer_tokens: list
    token_to_good: dict          # normalized token -> physical good index
    form_id: str = ""
    axes: dict = field(default_factory=dict)


def canonical_choice(form: NeutralForm, token: str) -> int | None:
    return form.token_to_good.get(_norm(token))


def _good_block(g: Goods, idx: int, l1: int, l2: int) -> dict:
    return {"n": g.name(idx).lower(),
            "a1": g.attribute(idx, 1), "a1v": g.value(idx, 1, l1),
            "a2": g.attribute(idx, 2), "a2v": g.value(idx, 2, l2)}


def build_neutral_forms(g: Goods, case_id: int, x_num: int, y_num: int, code: int):
    i, j, k, l = decode_attr(code)          # X carries (i,j); Y carries (k,l)
    forms = []
    for order in ORDERS:
        if order == "AX_BY":
            a_idx, a_l = x_num, (i, j)
            b_idx, b_l = y_num, (k, l)
        else:
            a_idx, a_l = y_num, (k, l)
            b_idx, b_l = x_num, (i, j)
        A = _good_block(g, a_idx, a_l[0], a_l[1])
        B = _good_block(g, b_idx, b_l[0], b_l[1])
        for para in PARAPHRASES:
            prompt = _TEMPLATE[para].format(
                na=A["n"], a1a=A["a1"], a1va=A["a1v"], a2a=A["a2"], a2va=A["a2v"],
                nb=B["n"], a1b=B["a1"], a1vb=B["a1v"], a2b=B["a2"], a2vb=B["a2v"])
            forms.append(NeutralForm(
                case_id=case_id, order=order, paraphrase=para,
                good_for_A=a_idx, good_for_B=b_idx, prompt=prompt,
                answer_tokens=["A", "B"], token_to_good={"a": a_idx, "b": b_idx},
                form_id=f"{case_id}:neutral:{order}:{para}",
                axes={"order": order, "paraphrase": para}))
    return forms


if __name__ == "__main__":
    g = Goods()
    fs = build_neutral_forms(g, 61, 0, 6, 16)
    print(f"{len(fs)} neutral forms/case")
    print(fs[0].prompt)
    print("canonical(A)=", g.name(canonical_choice(fs[0], "A")))
