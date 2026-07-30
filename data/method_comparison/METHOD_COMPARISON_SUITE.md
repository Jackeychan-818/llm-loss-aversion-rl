# Method-Comparison Suite — frozen, untouched (Task 1)

*Frozen 2026-07-30, before any full-grid SFT checkpoint was evaluated. Built by
the CPU-only work package (`NSCC_CPU_WORK_PROMPT.md`, Task 1). This document is
the design note; the machine-readable freeze is the manifest.*

## Purpose

A single evaluation suite reserved for the **final** comparison among:

- the exact matched local base,
- magnitude-weighted Qwen-own-delta GRPO,
- matched SFT (Qwen-own-delta targets), and
- sign-only / scale-matched GRPO controls.

It has **never** been used for training, reward construction, checkpoint
selection, or prior method development, so it can carry a confirmatory
cross-method claim that `test_goods` (validation) and the already-opened OOD-50
/ frozen-unused / framing suites cannot.

## Source population and sampling frame

Same 100 goods and 4,945 unordered goods pairs as the rest of the benchmark.
Each pair has **81 joint attribute codes** (3⁴: two attributes per good, three
levels each). Codes already consumed per pair:

| source | codes/pair | role |
|---|---:|---|
| `test_goods.json` | 2 | validation |
| `remaining_goods.json` | 10 | training |
| `frozen_unused_test_goods.json` | 10 | already-OPENED prospective suite |
| **total used** | **22** | |
| **untouched remainder** | **59** | candidate frame for this suite |

This suite selects **K = 4** untouched codes per pair → **19,780 cases**,
**39,560 X/Y prompts per model**, leaving 55 untouched codes/pair for any future
need.

## Selection (deterministic, response-independent)

For each pair, exclude the 22 used codes; sort the remaining 59 by
`SHA-256("{seed}:{x}:{y}:{code}")` with **FROZEN_SEED = 20260730** (distinct from
the frozen-unused seed 20260723); take the first 4; sort. The shuffle is
independent of Python's `random`, so the suite is byte-reproducible from tracked
inputs alone (`--check` enforces this). No model response influences selection.

## Overlap prevention (verified by tests)

- Zero `(pair, code)` intersection with `test_goods`, `remaining_goods`, and
  `frozen_unused_test_goods` — asserted in the generator and independently in
  `test_method_comparison_suite.py`.
- OOD-50 (`ood_new_goods_50.json`) uses a **different goods population** (unseen
  goods), so it shares no `(pair, code)` tuples with the 100 main goods by
  construction. The manifest records this.

## Case IDs and X/Y pairing

Stable integer `case_id` from offset **2,000,000** (contiguous, cannot collide
with the 1..59,400 trial/test/remaining namespace or the frozen-unused suite),
ordered over `(X, Y, code)`. Each `case_id` yields exactly one X-endowed and one
Y-endowed prompt (one keep, one trade), so prompts/model = 2 × cases.

## Strata (predefined; reporting only)

`method_comparison_strata.json` records a **predicted δ̃ = U_X − U_Y** per case,
computed from the FROZEN base utility table
(`baseline/Qwen-7B/Model_1/Qwen-7B_utility_of_each_goods_Model_A.csv`). This is a
deterministic function of already-fitted parameters — **not a model response and
not used in selection**. Difficulty bins (`|δ̃|≤0.5`, `0.5–1.0`, `1.0–2.0`, `>2.0`)
support stratified reporting. Current counts: 10,513 / 3,939 / 4,333 / 995.

## Semantic-counterbalancing component

`semantic_counterbalancing.json` freezes a stratified subset (40 cases per
predicted-δ̃ bin = **160 cases**) and the surface-form variant axes, expanded at
GPU-eval time to **48 forms/case**:

1. `response_mode`: Yes/No vs keep/trade
2. `item_labels`: X/Y vs A/B
3. `display_order`: normal vs reversed
4. `attr_order`: normal vs reversed
5. `paraphrase`: `baseline_v1`, `concise_v1`, `explicit_v1`

Primary outcome: whether the same final good is chosen across equivalent forms.
Permitted models: matched base, seed 1 @ 2,000, seed 2 @ 6,000.

## Freeze policy

- Training / reward construction / checkpoint selection / method tuning:
  **PROHIBITED**.
- Open **once**, only after every per-seed checkpoint is selected on
  `test_goods` validation, for the pre-registered comparison in
  `METHOD_COMPARISON_PROTOCOL.md`.
- Scope: same goods and pairs, untouched configurations — **not** unseen-goods
  OOD.

## Files

| file | contents |
|---|---|
| `build_method_comparison_suite.py` | deterministic generator (`--check`) |
| `method_comparison_suite.json` | 4,945 rows `[X, Y, [4 codes]]` |
| `method_comparison_suite.manifest.json` | SHA-256, seed, input hashes, overlap + balance validation, case-ID scheme, freeze policy |
| `method_comparison_strata.json` | predicted-δ̃ per case (enrichment) |
| `build_semantic_counterbalancing.py` | deterministic subset + variant axes (`--check`) |
| `semantic_counterbalancing.json` / `.manifest.json` | frozen subset + axes |
| `test_method_comparison_suite.py` | 31 integrity checks (overlap, pairing, IDs, strata, hash drift) |

## Reproduce / verify

```bash
python3 data/method_comparison/build_method_comparison_suite.py --check
python3 data/method_comparison/build_semantic_counterbalancing.py --check
python3 data/method_comparison/test_method_comparison_suite.py
```

**No model is evaluated on this suite during the CPU phase.**
