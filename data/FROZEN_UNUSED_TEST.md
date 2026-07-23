# Frozen Prospective Unused-Configuration Test

**Status:** `FROZEN_PROSPECTIVE_UNEVALUATED`

**Freeze date:** July 23, 2026

**Dataset:** `data/frozen_unused_test_goods.json`

**Manifest:** `data/frozen_unused_test_goods.manifest.json`

## Purpose and scope

This is a prospective supplemental test of generalization to previously unused
attribute configurations. It contains the same 100 goods and the same 4,945
unordered goods pairs used by `test_goods.json` and `remaining_goods.json`.
Therefore, it is a **within-benchmark configuration test**, not an unseen-goods
OOD test.

The file was generated and committed only after training, checkpoint selection,
and the confirmatory OOD-50/GSM8K verdict were complete. Its outcomes were not
used to choose the seed, balance gates, checkpoints, or evaluation rules.

## Frozen construction

Each joint attribute code is an integer from 0 through 80 encoding four
three-level attribute positions. For every goods pair, construction:

1. removes the 2 codes already in `test_goods.json`;
2. removes the 10 codes already in `remaining_goods.json`;
3. hash-shuffles the 69 unused codes using frozen seed `20260723`;
4. takes the first 10 codes.

The stable shuffle key is:

```text
SHA256("{seed}:{x}:{y}:{code}")
```

This produces:

| Quantity | Frozen value |
|---|---:|
| Goods pairs | 4,945 |
| New configurations per pair | 10 |
| Cases | 49,450 |
| X/Y prompts per model | 98,900 |
| Remaining unused configurations after this test | 291,755 |

## Validation

The generator hard-fails unless pair coverage, uniqueness, overlap, and balance
checks pass. For the committed file:

- overlap with the 2 existing validation codes per pair: **zero**;
- overlap with the 10 existing training codes per pair: **zero**;
- selected codes are unique within every pair: **passed**;
- all 4,945 pairs have exactly 10 selected codes: **passed**;
- joint-code counts across codes 0–80: **547–665**, versus 610.49 expected;
- joint-code uniformity: **chi-square(80) = 73.10**;
- maximum joint-code relative deviation: **10.400%**, below the frozen 12% gate;
- marginal uniformity chi-square(2), by attribute position:
  **1.22, 1.26, 2.15, 3.53**;
- maximum marginal share deviation across the four attribute positions and
  three levels: **0.387 percentage points**, below the frozen 0.5-point gate.

The manifest records the complete 81-code frequency vector, all 12 marginal
counts, the source hashes, construction rule, output hash, and freeze policy.

## Freeze policy

- This file must **never** be used for training, reward construction, checkpoint
  selection, early stopping, or hyperparameter changes.
- Primary evaluation is limited to the exact matched local base and the two
  already-selected confirmatory checkpoints: seed 1 at step 2,000 and seed 2 at
  step 6,000.
- Any other already-trained model, including exploratory seed 42 at step 8,000,
  must be labelled exploratory and cannot change the frozen selections or the
  existing confirmatory verdict.
- Report lambda, eta, the combined objective, direct paired-choice outcomes,
  and utility/preference preservation for each seed separately.
- Do not describe this result as unseen-goods OOD evidence. OOD-50 remains the
  new-goods generalization test.

## Reproduce before evaluation

From the repository root:

```bash
python3 data/build_frozen_unused_test.py --check
```

To regenerate both tracked artifacts from the two original inputs:

```bash
python3 data/build_frozen_unused_test.py
python3 data/build_frozen_unused_test.py --check
```

The committed dataset SHA-256 is:

```text
793c472157fb46fbc53cb24df644b04b84ade7aa8a1a0b6ea93567b4838a55e5
```
