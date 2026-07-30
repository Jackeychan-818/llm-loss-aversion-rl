# Method-Comparison Protocol — matched base vs GRPO vs SFT vs sign-only

*Frozen 2026-07-30, **before any full-grid SFT checkpoint was evaluated** and
before the new untouched suite (`data/method_comparison/`) was opened. This is a
pre-registration: the eligible models, grids, selector, outcomes, and claim
language below are fixed in advance and applied mechanically. The append-only
execution record at the end documents later runs without rewriting the rules.*

**Freeze-time statement:** at freeze time, the only evaluated causal-baseline
evidence is the exploratory seed-1 6k pilot (`results/causal_baseline_pilot/`).
No full-run SFT checkpoint had been behaviorally evaluated and no selector had
been run on the full grids. Training completion is **not** a behavioral result.

## 1. Purpose and the questions

| Comparison | Question |
|---|---|
| GRPO (magnitude) vs matched base | Does the primary intervention reduce ownership-dependent choice? (already confirmed on other suites) |
| SFT vs GRPO | Is **RL necessary**, or does supervised learning on the same rational-choice targets suffice? |
| sign-only / scale-matched GRPO vs GRPO | Does reward **magnitude** `|δ̃|` add beyond the **sign**? |

These are distinct from the completed frontier-consensus-delta run (a reward-
**source** ablation) and must not be conflated.

## 2. Eligible models, seeds, and checkpoint grids

| model family | seeds | checkpoint grid |
|---|---|---|
| exact matched local base | — | single fixed base (no selection) |
| Qwen-own-delta GRPO (magnitude) | 1, 2 | frozen 2k…30k @ 2k (15) |
| matched SFT (Qwen-own-delta targets) | 1, 2 | frozen 2k…30k @ 2k (15) |
| sign-only / scale-matched GRPO | 1, 2 | frozen 2k…30k @ 2k (15) — **pending training** |

All grids are the confirmatory 15-checkpoint grid. Off-grid diagnostics (e.g.
step 600) are **ineligible** for selection.

## 3. Selection — unchanged `eval/select_checkpoint.py`

Per model family × seed, selection uses the **existing, unchanged** selector:

- **Eligibility:** `consistency ≥ 0.50` **and** finite non-zero SEs for λ̂, η̂.
- **Primary:** minimise `d = sqrt(λ̂² + η̂²)`.
- **Tie-break 1:** within `D_TOL` on `d`, prefer higher consistency.
- **Tie-break 2:** within `CONS_TOL` on consistency, prefer the earliest checkpoint.
- **If nothing is eligible:** report it; **do not** relax thresholds post hoc.

Selection is run per family with a family-specific `--pattern`, on **separate
evaluation directories**, and does **not** touch the frozen confirmatory
selections (seed 1 → 2,000; seed 2 → 6,000). No new or modified selector is
introduced. The selector **hard-fails on an incomplete 15-point grid**, so all 15
checkpoints per seed must be evaluated first.

`test_goods.json` is **VALIDATION ONLY** for checkpoint selection and reported
λ̂; it is not an untouched final test.

**Ordering rule (critical):** *all* per-seed checkpoints for *all* families are
selected on `test_goods` validation **before** the new untouched suite is opened.
The untouched suite is opened **once**, only on the already-selected checkpoints
(+ the matched base), for the comparison in §5.

## 4. Provenance requirement

Every selector invocation for these baselines MUST pass `--provenance_note`
recording that the previous final suites are already opened:

> "OOD-50 and frozen-unused final suites already opened (before these baselines);
> selection is validation-only on test_goods; SFT-vs-GRPO / sign-vs-magnitude
> comparisons are post-hoc until the new untouched suite is opened."

The new untouched suite (`data/method_comparison/`) is the only set that can
carry a **confirmatory** cross-method claim, and only under this protocol.

## 5. Outcomes

- **Primary:** structural **λ̂**, with **η̂ reported jointly**, on the new
  untouched suite for each selected checkpoint and the matched base.
- **Joint distance:** `d = sqrt(λ̂² + η̂²)`.
- **Direct choice outcomes:** consistency, keep-both, trade-both, target
  agreement (fraction matching the frozen rational target), and frozen
  ownership-free preference preservation where available.
- **Uncertainty:** pair/good-aware (pair-clustered bootstrap; see
  `eval/robust_inference.py`), reported **per seed** — never pooled to hide seed
  variation.
- **Capability:** GSM8K / IFEval non-inferiority reported separately (not used
  for selection).

## 6. Malformed output and missing-run handling

- Non-`{Yes,No}` (or non-`{keep,trade}`) responses are **parse failures**,
  counted as such and reported; they are **not** silently dropped and **not**
  counted as correct.
- A missing checkpoint or missing eval makes that family/seed grid **incomplete**;
  the selector must reject it rather than select from a partial grid.
- Every reported cell must trace to a machine-readable input (see Task 5
  aggregation); no hand-entered numbers.

## 7. Capability non-inferiority margins

Frozen before evaluation, consistent with the existing gates: GSM8K paired-CI
lower bound ≥ −3 pp per selected seed vs the same base responses; IFEval margin
frozen when IFEval is run. Capability results **never** drive checkpoint
selection.

## 8. Confirmatory vs exploratory claim language

- A cross-method difference is **confirmatory** only if it (a) uses the new
  untouched suite opened once under this protocol, (b) holds per seed under
  pair-aware uncertainty, and (c) was not selected or tuned on this suite.
- Any comparison drawn from `test_goods`, OOD-50, frozen-unused, or framing is
  **post-hoc / validation / exploratory** and must be labelled so.
- A full `test_goods` grid does **not** make SFT-vs-GRPO confirmatory. This
  protocol explicitly rejects that claim.

## 9. Prohibited selection inputs

Checkpoints may **not** be selected using training loss, GSM8K, IFEval, framing,
OOD-50, the frozen-unused suite, or the new untouched suite. Selection is
`test_goods` validation only.

## 10. Already-opened suites (recorded)

OOD-50, the frozen-unused prospective-configuration suite, and the framing suite
are **already opened** and are closed to tuning, selection, and method revision
for this comparison.

## Append-only execution record

*(none yet — no full-grid SFT checkpoint has been evaluated; the untouched suite
is unopened. Add entries here when the GPU phase runs, without editing the rules
above.)*
