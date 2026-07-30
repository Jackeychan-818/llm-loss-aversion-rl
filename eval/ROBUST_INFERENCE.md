# Robust structural inference (Task 4)

*Added 2026-07-30. Separate, documented robustness layer. Does **not** modify or
replace the frozen headline estimator (`eval/core_exp_refactored.py`, Model A
full fixed effects), which stays the reference for point estimates.*

## What this adds (INFER-001, PAIR-001)

1. **Strict ID-based X/Y join** (`load_paired_predictions` / `join_paired_records`)
   with hard assertions: missing IDs, duplicate IDs, asymmetric pairing, and
   goods/attribute mismatch all raise. This is the PAIR-001 fix, usable by any
   estimator.
2. **Pair-clustered bootstrap** for `(lambda, eta)` — the cluster is one goods
   pair keyed by `(X_num, Y_num)`; each replicate resamples WHOLE pairs with
   replacement, so all of a pair's configurations and both perspectives move
   together and can never be split across clusters (proven by
   `test_robust_inference.py`). This respects the paired/repeated structure the
   frozen iid `curve_fit` covariance ignores.
3. **Leave-one-good-out**, **Jacobian conditioning**, and **multi-start**
   diagnostics.
4. **Estimator-recovery** (`estimator_recovery.py`): known `(lambda*, eta*)` →
   bias, clustered-CI coverage, failure rate, weak-identification.

## Design and its limitation (read before citing)

Re-fitting the full ~109-parameter FE model inside every bootstrap replicate is
expensive and couples to the frozen engine's file I/O. So the robustness layer
**conditions on plug-in per-case utilities** `V_X = exp(U_X)`, `V_Y = exp(U_Y)`
and re-estimates only `(lambda, eta)` via the identical link
(`z = (1+lambda)·V_X − V_Y + eta` for the X-perspective; symmetric for Y;
`P(No) = sigmoid(z/T)`, `T = 1`).

**Consequence (must be stated honestly):** when `V` is taken from the base
utility table, the layer's *point* `(lambda, eta)` is on the plug-in scale and is
**not** the frozen headline `lambda = 7.637`. On the matched base it returns a
much larger conditional `lambda` because the plug-in `V` scale differs from how
the joint estimator co-fits `lambda` with the fixed effects. Its validated
contribution is the **clustered-uncertainty methodology**, not a replacement
point estimate.

The recovery study proves the estimator is **unbiased with ~nominal clustered
coverage when the plug-in `V` is the data-generating `V`**. Therefore:

- Use it now to (a) validate that pair-clustered inference is well-behaved
  (recovery grid), and (b) get clustered CIs around whatever `(lambda, eta)` the
  supplied `V` implies.
- **Closing INFER-001 on the headline numbers** requires feeding each model's own
  frozen-estimator utilities into the bootstrap (or refitting the FE jointly per
  replicate). That wiring is the remaining step and is called out here rather than
  overclaimed.

## Files

| file | role |
|---|---|
| `eval/robust_inference.py` | ID-join + assertions, conditional `(lambda,eta)` NLS, pair-clustered bootstrap, leave-one-good-out, Jacobian/multistart |
| `eval/estimator_recovery.py` | recovery grid → bias / coverage / failure / weak-ID |
| `eval/run_robust_bootstrap.py` | driver over EXISTING loss_aversion_X/Y + utility CSV (no inference) |
| `eval/test_robust_inference.py` | 16 tests (join hard-fails, recovery, bootstrap, LOGO, conditioning) |
| `train/submit_cpu_recovery.pbs` | CPU-only recovery grid (no GPU) |
| `train/submit_cpu_bootstrap.pbs` | CPU-only bootstrap over existing predictions (no GPU) |

## Reproduce

```bash
python3 eval/test_robust_inference.py            # unit tests
python3 eval/estimator_recovery.py --quick       # fast recovery check
# full CPU jobs (NSCC):
qsub train/submit_cpu_recovery.pbs
qsub -v XJSON=...,YJSON=...,UTIL=...,TAG=base train/submit_cpu_bootstrap.pbs
```

Small checks run interactively; the high-replicate grid/bootstrap are the PBS
jobs. No model inference is performed and no checkpoint or frozen suite is
touched.
