# Amendment-3 probe — lr 1e-4 versus 1e-5 (seed 1, bracketing only)

*Off-grid bracketing probe. **One seed. It may bracket the optimum; under
Amendment 3 it may NOT promote a setting or change the frozen selection.***
Protocol commit `32911dd`; code `91d21c9`.

## Result: 1e-4 is dramatically better on every axis

| cell | validation CE | λ | η | consistency | keep-both | trade-both |
|---|---:|---:|---:|---:|---:|---:|
| eb16 @ 1e-5 | 0.4448 | +0.096 | −0.367 | 0.7585 | 0.063 | 0.179 |
| **eb16 @ 1e-4** | **0.2108** | +0.037 | −0.228 | **0.9145** | 0.029 | 0.057 |
| eb32 @ 1e-5 | 0.4958 | +0.054 | −0.264 | 0.7160 | 0.078 | 0.206 |
| **eb32 @ 1e-4** | **0.2206** | +0.082 | −0.318 | **0.8930** | 0.033 | 0.074 |

Validation cross-entropy **more than halves** at both batches. It is not a
CE-only effect: λ moves toward zero at eb16, consistency rises by ~0.16, and
both degenerate modes (keep-both, trade-both) fall by roughly two thirds. The
behavioural diagnostics and the selection criterion point the same way.

Training was clean — no divergence, **zero non-finite records**, no loss spikes.

## The optimum is still not bracketed

The probe was meant to trap a turnover between 1e-5 and 1e-4. It did not: 1e-4
wins outright, so the optimum lies at or **above** 1e-4 and remains unbracketed
after searching two orders of magnitude.

## The batch difference shrinks as the LR approaches a sane value

| LR | eb16 | eb32 | gap |
|---|---:|---:|---:|
| 1e-6 | 0.5947 | 0.6622 | 0.0675 |
| 1e-5 | 0.4413 | 0.4986 | 0.0573 |
| **1e-4** | **0.2108** | **0.2206** | **0.0098** |

This supports the caution recorded in the Stage-1 report: the apparent
"smaller batch is better" ordering was largely an artefact of **all** batches
being under-trained at a shared, far-too-low learning rate. At 1e-4 the two are
nearly tied — the gap is down to ~2× eb32's noise floor, from ~11× at 1e-5.
Choosing a batch on the Stage-1 ranking would have been choosing on an artefact.

## The finding that matters beyond this experiment

The matched-SFT baseline uses **lr 1e-6**, inherited from the GRPO config for
matching rather than chosen for SFT. On this evidence that value is roughly two
orders of magnitude too low for supervised fine-tuning on this task.

For scale: the frozen matched-SFT selection (seed 1, step 4,000) reaches
consistency **0.7298**, and its 30,000-prompt endpoint reaches **0.867**. This
probe reaches **0.9145 in 6,016 prompts** — five times less data than the
endpoint of the full run.

That does not invalidate the matched-SFT baseline, which is *deliberately*
matched on hyper-parameters. But it sharpens what that baseline is: a
**matched-hyper-parameter** SFT baseline, not a best-effort one. Any claim of
the form "RL is necessary because SFT does worse" has to say which of the two it
means, because a best-effort SFT baseline would plainly be configured
differently. That belongs in `METHOD_COMPARISON_PROTOCOL.md`, which governs
cross-method claims — not here.

## Limits

One seed per cell; no promotion is possible under Amendment 3. eb64 was not
probed. `test_goods` is validation data that informed reward construction and
prior checkpoint selection. Everything is conditional on the 6,016-prompt cosine
schedule and clipping 0.1 — the coupling between LR and horizon means 1e-4 must
**not** be carried to a 30,016-prompt run without separate testing; five times
the updates at the same peak LR is roughly five times the path length.

No frozen or untouched suite was opened.
