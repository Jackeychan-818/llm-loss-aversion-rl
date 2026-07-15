# Project Timeline — lambda-zero

**Goal:** Fine-tune Qwen2.5-7B-Instruct with GRPO to reduce
ownership-dependent choice behaviour while preserving Qwen's own estimated
preferences, and publish the result at a top AI/ML venue.

**Strategy:** NeurIPS 2026 workshop paper (early results) → ICML 2027 main track (full paper)

---

## Research Phases

| Period | Phase | Goal | Status |
|---|---|---|---|
| Apr 14, 2026 | Phase 1 — Small-model baseline | Historical λ̂_before = **11.75**; matched local rerun required | 🟡 |
| Apr 27, 2026 | Phase 2 — Consensus reward design | Utility-weighted frontier consensus δ̃ v3 | ✅ Reference/ablation |
| Apr–Jun 2026 | Phase 3 — GRPO training | LoRA fine-tuning on Qwen2.5-7B; NSCC resume/eval path hardened | ✅ Done |
| Jul 3–15, 2026 | Phase 4 — Post-training evaluation | λ̂_after = **0.177** (SE = 0.005), 98.5% reduction; held-out attribute sensitivity confirmed | ✅ Done |
| Jul 7, 2026 | Treatment robustness | Debias λ̂ = 0.205; forced λ̂ = 0.173 | ✅ Done |
| Jul 2026 | Primary-model decision | Qwen-own-delta step 8k primary; consensus delta reward-source ablation | ✅ Decided |
| Jul 2026 | Qwen-own evaluation | ID λ̂ = 0.111 is validation; reported OOD λ̂ = 0.226 pending artifact archival | 🟡 |
| Jul 2026 | Full-paper validation | Matched base, delta validation, fixed selection rule, ≥3 seeds, SFT, robust inference, capability checks | ⏳ Next |
| Jul 2026 | Cross-model comparison | Retain only under harmonized protocols | ⏳ Optional to claim scope |
| Aug 2026 onward | Phase 7 — Paper writeup | Workshop paper, then full paper | ⏳ |

---

## Commit History Integrated

| Date | Commit | Project meaning |
|---|---|---|
| Apr 27, 2026 | `40f4b0e` | Added GRPO training code, baseline results, consensus δ̃ files, and NSCC setup. |
| Apr 27, 2026 | `c7088c7` | Updated agent context after the reward decision and corrected GRPO hyperparameters (`G=16`, temp `1.5`). |
| Jun 18, 2026 | `d26ac54` | Fixed GRPO resume behavior and vLLM training path; added long-queue PBS script. |
| Jun 18, 2026 | `e938efa` | Restored NSCC-validated PyTorch module and gradient accumulation after Mac merge drift. |
| Jun 18, 2026 | `dce0550` | Fixed `glong` routing via the normal PBS queue. |
| Jun 23, 2026 | `b9c2041` | Disabled broken vLLM dependency on NSCC and fell back to plain HF generation. |
| Jun 25, 2026 | `f4a2c9a` | Added vLLM-free Qwen evaluation path (`eval/run_qwen_local.py`, `eval/estimate_qwen_grpo.py`). |
| Jul 3, 2026 | `b511e53` | Added held-out Phase 4 baseline results: λ̂_after = 0.177, below human benchmark. |
| Jul 7, 2026 | `c643617` | Added debias/forced results, Qwen-delta ablation config/data, and training docs. |
| Jul 9, 2026 | `f2a8710` | Added monitoring and plotting tools; changed PBS logs to append mode. |

---

## Publication Track

### 🎯 Target 1 — NeurIPS 2026 Workshop (Early Results)

| Milestone | Estimated Date |
|---|---|
| Identify relevant workshop (RL for LLMs / behavioral AI) | May 2026 |
| Workshop call for papers opens | ~Jun 2026 |
| Workshop paper submission deadline | ~Aug 2026 *(TBC — not yet announced)* |
| Workshop acceptance notification | ~Oct 2026 |
| NeurIPS 2026 Conference | **Dec 6–12, 2026** |

**What to submit:** Preliminary results — baseline λ̂_before, GRPO setup, early λ̂_after if training is complete. 4–6 pages. Non-archival — does not block future submission.

---

### 🎯 Target 2 — ICML 2027 Main Track (Full Paper)

| Milestone | Estimated Date |
|---|---|
| Incorporate workshop feedback | Dec 2026 – Jan 2027 |
| ICML 2027 abstract deadline | ~Jan 2027 *(TBC — not yet announced)* |
| ICML 2027 full paper deadline | ~Feb 2027 *(TBC)* |
| ICML 2027 acceptance notification | ~Apr 2027 |
| ICML 2027 Conference | ~Jul 2027 |

**What to submit:** Complete Qwen-own-delta paper with a matched local base,
three or more seeds, matched SFT/sign-only baselines, consensus reward-source
ablation, frozen final evaluation, robust inference, capability retention, and
reproducible raw artifacts. See `PAPER_READINESS.md`.

---

## Key Dates to Watch

- **NeurIPS 2026 main track deadline:** May 6, 2026 *(already very tight — not recommended)*
- **NeurIPS 2026 workshop deadlines:** Not yet announced. Check [neurips.cc](https://neurips.cc/Conferences/2026) regularly from May 2026.
- **ICML 2027 deadlines:** Not yet announced. Check [icml.cc](https://icml.cc) from Oct 2026.

---

## Notes

- Workshop papers at NeurIPS are non-archival — submitting there does **not** prevent ICML 2027 submission.
- Confirm the specific workshop's non-archival policy before submitting.
- Qwen-own-delta step 8k is the primary paper model; consensus delta is the
  reward-source ablation.
- Do not use the historical human or frontier comparisons as headline claims
  until tasks, estimands, model endpoints, samples, scorers, and estimators are
  comparable.
- Train and test contain no repeated prompts or exact configurations. The held-out attribute profiles explain 48.8% of within-pair/perspective variation and change at least one answer for 42.18% of goods pairs.
- The attribute split supports within-benchmark configuration generalization.
  For Qwen-own delta, `test_goods` is validation because it informed reward
  construction and checkpoint selection; a separately frozen evaluation is
  needed for final claims.

*Last updated: July 15, 2026*
