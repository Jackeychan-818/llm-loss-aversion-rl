# Project Timeline — lambda-zero

**Goal:** Fine-tune Qwen2.5-7B-Instruct with GRPO to reduce loss aversion (λ → 0) below frontier models, and publish at a top ML venue.

**Strategy:** NeurIPS 2026 workshop paper (early results) → ICML 2027 main track (full paper)

---

## Research Phases

| Period | Phase | Goal | Status |
|---|---|---|---|
| Apr 14, 2026 | Phase 1 — Small-model baseline | λ̂_before = **11.75** (SE = 1.22) | ✅ Done |
| Apr 2026 | Phase 2 — Reward function design | Finalize reward signal for GRPO | 🔴 Current |
| May 2026 | Phase 3 — GRPO training | LoRA fine-tuning on Qwen2.5-7B | ⏳ Blocked on 2 |
| Jun 2026 | Phase 4 — Post-training evaluation | λ̂_after — did GRPO reduce λ? | ⏳ |
| Jun–Jul 2026 | Phase 5 — Cross-model comparison | Qwen-7B vs frontier models | ⏳ |
| Jul 2026 | Phase 6 — Ablations | Reward type, model size, generalization | ⏳ |

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

**What to submit:** Complete paper — full results, ablations, cross-model comparison, paper figures from `notebooks/compare_lambdas.ipynb`.

---

## Key Dates to Watch

- **NeurIPS 2026 main track deadline:** May 6, 2026 *(already very tight — not recommended)*
- **NeurIPS 2026 workshop deadlines:** Not yet announced. Check [neurips.cc](https://neurips.cc/Conferences/2026) regularly from May 2026.
- **ICML 2027 deadlines:** Not yet announced. Check [icml.cc](https://icml.cc) from Oct 2026.

---

## Notes

- Workshop papers at NeurIPS are non-archival — submitting there does **not** prevent ICML 2027 submission.
- Confirm the specific workshop's non-archival policy before submitting.
- Human benchmark for comparison: λ ≈ 2–2.5 (Kahneman & Tversky, 1979).
- Current Qwen-7B baseline: λ̂ = 11.75 — roughly 5× human level, expected to be well above frontier models.

*Last updated: April 14, 2026*
