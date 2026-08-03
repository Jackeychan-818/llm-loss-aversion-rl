# SFT vs sign-only pilot diagnostic — structural + ownership (CPU)

*EXPLORATORY / POST-HOC. test_goods validation. Reward-hacking vs task-simplicity diagnostic, NOT a method winner. Surface-form stress tests and new goods (OOD-50) are GPU-pending.*

| model | λ (SE) | η (SE) | d | keep-both | trade-both | consistency | target-agree | yes-rate | parse |
|---|---|---|---|---|---|---|---|---|---|
| matched_base | 7.637 (0.627) | 1.007 (0.120) | 7.703 | 0.993 | 0.000 | 0.007 | 0.503 | 0.004 | 0.000 |
| sft_seed1_step6000 | 0.047 (0.014) | 0.023 (0.033) | 0.052 | 0.158 | 0.117 | 0.725 | 0.740 | 0.480 | 0.000 |
| sign_only_seed1_step6000 | 0.388 (0.021) | 0.039 (0.038) | 0.390 | 0.292 | 0.057 | 0.651 | 0.721 | 0.386 | 0.000 |

Base keep-both is the loss-averse signature (endowed good kept from both sides). Lower keep-both + lower λ + higher consistency = reduced ownership dependence.
