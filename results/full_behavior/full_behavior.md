# Full-behavior aggregation (Task 5)

*Rendered by `eval/aggregate_full_behavior.py` from the tracked `full_behavior_snapshot.json` (NSCC discovery is behind `--refresh`). `test_goods` = validation. A small λ is not called success when caveats fire.*

| model | seed | step | λ (SE) | η (SE) | d | cons | keep | trade | tgt | W | clean | caveats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen-7B-Base-Local | base | 0 | 7.637 (0.627) | 1.007 (0.120) | 7.703 | 0.007 | 0.993 | 0.000 | 0.503 | 0.744 | NO | |eta|=1.01>1.0: status-quo bias remains; consistency=0.01<0.5; choice collapse (a keep/trade side <2%) |
| Qwen-7B-GRPO-qd-ckpt5000 | exploratory | 5000 | 0.226 (0.020) | 0.566 (0.036) | 0.609 | 0.593 | 0.360 | 0.047 | 0.708 | — | yes | — |
| Qwen-7B-GRPO-qd-ckpt8000 | exploratory | 8000 | 0.111 (0.014) | 0.504 (0.029) | 0.516 | 0.685 | 0.261 | 0.054 | 0.751 | — | yes | — |
| Qwen-7B-GRPO-qd-ckpt10000 | exploratory | 10000 | 0.131 (0.013) | 0.650 (0.029) | 0.663 | 0.701 | 0.266 | 0.033 | 0.768 | — | yes | — |
| Qwen-7B-GRPO-qd-ckpt12000 | exploratory | 12000 | 0.430 (0.019) | 0.717 (0.031) | 0.836 | 0.642 | 0.346 | 0.012 | 0.766 | — | NO | choice collapse (a keep/trade side <2%) |
| Qwen-7B-GRPO-qd-ckpt15000 | exploratory | 15000 | 0.673 (0.023) | 0.576 (0.032) | 0.886 | 0.639 | 0.354 | 0.006 | 0.779 | — | NO | choice collapse (a keep/trade side <2%) |
| Qwen-7B-GRPO-qd-ckpt20000 | exploratory | 20000 | -0.083 (0.008) | 1.263 (0.027) | 1.266 | 0.758 | 0.219 | 0.023 | 0.835 | — | NO | |eta|=1.26>1.0: status-quo bias remains |
| Qwen-7B-GRPO-qd-ckpt30000 | exploratory | 30000 | 0.593 (0.017) | 0.815 (0.026) | 1.008 | 0.723 | 0.272 | 0.005 | 0.841 | — | NO | choice collapse (a keep/trade side <2%) |
| Qwen-7B | other | -1 | 11.752 (1.222) | 1.518 (0.085) | 11.850 | 0.009 | 0.991 | 0.000 | 0.505 | — | NO | |eta|=1.52>1.0: status-quo bias remains; consistency=0.01<0.5; choice collapse (a keep/trade side <2%) |
| Qwen-7B-GRPO | other | -1 | 0.177 (0.005) | -0.048 (0.024) | 0.183 | 0.907 | 0.080 | 0.013 | 0.725 | — | NO | choice collapse (a keep/trade side <2%) |
| Qwen-7B-GRPO-qd-seed1-ckpt2000 | seed1 | 2000 | 0.032 (0.015) | 0.091 (0.031) | 0.096 | 0.658 | 0.207 | 0.135 | 0.711 | 0.883 | yes | — |
| Qwen-7B-GRPO-qd-seed1-ckpt4000 | seed1 | 4000 | 0.050 (0.016) | 1.143 (0.034) | 1.144 | 0.569 | 0.403 | 0.028 | 0.707 | 0.885 | NO | |eta|=1.14>1.0: status-quo bias remains |
| Qwen-7B-GRPO-qd-seed1-ckpt6000 | seed1 | 6000 | 0.083 (0.015) | 0.983 (0.032) | 0.987 | 0.638 | 0.333 | 0.029 | 0.738 | 0.903 | yes | — |
| Qwen-7B-GRPO-qd-seed1-ckpt8000 | seed1 | 8000 | 0.163 (0.017) | 1.340 (0.033) | 1.350 | 0.554 | 0.434 | 0.012 | 0.728 | 0.899 | NO | |eta|=1.34>1.0: status-quo bias remains; choice collapse (a keep/trade side <2%) |
| Qwen-7B-GRPO-qd-seed1-ckpt10000 | seed1 | 10000 | -0.105 (0.009) | 0.917 (0.026) | 0.923 | 0.744 | 0.207 | 0.048 | 0.784 | 0.929 | yes | — |
| Qwen-7B-GRPO-qd-seed1-ckpt12000 | seed1 | 12000 | -0.056 (0.009) | 0.760 (0.024) | 0.762 | 0.773 | 0.180 | 0.047 | 0.805 | 0.940 | yes | — |
| Qwen-7B-GRPO-qd-seed1-ckpt14000 | seed1 | 14000 | -0.035 (0.009) | 0.960 (0.025) | 0.961 | 0.754 | 0.215 | 0.031 | 0.806 | 0.941 | yes | — |
| Qwen-7B-GRPO-qd-seed1-ckpt16000 | seed1 | 16000 | 0.308 (0.015) | 0.853 (0.028) | 0.906 | 0.693 | 0.295 | 0.012 | 0.796 | 0.936 | NO | choice collapse (a keep/trade side <2%) |
| Qwen-7B-GRPO-qd-seed1-ckpt18000 | seed1 | 18000 | 0.207 (0.012) | 0.634 (0.025) | 0.667 | 0.760 | 0.215 | 0.025 | 0.819 | 0.948 | yes | — |
| Qwen-7B-GRPO-qd-seed1-ckpt20000 | seed1 | 20000 | 0.187 (0.011) | 0.597 (0.024) | 0.626 | 0.775 | 0.197 | 0.028 | 0.826 | 0.951 | yes | — |
| Qwen-7B-GRPO-qd-seed1-ckpt22000 | seed1 | 22000 | 0.166 (0.011) | 0.621 (0.024) | 0.643 | 0.778 | 0.193 | 0.029 | 0.830 | 0.953 | yes | — |
| Qwen-7B-GRPO-qd-seed1-ckpt24000 | seed1 | 24000 | 0.200 (0.011) | 0.543 (0.024) | 0.579 | 0.786 | 0.185 | 0.029 | 0.833 | 0.955 | yes | — |
| Qwen-7B-GRPO-qd-seed1-ckpt26000 | seed1 | 26000 | 0.220 (0.012) | 0.649 (0.025) | 0.685 | 0.768 | 0.210 | 0.022 | 0.829 | 0.953 | yes | — |
| Qwen-7B-GRPO-qd-seed1-ckpt28000 | seed1 | 28000 | 0.039 (0.009) | 0.710 (0.024) | 0.711 | 0.796 | 0.170 | 0.034 | 0.838 | 0.956 | yes | — |
| Qwen-7B-GRPO-qd-seed1-ckpt30000 | seed1 | 30000 | 0.090 (0.010) | 0.663 (0.024) | 0.669 | 0.792 | 0.175 | 0.033 | 0.837 | 0.956 | yes | — |
| Qwen-7B-GRPO-qd-seed2-ckpt2000 | seed2 | 2000 | 0.840 (0.036) | 0.485 (0.046) | 0.970 | 0.396 | 0.594 | 0.010 | 0.648 | 0.850 | NO | consistency=0.40<0.5; choice collapse (a keep/trade side <2%) |
| Qwen-7B-GRPO-qd-seed2-ckpt4000 | seed2 | 4000 | 0.309 (0.021) | 0.832 (0.037) | 0.888 | 0.543 | 0.438 | 0.019 | 0.700 | 0.880 | NO | choice collapse (a keep/trade side <2%) |
| Qwen-7B-GRPO-qd-seed2-ckpt6000 | seed2 | 6000 | -0.042 (0.012) | 0.659 (0.029) | 0.660 | 0.711 | 0.226 | 0.063 | 0.755 | 0.910 | yes | — |
| Qwen-7B-GRPO-qd-seed2-ckpt8000 | seed2 | 8000 | -0.019 (0.011) | 0.887 (0.028) | 0.887 | 0.704 | 0.260 | 0.036 | 0.765 | 0.918 | yes | — |
| Qwen-7B-GRPO-qd-seed2-ckpt10000 | seed2 | 10000 | 0.031 (0.011) | 0.834 (0.027) | 0.835 | 0.718 | 0.252 | 0.030 | 0.781 | 0.927 | yes | — |
| Qwen-7B-GRPO-qd-seed2-ckpt12000 | seed2 | 12000 | 0.397 (0.018) | 0.858 (0.031) | 0.946 | 0.622 | 0.370 | 0.008 | 0.766 | 0.921 | NO | choice collapse (a keep/trade side <2%) |
| Qwen-7B-GRPO-qd-seed2-ckpt14000 | seed2 | 14000 | -0.015 (0.010) | 1.019 (0.026) | 1.019 | 0.737 | 0.236 | 0.027 | 0.803 | 0.940 | NO | |eta|=1.02>1.0: status-quo bias remains |
| Qwen-7B-GRPO-qd-seed2-ckpt16000 | seed2 | 16000 | 0.029 (0.009) | 0.804 (0.025) | 0.804 | 0.766 | 0.206 | 0.028 | 0.819 | 0.948 | yes | — |
| Qwen-7B-GRPO-qd-seed2-ckpt18000 | seed2 | 18000 | 0.188 (0.012) | 0.780 (0.027) | 0.803 | 0.739 | 0.243 | 0.018 | 0.817 | 0.947 | NO | choice collapse (a keep/trade side <2%) |
| Qwen-7B-GRPO-qd-seed2-ckpt20000 | seed2 | 20000 | -0.018 (0.008) | 0.755 (0.025) | 0.756 | 0.800 | 0.165 | 0.035 | 0.837 | 0.956 | yes | — |
| Qwen-7B-GRPO-qd-seed2-ckpt22000 | seed2 | 22000 | 0.001 (0.008) | 0.772 (0.024) | 0.772 | 0.795 | 0.173 | 0.032 | 0.838 | 0.957 | yes | — |
| Qwen-7B-GRPO-qd-seed2-ckpt24000 | seed2 | 24000 | 0.035 (0.009) | 0.775 (0.025) | 0.775 | 0.790 | 0.181 | 0.029 | 0.839 | 0.957 | yes | — |
| Qwen-7B-GRPO-qd-seed2-ckpt26000 | seed2 | 26000 | 0.075 (0.010) | 0.822 (0.025) | 0.826 | 0.772 | 0.205 | 0.023 | 0.833 | 0.955 | yes | — |
| Qwen-7B-GRPO-qd-seed2-ckpt28000 | seed2 | 28000 | 0.016 (0.008) | 0.794 (0.024) | 0.794 | 0.793 | 0.177 | 0.030 | 0.839 | 0.958 | yes | — |
| Qwen-7B-GRPO-qd-seed2-ckpt30000 | seed2 | 30000 | 0.000 (0.008) | 0.783 (0.024) | 0.783 | 0.798 | 0.170 | 0.032 | 0.842 | 0.958 | yes | — |

Rows: 40. **15/40 rows are `clean=NO`.** Of those, **10** have `|λ|<0.5` *plus* a contradictory caveat (η, inconsistency, parse failures, or choice collapse) — the direct evidence that λ alone can mislead; the rest are `clean=NO` only because `|λ|≥0.5`.
