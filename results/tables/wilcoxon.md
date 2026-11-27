# Wilcoxon signed-rank tests (paired by seed)

| instance | pair | n_pairs | n_nonzero | result | statistic | p | p_holm | median_diff |
|---|---|---|---|---|---|---|---|---|
| A-n32-k5 | ga_ls vs tabu | 10 | 0 | all ties | – | – | – | 0.00 |
| A-n45-k6 | ga_ls vs tabu | 10 | 5 | tested | 1.00 | 0.125 | 0.25 | 0.00 |
| A-n60-k9 | ga_ls vs tabu | 10 | 6 | tested | 0.00 | 0.03125 | 0.09375 | -1.00 |
| A-n80-k10 | ga_ls vs tabu | 10 | 10 | tested | 22.00 | 0.625 | 0.625 | -4.50 |
| team | ga_ls vs tabu | 10 | 10 | tested | 0.00 | 0.001953 | 0.007812 | -154.78 |
| tiny_n7 | ga_ls vs tabu | 10 | 0 | all ties | – | – | – | 0.00 |
| tiny_n8 | ga_ls vs tabu | 10 | 0 | all ties | – | – | – | 0.00 |
| tiny_n9 | ga_ls vs tabu | 10 | 0 | all ties | – | – | – | 0.00 |
| A-n32-k5 | ga_ls vs ga | 10 | 10 | tested | 0.00 | 0.001953 | 0.009766 | -43.00 |
| A-n45-k6 | ga_ls vs ga | 10 | 10 | tested | 0.00 | 0.001953 | 0.009766 | -31.00 |
| A-n60-k9 | ga_ls vs ga | 10 | 10 | tested | 0.00 | 0.001953 | 0.009766 | -52.00 |
| A-n80-k10 | ga_ls vs ga | 10 | 10 | tested | 0.00 | 0.001953 | 0.009766 | -64.50 |
| team | ga_ls vs ga | 10 | 10 | tested | 0.00 | 0.001953 | 0.009766 | -446.33 |
| tiny_n7 | ga_ls vs ga | 10 | 0 | all ties | – | – | – | 0.00 |
| tiny_n8 | ga_ls vs ga | 10 | 0 | all ties | – | – | – | 0.00 |
| tiny_n9 | ga_ls vs ga | 10 | 0 | all ties | – | – | – | 0.00 |
| A-n32-k5 | tabu vs ga | 10 | 10 | tested | 0.00 | 0.001953 | 0.009766 | -43.00 |
| A-n45-k6 | tabu vs ga | 10 | 10 | tested | 0.00 | 0.001953 | 0.009766 | -32.00 |
| A-n60-k9 | tabu vs ga | 10 | 10 | tested | 0.00 | 0.001953 | 0.009766 | -51.00 |
| A-n80-k10 | tabu vs ga | 10 | 10 | tested | 0.00 | 0.001953 | 0.009766 | -62.00 |
| team | tabu vs ga | 10 | 10 | tested | 0.00 | 0.001953 | 0.009766 | -313.32 |
| tiny_n7 | tabu vs ga | 10 | 0 | all ties | – | – | – | 0.00 |
| tiny_n8 | tabu vs ga | 10 | 0 | all ties | – | – | – | 0.00 |
| tiny_n9 | tabu vs ga | 10 | 0 | all ties | – | – | – | 0.00 |

d = cost(first) - cost(second), paired by run index (same seed); negative median_diff = the first algorithm is better. Two-sided Wilcoxon signed-rank test (scipy.stats.wilcoxon, zero differences dropped). 'all ties' = every paired difference is 0, so the test is not defined and was not called. p_holm = Holm adjustment per pair over the benchmark + team instances that were tested.

Limitation: only 10 pairs per test. The smallest possible exact two-sided p is 2/2^10 = 0.00195, and with fewer non-zero pairs it is much larger (5 non-zero pairs: 0.0625). A large p is weak evidence of 'no difference'; the test says nothing about the size of the difference (see median_diff and main_results).
