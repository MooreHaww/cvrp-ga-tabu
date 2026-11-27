# Improvement over Clarke-Wright

| instance | cw_cost | cw_routes | ga_mean_impr_pct | ga_runs_below_cw | ga_ls_mean_impr_pct | ga_ls_runs_below_cw | tabu_mean_impr_pct | tabu_runs_below_cw |
|---|---|---|---|---|---|---|---|---|
| A-n32-k5 | 842.00 | 5 | 1.781 | 10 | 6.888 | 10 | 6.888 | 10 |
| A-n45-k6 | 997.00 | 7 | 2.036 | 10 | 4.955 | 10 | 5.226 | 10 |
| A-n60-k9 | 1408.00 | 9 | 0.128 | 9 | 3.835 | 10 | 3.750 | 10 |
| A-n80-k10 | 1840.00 | 10 | 0.223 | 10 | 3.668 | 10 | 3.543 | 10 |
| team | 16599.05 | 15 | 1.854 | 10 | 4.495 | 10 | 3.717 | 10 |
| tiny_n7 | 289.00 | 4 | 3.806 | 10 | 3.806 | 10 | 3.806 | 10 |
| tiny_n8 | 449.00 | 3 | 3.341 | 10 | 3.341 | 10 | 3.341 | 10 |
| tiny_n9 | 390.00 | 3 | 4.359 | 10 | 4.359 | 10 | 4.359 | 10 |

mean_impr_pct = 100 * (CW - mean cost) / CW; runs_below_cw = runs strictly below CW.
