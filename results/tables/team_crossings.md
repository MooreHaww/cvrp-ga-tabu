# Team instance: Bosphorus crossings

| algo | n_runs | mean_routes | mean_routes_with_E | mean_crossings | min_crossings | max_crossings | mean_penalty_share_pct | best_run | best_cost | best_routes | best_routes_with_E | best_crossings | best_penalty_share_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ga | 10 | 15.0 | 10.0 | 20.0 | 20 | 20 | 18.41 | 3 | 16204.76 | 15 | 10 | 20 | 18.51 |
| ga_ls | 10 | 15.0 | 10.0 | 20.0 | 20 | 20 | 18.92 | 0 | 15816.26 | 15 | 10 | 20 | 18.97 |
| tabu | 10 | 15.0 | 10.0 | 20.0 | 20 | 20 | 18.77 | 6 | 15869.75 | 15 | 10 | 20 | 18.90 |

crossings = legs whose endpoints have different side labels (depot legs included) = number of times P = 150 is charged; penalty share = crossings * P / cost. Every route with an E customer crosses at least twice.
