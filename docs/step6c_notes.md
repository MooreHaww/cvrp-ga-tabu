# Step 6c – Analysis of the main run: notes

Code: `experiments/analysis.py`. Tests: `tests/test_analysis.py` (a synthetic raw folder; no experiment runs).

```powershell
python -m experiments.analysis
```

The command reads `results/raw/` (read-only since Step 6a-4) and writes `results/tables/*.csv` and `*.md`, and
`results/figures/*.png`.

**Inputs:**
- `runs.csv`, one row per run;
- `progress.csv`, one row per improvement (`instance, algo, run, seed, elapsed_s, best_cost`);
- `solutions/*.json`, the routes of every run, read with `experiments/solutions.py`.

**Determinism:**
- fixed row and column order: instances in `INSTANCE_ORDER`, algorithms `ga, ga_ls, tabu`;
- CSV floats rounded to 6 decimals, with LF line endings;
- a test runs the analysis twice and compares the bytes of every table.

The algorithm colours are the same in every figure: GA blue, GA+LS (memetic) green, Tabu Search red.

## Reference values
| Instance | Reference |
|---|---|
| A-n32-k5 … A-n80-k10 | BKS from the `.sol` file (column `bks_or_opt`) |
| tiny_n7/8/9 | exact optimum from `.opt.json` |
| team | **best known to us: 15816.26** = best feasible team cost over the main run *and* both tuning rounds (`results/tuning*`). It is **not** a proven optimum, so team gaps are relative. |

## Tables
| Table | Content and formulas |
|---|---|
| `main_results` | Per instance × algo: n, failures, best, mean, median, std (ddof 1), `mean_gap_pct = 100·(mean − ref)/ref`, `hits` = runs with \|cost − ref\| < 1e-6, mean wall-clock runtime |
| `cw_baseline` | Clarke–Wright cost and routes (`cvrp.savings.clarke_wright`); per algo `mean_impr_pct = 100·(CW − mean)/CW` and `runs_below_cw` (strictly below) |
| `tiny_results` | Per tiny instance × algo: optimum hit rate; mean / max time-to-best = elapsed time of the **last** improvement record of a run in `progress.csv` |
| `wilcoxon` | See below |
| `team_crossings` | See below |

**Wilcoxon signed-rank test.**
- Pairs: GA+LS vs TS, GA+LS vs GA, and TS vs GA, on every instance.
- Runs are paired by run index. The same run index means the same seed (20260 + r); this is asserted.
- `d = cost(first) − cost(second)`, rounded to 1e-9 so that float noise does not count as a difference. A negative
  `median_diff` means the first algorithm is better.
- Two-sided `scipy.stats.wilcoxon(d, zero_method="wilcox")`: zero differences are dropped, `n_nonzero` is reported,
  and SciPy chooses the exact or the approximate p.
- **"all ties"**: if every d is 0 (all tiny instances; A-n32-k5 for GA+LS vs TS), the test is undefined and is not
  called.
- **Holm** adjustment per pair, over the benchmark + team instances that were tested (`p_holm`).
- **Limitation (10 pairs):**
  - the smallest exact two-sided p is 2/2¹⁰ = 0.00195; with 5 non-zero pairs it is 0.0625, so such a cell can never
    be significant at 5%;
  - a large p is weak evidence of "no difference", and the test does not measure the effect size (use `median_diff`
    and `main_results`).

**Bosphorus crossings** (`team_crossings`, team instance, every run of every algorithm):
- **(i) `routes_with_E`:** routes that visit at least one customer on the E side.
- **(ii) `crossings`:** legs of `0 → route → 0` whose two endpoints have different side labels, depot legs
  included.
  - The side rule is `side(v) = "E" if x < 500 else "A"`, and the depot (500, 500) is "A".
  - This is exactly the number of times P = 150 is charged, because `bosphorus_matrix` adds P iff
    side(i) ≠ side(j).
  - The geometric rule "the segment crosses x = 500" is **not** used.
  - For every solution the code checks **Euclidean length + crossings·P = cost** (tolerance 1e-6) and compares the
    recomputed cost with the stored one.
- **(iii) `penalty_share_pct`** = 100·crossings·P/cost.
- The table gives the means over the 10 runs per algorithm and the values of each algorithm's best run.
- Lower bound: every route with an E customer leaves and re-enters the A-side depot, so crossings ≥ 2·(i).

**Result of the main run.** Every one of the 30 team solutions has 15 routes, 10 of them with E customers, and
exactly **20 crossings**, i.e. the lower bound. Every algorithm crosses the strait only once in each direction per
E route. The penalty is about 18–19% of the cost. The best run overall is GA+LS run 0, with cost 15816.26,
20 crossings and a penalty share of 18.97%.

## Figures
| Figure | Content |
|---|---|
| `convergence_<inst>.png` (A-n45-k6, A-n60-k9, A-n80-k10, team) | Median best-so-far cost over the 10 runs (step functions on a 300-point grid) with the inter-quartile band; **log x axis**, because most improvement happens in the first second; dashed CW and dotted BKS/best-known lines. Each line starts when **all** 10 runs of that algorithm have a feasible solution. |
| `boxplots_final.png` | Final gap % (to the BKS; to best known for the team) per algorithm, 5 panels; every run is also drawn as a point because many cells have identical values |
| `team_routes_ga_ls.png`, `team_routes_tabu.png` | Best team solution of GA+LS and TS (`best_solution`): one colour per route, depot legs drawn, depot star, Bosphorus line x = 500, side labels; the title gives cost, routes, crossings and penalty share |

## Caveats (state them in the report)
- **10 runs per cell.** Means and medians are noisy, and the Wilcoxon power is limited (see above).
- **Wall-clock budgets.** The main run used 3 parallel workers (`results/raw/hardware.txt`). The budgets are equal for
  all algorithms, but the absolute iteration counts depend on the machine.
- **The team reference is not an optimum.** Team gaps compare the algorithms with each other, not with the true
  optimum.
- **The tiny instances do not discriminate.** Every algorithm finds the optimum in every run within milliseconds. They
  only validate correctness.
- **Plain GA.** It improves on its CW seed in almost every run, but only by 0.1–2% (`cw_baseline`). The inter-route
  descent of GA+LS is what closes the gap.
