# Step 6d – Complexity validation (timing experiment): notes

Code: `experiments/timing.py`. Tests: `tests/test_timing.py`, which checks the fits, the verdict rule, the writers
and the smoke run, but never asserts on real timings.

Goal: measure how the running time grows with n and compare the log-log slope with the theoretical bounds.

## How to run (quiet CPU: close other programs, laptop on mains power)
```powershell
python -m experiments.timing
```
This runs sizes 50, 100, 200, 400 and 800 with 7 repetitions and takes about 5–7 minutes. The raw data goes to
`results/timing/`, the table to `results/tables/complexity_fits.{csv,md}` and the figures to `results/figures/`.

```powershell
python -m experiments.timing --smoke
```
The smoke run uses sizes 20 and 40 with 2 repetitions and takes a few seconds. Everything goes to
`results/smoke_timing/`, which is git-ignored.

## Design
- **No algorithm code was changed.** Fixed-count runs already exist:
  - `GAParams.max_generations`;
  - `GAResult.gen_times`, the seconds of each generation, without the initial population;
  - `TabuParams.max_iterations`.

  The main results stay reproducible.
- **Instances:** `generate_team_instance(seed=n, n=n)`, the team-style generator with a fixed seed per n.
  - **Q = 100 for every n**, and demands are U[1, 30] with mean 15.5.
  - So the customers per route stay at **L ≈ 6–7 for every n**, while **K = ⌈Σq/Q⌉ + 1 ≈ 0.155·n grows linearly**.
- **Q-sweep:** n = 200 and Q ∈ {50, 100, 200, 400, 800}, so that L varies at fixed n. The measured
  `L_eff = arcs examined / n` is used on the x axis.
- **Protocol per (component, n):**
  - one untimed warm-up, then 7 timed repetitions;
  - `time.perf_counter`;
  - the garbage collector is off during the timed call (`gc.collect()` before);
  - repetition r uses seed r;
  - one process: the 3 workers of the main run are **not** used here, so runs do not compete for the CPU.

| Component | One repetition | Theory | Theory slope |
|---|---|---|---|
| cw | `clarke_wright` | O(n² log n) | fitted from n² ln n (≈ 2.17 on 100..800) |
| split_unlimited | mean over 20 random tours | O(n·L), L constant | 1 |
| split_k | mean over 5 random tours, `max_routes=K` | O(K·n·L), K ∝ n | 2 |
| ga_generation | mean of `gen_times`, 10 generations, plain GA, tuned (P = 25) | O(P·n·L) | 1 (see caveats) |
| gals_generation | the same for GA+LS (P = 10), 5 generations | O(P·(n·L + passes·n·κ)) | 1 |
| tabu_iteration | (t(50 iterations) − t(0 iterations)) / 50, the same seed, so the CW + neighbour-list setup cancels | O(n·κ) | 1 |
| tabu_delta | 2000 random relocates by the O(1) delta, time per move | O(1) | 0 |
| tabu_full_eval | the same moves by copy + apply + full recomputation | O(n) | 1 |
| descent | one descent from the CW routes, κ = 10 | O(passes·n·κ) | 1 per pass |
| descent_per_pass | the same runs, time / passes | O(n·κ) | 1 |

**Cost-share profile** (`results/timing/share_ga_ls.csv`):
- GA+LS with the tuned parameters, 20 generations, on the team instance (n = 100) and on A-n80-k10.
- For that run only, `cvrp.ga.decode` and `cvrp.ga.descent` are wrapped by perf_counter timers; the originals are
  restored afterwards. This is lighter than cProfile, whose per-call overhead would inflate the many small Split
  calls.
- The shares are descent / Split decode / the rest of the run, initial population included.

## Fits (`complexity_fits.csv` / `.md`)
- **The fit:** `log10(t) = a + b·log10(n)` by least squares (`numpy.polyfit`), on the **median** of the 7 repetitions
  per n.
  - The median is used because timing noise is one-sided: an interruption only makes a repetition slower.
  - SE(b) = √(SSR/(m − 2) / Σ(x − x̄)²); R² = 1 − SSR/SST.
- **Two fits:** all n, and **n ≥ 100** only. At small n the constant per-call costs (function calls, list
  allocation, the tour check in Split) hide the asymptotic growth.
- **theory_slope** is the slope of the theoretical cost function fitted over the *same* sizes. For n² log n on
  100..800 it is ≈ 2.17, not 2, so the log factor is not blamed on the measurement.
- **verdict:** "consistent" if **|slope(n ≥ 100) − theory_slope| ≤ max(2·SE, 0.15)**, otherwise "differs". The
  Q-sweep rows use the fit over all Q values.
  - The **0.15 tolerance** is there because with only 4 sizes the SE itself is unreliable, timer noise is a few
    percent, and cache effects bend the curve slightly.
  - Slope, SE and R² are reported unchanged, so the reader can apply a stricter rule.

## Figures
| Figure | Content |
|---|---|
| `complexity_<component>.png` | log-log, median per n with min–max bars, the n ≥ 100 fit, grey reference slopes 1 (dashed) and 2 (dotted) through the first point |
| `complexity_overview.png` | the same for 9 components in one 3×3 figure |
| `split_vs_L.png` | Split time vs L_eff at n = 200: the unlimited Split grows ∝ L; the K-limited Split stays about flat, because K ∝ 1/Q cancels L |
| `delta_vs_full.png` | µs per relocate, delta vs full re-evaluation, with the speedup (∝ n) on the right axis |

## Caveats (state them in the report)
- **CPU frequency scaling and turbo:** run on mains power with no other load. The hardware line is stored in every
  row and in `hardware.txt`.
- **Python interpreter overhead:** a large constant per operation, so small n look flatter than the asymptotic
  slope.
- **Cache and memory effects:** the n × n distance matrix (as a numpy array and as nested lists) stops fitting in the
  caches at large n, which can push slopes up.
- **Log factors:** CW sorts n² savings, and the neighbour lists sort n lists of n entries.
- **The GA generation is not purely O(P·n·L) here.** An offspring whose unlimited Split needs more than K routes is
  decoded again with the K-limited Split, O(K·n·L). Since K ∝ n in this experiment, that fallback is O(n²).
  - `arcs_per_decode` in `extra_json` does **not** show this. `decode` adds the arcs of a K-limited Split only when
    that Split succeeds, so a failed fallback costs time but adds no arcs.
  - The measured numbers are in "Checks on the full run" below.
- **Descent passes grow with n** (2 at n = 100, 3 at n = 400 in a pilot run), so the whole descent grows faster than
  one pass.
- **Only 5 sizes:** the fits describe this range (50–800), not an asymptotic limit.

## Checks on the full run (read from `results/timing/timing.csv`, medians over 7 repetitions)
- **GA fallback.**
  - The fallback share (decodes that needed the K-limited Split) is **not stored**. It was estimated from time as
    (ms per decode − unlimited Split ms) / K-limited Split ms.
  - That estimate counts all other generation work (selection, crossover, mutation) as fallback, so it is an
    **upper bound**: 0.61–0.77 for n = 50..800.
  - From n = 200 on, the K-limited Split never succeeds on random tours (`split_k` feasible share 0.0). Its cost is
    paid with no benefit: the offspring ends up infeasible in stage 3 anyway.
  - The GA time per decode is about **0.75 × the K-limited Split time** at every n ≥ 200. The GA generation slope for
    n = 100..800 is **1.99**, measured from the two end points only; the least-squares fit in `complexity_fits` gives
    2.01. So the generation is O(n²) in this experiment because of the fallback.
- **Tabu Search iteration.** These slopes are also measured from the two end points, n = 100 and 800 (the fit in the
  table gives 1.18 too). The extra slope, **1.18** against a theory slope of 1.00, splits into two parts:
  - **1.07** from more moves per iteration: 78 → 90 moves evaluated per customer;
  - **0.11** from the time per move: 691 → 866 ns (n = 100 → 800).

  Each move simply gets slower to evaluate as n grows (a cache and memory effect).
- **Delta vs full evaluation.**
  - The O(1) delta still slows by 73% from n = 50 to 800 (0.515 → 0.890 µs per move), which is the same cache and
    memory effect.
  - The full re-evaluation has slope **1.02**, as expected for O(n).
  - The speedup grows from **10×** (n = 50) to **92×** (n = 800).
- **Split vs L** (Q-sweep, n = 200).
  - The arcs examined are exactly n·L.
  - The time is about **126 µs fixed per call + 0.076 µs per arc**. So the time grows less than linearly in L: the
    fixed part (prefix sums, the tour check, allocation) does not depend on L.
