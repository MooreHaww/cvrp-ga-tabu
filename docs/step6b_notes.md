# Step 6b – Parameter study: notes

Code:
- `experiments/tuning_grid.py`: the grid;
- `experiments/tuning.py`: the runner and the analysis;
- a `penalty_scale` override in `experiments/config.py`.

Tests: `tests/test_tuning.py`.

Goals:
1. The sensitivity study required by Task 7 ("at least one parameter study with a plot").
2. A documented rule that picks the parameter values for the final experiments **without touching the final seeds**.

## Design

| Item | Choice |
|---|---|
| Method | One factor at a time (OFAT) around the current defaults. |
| Seeds | Tuning run r uses **1000 + r**. Selection uses r = 0..4 (seeds 1000–1004) and validation r = 5..9 (1005–1009). The final seeds 20260–20269 are never used; `tuning_seed()` raises if one would be. |
| Instances | A-n45-k6, A-n60-k9, and the team instance (seed 14835). |
| Budget | 10 s per run. `--budget-scale` multiplies it. |
| Order | Interleaved: instance → run → algorithm → configuration. |
| Validation of runs | Every run goes through `run_experiments.run_one`, which calls `checker.check`. |

**Grid** (`experiments/tuning_grid.py`, defaults in brackets):

| Algorithm | Parameter | Values | Configs |
|---|---|---|---|
| tabu | tenure (tenure_min-tenure_max) | 2-6, [5-15], 10-25 | |
| tabu | candidate_size | 5, [10], 20, None | |
| tabu | lambda_adapt (lam_factor-adapt_every) | [1.5-10], 2-5, 3-3 | |
| tabu | repair_after | 5, [10], 30 | |
| tabu | **total** | 9 non-default + 1 baseline | **10** |
| ga | pop_size | 25, [50], 100 | |
| ga | mutation_rate | 0.05, [0.2], 0.5 | |
| ga | crossover_rate | 0.6, [0.9] | |
| ga | elite_count | [2], 5, 10, 20 | |
| ga | seed_fraction | 0, [0.1], 0.3 | |
| ga | penalty_scale | 0.5, [1], 2 | |
| ga | **total** | 12 + 1 | **13** |
| ga_memetic | local_search_prob | [0.1], 0.5, 1.0 | **3** |

That is **26 configurations** in total.

**Shared baseline.** The all-defaults configuration is the same for every factor of an algorithm. It is run once per
(algorithm, instance, seed), with `param_name = baseline`, and it supplies the default point of every factor's
statistics and plot. A test checks that the grid defaults equal the `GAParams()` / `TabuParams()` defaults.

**Penalty scale.** The GA's default λ = 2·mean d(0,i) depends on the instance. The grid and `tuned_params.json`
therefore store a multiplier, `penalty_scale`. `experiments/config.py` turns it into the absolute
`penalty = scale · default_penalty(inst)` for each instance. `params_json` records both values. Giving both `penalty`
and `penalty_scale` is an error.

## Metric

The normalised gap of a run is `100·(cost − ref)/ref`:
- Augerat instances: ref = the BKS, as computed at run time.
- Team instance: ref = the best feasible cost seen in **any** tuning or validation run, computed at analysis time.
  The `normalised_gap` column is therefore empty for team rows in `runs.csv`.

Infeasible runs are **failures**. They are counted in the `failures` column and excluded from the mean.

Per cell (algorithm, parameter, value, instance) we report:
- n;
- the mean;
- the std (ddof = 1);
- SE = std/√n;
- the failures.

The instance average is the mean of the per-instance means, with SE_avg = √(Σ SE_i²)/#instances (the instances are
independent). It is undefined if an instance has no feasible run.

## Selection rule (`select` in `tuning.py`)

For each algorithm and each parameter, independently:
1. **best** = the value with the lowest instance-averaged mean gap (ties go to the default).
2. Change from the default **only if** `mean_default − mean_best > max(SE_default, SE_best)` **and** best has no
   more failures than the default.
3. Otherwise keep the default. The default is also kept when an SE is unknown, that is, when some cell has fewer
   than 2 feasible runs. This is why the 1-run smoke analysis changes nothing.

`experiments/tuned_params.json`:
- holds only the changed keys, with pairs expanded (`tenure_min` + `tenure_max`) and the penalty written as
  `penalty_scale`;
- always has the sections `ga`, `tabu` and `ga_memetic`;
- is in the `--params-file` format of `run_experiments`;
- is checked against `PROTECTED`, so it never contains seed, time_limit or max_*.

`ga_memetic` receives only its own `local_search_prob`, because its study ran with the GA defaults. Every decision,
with its means, SEs, margin and failures, is written to `results/tables/tuning_selection.csv`.

**Validation.** OFAT picks each parameter separately, so it ignores interactions. Two changes that each help can hurt
in combination. `run --validate` therefore runs tuned vs default for GA and TS on the 3 instances × 5 **held-out**
seeds (1005–1009). Using the selection seeds would favour the tuned configuration, because it was chosen on those
runs.

`analyze` writes `results/tables/tuning_validation.csv`. It prints a WARNING if the tuned configuration is worse on
average by more than the larger SE, or has more failures. The user then decides whether to drop that algorithm's
overrides; they are not reverted automatically.

## Outputs

| File | Content |
|---|---|
| `results/tuning/runs.csv` | one row per run: instance, algo, param_name, param_value, run, seed, budget_s, cost, n_routes, feasible, normalised_gap, iterations, work, params_json, git_commit |
| `results/tuning/validation_runs.csv` | same columns; param_name = validation, param_value = default / tuned |
| `results/tuning/hardware.txt` | hardware report (same function as Step 6a) |
| `results/tables/tuning_summary.csv` | mean, std, SE, failures per (algo, param, value, instance) and the instance average |
| `results/tables/tuning_selection.csv` | the selection decisions |
| `results/tables/tuning_validation.csv` | tuned vs default |
| `results/figures/sensitivity_<algo>_<param>.png` | 11 plots |
| `experiments/tuned_params.json` | the chosen overrides |

The plots have the parameter values on the x axis, as categories because some values are pairs or None. The y axis is
the mean normalised gap in %. Each instance is a thin line with SE error bars, the instance average is a thick black
line, and the default is shaded and labelled "(default)".

**Resume.** Rows are flushed after every run. A restarted run skips every (instance, algo, param_name, param_value,
run) already in the CSV.

## Runtime

| Stage | Count | 1 worker | `--workers 3` |
|---|---|---|---|
| Study | 26 configs × 3 instances × 5 runs = 390 runs × 10 s = 3,900 s | ≈ 65 min | ≈ 22 min |
| Validation | 2 algorithms × 2 configs × 3 instances × 5 runs = 60 runs × 10 s = 600 s | 10 min | ≈ 3.5 min |
| Smoke run | 26 × 3 × 1 run × 0.5 s | ≈ 40 s | |

The budgets are wall-clock, and parallel runs slow each other down slightly. Use the **same** `--workers` setting for
the whole study and its validation. The machine has 8 cores, so 3 workers leave room for the OS. Overhead per run is
small (Step 6a measured about 4%).

## Commands (PowerShell, repo root, venv active)

```powershell
python -m experiments.tuning run --smoke
```

```powershell
python -m experiments.tuning run --workers 3
```

```powershell
python -m experiments.tuning analyze
```

```powershell
python -m experiments.tuning run --validate --workers 3
```

```powershell
python -m experiments.tuning analyze
```

```powershell
python -m experiments.run_experiments --params-file experiments/tuned_params.json
```

`analyze --smoke` analyses the smoke output and writes all its tables, plots and JSON inside
`results/smoke_tuning/`, so the committed outputs are never overwritten by smoke data.

## Limitations (state them in the report)

- **OFAT ignores interactions** between parameters. The validation step guards only against a bad combination of the
  chosen values; it does not search for a better one.
- **5 runs per cell.** SEs are wide, so only clear effects pass the selection rule. This is conservative by design.
- **Only 3 instances.** The choice may not transfer to A-n32-k5 / A-n80-k10 or the tiny instances.
- **Short budgets.** The study uses 10 s versus 30–60 s in the final runs. Settings that pay off late, such as a
  larger population, may be undervalued.
- **Team reference = the best cost seen.** It is not a proven optimum, so team gaps are relative, not absolute.
