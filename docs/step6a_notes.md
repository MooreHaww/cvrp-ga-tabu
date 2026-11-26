# Step 6a – Experiment runner: notes

Code: `experiments/config.py` (instances, budgets, seeds, algorithm registry) and `experiments/run_experiments.py`
(runner CLI). Tests: `tests/test_experiments.py`.

## Protocol (course brief, Task 7)

| Rule | How it is implemented |
|---|---|
| 10 independent runs per (instance, algorithm) | `--runs 10` (default). Run r uses **seed = BASE_SEED + r**, with `BASE_SEED = 20260` in `config.py`. |
| Seeds recorded | `seed` column in every row. GA and TS use the **same** seed values (tested). |
| Equal time budget | `BUDGETS` in `config.py`, the same for every algorithm on an instance: A-n32-k5 30 s, A-n45-k6 30 s, A-n60-k9 60 s, A-n80-k10 60 s, team 60 s, tiny_n7/8/9 2 s. `--budget-scale` multiplies all budgets, which is only used for smoke runs. |
| Same machine conditions | Interleaved order: for each instance → for each run r → for each algorithm. GA run r and TS run r follow each other. |
| Validation | Every result goes through `checker.check(inst, solution, claimed_cost)`. A run with no feasible solution is recorded with `feasible=False` and an empty cost, never dropped. |
| Hardware reported | `hardware.txt` is written at the start of every invocation. |
| Raw CSV of all runs | `runs.csv` and `progress.csv`. |

**Algorithms.**
- `ga`: `GAParams` defaults.
- `tabu`: `TabuParams` defaults.
- `ga_memetic`: GA with `memetic=True`.

The default selection is `--algos ga,tabu`.

**Parameter overrides (for Step 6b).** `--params-file` takes JSON such as `{"ga": {"pop_size": 80}, "tabu":
{"tenure_max": 20}}`. Each algorithm uses its own section.
- `seed`, `time_limit`, `max_generations` and `max_iterations` cannot be overridden: they are fixed by the protocol,
  and trying raises ValueError.
- The exact parameters of each run are stored in its `params_json` column.

**Team instance.** `--team-seed S`, default 0, with a TODO for the real seed. The file
`data/team/team_n100_seed{S}.vrp` is generated if it is missing.

## File formats (`--out`, default `results/raw`)

**`runs.csv`**: one row per run.

| column | meaning |
|---|---|
| instance, algo, run, seed | identification; seed = 20260 + run |
| budget_s, elapsed_s | time limit and actual wall-clock time (GA may exceed the limit by at most one generation) |
| cost, n_routes, K | best feasible distance (empty if none), its non-empty routes, routes allowed |
| feasible, check_errors | checker verdict and its messages (`;`-separated) |
| bks_or_opt | BKS from the .sol file (Augerat), exact optimum from .opt.json (tiny), empty (team) |
| gap_pct | 100·(cost − bks_or_opt)/bks_or_opt; empty for the team instance and infeasible runs |
| iterations | GA generations or TS iterations |
| work | GA decodes or TS moves evaluated |
| reached_optimum | tiny instances only: \|cost − opt\| < 1e-6 |
| time_to_best_s | elapsed time of the last improvement |
| params_json | all algorithm parameters actually used, including seed and time limit |
| git_commit | commit hash; `-dirty` if tracked files were modified (official runs must be clean) |

**`progress.csv`**: one row per improvement of the best feasible cost: instance, algo, run, seed, elapsed_s, best_cost.
These rows are the convergence curves.

**`hardware.txt`**: date, OS, CPU model, physical and logical cores, RAM, Python and numpy versions, git commit,
and worker count. It uses the standard library only; any item it cannot determine reads "unknown".

## How to run

The smoke run covers all instances with 2 runs and 5% budgets, and writes to `results/smoke/`. That folder is
git-ignored, so `results/raw` is never touched. It takes about 1 minute:

```powershell
python -m experiments.run_experiments --smoke
```

The full protocol writes to `results/raw/`:

```powershell
python -m experiments.run_experiments
```

Options: `--algos ga,tabu,ga_memetic`, `--instances A-n32-k5,team,...`, `--runs`, `--team-seed`, `--params-file`,
`--workers`.

**Resume.** Rows are appended and flushed after every run. When restarted with the same `--out`, the runner skips
every (instance, algo, run) already in `runs.csv`, so an interrupted run simply continues. If you change parameters,
use a new `--out` folder, because resume does not compare parameters.

**Workers.** `--workers 1` is the default: runs are sequential, which gives the best timing quality. With
`--workers N`, runs go to a process pool and the main process writes the rows. For official numbers:
- use few workers;
- keep at least 2 cores free for the OS;
- use the same setting for all runs, because concurrent runs slow each other down and the budgets are wall-clock.

The machine has 8 cores, so use at most 4 workers. Even then, compare only results produced with the same setting.

## Estimated time of the full run (`ga,tabu`, 10 runs, sequential)

| part | calculation | time |
|---|---|---|
| Augerat | (30 + 30 + 60 + 60) s × 10 runs × 2 algorithms | 3,600 s |
| team instance | 60 s × 10 × 2 | 1,200 s |
| tiny instances | 3 × 2 s × 10 × 2 | 120 s |
| **total** | | **4,920 s ≈ 82 min** |

Overhead per run is small: the smoke run's budgets add up to 49.2 s, and it took 51 s in total. Adding `ga_memetic`
costs another ≈ 41 min. With `--workers 4` the wall-clock time is about 21 min, but timings are less clean.

## Smoke run (this step)

`python -m experiments.run_experiments --smoke` produced 32 rows, all feasible and validated, plus 93 progress rows
and `hardware.txt` (AMD Ryzen 7 9800X3D, 8 physical / 8 logical cores, 31.1 GB RAM, Python 3.12.10, numpy 2.5.5).

The smoke budgets are tiny (1.5–3 s), so its numbers are **not** results.

## Known issues found by the smoke run (to resolve before the full run)

1. **The tiny-instance optimum check is trivial.**
   - Clarke–Wright is already optimal on tiny_n7/8/9, and both algorithms start from it by default (the GA seeds its
     population with CW, and TS starts at CW).
   - So `reached_optimum` is always True and says nothing about the algorithms.
   - For the GA, a params file with `"ga": {"seed_fraction": 0}` makes the test meaningful.
   - TS always starts from CW; giving it a different start would need a TS change.
2. **Tabu Search gets stuck slightly infeasible on tight instances.**
   - A-n80-k10 fills 94.2% of the total capacity K·Q (942 of 1000).
   - TS leaves the feasible region early, while λ is small, and then cannot remove the last unit of overload. Every
     move that could fix it lies outside the 10-nearest-neighbour candidate list.
   - λ then climbs to its cap (10,000), and no feasible improvement over Clarke–Wright is found, even in 10 s.
   - Probe runs (10 s, seed 20260):

     | | candidate_size 10 | 30 | all |
     |---|---|---|---|
     | A-n60-k9 | 3.62%, λ at cap | 0.30% | 0.74% |
     | A-n80-k10 | 4.37%, λ at cap | 4.31% | 4.31% |

   - This needs a fix in `tabu.py`, for example a repair neighbourhood that is always evaluated while the solution
     is infeasible, or a larger candidate list. It must be fixed before the official run.
   - **Resolved** by the commit "Fix TS: skip null moves and add overload repair neighbourhood". The diagnosis found a
     second cause, null moves, and the before/after table is in `docs/step5_notes.md`, section "Fix".
