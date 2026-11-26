# Drones, Deadlines and NP-Hardness – CVRP with a Genetic Algorithm and Tabu Search

CMP 3005 – Analysis of Algorithms, term project.

## Why this project exists

MediDrone Istanbul (a fictional start-up) delivers medical supplies by drone from a central lab to
clinics on both sides of the Bosphorus. Every drone has a payload capacity, so the deliveries must be
split into several routes of minimum total flight distance. This is the Capacitated Vehicle Routing
Problem (CVRP), an NP-hard generalization of the TSP. The project proves NP-hardness, then designs,
implements, analyzes and evaluates a constructive baseline and two metaheuristics.

## Team

- Joohyun Lee
- Ahmet Emre Cinel
- Mohammad Ammar Hawwari

Team-instance seed = **14835**, the sum of the last four digits of the three members' IDs (see the report).

## What it does

- Parses CVRPLIB / TSPLIB instances (Augerat set A) and their best-known solutions.
- Generates our own MediDrone team instance (100 customers, Bosphorus penalty P = 150) and tiny
  instances (n = 7, 8, 9) with exact optima.
- Solves CVRP with:
  - Clarke–Wright savings (deterministic baseline),
  - a Genetic Algorithm on giant tours decoded by the Split dynamic program,
  - Tabu Search on route-based solutions.
- Validates every reported solution with an independent feasibility checker.
- Runs reproducible experiments (fixed seeds, equal time budgets) and writes raw CSV results, tables and plots.

## Setup

Windows, PowerShell, Python 3.11+ (we use 3.12). From the repository root:

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

```powershell
pip install -e .
```

```powershell
python -c "import numpy, matplotlib, pytest, scipy, cvrp; print('ok')"
```

If activation is blocked by the execution policy, run this once and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Reproducing results

Run all commands from the repository root with the venv active. The data files are already committed. These
commands recreate them.

**Benchmark data (Augerat set A, CVRPLIB).** The files are already in `data/cvrplib/`. To re-download them:

```powershell
$base = "https://galgos.inf.puc-rio.br/cvrplib/en/download"
$ids = @{ "A-n32-k5" = 4; "A-n45-k6" = 16; "A-n60-k9" = 23; "A-n80-k10" = 31 }
foreach ($name in $ids.Keys) {
    Invoke-WebRequest "$base/instance/$($ids[$name])" -OutFile "data/cvrplib/$name.vrp"
    Invoke-WebRequest "$base/bks/$($ids[$name])" -OutFile "data/cvrplib/$name.sol"
}
```

**Team instance.** Writes `data/team/team_n100_seed<N>.vrp`. The team seed is 14835 (the default), the sum of the
last four digits of the three members' IDs. The old placeholder file `team_n100_seed0.vrp` is kept for the tests and
the Step 3/5 benchmarks.

```powershell
python -m cvrp.generator --seed 14835
```

**Tiny instances and their exact optima.** Writes `data/tiny/tiny_n{7,8,9}.vrp` and `.opt.json`. The generator
searches for the first seed where Clarke–Wright is not optimal; GA and TS start from CW, so reaching the optimum is a
real test. The chosen seeds are **n=7: 12, n=8: 22, n=9: 19**, with CW 3.96%, 3.46% and 4.56% above the optimum.

```powershell
python -m cvrp.generator --tiny
```

```powershell
python -m cvrp.exact_small
```

**Milestone 1: summary table and team-instance plot.** The plot goes to `results/figures/team_instance.png`.

```powershell
python -m cvrp.summary
```

```powershell
python -m cvrp.plot_instance data/team/team_n100_seed14835.vrp --out results/figures/team_instance.png
```

**Clarke–Wright baseline.** Prints the cost, number of routes vs K, gap to the BKS, runtime, the checker verdict
and the routes:

```powershell
python -m cvrp.savings data/cvrplib/A-n32-k5.vrp
```

**Split DP timing.** Mean time per `split()` call on 200 random tours:

```powershell
python -m cvrp.split --benchmark
```

**Genetic Algorithm.** Giant tour + Split. Add `--memetic` for intra-route 2-opt. The other options are `--pop`,
`--pc`, `--pm`, `--gens` and `--penalty`:

```powershell
python -m cvrp.ga data/cvrplib/A-n32-k5.vrp --seed 0 --time 10
```

**Tabu Search.** Granular, with soft capacity and adaptive λ. The other options are `--iters`, `--tenure-min`,
`--tenure-max`, `--cand` and `--moves relocate,swap,two_opt_star,two_opt`:

```powershell
python -m cvrp.tabu data/cvrplib/A-n32-k5.vrp --seed 0 --time 10
```

**Delta-evaluation benchmark** (Task 6). Compares the O(1) delta with a full re-evaluation per move:

```powershell
python -m cvrp.tabu --benchmark-delta
```

### Main experiments (Task 7)

The protocol is 10 seeded runs per instance and algorithm, with equal time budgets for GA and TS. Raw CSV goes to
`results/raw/`. Details are in `docs/step6a_notes.md`.

**Quick check of the whole pipeline.** About 1 minute; writes to `results/smoke/`, which is not committed:

```powershell
python -m experiments.run_experiments --smoke
```

**Full run.** GA and Tabu Search, 10 runs, sequential. About 82 minutes. It can be resumed after an interruption,
because finished runs are skipped:

```powershell
python -m experiments.run_experiments
```

### Parameter study (Task 7, Step 6b)

One-factor-at-a-time study around the defaults: 26 configurations × 3 instances (A-n45-k6, A-n60-k9, team) × 5 runs
× 10 s, on tuning seeds 1000–1004 (never the final seeds 20260–20269). Details are in `docs/step6b_notes.md`.

**Quick check.** 1 run, 0.5 s budgets, into `results/smoke_tuning/` (not committed):

```powershell
python -m experiments.tuning run --smoke
```

**Study.** About 65 minutes, or about 22 minutes with `--workers 3`. It can be resumed:

```powershell
python -m experiments.tuning run --workers 3
```

**Analysis.** Writes `results/tables/tuning_summary.csv`, `tuning_selection.csv`, the plots
`results/figures/sensitivity_<algo>_<param>.png` and `experiments/tuned_params.json`:

```powershell
python -m experiments.tuning analyze
```

**Validation.** Tuned vs default for GA and TS on the held-out seeds 1005–1009, then analyze again to write
`results/tables/tuning_validation.csv`:

```powershell
python -m experiments.tuning run --validate --workers 3
```

```powershell
python -m experiments.tuning analyze
```

**Final run with the tuned parameters:**

```powershell
python -m experiments.run_experiments --params-file experiments/tuned_params.json
```

Still to come:
- TODO Step 6c: summary tables, convergence plots, route plots and Wilcoxon test;
- TODO Step 7: final report figures.

## Project structure

```
cvrp-project/
  CLAUDE.md            project rules for AI-assisted sessions
  README.md
  requirements.txt
  pyproject.toml
  data/
    cvrplib/           Augerat A instances (.vrp) + best-known solutions (.sol)
    team/              our MediDrone instance
    tiny/              tiny instances (n = 7, 8, 9) with exact optima
  src/cvrp/
    instance.py        Instance dataclass + distance functions
    parser.py          CVRPLIB/TSPLIB .vrp and .sol parsing
    generator.py       team + tiny instance generator
    solution.py        solution representation + cost
    checker.py         independent feasibility checker
    exact_small.py     exact solver for n <= 9
    summary.py         Milestone 1 summary table
    plot_instance.py   Milestone 1 instance plot
    savings.py         Clarke–Wright savings
    split.py           Split DP (incl. max-K-routes version)
    ga.py              Genetic Algorithm
    local_search.py    2-opt (memetic GA) + O(1) move deltas (relocate, swap, 2-opt*)
    tabu.py            Tabu Search
  experiments/         experiment scripts (config.py, run_experiments.py, tuning.py, tuning_grid.py)
  results/
    raw/               raw per-run CSV results
    tables/            summary tables
    figures/           plots and route maps
  tests/               pytest test suite
  docs/                course brief, theory write-up, AI-usage log
```

## Tests

Run the full suite:

```powershell
python -m pytest
```

Slow exhaustive checks (for example Split over all 9! permutations, about 7 s) are skipped by default. Run them with:

```powershell
python -m pytest -m slow
```

Run one test file:

```powershell
python -m pytest tests/test_checker.py
```

The suite covers parsing of all four Augerat instances. It checks that each BKS .sol passes the checker with the
stated cost, and that the checker catches every kind of violation. For the team instance it tests the triangle
inequality on all 101³ triples, the side labels, determinism and the write → parse round trip. It also checks that
the exact solver matches a brute force on `tiny_n7`. Step-by-step design notes are in `docs/step1_notes.md`.

## Sources

- G. Clarke & J. W. Wright, "Scheduling of vehicles from a central depot to a number of delivery points," *Operations Research*, 1964.
- C. Prins, "A simple and effective evolutionary algorithm for the vehicle routing problem," *Computers & Operations Research*, 2004.
- F. Glover & M. Laguna, *Tabu Search*, Kluwer, 1997.
- P. Toth & D. Vigo (eds.), *Vehicle Routing: Problems, Methods, and Applications*, 2nd ed., SIAM, 2014.
- M. R. Garey & D. S. Johnson, *Computers and Intractability*, 1979.
- CVRPLIB benchmark library (instances and best-known solutions).

---

© 2026 Mohammad (Moore) Hawwari and team.
