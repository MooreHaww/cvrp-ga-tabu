# Drones, Deadlines and NP-Hardness – CVRP with a Genetic Algorithm and Tabu Search

CMP 3005 – Analysis of Algorithms, term project.

## Why this project exists

MediDrone Istanbul (a fictional start-up) delivers medical supplies by drone from a central lab to
clinics on both sides of the Bosphorus. Every drone has a payload capacity, so the deliveries must be
split into several routes of minimum total flight distance. This is the Capacitated Vehicle Routing
Problem (CVRP), an NP-hard generalization of the TSP. The project proves NP-hardness, then designs,
implements, analyzes and evaluates a constructive baseline and two metaheuristics.

## Team

| Name | Student ID (last 4 digits) | Main responsibility |
|---|---|---|
| Mohammad (Moore) Hawwari | _TBD_ | _TBD_ |
| _TBD_ | _TBD_ | _TBD_ |
| _TBD_ | _TBD_ | _TBD_ |

Team-instance seed = sum of the last four digits of all members' student IDs (see the report).

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

_To be filled in at Step 6/7: exact commands for every table and figure in the report._

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
    savings.py         Clarke–Wright savings
    split.py           Split DP (incl. max-K-routes version)
    ga.py              Genetic Algorithm
    tabu.py            Tabu Search
  experiments/         experiment scripts
  results/
    raw/               raw per-run CSV results
    tables/            summary tables
    figures/           plots and route maps
  tests/               pytest test suite
  docs/                course brief, theory write-up, AI-usage log
```

## Tests

```powershell
python -m pytest
```

## Sources

- G. Clarke & J. W. Wright, "Scheduling of vehicles from a central depot to a number of delivery points," *Operations Research*, 1964.
- C. Prins, "A simple and effective evolutionary algorithm for the vehicle routing problem," *Computers & Operations Research*, 2004.
- F. Glover & M. Laguna, *Tabu Search*, Kluwer, 1997.
- P. Toth & D. Vigo (eds.), *Vehicle Routing: Problems, Methods, and Applications*, 2nd ed., SIAM, 2014.
- M. R. Garey & D. S. Johnson, *Computers and Intractability*, 1979.
- CVRPLIB benchmark library (instances and best-known solutions).

---

© 2026 Mohammad (Moore) Hawwari and team.
