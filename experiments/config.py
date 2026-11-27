# Original code
"""Step 6a: experiment configuration – instances, time budgets, seeds, algorithm registry.

Protocol (course brief, Task 7):
  - 10 independent runs per (instance, algorithm); run r uses seed BASE_SEED + r,
    the SAME seed values for every algorithm;
  - the SAME wall-clock budget for every algorithm on an instance (BUDGETS below);
  - every result is validated by cvrp.checker.check.

Parameter overrides (--params-file) may use 'penalty_scale' for the GA: the absolute
penalty is then penalty_scale * default_penalty(inst), computed per instance (Step 6b).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from cvrp.ga import GAParams, default_penalty, genetic_algorithm
from cvrp.instance import Instance
from cvrp.solution import Solution
from cvrp.tabu import TabuParams, tabu_search

ROOT = Path(__file__).resolve().parents[1]

# Seeds: run r (0-based) uses BASE_SEED + r. The value itself is arbitrary but FIXED,
# so every reported number can be reproduced exactly.
BASE_SEED = 20260
DEFAULT_RUNS = 10

# Team seed: sum of the last four digits of the three members' IDs, see report.
TEAM_SEED_DEFAULT = 14835

# Wall-clock budget per run in seconds (equal for all algorithms on an instance).
BUDGETS = {
    "A-n32-k5": 30.0,
    "A-n45-k6": 30.0,
    "A-n60-k9": 60.0,
    "A-n80-k10": 60.0,
    "team": 60.0,
    "tiny_n7": 2.0,
    "tiny_n8": 2.0,
    "tiny_n9": 2.0,
}
INSTANCE_ORDER = tuple(BUDGETS)


def instance_path(name: str, team_seed: int = TEAM_SEED_DEFAULT) -> Path:
    """Path of the .vrp file for an instance name used in this config. Time: O(1)."""
    if name == "team":
        return ROOT / "data" / "team" / f"team_n100_seed{team_seed}.vrp"
    if name.startswith("tiny_"):
        return ROOT / "data" / "tiny" / f"{name}.vrp"
    if name in BUDGETS:
        return ROOT / "data" / "cvrplib" / f"{name}.vrp"
    raise ValueError(f"unknown instance {name!r}; choose from {INSTANCE_ORDER}")


# --------------------------------------------------------------------------- algorithms

@dataclass
class Outcome:
    """Uniform view of one algorithm run, independent of GA/TS result types."""

    solution: Solution | None
    cost: float | None
    n_routes: int | None
    iterations: int            # GA generations or TS iterations
    work: int                  # GA decodes or TS moves evaluated
    progress: list[tuple[float, float]]
    elapsed: float
    params: dict


# Parameters fixed by the protocol: they may NOT come from a params file.
PROTECTED = {"seed", "time_limit", "max_generations", "max_iterations"}


def _checked_overrides(overrides: dict) -> dict:
    """Reject overrides of protocol-controlled fields. Time: O(#overrides)."""
    bad = PROTECTED & set(overrides)
    if bad:
        raise ValueError(f"these parameters are fixed by the protocol and cannot be overridden: {sorted(bad)}")
    return overrides


def _resolve_penalty_scale(inst: Instance, overrides: dict) -> tuple[dict, float | None]:
    """Turn the override 'penalty_scale' s into the absolute GA penalty s * default_penalty(inst).

    The default lambda depends on the instance (2 * mean depot distance), so a params file
    stores the multiplier, not the absolute value. Time: O(n).
    """
    overrides = dict(overrides)
    scale = overrides.pop("penalty_scale", None)
    if scale is None:
        return overrides, None
    if overrides.get("penalty") is not None:
        raise ValueError("give either 'penalty' or 'penalty_scale', not both")
    overrides["penalty"] = float(scale) * default_penalty(inst)
    return overrides, float(scale)


def _run_ga(inst: Instance, seed: int, budget: float, overrides: dict, base: dict) -> Outcome:
    """GA with GAParams defaults, then the variant's `base` settings, then the overrides."""
    overrides, scale = _resolve_penalty_scale(inst, _checked_overrides(overrides))
    params = GAParams(**{**base, **overrides}, seed=seed, time_limit=budget)
    res = genetic_algorithm(inst, params)
    p = asdict(params)
    if scale is not None:
        p["penalty_scale"] = scale      # record the multiplier next to the absolute lambda
    return Outcome(res.solution, res.best_cost, res.n_routes, res.generations, res.decodes,
                   res.progress, res.elapsed, p)


def run_ga(inst: Instance, seed: int, budget: float, overrides: dict) -> Outcome:
    """GA with GAParams defaults (+ overrides). Time: the budget."""
    return _run_ga(inst, seed, budget, overrides, base={"memetic": False})


def run_ga_memetic(inst: Instance, seed: int, budget: float, overrides: dict) -> Outcome:
    """Memetic GA (intra-route 2-opt on offspring). Time: the budget."""
    return _run_ga(inst, seed, budget, overrides, base={"memetic": True})


# Settings that define the ga_ls variant (on top of the GAParams defaults); overrides win.
GA_LS_BASE = {"memetic": True, "ls_mode": "full", "local_search_prob": 0.5}


def run_ga_ls(inst: Instance, seed: int, budget: float, overrides: dict) -> Outcome:
    """Memetic GA with the granular inter-route descent (Step 4c). Time: the budget."""
    return _run_ga(inst, seed, budget, overrides, base=GA_LS_BASE)


def run_tabu(inst: Instance, seed: int, budget: float, overrides: dict) -> Outcome:
    """Tabu Search with TabuParams defaults (+ overrides). Time: the budget."""
    overrides = dict(_checked_overrides(overrides))
    if "moves" in overrides:
        overrides["moves"] = tuple(overrides["moves"])
    params = TabuParams(**overrides, seed=seed, time_limit=budget)
    res = tabu_search(inst, params)
    p = asdict(params)
    p["moves"] = list(p["moves"])
    return Outcome(res.solution, res.cost, res.n_routes, res.iterations, res.moves_evaluated,
                   res.progress, res.elapsed, p)


ALGORITHMS = {"ga": run_ga, "ga_memetic": run_ga_memetic, "ga_ls": run_ga_ls, "tabu": run_tabu}
DEFAULT_ALGOS = ("ga", "tabu")
