# Original code
"""Step 6b: the one-factor-at-a-time (OFAT) parameter grid (round 1: GRID;
round 2: GRID_R2 for ga_ls and the named Tabu Search configurations TS_CONFIGS_R2).

Each Factor varies one parameter (or a fixed pair, e.g. tenure_min/tenure_max) while
all other parameters stay at their defaults. The all-defaults configuration is the
same for every factor of an algorithm, so it is run once ("baseline") and shared.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Factor:
    """One studied parameter: the GAParams/TabuParams fields it sets and its values."""

    name: str                      # short name, used in file names and CSV rows
    keys: tuple[str, ...]          # override keys set by one value (pairs: two keys)
    values: tuple[tuple, ...]      # each value is a tuple with one entry per key
    default: tuple                 # the current default (must be one of values)
    label: str                     # x-axis label, with units

    def overrides(self, value: tuple) -> dict:
        """Override dict for one value, e.g. {'tenure_min': 2, 'tenure_max': 6}. Time: O(#keys)."""
        return dict(zip(self.keys, value))


def value_label(value: tuple) -> str:
    """Readable label of a value: '10', 'None', '2-6'. Time: O(#keys)."""
    return "-".join(str(v) for v in value)


GRID: dict[str, list[Factor]] = {
    "tabu": [
        Factor("tenure", ("tenure_min", "tenure_max"), ((2, 6), (5, 15), (10, 25)), (5, 15),
               "tabu tenure range [min-max] (iterations)"),
        Factor("candidate_size", ("candidate_size",), ((5,), (10,), (20,), (None,)), (10,),
               "candidate list size kappa (nearest neighbours; None = all)"),
        Factor("lambda_adapt", ("lam_factor", "adapt_every"), ((1.5, 10), (2, 5), (3, 3)), (1.5, 10),
               "lambda adaptation [factor-window] (x, iterations)"),
        Factor("repair_after", ("repair_after",), ((5,), (10,), (30,)), (10,),
               "repair neighbourhood after (infeasible iterations)"),
    ],
    "ga": [
        Factor("pop_size", ("pop_size",), ((25,), (50,), (100,)), (50,),
               "population size (individuals)"),
        Factor("mutation_rate", ("mutation_rate",), ((0.05,), (0.2,), (0.5,)), (0.2,),
               "mutation rate (probability per offspring)"),
        Factor("crossover_rate", ("crossover_rate",), ((0.6,), (0.9,)), (0.9,),
               "crossover rate (probability per offspring)"),
        Factor("elite_count", ("elite_count",), ((2,), (5,), (10,), (20,)), (2,),
               "elite count (individuals)"),
        Factor("seed_fraction", ("seed_fraction",), ((0,), (0.1,), (0.3,)), (0.1,),
               "Clarke-Wright seed fraction (share of population)"),
        Factor("penalty_scale", ("penalty_scale",), ((0.5,), (1,), (2,)), (1,),
               "penalty scale (x default lambda = 2 * mean d(0,i))"),
    ],
    "ga_memetic": [
        Factor("local_search_prob", ("local_search_prob",), ((0.1,), (0.5,), (1.0,)), (0.1,),
               "2-opt probability (per offspring)"),
    ],
}

BASELINE = ("baseline", "default")   # (param_name, param_value) of the shared all-defaults config


# --------------------------------------------------------------------------- round 2 (Step 6b-2)

# ga_ls: OFAT around the ga_ls defaults (GAParams defaults + experiments.config.GA_LS_BASE).
GRID_R2: dict[str, list[Factor]] = {
    "ga_ls": [
        Factor("local_search_prob", ("local_search_prob",), ((0.1,), (0.3,), (0.5,), (1.0,)), (0.5,),
               "descent probability (per offspring)"),
        Factor("pop_size", ("pop_size",), ((10,), (25,), (50,)), (50,), "population size (individuals)"),
        Factor("ls_neighbours", ("ls_neighbours",), ((5,), (10,), (20,)), (10,),
               "descent neighbour list size kappa (nearest customers)"),
        Factor("elite_count", ("elite_count",), ((2,), (5,), (10,)), (2,), "elite count (individuals)"),
    ],
}

# Tabu Search: named configurations, not OFAT. "tuned_all" is the round-1 choice; each
# "tuned_minus_*" drops one of its three changes, to expose interactions (Step 6b-2).
_TUNED_ALL = {"tenure_min": 10, "tenure_max": 25, "candidate_size": 20, "lam_factor": 3, "adapt_every": 3}
TS_CONFIGS_R2: list[tuple[str, dict]] = [
    ("default", {}),
    ("tuned_all", dict(_TUNED_ALL)),
    ("tuned_minus_tenure", {**_TUNED_ALL, "tenure_min": 5, "tenure_max": 15}),
    ("tuned_minus_cand", {**_TUNED_ALL, "candidate_size": 10}),
    ("tuned_minus_lambda", {**_TUNED_ALL, "lam_factor": 1.5, "adapt_every": 10}),
    ("tuned_all_lam0x3", {**_TUNED_ALL, "lam0_scale": 3}),
    ("default_lam0x3", {"lam0_scale": 3}),
]

# Parameters that are changed together count as ONE change.
_GROUP = {"tenure_min": "tenure", "tenure_max": "tenure", "lam_factor": "lambda_adapt", "adapt_every": "lambda_adapt"}


def n_changes(overrides: dict, defaults: dict) -> int:
    """Number of parameter groups whose value differs from the defaults. Time: O(#overrides)."""
    return len({_GROUP.get(k, k) for k, v in overrides.items() if defaults[k] != v})


def build_configs(algo: str, grid: dict[str, list[Factor]] | None = None) -> list[tuple[str, str, dict]]:
    """All configurations of one algorithm: the shared baseline first, then every
    NON-default value of every factor. Returns (param_name, param_value_label, overrides).
    grid defaults to the round-1 GRID.

    Time: O(total number of values).
    """
    grid = GRID if grid is None else grid
    configs = [(BASELINE[0], BASELINE[1], {})]
    for factor in grid[algo]:
        assert factor.default in factor.values, f"default of {factor.name} not in its values"
        for value in factor.values:
            if value != factor.default:
                configs.append((factor.name, value_label(value), factor.overrides(value)))
    return configs


def factor_by_name(algo: str, name: str, grid: dict[str, list[Factor]] | None = None) -> Factor:
    """Look up a factor of an algorithm by its name. Time: O(#factors)."""
    for factor in (GRID if grid is None else grid)[algo]:
        if factor.name == name:
            return factor
    raise KeyError(f"{algo} has no factor {name!r}")
