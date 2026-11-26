# Original code
"""Step 6b: the one-factor-at-a-time (OFAT) parameter grid.

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


def build_configs(algo: str) -> list[tuple[str, str, dict]]:
    """All configurations of one algorithm: the shared baseline first, then every
    NON-default value of every factor. Returns (param_name, param_value_label, overrides).

    Time: O(total number of values).
    """
    configs = [(BASELINE[0], BASELINE[1], {})]
    for factor in GRID[algo]:
        assert factor.default in factor.values, f"default of {factor.name} not in its values"
        for value in factor.values:
            if value != factor.default:
                configs.append((factor.name, value_label(value), factor.overrides(value)))
    return configs


def factor_by_name(algo: str, name: str) -> Factor:
    """Look up a factor of an algorithm by its name. Time: O(#factors)."""
    for factor in GRID[algo]:
        if factor.name == name:
            return factor
    raise KeyError(f"{algo} has no factor {name!r}")
