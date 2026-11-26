# Original code (CMP 3005 term project).
"""exact_small agrees with a slow brute force; stored optima pass the checker."""

import itertools
import json

import pytest

from conftest import ROOT
from cvrp.checker import check
from cvrp.exact_small import solve_exact
from cvrp.parser import parse_vrp


def brute_force(inst) -> float:
    """All permutations x all ways to cut them into consecutive routes. O(n! * 2^(n-1) * n)."""
    d, best = inst.dist, float("inf")
    for perm in itertools.permutations(range(1, inst.n + 1)):
        for cuts in range(1 << (inst.n - 1)):         # bit i set = cut after position i
            routes, current = [], [perm[0]]
            for i in range(1, inst.n):
                if cuts >> (i - 1) & 1:
                    routes.append(current)
                    current = []
                current.append(perm[i])
            routes.append(current)
            if len(routes) > inst.K or any(sum(inst.demand[v] for v in r) > inst.Q for r in routes):
                continue
            cost = sum(d[0, r[0]] + d[r[-1], 0] + sum(d[a, b] for a, b in zip(r, r[1:])) for r in routes)
            best = min(best, cost)
    return best


def test_exact_matches_brute_force_n7():
    inst = parse_vrp(ROOT / "data/tiny/tiny_n7.vrp")
    cost, routes = solve_exact(inst)
    assert cost == brute_force(inst)
    assert check(inst, routes, claimed_cost=cost).feasible


@pytest.mark.parametrize("n", [7, 8, 9])
def test_stored_optimum_is_valid(n):
    inst = parse_vrp(ROOT / "data/tiny" / f"tiny_n{n}.vrp")
    stored = json.loads((ROOT / "data/tiny" / f"tiny_n{n}.opt.json").read_text())
    assert check(inst, stored["routes"], claimed_cost=stored["cost"]).feasible
    assert stored["cost"] == solve_exact(inst)[0]
    assert len(stored["routes"]) >= 2
