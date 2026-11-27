# Original code (CMP 3005 term project).
"""Step 4c: granular first-improvement descent (local_search.descent)."""

import random

import pytest

from conftest import ROOT
from cvrp.checker import check
from cvrp.local_search import (EPS, descent, full_distance, neighbours, node, relocate_delta, swap_delta,
                               two_opt_intra_delta, two_opt_star_delta)
from cvrp.parser import load_instance
from cvrp.savings import clarke_wright
from cvrp.solution import Solution
from cvrp.split import split

A32 = ROOT / "data/cvrplib/A-n32-k5.vrp"
TEAM = ROOT / "data/team/team_n100_seed14835.vrp"


@pytest.fixture(scope="module", params=[A32, TEAM], ids=["A-n32-k5", "team"])
def setup(request):
    """(instance, D, neighbour lists, Clarke–Wright routes)."""
    inst = load_instance(request.param)
    D = inst.dist.tolist()
    return inst, D, neighbours(D, inst.n, 10), clarke_wright(inst, warn=False).solution.routes


def random_split_routes(inst, seed: int) -> list[list[int]]:
    """Routes of the K-limited Split of a seeded random giant tour (retry until one exists)."""
    rng = random.Random(seed)
    for _ in range(200):
        tour = list(range(1, inst.n + 1))
        rng.shuffle(tour)
        res = split(inst, tour, max_routes=inst.K)
        if res is not None:
            return res.solution.routes
    raise AssertionError("no random tour with a <= K split")


def loads_of(inst, routes):
    return [sum(int(inst.demand[v]) for v in r) for r in routes]


def assert_invariants(inst, D, before, after):
    """Distance not higher, loads <= Q, every customer once, no more routes."""
    assert full_distance(D, after) <= full_distance(D, before) + 1e-9
    assert all(load <= inst.Q for load in loads_of(inst, after))
    assert sorted(v for r in after for v in r) == list(range(1, inst.n + 1))
    assert all(after) and len(after) <= len([r for r in before if r])


# --------------------------------------------------------------------------- independent oracle

def improving_granular_moves(inst, D, neigh, routes) -> list:
    """Brute force over ALL relocate / swap / 2-opt* / 2-opt moves; keep the granular
    ones (the move creates an arc between a customer and one of its neighbours, as
    defined in local_search.customer_moves) that improve the distance by more than
    1e-9 and keep every load <= Q. Written without customer_moves on purpose."""
    q = [int(x) for x in inst.demand]
    Q = inst.Q
    loads = loads_of(inst, routes)
    prefix = []
    for r in routes:
        pre = [0]
        for v in r:
            pre.append(pre[-1] + q[v])
        prefix.append(pre)
    N = [set(x) for x in neigh]
    found = []

    def keep(kind, args, dd, dover):
        if dd < -EPS and dover == 0:
            found.append((kind, args, dd))

    for r1, route1 in enumerate(routes):
        for i, c in enumerate(route1):
            # relocate c to position j of r2: new neighbours a, b of c
            for r2, route2 in enumerate(routes):
                target = route2 if r2 != r1 else route1[:i] + route1[i + 1:]
                for j in range(len(target) + 1):
                    if r2 == r1 and j == i:
                        continue                               # puts c back where it was
                    a, b = node(target, j - 1), node(target, j)
                    if a in N[c] or b in N[c]:
                        keep("relocate", (r1, i, r2, j), *relocate_delta(D, q, Q, routes, loads, r1, i, r2, j))
                # swap c with c2
                for j, c2 in enumerate(route2):
                    if (r2, j) != (r1, i) and (c2 in N[c] or c in N[c2]):
                        keep("swap", (r1, i, r2, j), *swap_delta(D, q, Q, routes, loads, r1, i, r2, j))
        # 2-opt*: new arc (u1, v2) = (route1[i-1], route2[j]), both customers, v2 in N(u1)
        for r2, route2 in enumerate(routes):
            if r2 == r1:
                continue
            for i in range(1, len(route1) + 1):
                for j in range(len(route2)):
                    if route2[j] in N[route1[i - 1]]:
                        keep("two_opt_star", (r1, i, r2, j),
                             *two_opt_star_delta(D, Q, routes, loads, prefix, r1, i, r2, j))
        # intra 2-opt reversing [a..b]: new arc (p, x_b), p = route1[a-1] a customer
        for a in range(1, len(route1)):
            for b in range(a + 1, len(route1)):
                p, xb = route1[a - 1], route1[b]
                if xb in N[p] or p in N[xb]:
                    keep("two_opt", (r1, a, b), *two_opt_intra_delta(D, routes, r1, a, b))
    return found


# --------------------------------------------------------------------------- tests

def test_invariants_from_cw_and_random_split(setup):
    inst, D, neigh, cw_routes = setup
    for start in (cw_routes, random_split_routes(inst, 1)):
        after = descent(inst, start, neigh, D=D)
        assert_invariants(inst, D, start, after)


def test_reaches_granular_local_optimum(setup):
    inst, D, neigh, cw_routes = setup
    for start in (cw_routes, random_split_routes(inst, 2)):
        stats = {}
        after = descent(inst, start, neigh, D=D, stats=stats)
        assert stats["moves_applied"] > 0                      # it did something
        assert improving_granular_moves(inst, D, neigh, after) == []


def test_oracle_finds_moves_before_descent(setup):
    """Sanity check of the oracle itself: a random split is far from a local optimum."""
    inst, D, neigh, _ = setup
    assert improving_granular_moves(inst, D, neigh, random_split_routes(inst, 3))


@pytest.mark.parametrize("start", ["cw", "random"])
def test_result_passes_checker_on_a32(start):
    inst = load_instance(A32)
    D = inst.dist.tolist()
    neigh = neighbours(D, inst.n, 10)
    routes = (clarke_wright(inst, warn=False).solution.routes if start == "cw"
              else random_split_routes(inst, 4))
    after = descent(inst, routes, neigh)
    report = check(inst, Solution(after), claimed_cost=full_distance(D, after))
    assert report.feasible, report.errors


def test_deterministic(setup):
    inst, D, neigh, _ = setup
    start = random_split_routes(inst, 5)
    assert descent(inst, start, neigh, D=D) == descent(inst, start, neigh, D=D)
    a = descent(inst, start, neigh, D=D, rng=random.Random(7))
    b = descent(inst, start, neigh, D=D, rng=random.Random(7))
    assert a == b
    assert_invariants(inst, D, start, a)


def test_does_not_modify_input_and_respects_max_passes(setup):
    inst, D, neigh, _ = setup
    start = random_split_routes(inst, 6)
    copy = [list(r) for r in start]
    stats = {}
    after = descent(inst, start, neigh, D=D, max_passes=1, stats=stats)
    assert start == copy and stats["passes"] == 1
    assert_invariants(inst, D, start, after)


def test_rejects_overloaded_input_and_bad_moves():
    inst = load_instance(A32)
    neigh = neighbours(inst.dist.tolist(), inst.n, 10)
    with pytest.raises(ValueError):
        descent(inst, [list(range(1, inst.n + 1))], neigh)     # one route with all demand
    with pytest.raises(ValueError):
        descent(inst, clarke_wright(inst, warn=False).solution.routes, neigh, moves=("or_opt",))
