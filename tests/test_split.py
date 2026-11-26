# Original code (CMP 3005 term project).
"""Split DP: hand example, brute force, optimality link, BKS, max_routes, feasibility."""

import itertools
import json
import random

import pytest

from conftest import ROOT
from cvrp.checker import check
from cvrp.instance import Instance
from cvrp.parser import load_bks_solution, load_instance, parse_vrp
from cvrp.savings import clarke_wright
from cvrp.split import _prefix_sums, arc_cost, split

# ---------------------------------------------------------------------------
# Hand example: 6 customers on the x-axis (y = 0), depot at x = 0, so every
# distance is |x_a - x_b| (an exact integer). Q = 10. Giant tour T = (1,2,3,4,5,6).
#
#   customer : 1   2   3   4   5   6
#   x        : 2   5  -3  -4   6   1
#   demand   : 4   3   5   2   4   3        total 21
#   d(0,v)   : 2   5   3   4   6   1
#
# Consecutive distances: d12=3  d23=8  d34=1  d45=10  d56=5
# dist_prefix (by position k): [0, 0, 3, 11, 12, 22, 27]
# load_prefix:                 [0, 4, 7, 12, 14, 18, 21]
#
# Arcs (i -> j) = route 0-t_{i+1}..t_j-0 with load <= 10 (loop i = j-1 down, break when load > 10):
#   j=1: (0,1) load 4  cost 2 + 0 + 2         = 4
#   j=2: (1,2) load 3  cost 5 + 0 + 5         = 10
#        (0,2) load 7  cost 2 + 3 + 5         = 10
#   j=3: (2,3) load 5  cost 3 + 0 + 3         = 6
#        (1,3) load 8  cost 5 + 8 + 3         = 16
#        (0,3) load 12 > 10 -> break
#   j=4: (3,4) load 2  cost 4 + 0 + 4         = 8
#        (2,4) load 7  cost 3 + 1 + 4         = 8
#        (1,4) load 10 cost 5 + 9 + 4         = 18
#        (0,4) load 14 -> break
#   j=5: (4,5) load 4  cost 6 + 0 + 6         = 12
#        (3,5) load 6  cost 4 + 10 + 6        = 20
#        (2,5) load 11 -> break
#   j=6: (5,6) load 3  cost 1 + 0 + 1         = 2
#        (4,6) load 7  cost 6 + 5 + 1         = 12
#        (3,6) load 9  cost 4 + 15 + 1        = 20
#        (2,6) load 14 -> break
# Feasible arcs examined: 1 + 2 + 2 + 3 + 2 + 3 = 13
#
# DP f[j] = min_i f[i] + c(i,j):
#   f0 = 0
#   f1 = f0 + 4                                  = 4   (pred 0)
#   f2 = min(f1+10=14, f0+10=10)                 = 10  (pred 0)
#   f3 = min(f2+6=16, f1+16=20)                  = 16  (pred 2)
#   f4 = min(f3+8=24, f2+8=18, f1+18=22)         = 18  (pred 2)
#   f5 = min(f4+12=30, f3+20=36)                 = 30  (pred 4)
#   f6 = min(f5+2=32, f4+12=30, f3+20=36)        = 30  (pred 4)
# Path 0 -> 2 -> 4 -> 6: routes [1,2], [3,4], [5,6], cost 10 + 8 + 12 = 30.
# With at most 2 routes: impossible (2 * Q = 20 < 21), so split returns None.
# ---------------------------------------------------------------------------

HAND_ARCS = {(0, 1): 4, (1, 2): 10, (0, 2): 10, (2, 3): 6, (1, 3): 16, (3, 4): 8, (2, 4): 8,
             (1, 4): 18, (4, 5): 12, (3, 5): 20, (5, 6): 2, (4, 6): 12, (3, 6): 20}


@pytest.fixture
def hand():
    xs = [0, 2, 5, -3, -4, 6, 1]
    return Instance.from_euc2d_rounded("hand6", [(x, 0) for x in xs], [0, 4, 3, 5, 2, 4, 3], Q=10, K=3)


def test_hand_arc_table(hand):
    tour = [1, 2, 3, 4, 5, 6]
    _, dist_prefix = _prefix_sums(hand, tour)
    assert dist_prefix == [0, 0, 3, 11, 12, 22, 27]
    for (i, j), c in HAND_ARCS.items():
        assert arc_cost(hand, tour, dist_prefix, i, j) == c


def test_hand_split(hand):
    res = split(hand, [1, 2, 3, 4, 5, 6])
    assert res.solution.routes == [[1, 2], [3, 4], [5, 6]]
    assert (res.cost, res.arcs_examined) == (30, 13)
    assert split(hand, [1, 2, 3, 4, 5, 6], max_routes=3).cost == 30
    assert split(hand, [1, 2, 3, 4, 5, 6], max_routes=2) is None


# --- independent slow reference: every way to cut the tour ------------------

def brute_force_cuts(inst, tour, max_routes=None) -> float:
    """Min cost over all 2^(n-1) cut patterns of a fixed tour (capacity- and K-feasible)."""
    d, n, best = inst.dist, len(tour), float("inf")
    for mask in range(1 << (n - 1)):              # bit k set = cut after position k+1
        routes, current = [], [tour[0]]
        for k in range(1, n):
            if mask >> (k - 1) & 1:
                routes.append(current)
                current = []
            current.append(tour[k])
        routes.append(current)
        if max_routes is not None and len(routes) > max_routes:
            continue
        if any(sum(inst.demand[v] for v in r) > inst.Q for r in routes):
            continue
        cost = sum(d[0, r[0]] + d[r[-1], 0] + sum(d[a, b] for a, b in zip(r, r[1:])) for r in routes)
        best = min(best, cost)
    return best


def test_matches_brute_force_cuts_tiny_n7():
    inst = parse_vrp(ROOT / "data/tiny/tiny_n7.vrp")
    rng = random.Random(1)
    for _ in range(30):
        tour = list(range(1, 8))
        rng.shuffle(tour)
        assert split(inst, tour).cost == brute_force_cuts(inst, tour)
        limited = split(inst, tour, max_routes=inst.K)
        expected = brute_force_cuts(inst, tour, max_routes=inst.K)
        assert (limited.cost if limited else float("inf")) == expected


# --- optimality link: some order + Split = the exact optimum ------------------

def _min_over_permutations(n: int) -> tuple[float, float]:
    inst = parse_vrp(ROOT / "data/tiny" / f"tiny_n{n}.vrp")
    opt = json.loads((ROOT / "data/tiny" / f"tiny_n{n}.opt.json").read_text())["cost"]
    best = float("inf")
    for perm in itertools.permutations(range(1, n + 1)):
        res = split(inst, list(perm), max_routes=inst.K)
        if res is not None and res.cost < best:
            best = res.cost
    return best, opt


@pytest.mark.parametrize("n", [7, 8])
def test_min_over_permutations_equals_optimum(n):
    best, opt = _min_over_permutations(n)
    assert best == opt


@pytest.mark.slow
def test_min_over_permutations_equals_optimum_n9():
    best, opt = _min_over_permutations(9)
    assert best == opt


# --- links with Clarke–Wright and the BKS -------------------------------------

@pytest.mark.parametrize("name", ["A-n32-k5", "A-n60-k9"])
def test_split_of_cw_tour_not_worse(name):
    inst = load_instance(ROOT / "data/cvrplib" / f"{name}.vrp")
    cw = clarke_wright(inst)
    tour = [v for r in cw.solution.routes for v in r]
    assert split(inst, tour).cost <= cw.cost
    assert cw.n_routes <= inst.K
    assert split(inst, tour, max_routes=inst.K).cost <= cw.cost


def test_split_of_bks_tour_equals_bks():
    inst, routes, stated = load_bks_solution(ROOT / "data/cvrplib/A-n32-k5.vrp")
    tour = [v for r in routes for v in r]
    res = split(inst, tour, max_routes=inst.K)
    assert res.cost == stated == inst.bks == 784
    assert check(inst, res.solution, claimed_cost=res.cost).feasible


# --- max_routes on A-n45-k6 ---------------------------------------------------

def _tours_a45():
    """BKS order (has a 6-route split), CW order (CW used 7 routes) and a random order."""
    inst, bks_routes, _ = load_bks_solution(ROOT / "data/cvrplib/A-n45-k6.vrp")
    bks_tour = [v for r in bks_routes for v in r]
    cw_tour = [v for r in clarke_wright(inst).solution.routes for v in r]
    rnd = list(range(1, inst.n + 1))
    random.Random(2).shuffle(rnd)
    return inst, [bks_tour, cw_tour, rnd]


def test_max_routes_a45():
    inst, tours = _tours_a45()
    for tour in tours:
        assert split(inst, tour, max_routes=1) is None          # 593 > Q = 100
        unlimited = split(inst, tour)
        other = [e for e in check(inst, unlimited.solution, unlimited.cost).errors
                 if not e.startswith("too many routes")]
        assert other == []
        limited = split(inst, tour, max_routes=inst.K)
        if limited is not None:
            assert check(inst, limited.solution, claimed_cost=limited.cost).feasible
            assert limited.cost >= unlimited.cost
    # The BKS order can be cut into <= K routes, and Split recovers the BKS exactly.
    limited_bks = split(inst, tours[0], max_routes=inst.K)
    assert limited_bks is not None and len(limited_bks.solution.routes) <= inst.K
    assert limited_bks.cost == inst.bks == 944
    # The CW order (7 routes) admits NO split into <= 6 routes: max_routes really constrains.
    assert split(inst, tours[1], max_routes=inst.K) is None
    assert len(split(inst, tours[1]).solution.routes) > inst.K


# --- random tours always give checker-valid solutions -------------------------

def test_random_tours_a32_feasible():
    inst = load_instance(ROOT / "data/cvrplib/A-n32-k5.vrp")
    rng = random.Random(3)
    for _ in range(50):
        tour = list(range(1, inst.n + 1))
        rng.shuffle(tour)
        res = split(inst, tour)
        report = check(inst, res.solution, claimed_cost=res.cost)
        assert [e for e in report.errors if not e.startswith("too many routes")] == []
        limited = split(inst, tour, max_routes=inst.K)
        if limited is not None:
            assert check(inst, limited.solution, claimed_cost=limited.cost).feasible


def test_invalid_tour_rejected(hand):
    with pytest.raises(ValueError):
        split(hand, [1, 2, 3, 4, 5, 5])
    with pytest.raises(ValueError):
        split(hand, [1, 2, 3, 4, 5])
