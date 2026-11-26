# Original code (CMP 3005 term project).
"""Clarke–Wright: hand-checked example, feasibility, never below optimum, determinism."""

import json

import pytest

from conftest import AUGERAT, ROOT
from cvrp.checker import check
from cvrp.generator import generate_team_instance
from cvrp.instance import Instance
from cvrp.parser import load_instance, parse_vrp
from cvrp.savings import clarke_wright, compute_savings

# ---------------------------------------------------------------------------
# Hand-made instance (TSPLIB-rounded Euclidean distances, Q = 10, K = 3)
#
#   vertex : 0      1      2      3      4       5
#   (x, y) : (0,0)  (6,0)  (6,8)  (0,8)  (-6,0)  (-6,-8)
#   demand : 0      4      3      3      4       5        total 19 > Q -> >= 2 routes
#
# Distances from the depot: d01=6, d02=10, d03=8, d04=6, d05=10
# Other pairs:  d12=8   d13=10  d14=12  d15=sqrt(208)=14.42->14
#               d23=6   d24=sqrt(208)->14  d25=sqrt(400)=20
#               d34=10  d35=sqrt(292)=17.09->17   d45=8
#
# Savings s(i,j) = d0i + d0j - dij:
#   s12=6+10-8=8   s13=6+8-10=4   s14=6+6-12=0 (dropped)   s15=6+10-14=2
#   s23=10+8-6=12  s24=10+6-14=2  s25=10+10-20=0 (dropped)
#   s34=8+6-10=4   s35=8+10-17=1  s45=6+10-8=8
#
# Sorted (s desc, i asc, j asc):
#   (12,2,3) (8,1,2) (8,4,5) (4,1,3) (4,3,4) (2,1,5) (2,2,4) (1,3,5)
#
# Merge loop:
#   (2,3): [2] + [3]          -> [2,3]    load 6    MERGE
#   (1,2): [1] + [2,3]        -> [1,2,3]  load 10   MERGE (10 <= Q)
#   (4,5): [4] + [5]          -> [4,5]    load 9    MERGE
#   (1,3): same route                               skip
#   (3,4): load 10 + 9 = 19 > 10                    reject (capacity)
#   (1,5): load 19 > 10                             reject (capacity)
#   (2,4): 2 is interior in [1,2,3]                 reject (endpoint rule)
#   (3,5): load 19 > 10                             reject (capacity)
#
# Result: routes [1,2,3] and [4,5]; 8 pairs examined, 3 merges.
# Cost: (6 + 8 + 6 + 8) + (6 + 8 + 10) = 28 + 24 = 52.
# ---------------------------------------------------------------------------


@pytest.fixture
def hand():
    coords = [(0, 0), (6, 0), (6, 8), (0, 8), (-6, 0), (-6, -8)]
    return Instance.from_euc2d_rounded("hand5", coords, [0, 4, 3, 3, 4, 5], Q=10, K=3)


def test_hand_savings_order(hand):
    assert compute_savings(hand) == [(12, 2, 3), (8, 1, 2), (8, 4, 5), (4, 1, 3),
                                     (4, 3, 4), (2, 1, 5), (2, 2, 4), (1, 3, 5)]


def test_hand_merges(hand):
    res = clarke_wright(hand)
    routes = [r if r[0] < r[-1] else r[::-1] for r in res.solution.routes]  # normalise direction
    assert sorted(routes) == [[1, 2, 3], [4, 5]]
    assert (res.cost, res.n_routes, res.iterations, res.merges) == (52, 2, 8, 3)
    assert check(hand, res.solution, claimed_cost=52).feasible


def _instances():
    for name in AUGERAT:
        yield load_instance(ROOT / "data/cvrplib" / f"{name}.vrp")
    yield generate_team_instance(seed=0)


@pytest.mark.parametrize("inst", list(_instances()), ids=lambda i: i.name)
def test_feasible_capacity_and_customers(inst):
    res = clarke_wright(inst)
    report = check(inst, res.solution, claimed_cost=res.cost)
    # Clarke-Wright does not enforce K, so only a "too many routes" error is tolerated here.
    other = [e for e in report.errors if not e.startswith("too many routes")]
    assert other == []
    assert res.n_routes == report.n_routes == len(res.solution.non_empty_routes())
    assert res.merges == inst.n - res.n_routes          # each merge removes exactly one route
    assert res.progress and res.progress[-1][1] == res.cost


@pytest.mark.parametrize("n", [7, 8, 9])
def test_never_better_than_optimum(n):
    inst = parse_vrp(ROOT / "data/tiny" / f"tiny_n{n}.vrp")
    opt = json.loads((ROOT / "data/tiny" / f"tiny_n{n}.opt.json").read_text())["cost"]
    res = clarke_wright(inst)
    assert check(inst, res.solution).feasible
    assert res.cost >= opt


def test_deterministic():
    inst = load_instance(ROOT / "data/cvrplib/A-n45-k6.vrp")
    a, b = clarke_wright(inst), clarke_wright(inst)
    assert a.solution.routes == b.solution.routes
    assert (a.cost, a.iterations, a.merges) == (b.cost, b.iterations, b.merges)
