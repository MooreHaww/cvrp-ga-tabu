# Original code (CMP 3005 term project).
"""Tabu Search: O(1) deltas vs full recomputation, state consistency, mechanics, whole runs."""

import random

import pytest

from conftest import ROOT
from cvrp.checker import check
from cvrp.generator import generate_team_instance
from cvrp.local_search import (apply_relocate, apply_swap, apply_two_opt_intra, apply_two_opt_star,
                               full_distance, relocate_delta, solution_overload, swap_delta,
                               two_opt_intra_delta, two_opt_star_delta)
from cvrp.parser import load_instance
from cvrp.savings import clarke_wright
from cvrp.tabu import (LambdaController, TabuList, TabuParams, _State, initial_routes, is_admissible,
                       tabu_search)


@pytest.fixture(scope="module")
def a32():
    return load_instance(ROOT / "data/cvrplib/A-n32-k5.vrp")


@pytest.fixture(scope="module")
def team():
    return generate_team_instance(seed=0)


def _test_routes(inst):
    """CW routes with the two first routes merged (=> overload) plus two empty routes."""
    routes = [list(r) for r in clarke_wright(inst, warn=False).solution.routes]
    routes = [routes[0] + routes[1]] + routes[2:] + [[], []]
    return routes


def _random_move(kind, routes, rng):
    """Random valid arguments for a move type; edge positions occur naturally (and adjacent swaps are forced
    in a quarter of the swap trials)."""
    non_empty = [r for r, route in enumerate(routes) if route]
    if kind == "relocate":
        r1 = rng.choice(non_empty)
        i = rng.randrange(len(routes[r1]))
        r2 = rng.randrange(len(routes))
        j = rng.randrange(len(routes[r2]) + (0 if r2 == r1 else 1))
        return r1, i, r2, j
    if kind == "swap":
        if rng.random() < 0.25:                                     # adjacent, same route
            r = rng.choice([r for r in non_empty if len(routes[r]) >= 2])
            i = rng.randrange(len(routes[r]) - 1)
            return (r, i, r, i + 1) if rng.random() < 0.5 else (r, i + 1, r, i)
        r1, r2 = rng.choice(non_empty), rng.choice(non_empty)
        i, j = rng.randrange(len(routes[r1])), rng.randrange(len(routes[r2]))
        if r1 == r2 and i == j:
            return None
        return r1, i, r2, j
    if kind == "two_opt_star":
        r1, r2 = rng.sample(range(len(routes)), 2)
        return r1, rng.randint(0, len(routes[r1])), r2, rng.randint(0, len(routes[r2]))
    r = rng.choice([r for r in non_empty if len(routes[r]) >= 2])
    a, b = sorted(rng.sample(range(len(routes[r])), 2))
    return r, a, b


APPLY = {"relocate": apply_relocate, "swap": apply_swap,
         "two_opt_star": apply_two_opt_star, "two_opt": apply_two_opt_intra}


@pytest.mark.parametrize("which", ["a32", "team"])
@pytest.mark.parametrize("kind", ["relocate", "swap", "two_opt_star", "two_opt"])
def test_delta_equals_full_recomputation(which, kind, request):
    inst = request.getfixturevalue(which)
    state = _State(inst, _test_routes(inst))
    D, q, Q = state.D, state.q, state.Q
    rng = random.Random(42)
    checked = overload_changes = 0
    while checked < 1000:
        args = _random_move(kind, state.routes, rng)
        if args is None:
            continue
        routes = state.routes
        if kind == "relocate":
            dd, dover = relocate_delta(D, q, Q, routes, state.loads, *args)
        elif kind == "swap":
            dd, dover = swap_delta(D, q, Q, routes, state.loads, *args)
        elif kind == "two_opt_star":
            dd, dover = two_opt_star_delta(D, Q, routes, state.loads, state.prefix, *args)
        else:
            dd, dover = two_opt_intra_delta(D, routes, *args)
        new = APPLY[kind](routes, *args)
        assert dd == pytest.approx(full_distance(D, new) - full_distance(D, routes), abs=1e-9)
        assert dover == solution_overload(q, Q, new) - solution_overload(q, Q, routes)
        overload_changes += dover != 0
        state.apply(kind, args, dd, dover)        # random walk: the next move starts from here
        checked += 1
    if kind != "two_opt":                         # intra-route moves never change loads
        assert overload_changes > 0


def test_partition_kept_after_many_moves(a32):
    state = _State(a32, _test_routes(a32))
    rng = random.Random(7)
    kinds = ["relocate", "swap", "two_opt_star", "two_opt"]
    for _ in range(2000):
        kind = rng.choice(kinds)
        args = _random_move(kind, state.routes, rng)
        if args is None:
            continue
        dd, dover = state.delta(kind, args)
        state.apply(kind, args, dd, dover)
    customers = sorted(v for r in state.routes for v in r)
    assert customers == list(range(1, a32.n + 1))
    for r, route in enumerate(state.routes):
        for k, c in enumerate(route):
            assert state.where[c] == (r, k)
        assert state.loads[r] == sum(state.q[c] for c in route)
    assert state.dist == pytest.approx(full_distance(state.D, state.routes), abs=1e-6)
    assert state.over == solution_overload(state.q, state.Q, state.routes)


def test_initial_routes_have_k_slots():
    inst = load_instance(ROOT / "data/cvrplib/A-n45-k6.vrp")    # CW gives 7 routes, K = 6
    routes, cw_cost = initial_routes(inst)
    assert len(routes) == inst.K
    assert sorted(v for r in routes for v in r) == list(range(1, inst.n + 1))
    team = generate_team_instance(seed=0)
    assert len(initial_routes(team)[0]) == team.K


def test_verify_deltas_run(a32):
    res = tabu_search(a32, TabuParams(seed=0, max_iterations=500, verify_deltas=True))
    assert res.iterations == 500


def test_a32_three_seconds(a32):
    res = tabu_search(a32, TabuParams(seed=0, time_limit=3.0))
    assert check(a32, res.solution, claimed_cost=res.cost).feasible
    assert res.cost <= clarke_wright(a32).cost
    costs = [c for _, c in res.progress]
    assert costs == sorted(costs, reverse=True) and costs[-1] == res.cost


def test_a45_becomes_feasible_with_k_routes():
    inst = load_instance(ROOT / "data/cvrplib/A-n45-k6.vrp")
    res = tabu_search(inst, TabuParams(seed=0, time_limit=5.0))
    assert res.solution is not None and res.n_routes <= inst.K == 6
    assert check(inst, res.solution, claimed_cost=res.cost).feasible


def test_a80_overload_trap_fixed(monkeypatch):
    """Regression test for the overload trap (docs/step5_notes.md, 'Fix').

    Before the fix, TS on A-n80-k10 stayed at overload 1 forever (no repair move in the
    granular list), applied null moves that change nothing, and never beat Clarke-Wright.
    Deterministic: max_iterations instead of a time limit.
    """
    import cvrp.tabu as tb
    inst = load_instance(ROOT / "data/cvrplib/A-n80-k10.vrp")
    removed_per_move = []
    original_apply = tb._State.apply

    def recording_apply(self, kind, args, dd, dover):
        removed = original_apply(self, kind, args, dd, dover)
        removed_per_move.append(len(removed))
        return removed

    monkeypatch.setattr(tb._State, "apply", recording_apply)
    res = tabu_search(inst, TabuParams(seed=0, max_iterations=1000))
    cw = clarke_wright(inst, warn=False).cost
    assert check(inst, res.solution, claimed_cost=res.cost).feasible
    assert res.cost < cw                                    # 1827 < 1840 (stuck at 1840 before the fix)
    assert res.final_lambda < TabuParams(seed=0, max_iterations=1).lam_max
    assert min(removed_per_move) > 0                        # every applied move really changed the solution


def test_null_moves_detected(a32):
    """Swapping the two customers of a 2-customer route only reverses it: a null move."""
    routes = [[1, 2], [3, 4, 5]] + [[c] for c in range(6, a32.n + 1)]
    state = _State(a32, routes)
    swap = state.added_arcs("swap", (0, 0, 0, 1))           # [1,2] -> [2,1]
    assert state.is_null_move(swap)
    whole = state.added_arcs("two_opt", (1, 0, 2))          # reverse all of [3,4,5]
    assert state.is_null_move(whole)
    real = state.added_arcs("relocate", (1, 0, 0, 1))       # 3 between 1 and 2
    assert not state.is_null_move(real)
    assert state.has_arc((0, 1)) and state.has_arc((1, 2)) and state.has_arc((0, 2))
    assert not state.has_arc((1, 3)) and not state.has_arc((0, 4))


def test_repair_candidates_reach_all_routes(a32):
    """Repair moves cover every position of every other route, not only nearest neighbours."""
    routes = _test_routes(a32)                              # route 0 is overloaded (two CW routes merged)
    state = _State(a32, routes)
    assert state.loads[0] > a32.Q
    repairs = list(state.repair_candidates({"relocate", "swap"}))
    reloc_targets = {args[2] for kind, args in repairs if kind == "relocate"}
    assert reloc_targets == set(range(1, len(routes)))      # every other route, empty ones included
    assert all(args[0] == 0 for _, args in repairs)         # only customers of overloaded routes move
    n_reloc = sum(1 for k, _ in repairs if k == "relocate")
    assert n_reloc == len(routes[0]) * sum(len(r) + 1 for r in routes[1:])


def test_deterministic(a32):
    a = tabu_search(a32, TabuParams(seed=3, max_iterations=200))
    b = tabu_search(a32, TabuParams(seed=3, max_iterations=200))
    assert a.solution.routes == b.solution.routes
    assert (a.cost, a.iterations, a.moves_evaluated, a.final_lambda, a.aspirations_used) == \
           (b.cost, b.iterations, b.moves_evaluated, b.final_lambda, b.aspirations_used)


# --- tabu mechanics -------------------------------------------------------------

def test_tabu_list_forbids_then_frees():
    tl = TabuList(3, 3)                       # fixed tenure 3 for the test
    rng = random.Random(0)
    tl.forbid({(1, 2)}, iteration=10, rng=rng)
    assert all(tl.is_tabu([(1, 2)], it) for it in (11, 12, 13))
    assert not tl.is_tabu([(1, 2)], 14)
    assert not tl.is_tabu([(2, 3)], 11)       # other arcs unaffected


def test_aspiration_rule():
    assert is_admissible(False, 5, 1000.0, 900.0)          # not tabu: always admissible
    assert not is_admissible(True, 0, 950.0, 900.0)        # tabu, feasible but not better
    assert not is_admissible(True, 3, 800.0, 900.0)        # tabu, better but infeasible
    assert is_admissible(True, 0, 899.0, 900.0)            # tabu, feasible and new best -> aspiration


def test_tenure_range():
    tl = TabuList(5, 15)
    rng = random.Random(1)
    draws = [tl.draw_tenure(rng) for _ in range(1000)]
    assert min(draws) == 5 and max(draws) == 15


# --- adaptive lambda -----------------------------------------------------------

def test_lambda_adapts():
    ctrl = LambdaController(2.0, 1.5, 10, 0.01, 100.0)
    changed = [ctrl.record(False) for _ in range(10)]
    assert changed == [False] * 9 + [True] and ctrl.lam == pytest.approx(3.0)
    for _ in range(10):
        ctrl.record(True)
    assert ctrl.lam == pytest.approx(2.0)
    for k in range(10):
        ctrl.record(k % 2 == 0)                             # mixed window: unchanged
    assert ctrl.lam == pytest.approx(2.0)
    for _ in range(200):
        ctrl.record(False)                                  # clamped at lam_max
    assert ctrl.lam == 100.0


def test_params_validation():
    with pytest.raises(ValueError):
        TabuParams(seed=0)
    with pytest.raises(ValueError):
        TabuParams(seed=0, max_iterations=1, moves=("relocate", "teleport"))


# --- Step 5c / 6b-2: initial lambda scale -------------------------------------------

def test_lam0_scale_multiplies_initial_lambda():
    """With max_iterations=0 no iteration runs, so final_lambda is the starting lambda."""
    inst = load_instance(ROOT / "data/team/team_n100_seed14835.vrp")
    base = tabu_search(inst, TabuParams(seed=0, max_iterations=0)).final_lambda
    cw = clarke_wright(inst, warn=False).cost
    assert base == pytest.approx(cw / inst.total_demand())         # 11.7 on this instance
    three = tabu_search(inst, TabuParams(seed=0, max_iterations=0, lam0_scale=3)).final_lambda
    assert three == pytest.approx(3 * base)


@pytest.mark.parametrize("bad", [0, -1.0])
def test_lam0_scale_must_be_positive(bad):
    with pytest.raises(ValueError):
        TabuParams(seed=0, max_iterations=1, lam0_scale=bad)
