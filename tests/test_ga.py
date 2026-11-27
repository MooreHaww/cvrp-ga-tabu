# Original code (CMP 3005 term project).
"""Genetic Algorithm: operators, decoder, determinism, optimum on tiny, feasibility, diversity, memetic."""

import dataclasses
import json
import random

import pytest

from conftest import ROOT
from cvrp.checker import check
from cvrp.ga import (GAParams, decode, default_penalty, genetic_algorithm, insertion_mutation,
                     inversion_mutation, order_crossover, swap_mutation)
from cvrp.generator import generate_team_instance
from cvrp.local_search import two_opt_route
from cvrp.parser import load_bks_solution, load_instance, parse_vrp
from cvrp.savings import clarke_wright
from cvrp.solution import route_cost


@pytest.fixture(scope="module")
def a32():
    return load_instance(ROOT / "data/cvrplib/A-n32-k5.vrp")


# --- operators -------------------------------------------------------------------

def test_operators_return_permutations():
    rng = random.Random(0)
    base = list(range(1, 32))                      # n = 31
    for _ in range(1000):
        p1, p2 = base[:], base[:]
        rng.shuffle(p1)
        rng.shuffle(p2)
        assert sorted(order_crossover(p1, p2, rng)) == base
        for op in (swap_mutation, inversion_mutation, insertion_mutation):
            child = op(p1, rng)
            assert sorted(child) == base and child != p1
        assert sorted(p1) == base                  # operators do not modify their input


def test_ox_keeps_segment_from_parent1():
    rng = random.Random(5)
    p1, p2 = list(range(1, 11)), list(range(10, 0, -1))
    for _ in range(100):
        child = order_crossover(p1, p2, rng)
        # every position that equals p1 forms one contiguous block containing the cut segment
        same = [i for i in range(10) if child[i] == p1[i]]
        assert same and sorted(child) == p1


# --- decoder ---------------------------------------------------------------------

def test_decode_cost_matches_checker(a32):
    rng = random.Random(1)
    for _ in range(20):
        tour = list(range(1, a32.n + 1))
        rng.shuffle(tour)
        ind, arcs = decode(a32, tour, penalty=50.0)
        report = check(a32, ind.routes, claimed_cost=ind.cost)
        assert arcs > 0
        assert ind.fitness == ind.cost + 50.0 * ind.excess
        if ind.excess == 0:
            assert report.feasible                  # including the K rule
        else:
            assert len(ind.routes) == a32.K + ind.excess


def test_decode_stages_on_a45():
    inst, bks_routes, _ = load_bks_solution(ROOT / "data/cvrplib/A-n45-k6.vrp")
    cw_tour = [v for r in clarke_wright(inst, warn=False).solution.routes for v in r]
    bad, _ = decode(inst, cw_tour, penalty=100.0)   # no split into <= 6 routes exists (Step 3)
    assert bad.excess == 1 and len(bad.routes) == 7 and bad.fitness == bad.cost + 100.0
    good, _ = decode(inst, [v for r in bks_routes for v in r], penalty=100.0)
    assert good.excess == 0 and good.cost == good.fitness == 944
    assert check(inst, good.routes, claimed_cost=good.cost).feasible


# --- penalty ---------------------------------------------------------------------

def test_default_penalty_scales_with_instance(a32):
    team = generate_team_instance(seed=0)
    lam_a32, lam_team = default_penalty(a32), default_penalty(team)
    assert lam_a32 == pytest.approx(2 * a32.dist[0, 1:].mean())
    assert lam_team == pytest.approx(2 * team.dist[0, 1:].mean())
    assert lam_team > 5 * lam_a32                   # 1000x1000 square + Bosphorus vs 100x100 square
    res = genetic_algorithm(a32, GAParams(seed=0, max_generations=1))
    assert res.penalty_lambda == pytest.approx(lam_a32)
    res = genetic_algorithm(a32, GAParams(seed=0, max_generations=1, penalty=7.0))
    assert res.penalty_lambda == 7.0                # an explicit value overrides the default


def test_infeasible_never_returned(a32):
    # K = 4 is impossible on A-n32-k5 (total demand 410 > 4 * 100): every individual is infeasible.
    impossible = dataclasses.replace(a32, K=4)
    res = genetic_algorithm(impossible, GAParams(seed=0, max_generations=5))
    assert res.solution is None and res.best_cost is None and res.n_routes is None
    assert res.progress == []
    assert res.best_infeasible_fitness is not None and res.best_infeasible_fitness[0] >= 1


# --- whole runs ------------------------------------------------------------------

def test_deterministic(a32):
    a = genetic_algorithm(a32, GAParams(seed=7, max_generations=20))
    b = genetic_algorithm(a32, GAParams(seed=7, max_generations=20))
    assert a.best_cost == b.best_cost and a.solution.routes == b.solution.routes
    assert (a.decodes, a.generations, a.splits_arcs_examined) == (b.decodes, b.generations, b.splits_arcs_examined)
    assert [c for _, c in a.progress] == [c for _, c in b.progress]
    assert a.final_population == b.final_population


@pytest.mark.parametrize("n", [7, 8])
@pytest.mark.parametrize("seed_fraction", [0.1, 0.0], ids=["cw-seeded", "random-only"])
def test_reaches_optimum_on_tiny(n, seed_fraction):
    """CW is already optimal on the tiny instances, so the random-only case is the real test."""
    inst = parse_vrp(ROOT / "data/tiny" / f"tiny_n{n}.vrp")
    opt = json.loads((ROOT / "data/tiny" / f"tiny_n{n}.opt.json").read_text())["cost"]
    hits = 0
    for seed in range(5):
        res = genetic_algorithm(inst, GAParams(seed=seed, max_generations=200, seed_fraction=seed_fraction))
        assert res.best_cost >= opt
        hits += res.best_cost == opt
    assert hits >= 4


def test_a32_three_seconds(a32):
    res = genetic_algorithm(a32, GAParams(seed=0, time_limit=3.0))
    report = check(a32, res.solution, claimed_cost=res.best_cost)
    assert report.feasible
    assert res.best_cost <= clarke_wright(a32).cost
    costs = [c for _, c in res.progress]
    assert costs == sorted(costs, reverse=True) and costs[-1] == res.best_cost
    times = [t for t, _ in res.progress]
    assert times == sorted(times)
    assert len(res.gen_times) == res.generations > 0


def test_final_population_is_diverse(a32):
    res = genetic_algorithm(a32, GAParams(seed=3, max_generations=20))
    fit = res.final_population                      # sorted (excess, cost) pairs
    assert len(fit) == 50
    for (e1, c1), (e2, c2) in zip(fit, fit[1:]):    # sorted, so neighbours are the closest pairs
        assert e1 != e2 or abs(c1 - c2) > 1e-6
    assert res.immigrant_duplicates == 0


def test_memetic_run(a32):
    res = genetic_algorithm(a32, GAParams(seed=0, max_generations=30, memetic=True))
    assert check(a32, res.solution, claimed_cost=res.best_cost).feasible


def test_cw_seeding_survives_diversity_rule(a32):
    """CW variants get one mutation, so they are not all rejected as duplicates (no immigrants needed)."""
    res = genetic_algorithm(a32, GAParams(seed=0, max_generations=1))
    assert res.immigrants == 0


def test_two_opt_never_worse(a32):
    rng = random.Random(2)
    for _ in range(50):
        route = rng.sample(range(1, a32.n + 1), 8)
        better = two_opt_route(a32, route)
        assert sorted(better) == sorted(route)
        assert route_cost(a32, better) <= route_cost(a32, route)


def test_params_need_a_stopping_rule():
    with pytest.raises(ValueError):
        GAParams(seed=0)


# --- Step 4c: memetic GA with inter-route descent (ls_mode="full") ---------------

def test_ls_mode_validation():
    with pytest.raises(ValueError):
        GAParams(seed=0, max_generations=1, ls_mode="or_opt")
    with pytest.raises(ValueError):
        GAParams(seed=0, max_generations=1, ls_neighbours=0)
    assert GAParams(seed=0, max_generations=1).ls_mode == "2opt"      # default = old memetic behaviour


def test_plain_ga_reports_no_local_search(a32):
    res = genetic_algorithm(a32, GAParams(seed=0, max_generations=5))
    assert res.local_searches == 0 and res.ls_time == 0.0


def test_memetic_full_deterministic(a32):
    p = dict(seed=4, max_generations=5, memetic=True, ls_mode="full", local_search_prob=0.5)
    a, b = genetic_algorithm(a32, GAParams(**p)), genetic_algorithm(a32, GAParams(**p))
    assert a.best_cost == b.best_cost and a.solution.routes == b.solution.routes
    assert a.decodes == b.decodes and a.local_searches == b.local_searches > 0


def test_memetic_full_beats_cw_on_a60():
    """Regression test for the Step 6b finding: the plain GA never got below the
    Clarke–Wright cost (1408) on A-n60-k9. 25 generations take about 5 s."""
    inst = load_instance(ROOT / "data/cvrplib/A-n60-k9.vrp")
    cw = clarke_wright(inst, warn=False).cost
    res = genetic_algorithm(inst, GAParams(seed=0, max_generations=25, memetic=True, ls_mode="full",
                                           local_search_prob=0.5))
    assert check(inst, res.solution, claimed_cost=res.best_cost).feasible
    assert res.best_cost <= 0.99 * cw
    assert res.local_searches > 0 and 0 < res.ls_time <= res.elapsed
