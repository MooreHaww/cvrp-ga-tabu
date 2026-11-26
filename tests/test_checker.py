# Original code (CMP 3005 term project).
"""check() accepts the BKS and catches every kind of violation."""

import copy

import pytest

from conftest import ROOT
from cvrp.checker import check
from cvrp.parser import load_bks_solution
from cvrp.solution import Solution, total_cost


@pytest.fixture
def bks():
    """(instance, BKS routes, stated cost) for A-n32-k5."""
    return load_bks_solution(ROOT / "data/cvrplib/A-n32-k5.vrp")


def _errors_with(inst, routes, needle, claimed=None):
    report = check(inst, routes, claimed_cost=claimed)
    assert not report.feasible
    assert any(needle in e for e in report.errors), report.errors


def test_bks_ok_and_matches_solution_module(bks):
    inst, routes, cost = bks
    report = check(inst, Solution(routes), claimed_cost=cost)
    assert report.feasible and report.n_routes == 5 and sum(report.loads) == 410
    assert total_cost(inst, routes) == report.cost


def test_duplicate(bks):
    inst, routes, _ = bks
    r = copy.deepcopy(routes)
    r[0].append(r[1][0])
    _errors_with(inst, r, "more than once")


def test_missing(bks):
    inst, routes, _ = bks
    r = copy.deepcopy(routes)
    r[0].pop()
    _errors_with(inst, r, "not visited")


@pytest.mark.parametrize("bad_id", [0, 32, -1])
def test_out_of_range(bks, bad_id):
    inst, routes, _ = bks
    r = copy.deepcopy(routes)
    r[0].append(bad_id)
    _errors_with(inst, r, "out of range")


def test_overload(bks):
    inst, routes, _ = bks
    r = copy.deepcopy(routes)
    r[0] += r[1]                    # merge two routes: load > 100, still 4 routes
    del r[1]
    _errors_with(inst, r, "load")


def test_too_many_routes(bks):
    inst, routes, _ = bks
    r = copy.deepcopy(routes)
    r.append([r[0].pop()])          # 6 routes, K = 5
    _errors_with(inst, r, "too many routes")


def test_wrong_claimed_cost(bks):
    inst, routes, cost = bks
    _errors_with(inst, routes, "claimed cost", claimed=cost + 1)


def test_empty_routes_ignored(bks):
    inst, routes, cost = bks
    report = check(inst, routes + [[], []], claimed_cost=cost)
    assert report.feasible and report.n_routes == 5
