# Original code (CMP 3005 term project).
"""Team and tiny instance generators: metric, labels, determinism, round trip."""

import json
import math

import numpy as np
import pytest

from cvrp.generator import (TINY_MIN_CW_GAP, ffd_bin_count, find_tiny_seed, generate_team_instance,
                            generate_tiny_instance, tiny_cw_gap, write_vrp)
from cvrp.instance import side_label
from cvrp.parser import parse_vrp


@pytest.fixture(scope="module")
def team():
    return generate_team_instance(seed=0)


def test_team_basic_shape(team):
    assert team.n == 100 and team.Q == 100
    assert team.K == math.ceil(team.total_demand() / team.Q) + 1
    assert team.demand[0] == 0 and team.demand[1:].min() >= 1 and team.demand[1:].max() <= 30
    assert (team.coords >= 0).all() and (team.coords <= 1000).all()
    assert tuple(team.coords[0]) == (500.0, 500.0)
    assert {"E", "A"} <= set(team.side[1:])     # customers on both sides


def test_team_depot_side_and_labels(team):
    assert team.side[0] == "A"
    assert team.side == [side_label(x) for x in team.coords[:, 0]]
    assert side_label(499.99) == "E" and side_label(500.0) == "A"


def test_team_symmetric_zero_diagonal(team):
    assert np.allclose(team.dist, team.dist.T)
    assert (np.diag(team.dist) == 0).all()


def test_team_penalty_only_across_sides(team):
    diff = team.coords[:, None, :] - team.coords[None, :, :]
    euclid = np.sqrt((diff ** 2).sum(axis=2))
    sides = np.array(team.side)
    cross = sides[:, None] != sides[None, :]
    assert np.allclose(team.dist - euclid, 150.0 * cross)


def test_team_triangle_inequality_all_triples(team):
    d = team.dist                     # 101^3 ≈ 1.03e6 triples, vectorized
    # lhs[i, j, k] = d[i, k];  rhs[i, j, k] = d[i, j] + d[j, k]
    assert (d[:, None, :] <= d[:, :, None] + d[None, :, :] + 1e-9).all()


def test_team_round_trip(team, tmp_path):
    p = write_vrp(team, tmp_path / "t.vrp", "test")
    back = parse_vrp(p)
    assert np.array_equal(back.coords, team.coords)
    assert np.array_equal(back.demand, team.demand)
    assert back.side == team.side                       # labels recomputed from file x agree
    assert np.array_equal(back.dist, team.dist)
    assert (back.Q, back.K, back.penalty) == (team.Q, team.K, team.penalty)


def test_same_seed_identical_bytes(tmp_path):
    a = write_vrp(generate_team_instance(3), tmp_path / "a.vrp", "c").read_bytes()
    b = write_vrp(generate_team_instance(3), tmp_path / "b.vrp", "c").read_bytes()
    assert a == b


def test_different_seed_different_bytes(tmp_path):
    a = write_vrp(generate_team_instance(3), tmp_path / "a.vrp", "c").read_bytes()
    b = write_vrp(generate_team_instance(4), tmp_path / "b.vrp", "c").read_bytes()
    assert a != b


def test_ffd_bin_count():
    assert ffd_bin_count([60, 50, 40, 30, 20], 100) == 2   # 60+40, 50+30+20
    assert ffd_bin_count([51, 51, 51], 100) == 3


@pytest.mark.parametrize("n", [7, 8, 9])
def test_tiny_capacity_binds(n, root):
    inst = parse_vrp(root / "data/tiny" / f"tiny_n{n}.vrp")
    assert inst.n == n
    assert inst.total_demand() > inst.Q                      # at least 2 routes needed
    assert inst.K >= ffd_bin_count(list(inst.demand[1:]), inst.Q)
    seed, met_gap = find_tiny_seed(n)                        # committed file == generator output
    gen = generate_tiny_instance(n, seed)
    assert np.array_equal(gen.coords, inst.coords) and np.array_equal(gen.demand, inst.demand)
    # Clarke–Wright must NOT be optimal (GA and TS start from it), here by >= 3%
    cw, opt, gap = tiny_cw_gap(inst)
    assert met_gap and cw > opt + 1e-9 and gap >= TINY_MIN_CW_GAP
    stored = json.loads((root / "data/tiny" / f"tiny_n{n}.opt.json").read_text())["cost"]
    assert stored == opt
