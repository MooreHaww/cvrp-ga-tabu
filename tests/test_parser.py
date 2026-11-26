# Original code (CMP 3005 term project).
"""Augerat instances: parsing, K, demand, and the .sol files as checker-validated BKS."""

import pytest

from conftest import AUGERAT, ROOT
from cvrp.checker import check
from cvrp.parser import load_bks_solution, load_instance, parse_vrp, read_k

# (n customers, Q, K, total demand). Total demand was summed from DEMAND_SECTION of each file.
EXPECTED = {
    "A-n32-k5": (31, 100, 5, 410),
    "A-n45-k6": (44, 100, 6, 593),
    "A-n60-k9": (59, 100, 9, 829),
    "A-n80-k10": (79, 100, 10, 942),
}
# Published CVRPLIB values, used only as a sanity check of what the .sol files say.
EXPECTED_BKS = {"A-n32-k5": 784, "A-n45-k6": 944, "A-n60-k9": 1354, "A-n80-k10": 1763}


def _raw_demand_sum(name: str) -> int:
    """Sum the DEMAND_SECTION straight from the text file (independent of the parser)."""
    lines = (ROOT / "data/cvrplib" / f"{name}.vrp").read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "DEMAND_SECTION")
    end = next(i for i, l in enumerate(lines) if l.strip() == "DEPOT_SECTION")
    return sum(int(l.split()[1]) for l in lines[start + 1:end])


@pytest.mark.parametrize("name", AUGERAT)
def test_augerat_parses(name):
    inst = parse_vrp(ROOT / "data/cvrplib" / f"{name}.vrp")
    n, Q, K, total = EXPECTED[name]
    assert (inst.name, inst.n, inst.Q, inst.K) == (name, n, Q, K)
    assert inst.total_demand() == total == _raw_demand_sum(name)
    assert inst.demand[0] == 0 and inst.coords.shape == (n + 1, 2)
    assert inst.dist.shape == (n + 1, n + 1)
    assert (inst.dist == inst.dist.T).all() and (inst.dist.diagonal() == 0).all()


def test_euc2d_rounding_matches_tsplib():
    inst = parse_vrp(ROOT / "data/cvrplib/A-n32-k5.vrp")
    (x0, y0), (x1, y1) = inst.coords[0], inst.coords[1]  # (82,76) and (96,44)
    assert inst.dist[0, 1] == int(((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5 + 0.5) == 35


@pytest.mark.parametrize("name", AUGERAT)
def test_bks_solution_is_feasible_and_cost_matches(name):
    inst, routes, stated = load_bks_solution(ROOT / "data/cvrplib" / f"{name}.vrp")
    report = check(inst, routes, claimed_cost=stated)
    assert report.feasible, report.errors
    assert report.cost == stated == inst.bks == EXPECTED_BKS[name]


def test_load_instance_reads_bks_from_sol():
    assert load_instance(ROOT / "data/cvrplib/A-n32-k5.vrp").bks == 784


def test_read_k_sources(capsys):
    assert read_k("A-n32-k5", "No of trucks: 5") == 5
    assert read_k("tiny_n7", "Tiny, No of trucks: 4") == 4
    assert read_k("A-n32-k5", "No of trucks: 6") == 5
    assert "WARNING" in capsys.readouterr().out
    with pytest.raises(ValueError):
        read_k("nothing", "")
