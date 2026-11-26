# Original code (CMP 3005 term project).
"""Experiment runner: CSV format, seeds, validation, progress, hardware, resume, params file."""

import csv
import json
from collections import defaultdict

import pytest

from experiments.config import BASE_SEED, run_tabu
from experiments.run_experiments import PROGRESS_COLUMNS, RUN_COLUMNS, main

ARGS = ["--instances", "tiny_n7,A-n32-k5", "--runs", "2", "--budget-scale", "0.02"]


def _read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    main(ARGS + ["--out", str(out)])
    return out


def test_runs_csv_format_and_count(smoke):
    with (smoke / "runs.csv").open(newline="") as f:
        assert next(csv.reader(f)) == RUN_COLUMNS
    rows = _read(smoke / "runs.csv")
    assert len(rows) == 2 * 2 * 2                                  # instances x algos x runs
    assert {(r["instance"], r["algo"]) for r in rows} == {
        (i, a) for i in ("tiny_n7", "A-n32-k5") for a in ("ga", "tabu")}


def test_seeds_and_interleaving(smoke):
    rows = _read(smoke / "runs.csv")
    for r in rows:
        assert int(r["seed"]) == BASE_SEED + int(r["run"])
    # interleaved order: instance -> run -> algo
    order = [(r["instance"], r["run"], r["algo"]) for r in rows]
    assert order == [(i, str(k), a) for i in ("tiny_n7", "A-n32-k5") for k in range(2) for a in ("ga", "tabu")]


def test_all_feasible_and_validated(smoke):
    for r in _read(smoke / "runs.csv"):
        assert r["feasible"] == "True" and r["check_errors"] == ""
        assert r["cost"] and r["gap_pct"] != ""
        assert float(r["budget_s"]) == pytest.approx(30 * 0.02 if r["instance"] == "A-n32-k5" else 2 * 0.02)
        params = json.loads(r["params_json"])
        assert params["seed"] == int(r["seed"]) and params["time_limit"] == float(r["budget_s"])
        if r["instance"] == "tiny_n7":
            assert r["reached_optimum"] in ("True", "False")
        else:
            assert r["reached_optimum"] == ""


def test_progress_non_increasing(smoke):
    rows = _read(smoke / "progress.csv")
    assert rows and list(rows[0]) == PROGRESS_COLUMNS
    per_run = defaultdict(list)
    for r in rows:
        per_run[(r["instance"], r["algo"], r["run"])].append(float(r["best_cost"]))
    assert len(per_run) == 8
    for costs in per_run.values():
        assert costs == sorted(costs, reverse=True)


def test_hardware_file(smoke):
    text = (smoke / "hardware.txt").read_text()
    for key in ("cpu:", "logical_cores:", "ram_gb:", "python:", "numpy:", "git_commit:"):
        assert key in text


def test_resume_adds_no_rows(smoke):
    before = len(_read(smoke / "runs.csv"))
    main(ARGS + ["--out", str(smoke)])
    assert len(_read(smoke / "runs.csv")) == before


def test_params_file_overrides(tmp_path):
    pf = tmp_path / "params.json"
    pf.write_text(json.dumps({"tabu": {"tenure_max": 9}}))
    main(["--instances", "tiny_n7", "--algos", "tabu", "--runs", "1", "--budget-scale", "0.02",
          "--params-file", str(pf), "--out", str(tmp_path / "out")])
    row = _read(tmp_path / "out" / "runs.csv")[0]
    assert json.loads(row["params_json"])["tenure_max"] == 9


def test_protected_params_rejected():
    from cvrp.parser import parse_vrp
    from experiments.config import instance_path
    inst = parse_vrp(instance_path("tiny_n7"))
    with pytest.raises(ValueError):
        run_tabu(inst, 0, 0.1, {"time_limit": 999})
