# Original code (CMP 3005 term project).
"""Experiment runner: CSV format, seeds, validation, progress, hardware, resume, params file,
per-run solution files (Step 6a-3)."""

import csv
import json
from collections import defaultdict

import pytest

from cvrp.checker import check
from cvrp.parser import load_instance
from experiments.config import BASE_SEED, instance_path, run_tabu
from experiments.run_experiments import PROGRESS_COLUMNS, RUN_COLUMNS, main
from experiments.solutions import best_solution, load_all, load_solution

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


def test_every_row_has_its_solution_file(smoke):
    for r in _read(smoke / "runs.csv"):
        path = smoke / "solutions" / f"{r['instance']}_{r['algo']}_run{r['run']}.json"
        sol = load_solution(path)
        assert (sol["instance"], sol["algo"], sol["run"], sol["seed"]) == (
            r["instance"], r["algo"], int(r["run"]), int(r["seed"]))
        assert str(sol["feasible"]) == r["feasible"] and sol["cost"] == float(r["cost"])
        assert sol["n_routes"] == int(r["n_routes"]) == len(sol["routes"])
        text = path.read_bytes().decode()
        assert text == json.dumps(sol, sort_keys=True) + "\n" and "\r" not in text   # deterministic, LF


def test_solution_routes_recheck_to_same_cost(smoke):
    for sol in load_all(smoke):
        report = check(load_instance(instance_path(sol["instance"])), sol["routes"])
        assert report.feasible and report.cost == pytest.approx(sol["cost"])


def test_best_solution_is_lowest_cost(smoke):
    assert len(load_all(smoke)) == 8
    rows = [r for r in _read(smoke / "runs.csv") if r["instance"] == "A-n32-k5" and r["algo"] == "tabu"]
    expected = min(rows, key=lambda r: (float(r["cost"]), int(r["run"])))
    best = best_solution(smoke, "A-n32-k5", "tabu")
    assert best["cost"] == float(expected["cost"]) and best["run"] == int(expected["run"])
    assert best_solution(smoke, "team", "tabu") is None


def test_resume_adds_no_rows(smoke):
    before = len(_read(smoke / "runs.csv"))
    files = {p.name: p.read_bytes() for p in (smoke / "solutions").iterdir()}
    main(ARGS + ["--out", str(smoke)])
    assert len(_read(smoke / "runs.csv")) == before
    assert {p.name: p.read_bytes() for p in (smoke / "solutions").iterdir()} == files   # no new or changed files


def test_missing_solution_file_warns_but_is_not_rerun(tmp_path, capsys):
    args = ["--instances", "tiny_n7", "--algos", "tabu", "--runs", "1", "--budget-scale", "0.02",
            "--out", str(tmp_path)]
    main(args)
    path = tmp_path / "solutions" / "tiny_n7_tabu_run0.json"
    path.unlink()
    capsys.readouterr()
    main(args)
    assert "WARNING: 1 runs" in capsys.readouterr().out
    assert not path.exists() and len(_read(tmp_path / "runs.csv")) == 1


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
