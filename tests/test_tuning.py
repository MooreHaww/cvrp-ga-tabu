# Original code (CMP 3005 term project).
"""Step 6b: OFAT grid, seeds, selection rule, analysis outputs and the smoke runner."""

import csv
import json
from dataclasses import fields

import pytest

from cvrp.ga import GAParams, default_penalty
from cvrp.parser import load_instance
from cvrp.tabu import TabuParams
from experiments.config import BASE_SEED, DEFAULT_RUNS, PROTECTED, instance_path, run_ga
from experiments.tuning import (COLUMNS, TUNING_INSTANCES, analyze, build_tasks, build_validation_tasks, main,
                                select, summarise)
from experiments.tuning_grid import BASELINE, GRID, build_configs, value_label

FINAL = set(range(BASE_SEED, BASE_SEED + DEFAULT_RUNS))


# --------------------------------------------------------------------------- grid

@pytest.mark.parametrize("algo, expected", [("tabu", 10), ("ga", 13), ("ga_memetic", 3)])
def test_grid_sizes_and_single_baseline(algo, expected):
    configs = build_configs(algo)
    assert len(configs) == expected
    assert [c[:2] for c in configs].count(BASELINE) == 1 and configs[0] == (*BASELINE, {})
    # no configuration equals the all-defaults one except the baseline itself
    for factor in GRID[algo]:
        assert (factor.name, value_label(factor.default)) not in [c[:2] for c in configs]
    assert len({(c[0], c[1]) for c in configs}) == len(configs)


def test_grid_defaults_match_dataclass_defaults():
    ga = {f.name: f.default for f in fields(GAParams)}
    ts = {f.name: f.default for f in fields(TabuParams)}
    for algo, defaults in (("ga", ga), ("ga_memetic", ga), ("tabu", ts)):
        for factor in GRID[algo]:
            for key, value in zip(factor.keys, factor.default):
                if key == "penalty_scale":
                    assert value == 1 and ga["penalty"] is None       # scale 1 == default lambda
                else:
                    assert defaults[key] == value, (algo, key)


def test_seeds_are_tuning_seeds_only():
    tasks = build_tasks(TUNING_INSTANCES, ["ga", "tabu", "ga_memetic"], 5, 1.0)
    assert len(tasks) == 26 * 3 * 5
    assert {t["seed"] for t in tasks} == {1000 + r for r in range(5)}
    assert all(t["seed"] == 1000 + t["run"] for t in tasks)
    assert not {t["seed"] for t in tasks} & FINAL
    val = build_validation_tasks(TUNING_INSTANCES, {"ga": {"pop_size": 100}}, 5, 1.0)
    assert len(val) == 3 * 5 * 2 * 2
    assert {t["seed"] for t in val} == {1005, 1006, 1007, 1008, 1009}
    assert not {t["seed"] for t in val} & FINAL
    with pytest.raises(ValueError):
        build_tasks(["A-n45-k6"], ["ga"], BASE_SEED - 1000 + 1, 1.0)   # would reach seed 20260


def test_penalty_scale_override():
    inst = load_instance(instance_path("A-n32-k5"))
    out = run_ga(inst, seed=1, budget=0.05, overrides={"penalty_scale": 2})
    assert out.params["penalty"] == pytest.approx(2 * default_penalty(inst))
    assert out.params["penalty_scale"] == 2
    with pytest.raises(ValueError):
        run_ga(inst, seed=1, budget=0.05, overrides={"penalty_scale": 2, "penalty": 5.0})


# --------------------------------------------------------------------------- synthetic analysis

def _rows(algo, pname, pvalue, instance, gaps, feasible=None, cost0=1000.0):
    """Synthetic runs.csv rows. For the team instance, cost = cost0 * (1 + gap/100)."""
    rows = []
    for r, g in enumerate(gaps):
        ok = True if feasible is None else feasible[r]
        rows.append({"instance": instance, "algo": algo, "param_name": pname, "param_value": pvalue, "run": r,
                     "seed": 1000 + r, "budget_s": 10.0,
                     "cost": (cost0 * (1 + g / 100) if ok else ""), "n_routes": 5, "feasible": ok,
                     "normalised_gap": ("" if instance == "team" or not ok else g),
                     "iterations": 1, "work": 1, "params_json": "{}", "git_commit": "x"})
    return rows


def _synthetic():
    """Baseline gap 5 on both instances; pop_size=100 clearly better (gap 1, tiny spread);
    pop_size=25 slightly better than default (4.9) but within the SE; tabu tenure 2-6 better
    but with one infeasible run (more failures than the default)."""
    rows = []
    for inst in ("A-n45-k6", "team"):
        rows += _rows("ga", "baseline", "default", inst, [4.0, 5.0, 6.0])
        rows += _rows("ga", "pop_size", "100", inst, [0.9, 1.0, 1.1])
        rows += _rows("ga", "pop_size", "25", inst, [3.9, 4.9, 5.9])
        rows += _rows("tabu", "baseline", "default", inst, [2.0, 2.1, 2.2])
        rows += _rows("tabu", "tenure", "2-6", inst, [0.5, 0.6, 0.0], feasible=[True, True, False])
    # another algorithm reaches cost0 on the team instance -> team reference = cost0 = 1000
    rows += _rows("ga_memetic", "baseline", "default", "team", [0.0])
    return rows


def _summary():
    rows = _synthetic()
    return summarise(rows, team_ref=1000.0)


def test_baseline_shared_by_every_factor():
    summary = _summary()
    defaults = [s for s in summary if s["algo"] == "ga" and s["is_default"] and s["instance"] == "A-n45-k6"]
    # every GA factor gets a default point, all from the same baseline runs
    assert {s["param_name"] for s in defaults} == {f.name for f in GRID["ga"]}
    assert {(s["n"], s["mean"], s["se"]) for s in defaults} == {(3, 5.0, defaults[0]["se"])}
    tabu_default = [s for s in summary if s["algo"] == "tabu" and s["param_name"] == "tenure" and s["is_default"]]
    assert tabu_default and all(s["param_value"] == "5-15" for s in tabu_default)
    assert tabu_default[0]["mean"] == pytest.approx(2.1)


def test_selection_rule():
    tuned, decisions = select(_summary())
    assert tuned["ga"] == {"pop_size": 100}                # big gain > SE bound -> changed
    assert tuned["tabu"] == {}                             # better mean but more failures -> kept
    assert tuned["ga_memetic"] == {}
    d = {(x["algo"], x["param_name"]): x for x in decisions}
    assert d[("ga", "pop_size")]["changed"] and d[("ga", "pop_size")]["best"] == "100"
    assert not d[("tabu", "tenure")]["changed"]


def test_selection_keeps_default_when_gain_within_se():
    rows = []
    for inst in ("A-n45-k6", "A-n60-k9"):
        rows += _rows("ga", "baseline", "default", inst, [4.0, 5.0, 6.0])
        rows += _rows("ga", "pop_size", "25", inst, [3.9, 4.9, 5.9])     # 0.1 better, SE about 0.41
    tuned, decisions = select(summarise(rows, None))
    assert tuned["ga"] == {} and decisions[0]["best"] == "25" and not decisions[0]["changed"]


def test_selection_keeps_default_when_se_unknown():
    rows = []
    for inst in ("A-n45-k6", "A-n60-k9"):
        rows += _rows("ga", "baseline", "default", inst, [5.0])          # 1 run: no SE
        rows += _rows("ga", "pop_size", "100", inst, [1.0])
    tuned, decisions = select(summarise(rows, None))
    assert tuned["ga"] == {} and decisions[0]["best"] == "100" and not decisions[0]["changed"]


def test_analyze_writes_outputs(tmp_path):
    out = tmp_path / "tuning"
    out.mkdir()
    with (out / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(_synthetic())
    val = (_rows("ga", "validation", "default", "A-n45-k6", [5.0, 5.2]) +
           _rows("ga", "validation", "tuned", "A-n45-k6", [1.0, 1.2]))
    with (out / "validation_runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(val)
    tables, figures, tuned_path = tmp_path / "tables", tmp_path / "figures", tmp_path / "tuned.json"
    res = analyze(out, tables, figures, tuned_path)
    assert (tables / "tuning_summary.csv").exists() and (tables / "tuning_selection.csv").exists()
    assert (tables / "tuning_validation.csv").exists() and not res["warnings"]
    assert (figures / "sensitivity_ga_pop_size.png").exists()
    assert (figures / "sensitivity_tabu_tenure.png").exists()
    tuned = json.loads(tuned_path.read_text())
    assert set(tuned) == {"ga", "tabu", "ga_memetic"} and tuned["ga"] == {"pop_size": 100}
    assert not any(PROTECTED & set(section) for section in tuned.values())


# --------------------------------------------------------------------------- smoke runner

def test_smoke_run_and_resume(tmp_path):
    args = ["run", "--algos", "ga_memetic", "--instances", "A-n45-k6", "--runs", "1",
            "--budget-scale", "0.01", "--out", str(tmp_path)]
    main(args)
    with (tmp_path / "runs.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and list(rows[0]) == COLUMNS
    assert {r["param_value"] for r in rows} == {"default", "0.5", "1.0"}
    for r in rows:
        assert r["seed"] == "1000" and r["feasible"] in ("True", "False")
        if r["feasible"] == "True":                        # validated by checker.check inside run_one
            assert float(r["normalised_gap"]) >= 0 and float(r["cost"]) > 0
        else:                                              # 0.1 s may be too short: kept as a failure
            assert r["cost"] == "" and r["normalised_gap"] == ""
        params = json.loads(r["params_json"])
        expected = {"default": 0.1, "0.5": 0.5, "1.0": 1.0}[r["param_value"]]
        assert params["local_search_prob"] == expected and params["memetic"] is True
    assert (tmp_path / "hardware.txt").exists()
    main(args)                                            # resume: nothing new
    with (tmp_path / "runs.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 3
