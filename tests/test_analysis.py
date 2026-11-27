# Original code (CMP 3005 term project).
"""Step 6c: analysis tables, Wilcoxon (incl. 'all ties'), Holm, crossings, CW baseline,
determinism - on a small synthetic raw folder (no experiment runs)."""

import csv
import statistics

import pytest

from cvrp.parser import load_instance
from cvrp.savings import clarke_wright
from experiments.analysis import analyze, crossing_stats, cw_baseline, holm, main, read_rows
from experiments.config import BASE_SEED, instance_path
from experiments.run_experiments import PROGRESS_COLUMNS, RUN_COLUMNS, write_solution

GA_COSTS = [800.0 + r for r in range(10)]          # A-n32-k5: plain GA 800..809, the others at the BKS 784


def _row(instance, algo, run, cost, ref, budget=1.0):
    return {"instance": instance, "algo": algo, "run": run, "seed": BASE_SEED + run, "budget_s": budget,
            "elapsed_s": budget, "cost": cost, "n_routes": 5, "K": 5, "feasible": True, "check_errors": "",
            "bks_or_opt": ref if ref is not None else "", "gap_pct": "", "iterations": 1, "work": 1,
            "reached_optimum": "", "time_to_best_s": 0.5, "params_json": "{}", "git_commit": "x"}


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    """Synthetic raw folder: A-n32-k5 (10 runs, GA worse), tiny_n7 (all at the optimum: ties)
    and team (2 runs, the real CW routes, so the crossing check uses the real cost formula)."""
    raw = tmp_path_factory.mktemp("raw")
    team = load_instance(instance_path("team"))
    cw = clarke_wright(team, warn=False)
    team_routes = [list(r) for r in cw.solution.non_empty_routes()]
    rows, progress = [], []
    for algo in ("ga", "ga_ls", "tabu"):
        for run in range(10):
            for inst, cost, ref in (("A-n32-k5", GA_COSTS[run] if algo == "ga" else 784.0, 784.0),
                                    ("tiny_n7", 278.0, 278.0)):
                rows.append(_row(inst, algo, run, cost, ref))
                progress += [{"instance": inst, "algo": algo, "run": run, "seed": BASE_SEED + run,
                              "elapsed_s": t, "best_cost": c} for t, c in ((0.001, cost + 10), (0.01 * (run + 1), cost))]
                write_solution(raw / "solutions", {"instance": inst, "algo": algo, "run": run, "seed": BASE_SEED + run,
                                                   "budget_s": 1.0, "cost": cost, "n_routes": 1, "feasible": True,
                                                   "routes": [[1]]})
        for run in range(2):
            rows.append(_row("team", algo, run, cw.cost, None))
            progress.append({"instance": "team", "algo": algo, "run": run, "seed": BASE_SEED + run,
                             "elapsed_s": 0.001, "best_cost": cw.cost})
            write_solution(raw / "solutions", {"instance": "team", "algo": algo, "run": run, "seed": BASE_SEED + run,
                                               "budget_s": 1.0, "cost": cw.cost, "n_routes": len(team_routes),
                                               "feasible": True, "routes": team_routes})
    for name, columns, data in (("runs.csv", RUN_COLUMNS, rows), ("progress.csv", PROGRESS_COLUMNS, progress)):
        with (raw / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
    return raw


@pytest.fixture(scope="module")
def result(raw, tmp_path_factory):
    return analyze(raw, tmp_path_factory.mktemp("out"))


def _get(rows, **keys):
    return next(r for r in rows if all(r[k] == v for k, v in keys.items()))


def test_main_results_values(result):
    ga = _get(result["main"], instance="A-n32-k5", algo="ga")
    assert ga["n"] == 10 and ga["best"] == 800.0 and ga["mean"] == pytest.approx(804.5)
    assert ga["median"] == pytest.approx(804.5) and ga["std"] == pytest.approx(statistics.stdev(GA_COSTS))
    assert ga["mean_gap_pct"] == pytest.approx(100 * (804.5 - 784) / 784) and ga["hits"] == 0
    tabu = _get(result["main"], instance="A-n32-k5", algo="tabu")
    assert tabu["hits"] == 10 and tabu["mean_gap_pct"] == 0.0 and tabu["ref_type"] == "BKS"
    team = _get(result["main"], instance="team", algo="ga_ls")
    assert result["team_ref"] == pytest.approx(team["best"]) and team["ref_type"].startswith("best known to us")
    assert [r["instance"] for r in result["main"]][::3] == ["A-n32-k5", "team", "tiny_n7"]   # INSTANCE_ORDER


def test_tiny_results(result):
    row = _get(result["tiny"], instance="tiny_n7", algo="tabu")
    assert row["hit_rate"] == 1.0 and row["optimum"] == 278.0
    assert row["mean_time_to_best_s"] == pytest.approx(statistics.fmean(0.01 * (r + 1) for r in range(10)))
    assert row["max_time_to_best_s"] == pytest.approx(0.1)


def test_wilcoxon_values_and_all_ties(result):
    tested = _get(result["wilcoxon"], instance="A-n32-k5", pair="tabu vs ga")
    assert tested["result"] == "tested" and tested["n_nonzero"] == 10
    assert tested["statistic"] == 0.0 and tested["p"] == pytest.approx(2 / 1024)        # exact, all d < 0
    assert tested["median_diff"] == pytest.approx(-20.5)
    for inst, pair in (("A-n32-k5", "ga_ls vs tabu"), ("tiny_n7", "tabu vs ga"), ("team", "ga_ls vs ga")):
        tie = _get(result["wilcoxon"], instance=inst, pair=pair)
        assert tie["result"] == "all ties" and tie["p"] is None and tie["n_nonzero"] == 0
    # Holm family of 'tabu vs ga' = the tested benchmark/team instances = only A-n32-k5 -> unchanged
    assert tested["p_holm"] == pytest.approx(tested["p"])


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm([]) == []


def test_cw_baseline_matches_savings(raw, result):
    row = _get(result["cw"], instance="A-n32-k5")
    res = clarke_wright(load_instance(instance_path("A-n32-k5")), warn=False)
    assert row["cw_cost"] == res.cost and row["cw_routes"] == res.n_routes
    assert row["tabu_mean_impr_pct"] == pytest.approx(100 * (res.cost - 784) / res.cost)
    assert row["tabu_runs_below_cw"] == 10
    team = _get(cw_baseline(read_rows(raw / "runs.csv")), instance="team")
    assert team["ga_runs_below_cw"] == 0 and team["ga_mean_impr_pct"] == 0.0        # synthetic team = CW


def test_crossings_match_independent_count_and_cost(result):
    team = load_instance(instance_path("team"))
    routes = [list(r) for r in clarke_wright(team, warn=False).solution.non_empty_routes()]
    st = crossing_stats(team, routes)
    recount = 0
    for r in routes:                                   # independent recount from the side labels
        labels = ["A"] + [team.side[v] for v in r] + ["A"]          # the depot (500, 500) is side A
        recount += sum(1 for x, y in zip(labels, labels[1:]) if x != y)
    assert st["crossings"] == recount and st["crossings"] >= 2 * st["routes_with_E"]
    row = _get(result["crossings"], algo="tabu")
    assert row["n_runs"] == 2 and row["mean_crossings"] == recount and row["best_run"] == 0
    assert row["best_penalty_share_pct"] == pytest.approx(100 * recount * 150 / st["cost"])


def test_figures_written(result):
    names = {p.name for p in result["figures"]}
    assert {"team_routes_ga_ls.png", "team_routes_tabu.png", "convergence_team.png", "boxplots_final.png"} <= names


def test_deterministic_tables(raw, tmp_path):
    a, b = main(["--raw", str(raw), "--out", str(tmp_path / "a"), "--tuning-dirs", ""]), \
        main(["--raw", str(raw), "--out", str(tmp_path / "b"), "--tuning-dirs", ""])
    names = sorted(p.name for p in a["tables"])
    assert names == sorted(p.name for p in b["tables"]) and len(names) == 10
    for name in names:
        data = (tmp_path / "a" / "tables" / name).read_bytes()
        assert data == (tmp_path / "b" / "tables" / name).read_bytes() and b"\r" not in data
