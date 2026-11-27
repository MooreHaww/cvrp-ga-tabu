# Original code (CMP 3005 term project).
"""Step 7a: the report numbers sheet on a small synthetic results folder.

Checks: deterministic LF output, every commit hash in the sheet comes from the inputs, the computed findings follow
the input numbers, and the parameter and instance sections read the files."""

import csv
import json
import re
import shutil
from pathlib import Path

from cvrp.parser import load_instance
from experiments.report_numbers import build_sheet, main

REPO = Path(__file__).resolve().parents[1]
MAIN_HASH, TIMING_HASH, HEAD_HASH = "a1" * 20, "b2" * 20, "c3" * 20
ENV = {"python": "9.9.9", "numpy": "0.0.1"}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _hardware(path: Path, commit: str, workers: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"cpu: TestCPU 9000\ngit_commit: {commit}\nworkers: {workers}\n", newline="\n")


def make_inputs(root: Path) -> Path:
    """A tiny results folder: 2 instances x 2 algorithms x 2 runs, timings that follow known power laws."""
    for rel in ("data/cvrplib/A-n32-k5.vrp", "data/cvrplib/A-n32-k5.sol", "data/team/team_n100_seed14835.vrp"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, root / rel)
    runs = []
    for inst, budget in (("A-n32-k5", "30.0"), ("team", "60.0")):
        for algo in ("ga", "tabu"):
            for r in range(2):
                params = {"pop_size": 31, "crossover_rate": 0.9} if algo == "ga" else {"candidate_size": 10}
                runs.append({"instance": inst, "algo": algo, "run": r, "seed": 777 + r, "budget_s": budget,
                             "params_json": json.dumps(params), "git_commit": MAIN_HASH})
    _write_csv(root / "results/raw/runs.csv", runs)
    _hardware(root / "results/raw/hardware.txt", MAIN_HASH, 2)
    _hardware(root / "results/timing/hardware.txt", TIMING_HASH, 1)
    timing = []
    for n in (50, 100, 800):
        timing.append({"component": "ga_generation", "n": n, "rep": 0, "seconds": 1e-6 * n * n,
                       "units_of_work": 10, "git_commit": TIMING_HASH})
        moves = 80 * n                                 # moves per iteration ~ n, time per move ~ sqrt(n)
        timing.append({"component": "tabu_iteration", "n": n, "rep": 0, "seconds": moves * 1e-6 * (n / 100) ** 0.5,
                       "units_of_work": moves, "git_commit": TIMING_HASH})
    _write_csv(root / "results/timing/timing.csv", timing)
    _write_csv(root / "results/timing/share_ga_ls.csv",
               [{"instance": "team_n100_seed14835", "descent_share": 0.8771, "split_decode_share": 0.1, "rest_share": 0.0229}])
    tables = root / "results/tables"
    _write_csv(tables / "main_results.csv", [
        {"instance": "A-n32-k5", "algo": "ga", "ref": 784.0, "ref_type": "BKS", "n": 2, "mean": 801.5, "hits": 0},
        {"instance": "A-n32-k5", "algo": "tabu", "ref": 784.0, "ref_type": "BKS", "n": 2, "mean": 790.25, "hits": 1},
        {"instance": "team", "algo": "ga", "ref": 15000.5, "ref_type": "ref", "n": 2, "mean": 15100.0, "hits": 0},
        {"instance": "team", "algo": "tabu", "ref": 15000.5, "ref_type": "ref", "n": 2, "mean": 15050.0, "hits": 2}])
    _write_csv(tables / "cw_baseline.csv", [
        {"instance": "A-n32-k5", "ga_mean_impr_pct": 1.25, "ga_runs_below_cw": 2},
        {"instance": "team", "ga_mean_impr_pct": 3.5, "ga_runs_below_cw": 1}])
    _write_csv(tables / "wilcoxon.csv", [
        {"instance": "A-n32-k5", "pair": "tabu vs ga", "result": "tested", "p_holm": 0.5, "median_diff": -11.5},
        {"instance": "team", "pair": "tabu vs ga", "result": "tested", "p_holm": 0.0123, "median_diff": -50.0}])
    _write_csv(tables / "team_crossings.csv", [
        {"algo": a, "n_runs": 2, "mean_routes_with_E": "7.0", "min_crossings": 14, "max_crossings": 14}
        for a in ("ga", "tabu")])
    _write_csv(tables / "complexity_fits.csv", [
        {"component": "ga_generation", "theory_slope": 1.0, "fit_n100_slope": 2.0},
        {"component": "tabu_iteration", "theory_slope": 1.0, "fit_n100_slope": 1.5}])
    (root / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "experiments/tuned_params.json").write_text(json.dumps({"ga": {"pop_size": 31}}))
    return root


def test_output_is_deterministic_and_lf(tmp_path):
    root = make_inputs(tmp_path / "in")
    a = main(["--root", str(root), "--out", str(tmp_path / "a.md")]).read_bytes()
    b = main(["--root", str(root), "--out", str(tmp_path / "b.md")]).read_bytes()
    assert a == b and b"\r" not in a
    assert build_sheet(root, HEAD_HASH, ENV) == build_sheet(root, HEAD_HASH, ENV)


def test_every_hash_comes_from_the_inputs(tmp_path):
    text = build_sheet(make_inputs(tmp_path), HEAD_HASH, ENV)
    hashes = set(re.findall(r"\b[0-9a-f]{40}\b", text))
    assert hashes == {MAIN_HASH, TIMING_HASH, HEAD_HASH}


def test_setup_reads_seeds_hardware_and_budgets(tmp_path):
    text = build_sheet(make_inputs(tmp_path), HEAD_HASH, ENV)
    assert "Seeds used by the main run: 777, 778" in text
    assert "Runs in results/raw/runs.csv: 8 (ga 4, tabu 4)" in text
    assert "Workers: main run 2, timing run 1" in text
    assert "| cpu | TestCPU 9000 | TestCPU 9000 |" in text
    assert "| python | 9.9.9 |" in text
    assert "| A-n32-k5 | 30.0 | 30.0 |" in text and "| team | 60.0 | 60.0 |" in text


def test_findings_follow_the_input_numbers(tmp_path):
    root = make_inputs(tmp_path)
    text = build_sheet(root, HEAD_HASH, ENV)
    assert "A-n32-k5: the lowest mean cost is 790.25, reached by tabu." in text
    assert "the runs that reach it are ga 0 of 2, tabu 1 of 2" in text
    assert "1.25% (A-n32-k5) to 3.50% (team)" in text and "3 of 4 runs" in text
    assert "on team (p_holm = 0.0123, median diff = -50.00); not significant on A-n32-k5" in text
    assert "cross the strait exactly 14 times" in text
    # GA time ~ n^2; Tabu time = moves (~ n) x time per move (~ n^0.5)
    assert "two-point slope from n = 100 to 800 is 2.00" in text
    assert "is 1.50, of which 1.00 comes from the moves" in text and "0.50 from the time per move" in text
    assert "87.7% of the run time on team_n100_seed14835" in text
    # change one input number: the sentence must change with it
    path = root / "results/tables/main_results.csv"
    path.write_text(path.read_text().replace("790.25", "799.75"))
    changed = build_sheet(root, HEAD_HASH, ENV)
    assert "lowest mean cost is 799.75" in changed and "790.25" not in changed


def test_parameters_and_instances_are_read_from_files(tmp_path):
    root = make_inputs(tmp_path)
    text = build_sheet(root, HEAD_HASH, ENV)
    assert "| pop_size | 50 | 31 | 31 | yes |" in text               # default, tuned, used
    assert "| candidate_size | 10 |  | 10 | yes |" not in text        # empty tuned cell is written as '–'
    assert "| candidate_size | 10 | – | 10 | yes |" in text
    team = load_instance(root / "data/team/team_n100_seed14835.vrp")
    sides = team.side[1:]
    assert f"Customers on side E: {sides.count('E')}, on side A: {sides.count('A')}" in text
    assert "Seed: 14835 (from the file name)" in text and f"P = {team.penalty:g}" in text
    assert f"| team | {team.name} | {team.n} | {team.Q} | {team.K} | {int(team.demand.sum())} |" in text
    # tables are copied cell by cell; a missing table is reported, not invented
    assert "| A-n32-k5 | tabu | 784.0 | BKS | 2 | 790.25 | 1 |" in text
    assert "Source: results/tables/tiny_results.csv\n\n(file missing or empty)" in text
