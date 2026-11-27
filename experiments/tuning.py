# Original code
"""Step 6b: parameter study (one factor at a time) – runner and analysis.

    python -m experiments.tuning run                 # 26 configs x 3 instances x 5 runs x 10 s
    python -m experiments.tuning analyze             # tables, sensitivity plots, tuned_params.json
    python -m experiments.tuning run --validate      # tuned vs default on held-out tuning seeds
    python -m experiments.tuning run --smoke         # 1 run, 5% budgets, into results/smoke_tuning

Seeds: tuning run r uses TUNING_BASE_SEED + r = 1000 + r. The final-experiment seeds
(BASE_SEED + r = 20260..20269) are NEVER used here, so parameter choice and final
evaluation use independent random streams. Selection uses runs 0..4 (seeds 1000..1004),
validation uses runs 5..9 (seeds 1005..1009).

Round 2 (Step 6b-2, `--round 2`): named Tabu Search configurations (interaction check,
lam0_scale) and OFAT for ga_ls; 4 instances, 15 s, selection runs 10..14 (seeds
1010..1014), validation runs 15..19 (seeds 1015..1019); outputs in results/tuning2 and
results/tables/tuning2_*.csv. Round-1 data is never touched.

    python -m experiments.tuning run --round 2 [--workers 3]
    python -m experiments.tuning analyze --round 2
    python -m experiments.tuning run --round 2 --validate

Every run goes through run_experiments.run_one, i.e. through checker.check().
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path

from cvrp.tabu import TabuParams
from experiments.config import ALGORITHMS, BASE_SEED, DEFAULT_RUNS, TEAM_SEED_DEFAULT, _checked_overrides, instance_path
from experiments.run_experiments import _Appender, ensure_team_file, git_commit, run_one, write_hardware
from experiments.tuning_grid import (BASELINE, GRID, GRID_R2, TS_CONFIGS_R2, Factor, build_configs, n_changes,
                                     value_label)

TUNING_BASE_SEED = 1000
TUNING_RUNS = 5
TUNING_BUDGET_S = 10.0
TUNING_INSTANCES = ("A-n45-k6", "A-n60-k9", "team")
TUNING_ALGOS = ("ga", "tabu", "ga_memetic")
VALIDATION_ALGOS = ("ga", "tabu")
VALIDATION_FIRST_RUN = TUNING_RUNS          # validation runs 5..9 -> seeds 1005..1009
FINAL_SEEDS = range(BASE_SEED, BASE_SEED + DEFAULT_RUNS)

COLUMNS = ["instance", "algo", "param_name", "param_value", "run", "seed", "budget_s", "cost", "n_routes",
           "feasible", "normalised_gap", "iterations", "work", "params_json", "git_commit"]
SUMMARY_COLUMNS = ["algo", "param_name", "param_value", "instance", "n", "mean", "std", "se", "failures",
                   "is_default"]
SELECTION_COLUMNS = ["algo", "param_name", "default", "best", "mean_default", "se_default", "mean_best", "se_best",
                     "improvement", "se_bound", "failures_default", "failures_best", "changed"]
VALIDATION_COLUMNS = ["algo", "instance", "n_default", "mean_default", "se_default", "failures_default",
                      "n_tuned", "mean_tuned", "se_tuned", "failures_tuned", "diff_tuned_minus_default"]
INSTANCE_ORDER = ("A-n45-k6", "A-n60-k9", "A-n80-k10", "team")   # column / legend order in tables and plots
LOSS_PP = 0.5          # round 2: flag an instance where the choice is worse than default by > 0.5 percentage points


@dataclass(frozen=True)
class TuningRound:
    """Everything that differs between tuning rounds."""

    number: int
    out: str                          # raw CSV folder
    smoke_out: str                    # smoke-run folder (git-ignored)
    prefix: str                       # table names: <prefix>_summary.csv, ...
    first_run: int                    # selection runs first_run .. first_run + runs - 1 (seed = 1000 + run)
    validation_first_run: int         # validation runs (held out from selection)
    runs: int
    budget_s: float
    instances: tuple[str, ...]
    algos: tuple[str, ...]
    validation_algos: tuple[str, ...]


ROUND1 = TuningRound(1, "results/tuning", "results/smoke_tuning", "tuning", 0, VALIDATION_FIRST_RUN, TUNING_RUNS,
                     TUNING_BUDGET_S, TUNING_INSTANCES, TUNING_ALGOS, VALIDATION_ALGOS)
# Round 2: 15 s because ga_ls spends ~90% of its time in the descent and completes only
# about 30-60 generations in 10 s (Step 4c).
ROUND2 = TuningRound(2, "results/tuning2", "results/smoke_tuning2", "tuning2", 10, 15, 5, 15.0,
                     ("A-n45-k6", "A-n60-k9", "A-n80-k10", "team"), ("tabu", "ga_ls"), ("tabu", "ga_ls"))
ROUNDS = {1: ROUND1, 2: ROUND2}
TABU_DEFAULTS = {f.name: f.default for f in fields(TabuParams)}


def configs_for(rnd: TuningRound, algo: str) -> list[tuple[str, str, dict]]:
    """(param_name, param_value, overrides) of every configuration of an algorithm in a round.
    Round 2 Tabu Search uses the named configurations (param_name 'config'). Time: O(#configs)."""
    if rnd.number == 1:
        return build_configs(algo)
    if algo == "tabu":
        return [("config", name, dict(overrides)) for name, overrides in TS_CONFIGS_R2]
    return build_configs(algo, GRID_R2)


# --------------------------------------------------------------------------- tasks and runs

def tuning_seed(run: int) -> int:
    """Seed of tuning run r; refuses to return one of the final-experiment seeds. Time: O(1)."""
    seed = TUNING_BASE_SEED + run
    if seed in FINAL_SEEDS:
        raise ValueError(f"seed {seed} is reserved for the final experiments")
    return seed


def _task(name, algo, run, budget, overrides, param_name, param_value, commit) -> dict:
    return {"instance": name, "algo": algo, "run": run, "seed": tuning_seed(run), "budget": budget,
            "path": str(instance_path(name, TEAM_SEED_DEFAULT)), "overrides": overrides,
            "param_name": param_name, "param_value": param_value, "git_commit": commit}


def build_tasks(instances, algos, runs, budget_scale, commit="", rnd: TuningRound = ROUND1) -> list[dict]:
    """Study tasks, interleaved: instance -> run -> algorithm -> config.
    The all-defaults baseline appears once per (instance, run, algorithm). Time: O(#tasks)."""
    budget = rnd.budget_s * budget_scale
    return [_task(name, algo, run, budget, overrides, pname, pvalue, commit)
            for name in instances for run in range(rnd.first_run, rnd.first_run + runs) for algo in algos
            for pname, pvalue, overrides in configs_for(rnd, algo)]


def build_validation_tasks(instances, tuned: dict, runs, budget_scale, commit="",
                           rnd: TuningRound = ROUND1) -> list[dict]:
    """Tuned vs default on the held-out validation runs of the round. Time: O(#tasks)."""
    budget = rnd.budget_s * budget_scale
    first = rnd.validation_first_run
    return [_task(name, algo, run, budget, overrides, "validation", config, commit)
            for name in instances for run in range(first, first + runs)
            for algo in rnd.validation_algos
            for config, overrides in (("default", {}), ("tuned", tuned.get(algo, {})))]


def tuning_row(task: dict) -> dict:
    """Run one task with run_experiments.run_one (which validates with checker.check)
    and keep the tuning columns. Top-level so it can be sent to a worker process."""
    row, _progress = run_one(task)
    return {"instance": row["instance"], "algo": row["algo"], "param_name": task["param_name"],
            "param_value": task["param_value"], "run": row["run"], "seed": row["seed"],
            "budget_s": row["budget_s"], "cost": row["cost"], "n_routes": row["n_routes"],
            "feasible": row["feasible"],
            "normalised_gap": row["gap_pct"],      # vs BKS; empty for the team instance (set in analyze)
            "iterations": row["iterations"], "work": row["work"], "params_json": row["params_json"],
            "git_commit": row["git_commit"]}


def _key(row: dict) -> tuple:
    return row["instance"], row["algo"], row["param_name"], str(row["param_value"]), int(row["run"])


def read_rows(path: Path) -> list[dict]:
    """All rows of a CSV file (empty list if it does not exist). Time: O(rows)."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def run_tasks(tasks: list[dict], csv_path: Path, workers: int) -> int:
    """Run every task whose key is not yet in csv_path (resume); append + flush per run.
    Returns the number of rows written."""
    done = {_key(r) for r in read_rows(csv_path)}
    todo = [t for t in tasks if _key(t) not in done]
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done (resume), {len(todo)} to run -> {csv_path}")
    out = _Appender(csv_path, COLUMNS)
    infeasible = 0
    try:
        def record(row: dict) -> None:
            nonlocal infeasible
            out.write([row])
            infeasible += not row["feasible"]
            cost = f"{row['cost']:.2f}" if row["feasible"] else "-"
            print(f"{row['instance']:<9} {row['algo']:<10} {row['param_name']:<17} {str(row['param_value']):<8} "
                  f"run {row['run']} seed {row['seed']}  cost {cost:>10}", flush=True)

        if workers <= 1:
            for task in todo:
                record(tuning_row(task))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(tuning_row, t) for t in todo]
                for fut in as_completed(futures):
                    record(fut.result())
    finally:
        out.close()
    print(f"done: {len(todo)} runs written, {len(tasks) - len(todo)} skipped, {infeasible} infeasible")
    return len(todo)


# --------------------------------------------------------------------------- analysis: gaps and statistics

def _feasible(row: dict) -> bool:
    return str(row["feasible"]) == "True"


def team_reference(rows: list[dict]) -> float | None:
    """Best feasible team cost in ANY tuning/validation run (the team instance has no BKS). Time: O(rows)."""
    costs = [float(r["cost"]) for r in rows if r["instance"] == "team" and _feasible(r)]
    return min(costs) if costs else None


def row_gap(row: dict, team_ref: float | None) -> float | None:
    """Normalised gap of one run in %, or None for an infeasible run (= failure). Time: O(1)."""
    if not _feasible(row):
        return None
    if row["instance"] == "team":
        return 100.0 * (float(row["cost"]) - team_ref) / team_ref
    return float(row["normalised_gap"])        # vs BKS, computed at run time


def cell_stats(gaps: list[float | None]) -> dict:
    """n, mean, std (ddof 1), SE = std/sqrt(n) and failures of one cell.
    Failures are excluded from the mean. With n < 2 the spread is unknown: std = SE = nan.
    Time: O(len(gaps))."""
    ok = [g for g in gaps if g is not None]
    n = len(ok)
    mean = statistics.fmean(ok) if n else math.nan
    std = statistics.stdev(ok) if n >= 2 else math.nan
    se = std / math.sqrt(n) if n >= 2 else math.nan
    return {"n": n, "mean": mean, "std": std, "se": se, "failures": len(gaps) - n}


def average_stats(per_instance: list[dict]) -> dict:
    """Instance average: mean of the instance means, SE = sqrt(sum SE_i^2) / #instances
    (independent instances). Undefined (nan) if some instance has no feasible run. Time: O(#instances)."""
    k = len(per_instance)
    means = [s["mean"] for s in per_instance]
    defined = k > 0 and all(not math.isnan(m) for m in means)
    return {"n": sum(s["n"] for s in per_instance),
            "mean": statistics.fmean(means) if defined else math.nan,
            "std": math.nan,
            "se": math.sqrt(sum(s["se"] ** 2 for s in per_instance)) / k if defined else math.nan,
            "failures": sum(s["failures"] for s in per_instance)}


def _instances_in(rows: list[dict]) -> list[str]:
    present = {r["instance"] for r in rows}
    return [i for i in INSTANCE_ORDER if i in present] + sorted(present - set(INSTANCE_ORDER))


def _group_gaps(rows: list[dict], team_ref: float | None) -> dict[tuple, list]:
    """Gaps grouped by (algo, param_name, param_value, instance). Time: O(rows)."""
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["algo"], r["param_name"], r["param_value"], r["instance"]), []).append(
            row_gap(r, team_ref))
    return groups


def summarise(rows: list[dict], team_ref: float | None, grid: dict[str, list[Factor]] | None = None) -> list[dict]:
    """Summary rows per (algo, param, value, instance) plus instance='average'.
    The default value of every factor uses the SHARED baseline rows. grid defaults to the
    round-1 GRID. Time: O(rows + grid)."""
    groups = _group_gaps(rows, team_ref)
    instances = _instances_in(rows)
    out = []
    for algo, factors in (GRID if grid is None else grid).items():
        for factor in factors:
            for value in factor.values:
                is_default = value == factor.default
                source = BASELINE if is_default else (factor.name, value_label(value))
                cells = [groups.get((algo, *source, inst)) for inst in instances]
                if all(c is None for c in cells):
                    continue                                      # value not run (partial data)
                stats = [cell_stats(c or []) for c in cells]
                for inst, s in zip(instances + ["average"], stats + [average_stats(stats)]):
                    out.append({"algo": algo, "param_name": factor.name, "param_value": value_label(value),
                                "instance": inst, **s, "is_default": is_default})
    return out


def summarise_configs(rows: list[dict], team_ref: float | None, algo: str,
                      configs: list[tuple[str, dict]]) -> list[dict]:
    """Summary rows for NAMED configurations (round-2 Tabu Search): per instance and the
    instance average; param_name = 'config'. Time: O(rows + #configs * #instances)."""
    groups = _group_gaps(rows, team_ref)
    instances = _instances_in(rows)
    out = []
    for name, _overrides in configs:
        cells = [groups.get((algo, "config", name, inst)) for inst in instances]
        if all(c is None for c in cells):
            continue
        stats = [cell_stats(c or []) for c in cells]
        for inst, s in zip(instances + ["average"], stats + [average_stats(stats)]):
            out.append({"algo": algo, "param_name": "config", "param_value": name, "instance": inst, **s,
                        "is_default": name == "default"})
    return out


# --------------------------------------------------------------------------- analysis: selection

def select_config(summary: list[dict], algo: str, configs: list[tuple[str, dict]],
                  defaults: dict) -> tuple[str, dict, set[str]]:
    """Selection among named configurations (round 2):
    eligible = configs with a known SE and no more failures than the default;
    best = the eligible config with the lowest instance-averaged mean gap;
    within = eligible configs with mean - mean_best <= max(SE_best, SE_config) (within one SE);
    chosen = the config in `within` with the FEWEST changes relative to the defaults
             (ties: lower mean). If the default has no known SE, keep the default.
    Returns (chosen name, its overrides, names within one SE). Time: O(#configs)."""
    avg = {s["param_value"]: s for s in summary
           if s["algo"] == algo and s["param_name"] == "config" and s["instance"] == "average"}
    default = avg.get("default")
    if default is None or math.isnan(default["se"]):
        return "default", {}, set()
    eligible = [s for s in avg.values()
                if not math.isnan(s["mean"]) and not math.isnan(s["se"]) and s["failures"] <= default["failures"]]
    best = min(eligible, key=lambda s: s["mean"])
    within = [s for s in eligible if s["mean"] - best["mean"] <= max(best["se"], s["se"])]
    overrides = dict(configs)
    chosen = min(within, key=lambda s: (n_changes(overrides[s["param_value"]], defaults), s["mean"]))
    name = chosen["param_value"]
    return name, dict(overrides[name]), {s["param_value"] for s in within}


def loss_flags(summary: list[dict], algo: str, param_name: str, chosen: str, default: str,
               threshold: float = LOSS_PP) -> str:
    """Instances where the chosen value/config is worse than the default by more than
    `threshold` percentage points ('' if none or if the choice is the default). Time: O(len(summary))."""
    if chosen == default:
        return ""
    cell = {(s["param_value"], s["instance"]): s["mean"] for s in summary
            if s["algo"] == algo and s["param_name"] == param_name and s["instance"] != "average"}
    flags = []
    for (value, inst), mean in sorted(cell.items()):
        if value == chosen and (default, inst) in cell:
            diff = mean - cell[(default, inst)]
            if diff > threshold:
                flags.append(f"{inst} +{diff:.2f} pp")
    return "; ".join(flags)


def select(summary: list[dict], grid: dict[str, list[Factor]] | None = None) -> tuple[dict, list[dict]]:
    """Selection rule (per algorithm, per factor, independently = OFAT):
    best = value with the lowest instance-averaged mean gap. Change from the default only if
        mean_default - mean_best > max(SE_default, SE_best)
    and the best value has no more failures than the default; otherwise keep the default.
    If either SE is unknown (fewer than 2 feasible runs somewhere), the default is kept.
    grid defaults to the round-1 GRID.
    Returns (tuned overrides per algorithm, decision rows). Time: O(len(summary))."""
    grid = GRID if grid is None else grid
    tuned: dict[str, dict] = {algo: {} for algo in grid}
    decisions = []
    avg = [s for s in summary if s["instance"] == "average"]
    for algo, factors in grid.items():
        for factor in factors:
            rows = [s for s in avg if s["algo"] == algo and s["param_name"] == factor.name]
            default = next((s for s in rows if s["is_default"]), None)
            usable = [s for s in rows if not math.isnan(s["mean"])]
            if default is None or not usable:
                continue
            best = min(usable, key=lambda s: (s["mean"], not s["is_default"]))   # ties -> default
            improvement = default["mean"] - best["mean"]
            known = not (math.isnan(default["se"]) or math.isnan(best["se"]))
            bound = max(default["se"], best["se"]) if known else math.nan
            changed = (known and not best["is_default"] and improvement > bound
                       and best["failures"] <= default["failures"])
            if changed:
                value = next(v for v in factor.values if value_label(v) == best["param_value"])
                tuned[algo].update(factor.overrides(value))
            decisions.append({"algo": algo, "param_name": factor.name, "default": value_label(factor.default),
                              "best": best["param_value"], "mean_default": default["mean"],
                              "se_default": default["se"], "mean_best": best["mean"], "se_best": best["se"],
                              "improvement": improvement, "se_bound": bound,
                              "failures_default": default["failures"], "failures_best": best["failures"],
                              "changed": changed})
    for algo in tuned:
        _checked_overrides(tuned[algo])          # never seed / time_limit / max_*
    return tuned, decisions


# --------------------------------------------------------------------------- analysis: plots

def plot_factor(summary: list[dict], algo: str, factor: Factor, out_path: Path, subtitle: str) -> Path:
    """Sensitivity plot of one factor: x = value (categorical), y = mean normalised gap,
    one line per instance (error bars = SE) and a thick line for the instance average."""
    import matplotlib
    matplotlib.use("Agg")                     # file output only
    import matplotlib.pyplot as plt

    rows = [s for s in summary if s["algo"] == algo and s["param_name"] == factor.name]
    labels = [value_label(v) for v in factor.values if any(s["param_value"] == value_label(v) for s in rows)]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for inst in dict.fromkeys(s["instance"] for s in rows):
        cells = {s["param_value"]: s for s in rows if s["instance"] == inst}
        y = [cells[lab]["mean"] for lab in labels]
        err = [0.0 if math.isnan(cells[lab]["se"]) else cells[lab]["se"] for lab in labels]
        if inst == "average":
            ax.errorbar(x, y, yerr=err, color="black", linewidth=3, marker="o", markersize=8, capsize=5,
                        label="instance average", zorder=3)
        else:
            ax.errorbar(x, y, yerr=err, linewidth=1.3, marker="o", capsize=3, alpha=0.85, label=inst)
    default_label = value_label(factor.default)
    tick_labels = []
    for i, lab in enumerate(labels):
        if lab == default_label:
            ax.axvspan(i - 0.35, i + 0.35, color="grey", alpha=0.15)
            lab += "\n(default)"
        tick_labels.append(lab)
    ax.set_xticks(x, tick_labels, fontsize=11)
    ax.set_xlabel(factor.label, fontsize=12)
    ax.set_ylabel("mean normalised gap (%)", fontsize=12)
    failures = sum(s["failures"] for s in rows if s["instance"] != "average")
    fail_txt = f", {failures} infeasible runs excluded" if failures else ""
    ax.set_title(f"{algo}: sensitivity to {factor.name}\n{subtitle}{fail_txt}", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_configs(summary: list[dict], algo: str, configs: list[tuple[str, dict]], out_path: Path,
                 subtitle: str) -> Path:
    """Point plot of named configurations: x = configuration, y = mean normalised gap,
    one series per instance (error bars = SE, slightly offset) and the instance average."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [s for s in summary if s["algo"] == algo and s["param_name"] == "config"]
    names = [name for name, _ in configs if any(s["param_value"] == name for s in rows)]
    series = list(dict.fromkeys(s["instance"] for s in rows))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    k = len(series)
    for idx, inst in enumerate(series):
        cells = {s["param_value"]: s for s in rows if s["instance"] == inst}
        x = [i + (idx - (k - 1) / 2) * 0.1 for i in range(len(names))]
        y = [cells[n]["mean"] for n in names]
        err = [0.0 if math.isnan(cells[n]["se"]) else cells[n]["se"] for n in names]
        if inst == "average":
            ax.errorbar(x, y, yerr=err, color="black", linewidth=3, marker="D", markersize=8, capsize=5,
                        label="instance average", zorder=3)
        else:
            ax.errorbar(x, y, yerr=err, linestyle="none", marker="o", capsize=3, label=inst)
    if "default" in names:
        i = names.index("default")
        ax.axvspan(i - 0.4, i + 0.4, color="grey", alpha=0.15)
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize=10)
    ax.set_xlabel("Tabu Search configuration", fontsize=12)
    ax.set_ylabel("mean normalised gap (%)", fontsize=12)
    failures = sum(s["failures"] for s in rows if s["instance"] != "average")
    fail_txt = f", {failures} infeasible runs excluded" if failures else ""
    ax.set_title(f"{algo}: named configurations\n{subtitle}{fail_txt}", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _subtitle(rows: list[dict], design: str = "OFAT") -> str:
    runs = sorted({int(r["run"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    budget = statistics.median(float(r["budget_s"]) for r in rows)
    plural = "s" if len(runs) != 1 else ""
    return f"{design}, {len(runs)} run{plural} x {budget:g} s per cell, seeds {seeds[0]}-{seeds[-1]}"


# --------------------------------------------------------------------------- analysis: validation

def summarise_validation(rows: list[dict], team_ref: float | None, algos=VALIDATION_ALGOS,
                         instance_pp: float | None = None) -> tuple[list[dict], list[str]]:
    """Tuned vs default per (algo, instance) + average; warnings when tuned is worse on
    average by more than the larger SE, or (if instance_pp is given) worse on a single
    instance by more than instance_pp percentage points. Time: O(rows)."""
    out, warnings = [], []
    instances = _instances_in(rows)
    for algo in algos:
        stats = {}
        for config in ("default", "tuned"):
            per_inst = [cell_stats([row_gap(r, team_ref) for r in rows
                                    if r["algo"] == algo and r["param_value"] == config and r["instance"] == inst])
                        for inst in instances]
            stats[config] = per_inst + [average_stats(per_inst)]
        if not any(s["n"] or s["failures"] for s in stats["default"][:-1]):
            continue                                                # algorithm not validated
        for inst, d, t in zip(instances + ["average"], stats["default"], stats["tuned"]):
            out.append({"algo": algo, "instance": inst,
                        "n_default": d["n"], "mean_default": d["mean"], "se_default": d["se"],
                        "failures_default": d["failures"], "n_tuned": t["n"], "mean_tuned": t["mean"],
                        "se_tuned": t["se"], "failures_tuned": t["failures"],
                        "diff_tuned_minus_default": t["mean"] - d["mean"]})
            if instance_pp is not None and inst != "average" and t["mean"] - d["mean"] > instance_pp:
                warnings.append(f"WARNING: tuned {algo} is worse than its default on {inst} by "
                                f"{t['mean'] - d['mean']:.2f} pp in validation (> {instance_pp} pp)")
        d, t = stats["default"][-1], stats["tuned"][-1]
        worse = t["mean"] - d["mean"] > max(d["se"], t["se"]) if not (math.isnan(d["se"]) or math.isnan(t["se"])) \
            else t["mean"] > d["mean"]
        if worse or t["failures"] > d["failures"]:
            warnings.append(f"WARNING: tuned {algo} is worse than its default in validation "
                            f"({t['mean']:.3f}% vs {d['mean']:.3f}%, failures {t['failures']} vs {d['failures']}); "
                            f"consider removing its overrides from tuned_params.json")
    return out, warnings


# --------------------------------------------------------------------------- analyze entry point

def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) and not math.isnan(v) else
                                 ("" if isinstance(v, float) else v)) for k, v in r.items()})
    return path


def analyze(out_dir: Path, tables_dir: Path, figures_dir: Path, tuned_path: Path, rnd: TuningRound = ROUND1,
            ref_dirs: tuple[Path, ...] = ()) -> dict:
    """Build the summary table, the sensitivity plots, tuned_params.json and (if
    validation runs exist) the validation table. Returns the decisions and warnings.
    Round 2 is handled by analyze_round2; ref_dirs are only used there."""
    if rnd.number == 2:
        return analyze_round2(out_dir, tables_dir, figures_dir, tuned_path, ref_dirs)
    if tuned_path.exists() and "ga_ls" in json.loads(tuned_path.read_text()):
        raise SystemExit(f"{tuned_path} already holds round-2 choices (section 'ga_ls'); a round-1 analysis "
                         f"would overwrite them. Use --tuned <other file> or 'analyze --round 2'.")
    rows = read_rows(out_dir / "runs.csv")
    val_rows = read_rows(out_dir / "validation_runs.csv")
    if not rows:
        raise SystemExit(f"no tuning runs in {out_dir / 'runs.csv'}; run 'python -m experiments.tuning run' first")
    team_ref = team_reference(rows + val_rows)
    if team_ref is not None:
        print(f"team reference = best feasible team cost in any tuning run = {team_ref:.2f}")

    summary = summarise(rows, team_ref)
    _write_csv(tables_dir / "tuning_summary.csv", SUMMARY_COLUMNS, summary)
    subtitle = _subtitle(rows)
    plots = [plot_factor(summary, algo, f, figures_dir / f"sensitivity_{algo}_{f.name}.png", subtitle)
             for algo, factors in GRID.items() for f in factors
             if any(s["algo"] == algo and s["param_name"] == f.name for s in summary)]

    tuned, decisions = select(summary)
    _write_csv(tables_dir / "tuning_selection.csv", SELECTION_COLUMNS, decisions)
    tuned_path.parent.mkdir(parents=True, exist_ok=True)
    tuned_path.write_text(json.dumps(tuned, indent=2, sort_keys=True) + "\n", newline="\n")
    for d in decisions:
        verdict = "CHANGE" if d["changed"] else "keep"
        print(f"{d['algo']:<10} {d['param_name']:<17} default {d['default']:<7} best {d['best']:<7} "
              f"gain {d['improvement']:+.3f} vs SE bound {d['se_bound']:.3f} -> {verdict}"
              + ("  (SE unknown: < 2 feasible runs)" if math.isnan(d["se_bound"]) else ""))
    print(f"tuned parameters -> {tuned_path}: {json.dumps(tuned, sort_keys=True)}")

    warnings = []
    if val_rows:
        val_summary, warnings = summarise_validation(val_rows, team_ref)
        _write_csv(tables_dir / "tuning_validation.csv", VALIDATION_COLUMNS, val_summary)
    for w in warnings:
        print(w)
    return {"summary": summary, "decisions": decisions, "tuned": tuned, "plots": plots, "warnings": warnings}


# --------------------------------------------------------------------------- round-2 analysis (Step 6b-2)

DEFAULT_TUNED = "experiments/tuned_params.json"


def _within_one_se(avg_rows: list[dict]) -> set[str]:
    """Values whose average mean is within one SE of the best: mean - best <= max(SE_best, SE).
    Rows with an unknown mean or SE are never 'within'. Time: O(len(avg_rows))."""
    usable = [s for s in avg_rows if not (math.isnan(s["mean"]) or math.isnan(s["se"]))]
    if not usable:
        return set()
    best = min(usable, key=lambda s: s["mean"])
    return {s["param_value"] for s in usable if s["mean"] - best["mean"] <= max(best["se"], s["se"])}


def selection_rows(summary: list[dict], algo: str, param_name: str, labels: list[str], default: str,
                   chosen: str, within: set[str], changes: dict[str, int]) -> list[dict]:
    """Round-2 selection table: one row per candidate value/config of (algo, param_name) with
    its per-instance means, whether it is within one SE of the best, whether it was chosen,
    and (on the chosen row) the instances where it loses > LOSS_PP to the default.
    Time: O(len(summary))."""
    cells = {(s["param_value"], s["instance"]): s for s in summary
             if s["algo"] == algo and s["param_name"] == param_name}
    instances = [i for i in dict.fromkeys(inst for _, inst in cells) if i != "average"]
    flag = loss_flags(summary, algo, param_name, chosen, default)
    out = []
    for lab in labels:
        avg = cells.get((lab, "average"))
        if avg is None:
            continue                                   # not run (partial data)
        out.append({"algo": algo, "param_name": param_name, "param_value": lab, "n_changes": changes[lab],
                    "mean_avg": avg["mean"], "se_avg": avg["se"], "failures": avg["failures"],
                    **{f"mean_{i}": cells[(lab, i)]["mean"] for i in instances},
                    "is_default": lab == default, "within_se": lab in within, "chosen": lab == chosen,
                    "loss_flag": flag if lab == chosen else ""})
    return out


def _selection_columns(rows: list[dict]) -> list[str]:
    means = list(dict.fromkeys(k for r in rows for k in r if k.startswith("mean_") and k != "mean_avg"))
    return (["algo", "param_name", "param_value", "n_changes", "mean_avg", "se_avg", "failures"] + means
            + ["is_default", "within_se", "chosen", "loss_flag"])


def _round1_ga_section(tuned_path: Path) -> dict:
    """The plain-GA overrides chosen in round 1 (kept unchanged in round 2): read from
    tuned_path, or from the repository file if tuned_path does not exist yet. Time: O(file)."""
    source = tuned_path if tuned_path.exists() else Path(DEFAULT_TUNED)
    return json.loads(source.read_text()).get("ga", {}) if source.exists() else {}


def analyze_round2(out_dir: Path, tables_dir: Path, figures_dir: Path, tuned_path: Path,
                   ref_dirs: tuple[Path, ...] = ()) -> dict:
    """Round 2: summary of ga_ls (OFAT) and of the named TS configurations, the selection
    table with loss flags, the plots, tuned_params.json with four sections and (if validation
    runs exist) the validation table. The team reference is the best feasible team cost in
    this round AND in the runs found in ref_dirs (round 1)."""
    rows = read_rows(out_dir / "runs.csv")
    val_rows = read_rows(out_dir / "validation_runs.csv")
    if not rows:
        raise SystemExit(f"no tuning runs in {out_dir / 'runs.csv'}; run 'python -m experiments.tuning run "
                         f"--round 2' first")
    ref_rows = [r for d in ref_dirs for f in ("runs.csv", "validation_runs.csv") for r in read_rows(d / f)]
    team_ref = team_reference(rows + val_rows + ref_rows)
    if team_ref is not None:
        print(f"team reference = best feasible team cost in rounds 1 + 2 = {team_ref:.2f}")

    ga_summary = summarise(rows, team_ref, GRID_R2)
    ts_summary = summarise_configs(rows, team_ref, "tabu", TS_CONFIGS_R2)
    summary = ga_summary + ts_summary
    _write_csv(tables_dir / "tuning2_summary.csv", SUMMARY_COLUMNS, summary)
    plots = [plot_factor(ga_summary, "ga_ls", f, figures_dir / f"sensitivity_ga_ls_{f.name}.png", _subtitle(rows))
             for f in GRID_R2["ga_ls"] if any(s["param_name"] == f.name for s in ga_summary)]
    if ts_summary:
        plots.append(plot_configs(ts_summary, "tabu", TS_CONFIGS_R2, figures_dir / "tuning2_tabu_configs.png",
                                  _subtitle(rows, "interaction check")))

    # ga_ls: the round-1 OFAT rule, factor by factor.
    tuned_ga_ls, decisions = select(ga_summary, GRID_R2)
    table = []
    for d in decisions:
        factor = next(f for f in GRID_R2["ga_ls"] if f.name == d["param_name"])
        labels = [value_label(v) for v in factor.values]
        avg = [s for s in ga_summary if s["param_name"] == factor.name and s["instance"] == "average"]
        chosen = d["best"] if d["changed"] else d["default"]
        table += selection_rows(ga_summary, "ga_ls", factor.name, labels, d["default"], chosen, _within_one_se(avg),
                                {lab: int(lab != d["default"]) for lab in labels})
        print(f"ga_ls      {d['param_name']:<17} default {d['default']:<5} best {d['best']:<5} "
              f"gain {d['improvement']:+.3f} vs SE bound {d['se_bound']:.3f} -> {'CHANGE' if d['changed'] else 'keep'}")

    # Tabu Search: fewest changes within one SE of the best configuration.
    tabu_name, tuned_tabu, within = ("default", {}, set())
    if ts_summary:
        tabu_name, tuned_tabu, within = select_config(ts_summary, "tabu", TS_CONFIGS_R2, TABU_DEFAULTS)
        changes = {name: n_changes(ov, TABU_DEFAULTS) for name, ov in TS_CONFIGS_R2}
        table += selection_rows(ts_summary, "tabu", "config", [n for n, _ in TS_CONFIGS_R2], "default", tabu_name,
                                within, changes)
        print(f"tabu       within one SE of the best: {sorted(within) or '- (SE unknown)'} -> {tabu_name} "
              f"({changes[tabu_name]} changes)")
    _write_csv(tables_dir / "tuning2_selection.csv", _selection_columns(table), table)

    warnings = [f"WARNING: chosen {r['algo']} {r['param_name']}={r['param_value']} is worse than the default "
                f"by more than {LOSS_PP} pp on: {r['loss_flag']}" for r in table if r["loss_flag"]]
    tuned = {"ga": _round1_ga_section(tuned_path), "ga_ls": tuned_ga_ls["ga_ls"], "tabu": tuned_tabu,
             "ga_memetic": {}}
    for section in tuned.values():
        _checked_overrides(section)                     # never seed / time_limit / max_*
    tuned_path.parent.mkdir(parents=True, exist_ok=True)
    tuned_path.write_text(json.dumps(tuned, indent=2, sort_keys=True) + "\n", newline="\n")
    print(f"tuned parameters -> {tuned_path}: {json.dumps(tuned, sort_keys=True)}")

    if val_rows:
        val_summary, val_warnings = summarise_validation(val_rows, team_ref, ROUND2.validation_algos, LOSS_PP)
        _write_csv(tables_dir / "tuning2_validation.csv", VALIDATION_COLUMNS, val_summary)
        warnings += val_warnings
    for w in warnings:
        print(w)
    return {"summary": summary, "selection": table, "tuned": tuned, "plots": plots, "warnings": warnings,
            "team_ref": team_ref}


# --------------------------------------------------------------------------- CLI

SMOKE_OUT = ROUND1.smoke_out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Parameter study for GA and TS (Step 6b, round 2: Step 6b-2).")
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the tuning study (or --validate)")
    run.add_argument("--round", type=int, choices=sorted(ROUNDS), default=1, help="tuning round (default 1)")
    run.add_argument("--algos", default=None, help="subset of the round's algorithms (round 1: ga,tabu,ga_memetic; "
                                                   "round 2: tabu,ga_ls)")
    run.add_argument("--instances", default=None, help="default: the round's instances")
    run.add_argument("--runs", type=int, default=None, help="runs per cell (default 5)")
    run.add_argument("--budget-scale", type=float, default=1.0, help="multiplies the round's budget (10 s / 15 s)")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--out", default=None, help="default results/tuning (round 2: results/tuning2)")
    run.add_argument("--validate", action="store_true",
                     help="tuned vs default on the held-out seeds (round 1: 1005..1009, round 2: 1015..1019)")
    run.add_argument("--tuned", default=None, help=f"tuned params JSON (default {DEFAULT_TUNED})")
    run.add_argument("--smoke", action="store_true", help="= --runs 1 --budget-scale 0.05 into the round's smoke folder")
    an = sub.add_parser("analyze", help="tables, sensitivity plots and tuned_params.json")
    an.add_argument("--round", type=int, choices=sorted(ROUNDS), default=1, help="tuning round (default 1)")
    an.add_argument("--out", default=None)
    an.add_argument("--tables", default="results/tables")
    an.add_argument("--figures", default="results/figures")
    an.add_argument("--tuned", default=None)
    an.add_argument("--smoke", action="store_true", help="read and write everything inside the round's smoke folder")
    args = ap.parse_args(argv)
    rnd = ROUNDS[args.round]
    if args.out is None:
        args.out = rnd.out
    if args.command == "run":
        args.algos = args.algos or ",".join(rnd.algos)
        args.instances = args.instances or ",".join(rnd.instances)
        args.runs = rnd.runs if args.runs is None else args.runs
    if args.smoke:
        args.out = rnd.smoke_out
        if args.command == "run":
            args.runs, args.budget_scale = 1, 0.05
        else:
            args.tables = args.figures = rnd.smoke_out
    if args.tuned is None:
        args.tuned = f"{args.out}/tuned_params.json" if args.smoke else DEFAULT_TUNED
    return args


def main(argv=None) -> Path:
    """Entry point; returns the output directory."""
    args = parse_args(argv)
    rnd = ROUNDS[args.round]
    out_dir = Path(args.out)
    if args.command == "analyze":
        analyze(out_dir, Path(args.tables), Path(args.figures), Path(args.tuned), rnd, ref_dirs=(Path(ROUND1.out),))
        return out_dir

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    instances = [i.strip() for i in args.instances.split(",") if i.strip()]
    unknown = [a for a in algos if a not in ALGORITHMS or a not in rnd.algos]
    if unknown:
        raise SystemExit(f"unknown algorithms for round {rnd.number}: {unknown}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if "team" in instances:
        ensure_team_file(TEAM_SEED_DEFAULT)
    write_hardware(out_dir, args.workers)
    commit = git_commit()
    if args.validate:
        tuned_path = Path(args.tuned)
        if not tuned_path.exists():
            raise SystemExit(f"{tuned_path} not found; run 'python -m experiments.tuning analyze' first")
        tuned = json.loads(tuned_path.read_text())
        tasks = build_validation_tasks(instances, tuned, args.runs, args.budget_scale, commit, rnd)
        run_tasks(tasks, out_dir / "validation_runs.csv", args.workers)
    else:
        tasks = build_tasks(instances, algos, args.runs, args.budget_scale, commit, rnd)
        run_tasks(tasks, out_dir / "runs.csv", args.workers)
    return out_dir


if __name__ == "__main__":
    main()
