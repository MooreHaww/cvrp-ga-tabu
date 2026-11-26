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

Every run goes through run_experiments.run_one, i.e. through checker.check().
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiments.config import ALGORITHMS, BASE_SEED, DEFAULT_RUNS, TEAM_SEED_DEFAULT, _checked_overrides, instance_path
from experiments.run_experiments import _Appender, ensure_team_file, git_commit, run_one, write_hardware
from experiments.tuning_grid import BASELINE, GRID, Factor, build_configs, value_label

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


def build_tasks(instances, algos, runs, budget_scale, commit="") -> list[dict]:
    """OFAT study tasks, interleaved: instance -> run -> algorithm -> config.
    The all-defaults baseline appears once per (instance, run, algorithm). Time: O(#tasks)."""
    budget = TUNING_BUDGET_S * budget_scale
    return [_task(name, algo, run, budget, overrides, pname, pvalue, commit)
            for name in instances for run in range(runs) for algo in algos
            for pname, pvalue, overrides in build_configs(algo)]


def build_validation_tasks(instances, tuned: dict, runs, budget_scale, commit="") -> list[dict]:
    """Tuned vs default for GA and TS on the held-out tuning runs 5..9. Time: O(#tasks)."""
    budget = TUNING_BUDGET_S * budget_scale
    return [_task(name, algo, run, budget, overrides, "validation", config, commit)
            for name in instances for run in range(VALIDATION_FIRST_RUN, VALIDATION_FIRST_RUN + runs)
            for algo in VALIDATION_ALGOS
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
    return [i for i in TUNING_INSTANCES if i in present] + sorted(present - set(TUNING_INSTANCES))


def summarise(rows: list[dict], team_ref: float | None) -> list[dict]:
    """Summary rows per (algo, param, value, instance) plus instance='average'.
    The default value of every factor uses the SHARED baseline rows. Time: O(rows + grid)."""
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["algo"], r["param_name"], r["param_value"], r["instance"]), []).append(
            row_gap(r, team_ref))
    instances = _instances_in(rows)
    out = []
    for algo, factors in GRID.items():
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


# --------------------------------------------------------------------------- analysis: selection

def select(summary: list[dict]) -> tuple[dict, list[dict]]:
    """Selection rule (per algorithm, per factor, independently = OFAT):
    best = value with the lowest instance-averaged mean gap. Change from the default only if
        mean_default - mean_best > max(SE_default, SE_best)
    and the best value has no more failures than the default; otherwise keep the default.
    If either SE is unknown (fewer than 2 feasible runs somewhere), the default is kept.
    Returns (tuned overrides per algorithm, decision rows). Time: O(len(summary))."""
    tuned: dict[str, dict] = {algo: {} for algo in GRID}
    decisions = []
    avg = [s for s in summary if s["instance"] == "average"]
    for algo, factors in GRID.items():
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


def _subtitle(rows: list[dict]) -> str:
    runs = sorted({int(r["run"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    budget = statistics.median(float(r["budget_s"]) for r in rows)
    plural = "s" if len(runs) != 1 else ""
    return f"OFAT, {len(runs)} run{plural} x {budget:g} s per cell, seeds {seeds[0]}-{seeds[-1]}"


# --------------------------------------------------------------------------- analysis: validation

def summarise_validation(rows: list[dict], team_ref: float | None) -> tuple[list[dict], list[str]]:
    """Tuned vs default per (algo, instance) + average; warnings when tuned is worse on
    average by more than the larger SE. Time: O(rows)."""
    out, warnings = [], []
    instances = _instances_in(rows)
    for algo in VALIDATION_ALGOS:
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


def analyze(out_dir: Path, tables_dir: Path, figures_dir: Path, tuned_path: Path) -> dict:
    """Build the summary table, the sensitivity plots, tuned_params.json and (if
    validation runs exist) the validation table. Returns the decisions and warnings."""
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


# --------------------------------------------------------------------------- CLI

SMOKE_OUT = "results/smoke_tuning"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Parameter study (OFAT) for GA and TS (Step 6b).")
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the tuning study (or --validate)")
    run.add_argument("--algos", default=",".join(TUNING_ALGOS), help=f"subset of {TUNING_ALGOS}")
    run.add_argument("--instances", default=",".join(TUNING_INSTANCES))
    run.add_argument("--runs", type=int, default=TUNING_RUNS)
    run.add_argument("--budget-scale", type=float, default=1.0, help=f"multiplies the {TUNING_BUDGET_S:g} s budget")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--out", default="results/tuning")
    run.add_argument("--validate", action="store_true", help="tuned vs default on seeds 1005..1009")
    run.add_argument("--tuned", default=None, help="tuned params JSON (default experiments/tuned_params.json)")
    run.add_argument("--smoke", action="store_true", help=f"= --runs 1 --budget-scale 0.05 --out {SMOKE_OUT}")
    an = sub.add_parser("analyze", help="tables, sensitivity plots and tuned_params.json")
    an.add_argument("--out", default="results/tuning")
    an.add_argument("--tables", default="results/tables")
    an.add_argument("--figures", default="results/figures")
    an.add_argument("--tuned", default=None)
    an.add_argument("--smoke", action="store_true", help=f"read and write everything inside {SMOKE_OUT}")
    args = ap.parse_args(argv)
    if args.smoke:
        args.out = SMOKE_OUT
        if args.command == "run":
            args.runs, args.budget_scale = 1, 0.05
        else:
            args.tables = args.figures = SMOKE_OUT
    if args.tuned is None:
        args.tuned = f"{args.out}/tuned_params.json" if args.smoke else "experiments/tuned_params.json"
    return args


def main(argv=None) -> Path:
    """Entry point; returns the output directory."""
    args = parse_args(argv)
    out_dir = Path(args.out)
    if args.command == "analyze":
        analyze(out_dir, Path(args.tables), Path(args.figures), Path(args.tuned))
        return out_dir

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    instances = [i.strip() for i in args.instances.split(",") if i.strip()]
    unknown = [a for a in algos if a not in ALGORITHMS or a not in GRID]
    if unknown:
        raise SystemExit(f"unknown algorithms: {unknown}")
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
        tasks = build_validation_tasks(instances, tuned, args.runs, args.budget_scale, commit)
        run_tasks(tasks, out_dir / "validation_runs.csv", args.workers)
    else:
        tasks = build_tasks(instances, algos, args.runs, args.budget_scale, commit)
        run_tasks(tasks, out_dir / "runs.csv", args.workers)
    return out_dir


if __name__ == "__main__":
    main()
