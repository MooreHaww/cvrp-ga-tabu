# Original code
"""Step 6c: analysis of the main run (course brief, Task 7).

    python -m experiments.analysis [--raw results/raw] [--out results]
                                   [--tuning-dirs results/tuning,results/tuning2]

Reads <raw>/runs.csv, <raw>/progress.csv and <raw>/solutions/ (read-only) and writes
  <out>/tables/{main_results,cw_baseline,tiny_results,wilcoxon,team_crossings}.{csv,md}
  <out>/figures/convergence_<instance>.png, boxplots_final.png, team_routes_<algo>.png

Team reference = the best feasible team cost in the main run AND the tuning rounds; it is
the best solution KNOWN TO US, not a proven optimum. All outputs are deterministic
(fixed orders, rounded floats, LF line endings).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import warnings
from pathlib import Path

import numpy as np

from cvrp.instance import Instance
from cvrp.parser import load_instance
from cvrp.savings import clarke_wright
from cvrp.solution import total_cost
from experiments.config import INSTANCE_ORDER, instance_path
from experiments.solutions import best_solution, load_all
from experiments.tuning import read_rows, team_reference

ALGOS = ("ga", "ga_ls", "tabu")
ALGO_LABEL = {"ga": "GA", "ga_ls": "GA+LS (memetic)", "tabu": "Tabu Search"}
ALGO_COLOUR = {"ga": "tab:blue", "ga_ls": "tab:green", "tabu": "tab:red"}     # same in every figure
PAIRS = (("ga_ls", "tabu"), ("ga_ls", "ga"), ("tabu", "ga"))
BENCHMARK_AND_TEAM = ("A-n32-k5", "A-n45-k6", "A-n60-k9", "A-n80-k10", "team")   # Holm family per pair
CONVERGENCE_INSTANCES = ("A-n45-k6", "A-n60-k9", "A-n80-k10", "team")
TEAM_REF_LABEL = "best known to us (not proven optimal)"
HIT_TOL = 1e-6                  # |cost - reference| below this counts as a hit
DEFAULT_TUNING_DIRS = "results/tuning,results/tuning2"


# --------------------------------------------------------------------------- reading

def _feasible(row: dict) -> bool:
    return str(row["feasible"]) == "True"


def _ordered(names, order) -> list[str]:
    """Names in the given order first, then any others sorted. Time: O(k log k)."""
    names = set(names)
    return [n for n in order if n in names] + sorted(names - set(order))


def cell(rows: list[dict], instance: str, algo: str) -> list[dict]:
    """Rows of one (instance, algo), sorted by run. Time: O(rows)."""
    return sorted((r for r in rows if r["instance"] == instance and r["algo"] == algo), key=lambda r: int(r["run"]))


def reference(rows: list[dict], instance: str, team_ref: float | None) -> tuple[float | None, str]:
    """(reference cost, its type): BKS (Augerat), optimum (tiny) or the team reference. Time: O(rows)."""
    if instance == "team":
        return team_ref, TEAM_REF_LABEL
    value = next((r["bks_or_opt"] for r in rows if r["instance"] == instance and r["bks_or_opt"] != ""), None)
    return (float(value) if value is not None else None), ("optimum" if instance.startswith("tiny_") else "BKS")


def progress_by_run(progress: list[dict]) -> dict[tuple[str, str, int], list[tuple[float, float]]]:
    """Best-so-far curve of every run: (instance, algo, run) -> [(elapsed_s, best_cost), ...]
    sorted by time. Time: O(P log P)."""
    curves: dict[tuple[str, str, int], list[tuple[float, float]]] = {}
    for r in progress:
        curves.setdefault((r["instance"], r["algo"], int(r["run"])), []).append(
            (float(r["elapsed_s"]), float(r["best_cost"])))
    return {k: sorted(v) for k, v in curves.items()}


# --------------------------------------------------------------------------- tables

def main_results(rows: list[dict], team_ref: float | None) -> list[dict]:
    """Per instance x algo: n, failures, best, mean, median, std (ddof 1), mean gap % to the
    reference, hits (|cost - ref| < 1e-6) and mean runtime. Time: O(rows * #cells)."""
    out = []
    for inst in _ordered({r["instance"] for r in rows}, INSTANCE_ORDER):
        ref, ref_type = reference(rows, inst, team_ref)
        for algo in _ordered({r["algo"] for r in rows if r["instance"] == inst}, ALGOS):
            rs = cell(rows, inst, algo)
            costs = [float(r["cost"]) for r in rs if _feasible(r)]
            mean = statistics.fmean(costs) if costs else math.nan
            out.append({
                "instance": inst, "algo": algo, "ref": ref, "ref_type": ref_type, "n": len(rs),
                "failures": len(rs) - len(costs),
                "best": min(costs, default=math.nan), "mean": mean,
                "median": statistics.median(costs) if costs else math.nan,
                "std": statistics.stdev(costs) if len(costs) >= 2 else math.nan,
                "mean_gap_pct": 100.0 * (mean - ref) / ref if ref else math.nan,
                "hits": sum(abs(c - ref) < HIT_TOL for c in costs) if ref else "",
                "mean_runtime_s": statistics.fmean(float(r["elapsed_s"]) for r in rs)})
    return out


def cw_cost(instance: str) -> tuple[float, int]:
    """Clarke-Wright cost and route count of an instance (cvrp.savings). Time: O(n^2 log n)."""
    res = clarke_wright(load_instance(instance_path(instance)), warn=False)
    return res.cost, res.n_routes


def cw_baseline(rows: list[dict]) -> list[dict]:
    """Per instance: CW cost and, per algo, the mean improvement over CW in % and the number
    of runs strictly below CW. Shows whether an algorithm leaves its CW seed at all.
    Time: O(#instances * CW + rows)."""
    out = []
    for inst in _ordered({r["instance"] for r in rows}, INSTANCE_ORDER):
        cw, routes = cw_cost(inst)
        row = {"instance": inst, "cw_cost": cw, "cw_routes": routes}
        for algo in ALGOS:
            costs = [float(r["cost"]) for r in cell(rows, inst, algo) if _feasible(r)]
            row[f"{algo}_mean_impr_pct"] = 100.0 * (cw - statistics.fmean(costs)) / cw if costs else math.nan
            row[f"{algo}_runs_below_cw"] = sum(c < cw - HIT_TOL for c in costs) if costs else ""
        out.append(row)
    return out


def tiny_results(rows: list[dict], curves: dict) -> list[dict]:
    """Per tiny instance x algo: optimum hit rate and the mean / max time-to-best, where
    time-to-best = elapsed time of the LAST improvement record of a run in progress.csv
    (the moment the final cost was first reached). Time: O(rows)."""
    out = []
    for inst in _ordered({r["instance"] for r in rows if r["instance"].startswith("tiny_")}, INSTANCE_ORDER):
        opt, _ = reference(rows, inst, None)
        for algo in _ordered({r["algo"] for r in rows if r["instance"] == inst}, ALGOS):
            rs = cell(rows, inst, algo)
            hits = sum(_feasible(r) and abs(float(r["cost"]) - opt) < HIT_TOL for r in rs)
            ttb = [curves[(inst, algo, int(r["run"]))][-1][0] for r in rs if (inst, algo, int(r["run"])) in curves]
            out.append({"instance": inst, "algo": algo, "optimum": opt, "n": len(rs), "hit_rate": hits / len(rs),
                        "mean_time_to_best_s": statistics.fmean(ttb) if ttb else math.nan,
                        "max_time_to_best_s": max(ttb, default=math.nan)})
    return out


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order:
    p_(k) -> max over j <= k of min(1, (m - j + 1) * p_(j)) for the sorted p-values. Time: O(m log m)."""
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired_costs(rows: list[dict], instance: str, a: str, b: str) -> tuple[list[float], list[float]]:
    """Costs of algorithms a and b paired by run index (same seed is asserted). Time: O(rows)."""
    ra, rb = cell(rows, instance, a), cell(rows, instance, b)
    if [(r["run"], r["seed"]) for r in ra] != [(r["run"], r["seed"]) for r in rb]:
        raise ValueError(f"{instance}: runs/seeds of {a} and {b} do not match; cannot pair them")
    return [float(r["cost"]) for r in ra], [float(r["cost"]) for r in rb]


def wilcoxon_table(rows: list[dict]) -> list[dict]:
    """Two-sided Wilcoxon signed-rank test per instance and pair, paired by run index.
    d = cost_a - cost_b (negative = a better). Zero differences are dropped ('wilcox');
    if every d is zero the row reports 'all ties' and the test is not called. Holm-adjusted
    p-values per pair over the benchmark + team instances that were tested.
    Time: O(#instances * #pairs * runs log runs)."""
    from scipy.stats import wilcoxon

    instances = _ordered({r["instance"] for r in rows}, INSTANCE_ORDER)
    out = []
    for a, b in PAIRS:
        block = []
        for inst in instances:
            if not (cell(rows, inst, a) and cell(rows, inst, b)):
                continue
            xa, xb = paired_costs(rows, inst, a, b)
            d = [round(x - y, 9) for x, y in zip(xa, xb)]   # rounding: float noise is not a difference
            nonzero = sum(v != 0 for v in d)
            row = {"instance": inst, "pair": f"{a} vs {b}", "n_pairs": len(d), "n_nonzero": nonzero,
                   "median_diff": statistics.median(d), "result": "all ties", "statistic": None, "p": None,
                   "p_holm": None}
            if nonzero:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")              # small-sample / ties notices
                    res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
                row.update(result="tested", statistic=float(res.statistic), p=float(res.pvalue))
            block.append(row)
        family = [r for r in block if r["instance"] in BENCHMARK_AND_TEAM and r["p"] is not None]
        for r, adj in zip(family, holm([r["p"] for r in family])):
            r["p_holm"] = adj
        out += block
    return out


# --------------------------------------------------------------------------- team: strait crossings

def leg_crossings(inst: Instance, route: list[int]) -> int:
    """Legs of 0 -> route -> 0 whose endpoints have different side labels, i.e. how many
    times the penalty P is charged (bosphorus_matrix: + P iff side(i) != side(j)). Time: O(len(route))."""
    path = [0] + list(route) + [0]
    return sum(inst.side[u] != inst.side[v] for u, v in zip(path, path[1:]))


def euclidean_length(inst: Instance, routes: list[list[int]]) -> float:
    """Unpenalised Euclidean length of all routes (depot legs included). Time: O(n)."""
    total = 0.0
    for route in routes:
        path = [0] + list(route) + [0]
        for u, v in zip(path, path[1:]):
            total += float(np.hypot(*(inst.coords[u] - inst.coords[v])))
    return total


def crossing_stats(inst: Instance, routes: list[list[int]]) -> dict:
    """(i) routes with at least one E customer, (ii) side changes over all legs, (iii) penalty
    share of the cost. Checks cost == Euclidean length + crossings * P. Time: O(n)."""
    cost = total_cost(inst, routes)
    crossings = sum(leg_crossings(inst, r) for r in routes)
    if abs(euclidean_length(inst, routes) + crossings * inst.penalty - cost) > 1e-6:
        raise ValueError("cost != Euclidean length + crossings * P: crossing count does not match the cost formula")
    return {"cost": cost, "n_routes": len(routes),
            "routes_with_E": sum(any(inst.side[v] == "E" for v in r) for r in routes),
            "crossings": crossings, "penalty_share_pct": 100.0 * crossings * inst.penalty / cost}


def team_crossings(raw: Path, inst: Instance) -> list[dict]:
    """Per algo: means of the crossing statistics over its team runs, and the values of its
    best run. Every stored cost is re-checked against the instance. Time: O(runs * n)."""
    records = [s for s in load_all(raw) if s["instance"] == "team" and s["feasible"] and s["routes"]]
    out = []
    for algo in _ordered({s["algo"] for s in records}, ALGOS):
        stats = []
        for s in (s for s in records if s["algo"] == algo):
            st = crossing_stats(inst, s["routes"])
            if abs(st["cost"] - s["cost"]) > 1e-6:
                raise ValueError(f"team {algo} run {s['run']}: stored cost {s['cost']} != recomputed {st['cost']}")
            stats.append(st)
        best = best_solution(raw, "team", algo)
        b = crossing_stats(inst, best["routes"])
        out.append({"algo": algo, "n_runs": len(stats),
                    "mean_routes": statistics.fmean(s["n_routes"] for s in stats),
                    "mean_routes_with_E": statistics.fmean(s["routes_with_E"] for s in stats),
                    "mean_crossings": statistics.fmean(s["crossings"] for s in stats),
                    "min_crossings": min(s["crossings"] for s in stats),
                    "max_crossings": max(s["crossings"] for s in stats),
                    "mean_penalty_share_pct": statistics.fmean(s["penalty_share_pct"] for s in stats),
                    "best_run": best["run"], "best_cost": b["cost"], "best_routes": b["n_routes"],
                    "best_routes_with_E": b["routes_with_E"], "best_crossings": b["crossings"],
                    "best_penalty_share_pct": b["penalty_share_pct"]})
    return out


# --------------------------------------------------------------------------- writing tables

def _csv_value(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return round(v, 6) if isinstance(v, float) else v


def _md_value(v, fmt: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return "–"
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


def write_table(rows: list[dict], columns: list[str], stem: str, tables_dir: Path,
                formats: dict[str, str], title: str, note: str = "") -> list[Path]:
    """Write <stem>.csv (LF endings, floats rounded to 6 decimals) and <stem>.md (fixed
    formats per column, default '.3f'). Time: O(rows * columns)."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = tables_dir / f"{stem}.csv", tables_dir / f"{stem}.md"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for r in rows:
            writer.writerow([_csv_value(r[c]) for c in columns])
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join(_md_value(r[c], formats.get(c, ".3f")) for c in columns) + " |")
    if note:
        lines += ["", note]
    md_path.write_text("\n".join(lines) + "\n", newline="\n")
    return [csv_path, md_path]


MAIN_COLUMNS = ["instance", "algo", "ref", "ref_type", "n", "failures", "best", "mean", "median", "std",
                "mean_gap_pct", "hits", "mean_runtime_s"]
CW_COLUMNS = ["instance", "cw_cost", "cw_routes"] + [f"{a}_{k}" for a in ALGOS
                                                     for k in ("mean_impr_pct", "runs_below_cw")]
TINY_COLUMNS = ["instance", "algo", "optimum", "n", "hit_rate", "mean_time_to_best_s", "max_time_to_best_s"]
WILCOXON_COLUMNS = ["instance", "pair", "n_pairs", "n_nonzero", "result", "statistic", "p", "p_holm", "median_diff"]
CROSSING_COLUMNS = ["algo", "n_runs", "mean_routes", "mean_routes_with_E", "mean_crossings", "min_crossings",
                    "max_crossings", "mean_penalty_share_pct", "best_run", "best_cost", "best_routes",
                    "best_routes_with_E", "best_crossings", "best_penalty_share_pct"]
COST_FMT = ".2f"
P_FMT = ".4g"

WILCOXON_NOTE = (
    "d = cost(first) - cost(second), paired by run index (same seed); negative median_diff = the first "
    "algorithm is better. Two-sided Wilcoxon signed-rank test (scipy.stats.wilcoxon, zero differences dropped). "
    "'all ties' = every paired difference is 0, so the test is not defined and was not called. p_holm = Holm "
    "adjustment per pair over the benchmark + team instances that were tested.\n\n"
    "Limitation: only 10 pairs per test. The smallest possible exact two-sided p is 2/2^10 = 0.00195, and with "
    "fewer non-zero pairs it is much larger (5 non-zero pairs: 0.0625). A large p is weak evidence of 'no "
    "difference'; the test says nothing about the size of the difference (see median_diff and main_results).")


# --------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")                     # file output only
    import matplotlib.pyplot as plt
    return plt


def median_curve(curves: list[list[tuple[float, float]]], grid: np.ndarray) -> tuple[np.ndarray, ...]:
    """Median, 25th and 75th percentile of step-function best-so-far curves on a time grid
    (value at t = last record with time <= t). Time: O(runs * (len(grid) + records))."""
    values = np.empty((len(curves), len(grid)))
    for i, curve in enumerate(curves):
        times = np.array([t for t, _ in curve])
        costs = np.array([c for _, c in curve])
        idx = np.searchsorted(times, grid, side="right") - 1
        values[i] = np.where(idx >= 0, costs[np.maximum(idx, 0)], np.nan)
    return np.median(values, axis=0), np.percentile(values, 25, axis=0), np.percentile(values, 75, axis=0)


def plot_convergence(rows: list[dict], curves: dict, instance: str, cw: float, ref: float | None,
                     ref_label: str, out_path: Path) -> Path:
    """Median best-so-far cost vs time (log x) with the IQR band, one line per algorithm.
    Each line starts when ALL runs of that algorithm have a feasible solution."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for algo in ALGOS:
        rs = cell(rows, instance, algo)
        run_curves = [curves[(instance, algo, int(r["run"]))] for r in rs if (instance, algo, int(r["run"])) in curves]
        if not run_curves:
            continue
        budget = max(float(r["budget_s"]) for r in rs)
        start = max(max(c[0][0] for c in run_curves), 1e-3)
        grid = np.geomspace(start, budget, 300)
        med, q25, q75 = median_curve(run_curves, grid)
        ax.plot(grid, med, color=ALGO_COLOUR[algo], linewidth=2, label=f"{ALGO_LABEL[algo]} (median of {len(rs)})")
        ax.fill_between(grid, q25, q75, color=ALGO_COLOUR[algo], alpha=0.2, linewidth=0)
    ax.axhline(cw, color="grey", linestyle="--", linewidth=1.2, label=f"Clarke-Wright ({cw:.2f})")
    if ref is not None:
        name = "best known to us" if instance == "team" else "BKS"
        ax.axhline(ref, color="black", linestyle=":", linewidth=1.2, label=f"{name} ({ref:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("elapsed time (s, log scale)", fontsize=11)
    ax.set_ylabel("best-so-far cost", fontsize=11)
    ax.set_title(f"{instance}: convergence (median, band = interquartile range)", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_boxplots(rows: list[dict], team_ref: float | None, out_path: Path) -> Path:
    """Final-cost gap (%) per algorithm, one panel per benchmark + team instance; every run
    is also drawn as a point (many cells have identical values)."""
    plt = _plt()
    instances = [i for i in BENCHMARK_AND_TEAM if any(r["instance"] == i for r in rows)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, inst in zip(axes.flat, instances):
        ref, _ = reference(rows, inst, team_ref)
        algos = [a for a in ALGOS if cell(rows, inst, a)]
        data = [[100.0 * (float(r["cost"]) - ref) / ref for r in cell(rows, inst, a) if _feasible(r)] for a in algos]
        box = ax.boxplot(data, tick_labels=[ALGO_LABEL[a] for a in algos], widths=0.5, patch_artist=True,
                         showfliers=False, medianprops={"color": "black"})
        for patch, algo in zip(box["boxes"], algos):
            patch.set_facecolor(ALGO_COLOUR[algo])
            patch.set_alpha(0.35)
        for i, (gaps, algo) in enumerate(zip(data, algos), start=1):
            offsets = np.linspace(-0.12, 0.12, len(gaps)) if len(gaps) > 1 else np.zeros(1)
            ax.scatter(i + offsets, gaps, s=14, color=ALGO_COLOUR[algo], edgecolors="black", linewidths=0.3,
                       zorder=3)
        ax.set_title(inst, fontsize=11)
        ax.set_ylabel("gap to best known to us (%)" if inst == "team" else "gap to BKS (%)", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in list(axes.flat)[len(instances):]:
        ax.axis("off")
    fig.suptitle("Final cost per run (10 runs per algorithm; points = individual runs)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_team_routes(inst: Instance, solution: dict, out_path: Path) -> Path:
    """Route plot of one team solution: one colour per route (depot legs drawn), depot
    star, the Bosphorus line x = split_x and the side labels (as in cvrp.plot_instance)."""
    plt = _plt()
    st = crossing_stats(inst, solution["routes"])
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab20")
    for k, route in enumerate(solution["routes"]):
        path = [0] + route + [0]
        xy = inst.coords[path]
        ax.plot(xy[:, 0], xy[:, 1], color=cmap(k % 20), linewidth=1.3, alpha=0.9)
        ax.scatter(inst.coords[route, 0], inst.coords[route, 1], color=cmap(k % 20), s=22, edgecolors="black",
                   linewidths=0.3, zorder=3)
    ax.scatter(*inst.coords[0], marker="*", s=400, c="red", edgecolors="black", zorder=5, label="Depot (lab)")
    ax.axvline(inst.split_x, color="steelblue", linestyle="--", linewidth=1.5, label="Bosphorus (x = 500)")
    ax.text(inst.split_x / 2, 985, "European side (E)", ha="center", fontsize=11, color="dimgray")
    ax.text((inst.split_x + 1000) / 2, 985, "Asian side (A)", ha="center", fontsize=11, color="dimgray")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.set_title(f"{ALGO_LABEL[solution['algo']]}: best team solution (run {solution['run']})\n"
                 f"cost {st['cost']:.2f}, {st['n_routes']} routes, {st['crossings']} strait crossings "
                 f"(penalty share {st['penalty_share_pct']:.1f}%)", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- entry point

def analyze(raw: Path, out: Path, tuning_dirs: tuple[Path, ...] = ()) -> dict:
    """Build every table and figure of Step 6c from a raw results folder (read-only).
    Returns the table rows and the list of written files."""
    rows = read_rows(raw / "runs.csv")
    if not rows:
        raise SystemExit(f"no runs in {raw / 'runs.csv'}")
    curves = progress_by_run(read_rows(raw / "progress.csv"))
    ref_rows = [r for d in tuning_dirs for f in ("runs.csv", "validation_runs.csv") for r in read_rows(d / f)]
    team_ref = team_reference(rows + ref_rows)
    tables, figures = out / "tables", out / "figures"

    main = main_results(rows, team_ref)
    cw = cw_baseline(rows)
    tiny = tiny_results(rows, curves)
    wil = wilcoxon_table(rows)
    cost_fmt = {c: COST_FMT for c in ("ref", "best", "mean", "median", "std", "cw_cost", "optimum", "statistic",
                                      "median_diff", "best_cost")}
    written = write_table(main, MAIN_COLUMNS, "main_results", tables, {**cost_fmt, "mean_runtime_s": ".2f"},
                          "Main results (10 runs per instance and algorithm)",
                          f"Team reference = {team_ref:.2f}, the {TEAM_REF_LABEL}: best feasible team cost over the "
                          f"main run and both tuning rounds." if team_ref is not None else "")
    written += write_table(cw, CW_COLUMNS, "cw_baseline", tables, cost_fmt, "Improvement over Clarke-Wright",
                           "mean_impr_pct = 100 * (CW - mean cost) / CW; runs_below_cw = runs strictly below CW.")
    written += write_table(tiny, TINY_COLUMNS, "tiny_results", tables, {**cost_fmt, "mean_time_to_best_s": ".4f",
                                                                        "max_time_to_best_s": ".4f"},
                           "Tiny instances (exact optimum known)",
                           "time to best = elapsed time of the last improvement in progress.csv.")
    written += write_table(wil, WILCOXON_COLUMNS, "wilcoxon", tables, {**cost_fmt, "p": P_FMT, "p_holm": P_FMT},
                           "Wilcoxon signed-rank tests (paired by seed)", WILCOXON_NOTE)

    crossings, plots = [], []
    if any(r["instance"] == "team" for r in rows):
        team = load_instance(instance_path("team"))
        crossings = team_crossings(raw, team)
        written += write_table(crossings, CROSSING_COLUMNS, "team_crossings", tables,
                               {"best_cost": COST_FMT, "mean_routes": ".1f", "mean_routes_with_E": ".1f",
                                "mean_crossings": ".1f", "mean_penalty_share_pct": ".2f",
                                "best_penalty_share_pct": ".2f"},
                               "Team instance: Bosphorus crossings",
                               f"crossings = legs whose endpoints have different side labels (depot legs "
                               f"included) = number of times P = {team.penalty:g} is charged; penalty share = "
                               f"crossings * P / cost. Every route with an E customer crosses at least twice.")
        for algo in ("ga_ls", "tabu"):
            best = best_solution(raw, "team", algo)
            if best is not None:
                plots.append(plot_team_routes(team, best, figures / f"team_routes_{algo}.png"))

    cw_of = {r["instance"]: r["cw_cost"] for r in cw}
    for inst in CONVERGENCE_INSTANCES:
        if any(r["instance"] == inst for r in rows):
            ref, label = reference(rows, inst, team_ref)
            plots.append(plot_convergence(rows, curves, inst, cw_of[inst], ref, label,
                                          figures / f"convergence_{inst}.png"))
    if any(r["instance"] in BENCHMARK_AND_TEAM for r in rows):
        plots.append(plot_boxplots(rows, team_ref, figures / "boxplots_final.png"))
    return {"team_ref": team_ref, "main": main, "cw": cw, "tiny": tiny, "wilcoxon": wil, "crossings": crossings,
            "tables": written, "figures": sorted(plots)}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Step 6c: tables, tests and figures of the main run.")
    ap.add_argument("--raw", default="results/raw", help="raw results folder (read only)")
    ap.add_argument("--out", default="results", help="writes <out>/tables and <out>/figures")
    ap.add_argument("--tuning-dirs", default=DEFAULT_TUNING_DIRS,
                    help="comma-separated tuning folders also searched for the team reference")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    dirs = tuple(Path(d.strip()) for d in args.tuning_dirs.split(",") if d.strip())
    result = analyze(Path(args.raw), Path(args.out), dirs)
    print(f"team reference ({TEAM_REF_LABEL}) = {result['team_ref']:.2f}" if result["team_ref"] else "no team runs")
    for path in result["tables"]:
        print(f"table  {path}")
    for path in result["figures"]:
        print(f"figure {path}")
    return result


if __name__ == "__main__":
    main()
