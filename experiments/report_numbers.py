# Original code
"""Step 7a: the report numbers sheet, docs/report_numbers.md.

Every number in the sheet is read from a file:
  - results/raw, results/timing and results/tables;
  - the .vrp files in data/;
  - experiments/tuned_params.json;
  - the budgets in experiments/config.py and the parameter dataclasses (GAParams, TabuParams).
Nothing is typed by hand, so a number in the report can always be traced back to its source.

The output is deterministic: fixed order, fixed number formats and LF newlines, so two runs on the same inputs give the
same bytes. The commit hash of HEAD and the package versions of the environment are passed in by main(), so the
tests can fix them.

CLI:  python -m experiments.report_numbers [--root .] [--out docs/report_numbers.md]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import MISSING, fields
from importlib import metadata
from pathlib import Path

from cvrp.ga import GAParams
from cvrp.parser import load_instance
from cvrp.tabu import TabuParams
from experiments import config
from experiments.timing import N_FIT_MIN

ALGOS = ("ga", "ga_ls", "tabu")
ALPHA = 0.05                                   # significance level for the Holm-adjusted p values
PACKAGES = ("numpy", "matplotlib", "scipy", "pytest")

# (file stem, folder, title) of every table copied into the sheet
TABLES = (
    ("main_results", "results/tables", "Main results"),
    ("cw_baseline", "results/tables", "Improvement over Clarke–Wright"),
    ("wilcoxon", "results/tables", "Wilcoxon signed-rank tests (Holm-adjusted)"),
    ("team_crossings", "results/tables", "Team instance: Bosphorus crossings"),
    ("tiny_results", "results/tables", "Tiny instances (exact optimum known)"),
    ("complexity_fits", "results/tables", "Complexity validation: log-log slopes"),
    ("share_ga_ls", "results/timing", "GA+LS cost shares (20 generations)"),
)

# (report item, source file, command that writes it)
SOURCES = (
    ("Main results table", "results/tables/main_results.csv", "python -m experiments.analysis"),
    ("Clarke–Wright baseline table", "results/tables/cw_baseline.csv", "python -m experiments.analysis"),
    ("Wilcoxon table", "results/tables/wilcoxon.csv", "python -m experiments.analysis"),
    ("Team crossings table", "results/tables/team_crossings.csv", "python -m experiments.analysis"),
    ("Tiny instances table", "results/tables/tiny_results.csv", "python -m experiments.analysis"),
    ("Convergence figures", "results/figures/convergence_<instance>.png", "python -m experiments.analysis"),
    ("Boxplots of final costs", "results/figures/boxplots_final.png", "python -m experiments.analysis"),
    ("Team route maps", "results/figures/team_routes_{ga_ls,tabu}.png", "python -m experiments.analysis"),
    ("Complexity fits table", "results/tables/complexity_fits.csv", "python -m experiments.timing"),
    ("Complexity figures", "results/figures/complexity_<component>.png", "python -m experiments.timing"),
    ("Split vs L figure", "results/figures/split_vs_L.png", "python -m experiments.timing"),
    ("Delta vs full evaluation figure", "results/figures/delta_vs_full.png", "python -m experiments.timing"),
    ("GA+LS cost shares", "results/timing/share_ga_ls.csv", "python -m experiments.timing"),
    ("Raw timings", "results/timing/timing.csv", "python -m experiments.timing"),
    ("Tuning round 1 tables", "results/tables/tuning_{summary,selection,validation}.csv",
     "python -m experiments.tuning analyze"),
    ("Tuning round 2 tables", "results/tables/tuning2_{summary,selection,validation}.csv",
     "python -m experiments.tuning analyze --round 2"),
    ("Sensitivity figures", "results/figures/sensitivity_<algo>_<param>.png", "python -m experiments.tuning analyze"),
    ("Tabu configurations (round 2)", "results/figures/tuning2_tabu_configs.png",
     "python -m experiments.tuning analyze --round 2"),
    ("Tuned parameters", "experiments/tuned_params.json", "python -m experiments.tuning analyze --round 2"),
    ("Team instance plot", "results/figures/team_instance.png",
     "python -m cvrp.plot_instance data/team/team_n100_seed14835.vrp --out results/figures/team_instance.png"),
    ("Raw per-run results", "results/raw/runs.csv, progress.csv, solutions/",
     "python -m experiments.run_experiments --algos ga,ga_ls,tabu --params-file experiments/tuned_params.json "
     "--workers 3"),
    ("This sheet", "docs/report_numbers.md", "python -m experiments.report_numbers"),
)


# --------------------------------------------------------------------------- reading

def read_csv(path: Path) -> list[dict]:
    """All rows of a CSV file, as strings; [] if the file does not exist. Time: O(rows)."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_kv(path: Path) -> dict[str, str]:
    """'key: value' lines (hardware.txt) as a dict; {} if missing. Time: O(lines)."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def unique(rows: list[dict], column: str) -> list[str]:
    """Sorted distinct values of one column. Time: O(rows log rows)."""
    return sorted({r[column] for r in rows if r.get(column)})


def instance_file(root: Path, name: str) -> Path:
    """The .vrp file of an instance, below `root`, using the same naming as experiments/config.py. Time: O(1)."""
    return root / config.instance_path(name).relative_to(config.ROOT)


# --------------------------------------------------------------------------- formatting

def md_table(columns: list[str], rows: list[list]) -> list[str]:
    """A markdown table; empty cells become '–' and '|' is escaped. Time: O(cells)."""
    def cell(v) -> str:
        text = "–" if v is None or v == "" else str(v)
        return text.replace("|", "\\|")
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return lines


def value_text(v) -> str:
    """A parameter value as text: lists joined by commas, everything else str(). Time: O(len(v))."""
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def join_names(names: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'. Time: O(len(names))."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def two_point_slope(x0: float, y0: float, x1: float, y1: float) -> float:
    """Slope of log y against log x between two points. Time: O(1)."""
    return math.log(y1 / y0) / math.log(x1 / x0)


# --------------------------------------------------------------------------- sections

def section_setup(root: Path, runs: list[dict], timing: list[dict], head: str, env: dict[str, str]) -> list[str]:
    """Commits, hardware, versions, seeds, budgets, workers and run counts. Time: O(rows)."""
    hw_main = read_kv(root / "results/raw/hardware.txt")
    hw_timing = read_kv(root / "results/timing/hardware.txt")
    seeds = sorted({int(r["seed"]) for r in runs})
    budgets_runs = {}
    for r in runs:
        budgets_runs.setdefault(r["instance"], set()).add(r["budget_s"])
    lines = ["## Setup", "",
             f"- Main run commit (git_commit in results/raw/runs.csv): {', '.join(unique(runs, 'git_commit'))}",
             f"- Timing run commit (git_commit in results/timing/timing.csv): "
             f"{', '.join(unique(timing, 'git_commit'))}",
             f"- HEAD when this sheet was generated (git log): {head}",
             f"- Runs in results/raw/runs.csv: {len(runs)} "
             f"({', '.join(f'{a} {sum(r['algo'] == a for r in runs)}' for a in unique(runs, 'algo'))})",
             f"- Seeds used by the main run: {', '.join(str(s) for s in seeds)} "
             f"(experiments/config.py: BASE_SEED = {config.BASE_SEED}, run r uses BASE_SEED + r)",
             f"- Workers: main run {hw_main.get('workers', '–')}, timing run {hw_timing.get('workers', '–')}",
             "", "### Hardware and software (hardware.txt of each run)", ""]
    keys = sorted(set(hw_main) | set(hw_timing))
    lines += md_table(["key", "main run", "timing run"], [[k, hw_main.get(k), hw_timing.get(k)] for k in keys])
    lines += ["", "### Package versions of the environment that generated this sheet", ""]
    lines += md_table(["package", "version"], [[k, env[k]] for k in sorted(env)])
    lines += ["", "### Time budgets per run (seconds)", ""]
    names = [n for n in config.INSTANCE_ORDER if n in budgets_runs]
    lines += md_table(["instance", "budget in experiments/config.py", "budget_s in runs.csv"],
                      [[n, config.BUDGETS[n], ", ".join(sorted(budgets_runs[n]))] for n in names])
    return lines


def section_instances(root: Path, runs: list[dict], main_rows: list[dict]) -> list[str]:
    """n, Q, K, total demand and reference of every instance in the main run; team extras. Time: O(sum n)."""
    ref_of = {r["instance"]: (r["ref"], r["ref_type"]) for r in main_rows}
    names = [n for n in config.INSTANCE_ORDER if n in {r["instance"] for r in runs}]
    rows, team = [], None
    for name in names:
        inst = load_instance(instance_file(root, name))
        ref, ref_type = ref_of.get(name, ("", ""))
        total = int(inst.demand.sum())
        rows.append([name, inst.name, inst.n, inst.Q, inst.K, total, f"{total / inst.Q:.2f}", ref, ref_type,
                     inst.distance_type])
        if name == "team":
            team = inst
    lines = ["## Instances", "",
             "Read from the .vrp files; the reference and its type come from results/tables/main_results.csv.", ""]
    lines += md_table(["instance", "file name", "n", "Q", "K", "total demand", "total demand / Q", "reference",
                       "reference type", "distance"], rows)
    if team is not None:
        sides = team.side[1:]                          # customers 1..n; vertex 0 is the depot
        seed = re.search(r"seed(\d+)", team.name)
        lines += ["", "### Team instance", "",
                  f"- File: {instance_file(root, 'team').relative_to(root).as_posix()}",
                  f"- Seed: {seed.group(1) if seed else '–'} (from the file name)",
                  f"- Bosphorus penalty P = {team.penalty:g}, split_x = {team.split_x:g} "
                  f"(side E if x < split_x, else A; the depot is on side {team.side[0]})",
                  f"- Customers on side E: {sides.count('E')}, on side A: {sides.count('A')}",
                  f"- Depot at ({team.coords[0][0]:g}, {team.coords[0][1]:g})"]
    return lines


def default_params(algo: str) -> dict:
    """Default parameters of an algorithm variant: the dataclass defaults, plus GA_LS_BASE for ga_ls. Time: O(1)."""
    cls = TabuParams if algo == "tabu" else GAParams
    out = {f.name: f.default for f in fields(cls) if f.default is not MISSING}
    if algo == "ga":
        out["memetic"] = False
    if algo == "ga_ls":
        out.update(config.GA_LS_BASE)
    return out


def section_params(root: Path, runs: list[dict]) -> list[str]:
    """Default, tuned (tuned_params.json) and used (params_json of the main run) values side by side.
    Time: O(rows + parameters)."""
    path = root / "experiments/tuned_params.json"
    tuned = json.loads(path.read_text()) if path.exists() else {}
    lines = ["## Parameters", "",
             "default = the dataclass default (GAParams, TabuParams; ga_ls adds GA_LS_BASE from experiments/config.py); "
             "tuned = experiments/tuned_params.json; used = params_json of the first main run of that algorithm "
             "(results/raw/runs.csv). Protocol fields (seed, time limit, iteration caps) are left out.", ""]
    order = {n: i for i, n in enumerate(config.INSTANCE_ORDER)}
    for algo in [a for a in ALGOS if any(r["algo"] == a for r in runs)]:
        first = min((r for r in runs if r["algo"] == algo), key=lambda r: (order.get(r["instance"], 99), int(r["run"])))
        used = json.loads(first["params_json"])
        default, tuned_algo = default_params(algo), tuned.get(algo, {})
        rows = []
        for name in sorted(set(default) | set(used)):
            if name in config.PROTECTED:
                continue
            expected = tuned_algo.get(name, default.get(name))
            matches = "yes" if name in used and value_text(used[name]) == value_text(expected) else "no"
            rows.append([name, value_text(default.get(name, "–")),
                         value_text(tuned_algo[name]) if name in tuned_algo else "", value_text(used.get(name, "–")),
                         matches])
        lines += [f"### {algo}", ""] + md_table(["parameter", "default", "tuned", "used", "used = tuned or default"],
                                                rows) + [""]
    return lines[:-1]


def section_tables(root: Path) -> list[str]:
    """Every result table, copied cell by cell from its CSV. Time: O(cells)."""
    lines = ["## Tables (copied from the CSV files)"]
    for stem, folder, title in TABLES:
        path = root / folder / f"{stem}.csv"
        lines += ["", f"### {title}", "", f"Source: {folder}/{stem}.csv", ""]
        rows = read_csv(path)
        if not rows:
            lines.append("(file missing or empty)")
            continue
        columns = list(rows[0])
        lines += md_table(columns, [[r[c] for c in columns] for r in rows])
    return lines


# --------------------------------------------------------------------------- key findings

def finding_best_by_mean(main_rows: list[dict]) -> list[str]:
    """One sentence per instance: the lowest mean cost and the algorithms that reach it. Time: O(rows)."""
    out = []
    for name in sorted({r["instance"] for r in main_rows}, key=lambda n: config.INSTANCE_ORDER.index(n)
                       if n in config.INSTANCE_ORDER else 99):
        cell = [r for r in main_rows if r["instance"] == name]
        best = min(float(r["mean"]) for r in cell)
        algos = [r["algo"] for r in cell if float(r["mean"]) == best]
        out.append(f"{name}: the lowest mean cost is {best:.2f}, reached by {join_names(algos)}.")
    return out


def finding_hits(main_rows: list[dict]) -> list[str]:
    """One sentence per reference type: hits of the reference per algorithm. Time: O(rows)."""
    out = []
    for ref_type in sorted({r["ref_type"] for r in main_rows}):
        rows = [r for r in main_rows if r["ref_type"] == ref_type]
        names = sorted({r["instance"] for r in rows})
        parts = []
        for algo in sorted({r["algo"] for r in rows}):
            mine = [r for r in rows if r["algo"] == algo]
            parts.append(f"{algo} {sum(int(r['hits']) for r in mine)} of {sum(int(r['n']) for r in mine)}")
        out.append(f"Reference type '{ref_type}' on {len(names)} instance(s) ({', '.join(names)}): the runs that "
                   f"reach it are {', '.join(parts)}.")
    return out


def finding_ga_vs_cw(cw_rows: list[dict], main_rows: list[dict]) -> list[str]:
    """The range of the plain GA's mean improvement over Clarke–Wright. Time: O(rows)."""
    if not cw_rows:
        return []
    lo = min(cw_rows, key=lambda r: float(r["ga_mean_impr_pct"]))
    hi = max(cw_rows, key=lambda r: float(r["ga_mean_impr_pct"]))
    below = sum(int(r["ga_runs_below_cw"]) for r in cw_rows)
    total = sum(int(r["n"]) for r in main_rows if r["algo"] == "ga" and r["instance"] in {c["instance"] for c in cw_rows})
    return [f"The plain GA improves on Clarke–Wright by {float(lo['ga_mean_impr_pct']):.2f}% ({lo['instance']}) to "
            f"{float(hi['ga_mean_impr_pct']):.2f}% ({hi['instance']}) on average, and ends strictly below CW in "
            f"{below} of {total} runs."]


def finding_wilcoxon(rows: list[dict]) -> list[str]:
    """One sentence per pair: where the Holm-adjusted p is below ALPHA, where not, where all ties. Time: O(rows)."""
    out = []
    for pair in sorted({r["pair"] for r in rows}):
        mine = [r for r in rows if r["pair"] == pair]
        sig = [r for r in mine if r["result"] == "tested" and r["p_holm"] and float(r["p_holm"]) < ALPHA]
        not_sig = [r["instance"] for r in mine if r["result"] == "tested" and r not in sig]
        ties = [r["instance"] for r in mine if r["result"] == "all ties"]
        sig_text = ", ".join(f"{r['instance']} (p_holm = {r['p_holm']}, median diff = {float(r['median_diff']):.2f})"
                             for r in sig) or "none"
        out.append(f"{pair}: significant after Holm at alpha = {ALPHA} on {sig_text}; not significant on "
                   f"{', '.join(not_sig) or 'none'}; all ties on {', '.join(ties) or 'none'}.")
    return out


def finding_crossings(rows: list[dict]) -> list[str]:
    """Whether every team solution has the same number of strait crossings. Time: O(rows)."""
    if not rows:
        return []
    values = {r["min_crossings"] for r in rows} | {r["max_crossings"] for r in rows}
    algos = ", ".join(r["algo"] for r in rows)
    runs = sum(int(r["n_runs"]) for r in rows)
    if len(values) == 1:
        v = values.pop()
        e_routes = sorted({r["mean_routes_with_E"] for r in rows})
        return [f"All {runs} team solutions ({algos}) cross the strait exactly {v} times; the mean number of routes "
                f"with an E customer is {', '.join(e_routes)}, and every such route crosses at least twice."]
    return [f"The team solutions ({algos}) cross the strait between {min(values, key=float)} and "
            f"{max(values, key=float)} times."]


def _median(rows: list[dict], component: str, n: int, column: str) -> float:
    return statistics.median(float(r[column]) for r in rows if r["component"] == component and int(r["n"]) == n)


def finding_slopes(fits: list[dict], timing: list[dict]) -> list[str]:
    """GA generation slope, and the Tabu iteration slope split into moves per iteration and time per move.
    Two-point slopes use the medians at the smallest n >= N_FIT_MIN and the largest n. Time: O(rows)."""
    fit = {r["component"]: r for r in fits}
    out = []
    for comp, label in (("ga_generation", "GA generation"), ("tabu_iteration", "Tabu Search iteration")):
        ns = sorted({int(r["n"]) for r in timing if r["component"] == comp})
        big = [n for n in ns if n >= N_FIT_MIN]
        if comp not in fit or len(big) < 2:
            continue
        lo, hi = big[0], big[-1]
        t0, t1 = _median(timing, comp, lo, "seconds"), _median(timing, comp, hi, "seconds")
        total = two_point_slope(lo, t0, hi, t1)
        text = (f"{label}: fitted slope (n >= {N_FIT_MIN}) {float(fit[comp]['fit_n100_slope']):.2f} against a "
                f"theory slope of {float(fit[comp]['theory_slope']):.2f}; the two-point slope from n = {lo} to {hi} "
                f"is {total:.2f}")
        if comp == "tabu_iteration":
            m0, m1 = _median(timing, comp, lo, "units_of_work"), _median(timing, comp, hi, "units_of_work")
            moves = two_point_slope(lo, m0, hi, m1)
            text += (f", of which {moves:.2f} comes from the moves evaluated per iteration ({m0 / lo:.1f} -> "
                     f"{m1 / hi:.1f} per customer) and {total - moves:.2f} from the time per move "
                     f"({t0 / m0 * 1e9:.0f} -> {t1 / m1 * 1e9:.0f} ns)")
        out.append(text + ".")
    return out


def finding_shares(rows: list[dict]) -> list[str]:
    """Share of the GA+LS run time spent in the descent and in Split decoding. Time: O(rows)."""
    if not rows:
        return []
    parts = [f"{100 * float(r['descent_share']):.1f}% of the run time on {r['instance']} (Split decode "
             f"{100 * float(r['split_decode_share']):.1f}%)" for r in rows]
    return [f"In GA+LS the descent takes {join_names(parts)}."]


def section_findings(root: Path, timing: list[dict]) -> list[str]:
    """Every key finding as a sentence computed from the tables. Time: O(rows)."""
    tables = root / "results/tables"
    main_rows = read_csv(tables / "main_results.csv")
    groups = (
        ("Best algorithm per instance (lowest mean cost)", finding_best_by_mean(main_rows)),
        ("Reference hits", finding_hits(main_rows)),
        ("Plain GA vs Clarke–Wright", finding_ga_vs_cw(read_csv(tables / "cw_baseline.csv"), main_rows)),
        ("Wilcoxon tests", finding_wilcoxon(read_csv(tables / "wilcoxon.csv"))),
        ("Team instance crossings", finding_crossings(read_csv(tables / "team_crossings.csv"))),
        ("Complexity slopes", finding_slopes(read_csv(tables / "complexity_fits.csv"), timing)),
        ("GA+LS cost shares", finding_shares(read_csv(root / "results/timing/share_ga_ls.csv"))),
    )
    lines = ["## Key findings (computed from the tables)"]
    for title, sentences in groups:
        lines += ["", f"**{title}**", ""] + [f"- {s}" for s in sentences or ["(no data)"]]
    return lines


def section_sources(root: Path) -> list[str]:
    """Where every report table and figure comes from. Time: O(items)."""
    rows = []
    for item, source, command in SOURCES:
        first = source.split(",")[0].strip()
        if source == "docs/report_numbers.md":
            present = "this file"                      # not checked, so that two runs give the same bytes
        elif "<" in first or "{" in first:
            present = "pattern"
        else:
            present = "yes" if (root / first).exists() else "no"
        rows.append([item, f"`{source}`", f"`{command}`", present])
    return (["## How to cite a number", "",
             "Quote a number from the file in the second column. The command rewrites it; 'present' says whether "
             "the file exists in this checkout ('pattern' = several files).", ""]
            + md_table(["report item", "source file", "command", "present"], rows))


# --------------------------------------------------------------------------- the sheet

def build_sheet(root: Path, head: str, env: dict[str, str]) -> str:
    """The whole sheet as text (LF). Time: O(total input size)."""
    runs = read_csv(root / "results/raw/runs.csv")
    timing = read_csv(root / "results/timing/timing.csv")
    main_rows = read_csv(root / "results/tables/main_results.csv")
    lines = ["# Report numbers", "",
             "Generated by `python -m experiments.report_numbers`. Do not edit by hand: every number below is read "
             "from the files named in each section.", ""]
    for section in (section_setup(root, runs, timing, head, env), section_instances(root, runs, main_rows),
                    section_params(root, runs), section_findings(root, timing), section_tables(root),
                    section_sources(root)):
        lines += section + [""]
    return "\n".join(lines).rstrip("\n") + "\n"


def write_sheet(text: str, path: Path) -> Path:
    """Write the sheet with LF newlines. Time: O(len(text))."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def git_head(root: Path) -> str:
    """HEAD commit of the repository at `root`, plus a marker if the working tree has changes. Time: O(repo)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    head = git("rev-parse", "HEAD") or "unknown"
    return head + (" (+ uncommitted changes)" if git("status", "--porcelain") else "")


def environment_versions() -> dict[str, str]:
    """Python and package versions of the running environment. Time: O(1)."""
    env = {"python": sys.version.split()[0]}
    for name in PACKAGES:
        try:
            env[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            env[name] = "not installed"
    return env


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description="Step 7a: write docs/report_numbers.md from the result files.")
    ap.add_argument("--root", default=str(config.ROOT), help="repository root to read from (default: this repo)")
    ap.add_argument("--out", default=None, help="output file (default <root>/docs/report_numbers.md)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "docs" / "report_numbers.md"
    path = write_sheet(build_sheet(root, git_head(root), environment_versions()), out)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()
