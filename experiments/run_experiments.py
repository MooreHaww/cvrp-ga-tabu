# Original code
"""Step 6a: experiment runner (course brief, Task 7).

    python -m experiments.run_experiments                  # full protocol into results/raw
    python -m experiments.run_experiments --smoke          # 2 runs, 5% budgets, into results/smoke

Order: for each instance, for each run r, for each algorithm (interleaved, so all
algorithms see the same machine conditions). Run r uses seed BASE_SEED + r for
every algorithm. Every result is validated by checker.check; runs without a
feasible solution are kept with feasible=False. Rows are appended and flushed
after every run; existing (instance, algo, run) rows are skipped (resume).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cvrp.checker import check
from cvrp.generator import generate_team_instance, write_vrp
from cvrp.parser import load_instance
from cvrp.savings import gap_percent
from experiments.config import (ALGORITHMS, BASE_SEED, BUDGETS, DEFAULT_ALGOS, DEFAULT_RUNS,
                                INSTANCE_ORDER, ROOT, TEAM_SEED_DEFAULT, instance_path)

RUN_COLUMNS = ["instance", "algo", "run", "seed", "budget_s", "elapsed_s", "cost", "n_routes", "K",
               "feasible", "check_errors", "bks_or_opt", "gap_pct", "iterations", "work",
               "reached_optimum", "time_to_best_s", "params_json", "git_commit"]
PROGRESS_COLUMNS = ["instance", "algo", "run", "seed", "elapsed_s", "best_cost"]


# --------------------------------------------------------------------------- environment info

def git_commit() -> str:
    """Current commit hash, with '-dirty' if tracked files are modified; 'unknown' without git."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cpu_model() -> str:
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _physical_cores() -> str:
    try:
        if sys.platform == "win32":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            return out or "unknown"
        out = subprocess.run(["lscpu", "-p=Core,Socket"], capture_output=True, text=True, timeout=30).stdout
        cores = {line for line in out.splitlines() if line and not line.startswith("#")}
        return str(len(cores)) if cores else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _ram_gb() -> str:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return f"{status.ullTotalPhys / 2**30:.1f}"
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return f"{int(line.split()[1]) / 2**20:.1f}"
    except (OSError, AttributeError, ValueError):
        pass
    return "unknown"


def write_hardware(out_dir: Path, workers: int) -> Path:
    """Write hardware.txt (needed for the report: 'Report your hardware'). Time: a few subprocess calls."""
    lines = [
        f"date: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"os: {platform.platform()}",
        f"cpu: {_cpu_model()}",
        f"physical_cores: {_physical_cores()}",
        f"logical_cores: {os.cpu_count()}",
        f"ram_gb: {_ram_gb()}",
        f"python: {platform.python_version()}",
        f"numpy: {np.__version__}",
        f"git_commit: {git_commit()}",
        f"workers: {workers}",
    ]
    path = out_dir / "hardware.txt"
    path.write_text("\n".join(lines) + "\n", newline="\n")
    return path


# --------------------------------------------------------------------------- one run

def reference_value(name: str, inst) -> float | None:
    """BKS (Augerat, from the .sol) or exact optimum (tiny, from .opt.json); None for the team instance."""
    if inst.bks is not None:
        return float(inst.bks)
    opt = instance_path(name).with_suffix(".opt.json") if name.startswith("tiny_") else None
    if opt is not None and opt.exists():
        return float(json.loads(opt.read_text())["cost"])
    return None


def run_one(task: dict) -> tuple[dict, list[dict]]:
    """Run one (instance, algo, run) task, validate it, and build its CSV rows.

    Top-level function so it can be sent to a worker process.
    """
    name, algo, run, seed, budget = task["instance"], task["algo"], task["run"], task["seed"], task["budget"]
    inst = load_instance(task["path"])
    outcome = ALGORITHMS[algo](inst, seed, budget, task["overrides"])
    ref = reference_value(name, inst)

    feasible, errors, cost = False, "no feasible solution found", ""
    if outcome.solution is not None:
        report = check(inst, outcome.solution, claimed_cost=outcome.cost)
        feasible, errors, cost = report.feasible, "; ".join(report.errors), outcome.cost
    gap = gap_percent(cost, ref) if feasible and ref is not None else None
    reached = ""
    if name.startswith("tiny_") and ref is not None:
        reached = feasible and abs(cost - ref) < 1e-6

    row = {
        "instance": name, "algo": algo, "run": run, "seed": seed, "budget_s": budget,
        "elapsed_s": round(outcome.elapsed, 4), "cost": cost,
        "n_routes": outcome.n_routes if outcome.n_routes is not None else "",
        "K": inst.K, "feasible": feasible, "check_errors": errors,
        "bks_or_opt": ref if ref is not None else "",
        "gap_pct": round(gap, 4) if gap is not None else "",
        "iterations": outcome.iterations, "work": outcome.work, "reached_optimum": reached,
        "time_to_best_s": round(outcome.progress[-1][0], 4) if outcome.progress else "",
        "params_json": json.dumps(outcome.params, sort_keys=True),
        "git_commit": task["git_commit"],
    }
    progress = [{"instance": name, "algo": algo, "run": run, "seed": seed,
                 "elapsed_s": round(t, 4), "best_cost": c} for t, c in outcome.progress]
    return row, progress


# --------------------------------------------------------------------------- task list and files

def build_tasks(instances, algos, runs, budget_scale, team_seed, overrides, commit) -> list[dict]:
    """Interleaved task order: instance -> run -> algorithm. Time: O(#tasks)."""
    tasks = []
    for name in instances:
        for run in range(runs):
            for algo in algos:
                tasks.append({"instance": name, "algo": algo, "run": run, "seed": BASE_SEED + run,
                              "budget": BUDGETS[name] * budget_scale,
                              "path": str(instance_path(name, team_seed)),
                              "overrides": overrides.get(algo, {}), "git_commit": commit})
    return tasks


def done_keys(runs_csv: Path) -> set[tuple[str, str, int]]:
    """(instance, algo, run) keys already in runs.csv (for resuming). Time: O(rows)."""
    if not runs_csv.exists():
        return set()
    with runs_csv.open(newline="") as f:
        return {(r["instance"], r["algo"], int(r["run"])) for r in csv.DictReader(f)}


class _Appender:
    """Append rows to a CSV file, writing the header only for a new file; flush after each row."""

    def __init__(self, path: Path, columns: list[str]) -> None:
        new = not path.exists() or path.stat().st_size == 0
        self.file = path.open("a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if new:
            self.writer.writeheader()

    def write(self, rows: list[dict]) -> None:
        self.writer.writerows(rows)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def ensure_team_file(team_seed: int) -> None:
    """Generate data/team/team_n100_seed{S}.vrp if it does not exist yet."""
    path = instance_path("team", team_seed)
    if not path.exists():
        write_vrp(generate_team_instance(team_seed), path, f"MediDrone team instance, seed {team_seed}")
        print(f"generated {path}")


def _progress_line(row: dict) -> str:
    cost = f"{row['cost']:.2f}" if row["feasible"] else "-"
    gap = f"{row['gap_pct']:.2f}%" if row["gap_pct"] != "" else "-"
    return (f"{row['instance']:<16} {row['algo']:<10} run {row['run']:<2} seed {row['seed']}  "
            f"cost {cost:>10}  gap {gap:>7}  feasible {row['feasible']!s:<5}  "
            f"elapsed {row['elapsed_s']:.2f}s")


# --------------------------------------------------------------------------- CLI

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the GA/TS experiments (course brief, Task 7).")
    ap.add_argument("--algos", default=",".join(DEFAULT_ALGOS), help=f"subset of {sorted(ALGORITHMS)}")
    ap.add_argument("--instances", default=",".join(INSTANCE_ORDER), help=f"subset of {INSTANCE_ORDER}")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--budget-scale", type=float, default=1.0, help="multiplies every time budget")
    ap.add_argument("--team-seed", type=int, default=TEAM_SEED_DEFAULT)
    ap.add_argument("--params-file", default=None,
                    help='JSON like {"ga": {"pop_size": 80}, "tabu": {"tenure_max": 20}}')
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (default 1 = sequential)")
    ap.add_argument("--smoke", action="store_true", help="= --runs 2 --budget-scale 0.05 --out results/smoke")
    args = ap.parse_args(argv)
    if args.smoke:
        args.runs, args.budget_scale, args.out = 2, 0.05, "results/smoke"
    return args


def main(argv=None) -> Path:
    """Run all missing tasks and return the output directory."""
    args = parse_args(argv)
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    instances = [i.strip() for i in args.instances.split(",") if i.strip()]
    unknown = [a for a in algos if a not in ALGORITHMS] + [i for i in instances if i not in BUDGETS]
    if unknown:
        raise SystemExit(f"unknown algorithm/instance names: {unknown}")
    overrides = json.loads(Path(args.params_file).read_text()) if args.params_file else {}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "team" in instances:
        ensure_team_file(args.team_seed)
    write_hardware(out_dir, args.workers)

    runs_csv, progress_csv = out_dir / "runs.csv", out_dir / "progress.csv"
    done = done_keys(runs_csv)
    tasks = build_tasks(instances, algos, args.runs, args.budget_scale, args.team_seed, overrides, git_commit())
    todo = [t for t in tasks if (t["instance"], t["algo"], t["run"]) not in done]
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done (resume), {len(todo)} to run "
          f"-> {out_dir}")

    runs_out, progress_out = _Appender(runs_csv, RUN_COLUMNS), _Appender(progress_csv, PROGRESS_COLUMNS)
    infeasible = 0
    try:
        def record(row, progress):
            nonlocal infeasible
            runs_out.write([row])
            progress_out.write(progress)
            infeasible += not row["feasible"]
            print(_progress_line(row), flush=True)

        if args.workers <= 1:
            for task in todo:
                record(*run_one(task))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_one, t) for t in todo]   # submitted in interleaved order
                for fut in as_completed(futures):
                    record(*fut.result())
    finally:
        runs_out.close()
        progress_out.close()
    print(f"done: {len(todo)} runs written, {len(tasks) - len(todo)} skipped, {infeasible} infeasible")
    return out_dir


if __name__ == "__main__":
    main()
