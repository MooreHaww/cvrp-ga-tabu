# Original code
"""Step 6a-3: read the per-run solution files written by run_experiments.

Each file <out>/solutions/<instance>_<algo>_run<r>.json holds
{"instance", "algo", "run", "seed", "budget_s", "cost", "n_routes", "feasible", "routes"};
routes use the internal numbering (customers 1..n, depot 0 implicit) and are null
when the run found no solution.

    from experiments.solutions import best_solution
    best = best_solution("results/raw", "team", "tabu")    # best["routes"] for the route plot
"""

from __future__ import annotations

import json
from pathlib import Path


def load_solution(path: str | Path) -> dict:
    """One solution record. Time: O(file size)."""
    return json.loads(Path(path).read_text())


def load_all(out_dir: str | Path) -> list[dict]:
    """Every record in <out_dir>/solutions, sorted by (instance, algo, run); [] if the
    folder does not exist. Time: O(total size + F log F) for F files."""
    folder = Path(out_dir) / "solutions"
    if not folder.is_dir():
        return []
    records = [load_solution(p) for p in folder.glob("*.json")]
    return sorted(records, key=lambda s: (s["instance"], s["algo"], s["run"]))


def best_solution(out_dir: str | Path, instance: str, algo: str) -> dict | None:
    """The feasible record of (instance, algo) with the lowest cost (ties: the lowest run);
    None if there is none. Time: O(total size of the files)."""
    feasible = [s for s in load_all(out_dir)
                if s["instance"] == instance and s["algo"] == algo and s["feasible"] and s["routes"] is not None]
    return min(feasible, key=lambda s: (s["cost"], s["run"]), default=None)
