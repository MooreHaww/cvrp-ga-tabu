# Original code (CMP 3005 term project).
"""Step 1 / Milestone 1: summary table of all instances.

    python -m cvrp.summary

Columns: instance | n | Q | K | sum(q) | ceil(sum(q)/Q) | BKS/optimum | FFD bins.
BKS values are read from the .sol files, tiny optima from the .opt.json files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from cvrp.generator import ffd_bin_count
from cvrp.parser import load_instance

# Sanity check only (published CVRPLIB values); the table itself uses the .sol files.
EXPECTED_BKS = {"A-n32-k5": 784, "A-n45-k6": 944, "A-n60-k9": 1354, "A-n80-k10": 1763}
FOLDERS = ("data/cvrplib", "data/team", "data/tiny")


def reference_value(path: Path, bks: float | None) -> str:
    """BKS from the .sol (already in bks) or optimum from <stem>.opt.json, else "-". Time: O(1)."""
    if bks is not None:
        return f"{bks:g}"
    opt = path.with_suffix(".opt.json")
    if opt.exists():
        return f"{json.loads(opt.read_text())['cost']:g} (opt)"
    return "-"


def summary_rows() -> tuple[list[list[str]], list[str]]:
    """One row per .vrp file in FOLDERS, plus a list of BKS mismatch messages. Time: O(sum of n^2)."""
    rows, problems = [], []
    for folder in FOLDERS:
        for path in sorted(Path(folder).glob("*.vrp"), key=lambda p: (len(p.stem), p.stem)):
            inst = load_instance(path)
            total = inst.total_demand()
            if inst.name in EXPECTED_BKS and inst.bks != EXPECTED_BKS[inst.name]:
                problems.append(f"MISMATCH: {inst.name} BKS {inst.bks} != expected {EXPECTED_BKS[inst.name]}")
            rows.append([inst.name, str(inst.n), str(inst.Q), str(inst.K), str(total),
                         str(math.ceil(total / inst.Q)), reference_value(path, inst.bks),
                         str(ffd_bin_count(list(inst.demand[1:]), inst.Q))])
    return rows, problems


def format_table(header: list[str], rows: list[list[str]]) -> str:
    """Plain-text table with aligned columns. Time: O(cells)."""
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: " | ".join(c.ljust(w) for c, w in zip(r, widths))
    return "\n".join([line(header), "-+-".join("-" * w for w in widths)] + [line(r) for r in rows])


def main() -> None:
    """Print the table and any BKS mismatches."""
    header = ["instance", "n", "Q", "K", "sum(q)", "ceil(sum(q)/Q)", "BKS/optimum", "FFD bins"]
    rows, problems = summary_rows()
    print(format_table(header, rows))
    for msg in problems:
        print(msg)


if __name__ == "__main__":
    main()
