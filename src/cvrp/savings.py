# Original code (algorithm: Clarke & Wright 1964)
"""Step 2: Clarke–Wright savings, classical PARALLEL version.

G. Clarke & J. W. Wright, "Scheduling of vehicles from a central depot to a
number of delivery points", Operations Research 12(4), 1964.

1. Start with one route 0-i-0 per customer.
2. Saving of serving i and j on one route: s(i, j) = d(0,i) + d(0,j) - d(i,j).
3. Sort the pairs by (saving desc, i asc, j asc), keep only s > 0.
4. Merge the routes of i and j when they differ, i and j are both endpoints
   of their routes, and the combined load fits in Q.

The number of routes K is NOT enforced: if the result uses more than K routes,
a warning is printed and the solution is returned unchanged.

Total: O(n^2 log n) time, O(n^2) space.

CLI: python -m cvrp.savings data/cvrplib/A-n32-k5.vrp
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from cvrp.checker import check
from cvrp.instance import Instance
from cvrp.parser import load_instance
from cvrp.solution import Solution, total_cost


@dataclass
class CWResult:
    """Output of clarke_wright()."""

    solution: Solution
    cost: float
    n_routes: int
    iterations: int  # number of savings pairs examined
    merges: int
    progress: list[tuple[float, float]] = field(default_factory=list)  # (elapsed_seconds, best_cost)


def compute_savings(inst: Instance) -> list[tuple[float, int, int]]:
    """All positive savings (s, i, j) with i < j, sorted by (s desc, i asc, j asc).

    Time: O(n^2) to compute, O(n^2 log n) to sort. Space: O(n^2).
    """
    d = inst.dist
    savings = []
    for i in range(1, inst.n + 1):
        for j in range(i + 1, inst.n + 1):
            s = float(d[0, i] + d[0, j] - d[i, j])
            if s > 0:
                savings.append((s, i, j))
    savings.sort(key=lambda t: (-t[0], t[1], t[2]))
    return savings


def _is_endpoint(route: list[int], v: int) -> bool:
    """True if v is the first or the last customer of the route. Time: O(1)."""
    return route[0] == v or route[-1] == v


def _merge(route_i: list[int], route_j: list[int], i: int, j: int) -> list[int]:
    """Join two routes so that i and j become adjacent: (... i)(j ...).

    route_i is reversed if i is its first customer (and it has more than one),
    route_j is reversed if j is its last customer. Reversing a route does not
    change its cost because d is symmetric. Time: O(len(route_i) + len(route_j)).
    """
    if route_i[-1] != i:
        route_i = route_i[::-1]
    if route_j[0] != j:
        route_j = route_j[::-1]
    return route_i + route_j


def clarke_wright(inst: Instance, warn: bool = True) -> CWResult:
    """Parallel Clarke–Wright savings heuristic.

    warn=False suppresses the "routes > K" message (used when CW only seeds the GA).

    Time: O(n^2 log n) (dominated by sorting the savings); the merge loop is
    O(n^2): O(1) checks per pair, and at most n-1 merges of O(n) each.
    Space: O(n^2) for the savings list.
    """
    start = time.perf_counter()

    # route_of[v] = id of the route containing customer v; routes[id] = customers; load[id] = demand.
    route_of = list(range(inst.n + 1))
    routes: dict[int, list[int]] = {v: [v] for v in range(1, inst.n + 1)}
    load: dict[int, int] = {v: int(inst.demand[v]) for v in range(1, inst.n + 1)}

    iterations = merges = 0
    for _, i, j in compute_savings(inst):
        iterations += 1
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue                                   # (a) already on the same route
        if not (_is_endpoint(routes[ri], i) and _is_endpoint(routes[rj], j)):
            continue                                   # (b) i or j is an interior customer
        if load[ri] + load[rj] > inst.Q:
            continue                                   # (c) capacity would be exceeded
        merged = _merge(routes[ri], routes[rj], i, j)
        # Keep the id of the longer route; relabel only the customers of the shorter one.
        keep, drop = (ri, rj) if len(routes[ri]) >= len(routes[rj]) else (rj, ri)
        for v in routes[drop]:
            route_of[v] = keep
        routes[keep] = merged
        load[keep] = load[ri] + load[rj]
        del routes[drop], load[drop]
        merges += 1

    final = sorted(routes.values(), key=lambda r: r[0])  # deterministic output order
    cost = total_cost(inst, final)
    if warn and len(final) > inst.K:
        print(f"WARNING: {inst.name}: Clarke-Wright used {len(final)} routes > K = {inst.K}")
    elapsed = time.perf_counter() - start
    return CWResult(solution=Solution(final), cost=cost, n_routes=len(final),
                    iterations=iterations, merges=merges, progress=[(elapsed, cost)])


def gap_percent(cost: float, bks: float | None) -> float | None:
    """Gap (%) = 100 * (cost - BKS) / BKS, or None if the BKS is unknown. Time: O(1)."""
    return None if bks is None else 100.0 * (cost - bks) / bks


def main() -> None:
    """Command-line entry point (see module docstring)."""
    ap = argparse.ArgumentParser(description="Run Clarke-Wright savings on a .vrp instance.")
    ap.add_argument("path", help=".vrp file")
    args = ap.parse_args()

    inst = load_instance(args.path)
    result = clarke_wright(inst)
    report = check(inst, result.solution, claimed_cost=result.cost)
    gap = gap_percent(result.cost, inst.bks)

    print(f"instance : {inst.name} (n={inst.n}, Q={inst.Q}, K={inst.K})")
    print(f"cost     : {result.cost:.2f}")
    print(f"routes   : {result.n_routes} (K = {inst.K})")
    print(f"BKS      : {inst.bks if inst.bks is not None else '-'}")
    print(f"gap      : {f'{gap:.2f} %' if gap is not None else '-'}")
    print(f"runtime  : {result.progress[-1][0] * 1000:.1f} ms "
          f"({result.iterations} pairs examined, {result.merges} merges)")
    print(f"checker  : {'feasible' if report.feasible else 'INFEASIBLE'}")
    for err in report.errors:
        print(f"  - {err}")
    print(result.solution.describe(inst))


if __name__ == "__main__":
    main()
