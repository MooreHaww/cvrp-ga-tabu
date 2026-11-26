# Original code (CMP 3005 term project). Held–Karp DP: Bellman (1962), Held & Karp (1962).
"""Step 1: exact solver for n <= 9 customers (ground truth for the tiny instances).

It reuses no heuristic code. Two dynamic programs over bitmask subsets:

(1) Held–Karp, for every customer subset S with demand(S) <= Q:
        path[S][j] = shortest path 0 -> (all of S) ending at j in S
        path[{j}][j] = d(0, j)
        path[S ∪ {k}][k] = min_j path[S][j] + d(j, k)
        route[S] = min_j path[S][j] + d(j, 0)
    Time O(2^n * n^2), memory O(2^n * n).

(2) Set partition with at most K routes:
        best[0][∅] = 0,  best[k][S] = min( best[k-1][S],
                    min over T ⊆ S, lowest(S) ∈ T, demand(T) <= Q of route[T] + best[k-1][S \\ T] )
    Fixing the lowest customer of S inside T avoids counting the same partition
    several times. Time O(K * 3^n), memory O(K * 2^n).

Total time O(2^n * n^2 + K * 3^n); for n = 9 that is well under a second.

CLI: python -m cvrp.exact_small   # solves data/tiny/*.vrp, writes tiny_n{n}.opt.json
"""

from __future__ import annotations

import json
from pathlib import Path

from cvrp.instance import Instance

INF = float("inf")
MAX_N = 9


def _subset_loads(inst: Instance) -> list[int]:
    """load[S] = total demand of subset S (bit i = customer i+1). Time: O(2^n)."""
    n = inst.n
    load = [0] * (1 << n)
    for S in range(1, 1 << n):
        low = (S & -S).bit_length() - 1          # index of the lowest set bit
        load[S] = load[S & (S - 1)] + int(inst.demand[low + 1])
    return load


def best_single_routes(inst: Instance, load: list[int]) -> tuple[list[float], list[list[int]]]:
    """Held–Karp: optimal depot cycle cost and order for every feasible subset.

    Returns (route_cost, route_order) indexed by subset mask; infeasible subsets
    (load > Q) have cost INF. Time: O(2^n * n^2).
    """
    n, d = inst.n, inst.dist
    full = 1 << n
    path = [[INF] * n for _ in range(full)]
    parent = [[-1] * n for _ in range(full)]
    for j in range(n):
        if load[1 << j] <= inst.Q:
            path[1 << j][j] = d[0, j + 1]
    for S in range(1, full):
        if load[S] > inst.Q:
            continue
        for j in range(n):
            if path[S][j] == INF:
                continue
            for k in range(n):
                if S & (1 << k):
                    continue
                T = S | (1 << k)
                if load[T] > inst.Q:
                    continue
                cand = path[S][j] + d[j + 1, k + 1]
                if cand < path[T][k]:
                    path[T][k] = cand
                    parent[T][k] = j

    route_cost = [INF] * full
    route_order: list[list[int]] = [[] for _ in range(full)]
    route_cost[0] = 0.0
    for S in range(1, full):
        if load[S] > inst.Q:
            continue
        last = min(range(n), key=lambda j: path[S][j] + d[j + 1, 0])
        route_cost[S] = path[S][last] + d[last + 1, 0]
        order, mask, j = [], S, last          # walk the parent pointers backwards
        while j != -1:
            order.append(j + 1)
            prev = parent[mask][j]
            mask ^= 1 << j
            j = prev
        route_order[S] = order[::-1]
    return route_cost, route_order


def solve_exact(inst: Instance) -> tuple[float, list[list[int]]]:
    """Optimal CVRP cost and routes for n <= 9 using at most K routes.

    Time: O(2^n * n^2 + K * 3^n).
    """
    if inst.n > MAX_N:
        raise ValueError(f"exact_small supports n <= {MAX_N}, got {inst.n}")
    n, full = inst.n, 1 << inst.n
    load = _subset_loads(inst)
    route_cost, route_order = best_single_routes(inst, load)

    # best[k][S]: min cost to serve S with at most k routes; choice[k][S] = T used (0 = "use k-1").
    best = [[INF] * full for _ in range(inst.K + 1)]
    choice = [[0] * full for _ in range(inst.K + 1)]
    best[0][0] = 0.0
    for k in range(1, inst.K + 1):
        best[k][0] = 0.0
        for S in range(1, full):
            best[k][S] = best[k - 1][S]
            low = S & -S
            rest = S ^ low
            sub = rest
            while True:                       # enumerate all subsets of rest
                T = sub | low
                cand = route_cost[T] + best[k - 1][S ^ T]
                if cand < best[k][S]:
                    best[k][S] = cand
                    choice[k][S] = T
                if sub == 0:
                    break
                sub = (sub - 1) & rest

    if best[inst.K][full - 1] == INF:
        raise ValueError(f"{inst.name}: no feasible solution with K = {inst.K}")
    routes, k, S = [], inst.K, full - 1
    while S:                                   # rebuild the partition
        T = choice[k][S]
        if T:
            routes.append(route_order[T])
            S ^= T
        k -= 1
    return float(best[inst.K][full - 1]), routes


def main() -> None:
    """Solve every data/tiny/*.vrp and store the optimum next to it as .opt.json."""
    from cvrp.checker import check
    from cvrp.parser import parse_vrp

    for path in sorted(Path("data/tiny").glob("*.vrp")):
        inst = parse_vrp(path)
        cost, routes = solve_exact(inst)
        report = check(inst, routes, claimed_cost=cost)
        if not report.feasible:
            raise RuntimeError(f"{inst.name}: exact solution fails the checker: {report.errors}")
        out = path.with_suffix(".opt.json")
        out.write_text(json.dumps({"instance": inst.name, "cost": cost, "routes": routes}, indent=2) + "\n", newline="\n")
        print(f"{inst.name}: optimum {cost:g} with {len(routes)} routes -> {out}")


if __name__ == "__main__":
    main()
