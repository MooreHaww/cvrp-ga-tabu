# Original code (algorithm: Prins 2004, giant tour + Split)
"""Step 3: Split DP – cut a giant tour optimally into capacity-feasible routes.

C. Prins, "A simple and effective evolutionary algorithm for the vehicle
routing problem", Computers & Operations Research 31(12), 2004.

Giant tour T = (t_1, ..., t_n), a permutation of the customers. Positions are
1-based in the comments and in arc_cost(); the Python list index is k - 1.

DAG: nodes 0..n (node k = "the first k customers of T are served").
Arc (i -> j), i < j, exists iff load(i, j) = q(t_{i+1}) + ... + q(t_j) <= Q;
its cost is the route 0 - t_{i+1} - ... - t_j - 0:
    c(i, j) = d(0, t_{i+1}) + sum_{k=i+1}^{j-1} d(t_k, t_{k+1}) + d(t_j, 0).
A shortest path 0 -> n is an optimal split of THIS order (not of the problem).

CLI: python -m cvrp.split --benchmark      # time per split() call on random tours
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

from cvrp.instance import Instance
from cvrp.solution import Solution, total_cost

INF = float("inf")


@dataclass
class SplitResult:
    """Output of split(): the routes, their total cost and the number of DAG arcs evaluated."""

    solution: Solution
    cost: float
    arcs_examined: int


def _check_tour(inst: Instance, tour: list[int]) -> None:
    """Raise ValueError unless tour is a permutation of 1..n. Time: O(n)."""
    if len(tour) != inst.n or sorted(tour) != list(range(1, inst.n + 1)):
        raise ValueError(f"tour must be a permutation of 1..{inst.n}")


def _prefix_sums(inst: Instance, tour: list[int]) -> tuple[list[int], list[float]]:
    """Prefix sums along the tour, so every arc load and cost is O(1). Time: O(n).

    load_prefix[k] = q(t_1) + ... + q(t_k)                    (load_prefix[0] = 0)
    dist_prefix[k] = d(t_1,t_2) + ... + d(t_{k-1},t_k)         (dist_prefix[0] = dist_prefix[1] = 0)
    """
    n = len(tour)
    load_prefix = [0] * (n + 1)
    dist_prefix = [0.0] * (n + 1)
    for k in range(1, n + 1):
        load_prefix[k] = load_prefix[k - 1] + int(inst.demand[tour[k - 1]])
        if k >= 2:
            dist_prefix[k] = dist_prefix[k - 1] + float(inst.dist[tour[k - 2], tour[k - 1]])
    return load_prefix, dist_prefix


def arc_cost(inst: Instance, tour: list[int], dist_prefix: list[float], i: int, j: int) -> float:
    """Cost of arc (i -> j) = route 0 - t_{i+1} - ... - t_j - 0. Time: O(1).

    c(i, j) = d(0, t_{i+1}) + (dist_prefix[j] - dist_prefix[i+1]) + d(t_j, 0)
    """
    first, last = tour[i], tour[j - 1]  # t_{i+1} and t_j in 0-based list indexing
    return float(inst.dist[0, first]) + (dist_prefix[j] - dist_prefix[i + 1]) + float(inst.dist[last, 0])


def _routes_from_cuts(tour: list[int], cuts: list[int]) -> list[list[int]]:
    """cuts = [0, c_1, ..., n] -> list of consecutive segments of the tour. Time: O(n)."""
    return [tour[a:b] for a, b in zip(cuts, cuts[1:])]


def _split_unlimited(inst, tour, load_prefix, dist_prefix, d0):
    """Bellman DP on the DAG with any number of routes. Time: O(n * L), space O(n).

    f[0] = 0;  f[j] = min over i < j with load(i, j) <= Q of f[i] + c(i, j).
    The loop over i runs backwards from j-1 and stops at the first infeasible
    load (loads only grow as i decreases), so each j looks at <= L + 1 arcs.
    """
    n = len(tour)
    f = [INF] * (n + 1)
    pred = [-1] * (n + 1)
    f[0] = 0.0
    arcs = 0
    for j in range(1, n + 1):
        back = d0[tour[j - 1]]                     # d(t_j, 0)
        for i in range(j - 1, -1, -1):
            if load_prefix[j] - load_prefix[i] > inst.Q:
                break
            arcs += 1
            cost = f[i] + d0[tour[i]] + (dist_prefix[j] - dist_prefix[i + 1]) + back
            if cost < f[j]:
                f[j], pred[j] = cost, i
    cuts = [n]
    while cuts[-1] > 0:
        cuts.append(pred[cuts[-1]])
    return f[n], cuts[::-1], arcs


def _split_limited(inst, tour, load_prefix, dist_prefix, d0, K):
    """Layered DP with at most K routes. Time: O(K * n * L), space O(K * n).

    f[k][0] = 0 for all k (serving nobody needs no route), f[0][j > 0] = inf,
    f[k][j] = min over i < j with load(i, j) <= Q of f[k-1][i] + c(i, j).
    Because f[k-1][0] = 0, the last layer allows "at most k" routes, so the
    answer is f[K][n] (inf if no split with <= K routes exists).
    """
    n = len(tour)
    f = [[INF] * (n + 1) for _ in range(K + 1)]
    pred = [[-1] * (n + 1) for _ in range(K + 1)]
    for k in range(K + 1):
        f[k][0] = 0.0
    arcs = 0
    for k in range(1, K + 1):
        prev, cur, p = f[k - 1], f[k], pred[k]
        for j in range(1, n + 1):
            back = d0[tour[j - 1]]
            for i in range(j - 1, -1, -1):
                if load_prefix[j] - load_prefix[i] > inst.Q:
                    break
                arcs += 1
                if prev[i] == INF:
                    continue
                cost = prev[i] + d0[tour[i]] + (dist_prefix[j] - dist_prefix[i + 1]) + back
                if cost < cur[j]:
                    cur[j], p[j] = cost, i
    if f[K][n] == INF:
        return INF, None, arcs
    cuts, k = [n], K
    while cuts[-1] > 0:                            # walk back one layer per route
        cuts.append(pred[k][cuts[-1]])
        k -= 1
    return f[K][n], cuts[::-1], arcs


def split(inst: Instance, tour: list[int], max_routes: int | None = None) -> SplitResult | None:
    """Optimal split of a giant tour into capacity-feasible routes (Prins 2004).

    max_routes=None : any number of routes, time O(n * L), space O(n).
    max_routes=K    : at most K routes, time O(K * n * L), space O(K * n);
                      returns None if no such split exists.
    L = max number of customers that fit in one route (L <= n, so O(n^2) /
    O(K * n^2) in the worst case). Validation of the tour costs O(n).
    """
    tour = [int(v) for v in tour]
    _check_tour(inst, tour)
    load_prefix, dist_prefix = _prefix_sums(inst, tour)
    d0 = [float(x) for x in inst.dist[0]]          # d(0, v) as a plain list (fast O(1) access)
    if max_routes is None:
        dp_cost, cuts, arcs = _split_unlimited(inst, tour, load_prefix, dist_prefix, d0)
    else:
        dp_cost, cuts, arcs = _split_limited(inst, tour, load_prefix, dist_prefix, d0, max_routes)
        if cuts is None:
            return None
    routes = _routes_from_cuts(tour, cuts)
    cost = total_cost(inst, routes)                # recomputed: identical to what the checker sums
    assert abs(cost - dp_cost) <= 1e-6, (cost, dp_cost)
    return SplitResult(solution=Solution(routes), cost=cost, arcs_examined=arcs)


def benchmark(n_tours: int = 200, seed: int = 0) -> list[tuple]:
    """Mean time per split() call on random tours, unlimited and with max_routes=K.

    Returns rows (name, n, K, us_unlimited, us_limited, arcs_unlimited, arcs_limited,
    feasible_share_limited, L). Arcs are averaged over calls that returned a split;
    feasible_share_limited = fraction of random tours that can be cut into <= K routes;
    L = max customers on one route over all splits. Time: O(n_tours * K * n * L) per instance.
    """
    from cvrp.parser import load_instance

    paths = ["data/cvrplib/A-n32-k5.vrp", "data/cvrplib/A-n60-k9.vrp",
             "data/cvrplib/A-n80-k10.vrp", "data/team/team_n100_seed0.vrp"]
    rng = random.Random(seed)
    rows = []
    for path in paths:
        inst = load_instance(path)
        tours = []
        for _ in range(n_tours):
            t = list(range(1, inst.n + 1))
            rng.shuffle(t)
            tours.append(t)
        stats = []
        longest = 0
        for limit in (None, inst.K):
            arcs = found = 0
            start = time.perf_counter()
            for t in tours:
                res = split(inst, t, max_routes=limit)
                if res is not None:
                    found += 1
                    arcs += res.arcs_examined
                    longest = max(longest, max(len(r) for r in res.solution.routes))
            elapsed_us = (time.perf_counter() - start) / n_tours * 1e6
            stats.append((elapsed_us, arcs / max(found, 1), found / n_tours))
        rows.append((inst.name, inst.n, inst.K, stats[0][0], stats[1][0], stats[0][1], stats[1][1],
                     stats[1][2], longest))
    return rows


def main() -> None:
    """Command-line entry point (see module docstring)."""
    ap = argparse.ArgumentParser(description="Split DP utilities.")
    ap.add_argument("--benchmark", action="store_true", help="time split() on random tours")
    ap.add_argument("--tours", type=int, default=200)
    args = ap.parse_args()
    if not args.benchmark:
        ap.print_help()
        return
    print("instance | n | K | us/split (unlimited) | us/split (max_routes=K) | arcs (unl.) | arcs (K) "
          "| tours with a <=K split | max route len")
    for name, n, K, t_unl, t_k, a_unl, a_k, share, longest in benchmark(args.tours):
        print(f"{name} | {n} | {K} | {t_unl:.0f} | {t_k:.0f} | {a_unl:.0f} | {a_k:.0f} | {share:.0%} | {longest}")


if __name__ == "__main__":
    main()
