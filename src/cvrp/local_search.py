# Original code (2-opt move: Croes 1958)
"""Step 4 (memetic part): intra-route 2-opt local search.

G. A. Croes, "A method for solving traveling-salesman problems",
Operations Research 6(6), 1958.

A route 0 - r_1 - ... - r_m - 0 is treated as the path P = (0, r_1, ..., r_m, 0).
Reversing the segment P[a..b] (1 <= a < b <= m) replaces the edges
(P[a-1], P[a]) and (P[b], P[b+1]) by (P[a-1], P[b]) and (P[a], P[b+1]):

    delta = d(P[a-1], P[b]) + d(P[a], P[b+1]) - d(P[a-1], P[a]) - d(P[b], P[b+1])

computed in O(1) because d is symmetric (the reversed inner edges keep their length).
"""

from __future__ import annotations

from cvrp.instance import Instance

EPS = 1e-9  # a move must improve by more than this (avoids cycling on float noise)


def two_opt_route(inst: Instance, route: list[int]) -> list[int]:
    """First-improvement 2-opt inside one route until no improving move is left.

    Time: O(m^2) per pass (all pairs a < b), m = len(route); the number of
    passes is bounded because every applied move strictly lowers the cost.
    Returns a new list; the customers (and so the load) are unchanged.
    """
    d = inst.dist
    path = [0] + list(route) + [0]
    m = len(route)
    improved = True
    while improved:
        improved = False
        for a in range(1, m):
            for b in range(a + 1, m + 1):
                delta = (d[path[a - 1], path[b]] + d[path[a], path[b + 1]]
                         - d[path[a - 1], path[a]] - d[path[b], path[b + 1]])
                if delta < -EPS:
                    path[a:b + 1] = path[a:b + 1][::-1]
                    improved = True
                    break           # first improvement: restart the scan
            if improved:
                break
    return path[1:-1]


def improve_solution_2opt(inst: Instance, routes: list[list[int]]) -> list[list[int]]:
    """Apply two_opt_route to every route. Time: O(sum of m_r^2) per pass over all routes."""
    return [two_opt_route(inst, r) for r in routes]
