# Original code (2-opt move: Croes 1958; relocate / swap / 2-opt* moves: Toth & Vigo 2003)
"""Local-search moves: intra-route 2-opt (Step 4) and O(1) move deltas (Step 5).

G. A. Croes, "A method for solving traveling-salesman problems",
Operations Research 6(6), 1958.
P. Toth & D. Vigo, "The granular tabu search and its application to the
vehicle-routing problem", INFORMS Journal on Computing 15(4), 2003.

Part 1 (below): two_opt_route / improve_solution_2opt used by the memetic GA.
Part 2 (further down): pure delta functions for relocate, swap, 2-opt* and
intra-route 2-opt, used by Tabu Search and reusable by the GA.
Part 3 (end): granular candidate lists and moves (shared by Tabu Search and
descent) and descent(), the first-improvement local search of the "full"
memetic GA (Step 4c).

A route 0 - r_1 - ... - r_m - 0 is treated as the path P = (0, r_1, ..., r_m, 0).
Reversing the segment P[a..b] (1 <= a < b <= m) replaces the edges
(P[a-1], P[a]) and (P[b], P[b+1]) by (P[a-1], P[b]) and (P[a], P[b+1]):

    delta = d(P[a-1], P[b]) + d(P[a], P[b+1]) - d(P[a-1], P[a]) - d(P[b], P[b+1])

computed in O(1) because d is symmetric (the reversed inner edges keep their length).
"""

from __future__ import annotations

import random

from cvrp.instance import Instance

EPS = 1e-9  # a move must improve by more than this (avoids cycling on float noise)
MOVE_TYPES = ("relocate", "swap", "two_opt_star", "two_opt")


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


# =============================================================================
# Part 2: O(1) move deltas (Step 5).
#
# Conventions for all functions below:
#   D       distance matrix as nested Python lists (inst.dist.tolist(): fast O(1) access)
#   q       demand list, Q capacity
#   routes  list of routes (customer ids only, depot 0 implicit; a route may be empty)
#   loads   loads[r] = total demand of routes[r]
#   prefix  prefix[r][k] = load of routes[r][:k]
#   node(route, k) = route[k] if 0 <= k < len(route) else 0 (the depot), so first/last
#   positions and empty routes need no special cases.
# Every *_delta returns (delta_distance, delta_overload) and runs in O(1).
# Every apply_* returns a NEW outer list in which only the touched routes are new
# lists: O(K) for the outer copy + O(L) for the touched routes.
# =============================================================================


def node(route: list[int], k: int) -> int:
    """Customer at position k, or the depot (0) outside the route. Time: O(1)."""
    return route[k] if 0 <= k < len(route) else 0


def overload(load: int, Q: int) -> int:
    """Capacity violation of one route: max(0, load - Q). Time: O(1)."""
    return load - Q if load > Q else 0


def full_distance(D, routes: list[list[int]]) -> float:
    """Total distance recomputed from scratch (reference for the deltas). Time: O(n)."""
    total = 0.0
    for r in routes:
        if r:
            total += D[0][r[0]] + D[r[-1]][0]
            for a, b in zip(r, r[1:]):
                total += D[a][b]
    return total


def solution_overload(q, Q: int, routes: list[list[int]]) -> int:
    """Total overload recomputed from scratch. Time: O(n)."""
    return sum(overload(sum(q[v] for v in r), Q) for r in routes)


def arc(u: int, v: int) -> tuple[int, int]:
    """Unordered arc key (the distance is symmetric). Time: O(1)."""
    return (u, v) if u < v else (v, u)


def route_arcs(route: list[int]) -> set[tuple[int, int]]:
    """All arcs of 0 - route - 0 (empty route: none). Time: O(len(route))."""
    if not route:
        return set()
    path = [0] + route + [0]
    return {arc(a, b) for a, b in zip(path, path[1:])}


# --- relocate -----------------------------------------------------------------

def _relocate_neighbours(routes, r1, i, r2, j):
    """(p, c, s, a, b): c = routes[r1][i] with neighbours p, s; it is inserted between a and b.

    For r2 == r1, j indexes the route AFTER c is removed. Time: O(1).
    """
    route1 = routes[r1]
    c = route1[i]
    p, s = node(route1, i - 1), node(route1, i + 1)
    if r2 != r1:
        route2 = routes[r2]
        return p, c, s, node(route2, j - 1), node(route2, j)

    def without_c(k: int) -> int:
        """Position k of the route without c is position k (k < i) or k + 1 (k >= i) of route1."""
        if k < 0 or k >= len(route1) - 1:
            return 0
        return route1[k] if k < i else route1[k + 1]

    return p, c, s, without_c(j - 1), without_c(j)


def relocate_delta(D, q, Q, routes, loads, r1, i, r2, j) -> tuple[float, int]:
    """Move customer c = routes[r1][i] to position j of route r2.

    delta_dist = [d(a,c) + d(c,b) - d(a,b)] - [d(p,c) + d(c,s) - d(p,s)]
    (insertion cost between a and b minus removal saving between p and s).
    delta_over = change of max(0, load - Q) on r1 (- q_c) and r2 (+ q_c); 0 if r1 == r2.
    Time: O(1).
    """
    p, c, s, a, b = _relocate_neighbours(routes, r1, i, r2, j)
    dd = (D[a][c] + D[c][b] - D[a][b]) - (D[p][c] + D[c][s] - D[p][s])
    if r1 == r2:
        return dd, 0
    qc = q[c]
    dover = (overload(loads[r1] - qc, Q) - overload(loads[r1], Q)
             + overload(loads[r2] + qc, Q) - overload(loads[r2], Q))
    return dd, dover


def relocate_added_arcs(routes, r1, i, r2, j) -> list[tuple[int, int]]:
    """Arcs a relocate creates: (p,s), (a,c), (c,b) (depot-depot pairs skipped). Time: O(1)."""
    p, c, s, a, b = _relocate_neighbours(routes, r1, i, r2, j)
    return [arc(u, v) for u, v in ((p, s), (a, c), (c, b)) if u or v]


def apply_relocate(routes, r1, i, r2, j) -> list[list[int]]:
    """Routes after the relocate. Time: O(K + L)."""
    new = list(routes)
    route1 = list(routes[r1])
    c = route1.pop(i)
    if r2 == r1:
        route1.insert(j, c)
        new[r1] = route1
    else:
        route2 = list(routes[r2])
        route2.insert(j, c)
        new[r1], new[r2] = route1, route2
    return new


# --- swap ---------------------------------------------------------------------

def swap_delta(D, q, Q, routes, loads, r1, i, r2, j) -> tuple[float, int]:
    """Exchange c1 = routes[r1][i] and c2 = routes[r2][j].

    Different routes, or non-adjacent in one route (each slot replaced independently):
        delta = d(p1,c2) + d(c2,s1) - d(p1,c1) - d(c1,s1) + d(p2,c1) + d(c1,s2) - d(p2,c2) - d(c2,s2)
    Adjacent in one route (p, c1, c2, s  ->  p, c2, c1, s; the middle arc keeps its length):
        delta = d(p,c2) + d(c1,s) - d(p,c1) - d(c2,s)
    delta_over: loads change by q2 - q1 on r1 and q1 - q2 on r2 (0 within one route). Time: O(1).
    """
    if r1 == r2 and i > j:
        i, j = j, i
    route1, route2 = routes[r1], routes[r2]
    c1, c2 = route1[i], route2[j]
    if r1 == r2 and j == i + 1:
        p, s = node(route1, i - 1), node(route1, j + 1)
        return D[p][c2] + D[c1][s] - D[p][c1] - D[c2][s], 0
    p1, s1 = node(route1, i - 1), node(route1, i + 1)
    p2, s2 = node(route2, j - 1), node(route2, j + 1)
    dd = (D[p1][c2] + D[c2][s1] - D[p1][c1] - D[c1][s1]
          + D[p2][c1] + D[c1][s2] - D[p2][c2] - D[c2][s2])
    if r1 == r2:
        return dd, 0
    diff = q[c2] - q[c1]
    dover = (overload(loads[r1] + diff, Q) - overload(loads[r1], Q)
             + overload(loads[r2] - diff, Q) - overload(loads[r2], Q))
    return dd, dover


def swap_added_arcs(routes, r1, i, r2, j) -> list[tuple[int, int]]:
    """Arcs a swap creates. Time: O(1)."""
    if r1 == r2 and i > j:
        i, j = j, i
    route1, route2 = routes[r1], routes[r2]
    c1, c2 = route1[i], route2[j]
    if r1 == r2 and j == i + 1:
        pairs = ((node(route1, i - 1), c2), (c1, node(route1, j + 1)))
    else:
        pairs = ((node(route1, i - 1), c2), (c2, node(route1, i + 1)),
                 (node(route2, j - 1), c1), (c1, node(route2, j + 1)))
    return [arc(u, v) for u, v in pairs]


def apply_swap(routes, r1, i, r2, j) -> list[list[int]]:
    """Routes after the swap. Time: O(K + L)."""
    new = list(routes)
    if r1 == r2:
        route = list(routes[r1])
        route[i], route[j] = route[j], route[i]
        new[r1] = route
    else:
        route1, route2 = list(routes[r1]), list(routes[r2])
        route1[i], route2[j] = route2[j], route1[i]
        new[r1], new[r2] = route1, route2
    return new


# --- 2-opt* (tail exchange between two routes) ----------------------------------

def two_opt_star_delta(D, Q, routes, loads, prefix, r1, i, r2, j) -> tuple[float, int]:
    """Exchange tails: r1[:i] + r2[j:]  and  r2[:j] + r1[i:]  (r1 != r2, 0 <= i <= len1, 0 <= j <= len2).

    u1 = last kept customer of r1 (or depot), v1 = first moved customer of r1 (or depot),
    likewise u2, v2 for r2:
        delta_dist = d(u1,v2) + d(u2,v1) - d(u1,v1) - d(u2,v2)
    New loads from prefix sums: prefix1[i] + (load2 - prefix2[j]) and prefix2[j] + (load1 - prefix1[i]).
    Time: O(1).
    """
    route1, route2 = routes[r1], routes[r2]
    u1, v1 = node(route1, i - 1), node(route1, i)
    u2, v2 = node(route2, j - 1), node(route2, j)
    dd = D[u1][v2] + D[u2][v1] - D[u1][v1] - D[u2][v2]
    head1, head2 = prefix[r1][i], prefix[r2][j]
    new1 = head1 + (loads[r2] - head2)
    new2 = head2 + (loads[r1] - head1)
    dover = (overload(new1, Q) + overload(new2, Q)
             - overload(loads[r1], Q) - overload(loads[r2], Q))
    return dd, dover


def two_opt_star_added_arcs(routes, r1, i, r2, j) -> list[tuple[int, int]]:
    """Arcs a 2-opt* creates: (u1,v2), (u2,v1) (depot-depot pairs skipped). Time: O(1)."""
    route1, route2 = routes[r1], routes[r2]
    pairs = ((node(route1, i - 1), node(route2, j)), (node(route2, j - 1), node(route1, i)))
    return [arc(u, v) for u, v in pairs if u or v]


def apply_two_opt_star(routes, r1, i, r2, j) -> list[list[int]]:
    """Routes after the tail exchange. Time: O(K + L)."""
    new = list(routes)
    route1, route2 = routes[r1], routes[r2]
    new[r1] = route1[:i] + route2[j:]
    new[r2] = route2[:j] + route1[i:]
    return new


# --- intra-route 2-opt ----------------------------------------------------------

def two_opt_intra_delta(D, routes, r, a, b) -> tuple[float, int]:
    """Reverse routes[r][a..b] (a < b). Same formula as two_opt_route:
        delta = d(p, x_b) + d(x_a, s) - d(p, x_a) - d(x_b, s),  p = node(a-1), s = node(b+1).
    The load is unchanged, so delta_over = 0. Time: O(1).
    """
    route = routes[r]
    p, s = node(route, a - 1), node(route, b + 1)
    xa, xb = route[a], route[b]
    return D[p][xb] + D[xa][s] - D[p][xa] - D[xb][s], 0


def two_opt_intra_added_arcs(routes, r, a, b) -> list[tuple[int, int]]:
    """Arcs a reversal creates: (p, x_b), (x_a, s). Time: O(1)."""
    route = routes[r]
    return [arc(node(route, a - 1), route[b]), arc(route[a], node(route, b + 1))]


def apply_two_opt_intra(routes, r, a, b) -> list[list[int]]:
    """Routes after reversing routes[r][a..b]. Time: O(K + L)."""
    new = list(routes)
    route = list(routes[r])
    route[a:b + 1] = route[a:b + 1][::-1]
    new[r] = route
    return new


# =============================================================================
# Part 3: granular candidate moves (shared with Tabu Search) and descent (Step 4c).
# =============================================================================

def neighbours(D, n: int, k: int | None) -> list[list[int]]:
    """Granular candidate list: the k nearest customers of every customer (k=None: all).
    Index 0 (depot) gets an empty list. Time: O(n^2 log n), done once."""
    result: list[list[int]] = [[]]
    for c in range(1, n + 1):
        others = sorted((v for v in range(1, n + 1) if v != c), key=lambda v: (D[c][v], v))
        result.append(others if k is None else others[:k])
    return result


def customer_moves(c: int, routes, where, neigh_c: list[int], moves):
    """Granular candidate moves of customer c (generator): every move makes c adjacent
    to one of its nearest neighbours nb. where[v] = (route index, position) of customer v.

    The same definition is used by Tabu Search (tabu._State.candidates) and by descent().
    Time: O(len(neigh_c)) moves, at most 5 per neighbour.
    """
    r, i = where[c]
    for nb in neigh_c:
        rn, jn = where[nb]
        if "relocate" in moves:
            if rn != r:
                yield "relocate", (r, i, rn, jn)          # c just before nb
                yield "relocate", (r, i, rn, jn + 1)      # c just after nb
            else:
                jn2 = jn if jn < i else jn - 1           # index of nb once c is removed
                if jn2 != i:
                    yield "relocate", (r, i, r, jn2)
                if jn2 + 1 != i:
                    yield "relocate", (r, i, r, jn2 + 1)
        if "swap" in moves:
            yield "swap", (r, i, rn, jn)
        if rn != r and "two_opt_star" in moves:
            yield "two_opt_star", (r, i + 1, rn, jn)    # new arc (c, nb)
        if rn == r and "two_opt" in moves:
            if i + 1 < jn:
                yield "two_opt", (r, i + 1, jn)           # ... c, nb ...
            elif jn + 1 < i:
                yield "two_opt", (r, jn + 1, i)           # ... nb, c ...


def move_delta(kind: str, args: tuple, D, q, Q: int, routes, loads, prefix) -> tuple[float, int]:
    """(delta_distance, delta_overload) of a move with the O(1) functions above. Time: O(1)."""
    if kind == "relocate":
        return relocate_delta(D, q, Q, routes, loads, *args)
    if kind == "swap":
        return swap_delta(D, q, Q, routes, loads, *args)
    if kind == "two_opt_star":
        return two_opt_star_delta(D, Q, routes, loads, prefix, *args)
    return two_opt_intra_delta(D, routes, *args)


def apply_move(kind: str, args: tuple, routes) -> list[list[int]]:
    """Routes after a move (new outer list, touched routes copied). Time: O(K + L)."""
    if kind == "relocate":
        return apply_relocate(routes, *args)
    if kind == "swap":
        return apply_swap(routes, *args)
    if kind == "two_opt_star":
        return apply_two_opt_star(routes, *args)
    return apply_two_opt_intra(routes, *args)


def descent(inst: Instance, routes: list[list[int]], neighbour_lists: list[list[int]],
            rng: random.Random | None = None, moves: tuple[str, ...] = MOVE_TYPES,
            max_passes: int | None = None, D=None, stats: dict | None = None) -> list[list[int]]:
    """First-improvement descent over granular relocate / swap / 2-opt* / 2-opt moves.

    For every customer c (fixed order 1..n, or shuffled by rng each pass) the
    candidate moves of customer_moves() are scanned; the FIRST move with
    delta_distance < -1e-9 that keeps every load <= Q is applied, then the scan
    goes on with the next customer. It stops after a full pass without any
    improvement (a local optimum w.r.t. the granular neighbourhood) or after
    max_passes passes.

    Hard capacity: the input must be capacity-feasible (Split and Clarke–Wright
    always are; otherwise ValueError). Then all overloads are 0, so a move keeps
    every load <= Q exactly when its delta_overload is 0.
    No route is ever created (moves only target routes that hold a neighbour);
    routes that become empty are dropped, so the route count never increases.

    Time per pass: O(n * kappa * m) O(1) delta evaluations (kappa = neighbours per
    customer, m <= 5 moves per neighbour) + O(K + L) per applied move. Every
    applied move lowers the distance, so the number of passes is finite; it is an
    empirical quantity, reported in stats ('passes', 'moves_applied', 'evaluated').
    D: optional distance matrix as nested lists (saves an O(n^2) conversion per call).
    """
    if D is None:
        D = inst.dist.tolist()
    q, Q, n = [int(x) for x in inst.demand], inst.Q, inst.n
    move_set = set(moves)
    if not move_set or move_set - set(MOVE_TYPES):
        raise ValueError(f"moves must be a non-empty subset of {MOVE_TYPES}")
    routes = [list(r) for r in routes]
    loads, prefix = [0] * len(routes), [[0] for _ in routes]
    where = [(-1, -1)] * (n + 1)

    def refresh(r: int) -> None:
        """Recompute load, prefix loads and positions of route r. Time: O(len(route))."""
        pre = [0]
        for k, c in enumerate(routes[r]):
            pre.append(pre[-1] + q[c])
            where[c] = (r, k)
        prefix[r], loads[r] = pre, pre[-1]

    for r in range(len(routes)):
        refresh(r)
    if any(load > Q for load in loads):
        raise ValueError("descent needs a capacity-feasible input (every load <= Q)")

    order = list(range(1, n + 1))
    passes = applied = evaluated = 0
    while max_passes is None or passes < max_passes:
        passes += 1
        if rng is not None:
            rng.shuffle(order)
        improved = False
        for c in order:
            for kind, args in customer_moves(c, routes, where, neighbour_lists[c], move_set):
                evaluated += 1
                dd, dover = move_delta(kind, args, D, q, Q, routes, loads, prefix)
                if dd < -EPS and dover == 0:
                    routes = apply_move(kind, args, routes)
                    for r in ({args[0]} if kind == "two_opt" else {args[0], args[2]}):
                        refresh(r)
                    applied += 1
                    improved = True
                    break                    # positions changed: continue with the next customer
        if not improved:
            break
    if stats is not None:
        stats.update(passes=passes, moves_applied=applied, evaluated=evaluated)
    return [r for r in routes if r]
