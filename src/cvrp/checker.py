# Original code (CMP 3005 term project).
"""Step 1: independent feasibility checker.

This is also the polynomial-time NP verifier from our theory write-up: the
certificate is the list of routes. It deliberately imports nothing from the
algorithm modules (savings/split/ga/tabu) and not even solution.py's cost
functions, so a bug there cannot hide itself here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cvrp.instance import Instance


@dataclass
class CheckReport:
    """Result of check(): feasible iff errors is empty."""

    feasible: bool
    errors: list[str] = field(default_factory=list)
    cost: float = 0.0
    n_routes: int = 0
    loads: list[int] = field(default_factory=list)


def check(inst: Instance, solution, claimed_cost: float | None = None) -> CheckReport:
    """Verify a solution (a Solution object or a list of routes) from scratch.

    Checks: non-empty routes <= K; every id in 1..n; no customer repeated; no
    customer missing; every route load <= Q; cost recomputed from the distance
    matrix; |claimed_cost - cost| <= 1e-6 if claimed_cost is given.
    Empty routes are ignored and do not count toward K.

    Time: O(n + K) for the feasibility checks (one pass over all route entries,
    one pass over a seen-array of size n+1) plus O(n) for the cost sum.
    """
    routes = solution.routes if hasattr(solution, "routes") else solution
    routes = [list(r) for r in routes if len(r) > 0]
    errors: list[str] = []

    if len(routes) > inst.K:
        errors.append(f"too many routes: {len(routes)} > K = {inst.K}")

    seen = [0] * (inst.n + 1)  # seen[v] = how many times customer v appears
    loads: list[int] = []
    cost = 0.0
    for idx, route in enumerate(routes, start=1):
        load = 0
        prev = 0  # start at the depot
        for v in route:
            if not hasattr(v, "__index__"):  # accepts Python and numpy ints, rejects floats/strings
                errors.append(f"route {idx}: id {v!r} is not an integer")
                continue
            v = int(v)
            if v < 1 or v > inst.n:
                errors.append(f"route {idx}: id {v} out of range 1..{inst.n}")
                continue
            seen[v] += 1
            load += int(inst.demand[v])
            cost += float(inst.dist[prev, v])
            prev = v
        cost += float(inst.dist[prev, 0])  # return to the depot
        loads.append(load)
        if load > inst.Q:
            errors.append(f"route {idx}: load {load} > Q = {inst.Q}")

    duplicates = [v for v in range(1, inst.n + 1) if seen[v] > 1]
    missing = [v for v in range(1, inst.n + 1) if seen[v] == 0]
    if duplicates:
        errors.append(f"customers visited more than once: {duplicates}")
    if missing:
        errors.append(f"customers not visited: {missing}")
    if claimed_cost is not None and abs(claimed_cost - cost) > 1e-6:
        errors.append(f"claimed cost {claimed_cost} != recomputed cost {cost}")

    return CheckReport(feasible=not errors, errors=errors, cost=cost,
                       n_routes=len(routes), loads=loads)
