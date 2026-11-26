# Original code (CMP 3005 term project).
"""Step 1: solution representation + cost.

A solution is a list of routes; each route is a list of customer ids (1..n).
The depot (0) is NOT stored: every route implicitly starts and ends at 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cvrp.instance import Instance


def route_cost(inst: Instance, route: list[int]) -> float:
    """Length of 0 -> route[0] -> ... -> route[-1] -> 0. Empty route costs 0. Time: O(len(route))."""
    if not route:
        return 0.0
    cost = inst.dist[0, route[0]] + inst.dist[route[-1], 0]
    for a, b in zip(route, route[1:]):
        cost += inst.dist[a, b]
    return float(cost)


def route_load(inst: Instance, route: list[int]) -> int:
    """Total demand of the customers on a route. Time: O(len(route))."""
    return int(sum(inst.demand[v] for v in route))


def total_cost(inst: Instance, routes: list[list[int]]) -> float:
    """Sum of all route costs. Time: O(n)."""
    return sum(route_cost(inst, r) for r in routes)


@dataclass
class Solution:
    """A CVRP solution: a list of routes (customer ids only, depot implicit)."""

    routes: list[list[int]] = field(default_factory=list)

    def cost(self, inst: Instance) -> float:
        """Total distance of the solution. Time: O(n)."""
        return total_cost(inst, self.routes)

    def non_empty_routes(self) -> list[list[int]]:
        """Routes that visit at least one customer. Time: O(number of routes)."""
        return [r for r in self.routes if r]

    def describe(self, inst: Instance) -> str:
        """Readable text with load and cost per route. Time: O(n)."""
        lines = []
        for i, r in enumerate(self.non_empty_routes(), start=1):
            path = "-".join(str(v) for v in [0] + r + [0])
            lines.append(f"Route {i}: {path}  load {route_load(inst, r)}  cost {route_cost(inst, r):.2f}")
        lines.append(f"Total cost {self.cost(inst):.2f}")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Routes as depot-to-depot paths (no instance needed). Time: O(n)."""
        return "\n".join(f"Route {i}: " + "-".join(str(v) for v in [0] + r + [0])
                         for i, r in enumerate(self.non_empty_routes(), start=1))
