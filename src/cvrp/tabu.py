# Original code (Tabu Search: Glover & Laguna 1997; granular neighborhoods: Toth & Vigo 2003)
"""Step 5: Tabu Search on route-based solutions.

F. Glover & M. Laguna, Tabu Search, Kluwer, 1997.
P. Toth & D. Vigo, "The granular tabu search and its application to the
vehicle-routing problem", INFORMS Journal on Computing 15(4), 2003.

State: exactly K route slots (some may be empty). Soft capacity:
    f = distance + lambda * total_overload,   overload = sum_r max(0, load_r - Q),
with lambda adapted every `adapt_every` iterations. Moves: relocate, swap,
2-opt* and intra-route 2-opt, all with O(1) deltas (local_search.py), restricted
to a granular candidate list (each customer's `candidate_size` nearest customers).
Tabu attribute: arcs. Removed arcs become tabu; a move is tabu if it would add
a tabu arc, unless it gives a new best FEASIBLE solution (aspiration).
Only feasible solutions (overload 0) are ever reported.

One iteration: O(n * kappa * m) delta evaluations (kappa = candidate_size,
m = number of move types) + O(L + K) to apply the chosen move. After
`repair_after` infeasible iterations in a row, repair moves from the overloaded
routes to ALL positions are added: + O(L_over * n) evaluations.

CLI: python -m cvrp.tabu data/cvrplib/A-n32-k5.vrp --seed 0 --time 10
     python -m cvrp.tabu --benchmark-delta
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from dataclasses import dataclass, field

from cvrp.checker import check
from cvrp.instance import Instance
from cvrp.local_search import (apply_relocate, apply_swap, apply_two_opt_intra, apply_two_opt_star,
                               full_distance, relocate_added_arcs, relocate_delta, route_arcs,
                               solution_overload, swap_added_arcs, swap_delta, two_opt_intra_added_arcs,
                               two_opt_intra_delta, two_opt_star_added_arcs, two_opt_star_delta)
from cvrp.parser import load_instance
from cvrp.savings import clarke_wright, gap_percent
from cvrp.solution import Solution

MOVE_TYPES = ("relocate", "swap", "two_opt_star", "two_opt")
EPS = 1e-9


# --------------------------------------------------------------------------- parameters

@dataclass
class TabuParams:
    """Tabu Search parameters (defaults are a starting point, tuned in Step 6)."""

    seed: int                                  # seed of the single random.Random of the run
    tenure_min: int = 5                        # tabu tenure is drawn uniformly from [tenure_min, tenure_max]
    tenure_max: int = 15
    candidate_size: int | None = 10            # kappa: nearest neighbours per customer (None = all customers)
    lam_factor: float = 1.5                    # lambda is multiplied / divided by this when adapted
    adapt_every: int = 10                      # window length (iterations) for adapting lambda
    lam_min: float = 0.01                      # lambda is clamped to [lam_min, lam_max]
    lam_max: float = 1e4
    moves: tuple[str, ...] = MOVE_TYPES        # neighbourhoods used
    time_limit: float | None = None            # seconds (checked every iteration)
    max_iterations: int | None = None
    verify_deltas: bool = False                # tests only: recompute f after every move and compare
    repair_after: int | None = 10              # after this many infeasible iterations in a row, also try
                                               # repair moves from overloaded routes to ALL positions (None = off)

    def __post_init__(self) -> None:
        """Validate. Time: O(1)."""
        if self.time_limit is None and self.max_iterations is None:
            raise ValueError("set time_limit and/or max_iterations")
        unknown = set(self.moves) - set(MOVE_TYPES)
        if unknown or not self.moves:
            raise ValueError(f"moves must be a non-empty subset of {MOVE_TYPES}, got {self.moves}")
        if not 0 <= self.tenure_min <= self.tenure_max:
            raise ValueError("need 0 <= tenure_min <= tenure_max")
        if self.lam_factor <= 1 or self.adapt_every < 1 or not 0 < self.lam_min <= self.lam_max:
            raise ValueError("invalid lambda settings")
        if self.candidate_size is not None and self.candidate_size < 1:
            raise ValueError("candidate_size must be >= 1 or None")
        if self.repair_after is not None and self.repair_after < 0:
            raise ValueError("repair_after must be >= 0 or None")


@dataclass
class TabuResult:
    """Output of tabu_search()."""

    solution: Solution | None          # best FEASIBLE solution (non-empty routes), None if never feasible
    cost: float | None
    n_routes: int | None
    iterations: int
    moves_evaluated: int
    tabu_moves_forced: int             # iterations where every move was tabu (best tabu move applied)
    aspirations_used: int              # applied moves that were tabu but allowed by aspiration
    elapsed: float
    progress: list[tuple[float, float]] = field(default_factory=list)  # (elapsed, best feasible distance)
    final_lambda: float = 0.0
    params: TabuParams | None = None
    seed: int = 0


# --------------------------------------------------------------------------- small testable parts

class TabuList:
    """Arc-based tabu list: arc -> last iteration in which it is tabu. O(1) per arc."""

    def __init__(self, tenure_min: int, tenure_max: int) -> None:
        self.tenure_min, self.tenure_max = tenure_min, tenure_max
        self.until: dict[tuple[int, int], int] = {}

    def draw_tenure(self, rng: random.Random) -> int:
        """Random tenure in [tenure_min, tenure_max]. Time: O(1)."""
        return rng.randint(self.tenure_min, self.tenure_max)

    def forbid(self, arcs, iteration: int, rng: random.Random) -> int:
        """Make the arcs tabu for the next `tenure` iterations; returns the tenure. Time: O(#arcs)."""
        tenure = self.draw_tenure(rng)
        for a in arcs:
            self.until[a] = iteration + tenure
        return tenure

    def is_tabu(self, arcs, iteration: int) -> bool:
        """True if any of the arcs is still tabu at this iteration. Time: O(#arcs)."""
        return any(self.until.get(a, -1) >= iteration for a in arcs)


def is_admissible(tabu: bool, new_overload: float, new_distance: float, best_feasible: float) -> bool:
    """Non-tabu moves are admissible; a tabu move only by aspiration: the result is
    feasible and strictly better than the best feasible distance so far. Time: O(1)."""
    return (not tabu) or (new_overload == 0 and new_distance < best_feasible - EPS)


class LambdaController:
    """Adaptive penalty: every `every` recorded iterations, lambda /= factor if all
    were feasible, lambda *= factor if all were infeasible; then clamp. O(1) per record."""

    def __init__(self, lam0: float, factor: float, every: int, lam_min: float, lam_max: float) -> None:
        self.factor, self.every, self.lam_min, self.lam_max = factor, every, lam_min, lam_max
        self.lam = self.clamp(lam0)
        self.window: list[bool] = []

    def clamp(self, lam: float) -> float:
        return min(max(lam, self.lam_min), self.lam_max)

    def record(self, feasible: bool) -> bool:
        """Record one iteration's feasibility; returns True if lambda changed."""
        self.window.append(feasible)
        if len(self.window) < self.every:
            return False
        old = self.lam
        if all(self.window):
            self.lam = self.clamp(self.lam / self.factor)
        elif not any(self.window):
            self.lam = self.clamp(self.lam * self.factor)
        self.window = []
        return self.lam != old


def neighbours(D, n: int, k: int | None) -> list[list[int]]:
    """Granular candidate list: the k nearest customers of every customer (k=None: all).
    Index 0 (depot) gets an empty list. Time: O(n^2 log n), done once."""
    result: list[list[int]] = [[]]
    for c in range(1, n + 1):
        others = sorted((v for v in range(1, n + 1) if v != c), key=lambda v: (D[c][v], v))
        result.append(others if k is None else others[:k])
    return result


def initial_routes(inst: Instance) -> tuple[list[list[int]], float]:
    """Clarke–Wright routes, adjusted to exactly K slots; also returns the CW distance.

    More than K routes: repeatedly concatenate the two routes with the smallest
    loads (overload allowed). Fewer: pad with empty routes. Time: O(n^2 log n) for CW.
    """
    cw = clarke_wright(inst, warn=False)
    routes = [list(r) for r in cw.solution.routes]
    load = lambda r: sum(int(inst.demand[v]) for v in r)
    while len(routes) > inst.K:
        routes.sort(key=lambda r: (load(r), r[0]))
        a, b = routes.pop(0), routes.pop(0)
        routes.append(a + b)
    routes += [[] for _ in range(inst.K - len(routes))]
    return routes, cw.cost


# --------------------------------------------------------------------------- search state

class _State:
    """K route slots with loads, prefix loads, positions and incremental objective parts."""

    def __init__(self, inst: Instance, routes: list[list[int]]) -> None:
        self.D = inst.dist.tolist()                     # nested lists: fast O(1) lookups
        self.q = [int(x) for x in inst.demand]
        self.Q, self.n = inst.Q, inst.n
        self.routes = [list(r) for r in routes]
        self.loads = [0] * len(routes)
        self.prefix: list[list[int]] = [[0] for _ in routes]
        self.where = [(-1, -1)] * (inst.n + 1)          # where[c] = (route index, position)
        for r in range(len(routes)):
            self._refresh(r)
        self.dist = full_distance(self.D, self.routes)
        self.over = solution_overload(self.q, self.Q, self.routes)

    def _refresh(self, r: int) -> None:
        """Recompute load, prefix loads and positions of route r. Time: O(len(route))."""
        route = self.routes[r]
        pre = [0]
        for k, c in enumerate(route):
            pre.append(pre[-1] + self.q[c])
            self.where[c] = (r, k)
        self.prefix[r] = pre
        self.loads[r] = pre[-1]

    def delta(self, kind: str, args: tuple) -> tuple[float, int]:
        """O(1) (delta_distance, delta_overload) of a move."""
        D, q, Q, routes, loads = self.D, self.q, self.Q, self.routes, self.loads
        if kind == "relocate":
            return relocate_delta(D, q, Q, routes, loads, *args)
        if kind == "swap":
            return swap_delta(D, q, Q, routes, loads, *args)
        if kind == "two_opt_star":
            return two_opt_star_delta(D, Q, routes, loads, self.prefix, *args)
        return two_opt_intra_delta(D, routes, *args)

    def has_arc(self, a: tuple[int, int]) -> bool:
        """Is the (unordered) arc in the current solution? Uses the position array. Time: O(1)."""
        u, v = a
        if u == 0:                                       # depot arc: v starts or ends its route
            r, k = self.where[v]
            return k == 0 or k == len(self.routes[r]) - 1
        (ru, ku), (rv, kv) = self.where[u], self.where[v]
        return ru == rv and abs(ku - kv) == 1

    def is_null_move(self, arcs: list[tuple[int, int]]) -> bool:
        """A move that adds no NEW arc leaves the solution unchanged (e.g. swapping the two
        customers of a 2-customer route = reversing it). Such moves remove no arc, so they
        can never become tabu, and must be skipped or they stall the search. Time: O(#arcs)."""
        return all(self.has_arc(a) for a in arcs)

    def added_arcs(self, kind: str, args: tuple) -> list[tuple[int, int]]:
        """Arcs the move would create. Time: O(1)."""
        if kind == "relocate":
            return relocate_added_arcs(self.routes, *args)
        if kind == "swap":
            return swap_added_arcs(self.routes, *args)
        if kind == "two_opt_star":
            return two_opt_star_added_arcs(self.routes, *args)
        return two_opt_intra_added_arcs(self.routes, *args)

    def apply(self, kind: str, args: tuple, dd: float, dover: int) -> set[tuple[int, int]]:
        """Apply a move, update everything incrementally; return the arcs it removed. Time: O(K + L)."""
        touched = {args[0]} if kind == "two_opt" else {args[0], args[2]}
        before = set().union(*(route_arcs(self.routes[r]) for r in touched))
        if kind == "relocate":
            self.routes = apply_relocate(self.routes, *args)
        elif kind == "swap":
            self.routes = apply_swap(self.routes, *args)
        elif kind == "two_opt_star":
            self.routes = apply_two_opt_star(self.routes, *args)
        else:
            self.routes = apply_two_opt_intra(self.routes, *args)
        for r in touched:
            self._refresh(r)
        self.dist += dd
        self.over += dover
        after = set().union(*(route_arcs(self.routes[r]) for r in touched))
        return before - after

    def candidates(self, neigh: list[list[int]], moves: set[str]):
        """Granular candidate moves (generator). Each move makes c adjacent to one of its
        neighbours nb (or, for the empty-route options, to the depot). Time: O(n * kappa) moves."""
        routes, where = self.routes, self.where
        empty = next((r for r, route in enumerate(routes) if not route), None)
        for c in range(1, self.n + 1):
            r, i = where[c]
            for nb in neigh[c]:
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
            if empty is not None:
                if "relocate" in moves and len(routes[r]) > 1:
                    yield "relocate", (r, i, empty, 0)            # open a new route with c
                if "two_opt_star" in moves and i + 1 < len(routes[r]):
                    yield "two_opt_star", (r, i + 1, empty, 0)    # split the route after c

    def repair_candidates(self, moves: set[str]):
        """Repair moves, NOT restricted to the granular list: every customer of an
        overloaded route may be relocated to any position of any other route, or
        swapped with any customer of another route.

        Needed on tight instances, where the only routes with spare capacity are
        far away and therefore never among a customer's nearest neighbours.
        Time: O(L_over * n) moves, L_over = customers on overloaded routes.
        """
        routes = self.routes
        for r1, route1 in enumerate(routes):
            if self.loads[r1] <= self.Q:
                continue
            for i in range(len(route1)):
                for r2, route2 in enumerate(routes):
                    if r2 == r1:
                        continue
                    if "relocate" in moves:
                        for j in range(len(route2) + 1):
                            yield "relocate", (r1, i, r2, j)
                    if "swap" in moves:
                        for j in range(len(route2)):
                            yield "swap", (r1, i, r2, j)


# --------------------------------------------------------------------------- main loop

def tabu_search(inst: Instance, params: TabuParams) -> TabuResult:
    """Granular Tabu Search with soft capacity and adaptive penalty.

    Time per iteration: O(n * kappa * m) O(1) delta evaluations + O(K + L) to
    apply the move (m = number of move types, kappa = candidate_size).
    """
    start = time.perf_counter()
    rng = random.Random(params.seed)
    routes, cw_cost = initial_routes(inst)
    state = _State(inst, routes)
    neigh = neighbours(state.D, inst.n, params.candidate_size)
    moves = set(params.moves)
    tabu = TabuList(params.tenure_min, params.tenure_max)
    lam_ctrl = LambdaController(cw_cost / inst.total_demand(), params.lam_factor, params.adapt_every,
                                params.lam_min, params.lam_max)
    # The penalised objective f = state.dist + lambda * state.over is never stored: dist and over are
    # kept incrementally, so f is always recomputed with the CURRENT lambda (also right after it changes).

    best_routes, best_dist, progress = None, float("inf"), []
    if state.over == 0:
        best_routes, best_dist = [list(r) for r in state.routes if r], state.dist
        progress.append((time.perf_counter() - start, best_dist))

    it = evaluated = forced = aspirations = 0
    infeasible_streak = 0                                   # consecutive iterations ending infeasible
    while True:
        if params.max_iterations is not None and it >= params.max_iterations:
            break
        if params.time_limit is not None and time.perf_counter() - start >= params.time_limit:
            break
        it += 1
        lam = lam_ctrl.lam
        best_move = None          # (pen, kind, args, dd, dover, by_aspiration)
        ties = 0
        best_tabu = None          # (pen, kind, args, dd, dover)
        candidates = state.candidates(neigh, moves)
        if params.repair_after is not None and state.over > 0 and infeasible_streak >= params.repair_after:
            candidates = itertools.chain(candidates, state.repair_candidates(moves))
        for kind, args in candidates:
            dd, dover = state.delta(kind, args)
            evaluated += 1
            pen = dd + lam * dover
            if best_move is not None and pen > best_move[0] + EPS:
                continue                                    # cannot win: skip the tabu check
            added = state.added_arcs(kind, args)
            if state.is_null_move(added):
                continue                                    # changes nothing: never select it
            is_tabu_move = tabu.is_tabu(added, it)
            if not is_admissible(is_tabu_move, state.over + dover, state.dist + dd, best_dist):
                if best_tabu is None or pen < best_tabu[0]:
                    best_tabu = (pen, kind, args, dd, dover)
                continue
            if best_move is None or pen < best_move[0] - EPS:
                best_move, ties = (pen, kind, args, dd, dover, is_tabu_move), 1
            else:                                           # tie: reservoir sampling, uniform choice
                ties += 1
                if rng.random() < 1.0 / ties:
                    best_move = (pen, kind, args, dd, dover, is_tabu_move)

        if best_move is None:
            if best_tabu is None:
                break                                       # no move exists at all
            forced += 1
            _, kind, args, dd, dover = best_tabu
        else:
            _, kind, args, dd, dover, by_aspiration = best_move
            aspirations += by_aspiration

        removed = state.apply(kind, args, dd, dover)
        tabu.forbid(removed, it, rng)
        if params.verify_deltas:
            full_d = full_distance(state.D, state.routes)
            full_o = solution_overload(state.q, state.Q, state.routes)
            assert abs(full_d - state.dist) <= 1e-6 and full_o == state.over, \
                f"incremental ({state.dist}, {state.over}) != full ({full_d}, {full_o}) after {kind}{args}"

        lam_ctrl.record(state.over == 0)                    # may change lambda for the next iteration
        infeasible_streak = infeasible_streak + 1 if state.over > 0 else 0
        if state.over == 0 and state.dist < best_dist - EPS:
            best_routes, best_dist = [list(r) for r in state.routes if r], state.dist
            progress.append((time.perf_counter() - start, best_dist))

    return TabuResult(
        solution=Solution(best_routes) if best_routes is not None else None,
        cost=best_dist if best_routes is not None else None,
        n_routes=len(best_routes) if best_routes is not None else None,
        iterations=it, moves_evaluated=evaluated, tabu_moves_forced=forced,
        aspirations_used=aspirations, elapsed=time.perf_counter() - start,
        progress=progress, final_lambda=lam_ctrl.lam, params=params, seed=params.seed)


# --------------------------------------------------------------------------- delta benchmark

def benchmark_delta(n_moves: int = 2000, seed: int = 0) -> list[tuple[str, int, float, float]]:
    """Mean time of one random relocate evaluated with the O(1) delta versus a full
    re-evaluation (copy + apply + recompute distance and overload, O(n)).
    Returns rows (name, n, us_delta, us_full). Time: O(n_moves * n) per instance."""
    paths = ["data/cvrplib/A-n32-k5.vrp", "data/cvrplib/A-n60-k9.vrp",
             "data/cvrplib/A-n80-k10.vrp", "data/team/team_n100_seed0.vrp"]
    rng = random.Random(seed)
    rows = []
    for path in paths:
        inst = load_instance(path)
        state = _State(inst, initial_routes(inst)[0])
        D, q, Q, routes, loads = state.D, state.q, state.Q, state.routes, state.loads
        moves = []
        non_empty = [r for r, route in enumerate(routes) if route]
        for _ in range(n_moves):
            r1 = rng.choice(non_empty)
            i = rng.randrange(len(routes[r1]))
            r2 = rng.randrange(len(routes))
            j = rng.randrange(len(routes[r2]) + (0 if r2 == r1 else 1))
            moves.append((r1, i, r2, j))
        lam = 1.0
        t0 = time.perf_counter()
        for m in moves:
            dd, dover = relocate_delta(D, q, Q, routes, loads, *m)
            _ = dd + lam * dover
        t_delta = (time.perf_counter() - t0) / n_moves * 1e6
        t0 = time.perf_counter()
        for m in moves:
            new = apply_relocate(routes, *m)
            _ = full_distance(D, new) + lam * solution_overload(q, Q, new)
        t_full = (time.perf_counter() - t0) / n_moves * 1e6
        rows.append((inst.name, inst.n, t_delta, t_full))
    return rows


def main() -> None:
    """Command-line entry point (see module docstring)."""
    ap = argparse.ArgumentParser(description="Tabu Search for the CVRP.")
    ap.add_argument("path", nargs="?", help=".vrp file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", type=float, default=None, help="time limit (default 10 s if no --iters)")
    ap.add_argument("--iters", type=int, default=None, help="max iterations")
    ap.add_argument("--tenure-min", type=int, default=5)
    ap.add_argument("--tenure-max", type=int, default=15)
    ap.add_argument("--cand", type=int, default=10, help="candidate list size (0 = all customers)")
    ap.add_argument("--moves", default=",".join(MOVE_TYPES))
    ap.add_argument("--benchmark-delta", action="store_true")
    args = ap.parse_args()

    if args.benchmark_delta:
        print("instance | n | us per move (O(1) delta) | us per move (full re-evaluation) | ratio")
        for name, n, t_d, t_f in benchmark_delta():
            print(f"{name} | {n} | {t_d:.2f} | {t_f:.2f} | {t_f / t_d:.1f}x")
        return
    if not args.path:
        ap.error("path is required unless --benchmark-delta is given")

    time_limit = args.time if args.time is not None or args.iters is not None else 10.0
    inst = load_instance(args.path)
    params = TabuParams(seed=args.seed, tenure_min=args.tenure_min, tenure_max=args.tenure_max,
                        candidate_size=args.cand or None, moves=tuple(args.moves.split(",")),
                        time_limit=time_limit, max_iterations=args.iters)
    res = tabu_search(inst, params)
    print(f"instance    : {inst.name} (n={inst.n}, Q={inst.Q}, K={inst.K})")
    print(f"iterations  : {res.iterations} ({res.iterations / res.elapsed:.0f} it/s), "
          f"moves evaluated {res.moves_evaluated}, forced {res.tabu_moves_forced}, "
          f"aspirations {res.aspirations_used}, final lambda {res.final_lambda:.3f}")
    print(f"runtime     : {res.elapsed:.2f} s")
    if res.solution is None:
        print("WARNING: no feasible solution found")
        return
    gap = gap_percent(res.cost, inst.bks)
    report = check(inst, res.solution, claimed_cost=res.cost)
    print(f"cost        : {res.cost:.2f}")
    print(f"routes      : {res.n_routes} (K = {inst.K})")
    print(f"gap         : {f'{gap:.2f} %' if gap is not None else '-'}")
    print(f"checker     : {'feasible' if report.feasible else 'INFEASIBLE'}")
    for err in report.errors:
        print(f"  - {err}")
    print(res.solution.describe(inst))


if __name__ == "__main__":
    main()
