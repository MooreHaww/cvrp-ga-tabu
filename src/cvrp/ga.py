# Original code (representation: Prins 2004)
"""Step 4: Genetic Algorithm on giant tours, decoded by the Split DP.

C. Prins, "A simple and effective evolutionary algorithm for the vehicle
routing problem", Computers & Operations Research 31(12), 2004.

Chromosome = giant tour (a permutation of the customers 1..n). It is decoded
into routes by split.py with this rule (most random tours have no split into
<= K routes, see docs/step3_notes.md):
    1. unlimited Split; if it uses <= K routes -> feasible;
    2. else Split with max_routes=K; if a split exists -> feasible;
    3. else keep the unlimited split, infeasible with excess = routes - K.
Fitness (used for selection and replacement) = cost + lambda * excess, where by
default lambda = 2 * mean d(0, i) (one extra depot round trip). Only feasible
individuals (excess == 0) are ever reported or logged as progress.
A pure lexicographic (excess, cost) fitness was tried first and failed on
A-n45-k6 (81.8% gap); see docs/step4_notes.md.

One generation costs O(P * n * L) (O(P * K * n * L) when the K-limited
fallback runs), P = population size, L = max customers per route.

CLI: python -m cvrp.ga data/cvrplib/A-n32-k5.vrp --seed 0 --time 10 [--memetic]
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field

from cvrp.checker import check
from cvrp.instance import Instance
from cvrp.local_search import improve_solution_2opt
from cvrp.parser import load_instance
from cvrp.savings import clarke_wright, gap_percent
from cvrp.solution import Solution
from cvrp.split import split

MAX_TRIES = 20      # attempts to create a non-duplicate offspring before using an immigrant
COST_SCALE = 1e6    # fitness costs equal within 1e-6 count as duplicates


# --------------------------------------------------------------------------- parameters

@dataclass
class GAParams:
    """GA parameters. Defaults are a starting point; they are tuned in Step 6."""

    seed: int                         # seed of the single random.Random used by the run
    pop_size: int = 50                # P: number of individuals per generation
    crossover_rate: float = 0.9       # probability that an offspring is made by OX (else a copy of parent 1)
    mutation_rate: float = 0.2        # probability that an offspring gets exactly one mutation
    elite_count: int = 2              # best individuals copied unchanged into the next generation
    tournament_size: int = 2          # k in k-tournament selection (2 = binary tournament)
    seed_fraction: float = 0.1        # share of the initial population built from Clarke–Wright routes
    time_limit: float | None = None   # stop after this many seconds (checked between generations)
    max_generations: int | None = None  # stop after this many generations
    memetic: bool = False             # apply 2-opt inside routes to some offspring
    local_search_prob: float = 0.1    # probability that an offspring gets the 2-opt step (if memetic)
    penalty: float | None = None      # lambda per extra route; None -> default_penalty(inst)

    def __post_init__(self) -> None:
        """Validate the parameters. Time: O(1)."""
        if self.time_limit is None and self.max_generations is None:
            raise ValueError("set time_limit and/or max_generations")
        if self.penalty is not None and self.penalty < 0:
            raise ValueError("penalty must be >= 0")
        if self.pop_size < 2 or not 0 <= self.elite_count < self.pop_size:
            raise ValueError("need pop_size >= 2 and 0 <= elite_count < pop_size")
        if not 1 <= self.tournament_size <= self.pop_size:
            raise ValueError("tournament_size must be in 1..pop_size")
        for name in ("crossover_rate", "mutation_rate", "seed_fraction", "local_search_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass
class Individual:
    """A decoded chromosome."""

    tour: tuple[int, ...]       # giant tour (chromosome)
    routes: list[list[int]]     # decoded routes
    cost: float                 # total distance of the routes
    excess: int                 # max(0, number of routes - K); 0 = feasible
    fitness: float              # cost + lambda * excess (smaller is better)

    @property
    def key(self) -> tuple[int, float]:
        """(excess, cost), used by the diversity rule and for reporting. Time: O(1)."""
        return (self.excess, self.cost)


@dataclass
class GAResult:
    """Output of genetic_algorithm()."""

    solution: Solution | None            # best FEASIBLE solution, None if none was ever found
    best_cost: float | None
    best_infeasible_fitness: tuple[int, float] | None  # best (excess, cost) among infeasible decodes
    n_routes: int | None
    generations: int
    decodes: int                          # number of chromosomes decoded
    splits_arcs_examined: int             # DAG arcs evaluated by Split (successful splits only)
    elapsed: float
    progress: list[tuple[float, float]]   # (elapsed_seconds, best_feasible_cost), on improvement only
    gen_times: list[float]                # seconds per generation
    params: GAParams
    seed: int
    penalty_lambda: float = 0.0           # lambda actually used (default formula or params.penalty)
    immigrants: int = 0                   # random individuals inserted by the diversity rule
    immigrant_duplicates: int = 0         # immigrants accepted although duplicate (tiny n only)
    final_population: list[tuple[int, float]] = field(default_factory=list)  # sorted (excess, cost)


# --------------------------------------------------------------------------- decoder

def default_penalty(inst: Instance) -> float:
    """Default lambda = 2 * mean d(0, i) over the customers: the cost of one extra
    depot round trip, so it scales with the instance. Time: O(n)."""
    return 2.0 * float(inst.dist[0, 1:].mean())


def decode(inst: Instance, tour: list[int], penalty: float) -> tuple[Individual, int]:
    """Decode a giant tour with the 3-stage rule; returns (individual, arcs examined).

    Fitness = cost + penalty * excess.
    Time: O(n * L) for the unlimited Split, plus O(K * n * L) only when it
    uses more than K routes and the K-limited Split is tried.
    """
    unlimited = split(inst, tour)
    arcs = unlimited.arcs_examined
    routes = unlimited.solution.routes
    if len(routes) <= inst.K:                                    # stage 1
        return Individual(tuple(tour), routes, unlimited.cost, 0, unlimited.cost), arcs
    limited = split(inst, tour, max_routes=inst.K)               # stage 2
    if limited is not None:
        return (Individual(tuple(tour), limited.solution.routes, limited.cost, 0, limited.cost),
                arcs + limited.arcs_examined)
    excess = len(routes) - inst.K                                # stage 3: infeasible
    return Individual(tuple(tour), routes, unlimited.cost, excess, unlimited.cost + penalty * excess), arcs


# --------------------------------------------------------------------------- operators

def tournament(pop: list[Individual], k: int, rng: random.Random) -> Individual:
    """k-tournament: best fitness among k distinct random individuals. Time: O(k)."""
    return min(rng.sample(pop, k), key=lambda ind: ind.fitness)


def order_crossover(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
    """Order Crossover (OX), two cut points.

    The child keeps p1[a..b] in place; the other positions are filled, starting
    right after b and wrapping around, with the remaining customers in the order
    they appear in p2 (also read from b+1, wrapping). Time: O(n).
    """
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child: list[int | None] = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    used = set(p1[a:b + 1])
    pos = (b + 1) % n
    for k in range(n):
        v = p2[(b + 1 + k) % n]
        if v not in used:
            child[pos] = v
            pos = (pos + 1) % n
    return child  # type: ignore[return-value]


def swap_mutation(tour: list[int], rng: random.Random) -> list[int]:
    """Exchange two random positions. Time: O(n) for the copy, O(1) for the move."""
    t = list(tour)
    i, j = rng.sample(range(len(t)), 2)
    t[i], t[j] = t[j], t[i]
    return t


def inversion_mutation(tour: list[int], rng: random.Random) -> list[int]:
    """Reverse the segment between two random positions (a 2-opt move on the tour). Time: O(n)."""
    t = list(tour)
    a, b = sorted(rng.sample(range(len(t)), 2))
    t[a:b + 1] = t[a:b + 1][::-1]
    return t


def insertion_mutation(tour: list[int], rng: random.Random) -> list[int]:
    """Remove a random customer and reinsert it at another random position. Time: O(n)."""
    t = list(tour)
    i, j = rng.sample(range(len(t)), 2)
    t.insert(j, t.pop(i))
    return t


MUTATIONS = (swap_mutation, inversion_mutation, insertion_mutation)


def mutate(tour: list[int], rng: random.Random) -> list[int]:
    """Apply one mutation chosen uniformly among swap, inversion, insertion. Time: O(n)."""
    return rng.choice(MUTATIONS)(tour, rng)


# --------------------------------------------------------------------------- diversity

class _Registry:
    """Giant tours and (excess, cost) keys of the population being built (hash sets, O(1) per check)."""

    def __init__(self) -> None:
        self.tours: set[tuple[int, ...]] = set()
        self.keys: set[tuple[int, int]] = set()

    @staticmethod
    def _key(ind: Individual) -> tuple[int, int]:
        """(excess, cost in units of 1e-6). Time: O(1)."""
        return ind.excess, round(ind.cost * COST_SCALE)

    def is_duplicate(self, ind: Individual) -> bool:
        """Same tour, or same excess and cost within 1e-6 (checks the neighbouring keys). Time: O(1)."""
        if ind.tour in self.tours:
            return True
        excess, c = self._key(ind)
        return any((excess, c + delta) in self.keys for delta in (-1, 0, 1))

    def add(self, ind: Individual) -> None:
        """Register an accepted individual. Time: O(1) (hashing the tour is O(n))."""
        self.tours.add(ind.tour)
        self.keys.add(self._key(ind))


# --------------------------------------------------------------------------- the run

class _GARun:
    """State of one GA run: instance, parameters, random source, counters, best solution."""

    def __init__(self, inst: Instance, params: GAParams) -> None:
        self.inst, self.p = inst, params
        self.penalty = params.penalty if params.penalty is not None else default_penalty(inst)
        self.rng = random.Random(params.seed)
        self.start = time.perf_counter()
        self.decodes = self.arcs = self.immigrants = self.immigrant_duplicates = 0
        self.best: Individual | None = None                 # best feasible ever seen
        self.best_infeasible: tuple[int, float] | None = None
        self.progress: list[tuple[float, float]] = []

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def evaluate(self, tour: list[int]) -> Individual:
        """Decode, count, and update the best-so-far records.

        Only feasible individuals (excess == 0) can become the reported best and
        enter the progress log. Time: as decode().
        """
        ind, arcs = decode(self.inst, tour, self.penalty)
        self.decodes += 1
        self.arcs += arcs
        if ind.excess == 0:
            if self.best is None or ind.cost < self.best.cost - 1e-9:
                self.best = ind
                self.progress.append((self.elapsed(), ind.cost))
        elif self.best_infeasible is None or ind.key < self.best_infeasible:
            self.best_infeasible = ind.key
        return ind

    def random_tour(self) -> list[int]:
        """Uniform random permutation of 1..n. Time: O(n)."""
        t = list(range(1, self.inst.n + 1))
        self.rng.shuffle(t)
        return t

    def cw_variant(self, cw_routes: list[list[int]]) -> list[int]:
        """CW routes in random order, each reversed with probability 1/2, then ONE random mutation.

        Without the mutation every variant decodes to the same routes and cost
        as CW itself, so the diversity rule rejects all of them. Time: O(n).
        """
        routes = [list(r) for r in cw_routes]
        self.rng.shuffle(routes)
        tour = [v for r in routes for v in (r[::-1] if self.rng.random() < 0.5 else r)]
        return mutate(tour, self.rng)

    def offspring(self, pop: list[Individual]) -> Individual:
        """Selection, OX, mutation, decoding (+ optional Lamarckian 2-opt). Time: O(n * L) typical."""
        p1 = tournament(pop, self.p.tournament_size, self.rng)
        p2 = tournament(pop, self.p.tournament_size, self.rng)
        if self.rng.random() < self.p.crossover_rate:
            child = order_crossover(list(p1.tour), list(p2.tour), self.rng)
        else:
            child = list(p1.tour)
        if self.rng.random() < self.p.mutation_rate:
            child = mutate(child, self.rng)
        ind = self.evaluate(child)
        if self.p.memetic and self.rng.random() < self.p.local_search_prob:
            improved = improve_solution_2opt(self.inst, ind.routes)
            ind = self.evaluate([v for r in improved for v in r])   # write back and re-decode
        return ind

    def add_unique(self, pop: list[Individual], reg: _Registry, make) -> None:
        """Append make()'s individual unless it duplicates one in pop; after MAX_TRIES
        failures append a random immigrant instead. Time: O(MAX_TRIES) decodes worst case."""
        for _ in range(MAX_TRIES):
            ind = make()
            if not reg.is_duplicate(ind):
                pop.append(ind)
                reg.add(ind)
                return
        self.immigrants += 1
        for _ in range(MAX_TRIES):
            ind = self.evaluate(self.random_tour())
            if not reg.is_duplicate(ind):
                break
        else:
            self.immigrant_duplicates += 1      # only possible when almost all tours are already present
        pop.append(ind)
        reg.add(ind)

    def initial_population(self) -> list[Individual]:
        """CW tour, CW variants (seed_fraction of P in total), then random tours. Time: O(P * n * L) + CW.

        seed_fraction = 0 gives a purely random initial population (no Clarke–Wright at all).
        """
        n_seeded = int(self.p.seed_fraction * self.p.pop_size)
        pop: list[Individual] = []
        reg = _Registry()
        if n_seeded >= 1:
            cw_routes = clarke_wright(self.inst, warn=False).solution.routes
            self.add_unique(pop, reg, lambda: self.evaluate([v for r in cw_routes for v in r]))
            while len(pop) < n_seeded:
                self.add_unique(pop, reg, lambda: self.evaluate(self.cw_variant(cw_routes)))
        while len(pop) < self.p.pop_size:
            self.add_unique(pop, reg, lambda: self.evaluate(self.random_tour()))
        return pop

    def next_generation(self, pop: list[Individual]) -> list[Individual]:
        """Generational replacement with elitism and the diversity rule.

        Time: O(P log P) sort + O(P) offspring, each O(n) for OX/mutation and
        O(n * L) to decode -> O(P * n * L) (O(P * K * n * L) with the fallback).
        """
        pop = sorted(pop, key=lambda ind: ind.fitness)
        new = pop[:self.p.elite_count]
        reg = _Registry()
        for ind in new:
            reg.add(ind)
        while len(new) < self.p.pop_size:
            self.add_unique(new, reg, lambda: self.offspring(pop))
        return new

    def stop(self, generations: int) -> bool:
        """True when max_generations or time_limit is reached. Time: O(1)."""
        if self.p.max_generations is not None and generations >= self.p.max_generations:
            return True
        return self.p.time_limit is not None and self.elapsed() >= self.p.time_limit


def genetic_algorithm(inst: Instance, params: GAParams) -> GAResult:
    """Run the GA until max_generations or time_limit (whichever comes first).

    Time: O(G * P * n * L) for G generations (K-limited fallback: O(G * P * K * n * L)).
    The time limit is checked between generations, so a run can exceed it by
    at most one generation.
    """
    run = _GARun(inst, params)
    pop = run.initial_population()
    generations = 0
    gen_times: list[float] = []
    while not run.stop(generations):
        t0 = time.perf_counter()
        pop = run.next_generation(pop)
        generations += 1
        gen_times.append(time.perf_counter() - t0)

    best = run.best
    return GAResult(
        solution=Solution([list(r) for r in best.routes]) if best else None,
        best_cost=best.cost if best else None,
        best_infeasible_fitness=run.best_infeasible,
        n_routes=len(best.routes) if best else None,
        generations=generations, decodes=run.decodes, splits_arcs_examined=run.arcs,
        elapsed=run.elapsed(), progress=run.progress, gen_times=gen_times,
        params=params, seed=params.seed, penalty_lambda=run.penalty, immigrants=run.immigrants,
        immigrant_duplicates=run.immigrant_duplicates,
        final_population=sorted(ind.key for ind in pop))


def main() -> None:
    """Command-line entry point (see module docstring)."""
    ap = argparse.ArgumentParser(description="Genetic Algorithm (giant tour + Split) for the CVRP.")
    ap.add_argument("path", help=".vrp file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", type=float, default=None, help="time limit in seconds (default 10 if no --gens)")
    ap.add_argument("--gens", type=int, default=None, help="max generations")
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--pc", type=float, default=0.9, help="crossover rate")
    ap.add_argument("--pm", type=float, default=0.2, help="mutation rate")
    ap.add_argument("--memetic", action="store_true", help="enable intra-route 2-opt on offspring")
    ap.add_argument("--penalty", type=float, default=None, help="lambda per extra route (default 2*mean d(0,i))")
    args = ap.parse_args()
    time_limit = args.time if args.time is not None or args.gens is not None else 10.0

    inst = load_instance(args.path)
    params = GAParams(seed=args.seed, pop_size=args.pop, crossover_rate=args.pc, mutation_rate=args.pm,
                      time_limit=time_limit, max_generations=args.gens, memetic=args.memetic,
                      penalty=args.penalty)
    res = genetic_algorithm(inst, params)

    print(f"instance    : {inst.name} (n={inst.n}, Q={inst.Q}, K={inst.K})")
    print(f"generations : {res.generations}   decodes: {res.decodes}   immigrants: {res.immigrants}   "
          f"lambda: {res.penalty_lambda:.2f}")
    print(f"runtime     : {res.elapsed:.2f} s")
    if res.solution is None:
        print(f"WARNING: no feasible solution (<= K routes) found; best infeasible (excess, cost) = "
              f"{res.best_infeasible_fitness}")
        return
    gap = gap_percent(res.best_cost, inst.bks)
    report = check(inst, res.solution, claimed_cost=res.best_cost)
    print(f"cost        : {res.best_cost:.2f}")
    print(f"routes      : {res.n_routes} (K = {inst.K})")
    print(f"gap         : {f'{gap:.2f} %' if gap is not None else '-'}")
    print(f"checker     : {'feasible' if report.feasible else 'INFEASIBLE'}")
    for err in report.errors:
        print(f"  - {err}")
    print(res.solution.describe(inst))


if __name__ == "__main__":
    main()
