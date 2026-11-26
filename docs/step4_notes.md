# Step 4 – Genetic Algorithm: notes

Representation: C. Prins (2004), giant tour + Split. 2-opt: G. A. Croes (1958).
Code: `src/cvrp/ga.py` and `src/cvrp/local_search.py`. Tests: `tests/test_ga.py`.

Notation: P = population size, n = customers, K = routes allowed, L = max customers per route (about 13 here, see
`docs/step3_notes.md`), G = generations.

## (a) Pseudocode

```
GENETIC-ALGORITHM(inst, params):
    λ ← params.penalty, or default 2 · mean_i d(0,i)
    pop ← INITIAL-POPULATION
    while generations < max_generations and elapsed < time_limit:
        sort pop by fitness
        new ← the elite_count best of pop                         # elitism
        while |new| < P:
            ADD-UNIQUE(new, OFFSPRING(pop))
        pop ← new
    return the best FEASIBLE individual ever decoded (or None)

DECODE(tour):                                                    # 3-stage rule
    r ← SPLIT(tour)                                              # unlimited
    if |r.routes| ≤ K:              return (r, excess 0)
    r_K ← SPLIT(tour, max_routes = K)
    if r_K exists:                  return (r_K, excess 0)
    return (r, excess = |r.routes| − K)                          # infeasible
    fitness = cost + λ · excess

OFFSPRING(pop):
    p1, p2 ← TOURNAMENT(pop, k), TOURNAMENT(pop, k)
    child ← OX(p1, p2) with probability pc, else a copy of p1
    child ← MUTATE(child) with probability pm
    ind ← DECODE(child)
    if memetic and rand < local_search_prob:
        ind ← DECODE(concatenation of 2-OPT(route) for each route of ind)   # Lamarckian
    return ind

ADD-UNIQUE(new, make):                                           # diversity rule
    repeat up to 20 times:
        ind ← make()
        if ind.tour not in new and no member of new has the same excess and |cost − ind.cost| ≤ 1e-6:
            add ind; return
    add a random immigrant (up to 20 random tours are tried for a non-duplicate one)

INITIAL-POPULATION:
    s ← ⌊seed_fraction · P⌋
    if s ≥ 1: the CW routes concatenated, then s − 1 CW variants
              (routes shuffled, each reversed with probability ½, then ONE random mutation)
    fill the rest with random permutations, all through ADD-UNIQUE

OX(p1, p2):                    # two cut points a < b
    child[a..b] ← p1[a..b]
    fill the other positions, starting at b+1 and wrapping around, with the
    customers not yet used, in the order they appear in p2 starting at b+1

MUTATE(t): one of the following, chosen uniformly:
    SWAP:      exchange t[i] and t[j]
    INVERSION: reverse t[a..b]
    INSERTION: remove t[i] and reinsert it at position j

TOURNAMENT(pop, k): the individual with the lowest fitness among k distinct random ones
```

## (b) Why these operators

- **Giant tour + Split.** Every permutation decodes to the best routes for that order (Step 3). The GA therefore
  works on plain permutations: no repair, and no route delimiters to keep consistent. Some permutation always decodes
  to the optimum, so no part of the search space is lost.
- **Binary tournament.** Selection pressure is mild and easy to tune with k. It needs only comparisons, so the scale
  of the fitness does not matter, and it costs O(k).
- **Order Crossover (OX).** The child is always a valid permutation. OX keeps a whole block of p1, and consecutive
  customers in the giant tour become routes after Split, so good route fragments survive. The rest follows p2's
  relative order. It runs in O(n).
- **Three mutations**, chosen uniformly:
  - swap moves two customers far apart;
  - inversion is a 2-opt move on the giant tour (good for fixing crossings);
  - insertion moves one customer to another route or position.

  Mixing them gives different step sizes without extra parameters.
- **Generational replacement with elitism.** Simple to explain, and the best individuals are never lost.
- **Diversity rule plus immigrants.** Rejecting duplicate tours and duplicate (excess, cost) pairs stops the
  population from filling up with copies of one solution, a common cause of premature convergence. When 20 attempts
  in a row fail, a random immigrant adds new genetic material.
- **Optional memetic 2-opt.** Plain GAs are weak at fine-tuning inside a route. Intra-route 2-opt fixes that cheaply.
  Writing the improved routes back into the chromosome (Lamarckian) lets the improvement be inherited.

## (c) Complexity of ONE generation

| Part | Cost | Note |
|---|---|---|
| Sorting for elitism | O(P log P) | |
| Selection | O(P·k) = O(P) | two tournaments per offspring, k constant |
| Crossover | O(P·n) | OX is O(n) |
| Mutation | O(P·n) | each O(n) because of the copy |
| Decoding | O(P·n·L) | unlimited Split |
| K-limited fallback | O(P·K·n·L) worst case | only for offspring whose unlimited split uses more than K routes |
| Duplicate check | O(P) checks + O(P·n) to hash the tours | hash sets, O(1) per lookup |
| Memetic 2-opt (optional) | O(local_search_prob · P · Σ_r m_r²) per pass | m_r = route length ≤ L, so ≤ O(P·n·L) per pass |
| Retries | at most 20 + 20 decodes per slot | only when duplicates occur; rare on Augerat (0–4 immigrants per run) |

**Total: O(P·n·L) per generation, O(P·K·n·L) in the worst case**, dominated by decoding.
With P = 50 and the Step 3 timings (about 35–105 µs per unlimited split for n = 31–100), we expect a few ms per
generation. Measured in the sanity runs: about 4.8 ms per generation on A-n32-k5 and 10.5 ms on A-n45-k6, where
the K-limited fallback is called often. The per-generation times are stored in `GAResult.gen_times` for the Task 6
log–log plot.

## (d) Parameters (`GAParams`)

| Parameter | Default | Controls |
|---|---|---|
| `seed` | required | the only source of randomness (`random.Random(seed)`) |
| `pop_size` (P) | 50 | diversity versus generations per second |
| `crossover_rate` (pc) | 0.9 | share of offspring made by OX (otherwise a copy of parent 1) |
| `mutation_rate` (pm) | 0.2 | probability that an offspring gets exactly one mutation |
| `elite_count` | 2 | best individuals copied unchanged into the next generation |
| `tournament_size` (k) | 2 | selection pressure |
| `seed_fraction` | 0.1 | share of the initial population built from Clarke–Wright (0 = purely random) |
| `time_limit` | None | stop after this many seconds (checked between generations, so it can overrun by at most one generation) |
| `max_generations` | None | stop after this many generations; at least one stopping rule is required |
| `memetic` | False | apply intra-route 2-opt to some offspring |
| `local_search_prob` | 0.1 | probability that an offspring gets the 2-opt step (when memetic) |
| `penalty` (λ) | None, meaning 2 · mean d(0,i) | fitness cost of each route beyond K |

All of these are candidates for tuning in Step 6. **λ in particular should get a parameter-sensitivity study.**

## (e) Design decisions

**Decode rule (3 stages).** Step 3 showed that most random tours have no split into ≤ K routes:
- 48% of random tours have one on A-n60-k9;
- 26% on A-n80-k10;
- 6% on the team instance.

Using only the K-limited Split would leave most chromosomes without a solution. The unlimited Split is tried first
because it is the cheapest; the K-limited Split is tried only when needed. A chromosome is marked infeasible only
when no split into ≤ K routes exists for its order.

### From pure lexicographic fitness to a penalty

**First version: pure lexicographic fitness.** The brief specified fitness = (excess, cost), compared
lexicographically, so a feasible individual always wins and no penalty constant is needed. In the 10 s sanity
run on A-n45-k6 (seed 0) this gave **cost 1716, gap 81.78%**. A-n32-k5 gave 5.61%.

**Diagnosis.** There were two causes:
1. **The CW seeding never took effect.** Every CW variant (same routes, shuffled order and direction) decodes to the
   same routes and the same cost, 997, as CW itself. The diversity rule rejected them all as duplicates and
   replaced them with random immigrants: 4 immigrants in every run. Only one CW individual survived.
2. **No initial individual was feasible on A-n45-k6.** The CW tour needs 7 routes and has no 6-route split (Step 3).
   Its fitness was (1, 997), and random tours were worse. Under lexicographic comparison, the first feasible
   individual ever found (cost 2411) beat every good infeasible one. Elitism and tournaments then kept selecting
   poor feasible tours, and the good near-feasible material was lost.

**Fixes, tested on 10 s / seed 0 runs before choosing:**

| variant | A-n32-k5 gap | A-n45-k6 gap |
|---|---|---|
| pure lexicographic (first version) | 5.61% | 81.78% |
| fix 1: CW variants also get one mutation | 5.74% | 13.35% |
| fix 2: penalty fitness, λ = 100 | 5.87% | 7.73% |
| fix 1 + fix 2 (adopted) | 5.74% | 7.52% |

(These four rows ran in parallel processes, so their generation counts are not comparable with the final table below.)

**Adopted design** (team decision):
- **CW variants get one random mutation**, so they differ in cost and survive the diversity rule.
- **Fitness = cost + λ · excess** for selection and replacement. A good tour with one extra route can now beat a
  poor feasible one and pass its routes on.
- **Default λ = 2 · mean_i d(0,i)**, the cost of one extra round trip to the depot, computed once per instance.
  This scales with the instance: λ = 120.8 on A-n32-k5, 97.4 on A-n45-k6, 614.4 on the team instance (1000 × 1000
  square plus the Bosphorus penalty). A hardcoded constant would be too weak on the team instance. An explicit
  `GAParams.penalty` overrides the default.
- **Reporting stays strict.** Only individuals with excess = 0 can become the result or enter the progress log. If
  none is ever found, the result's `solution` is None (tested with an impossible K).

### Other decisions

- **Seeding from CW.** A few good starting tours speed up convergence. Keeping them at 10% (and mutated) avoids
  taking over the population. `seed_fraction = 0` gives a purely random start. The tiny-instance tests use it,
  because CW is already optimal on tiny_n7/8/9 and would make that test trivial. Starting at random, the GA still
  reaches the optimum in 5 of 5 runs on n = 7 and n = 8.
- **Immigrants.** They are the fallback of the diversity rule. On tiny instances, where only a few distinct
  solutions exist, an immigrant may itself be a duplicate; `immigrant_duplicates` counts this. It was 0 in all
  Augerat runs.
- **Counters.**
  - `decodes`, `generations` and `gen_times` are exact.
  - `splits_arcs_examined` counts arcs only from Split calls that returned a solution. A K-limited call that
    returns None does not report its arcs; this is a known small under-count.
- **Determinism.** One `random.Random(seed)` drives everything. With `max_generations` and no time limit, two runs
  give identical results (tested).

## Sanity runs (10 s, seed 0, default parameters, final design, sequential)

| instance | variant | λ | cost | gap % | routes / K | checker | generations | decodes | time to best (s) |
|---|---|---|---|---|---|---|---|---|---|
| A-n32-k5 | plain | 120.8 | 829 | 5.74 | 5 / 5 | feasible | 2081 | 107353 | 9.1 |
| A-n32-k5 | memetic | 120.8 | 827 | 5.48 | 5 / 5 | feasible | 1894 | 106875 | 3.0 |
| A-n45-k6 | plain | 97.4 | 1000 | 5.93 | 6 / 6 | feasible | 951 | 47838 | 8.8 |
| A-n45-k6 | memetic | 97.4 | 982 | 4.03 | 6 / 6 | feasible | 833 | 45858 | 2.5 |

For comparison, Clarke–Wright scores 7.40% on A-n32-k5 and 5.61% on A-n45-k6, but there it uses 7 routes > K and
is infeasible. The plain GA's gap of about 5–6% after 10 s and the gain from 2-opt both suggest the defaults are not
yet tuned (Step 6).
