# Step 4c – Memetic GA with inter-route local search: notes

Code:
- `src/cvrp/local_search.py`, part 3: `neighbours`, `customer_moves`, `move_delta`, `apply_move` and `descent`;
- `src/cvrp/ga.py`: `ls_mode`, `ls_neighbours`, and the counters `local_searches` / `ls_time`;
- `experiments/config.py`: the algorithm `ga_ls`.

Tests: `tests/test_local_search.py` and the Step 4c block at the end of `tests/test_ga.py`.

## Why (the Step 6b finding)

In the first-pass parameter study (10 s runs, seeds 1000–1004), the plain GA almost never got below its
Clarke–Wright seed:
- 1408 on A-n60-k9 in almost every configuration;
- 16599.05 on the team instance.

The 2-opt memetic GA was stuck as well. Every GA operator works on the giant tour, and Split only cuts that tour.
Nothing moves a customer from one route to another while keeping the rest of the routes intact, which is exactly what
improves a good CW solution. Tabu Search has these moves; the GA did not.

## Design

- **A new memetic mode, not a new GA.** `GAParams(memetic=True, ls_mode="full")` runs `descent()` on an offspring
  with probability `local_search_prob`.
  - `memetic=False` (plain GA) and `ls_mode="2opt"` (the Step 4 memetic GA) are unchanged and stay the baselines.
  - "full" draws no random numbers, so the other variants consume exactly the same random sequence.
  - A before/after fingerprint (TS on A-n32-k5 and A-n60-k9 for 300 iterations; plain and 2-opt GA on A-n32-k5 for
    30 generations) was byte-identical.
- **Lamarckian.** The improved routes are concatenated into a new giant tour and re-decoded. This is the same code path
  as the 2-opt variant. Split is optimal for that customer order, so the re-decoded cost is never higher than the
  descent result.
- **Hard capacity.** The GA only ever holds capacity-feasible routes, because Split never overloads a route. The
  descent keeps them that way: it requires a feasible input, and then a move keeps all loads ≤ Q exactly when its
  `delta_overload` is 0. The existing O(1) delta functions therefore decide feasibility as well. (Tabu Search, in
  contrast, uses soft capacity with a penalty.)
- **No new routes.** Moves only target routes that contain a neighbour customer, so a route is never opened. Routes
  that become empty are dropped, so the number of routes can only fall. That can turn an individual with too many
  routes into a feasible one.
- **Granular lists shared with Tabu Search.** `neighbours()` and the per-customer candidate generator
  `customer_moves()` moved from `tabu.py` to `local_search.py`.
  - Tabu Search calls the same generator in the same order, so its results are unchanged (the tests and the
    fingerprint confirm this).
  - "Granular move" now has one definition: a move that makes customer c adjacent to one of its κ nearest customers
    (Toth & Vigo 2003).

## Descent (pseudocode)

```
descent(routes, N):                 # N[c] = kappa nearest customers of c
    require every load <= Q
    build loads, prefix loads, position of every customer        O(n + K)
    repeat:                                                       # one pass
        improved = false
        for c in 1..n (or a shuffled order if an rng is given):
            for each candidate move m of customer_moves(c, N[c]):  # relocate, swap, 2-opt*, 2-opt
                (dd, dover) = O(1) delta of m
                if dd < -1e-9 and dover == 0:                      # strictly shorter, loads stay <= Q
                    apply m; refresh the 1-2 touched routes        O(K + L)
                    improved = true
                    break                                          # next customer
    until not improved (local optimum) or max_passes reached
    return non-empty routes
```

**Time.**
- One pass evaluates at most n·κ·5 candidate moves, each in O(1). That is O(n·κ·m), with m the number of move types,
  plus O(K + L) for each applied move.
- Every applied move lowers the distance by more than 1e-9, so the descent terminates. There is no useful polynomial
  bound on the number of passes; it is measured below.

**Empirical** (κ = 10, one descent; random starts are the unlimited Split of 20 random tours):

| instance | start | passes | moves applied | delta evaluations | ms per descent | distance before → after |
|---|---|---|---|---|---|---|
| A-n32-k5 | CW | 2.0 | 2 | 2,218 | 1.4 | 842 → 837 |
| A-n32-k5 | random | 5.6 | 61 | 4,572 | 2.7 | 2026 → 829 |
| A-n45-k6 | CW | 2.0 | 4 | 3,115 | 1.9 | 997 → 982 |
| A-n45-k6 | random | 5.8 | 87 | 6,628 | 4.1 | 2630 → 1013 |
| A-n60-k9 | CW | 2.0 | 1 | 4,315 | 2.5 | 1408 → 1406 |
| A-n60-k9 | random | 6.8 | 132 | 10,886 | 6.8 | 3485 → 1461 |
| A-n80-k10 | CW | 2.0 | 4 | 5,711 | 3.7 | 1840 → 1835 |
| A-n80-k10 | random | 7.8 | 199 | 17,022 | 10.4 | 4694 → 1900 |
| team (seed 14835) | CW | 2.0 | 8 | 7,152 | 4.5 | 16599 → 16564 |
| team (seed 14835) | random | 9.1 | 294 | 24,840 | 15.6 | 49588 → 17324 |

The number of passes grows slowly with n: about 6–9 from a random start, and 2 from CW (one improving pass and one
check pass). One descent costs a few milliseconds. A single descent from CW gains little; the gain comes from repeated
descents inside the GA on recombined offspring.

## Tests

- **Invariants**, on A-n32-k5 and the team instance, starting from CW and from a random-tour Split:
  - the distance never increases;
  - all loads stay ≤ Q;
  - every customer appears exactly once;
  - the number of routes never increases.
- **Local optimum, with an independent oracle.** The oracle enumerates **all** relocate, swap, 2-opt* and 2-opt moves
  by brute force. It keeps the granular ones, using the definition above but without calling `customer_moves`, and
  asserts that none improves by more than 1e-9 with feasible loads. A control test checks that the oracle does find
  moves on a random solution.
- **Checker:** the result passes `checker.check` on A-n32-k5, from CW and from a random K-split.
- **Determinism:** with and without an rng.
- **Other:** the input is not modified, `max_passes` is respected, and an overloaded input or an unknown move raises.
- **GA:**
  - `ls_mode` validation;
  - the plain GA reports 0 local searches;
  - "full" is deterministic;
  - **regression test:** on A-n60-k9, 25 generations (about 4.3 s) are feasible and at least 1% below CW (1408).
  - The existing plain-GA tests are untouched and pass.

## Sanity runs (10 s, seed 0, sequential, `ga_ls` = memetic full with local_search_prob 0.5)

The plain GA and TS (defaults) were rerun on the same machine on the same day. Doc values are in brackets where they
exist: step4_notes for the plain GA, step5_notes "after fix" for TS.

| instance | CW | plain GA | ga_ls | TS | ga_ls gap | ga_ls below CW | routes / K | GA gens (plain / ls) | local searches | LS time |
|---|---|---|---|---|---|---|---|---|---|---|
| A-n32-k5 | 842 | 838 (829) | **784** | **784** (784) | 0.00% | 6.89% | 5/5 | 1847 / 133 | 4,183 | 9.12 s |
| A-n45-k6 | 997 | 1000 (1000) | 948 | **944** (944) | 0.42% | 4.91% | 6/6 | 937 / 91 | 2,599 | 8.46 s |
| A-n60-k9 | 1408 | 1408 | **1358** | 1384 (1384) | 0.30% | 3.55% | 9/9 | 566 / 61 | 1,575 | 8.77 s |
| A-n80-k10 | 1840 | 1840 | **1785** | 1827 (1827) | 1.25% | 2.99% | 10/10 | 332 / 42 | 1,067 | 8.64 s |
| team (seed 14835) | 16599.05 | 16599.05 | **16288.32** | 16599.05 | – | 1.87% | 15/16 | 215 / 29 | 691 | 8.75 s |

All results are feasible and pass the checker.
- The plain GA on A-n32-k5 gives 838 here against 829 in step4_notes. The run is time-limited, and this run managed
  fewer generations (1847 vs 2081).
- About 85–90% of the `ga_ls` runtime goes into the local search, so `ga_ls` completes 7–10× fewer generations than
  the plain GA.

**Observations (single runs, not yet statistics):**
- `ga_ls` reaches the BKS on A-n32-k5 and is the best of the three algorithms on A-n60-k9, A-n80-k10 and the team
  instance.
- The stop rule of the prompt does not apply: `ga_ls` is 3.55% below CW on A-n60-k9.
- **Tabu Search (defaults) does not improve on CW on the new team instance (seed 14835) in 10 s.** On the old seed-0
  instance it gained 0.77%. The 6b study already tuned TS (candidate_size 20, λ factor 3 every 3 iterations, tenure
  10–25); whether the tuned TS escapes is a question for the final experiments.
- `local_search_prob` (0.5) and `ls_neighbours` (10) are untuned, and `ga_ls` is not in the tuning grid yet.
