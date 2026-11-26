# Step 5 – Tabu Search: notes

Sources: F. Glover & M. Laguna, *Tabu Search*, Kluwer, 1997 (tabu list, tenure, aspiration). P. Toth & D. Vigo,
"The granular tabu search and its application to the vehicle-routing problem", *INFORMS J. Computing* 15(4), 2003
(granular candidate lists). Code: `src/cvrp/tabu.py`. Move deltas: `src/cvrp/local_search.py`, part 2.
Tests: `tests/test_tabu.py`.

Notation: n customers, K routes, κ = `candidate_size`, m = number of move types, L = max route length, λ = penalty.

## (a) Pseudocode

```
TABU-SEARCH(inst, params):
    routes ← Clarke–Wright routes
    while #routes > K: concatenate the two routes with the smallest loads    # overload allowed
    pad with empty routes up to exactly K slots
    λ ← clamp(CW distance / Σq)                                   # cost per unit of demand
    best ← current routes if overload = 0, else none
    N[c] ← the κ nearest customers of c, for every c               # granular list, computed once
    it ← 0
    while it < max_iterations and elapsed < time_limit:
        it ← it + 1
        (move, kind) ← SELECT-MOVE()
        removed_arcs ← APPLY(move)                                 # O(K + L), updates dist and overload by the deltas
        tenure ← random integer in [tenure_min, tenure_max]
        for each arc in removed_arcs: tabu_until[arc] ← it + tenure
        record whether overload = 0; every adapt_every iterations:
            all feasible   → λ ← λ / lam_factor
            all infeasible → λ ← λ · lam_factor
            clamp λ to [lam_min, lam_max]
        if overload = 0 and distance < best distance: best ← current; log (elapsed, distance)
    return best (None if no feasible solution was ever seen)

SELECT-MOVE():
    best_adm ← none; best_tabu ← none
    for each customer c, and each nb in N[c]:                      # GRANULAR NEIGHBOURHOOD
        candidates: relocate c just before / just after nb
                    swap c and nb
                    2-opt* that creates arc (c, nb)            (nb in another route)
                    intra 2-opt that creates arc (c, nb)       (nb in the same route)
        plus, if an empty route exists: relocate c into it, and 2-opt* splitting c's route after c
    for each candidate:
        (Δd, Δo) ← O(1) delta;  pen ← Δd + λ·Δo
        if best_adm exists and pen > best_adm.pen: skip            # cannot win
        tabu ← some arc the move would ADD has tabu_until[arc] ≥ it
        admissible ← not tabu
                     or (overload + Δo = 0 and distance + Δd < best feasible distance)   # ASPIRATION
        if admissible: keep the smallest pen (ties: uniform random, reservoir sampling)
        else:          remember the smallest-pen tabu move
    if no admissible move: return the best tabu move (tabu_moves_forced += 1)
    return best_adm                                               # applied even if pen > 0 (it may worsen f)
```

Tabu management in short:
- **Insert**: arcs removed by the applied move become tabu for a random tenure.
- **Check**: a candidate is tabu if any arc it would add is tabu. The lookup is a dict, O(1) per arc.
- **Aspiration**: a tabu move is allowed if it gives a feasible solution better than the best feasible one so far.

## (b) Delta formulas and a worked example

Notation: route r is the path 0, r₁, …, r_m, 0. A position outside the route is the depot.
`node(route, k)` returns route[k], or 0 outside the route, so first/last positions and empty routes need no special cases.

- **Relocate.** Customer c leaves (p, c, s) and is inserted between a and b:
  `Δd = [d(a,c) + d(c,b) − d(a,b)] − [d(p,c) + d(c,s) − d(p,s)]`.
  The source route loses q_c and the target gains it, so
  `Δo = ov(L₁−q_c) − ov(L₁) + ov(L₂+q_c) − ov(L₂)`, where ov(x) = max(0, x − Q). Within one route, Δo = 0.
- **Swap.** c₁ (between p₁, s₁) and c₂ (between p₂, s₂) change places:
  `Δd = d(p₁,c₂) + d(c₂,s₁) − d(p₁,c₁) − d(c₁,s₁) + d(p₂,c₁) + d(c₁,s₂) − d(p₂,c₂) − d(c₂,s₂)`.
  When c₁ and c₂ are adjacent in one route (p, c₁, c₂, s → p, c₂, c₁, s), the middle arc keeps its length:
  `Δd = d(p,c₂) + d(c₁,s) − d(p,c₁) − d(c₂,s)`.
- **2-opt\*.** r₁[:i] + r₂[j:] and r₂[:j] + r₁[i:]. With u = last kept node and v = first moved node (or the depot):
  `Δd = d(u₁,v₂) + d(u₂,v₁) − d(u₁,v₁) − d(u₂,v₂)`.
  New loads in O(1) from the prefix loads P: `P₁[i] + (L₂ − P₂[j])` and `P₂[j] + (L₁ − P₁[i])`.
- **Intra-route 2-opt.** Reverse route[a..b]:
  `Δd = d(p, x_b) + d(x_a, s) − d(p, x_a) − d(x_b, s)`, with Δo = 0. This is the same formula as `two_opt_route`.

**Worked example** (the 6-customer line instance from the Step 3 hand test; depot at x = 0, d = |x − x'|, Q = 10):

| customer | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| x | 2 | 5 | −3 | −4 | 6 | 1 |
| q | 4 | 3 | 5 | 2 | 4 | 3 |

The current routes are A = [1,2] (cost 10, load 7), B = [3,4] (cost 8, load 7) and C = [5,6] (cost 12, load 7).
Total distance 30.

- **Relocate 6 to the front of B.**
  - p = 5, s = 0, a = 0, b = 3.
  - Removal: d(5,6) + d(6,0) − d(5,0) = 5 + 1 − 6 = 0.
  - Insertion: d(0,6) + d(6,3) − d(0,3) = 1 + 4 − 3 = 2.
  - **Δd = +2.** The loads become C = 4 and B = 10 ≤ Q, so Δo = 0.
  - Check: C = [5] costs 12 (unchanged) and B = [6,3,4] costs 1 + 4 + 1 + 4 = 10 (was 8), so +2. ✓
- **Swap 2 (in A) and 5 (in C).** p₁ = 1, s₁ = 0, p₂ = 0, s₂ = 6.
  - Δd = d(1,5) + d(5,0) − d(1,2) − d(2,0) + d(0,2) + d(2,6) − d(0,5) − d(5,6)
    = (4 + 6 − 3 − 5) + (5 + 4 − 6 − 5) = 2 − 2 = **0**.
  - Check: A = [1,5] costs 2 + 4 + 6 = 12 (+2) and C = [2,6] costs 5 + 4 + 1 = 10 (−2). ✓
  - Loads: A = 7 + 4 − 3 = 8, C = 7 + 3 − 4 = 6.
- **2-opt\* on A and B with i = 1, j = 1.** The new routes are [1] + [4] = [1,4] and [3] + [2] = [3,2].
  - u₁ = 1, v₁ = 2, u₂ = 3, v₂ = 4.
  - Δd = d(1,4) + d(3,2) − d(1,2) − d(3,4) = 6 + 8 − 3 − 1 = **+10**.
  - Check: [1,4] costs 2 + 6 + 4 = 12 and [3,2] costs 3 + 8 + 5 = 16, so 28 − 18 = +10. ✓
  - Loads: 4 + (7 − 5) = 6 and 5 + (7 − 4) = 8.

The tests check every formula against a full recomputation on 1,000 random moves per move type, on A-n32-k5 and on
the team instance (float distances plus Bosphorus penalty), always within 1e-9. The moves include empty routes,
first and last positions, adjacent swaps and overloaded routes.

## (c) Complexity of one iteration

| | with granular list (κ) | without (κ = n) |
|---|---|---|
| Candidate moves | O(n·κ·m) + O(n) empty-route options | O(n²·m) |
| Evaluation with O(1) deltas | **O(n·κ·m)** | **O(n²)** |
| Evaluation with full re-evaluation, O(n) per move | O(n·κ·m·n) = O(n²·κ) | **O(n³)** |
| Applying the chosen move | O(K + L): copy the outer list, rebuild 2 routes, loads, prefixes, positions | same |
| Tabu check | O(1) per candidate (2–4 arc lookups), only for moves that can still win | same |

With constant κ (10) and m (4), an iteration is **linear in n** instead of O(n²) or O(n³). Measured: about 1,160
candidate moves per iteration on A-n32-k5, which is roughly 31 · 10 · 3.75.

**Measured cost of one relocate** (`python -m cvrp.tabu --benchmark-delta`, 2,000 random moves, `random.Random(0)`,
Python 3.12, AMD64, 8 logical CPUs):

| instance | n | µs per move (O(1) delta) | µs per move (full re-evaluation) | ratio |
|---|---|---|---|---|
| A-n32-k5 | 31 | 0.52 | 3.37 | 6.5× |
| A-n60-k9 | 59 | 0.52 | 5.69 | 11.0× |
| A-n80-k10 | 79 | 0.51 | 7.46 | 14.5× |
| team_n100_seed0 | 100 | 0.55 | 9.61 | 17.4× |

The delta cost is flat in n, as expected for O(1). The full re-evaluation cost grows linearly in n, so the ratio
grows linearly too: about 0.17·n. Full re-evaluation here is already efficient: Python lists, only the two touched
routes copied. A naive version that copies all routes as a numpy solution would be slower still.

## (d) Parameters (`TabuParams`)

| Parameter | Default | Controls |
|---|---|---|
| `seed` | required | the only source of randomness (tenure draws, tie-breaks) |
| `tenure_min`, `tenure_max` | 5, 15 | how long removed arcs stay forbidden; drawn uniformly each move |
| `candidate_size` (κ) | 10 | granular list size; None = all customers (full neighbourhood) |
| `lam_factor` | 1.5 | how strongly λ reacts |
| `adapt_every` | 10 | window length for adapting λ |
| `lam_min`, `lam_max` | 0.01, 1e4 | clamp for λ |
| `moves` | all four | neighbourhoods used: relocate, swap, two_opt_star, two_opt |
| `time_limit` | None | seconds, checked every iteration |
| `max_iterations` | None | at least one stopping rule is required |
| `verify_deltas` | False | tests only: recompute the objective after every move and assert it matches |
| `repair_after` | 10 | after this many infeasible iterations in a row, also evaluate repair moves (customers of overloaded routes to ALL positions of all other routes); None = off. Added by the fix below. |

Candidates for the Step 6 sensitivity study: tenure range, κ, and lam_factor.

## (e) Design decisions

- **Soft capacity with adaptive λ.**
  - Allowing overloaded routes connects the feasible regions: TS can pass through infeasible solutions to reach
    better feasible ones.
  - λ starts at the cost per unit of demand (CW distance / Σq), so it scales with the instance (2.05 on A-n32-k5,
    9.6 on the team instance).
  - λ rises when the search stays infeasible and falls when it stays feasible.
  - Only feasible solutions are ever reported or logged.
- **Arc-based tabu.**
  - Forbidding the re-creation of recently removed arcs is simple, and it works for every move type alike.
  - It is more precise than "customer c is tabu", which would freeze c for all moves.
  - Arcs are unordered pairs, since d is symmetric, and arcs to the depot count too.
- **Exactly K slots.** The K rule holds by construction, because at most K non-empty routes can ever exist. Empty
  slots let relocate open a new route and let 2-opt* split one.
- **Merge to K.** CW can return more than K routes (7 on A-n45-k6). Concatenating the two lightest routes is the
  cheapest way to get down to K slots. The overload this creates is repaired by the search through the penalty. On
  A-n45-k6, TS starts from an infeasible solution and reaches the BKS 944.
- **Granular neighbourhood.** Good solutions almost never use long arcs, so evaluating only moves that connect a
  customer to one of its κ nearest neighbours loses little and makes iterations linear in n (Toth & Vigo 2003).
- **Best move even if worse.** This is what lets TS escape local optima. The tabu list stops it from undoing the move
  right away. The `tabu_moves_forced` counter stayed 0 in all sanity runs: an admissible move always existed.
- **Same-route swaps are allowed**, including the adjacent case with its own formula. Relocate and 2-opt cover most
  intra-route changes, but supporting this case keeps the move set complete.

## Sanity runs at Step 5 (10 s, seed 0, default parameters), BEFORE the fix below

| instance | n | cost | gap % | routes / K | checker | iterations | it/s | final λ | forced | aspirations | CW cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A-n32-k5 | 31 | 788.00 | 0.51 | 5 / 5 | feasible | 12454 | 1245 | 6.931 | 0 | 3 | 842.00 |
| A-n45-k6 | 44 | 944.00 | **0.00** | 6 / 6 | feasible | 8689 | 869 | 43.089 | 0 | 7 | 997.00 (7 routes) |
| A-n60-k9 | 59 | 1391.00 | 2.73 | 9 / 9 | feasible | 6579 | 658 | 3.821 | 0 | 7 | 1408.00 |
| team_n100_seed0 | 100 | 14284.05 | – | 16 / 16 | feasible | 3997 | 400 | 14.427 | 0 | 4 | 14388.43 |

TS is much stronger than the GA (5.7–5.9% after 10 s, Step 4). On the team instance it improves Clarke–Wright by
only 0.7% in 10 s. This is not explained yet. Two hypotheses, to test in Step 6:
- fewer iterations: 400 it/s at n = 100;
- the P = 150 crossing penalty may make single moves across the strait unattractive, so an improvement needs
  several steps.

Longer runs, tenure and λ are in the parameter study plan.

## Fix: the overload trap and null moves (found by the Step 6a smoke run)

**Symptom.** On A-n80-k10, 10 s runs never improved on Clarke–Wright (1840, a 4.37% gap), and λ climbed to its cap
of 10,000. A-n80-k10 uses 94.2% of the total capacity K·Q (942 of 1000).

### Diagnosis (10 s, seed 0)

| | A-n80-k10 | team_n100_seed0 |
|---|---|---|
| iterations | 5,100 | 3,907 |
| infeasible iterations | **5,100 (100%)**, from iteration 1 | 1,937 (49.6%); longest streak 70 |
| overload while stuck | **1 unit for 5,040 iterations** | short streaks, mostly 1 |
| λ | 1.95 → 112.6 at iteration 100 → **10,000 (the cap) at iteration 220** | 9.6 → 32.5 → 14.4, never capped |
| overload-reducing relocates/swaps at a stuck snapshot | 178 exist, **0 in the granular list** | 524 exist, 48 in the list (46 not tabu) |
| applied moves with Δdist = 0 (first 2,000 iterations) | 1,870, of which 1,228 removed **no arc** | – |
| distinct solutions visited (2,000 iterations) | **420** | – |

At the A-n80-k10 snapshot, every route near the overloaded one had 0–3 units of slack. The only route with room (50
units) was far away.

### Root cause: two defects

1. **Null moves.** Some moves leave the solution unchanged. Swapping the two customers of a 2-customer route, or
   reversing a whole route with 2-opt, only reverses the route. Such a move removes no arc, so it never becomes tabu,
   and its delta is 0. Whenever every real move worsened f, the search applied a null move instead of the least-bad
   real move. That cancels the central idea of Tabu Search, and the search cycled through about 420 states.
2. **No repair move in the granular list.** On a tight instance, the only routes with spare capacity are far from
   the overloaded route, so the moves that remove the last unit of overload are never among a customer's 10 nearest
   neighbours. No value of λ can make the search pick a move it never evaluates, which is why λ ran up to its cap.

The team instance has neither trap: it alternates normally between feasible and infeasible.

### Fix: the smallest change, with the protocol, CLI flags and result format unchanged

- **Skip null moves.** A candidate whose added arcs are all already in the solution is skipped (`_State.is_null_move`).
  Each arc is checked in O(1) with the position array (`_State.has_arc`). After the fix, 0 null moves are applied and
  1,956 distinct states are visited in 2,000 iterations.
- **Repair neighbourhood.** New parameter `TabuParams.repair_after`, default 10; None turns it off. After 10
  infeasible iterations in a row, `_State.repair_candidates` adds:
  - relocations of each customer of an overloaded route to every position of every other route;
  - swaps of such a customer with every customer of another route.

  This adds O(L_over · n) evaluations, and only on those iterations.

Both parts are needed. With only the null-move fix, A-n80-k10 stays at 1840; with both it reaches 1827 (1,000-iteration
check). The λ defaults (×1.5 every 10 iterations) were not changed. Faster settings (×2 every 5) did better in probes:
A-n80-k10 1786/1773 and A-n60-k9 1360/1362 for seeds 0/1, against 1827/1810 and 1384/1364 with the defaults. That is
tuning, so it goes to the Step 6b parameter study.

**Tests added.**
- A-n80-k10, 1,000 iterations, seed 0: feasible, below CW, λ below its cap, and every applied move removes at least
  one arc. Without the repair neighbourhood the run stays at 1840 and this test fails.
- A unit test for null-move detection.
- A unit test that repair candidates reach every position of every other route.

### Before / after (10 s, seed 0, default parameters, sequential)

| instance | cost before | gap before | cost after | gap after | below CW after | routes / K | iterations after | λ after |
|---|---|---|---|---|---|---|---|---|
| A-n32-k5 | 788.00 | 0.51% | **784.00** | **0.00%** | 6.89% | 5/5 | 10169 | 10.40 |
| A-n45-k6 | 944.00 | 0.00% | 944.00 | 0.00% | 5.32% | 6/6 | 7049 | 12.77 |
| A-n60-k9 | 1391.00 | 2.73% | 1384.00 | 2.22% | 1.70% | 9/9 | 5763 | 29.02 |
| A-n80-k10 | 1840.00 (stuck at CW, λ at cap) | 4.37% | 1827.00 | 3.63% | 0.71% | 10/10 | 4219 | 6.59 |
| team_n100_seed0 | 14284.05 | – | 14277.35 | – | 0.77% | 16/16 | 3151 | 109.55 |

All results pass the checker. Iterations per second drop by about 10–20%, the cost of the repair moves and the
null-move check. A-n60-k9 and A-n80-k10 are still a few percent above the BKS with default λ, and the team instance
gains little; these are targets for Step 6b.
