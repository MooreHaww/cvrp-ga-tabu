# Step 2 – Clarke–Wright savings: notes

Source: G. Clarke & J. W. Wright, "Scheduling of vehicles from a central depot to a number of delivery points",
*Operations Research* 12(4), 1964. The implementation is in `src/cvrp/savings.py` and the tests are in `tests/test_savings.py`.

## (a) Pseudocode (parallel version)

```
CLARKE-WRIGHT(G, d, q, Q):
    for each customer v:                          # one route 0-v-0 per customer
        route_of[v] ← v ;  routes[v] ← [v] ;  load[v] ← q_v
    S ← empty list
    for i ← 1 .. n:  for j ← i+1 .. n:
        s ← d(0,i) + d(0,j) − d(i,j)
        if s > 0:  append (s, i, j) to S
    sort S by (s descending, i ascending, j ascending)
    for (s, i, j) in S:
        ri ← route_of[i] ;  rj ← route_of[j]
        if ri = rj:                          continue   # already together
        if i or j is not an endpoint:        continue   # would break an existing route
        if load[ri] + load[rj] > Q:          continue   # capacity
        orient routes[ri] so it ends with i, routes[rj] so it starts with j
        new ← routes[ri] ++ routes[rj]
        keep the id of the longer route, relabel route_of[v] for v in the shorter one
        routes[keep] ← new ;  load[keep] ← load[ri] + load[rj] ;  delete the other route
    if #routes > K:  print WARNING            # K is not enforced here
    return the routes
```

## (b) Complexity

| Part | Cost | Why |
|---|---|---|
| Initial routes | O(n) | one route per customer |
| Savings | O(n²) | n(n−1)/2 pairs, O(1) each, using the precomputed distance matrix |
| Sort | O(n² log n) | sorting m = O(n²) items costs O(m log m) = O(n² log n²) = O(n² log n) |
| Merge loop, checks | O(n²) | each pair needs O(1) checks: route ids, endpoint test via `route[0]`/`route[-1]`, loads |
| Merge loop, merges | O(n²) | every merge removes one route, so there are at most n − 1 merges; each concatenates and relabels in O(n) |
| Final cost | O(n) | |

**Total time: O(n² log n)**, dominated by the sort. **Space: O(n²)** for the savings list. The routes and arrays
need only O(n).

Relabelling only the shorter route ("small into large") makes the relabelling itself O(n log n) in total. The list
concatenation is still O(n) per merge, so the merge phase stays O(n²) in the worst case. Either way it is below the
sort.

**Why we can keep the whole O(n²) pair list.** For n ≤ 100 there are at most 4 950 pairs. That is a few hundred KB
of Python tuples, sorted in about a millisecond. The measured runtime is 0.3 ms for n = 31 and 3.1 ms for n = 100.
Pruning ideas (heaps, neighbour lists) would only matter for n in the tens of thousands.

## (c) Design decisions

- **Parallel rather than sequential.** All routes can grow at the same time, following the global savings order.
  This is the classical version from the course brief, and it usually produces better solutions than growing one
  route at a time.
- **Endpoint rule.** A merge joins the route through the arc (i, j). This only keeps every route a simple path
  0 → … → 0 if both i and j are at an end of their routes. An interior customer already has two route neighbours.
- **Orientation.** Because d is symmetric, a route can be reversed without changing its cost. `_merge` therefore
  reverses one or both routes so that i ends the first route and j starts the second.
- **Deterministic tie-break.** Pairs are sorted by (saving desc, i asc, j asc). Equal savings occur often with
  rounded EUC_2D distances, so this makes the result reproducible and the hand example in the tests checkable.
- **Only s > 0 kept.** A non-positive saving means merging would not shorten the total distance. Under the triangle
  inequality s ≥ 0 always holds; with rounded distances s can be 0 or slightly negative.
- **K is not forced.** Classical CW minimises distance and does not control the number of routes. On A-n45-k6 it
  ends with 7 routes while K = 6. We print a WARNING and return the solution unchanged, and the checker reports
  "too many routes". The route count will be handled in Step 3 (Split with at most K routes) and Step 5 (Tabu Search).
- **Instrumentation.** `CWResult` holds the iteration count (pairs examined), the number of merges, and one progress
  entry (elapsed seconds, cost), since CW is constructive. As a consistency check, merges = n − n_routes.

## Results (current data, team seed 0)

| instance | n | K | CW cost | CW routes | BKS | gap % | runtime ms | checker |
|---|---|---|---|---|---|---|---|---|
| A-n32-k5 | 31 | 5 | 842 | 5 | 784 | 7.40 | 0.3 | feasible |
| A-n45-k6 | 44 | 6 | 997 | 7 | 944 | 5.61 | 0.5 | too many routes: 7 > K = 6 |
| A-n60-k9 | 59 | 9 | 1408 | 9 | 1354 | 3.99 | 0.9 | feasible |
| A-n80-k10 | 79 | 10 | 1840 | 10 | 1763 | 4.37 | 1.9 | feasible |
| team_n100_seed0 | 100 | 16 | 14388.43 | 16 | - | - | 3.1 | feasible |

Runtimes are single runs on the development laptop and are only indicative.
