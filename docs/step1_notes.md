# Step 1 – Data layer: notes

n = number of customers, K = number of routes allowed, B = number of bins opened by FFD.

## Files and functions

### `src/cvrp/instance.py`: Instance and distances
| Function | What it does | Time |
|---|---|---|
| `side_label(x)` | `"E"` if x < 500, else `"A"` | O(1) |
| `euclidean_matrix(coords)` | unrounded pairwise Euclidean distances (numpy broadcasting) | O(n²) |
| `euc2d_rounded_matrix(coords)` | TSPLIB EUC_2D: `floor(d + 0.5)` | O(n²) |
| `bosphorus_matrix(coords, sides, P)` | float Euclidean + P if the side labels differ | O(n²) |
| `Instance.from_euc2d_rounded(...)` | builds an Augerat/tiny instance; matrix computed once | O(n²) |
| `Instance.from_bosphorus(...)` | builds a team instance; computes labels from the coordinates | O(n²) |
| `Instance.total_demand()` | Σq | O(n) |

### `src/cvrp/parser.py`: CVRPLIB/TSPLIB files
| Function | What it does | Time |
|---|---|---|
| `parse_vrp(path)` | header + NODE_COORD / DEMAND / DEPOT sections → Instance | O(n²) (matrix), reading O(n) |
| `read_k(name, comment)` | K from the `-kX` name suffix and/or `No of trucks: X`; warns if they disagree | O(len) |
| `parse_sol(path)` | `Route #k:` lines and `Cost X`, numbering unchanged | O(file) |
| `map_sol_routes(inst, routes, cost)` | tries both numberings and keeps the one that reproduces the stated cost | O(n) |
| `load_instance(path)` | `parse_vrp` + BKS from a `<stem>.sol` next to the file | O(n²) |
| `load_bks_solution(path)` | instance + BKS routes in our numbering + stated cost | O(n²) |

### `src/cvrp/generator.py`: team and tiny instances
| Function | What it does | Time |
|---|---|---|
| `generate_team_instance(seed, n=100, P=150, Q=100)` | MediDrone instance (Task 2b) | O(n²) |
| `generate_tiny_instance(n, seed)` | tiny instance where capacity binds (Task 2c) | O(n²) |
| `ffd_bin_count(demands, Q)` | First-Fit-Decreasing bin count | O(n log n + n·B) |
| `warn_if_ffd_exceeds_k(...)` | prints a WARNING if FFD needs more than K bins (does not fix it) | O(n log n + n·B) |
| `vrp_text` / `write_vrp` | deterministic TSPLIB-like text, `\n` newlines, written as bytes | O(n) |

### `src/cvrp/solution.py`: solution representation
A `Solution` holds `routes: list[list[int]]` with customer ids only; the depot is implicit.
`route_cost` O(len(route)), `route_load` O(len(route)), `total_cost` O(n), `__str__` / `describe` O(n).

### `src/cvrp/checker.py`: independent verifier (= NP verifier)
`check(inst, solution, claimed_cost=None) -> CheckReport(feasible, errors, cost, n_routes, loads)`.
It runs one pass over all route entries plus one pass over a seen-array: O(n + K) checks plus an O(n) cost sum.
The checker recomputes the cost with its own loop. It imports nothing from `solution`, `savings`, `split`, `ga` or `tabu`.
It collects all errors instead of stopping at the first one.

### `src/cvrp/exact_small.py`: exact solver for n ≤ 9
1. **Held–Karp** over subsets with load ≤ Q gives the best depot cycle per subset. Time O(2ⁿ·n²).
2. **Set-partition DP**, where `best[k][S]` is the best cost for S using at most k routes. It enumerates only
   subsets T ⊆ S that contain the lowest customer of S, which avoids counting the same partition more than once. Time O(K·3ⁿ).

Total time is O(2ⁿ·n² + K·3ⁿ), and solving all three tiny instances takes well under a second. It reuses no heuristic code.
The brute-force test enumerates all n! permutations × 2ⁿ⁻¹ cut patterns and gets the same optimum on `tiny_n7`.

### `src/cvrp/summary.py` and `src/cvrp/plot_instance.py`: Milestone 1
`summary_rows` takes O(Σ n²) because it parses every file. `plot_instance` takes O(n).

## Design decisions

- **Numbering.** Files number nodes 1..n+1 with the depot as node 1. In the code, vertex = file node − 1, so the
  depot is 0 and customers are 1..n. `parse_vrp` refuses a file whose depot is not node 1.
- **.sol numbering, verified rather than assumed.** CVRPLIB .sol files list customers only. We checked both possible
  mappings by recomputing the cost. For all four Augerat files the ids are already 1..n (offset 0), and the recomputed
  costs equal the stated 784 / 944 / 1354 / 1763. The BKS is read from the .sol and never hardcoded. The expected
  values appear only in a test and in the summary as a sanity check.
- **Rounding.** Augerat and tiny instances use `int(sqrt(dx²+dy²) + 0.5)`, the TSPLIB rule. Without it the published
  BKS cannot be reproduced. The team instance uses unrounded floats.
- **Side labels.** `side(v) = "E" if x < 500 else "A"`. The depot at (500, 500) is "A". The penalty depends only on
  the two endpoint labels and never on whether a segment crosses x = 500, which keeps the distance a metric. The
  tests check the triangle inequality on all 101³ triples.
- **2-decimal coordinates, rounded before labelling.** Team coordinates are rounded to 2 decimals first, and the
  labels are computed from the rounded values. A point at x = 499.996 is therefore written as 500.00 and labelled
  "A", which is the same label the parser computes. The round-trip test checks that coordinates, labels and the
  distance matrix all survive write → parse unchanged.
- **K in generated files.** It is written as `No of trucks: K` in COMMENT, the same mechanism CVRPLIB uses, so no
  new header field was needed. Extra header lines are `EDGE_WEIGHT_TYPE : EUC_2D_BOSPHORUS`, `PENALTY : 150` and
  `SPLIT_X : 500`.
- **Randomness.** `random.Random(seed)` from the standard library, with no global state. Its sequence is fixed
  across Python releases, unlike numpy's legacy API, so the same seed gives a byte-identical file (tested).
- **Team-instance layout.** 4–6 clusters. Centre 0 is forced onto the European side and centre 1 onto the Asian
  side; the others are uniform in [100, 900]². σ ∈ [40, 80] per cluster. 20 uniform customers, and the other 80 are
  split round-robin over the clusters.
- **Tiny instances.** Integer coordinates in [0, 100]², demands in [1, 10], Q = max(max q, ⌈Σq/2.5⌉) so that
  Σq > Q, which forces at least 2 routes (all three optima use 3). K = ⌈Σq/Q⌉ + 1. Seeds are fixed (seed = n).
- **FFD check.** If First-Fit-Decreasing needs more than K bins, the generator prints a WARNING and does not fix it.
  For seed 0: FFD = 15 bins ≤ K = 16.

## Current team instance (placeholder seed 0)
`data/team/team_n100_seed0.vrp`: Σq = 1496, K = 16, 57 vertices on side E and 44 on side A (including the depot).
Plot: `results/figures/team_instance.png`. **TODO:** regenerate with the real seed (sum of the last four digits of
all student IDs).
