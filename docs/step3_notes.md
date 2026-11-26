# Step 3 – Split DP: notes

Source: C. Prins, "A simple and effective evolutionary algorithm for the vehicle routing problem",
*Computers & Operations Research* 31(12), 2004. The implementation is in `src/cvrp/split.py`.

**Notation.** The giant tour is T = (t₁, …, tₙ), a permutation of the customers 1..n. q is demand, Q is capacity,
d is distance, and vertex 0 is the depot.

## (a) The DAG

- **Nodes** are 0, 1, …, n. Node k means "the first k customers of T are served".
- **Arcs.** There is an arc (i → j) for i < j if and only if load(i,j) = q(t_{i+1}) + … + q(t_j) ≤ Q.
  The arc stands for one route that serves the consecutive block t_{i+1}, …, t_j.
- **Arc cost** is that route's length:

      c(i,j) = d(0, t_{i+1}) + Σ_{k=i+1}^{j−1} d(t_k, t_{k+1}) + d(t_j, 0)

- **Acyclic.** Every arc goes from a smaller to a larger node, so there are no cycles and the nodes are already in
  topological order.
- **O(1) arcs.** With prefix sums `load_prefix[k] = Σ_{m≤k} q(t_m)` and `dist_prefix[k] = Σ_{m<k} d(t_m, t_{m+1})`,
  each arc's load and cost take O(1) time:

      load(i,j) = load_prefix[j] − load_prefix[i]
      c(i,j)    = d(0,t_{i+1}) + (dist_prefix[j] − dist_prefix[i+1]) + d(t_j,0)

## (b) Recurrences

**Any number of routes** (shortest path in a DAG, Bellman):

    f[0] = 0
    f[j] = min_{0 ≤ i < j, load(i,j) ≤ Q}  f[i] + c(i,j)        j = 1..n
    pred[j] = the i that attains the minimum

**At most K routes** (layered DP, where layer k uses at most k arcs):

    f[k][0] = 0                for k = 0..K
    f[0][j] = ∞                for j ≥ 1
    f[k][j] = min_{i < j, load(i,j) ≤ Q}  f[k−1][i] + c(i,j)

Because f[k−1][0] = 0, a path in layer k may use fewer than k arcs. f[k][j] is therefore the best cost of serving
t₁..t_j with **at most** k routes, and the answer is f[K][n]. If f[K][n] = ∞ there is no split with ≤ K routes, and
`split()` returns None.

## (c) Correctness

1. **Paths match partitions.** A path 0 = v₀ → v₁ → … → v_r = n picks cut points v₁ < … < v_{r−1}. These cut the
   fixed order T into r consecutive blocks (t_{v_{s−1}+1}, …, t_{v_s}). Every arc on the path exists, so every block
   has load ≤ Q. Conversely, every partition of T into consecutive blocks with load ≤ Q gives exactly one such path.
   The map is a bijection.
2. **Costs agree.** The path cost is the sum of its arc costs, and each arc cost is the length of the route for its
   block. So the path cost equals the cost of the corresponding solution.
3. **The DP finds a shortest path.** The DAG is processed in topological order (j increasing). When f[j] is computed,
   every f[i] with i < j is already final, so the recurrence is the standard shortest-path DP in a DAG. By induction
   on j, f[j] is the cost of the best partition of t₁..t_j.
4. **The early stop is safe.** For a fixed j, load(i,j) only grows as i decreases. Once load(i,j) > Q, every smaller
   i is infeasible too, so stopping there skips no arc.
5. **The K-limited version.** By the same argument on the layered DAG (k, j), with the extra invariant "at most k
   arcs", f[K][n] is the best partition of T into at most K feasible blocks.

Together, these show that Split returns an optimal solution among all solutions that visit the customers **in the
order T**.

## (d) Complexity

Let L be the maximum number of customers that can share one route.

- **Prefix sums:** O(n). Validating the tour as a permutation: O(n).
- **Unlimited DP.**
  - For each j, the backward loop evaluates at most L feasible arcs plus one infeasible arc, then breaks.
  - Each evaluation is O(1), so the DP takes **O(n·L)** time.
  - Since L ≤ n, the worst case is **O(n²)**: one huge Q holding everything gives L = n and n(n+1)/2 arcs.
  - Space is **O(n)** for f, pred and the prefix arrays.
- **Bound on L.** A route with L customers has load ≥ L · q_min, so L ≤ ⌊Q / q_min⌋.
  - On the team instance, q_min = 1, so in theory L ≤ 100, which is no better than n.
  - In practice, with demands averaging 15, routes hold far fewer customers. We measured at most 12–14 customers
    per route on random tours (table below).
  - So the DP behaves like **O(n·L) ≈ O(13·n)**: linear in n with a small constant.
- **K-limited DP.** Layers 1..K each repeat the same loop: **O(K·n·L)** time, O(K·n²) in the worst case.
  Space is **O(K·n)** for the f and pred tables.
- **Route rebuild:** O(n). The final cost is recomputed in O(n), so it is exactly what the checker computes.

## (e) Pseudocode

```
SPLIT(T, max_routes = None):
    compute load_prefix, dist_prefix                         # O(n)
    if max_routes is None:
        f[0] ← 0;  f[1..n] ← ∞
        for j ← 1 .. n:
            for i ← j−1 down to 0:
                if load_prefix[j] − load_prefix[i] > Q: break
                c ← d(0,t_{i+1}) + dist_prefix[j] − dist_prefix[i+1] + d(t_j,0)
                if f[i] + c < f[j]:  f[j] ← f[i] + c;  pred[j] ← i
        cuts ← follow pred from n back to 0
    else:
        f[k][0] ← 0 for all k;  all other f ← ∞
        for k ← 1 .. K:  (same double loop, reading f[k−1], writing f[k], pred[k])
        if f[K][n] = ∞: return None
        cuts ← follow pred[k] from (K, n), one layer down per route
    return the blocks of T between consecutive cuts
```

## (f) Optimal for the order, not for the problem

Split is exact **for the given order T only**. The best solution of the whole CVRP may visit the customers in a
different order, and then no cutting of T reaches it. Two facts show both sides:

- **Some order is perfect.** Concatenating the routes of an optimal solution gives a tour that Split cuts back into
  that optimal solution. The tests check this in three ways:
  - min over all permutations of Split = the exact optimum on tiny_n7/8/9;
  - Split(BKS order) = 784 on A-n32-k5;
  - Split(BKS order) = 944 on A-n45-k6.
- **Most orders are poor.** A random order is a bad tour, and even the best cut of it is bad.

**Consequence for the GA.** The search space is the n! orders, and Split acts as an exact decoder. That gives three
properties:
- every chromosome maps to its best feasible solution;
- crossover and mutation can work on plain permutations, with no repair needed for capacity;
- searching over permutations loses nothing, because the optimum is reachable.

The GA's job is to find a good order.

**Warning for Step 4: the K limit.** On larger instances, most random orders cannot be cut into at most K routes
(table below):
- 48% of random tours have a ≤ K split on A-n60-k9;
- 26% on A-n80-k10;
- only 6% on the team instance.

A GA that decodes with `max_routes=K` alone would mark most of the initial population infeasible. Step 4 needs a
plan for this, for example:
- decode with the unlimited Split and penalise routes beyond K; or
- fall back to the unlimited Split when the K-limited one returns None.

The same holds for the Clarke–Wright order of A-n45-k6: it has no split into ≤ 6 routes (tested). Split cannot
repair a bad order.

## Measured time per split() call

Command: `python -m cvrp.split --benchmark`. 200 random tours per instance, `random.Random(0)`, Python 3.12.10,
AMD64 (8 logical CPUs). Single runs, indicative only.

| instance | n | K | µs/split (unlimited) | µs/split (max_routes=K) | arcs (unl.) | arcs (K) | tours with a ≤K split | max route len |
|---|---|---|---|---|---|---|---|---|
| A-n32-k5 | 31 | 5 | 34 | 98 | 200 | 998 | 100% | 13 |
| A-n60-k9 | 59 | 9 | 65 | 296 | 381 | 3432 | 48% | 13 |
| A-n80-k10 | 79 | 10 | 93 | 494 | 608 | 6093 | 26% | 14 |
| team_n100_seed0 | 100 | 16 | 105 | 824 | 618 | 9913 | 6% | 12 |

- **Linear growth.** The unlimited arcs grow about linearly in n (≈ 6.2·n), as O(n·L) with L ≈ 13 predicts. The
  K version does about K times more work.
- **GA budget.** At about 0.1 ms per unlimited split for n = 100, a 60 s budget allows roughly 500 000 decodings
  before any other GA overhead.
