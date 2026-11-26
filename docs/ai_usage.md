# AI-tool usage log

Running log for the report's Appendix B (tools used, for which parts, 2–3 example prompts).

## Tools

| Tool | Used for |
|---|---|
| Claude Code (Anthropic) | Project setup; later steps logged below |

## Example prompts

### #1 – Step 0: project setup (Claude Code)
Asked the assistant to set up the repository only (no algorithm code): a Python 3.11+ virtual environment
with numpy, matplotlib, pytest and scipy (Wilcoxon only), the src/cvrp package skeleton with one
placeholder module per roadmap step, a CLAUDE.md with the project rules (no external solvers, distance
conventions, explicit seeds, feasibility checker), a README skeleton, and this log.

### #2 – Step 1: data layer (Claude Code)
Asked for the data layer only: an Instance dataclass with TSPLIB-rounded and Bosphorus distances, a CVRPLIB .vrp/.sol
parser that verifies the .sol numbering by recomputing the BKS cost, the team and tiny instance generators (seeded,
byte-identical output, FFD warning), an independent feasibility checker, and an exact Held–Karp + set-partition solver
for n ≤ 9 cross-checked against brute force. Also asked for pytest tests, a summary table and the team-instance plot.
In review we added two requirements: compute side labels from the coordinates after rounding, and check the NAME line
of each downloaded file.

### #3 – Step 2: Clarke–Wright savings (Claude Code)
Asked for the classical parallel Clarke–Wright algorithm only. Requirements: savings s = d0i + d0j − dij with
s > 0, a deterministic tie-break (s desc, i asc, j asc), and merging only at route endpoints under capacity. K is not
enforced: a warning is printed instead. The algorithm returns a result object with iteration and merge counters and
has a CLI that prints the gap and the checker verdict. The tests include a 5-customer example whose savings order
and merges were worked out by hand in a comment.

### #4 – Step 3: Split DP (Claude Code)
Asked for Prins' Split procedure as a shortest path in a DAG. Requirements: prefix sums for O(1) arcs, a backward
inner loop that stops at the first overload, O(n·L) time, and a layered version limited to at most K routes that
returns None when no split exists. The tests cover a 6-customer arc table worked out by hand, a brute force over all
cut patterns, "min over all permutations = exact optimum" on the tiny instances, and Split(BKS order) = BKS.
Benchmarking showed that most random tours on larger instances have no ≤ K split. That is input for the GA design
in Step 4.

### #5 – Step 4: Genetic Algorithm (Claude Code)
Asked for a GA on giant tours: a 3-stage Split decoder (unlimited, then K-limited, then infeasible with excess),
binary tournament, OX, swap/inversion/insertion mutation, elitism, a diversity rule with immigrants, CW seeding and
an optional Lamarckian 2-opt. The first sanity run of the specified lexicographic (excess, cost) fitness gave an
81.8% gap on A-n45-k6. Claude diagnosed two causes: all CW seeds were rejected as duplicates, and poor feasible
individuals dominated good near-feasible ones. It tested two fixes and asked us to choose. We chose mutated CW seeds
plus a penalty fitness, cost + λ·excess with an instance-scaled default λ = 2·mean d(0,i). The result was 5.9%
plain and 4.0% memetic. Lesson: check whether a sensible-looking rule actually works before building on it.

### #6 – Step 5: Tabu Search (Claude Code)
Asked for a granular Tabu Search. Specification:
- exactly K route slots (Clarke–Wright merged down to K if needed);
- soft capacity with an adaptive penalty λ;
- relocate, swap, 2-opt* and intra-route 2-opt moves with O(1) deltas, restricted to each customer's 10 nearest
  neighbours;
- an arc-based tabu list with random tenure and aspiration;
- only feasible solutions reported.

The tests check every delta formula against a full recomputation on 1,000 random moves per move type. The benchmark
shows the delta at about 0.5 µs per move for every n, against 3.4–9.6 µs for a full re-evaluation. In 10 s, Tabu
Search reached the BKS on A-n45-k6 and a 0.51% gap on A-n32-k5.

### #7 – Step 6a: experiment runner (Claude Code)
Asked for a runner implementing the Task 7 protocol:
- 10 runs per (instance, algorithm) with seeds BASE_SEED + r shared by GA and TS;
- equal wall-clock budgets and interleaved execution;
- `checker.check` on every result, with infeasible runs kept;
- raw `runs.csv` / `progress.csv`, a `hardware.txt` report, resume after interruption, parameter overrides from
  JSON, and a smoke mode.

The one-minute smoke run exposed two problems before the 82-minute official run:
- on the tiny instances, CW is already optimal, so `reached_optimum` would be trivially True;
- on tight instances (A-n60-k9, A-n80-k10), Tabu Search got stuck with one unit of overload. λ hit its cap because no
  move in the 10-nearest-neighbour list could repair the overload.

### #8 – Fix: Tabu Search overload trap (Claude Code)
Asked Claude to diagnose before fixing, with numbers: infeasible iterations, overload, λ trajectory, and why no move
removed the overload. The trace on A-n80-k10 showed:
- 100% infeasible iterations, with overload 1 for 5,040 iterations;
- λ at its cap from iteration 220;
- 178 repair moves existing, none in the granular candidate list.

A second check uncovered a bug nobody had suspected. 1,870 of 2,000 applied moves had Δ = 0, and only 420 distinct
solutions were visited. The cause was "null moves", such as swapping the two customers of a 2-customer route: they
change nothing, can never become tabu, and were being chosen instead of real worsening moves.

The fix skips null moves and adds a repair neighbourhood after 10 infeasible iterations. Results: A-n32-k5 reached
the optimum (784), and A-n80-k10 improved from 1840 (stuck) to 1827. We kept the λ defaults and left λ tuning to
Step 6b. Good example for the report: the bug was found only because the prompt asked for a diagnosis before a fix.

### #9 – Step 6a-2: tiny instances with a suboptimal start (Claude Code)
Asked to fix the data, not the algorithms. The generator now searches tiny seeds for instances where Clarke–Wright is
at least 3% above the exact optimum. The chosen seeds are 12, 22 and 19, with CW 3.96%, 3.46% and 4.56% above. Tests
that depended on the old instances now read the optima from the `.opt.json` files instead of hard-coding them.

### #10 – Step 6b: parameter study (Claude Code)
Asked for a one-factor-at-a-time study runner, not a tuning run. Requirements:
- separate tuning seeds (1000 + r), never the final seeds;
- 3 instances, 10 s per run;
- a shared all-defaults baseline per algorithm;
- a normalised gap (BKS, or the best team cost seen);
- sensitivity plots with SE error bars;
- a conservative selection rule: change a default only if it is beaten by more than the larger SE;
- a validation run of tuned vs default to catch bad combinations, since OFAT ignores interactions.

Claude pointed out two design gaps before coding:
- a per-instance λ cannot be written into a per-algorithm params file, so the GA penalty is stored as a multiplier
  (`penalty_scale`);
- validating on the selection seeds would be biased, so validation uses the held-out tuning seeds 1005–1009.
