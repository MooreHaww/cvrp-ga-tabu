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
