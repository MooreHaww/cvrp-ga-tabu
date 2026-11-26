# CMP 3005 Term Project Brief (original text, converted from Term_Project.docx)

**Term Project: Capacitated Routing + Genetic Algorithm & Tabu Search**

**Title: Drones, Deadlines and NP-Hardness: Metaheuristics for a Capacitated Medical Delivery Network**


## 1. Real-world Story

You have just joined MediDrone Istanbul, a (fictional) start-up that uses a fleet of delivery drones to carry blood samples, vaccines and urgent medication from a central laboratory to clinics on both the European and the Asian side of the city.

Every drone has a limited payload capacity, and every clinic orders a different amount each day. One drone usually cannot serve everyone, so the fleet must be split into several routes. Flying across the Bosphorus is allowed but costs extra (air-traffic corridor, wind, battery).

Your CTO gives you this task:

“Every morning, tell me which drone visits which clinics and in what order. Each clinic must be served exactly once, no drone may carry more than its capacity, every drone returns to the lab, and the total flight distance must be as small as possible. I need good plans in seconds, not a proof of optimality — but I want hard evidence that your method works and scales.”

You quickly recognize this as the Capacitated Vehicle Routing Problem (CVRP) — a generalization of the Traveling Salesperson Problem and one of the most studied NP-hard problems in operations research.

Your mission is to:

- Formalize the problem as an optimization and a decision problem.
- Prove NP-hardness through a chain of polynomial-time reductions.
- Design, implement and analyze a constructive baseline and two metaheuristics:
- a Genetic Algorithm (GA) — a population-based method,
- Tabu Search (TS) — a trajectory-based method.
- Analyze the running time of your algorithms, not only measure it.
- Evaluate them against published best-known solutions and defend your work orally.

## 2. Formal Problem Definition

You are given:

- A complete undirected graph G = (V, E) with V = {0, 1, …, n}, where vertex 0 is the depot (central lab) and vertices 1, …, n are customers (clinics).
- A distance function d : E → ℝ⁺ that satisfies the triangle inequality.
- A demand qᵢ ∈ ℤ⁺ for every customer i, with qᵢ ≤ Q.
- A vehicle capacity Q ∈ ℤ⁺ and a fleet of K identical vehicles.
A feasible solution is a set of at most K routes such that:

- each route starts and ends at the depot,
- each customer is visited by exactly one route, exactly once,
- the total demand on every route is at most Q.
Optimization version (primary focus): find a feasible solution of minimum total distance.

Decision version (for NP-hardness): given (G, d, q, Q, K) and a bound B ∈ ℝ⁺, is there a feasible solution with total distance at most B?


## 3. Learning Objectives

By the end of the project, you should be able to:

- Build a chain of reductions (Hamiltonian Cycle → TSP → CVRP) and explain why NP-hardness “travels” along it.
- Recognize hidden NP-hard subproblems inside a larger problem (here: Bin Packing inside CVRP).
- Implement and tune a Genetic Algorithm and a Tabu Search for a constrained problem.
- Derive the asymptotic cost of one iteration of each algorithm and validate it experimentally.
- Compare heuristics fairly (equal time budgets, multiple seeds, benchmark gaps).
- Reason about approximation: what we can guarantee in theory vs. what works in practice.

## 4. Tasks


### Task 1 – Theoretical Foundations (Reductions & NP-Hardness)

- NP membership. Show that the decision version of CVRP is in NP. Describe the certificate and give the running time of your verifier in terms of n.
- Reduction chain. Show that CVRP (decision) is NP-hard in two steps:
- Step A: Hamiltonian Cycle ≤ₚ TSP (decision). Give the construction, show it is polynomial, and prove both directions (YES ⇔ YES).
- Step B: TSP (decision) ≤ₚ CVRP (decision). Hint: how should you choose demands, Q and K so that the CVRP instance is “really” a TSP instance?
- Explain why the composition of two polynomial-time reductions is again a polynomial-time reduction.
- The hidden second NP-hard problem. Consider only the question: “Can all customers be served with at most K vehicles of capacity Q?” (ignore distances). Show that this feasibility question is itself NP-complete by a reduction from Bin Packing (or Partition). In one paragraph, explain what this means for your metaheuristics: why may even finding a feasible solution be hard in general?
- Your twist must keep the theory valid. In Task 2 you add a Bosphorus-crossing penalty to distances. Prove that the modified distance still satisfies the triangle inequality (or show a counter-example and fix the definition).
A fully formal textbook proof is not required, but every argument must be coherent, complete and logically sound.

Bonus (extra credit): In one paragraph, summarize what is known about approximation algorithms for CVRP (e.g. the iterated tour partitioning idea that turns a TSP tour into capacity-feasible routes) and how the TSP approximation ratio carries over.


### Task 2 – Data: Benchmarks + Your Own Team Instance

This year you will work with two kinds of data.

(a) Public benchmarks with known best solutions. Download the following instances from CVRPLIB (Augerat Set A):

Use the best-known solution (BKS) published on CVRPLIB as the reference value. Write a small parser for the TSPLIB/CVRPLIB file format yourself. Note: CVRPLIB rounds distances to the nearest integer — make sure your evaluation does the same, otherwise your gaps will be wrong.

(b) A unique MediDrone instance for your team. Generate your own instance with the rules below. The random seed is the sum of the last four digits of all team members’ student IDs — write this calculation in your report. Every team therefore gets different data.

- Place the depot at the center of a 1000 × 1000 square.
- Split the square vertically into a European side (x < 500) and an Asian side (x ≥ 500).
- Generate n = 100 customers in 4–6 clusters (spread over both sides), plus about 20% uniformly random customers.
- Demands qᵢ: integers drawn uniformly from [1, 30]; capacity Q = 100; choose K = ⌈Σqᵢ / Q⌉ + 1.
- Distance: d(i, j) = Euclidean distance + P if i and j lie on different sides, with P = 150. (You proved in Task 1 that this is still metric.)
(c) Ground truth for tiny instances. Generate at least three tiny instances (n = 7, 8, 9 customers) and compute their exact optimum by brute force / dynamic programming, or with an ILP solver. You will use these to check that your metaheuristics actually reach the optimum on small cases.


### Task 3 – Constructive Baseline: Clarke–Wright Savings

Implement the classical Clarke–Wright savings algorithm as a fast, deterministic baseline.

- Start with one route per customer: 0 → i → 0.
- Compute the saving s(i, j) = d(0, i) + d(0, j) − d(i, j) for every pair and sort them.
- Merge routes greedily in order of decreasing saving, whenever the merge is capacity-feasible.
- Analysis: give the time complexity of your implementation as a function of n and justify it.
This baseline is also an excellent initial solution for Tabu Search and a seed individual for the GA.


### Task 4 – Genetic Algorithm (GA)

Use the elegant “giant tour + split” representation:

- Chromosome: a permutation of the n customers without route delimiters (a “giant tour”).
- Decoding (Split procedure): given a giant tour, find the optimal way to cut it into capacity-feasible routes. This can be solved exactly as a shortest path in a directed acyclic graph, i.e. with dynamic programming. You must:
- describe the DAG (nodes, arcs, arc costs),
- write the recurrence of the DP,
- prove its time complexity (O(n²) in general; argue why it is closer to O(n·L) when each route contains at most L customers).
- Fitness: total distance of the decoded solution.
- Operators (describe and justify each choice):
- Selection: e.g. binary tournament.
- Crossover: Order Crossover (OX) or another permutation-preserving operator.
- Mutation: e.g. swap, inversion or insertion.
- Replacement & elitism: how the next generation is formed; how you prevent loss of diversity (e.g. reject duplicates).
- Parameters to tune and report: population size, crossover rate, mutation rate, number of generations / time limit.
- Optional (extra credit): a memetic GA that applies a quick local search (e.g. 2-opt inside each route) to offspring.

### Task 5 – Tabu Search (TS)

Implement a Tabu Search that works directly on a route-based solution.

- Initial solution: the Clarke–Wright result from Task 3.
- Neighborhood — implement at least three of the following moves:
- Relocate: move one customer to another position or route.
- Swap: exchange two customers between two routes.
- 2-opt: reverse a segment inside one route.
- 2-opt∗: exchange the tails of two routes.
- Delta evaluation: compute the change in cost of a move in O(1), without re-evaluating the whole solution. Show the formula for at least one move.
- Tabu list: what exactly is stored (moved customer? arc? route pair?), and the tabu tenure (fixed or random in a range).
- Aspiration criterion: a tabu move is allowed if it produces a new overall best solution.
- Capacity handling: either forbid infeasible moves, or allow them with a penalty term λ · (overload) and adapt λ during the search. Explain your choice.
- Parameters to tune and report: tenure, neighborhood size / candidate list, penalty λ, stopping criterion.

### Task 6 – Complexity Analysis of Your Own Code

This is an Analysis of Algorithms course — so analyze, don’t only measure.

- Give the asymptotic cost of one generation of your GA and one iteration of your TS as a function of n (and of population size, neighborhood size, etc.).
- Show how much delta evaluation saves compared with full re-evaluation of a solution.
- Validate empirically: plot the measured time per iteration against n on a log–log scale, estimate the slope, and compare it with your theoretical prediction. Explain any difference.

### Task 7 – Experiments, Comparison & Reflection

Fair experimental protocol (mandatory):

- Run every stochastic algorithm 10 times with different, recorded random seeds per instance.
- Give GA and TS the same wall-clock time budget per instance (e.g. 30 s for n ≤ 50, 60 s for larger n). Report your hardware.
- Report for each instance and algorithm: best, mean and standard deviation of the cost, runtime, and the gap to the best-known solution:
Gap (%) = 100 × (Cost − BKS) / BKS

- On the tiny instances, report whether each algorithm reaches the exact optimum and how often (out of 10 runs).
- Plot convergence curves (best cost vs. time) for GA and TS on at least one medium instance.
- Visualize the best routes on your MediDrone instance (color per route, show the Bosphorus line).
Discussion:

- Which algorithm is better on which instances, and why? Ground every claim in your data.
- How did the Bosphorus penalty change the shape of the routes? How many routes cross the strait?
- Parameter sensitivity: present at least one parameter study (e.g. tabu tenure vs. gap, or mutation rate vs. diversity) with a plot, and interpret it.
- Theory vs. practice (1–2 paragraphs): contrast proven approximation algorithms with metaheuristics. Why does industry accept algorithms without guarantees?
- Bonus: use a non-parametric statistical test (e.g. Wilcoxon signed-rank) to check whether the difference between GA and TS is significant.
Personal reflection (short, human part):

- What surprised your team the most?
- What was the hardest bug you found, and how did you find it?
- With more time, what would you try next (e.g. hybrid GA + TS, adaptive large neighborhood search, time windows)?

## 5. Deliverables

Milestone 1 – Theory & data (2–3 pages, ungraded feedback, but required): problem definitions, reduction sketches, seed calculation and a plot of your MediDrone instance.

Milestone 2 – Code demo (in lab / office hours, ~10 min): Clarke–Wright and at least one metaheuristic running on A-n32-k5.

Final submission:

- Report (PDF, 10–16 pages + appendices) in clear, human-readable English, containing:
- Introduction & problem modeling (your own short version of the story).
- Complexity: NP membership, reduction chain, Bin Packing argument, metric proof.
- Methodology: Clarke–Wright, GA (incl. Split DP), TS — with pseudocode for all three.
- Complexity analysis of your own code (Task 6).
- Experiments: setup, parameters, tables, plots, route visualizations.
- Discussion & reflection.
- Appendix A – Contribution table: who did what (percentages per member).
- Appendix B – AI-tool usage statement (see Section 7).
- Code (GitHub repository link or ZIP):
- Any language (Python, Java, C++, …).
- A README with exact commands to reproduce every table and figure.
- Clear separation of: instance parsing/generation, solution representation & evaluation, Clarke–Wright, GA, TS, experiment scripts.
- A feasibility checker that independently validates every reported solution (capacity, each customer exactly once, cost recomputed).
- Raw results (CSV) of all runs.
- Presentation & oral defense (last lecture of the semester, 10–15 minutes per team): a short slide presentation of your results, followed by questions. Every member must be able to explain any part of the report and code; questions will be asked to individual members. Upload your slides together with the final submission.

## 6. Grading Rubric

Individual grades may be adjusted according to the contribution table and the oral defense.


## 7. Rules, Allowed Resources and AI Tools

You may use:

- Standard libraries for random numbers, data structures, plotting, timing and statistics.
- An ILP solver only to compute ground truth for the tiny instances in Task 2(c).
- Published pseudocode for GA, TS, Split and Clarke–Wright as a starting point — but you must write your own implementation.
You may not use: existing VRP/metaheuristic solver libraries (e.g. OR-Tools routing, VROOM, HGS-CVRP, PyVRP, jMetal) to produce your reported solutions.

AI tools (ChatGPT, Claude, Copilot, …): allowed 100% and you do not have to state that you used it

Appendix B must state which tools you used, for which parts, and give 2–3 example prompts.

- You are fully responsible for the correctness of everything you submit; AI-generated proofs often contain subtle errors.
- Code or text that a team member cannot explain during the oral defense will not receive credit.
You must:

- Cite all external sources (books, papers, websites, repositories).
- Clearly mark which parts of the code are your own and which are adapted.
- Use your own team-specific instance; results reported on another team’s instance will not be accepted.

## 8. Suggested Reading

- Toth, P. & Vigo, D. (eds.), Vehicle Routing: Problems, Methods, and Applications, 2nd ed., SIAM, 2014.
- Clarke, G. & Wright, J. W., “Scheduling of vehicles from a central depot to a number of delivery points,” Operations Research, 1964.
- Prins, C., “A simple and effective evolutionary algorithm for the vehicle routing problem,” Computers & Operations Research, 2004 (giant tour + Split).
- Glover, F. & Laguna, M., Tabu Search, Kluwer, 1997.
- Garey, M. R. & Johnson, D. S., Computers and Intractability, 1979 (reductions, Bin Packing).
- CVRPLIB benchmark library (instances and best-known solutions).

## Instances (Task 2a)

| Instance | Size | Purpose |
|---|---|---|
| A-n32-k5 | Small | Correctness & tuning |
| A-n45-k6 | Medium | Main comparison |
| A-n60-k9 | Medium–large | Scalability |
| A-n80-k10 | Large | Scalability & stress test |

## Grading weights

| Component | Weight |
|---|---|
| Problem definition & theory (NP membership, reduction chain, Bin Packing, metric proof) | 20% |
| Algorithm design & implementation (Clarke–Wright, GA with Split, TS; code quality, feasibility checker) | 25% |
| Complexity analysis of own algorithms & empirical validation (Task 6) | 10% |
| Experimental rigor (benchmarks, team instance, seeds, equal budgets, gaps, plots) | 20% |
| Depth of analysis & reflection (data-grounded comparison, sensitivity, theory vs. practice) | 10% |
| Oral defense (individual understanding) | 10% |
| Writing quality & presentation | 5% |
| Bonus items (approximation paragraph, memetic GA, statistical test) | up to +10% |
