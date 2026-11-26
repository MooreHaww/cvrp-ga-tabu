# CLAUDE.md – CMP 3005 CVRP project (GA + Tabu Search)

Course brief: `docs/project_brief.md`. Every line of code must be defendable in an oral exam:
keep it simple, readable and commented.

## PROJECT RULES
- No external VRP/TSP/optimization solvers (OR-Tools, VROOM, HGS-CVRP, PyVRP, jMetal, PuLP,
  scipy.optimize, networkx algorithms). Everything from scratch. Allowed: numpy, matplotlib, pytest,
  scipy.stats (Wilcoxon only), Python standard library.
- Small functions, type hints, and docstrings that state the time complexity.
- Every file starts with a header comment: "Original code" or "Adapted from <source>" (course rule).
  Cite published pseudocode (Clarke & Wright 1964, Prins 2004, Glover & Laguna 1997) where it is used.
- Vertex 0 = depot, customers = 1..n everywhere in the code. Convert numbering only at file boundaries.
- Augerat instances: TSPLIB EUC_2D distance = int(sqrt(dx^2+dy^2) + 0.5), needed to match the published BKS.
- Team instance: UNROUNDED float Euclidean + P (=150) if the endpoints are on different sides.
- Side label: side(v) = "E" if x < 500 else "A". The depot (500,500) is side "A". Every vertex has exactly
  one label. Never use a "segment crosses x=500" rule (it breaks the triangle inequality).
- Every algorithm takes an Instance and returns a Solution. Every reported solution must pass checker.check().
- Every stochastic function takes an explicit seed or random.Random object. No global random state.
- Every algorithm exposes a progress log of (elapsed_seconds, best_cost) and an iteration counter.
- Experiments write raw per-run results as CSV to results/raw/ (course requirement).
- Roadmap: Step 0 setup -> Step 1 data layer -> Step 2 Clarke–Wright -> Step 3 Split DP (with a max-K-routes
  version) -> Step 4 GA (+ optional memetic 2-opt) -> Step 5 Tabu Search -> Step 6 experiments
  (+ Wilcoxon bonus) -> Step 7 report/README. Only work on the step you are asked for.
- Course milestones: M1 = theory + data + a plot of the team instance. M2 = live demo of Clarke–Wright plus
  one metaheuristic on A-n32-k5.

## WORKING RULES
- Testing: after a small change, run only the tests related to it. Run the full suite only at the end of
  a step. Do not run long experiments unless asked.
- Git: commit only when asked. Never push. Never add "Co-Authored-By" or any Claude/AI attribution
  to commit messages.
- Keep a running log of example prompts in docs/ai_usage.md (we need 2–3 for the report's Appendix B).

## Commands (PowerShell, from the repo root)
```powershell
.\.venv\Scripts\Activate.ps1          # activate the virtual environment (Python 3.12)
python -m pytest tests/test_x.py      # run one test file
python -m pytest                      # full suite (end of a step only)
```
