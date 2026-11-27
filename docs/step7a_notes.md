# Step 7a – Reproducibility check and report numbers sheet: notes

Code: `experiments/report_numbers.py`. Tests: `tests/test_report_numbers.py`. Output: `docs/report_numbers.md`.

No new experiment was run. `results/raw`, `results/timing` and `results/tuning*` were only read.

## 1. Reproducibility check (fresh clone)

**Setup.**
- The repository at commit `06e85f0` was cloned with `git clone` into a temporary folder outside the repository.
- The clone used the Python of the existing `.venv` (3.12.10), not a new virtual environment.
  - Because `cvrp` is installed in editable mode from the original checkout, `PYTHONPATH` was set to `<clone>/src`.
  - It was checked that `cvrp` and `experiments` were imported from the clone, not from the original folder.
- The temporary folder was deleted afterwards.

| Check | Command (in the clone) | Result |
|---|---|---|
| Full test suite | `python -m pytest` | **pass**: 168 passed, 1 deselected (slow), 54 s |
| Analysis tables | `python -m experiments.analysis --raw results/raw --out <temp>` | **pass**: all 10 files (`main_results`, `cw_baseline`, `tiny_results`, `wilcoxon`, `team_crossings`, each `.csv` and `.md`) byte-identical to `results/tables` |
| Analysis figures | the same run | **pass**: all 7 figures exist, size ratio 1.000; here even byte-identical |
| Complexity fits (extra) | `build_fits` + `write_fits` on the committed `results/timing/timing.csv` | **pass**: `complexity_fits.csv` and `.md` byte-identical |
| Timing smoke | `python -m experiments.timing --smoke` | **pass**: runs from a fresh checkout |
| Runner smoke | `python -m experiments.run_experiments --smoke` | **pass**: 32 runs, 0 infeasible |
| Clean tree | `git status` in the clone after all runs | **pass**: no tracked file changed; smoke output is git-ignored |

No file differed.

**Caveats.**
- **Same environment:** the same interpreter and the same package versions were used. A machine with other numpy,
  scipy or matplotlib versions may give different PNG bytes, because PNG output depends on the matplotlib version and
  its fonts. That is why figures are compared only by existence and size ratio. The tables should still match, since
  they are rounded to 6 decimals.
- **Not re-run:** the main run, the tuning runs and the timing run. Their results depend on wall-clock budgets and
  CPU speed, so they can only be reproduced statistically, not byte for byte. The check reproduces everything that is
  computed *from* the raw results.
- **What the analysis reads:** it reads the team reference from `results/tuning` and `results/tuning2`, and the
  solution files from `results/raw/solutions`. Both are committed, so the clone has them.

## 2. Report numbers sheet

```powershell
python -m experiments.report_numbers
```

It writes `docs/report_numbers.md`. Every value in it is read from a file:

| Section | Source |
|---|---|
| Setup: commits | `git_commit` columns of `results/raw/runs.csv` and `results/timing/timing.csv`; HEAD from `git rev-parse` |
| Setup: hardware, Python, numpy, workers | `results/raw/hardware.txt`, `results/timing/hardware.txt` |
| Setup: package versions | the environment that generates the sheet (`importlib.metadata`) |
| Setup: seeds, run counts, budgets | `runs.csv` (seed, algo, budget_s) and `experiments/config.py` (BASE_SEED, BUDGETS) |
| Instances | the `.vrp` files (n, Q, K, demand, P, split_x, side labels); reference and type from `main_results.csv` |
| Parameters | dataclass defaults (GAParams, TabuParams, GA_LS_BASE), `tuned_params.json`, and `params_json` of the first main run |
| Tables | the CSV files, copied cell by cell |
| Key findings | sentences computed from the tables, with the numbers inserted by code |

**Design points.**
- **Deterministic:** fixed order, fixed formats and LF newlines.
- **The HEAD line:** it is the HEAD *when the sheet was generated*. It is marked "+ uncommitted changes" if the tree
  was dirty. A sheet committed together with the code therefore names the commit before it.
- **Parameter check:** the "used = tuned or default" column compares the parameters actually recorded in `runs.csv`
  with `tuned_params.json`. Every row says "yes", so the main run really used the tuned values.
- **Significance:** Wilcoxon results count as significant if p_holm < 0.05. The table itself has no α; the 0.05 is
  stated in the sentence.
- **Two-point slopes:** the Tabu slope split and the GA two-point slope are computed between the smallest n ≥ 100 and
  the largest n. The fitted slopes come from `complexity_fits.csv`.

**Tests** (synthetic folder: 2 instances × 2 algorithms × 2 runs, with timings that follow known power laws):
- two runs are byte-identical, with no `\r`;
- every 40-character hash in the sheet is one of the three input hashes;
- seeds, hardware, budgets and versions appear as given;
- the findings follow the inputs. Examples: the GA slope is 2.00 for t ∝ n², and the Tabu split is 1.00 + 0.50 for
  moves ∝ n with time per move ∝ √n. Changing one mean in `main_results.csv` changes the sentence;
- parameters and team-instance facts match the files, and a missing table is reported, not invented.

## 3. README

The README was restructured, keeping the existing correct content:
- purpose, team, rules (no external solvers; allowed libraries), install, quick start and folder layout;
- a table "report item → command → output file";
- expected runtimes, seeds, reproducing everything (in pipeline order) and tests.

No student IDs appear anywhere: the seed section states the rule and the resulting value 14835 only.
