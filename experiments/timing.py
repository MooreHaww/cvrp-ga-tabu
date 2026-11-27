# Original code
"""Step 6d: complexity validation - how does the running time grow with n?

    python -m experiments.timing                       # sizes 50..800, 7 reps (quiet CPU!)
    python -m experiments.timing --smoke               # sizes 20,40, 2 reps, into results/smoke_timing

Instances: generate_team_instance(seed=n, n=n). Q = 100 for every n and demands are
U[1, 30], so L (customers per route) stays ~6-7 while K = ceil(sum q / Q) + 1 grows
linearly with n. A second experiment (Q-sweep) fixes n and varies Q, i.e. L.

Per (component, n): one untimed warm-up, then --reps timed repetitions (time.perf_counter,
garbage collector off during the timed call, rep r uses seed r). Fits: log10(t) =
a + b*log10(n) on the median per n, for all n and for n >= 100, compared with the slope
the theoretical bound has over the SAME n values (so a log factor is accounted for).

Outputs: <out>/timing.csv (one row per repetition), <out>/hardware.txt, <out>/share_ga_ls.csv,
<tables>/complexity_fits.{csv,md}, <figures>/complexity_*.png, split_vs_L.png, delta_vs_full.png.
No algorithm code is changed: GAParams.max_generations, GAResult.gen_times and
TabuParams.max_iterations already allow fixed-count runs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import cvrp.ga as ga_module
from cvrp.ga import GAParams, genetic_algorithm
from cvrp.generator import generate_team_instance
from cvrp.instance import Instance
from cvrp.local_search import apply_relocate, descent, full_distance, neighbours, relocate_delta, solution_overload
from cvrp.parser import load_instance
from cvrp.savings import clarke_wright
from cvrp.split import split
from cvrp.tabu import TabuParams, initial_routes, tabu_search
from experiments.analysis import write_table
from experiments.config import GA_LS_BASE, ROOT, instance_path
from experiments.run_experiments import git_commit, write_hardware

SLOPE_TOL = 0.15          # verdict tolerance on the slope: timer noise, cache effects, only 4-5 sizes
N_FIT_MIN = 100           # second fit uses n >= 100 only (small n: constant per-call overhead dominates)
TIMING_COLUMNS = ["component", "n", "rep", "seconds", "units_of_work", "units", "Q", "K", "L_eff", "extra_json",
                  "git_commit", "hardware"]
SHARE_COLUMNS = ["instance", "n", "generations", "total_s", "descent_s", "split_decode_s", "rest_s",
                 "descent_share", "split_decode_share", "rest_share"]
FIT_COLUMNS = ["component", "x", "theory", "theory_slope", "fit_all_slope", "fit_all_se", "fit_all_r2",
               "fit_n100_slope", "fit_n100_se", "fit_n100_r2", "verdict", "note"]


@dataclass(frozen=True)
class Config:
    """Work per repetition. FULL is the real measurement, SMOKE only checks the pipeline."""

    sizes: tuple[int, ...]
    reps: int
    split_tours: int          # random tours per split_unlimited repetition
    splitk_tours: int         # random tours per split_k repetition
    ga_generations: int       # plain GA: generations per run (time = mean of gen_times)
    gals_generations: int     # GA+LS
    tabu_iterations: int      # TS: time per iteration = (t(I) - t(0)) / I
    moves: int                # random relocates for delta vs full evaluation
    descent_kappa: int        # neighbour list size of the descent
    q_n: int                  # Q-sweep: fixed n
    q_values: tuple[int, ...]
    profile_generations: int  # GA+LS cost-share profile


FULL = Config((50, 100, 200, 400, 800), 7, 20, 5, 10, 5, 50, 2000, 10, 200, (50, 100, 200, 400, 800), 20)
SMOKE = Config((20, 40), 2, 3, 2, 2, 1, 5, 200, 10, 40, (50, 100, 200), 2)

# component -> (theory text, theoretical cost as a function of n, note on possible differences)
THEORY = {
    "cw": ("O(n^2 log n)", lambda n: n * n * math.log(n),
           "the log factor puts the expected slope a bit above 2; the n^2 savings list also stresses the cache"),
    "split_unlimited": ("O(n L), L constant", lambda n: n,
                        "L is constant (Q fixed); per-call overhead (tour check, prefix sums) flattens small n"),
    "split_k": ("O(K n L), K ~ n", lambda n: n * n,
                "K grows linearly with n because Q is fixed, so K*n*L is quadratic"),
    "ga_generation": ("O(P n L)", lambda n: n,
                      "offspring needing more than K routes also run the K-limited Split, O(K n L) ~ n^2, even "
                      "when it fails; arcs_per_decode counts successful K-limited splits only, so it does not "
                      "show this fallback"),
    "gals_generation": ("O(P (n L + passes n kappa))", lambda n: n,
                        "dominated by the descent, whose number of passes grows with n; decode may also fall back "
                        "to the K-limited Split"),
    "tabu_iteration": ("O(n kappa)", lambda n: n,
                       "candidate scan O(n kappa) plus O(K + L) to apply the move and update the tabu list"),
    "tabu_delta": ("O(1) per move", lambda n: 1.0, "the delta does not depend on n: the line should be flat"),
    "tabu_full_eval": ("O(n) per move", lambda n: n, "copying the routes and re-summing every route is O(n)"),
    "descent": ("O(passes n kappa)", lambda n: n,
                "the number of passes is empirical and grows with n, so the total grows faster than one pass"),
    "descent_per_pass": ("O(n kappa)", lambda n: n, "one pass scans n customers x kappa neighbours"),
}
N_COMPONENTS = tuple(THEORY)
L_COMPONENTS = ("split_unlimited_q", "split_k_q")       # Q-sweep: x = L_eff at fixed n


# --------------------------------------------------------------------------- timing primitives

def timed(fn):
    """Run fn() with the garbage collector off; returns (seconds, result). Time: one call of fn."""
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter()
        out = fn()
        return time.perf_counter() - start, out
    finally:
        gc.enable()


def load_tuned() -> dict:
    return json.loads((ROOT / "experiments" / "tuned_params.json").read_text())


def random_tours(n: int, count: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    tours = []
    for _ in range(count):
        t = list(range(1, n + 1))
        rng.shuffle(t)
        tours.append(t)
    return tours


def _m(component, seconds, units, units_name, L_eff=None, **extra) -> dict:
    return {"component": component, "seconds": seconds, "units_of_work": units, "units": units_name,
            "L_eff": L_eff, "extra": extra}


# --------------------------------------------------------------------------- measurements (one repetition each)

def m_cw(inst: Instance, rep: int, cfg: Config, tuned: dict) -> list[dict]:
    t, res = timed(lambda: clarke_wright(inst, warn=False))
    return [_m("cw", t, res.iterations, "savings pairs examined", routes=res.n_routes)]


def _split_batch(inst, tours, max_routes):
    out = []
    for t in tours:
        out.append(split(inst, t, max_routes=max_routes))
    return out


def m_split(inst: Instance, rep: int, cfg: Config, tuned: dict, suffix: str = "") -> list[dict]:
    """Mean time per split() call, unlimited and K-limited, on random tours (seed = rep)."""
    rows = []
    for name, count, limit in (("split_unlimited", cfg.split_tours, None), ("split_k", cfg.splitk_tours, inst.K)):
        tours = random_tours(inst.n, count, rep)
        t, results = timed(lambda: _split_batch(inst, tours, limit))
        found = [r for r in results if r is not None]
        arcs = statistics.fmean(r.arcs_examined for r in found) if found else math.nan
        L_eff = statistics.fmean(r.arcs_examined for r in found) / inst.n if found and limit is None else None
        rows.append(_m(name + suffix, t / count, arcs, "arcs examined per call", L_eff,
                       tours=count, feasible_share=len(found) / count))
    if rows[1]["L_eff"] is None:
        rows[1]["L_eff"] = rows[0]["L_eff"]          # same instance and Q: report the unlimited L_eff
    return rows


def m_ga(inst: Instance, rep: int, cfg: Config, tuned: dict) -> list[dict]:
    """Plain GA: mean seconds per generation (gen_times excludes the initial population)."""
    params = GAParams(seed=rep, max_generations=cfg.ga_generations, **tuned["ga"])
    _, res = timed(lambda: genetic_algorithm(inst, params))
    return [_m("ga_generation", statistics.fmean(res.gen_times), res.decodes, "decodes (run total)",
               generations=res.generations, arcs_per_decode=res.splits_arcs_examined / res.decodes)]


def m_gals(inst: Instance, rep: int, cfg: Config, tuned: dict) -> list[dict]:
    """GA+LS (memetic, full descent): mean seconds per generation."""
    params = GAParams(seed=rep, max_generations=cfg.gals_generations, **{**GA_LS_BASE, **tuned["ga_ls"]})
    _, res = timed(lambda: genetic_algorithm(inst, params))
    return [_m("gals_generation", statistics.fmean(res.gen_times), res.decodes, "decodes (run total)",
               generations=res.generations, local_searches=res.local_searches, ls_time=res.ls_time,
               arcs_per_decode=res.splits_arcs_examined / res.decodes)]


def m_tabu(inst: Instance, rep: int, cfg: Config, tuned: dict) -> list[dict]:
    """TS: (t(I iterations) - t(0 iterations)) / I, both with seed rep, so the setup
    (CW + neighbour lists, O(n^2 log n)) cancels out."""
    def run(iters):
        return timed(lambda: tabu_search(inst, TabuParams(seed=rep, max_iterations=iters, **tuned["tabu"])))
    t0, _ = run(0)
    t1, res = run(cfg.tabu_iterations)
    return [_m("tabu_iteration", (t1 - t0) / cfg.tabu_iterations, res.moves_evaluated / max(res.iterations, 1),
               "moves evaluated per iteration", setup_s=t0, iterations=res.iterations)]


def m_delta(inst: Instance, rep: int, cfg: Config, tuned: dict) -> list[dict]:
    """Seconds per random relocate: O(1) delta vs copy + apply + full recomputation, O(n)
    (the loop of tabu.benchmark_delta, on a generated instance)."""
    routes, _ = initial_routes(inst)
    D, q, Q = inst.dist.tolist(), [int(x) for x in inst.demand], inst.Q
    loads = [sum(q[v] for v in r) for r in routes]
    rng = random.Random(rep)
    non_empty = [r for r, route in enumerate(routes) if route]
    moves = []
    for _ in range(cfg.moves):
        r1 = rng.choice(non_empty)
        i = rng.randrange(len(routes[r1]))
        r2 = rng.randrange(len(routes))
        j = rng.randrange(len(routes[r2]) + (0 if r2 == r1 else 1))
        moves.append((r1, i, r2, j))

    def by_delta():
        for m in moves:
            dd, dover = relocate_delta(D, q, Q, routes, loads, *m)
            _ = dd + dover

    def by_full():
        for m in moves:
            new = apply_relocate(routes, *m)
            _ = full_distance(D, new) + solution_overload(q, Q, new)

    t_delta, _ = timed(by_delta)
    t_full, _ = timed(by_full)
    return [_m("tabu_delta", t_delta / cfg.moves, cfg.moves, "moves"),
            _m("tabu_full_eval", t_full / cfg.moves, cfg.moves, "moves", speedup=t_full / t_delta)]


def m_descent(inst: Instance, rep: int, cfg: Config, tuned: dict) -> list[dict]:
    """One first-improvement descent from the CW routes (deterministic: rep only repeats it)."""
    D = inst.dist.tolist()
    neigh = neighbours(D, inst.n, cfg.descent_kappa)
    start = clarke_wright(inst, warn=False).solution.routes
    stats: dict = {}
    t, _ = timed(lambda: descent(inst, start, neigh, D=D, stats=stats))
    return [_m("descent", t, stats["evaluated"], "moves evaluated", passes=stats["passes"],
               moves_applied=stats["moves_applied"]),
            _m("descent_per_pass", t / stats["passes"], stats["evaluated"] / stats["passes"],
               "moves evaluated per pass", passes=stats["passes"])]


MEASUREMENTS = (m_cw, m_split, m_ga, m_gals, m_tabu, m_delta, m_descent)


def _row(m: dict, inst: Instance, rep: int, commit: str, hardware: str) -> dict:
    return {"component": m["component"], "n": inst.n, "rep": rep, "seconds": m["seconds"],
            "units_of_work": m["units_of_work"], "units": m["units"], "Q": inst.Q, "K": inst.K,
            "L_eff": "" if m["L_eff"] is None else m["L_eff"],
            "extra_json": json.dumps(m["extra"], sort_keys=True), "git_commit": commit, "hardware": hardware}


def measure_all(cfg: Config, commit: str, hardware: str, log=print) -> list[dict]:
    """Every component at every size (warm-up + reps), then the Q-sweep of Split."""
    tuned = load_tuned()
    instances = {n: generate_team_instance(seed=n, n=n) for n in cfg.sizes}
    rows = []
    for measure in MEASUREMENTS:
        for n, inst in instances.items():
            measure(inst, 0, cfg, tuned)                                   # warm-up, not recorded
            for rep in range(cfg.reps):
                rows += [_row(m, inst, rep, commit, hardware) for m in measure(inst, rep, cfg, tuned)]
            log(f"{measure.__name__[2:]:<8} n={n:<4} done")
    for Q in cfg.q_values:
        inst = generate_team_instance(seed=cfg.q_n, n=cfg.q_n, Q=Q)
        m_split(inst, 0, cfg, tuned, suffix="_q")
        for rep in range(cfg.reps):
            rows += [_row(m, inst, rep, commit, hardware) for m in m_split(inst, rep, cfg, tuned, suffix="_q")]
        log(f"split Q-sweep n={cfg.q_n} Q={Q} done")
    return rows


# --------------------------------------------------------------------------- cost-share profile (GA+LS)

def profile_ga_ls(inst: Instance, generations: int, tuned: dict, seed: int = 0) -> dict:
    """Share of a GA+LS run spent in the descent, in Split decoding and elsewhere.
    Manual timers: cvrp.ga.decode and cvrp.ga.descent are wrapped for this run only (the
    originals are restored in 'finally'). Shares cover the whole run, initial population included."""
    spent = {"decode": 0.0, "descent": 0.0}
    originals = {"decode": ga_module.decode, "descent": ga_module.descent}

    def wrap(name):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return originals[name](*args, **kwargs)
            finally:
                spent[name] += time.perf_counter() - start
        return wrapper

    params = GAParams(seed=seed, max_generations=generations, **{**GA_LS_BASE, **tuned["ga_ls"]})
    try:
        ga_module.decode, ga_module.descent = wrap("decode"), wrap("descent")
        total, _ = timed(lambda: genetic_algorithm(inst, params))
    finally:
        ga_module.decode, ga_module.descent = originals["decode"], originals["descent"]
    rest = total - spent["decode"] - spent["descent"]
    return {"instance": inst.name, "n": inst.n, "generations": generations, "total_s": total,
            "descent_s": spent["descent"], "split_decode_s": spent["decode"], "rest_s": rest,
            "descent_share": spent["descent"] / total, "split_decode_share": spent["decode"] / total,
            "rest_share": rest / total}


# --------------------------------------------------------------------------- fits

def fit_loglog(x, y) -> dict:
    """Least squares log10(y) = a + b*log10(x) (numpy.polyfit). SE of b =
    sqrt(SSR/(m-2) / sum (u - mean u)^2), nan if m < 3; R^2 = 1 - SSR/SST. Time: O(m)."""
    u, v = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    m = len(u)
    if m < 2:
        return {"slope": math.nan, "intercept": math.nan, "se": math.nan, "r2": math.nan}
    b, a = np.polyfit(u, v, 1)
    resid = v - (a + b * u)
    ssr, sst, sxx = float(resid @ resid), float(((v - v.mean()) ** 2).sum()), float(((u - u.mean()) ** 2).sum())
    se = math.sqrt(ssr / (m - 2) / sxx) if m >= 3 else math.nan
    r2 = 1.0 - ssr / sst if sst > 0 else (1.0 if ssr == 0 else math.nan)
    return {"slope": float(b), "intercept": float(a), "se": se, "r2": r2}


def theory_slope(component: str, ns) -> float:
    """Slope of the theoretical cost over the same n values (n^2 log n -> a bit above 2)."""
    f = THEORY[component][1]
    return fit_loglog(ns, [f(n) for n in ns])["slope"]


def verdict(slope: float, se: float, theory: float, tol: float = SLOPE_TOL) -> str:
    """'consistent' if |slope - theory| <= max(2*SE, tol) (an unknown SE counts as 0),
    otherwise 'differs'; 'n/a' without a slope."""
    if math.isnan(slope) or math.isnan(theory):
        return "n/a"
    bound = max(2 * se if not math.isnan(se) else 0.0, tol)
    return "consistent" if abs(slope - theory) <= bound else "differs"


def medians(rows: list[dict], component: str, x: str = "n") -> list[tuple[float, float, float, float]]:
    """(x, median, min, max) of the seconds per x value (x = n, or the median L_eff for the
    Q-sweep), sorted by x; non-positive times are dropped. Time: O(rows)."""
    groups: dict = {}
    for r in rows:
        if r["component"] == component and float(r["seconds"]) > 0:
            key = int(r["n"]) if x == "n" else int(r["Q"])
            groups.setdefault(key, []).append(r)
    out = []
    for _, rs in groups.items():
        t = [float(r["seconds"]) for r in rs]
        xv = float(rs[0]["n"]) if x == "n" else statistics.median(float(r["L_eff"]) for r in rs)
        out.append((xv, statistics.median(t), min(t), max(t)))
    return sorted(out)


def _fit_row(component, x_name, theory_text, theory_value, fit_all, fit_big, note) -> dict:
    decisive = fit_big if x_name == "n" else fit_all
    return {"component": component, "x": x_name, "theory": theory_text, "theory_slope": theory_value,
            "fit_all_slope": fit_all["slope"], "fit_all_se": fit_all["se"], "fit_all_r2": fit_all["r2"],
            "fit_n100_slope": fit_big["slope"], "fit_n100_se": fit_big["se"], "fit_n100_r2": fit_big["r2"],
            "verdict": verdict(decisive["slope"], decisive["se"], theory_value), "note": note}


NAN_FIT = {"slope": math.nan, "se": math.nan, "r2": math.nan}


def build_fits(rows: list[dict]) -> list[dict]:
    """One fit row per component present in the timing rows. Time: O(rows)."""
    out = []
    for comp in N_COMPONENTS:
        pts = medians(rows, comp)
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        big = [(x, y) for x, y in zip(xs, ys) if x >= N_FIT_MIN]
        fit_big = fit_loglog(*zip(*big)) if len(big) >= 2 else NAN_FIT
        th = theory_slope(comp, [x for x, _ in big]) if len(big) >= 2 else math.nan
        out.append(_fit_row(comp, "n", THEORY[comp][0], th, fit_loglog(xs, ys), fit_big, THEORY[comp][2]))
    for comp, text, note in (("split_unlimited_q", "O(n L) at fixed n", "linear in L (Q grows, n fixed)"),
                             ("split_k_q", "O(K n L), K ~ 1/Q", "K*L is about constant: K shrinks as Q grows")):
        pts = medians(rows, comp, x="L")
        if len(pts) < 2:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if comp == "split_unlimited_q":
            th = fit_loglog(xs, xs)["slope"]
        else:                                           # theory K*L from the measured K and L_eff per Q
            k_of = {}
            for r in rows:
                if r["component"] == comp:
                    k_of[int(r["Q"])] = int(r["K"])
            th = fit_loglog(xs, [k * x for k, x in zip([k_of[q] for q in sorted(k_of)], xs)])["slope"]
        out.append(_fit_row(comp, "L", text, th, fit_loglog(xs, ys), NAN_FIT, note))
    return out


FITS_NOTE = (f"Fit: log10(time) = a + b*log10(x) by least squares on the median of the repetitions per size; "
             f"fit_n100 uses n >= {N_FIT_MIN} only (small n are dominated by constant per-call overhead). "
             f"theory_slope = slope of the theoretical cost over the same sizes (n^2 log n gives a bit more than 2). "
             f"verdict = consistent if |slope - theory_slope| <= max(2*SE, {SLOPE_TOL}) (n-components: the n >= "
             f"{N_FIT_MIN} fit; Q-sweep rows: the fit over all Q), else differs. The {SLOPE_TOL} tolerance covers "
             f"timer noise, cache effects and the few sizes; SE, slope and R^2 are reported unchanged.")


def write_fits(fits: list[dict], tables_dir: Path) -> list[Path]:
    """complexity_fits.csv and .md (deterministic, LF; experiments.analysis.write_table)."""
    fmt = {c: ".3f" for c in FIT_COLUMNS}
    fmt.update({"fit_all_r2": ".4f", "fit_n100_r2": ".4f"})
    return write_table(fits, FIT_COLUMNS, "complexity_fits", tables_dir, fmt,
                       "Complexity validation: log-log slopes", FITS_NOTE)


def write_rows(path: Path, columns: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


# --------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_component(ax, rows: list[dict], comp: str, fit: dict | None) -> None:
    pts = medians(rows, comp)
    xs, med = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    lo, hi = med - np.array([p[2] for p in pts]), np.array([p[3] for p in pts]) - med
    ax.errorbar(xs, med, yerr=[lo, hi], marker="o", color="tab:blue", capsize=3, linestyle="none",
                label="median (bars: min-max)")
    if fit is not None and not math.isnan(fit["fit_n100_slope"]):
        big = xs[xs >= N_FIT_MIN]
        f = fit_loglog(big, med[xs >= N_FIT_MIN])
        ax.plot(big, 10 ** (f["intercept"] + f["slope"] * np.log10(big)), color="tab:blue", linewidth=1.5,
                label=f"fit n>={N_FIT_MIN}: slope {f['slope']:.2f}")
    for k, style in ((1, (0, (4, 3))), (2, (0, (1, 2)))):
        ax.plot(xs, med[0] * (xs / xs[0]) ** k, color="grey", linestyle=style, linewidth=1, label=f"slope {k}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (customers)")
    ax.set_ylabel("seconds")
    ax.set_title(f"{comp}: {THEORY[comp][0]}", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


def plot_components(rows: list[dict], fits: list[dict], figures_dir: Path) -> list[Path]:
    """One log-log figure per component plus the 3x3 overview."""
    plt = _plt()
    fit_of = {f["component"]: f for f in fits}
    paths = []
    for comp in N_COMPONENTS:
        if not medians(rows, comp):
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        _draw_component(ax, rows, comp, fit_of.get(comp))
        ax.legend(fontsize=8)
        fig.tight_layout()
        paths.append(figures_dir / f"complexity_{comp}.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    overview = [c for c in N_COMPONENTS if c != "descent" and medians(rows, c)]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, comp in zip(axes.flat, overview):
        _draw_component(ax, rows, comp, fit_of.get(comp))
    for ax in list(axes.flat)[len(overview):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Running time vs n (log-log); grey: reference slopes 1 (dashed) and 2 (dotted)", fontsize=12)
    fig.tight_layout()
    paths.append(figures_dir / "complexity_overview.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


def plot_split_vs_L(rows: list[dict], out_path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for comp, colour, label in (("split_unlimited_q", "tab:blue", "unlimited Split, O(n L)"),
                                ("split_k_q", "tab:red", "K-limited Split, O(K n L), K ~ 1/Q")):
        pts = medians(rows, comp, x="L")
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], marker="o", capsize=3,
                    color=colour, label=label)
    n = next(int(r["n"]) for r in rows if r["component"] == "split_unlimited_q")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L_eff = arcs examined / n (customers per route)")
    ax.set_ylabel("seconds per split() call")
    ax.set_title(f"Split vs L at fixed n = {n} (Q varied)", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_delta_vs_full(rows: list[dict], out_path: Path) -> Path:
    plt = _plt()
    delta, full = medians(rows, "tabu_delta"), medians(rows, "tabu_full_eval")
    xs = [p[0] for p in delta]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(xs, [p[1] * 1e6 for p in delta], marker="o", color="tab:green", label="delta evaluation, O(1)")
    ax.plot(xs, [p[1] * 1e6 for p in full], marker="s", color="tab:red", label="full re-evaluation, O(n)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (customers)")
    ax.set_ylabel("microseconds per relocate move")
    ax2 = ax.twinx()
    ax2.plot(xs, [f[1] / d[1] for d, f in zip(delta, full)], color="black", linestyle="--", marker=".",
             label="speedup (right axis)")
    ax2.set_ylabel("speedup full / delta")
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=9, loc="upper left")
    ax.set_title("Tabu Search move evaluation: delta vs full", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- entry point

def run(cfg: Config, out: Path, tables: Path, figures: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    hw_path = write_hardware(out, 1)
    cpu = next((line.split(":", 1)[1].strip() for line in hw_path.read_text().splitlines()
                if line.startswith("cpu:")), "unknown")
    started = time.perf_counter()
    rows = measure_all(cfg, git_commit(), cpu)
    timing_csv = write_rows(out / "timing.csv", TIMING_COLUMNS, rows)
    tuned = load_tuned()
    share = [profile_ga_ls(load_instance(instance_path(name)), cfg.profile_generations, tuned)
             for name in ("team", "A-n80-k10")]
    share_csv = write_rows(out / "share_ga_ls.csv", SHARE_COLUMNS, share)
    fits = build_fits(rows)
    tables_written = write_fits(fits, tables)
    figs = plot_components(rows, fits, figures)
    figs += [plot_split_vs_L(rows, figures / "split_vs_L.png"), plot_delta_vs_full(rows, figures / "delta_vs_full.png")]
    print(f"done in {time.perf_counter() - started:.0f} s")
    return {"rows": rows, "fits": fits, "share": share,
            "files": [timing_csv, hw_path, share_csv, *tables_written, *figs]}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Step 6d: timing experiment (complexity validation).")
    ap.add_argument("--sizes", default=",".join(map(str, FULL.sizes)))
    ap.add_argument("--reps", type=int, default=FULL.reps)
    ap.add_argument("--out", default=None, help="raw output (default results/timing; smoke: results/smoke_timing)")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--smoke", action="store_true", help="sizes 20,40, 2 reps, tiny work; everything in --out")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.smoke:
        cfg = SMOKE
        out = Path(args.out or "results/smoke_timing")
        tables = figures = out
    else:
        cfg = Config(**{**FULL.__dict__, "sizes": tuple(int(s) for s in args.sizes.split(",")), "reps": args.reps})
        out, tables, figures = Path(args.out or "results/timing"), Path(args.tables), Path(args.figures)
    result = run(cfg, out, tables, figures)
    for path in result["files"]:
        print(f"wrote {path}")
    return result


if __name__ == "__main__":
    main()
