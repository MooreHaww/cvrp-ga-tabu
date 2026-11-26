# Original code (CMP 3005 term project). Instance rules: course brief, Task 2(b) and 2(c).
"""Step 1: team (MediDrone) + tiny instance generator.

All randomness comes from an explicit random.Random(seed) object (stdlib, so the
same seed gives the same instance on every machine and numpy version).

CLI:
    python -m cvrp.generator --seed N     # writes data/team/team_n100_seed{N}.vrp
    python -m cvrp.generator --tiny       # writes data/tiny/tiny_n{7,8,9}.vrp
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np

from cvrp.instance import EUC_2D_BOSPHORUS, Instance

SQUARE = 1000.0             # customers live in [0, 1000]^2
DEPOT_XY = (500.0, 500.0)   # depot at the centre of the square
RANDOM_SHARE = 0.20         # about 20% of customers are uniform over the square
TINY_SIZES = (7, 8, 9)

# TODO: replace 0 by the real seed = sum of the last four digits of all team members' student IDs.
DEFAULT_SEED = 0


def ffd_bin_count(demands: list[int], Q: int) -> int:
    """First-Fit-Decreasing bin packing: number of bins of size Q needed.

    Sort demands in decreasing order, put each into the first bin with room.
    Time: O(n log n + n * B), B = number of bins opened.
    """
    free: list[int] = []  # free[b] = remaining capacity of bin b
    for q in sorted(demands, reverse=True):
        for b in range(len(free)):
            if free[b] >= q:
                free[b] -= q
                break
        else:
            free.append(Q - q)
    return len(free)


def warn_if_ffd_exceeds_k(name: str, demands: list[int], Q: int, K: int) -> int:
    """Print a WARNING (no silent fix) if FFD needs more than K bins. Time: O(n log n + n * B)."""
    bins = ffd_bin_count(demands, Q)
    if bins > K:
        print(f"WARNING: {name}: First-Fit-Decreasing needs {bins} bins > K = {K}")
    return bins


def _clip(v: float) -> float:
    """Clamp a coordinate into [0, SQUARE]. Time: O(1)."""
    return min(max(v, 0.0), SQUARE)


def _cluster_centres(rng: random.Random, m: int) -> list[tuple[float, float]]:
    """m centres in [100, 900]^2; centre 0 on the European side, centre 1 on the Asian side.

    Forcing the first two guarantees at least one centre on each side of x = 500. Time: O(m).
    """
    centres = [(rng.uniform(100.0, 499.0), rng.uniform(100.0, 900.0)),
               (rng.uniform(500.0, 900.0), rng.uniform(100.0, 900.0))]
    for _ in range(m - 2):
        centres.append((rng.uniform(100.0, 900.0), rng.uniform(100.0, 900.0)))
    return centres


def _team_customer_points(rng: random.Random, n: int) -> list[tuple[float, float]]:
    """Clustered + uniform customer locations, rounded to 2 decimals. Time: O(n)."""
    m = rng.choice([4, 5, 6])
    centres = _cluster_centres(rng, m)
    sigmas = [rng.uniform(40.0, 80.0) for _ in range(m)]
    n_random = round(RANDOM_SHARE * n)
    n_clustered = n - n_random

    points = []
    for i in range(n_clustered):          # round-robin: cluster sizes differ by at most 1
        c = i % m
        cx, cy = centres[c]
        points.append((_clip(rng.gauss(cx, sigmas[c])), _clip(rng.gauss(cy, sigmas[c]))))
    for _ in range(n_random):
        points.append((rng.uniform(0.0, SQUARE), rng.uniform(0.0, SQUARE)))
    # Round FIRST: side labels are computed later from these rounded values,
    # exactly as the parser will see them in the file.
    return [(round(x, 2), round(y, 2)) for x, y in points]


def generate_team_instance(seed: int, n: int = 100, P: float = 150.0, Q: int = 100) -> Instance:
    """MediDrone team instance (course brief, Task 2b).

    Depot (500, 500); 4-6 Gaussian clusters (sigma in [40, 80]) + ~20% uniform
    customers; demands uniform in [1, 30]; K = ceil(sum(q) / Q) + 1;
    distance = float Euclidean + P across sides. Time: O(n^2) (distance matrix).
    """
    rng = random.Random(seed)
    points = _team_customer_points(rng, n)
    demands = [rng.randint(1, 30) for _ in range(n)]
    K = math.ceil(sum(demands) / Q) + 1
    name = f"team_n{n}_seed{seed}"
    warn_if_ffd_exceeds_k(name, demands, Q, K)
    coords = np.array([DEPOT_XY] + points)
    return Instance.from_bosphorus(name, coords, np.array([0] + demands), Q, K, penalty=P)


def generate_tiny_instance(n: int, seed: int) -> Instance:
    """Tiny instance with rounded EUC_2D distances where capacity binds.

    Integer coordinates in [0, 100]^2, depot (50, 50), demands in [1, 10],
    Q = max(max q, ceil(sum(q) / 2.5)) so sum(q) > Q and at least 2 routes are
    needed; K = ceil(sum(q) / Q) + 1. Time: O(n^2).
    """
    rng = random.Random(seed)
    points = [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
    demands = [rng.randint(1, 10) for _ in range(n)]
    Q = max(max(demands), math.ceil(sum(demands) / 2.5))
    K = math.ceil(sum(demands) / Q) + 1
    name = f"tiny_n{n}"
    warn_if_ffd_exceeds_k(name, demands, Q, K)
    coords = np.array([(50, 50)] + points)
    return Instance.from_euc2d_rounded(name, coords, np.array([0] + demands), Q, K)


def _fmt_number(v: float) -> str:
    """Integers without decimals, everything else with exactly 2 decimals. Time: O(1)."""
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"


def vrp_text(inst: Instance, comment: str) -> str:
    """TSPLIB-like text for an instance (fixed field order => deterministic). Time: O(n)."""
    bosphorus = inst.distance_type == EUC_2D_BOSPHORUS
    lines = [f"NAME : {inst.name}",
             f"COMMENT : {comment}, No of trucks: {inst.K}",
             "TYPE : CVRP",
             f"DIMENSION : {inst.n + 1}",
             f"EDGE_WEIGHT_TYPE : {EUC_2D_BOSPHORUS if bosphorus else 'EUC_2D'}"]
    if bosphorus:
        lines += [f"PENALTY : {_fmt_number(inst.penalty)}", f"SPLIT_X : {_fmt_number(inst.split_x)}"]
    lines += [f"CAPACITY : {inst.Q}", "NODE_COORD_SECTION"]
    # Team coordinates always use 2 decimals; tiny ones are integers.
    fmt = (lambda v: f"{v:.2f}") if bosphorus else (lambda v: str(int(v)))
    for v in range(inst.n + 1):  # vertex v is file node v + 1
        lines.append(f"{v + 1} {fmt(inst.coords[v, 0])} {fmt(inst.coords[v, 1])}")
    lines.append("DEMAND_SECTION")
    lines += [f"{v + 1} {int(inst.demand[v])}" for v in range(inst.n + 1)]
    lines += ["DEPOT_SECTION", "1", "-1", "EOF"]
    return "\n".join(lines) + "\n"


def write_vrp(inst: Instance, path: str | Path, comment: str) -> Path:
    """Write the instance as UTF-8 bytes with '\\n' newlines (byte-identical per seed). Time: O(n)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(vrp_text(inst, comment).encode("utf-8"))
    return path


def main() -> None:
    """Command-line entry point (see module docstring)."""
    ap = argparse.ArgumentParser(description="Generate the team or tiny CVRP instances.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="team-instance seed")
    ap.add_argument("--tiny", action="store_true", help="write tiny_n7/8/9 instead of the team instance")
    args = ap.parse_args()
    if args.tiny:
        for n in TINY_SIZES:
            inst = generate_tiny_instance(n, seed=n)  # fixed seed per size
            p = write_vrp(inst, f"data/tiny/tiny_n{n}.vrp", f"Tiny instance n={n}, seed {n}")
            print(f"wrote {p}  Q={inst.Q} K={inst.K} sum(q)={inst.total_demand()}")
    else:
        inst = generate_team_instance(args.seed)
        p = write_vrp(inst, f"data/team/team_n100_seed{args.seed}.vrp",
                      f"MediDrone team instance, seed {args.seed}")
        print(f"wrote {p}  Q={inst.Q} K={inst.K} sum(q)={inst.total_demand()} "
              f"FFD bins={ffd_bin_count(list(inst.demand[1:]), inst.Q)}")


if __name__ == "__main__":
    main()
