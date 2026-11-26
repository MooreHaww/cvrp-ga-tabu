# Original code (CMP 3005 term project). File formats: TSPLIB 95 / CVRPLIB.
"""Step 1: CVRPLIB/TSPLIB .vrp and .sol parsing.

Files number nodes 1..DIMENSION with the depot as node 1. Inside the code the
depot is 0 and customers are 1..n, so file node i becomes vertex i - 1.
CVRPLIB .sol files list customers only; their numbering is checked against the
.vrp by recomputing the cost of the routes (see map_sol_routes).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from cvrp.instance import DEFAULT_SPLIT_X, EUC_2D_BOSPHORUS, Instance

SECTION_NAMES = ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION")


def _split_header(line: str) -> tuple[str, str] | None:
    """Split "KEY : VALUE" (spaces optional) into (KEY, VALUE). Time: O(len(line))."""
    if ":" not in line:
        return None
    key, value = line.split(":", 1)
    return key.strip().upper(), value.strip()


def read_k(name: str, comment: str) -> int:
    """Number of vehicles K from the name suffix "-kX" and/or "No of trucks: X".

    Prints a warning if both are present and disagree (the name suffix wins).
    Raises ValueError if neither is present. Time: O(len(name) + len(comment)).
    """
    from_name = re.search(r"-k(\d+)$", name)
    from_comment = re.search(r"No of trucks:\s*(\d+)", comment)
    k_name = int(from_name.group(1)) if from_name else None
    k_comment = int(from_comment.group(1)) if from_comment else None
    if k_name is not None and k_comment is not None and k_name != k_comment:
        print(f"WARNING: {name}: K from name ({k_name}) != K from COMMENT ({k_comment}); using {k_name}")
    if k_name is not None:
        return k_name
    if k_comment is not None:
        return k_comment
    raise ValueError(f"{name}: cannot determine K (no '-kX' suffix and no 'No of trucks' in COMMENT)")


def _read_vrp_file(path: Path) -> tuple[dict[str, str], dict[str, list[list[str]]]]:
    """Read header fields and raw section rows from a .vrp file. Time: O(file size)."""
    header: dict[str, str] = {}
    sections: dict[str, list[list[str]]] = {name: [] for name in SECTION_NAMES}
    current = None  # name of the section being read, or None while in the header
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "EOF":
            break
        if line in SECTION_NAMES:
            current = line
            continue
        if current is None:
            pair = _split_header(line)
            if pair:
                header[pair[0]] = pair[1]
        else:
            sections[current].append(line.split())
    return header, sections


def parse_vrp(path: str | Path) -> Instance:
    """Parse a CVRPLIB/TSPLIB .vrp file (EUC_2D or our EUC_2D_BOSPHORUS) into an Instance.

    Time: O(n^2), dominated by building the distance matrix; reading is O(n).
    """
    path = Path(path)
    header, sections = _read_vrp_file(path)
    name = header.get("NAME", path.stem)
    dimension = int(header["DIMENSION"])  # number of nodes incl. depot = n + 1
    capacity = int(header["CAPACITY"])
    edge_type = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()

    # Depot must be file node 1 (that is what makes the "node i -> vertex i-1" shift valid).
    depots = [int(row[0]) for row in sections["DEPOT_SECTION"] if int(row[0]) != -1]
    if depots != [1]:
        raise ValueError(f"{name}: expected depot = node 1, got {depots}")

    coords = np.zeros((dimension, 2))
    demand = np.zeros(dimension, dtype=int)
    for node, x, y in sections["NODE_COORD_SECTION"]:
        coords[int(node) - 1] = (float(x), float(y))
    for node, q in sections["DEMAND_SECTION"]:
        demand[int(node) - 1] = int(q)
    if demand[0] != 0:
        raise ValueError(f"{name}: depot demand must be 0")

    K = read_k(name, header.get("COMMENT", ""))
    if edge_type == "EUC_2D":
        return Instance.from_euc2d_rounded(name, coords, demand, capacity, K)
    if edge_type == EUC_2D_BOSPHORUS:
        return Instance.from_bosphorus(name, coords, demand, capacity, K,
                                       penalty=float(header["PENALTY"]),
                                       split_x=float(header.get("SPLIT_X", DEFAULT_SPLIT_X)))
    raise ValueError(f"{name}: unsupported EDGE_WEIGHT_TYPE {edge_type}")


def parse_sol(path: str | Path) -> tuple[list[list[int]], float]:
    """Parse a CVRPLIB .sol file: "Route #k: a b c" lines and a "Cost X" line.

    Returns the routes exactly as written in the file (no renumbering) and the
    stated cost. Time: O(file size).
    """
    routes: list[list[int]] = []
    cost = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if line.startswith("Route"):
            routes.append([int(v) for v in line.split(":", 1)[1].split()])
        elif line.lower().startswith("cost"):
            cost = float(line.split()[1])
    if cost is None:
        raise ValueError(f"{path}: no 'Cost' line")
    return routes, cost


def _routes_cost(inst: Instance, routes: list[list[int]]) -> float:
    """Cost of depot-to-depot routes; ids outside 1..n give +inf. Time: O(n)."""
    total = 0.0
    for route in routes:
        if any(v < 1 or v > inst.n for v in route):
            return float("inf")
        path = [0] + route + [0]
        total += sum(inst.dist[a, b] for a, b in zip(path, path[1:]))
    return total


def map_sol_routes(inst: Instance, routes: list[list[int]], stated_cost: float) -> tuple[list[list[int]], int]:
    """Convert .sol ids to our vertex ids and verify the mapping by cost.

    Two candidates: ids are already customers 1..n (offset 0), or ids are file
    node numbers 2..n+1 (offset -1). The candidate whose recomputed cost equals
    the stated cost is returned with its offset. Raises ValueError if none match.
    Time: O(n).
    """
    for offset in (0, -1):
        mapped = [[v + offset for v in route] for route in routes]
        if abs(_routes_cost(inst, mapped) - stated_cost) <= 1e-6:
            return mapped, offset
    raise ValueError(f"{inst.name}: .sol routes do not reproduce the stated cost {stated_cost}")


def load_instance(path: str | Path) -> Instance:
    """parse_vrp + read the BKS from "<stem>.sol" next to the .vrp, if present.

    The BKS is never hardcoded: it is the cost stated in the .sol file, accepted
    only after map_sol_routes reproduced it. Time: O(n^2).
    """
    path = Path(path)
    inst = parse_vrp(path)
    sol_path = path.with_suffix(".sol")
    if sol_path.exists():
        routes, cost = parse_sol(sol_path)
        map_sol_routes(inst, routes, cost)  # raises if the numbering is wrong
        inst.bks = cost
    return inst


def load_bks_solution(vrp_path: str | Path) -> tuple[Instance, list[list[int]], float]:
    """Instance, BKS routes (our numbering) and stated BKS cost for a .vrp/.sol pair. Time: O(n^2)."""
    inst = load_instance(vrp_path)
    routes, cost = parse_sol(Path(vrp_path).with_suffix(".sol"))
    mapped, _ = map_sol_routes(inst, routes, cost)
    return inst, mapped, cost
