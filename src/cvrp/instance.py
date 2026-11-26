# Original code (CMP 3005 term project).
"""Step 1: Instance dataclass + distance functions.

Conventions (see CLAUDE.md):
  - vertex 0 is the depot, customers are 1..n;
  - Augerat instances use the TSPLIB EUC_2D rule  d = int(sqrt(dx^2 + dy^2) + 0.5);
  - the team (MediDrone) instance uses the unrounded Euclidean distance plus a
    penalty P whenever the two endpoints have different side labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Distance-type names stored in Instance.distance_type and in .vrp files.
EUC_2D_ROUNDED = "EUC_2D_ROUNDED"
EUC_2D_BOSPHORUS = "EUC_2D_BOSPHORUS"

# The vertical line x = SPLIT_X separates the European and the Asian side.
DEFAULT_SPLIT_X = 500.0


def side_label(x: float, split_x: float = DEFAULT_SPLIT_X) -> str:
    """Return "E" (European) if x < split_x, else "A" (Asian).

    A point exactly on the line (the depot at x = 500) is "A", so every vertex
    gets exactly one label. Time: O(1).
    """
    return "E" if x < split_x else "A"


def euclidean_matrix(coords: np.ndarray) -> np.ndarray:
    """Unrounded Euclidean distance between every pair of points.

    coords has shape (m, 2); result has shape (m, m). Time: O(m^2).
    """
    diff = coords[:, None, :] - coords[None, :, :]  # shape (m, m, 2)
    return np.sqrt((diff ** 2).sum(axis=2))


def euc2d_rounded_matrix(coords: np.ndarray) -> np.ndarray:
    """TSPLIB EUC_2D distances: int(sqrt(dx^2 + dy^2) + 0.5) for every pair.

    Needed to reproduce the published best-known solutions. Stored as float so
    all instances share one matrix type. Time: O(m^2).
    """
    return np.floor(euclidean_matrix(coords) + 0.5)


def bosphorus_matrix(coords: np.ndarray, sides: list[str], penalty: float) -> np.ndarray:
    """Unrounded Euclidean distance + penalty if side(i) != side(j).

    This stays a metric (proved in our theory write-up): the penalty is added
    based on the endpoint labels only, never on whether a segment crosses x = 500.
    Time: O(m^2).
    """
    labels = np.array(sides)
    different_side = labels[:, None] != labels[None, :]  # boolean (m, m)
    return euclidean_matrix(coords) + penalty * different_side


@dataclass
class Instance:
    """One CVRP instance with its distance matrix precomputed once (O(n^2)).

    coords : (n+1) x 2 float array, row 0 = depot
    demand : length n+1 int array, demand[0] = 0
    dist   : (n+1) x (n+1) float array, symmetric, zero diagonal
    side   : list of "E"/"A" labels (team instance) or None
    """

    name: str
    n: int
    coords: np.ndarray
    demand: np.ndarray
    Q: int
    K: int
    dist: np.ndarray
    side: list[str] | None
    penalty: float
    bks: float | None
    distance_type: str
    split_x: float = DEFAULT_SPLIT_X

    @classmethod
    def from_euc2d_rounded(cls, name: str, coords: np.ndarray, demand: np.ndarray,
                           Q: int, K: int, bks: float | None = None) -> Instance:
        """Build an instance with TSPLIB-rounded distances. Time: O(n^2)."""
        coords = np.asarray(coords, dtype=float)
        demand = np.asarray(demand, dtype=int)
        return cls(name=name, n=len(coords) - 1, coords=coords, demand=demand, Q=Q, K=K,
                   dist=euc2d_rounded_matrix(coords), side=None, penalty=0.0, bks=bks,
                   distance_type=EUC_2D_ROUNDED)

    @classmethod
    def from_bosphorus(cls, name: str, coords: np.ndarray, demand: np.ndarray,
                       Q: int, K: int, penalty: float = 150.0,
                       split_x: float = DEFAULT_SPLIT_X, bks: float | None = None) -> Instance:
        """Build a team instance: float Euclidean + penalty across sides. Time: O(n^2).

        Side labels are computed here from the given coordinates, so callers must
        pass the coordinates exactly as they will be written to file.
        """
        coords = np.asarray(coords, dtype=float)
        demand = np.asarray(demand, dtype=int)
        sides = [side_label(float(x), split_x) for x in coords[:, 0]]
        return cls(name=name, n=len(coords) - 1, coords=coords, demand=demand, Q=Q, K=K,
                   dist=bosphorus_matrix(coords, sides, penalty), side=sides,
                   penalty=float(penalty), bks=bks, distance_type=EUC_2D_BOSPHORUS,
                   split_x=float(split_x))

    def total_demand(self) -> int:
        """Sum of all customer demands. Time: O(n)."""
        return int(self.demand.sum())
