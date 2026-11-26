# Original code (CMP 3005 term project).
"""Step 1 / Milestone 1: plot an instance (designed for the team instance).

    python -m cvrp.plot_instance data/team/team_n100_seed0.vrp [--out results/figures/team_instance.png]

Customers coloured by side, depot marked, the line x = SPLIT_X labelled
"Bosphorus", marker size proportional to demand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt  # noqa: E402

from cvrp.parser import parse_vrp  # noqa: E402

SIDE_COLOURS = {"E": "tab:blue", "A": "tab:orange"}
SIDE_NAMES = {"E": "European side (E)", "A": "Asian side (A)"}


def plot_instance(vrp_path: str | Path, out_path: str | Path) -> Path:
    """Draw the instance and save it as a PNG. Time: O(n)."""
    inst = parse_vrp(vrp_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    xs, ys, q = inst.coords[1:, 0], inst.coords[1:, 1], inst.demand[1:]
    sides = inst.side[1:] if inst.side else ["E"] * inst.n
    for label in ("E", "A"):
        idx = [i for i, s in enumerate(sides) if s == label]
        if idx:
            ax.scatter(xs[idx], ys[idx], s=8 * q[idx], c=SIDE_COLOURS[label], alpha=0.7,
                       edgecolors="black", linewidths=0.4, label=SIDE_NAMES[label])
    ax.scatter(*inst.coords[0], marker="*", s=400, c="red", edgecolors="black", zorder=5, label="Depot (lab)")
    if inst.side:  # team instance: show the whole 1000 x 1000 square
        ax.set_xlim(-20, 1020)
        ax.set_ylim(-20, 1020)
        ax.axvline(inst.split_x, color="steelblue", linestyle="--", linewidth=1.5)
        ax.text(inst.split_x + 8, ax.get_ylim()[1] * 0.98, "Bosphorus", color="steelblue",
                rotation=90, va="top")
    ax.set_title(f"{inst.name}: n={inst.n}, Q={inst.Q}, K={inst.K}, P={inst.penalty:g}  "
                 f"(marker size ∝ demand)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=0.5, fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """Command-line entry point (see module docstring)."""
    ap = argparse.ArgumentParser(description="Plot a CVRP instance.")
    ap.add_argument("path", help=".vrp file")
    ap.add_argument("--out", default="results/figures/team_instance.png")
    args = ap.parse_args()
    print(f"saved {plot_instance(args.path, args.out)}")


if __name__ == "__main__":
    main()
