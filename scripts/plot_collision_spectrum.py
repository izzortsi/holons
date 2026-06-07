#!/usr/bin/env python3
"""Plot the glider collision defect spectrum.

Reads the CSV produced by scripts/collision_spectrum.py (columns:
separation, tau, defect). Each (separation, tau) pair maps to an integer
defect count, so the data forms a 2D grid. This renders:
  1. A heatmap of defect over the separation x tau grid.
  2. Line plots of defect vs tau, one line per separation.
Both panels are saved to a single PNG.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt


# Repo root is the parent of scripts/; results/ lives alongside scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plot the glider collision defect spectrum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=RESULTS_DIR / "collision_spectrum.csv",
        help="path to the CSV result file to plot",
    )
    p.add_argument(
        "--output-png",
        type=Path,
        default=RESULTS_DIR / "collision_spectrum.png",
        help="path to write the PNG figure",
    )
    return p.parse_args(argv)


def load(csv_path):
    separations = []
    taus = []
    defects = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("separation"):
                continue
            separations.append(int(row["separation"]))
            taus.append(int(row["tau"]))
            defects.append(int(row["defect"]))
    return np.array(separations), np.array(taus), np.array(defects)


def to_grid(separations, taus, defects):
    sep_axis = np.array(sorted(set(separations.tolist())))
    tau_axis = np.array(sorted(set(taus.tolist())))
    sep_index = {value: i for i, value in enumerate(sep_axis)}
    tau_index = {value: i for i, value in enumerate(tau_axis)}

    grid = np.full((len(sep_axis), len(tau_axis)), np.nan)
    for sep, tau, defect in zip(separations, taus, defects):
        grid[sep_index[sep], tau_index[tau]] = defect
    return sep_axis, tau_axis, grid


def main(argv=None):
    args = parse_args(argv)

    separations, taus, defects = load(args.input_csv)
    sep_axis, tau_axis, grid = to_grid(separations, taus, defects)

    fig, (ax_heat, ax_lines) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: heatmap
    image = ax_heat.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[tau_axis.min(), tau_axis.max(), sep_axis.min(), sep_axis.max()],
    )
    ax_heat.set_xlabel("tau")
    ax_heat.set_ylabel("separation")
    ax_heat.set_title("Defect count over separation x tau")
    colorbar = fig.colorbar(image, ax=ax_heat)
    colorbar.set_label("defect")

    # Panel 2: defect vs tau, one line per separation
    for i, sep in enumerate(sep_axis):
        ax_lines.plot(tau_axis, grid[i], marker="o", markersize=3, label=str(sep))
    ax_lines.set_xlabel("tau")
    ax_lines.set_ylabel("defect")
    ax_lines.set_title("Defect vs tau by separation")
    ax_lines.legend(title="separation", fontsize=7, ncol=2)
    ax_lines.grid(True, alpha=0.3)

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=150)
    print("Wrote " + str(args.output_png))


if __name__ == "__main__":
    main()
