#!/usr/bin/env python3
"""Experiment 4: Lenia organism collision defect spectrum.

The continuous-substrate counterpart of Experiment 1. Two Orbia are set on a
head-on course at a range of separations; for each separation the superposition
defect delta(tau) = ||evolve(A+B) - (evolve(A)+evolve(B))|| / mass is tracked
over tau. The defect is exactly 0 while the organisms are separated (the
dynamics is local, so the whole equals the non-interacting aggregate of its
parts) and becomes positive at contact -- the framework's structural+temporal
defect, measured directly on the field.

Writes a CSV (separation, tau, defect) and a two-panel figure: the
(separation, tau) defect spectrum and delta(tau) curves for a few separations.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Make the project importable without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holons.lenia import ORBIUM, kernel_fft
from holons.lenia_collision import (
    orbium_velocity,
    setup_collision,
    superposition_defect_trajectory,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Lenia Orbium collision defect spectrum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--grid-size", type=int, default=96, help="square Lenia field side")
    p.add_argument("--separation-min", type=float, default=28.0)
    p.add_argument("--separation-max", type=float, default=70.0)
    p.add_argument("--separation-step", type=float, default=3.0)
    p.add_argument("--tau-max", type=int, default=120)
    p.add_argument("--tau-step", type=int, default=2)
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args(argv)


def run(args):
    velocity = orbium_velocity(ORBIUM, args.grid_size)
    spectrum = kernel_fft(ORBIUM, args.grid_size)
    separations = list(
        np.arange(
            args.separation_min, args.separation_max + 1e-9, args.separation_step
        )
    )

    grid = []
    taus_ref = None
    rows = []
    for sep in separations:
        field_a, field_b = setup_collision(args.grid_size, sep, ORBIUM, velocity)
        taus, defects = superposition_defect_trajectory(
            field_a, field_b, spectrum, ORBIUM, args.tau_max, args.tau_step
        )
        taus_ref = taus
        grid.append(defects)
        for tau, d in zip(taus, defects):
            rows.append({"separation": sep, "tau": tau, "defect": d})
        onset = next((t for t, d in zip(taus, defects) if d > 0.01), None)
        print(
            "  separation %.1f: contact onset tau=%s, peak defect %.3f"
            % (sep, "none" if onset is None else str(onset), max(defects)),
            file=sys.stderr,
        )
    return separations, taus_ref, np.array(grid), rows


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["separation", "tau", "defect"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot(separations, taus, grid, png_path):
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt

    fig, (ax_spec, ax_curve) = plt.subplots(1, 2, figsize=(14, 6))

    extent = [min(taus), max(taus), min(separations), max(separations)]
    im = ax_spec.imshow(
        grid, origin="lower", aspect="auto", extent=extent, cmap="magma"
    )
    ax_spec.set_xlabel("fine time  $\\tau$")
    ax_spec.set_ylabel("separation (cells)")
    ax_spec.set_title("Lenia collision defect spectrum")
    fig.colorbar(im, ax=ax_spec, label="superposition defect (mass fraction)")

    # A few representative separation curves.
    n = len(separations)
    picks = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    for idx in picks:
        ax_curve.plot(
            taus, grid[idx], label="sep %.0f" % separations[idx], linewidth=1.5
        )
    ax_curve.set_xlabel("fine time  $\\tau$")
    ax_curve.set_ylabel("superposition defect (mass fraction)")
    ax_curve.set_title("Defect vs $\\tau$: 0 in free flight, rises at contact")
    ax_curve.legend(fontsize=8)
    ax_curve.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)


def main(argv=None):
    args = parse_args(argv)
    separations, taus, grid, rows = run(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "lenia_collision.csv"
    png_path = args.output_dir / "lenia_collision.png"
    write_csv(rows, csv_path)
    plot(separations, taus, grid, png_path)

    print("Wrote " + str(csv_path))
    print("Wrote " + str(png_path))


if __name__ == "__main__":
    main()
