#!/usr/bin/env python3
"""Experiment 3: Lenia stability boundary via the temporal defect.

Sweeps the growth parameters (mu, sigma) around the Orbium's stable point and,
for each, measures the temporal-emergence defect of the centroid coarse model
(deviation of the organism's centroid path from constant-velocity motion) and
the survival (log mass ratio). The result is two heatmaps over (mu, sigma):

  - centroid defect: ~0 on the glider island (coherent translation), rising at
    the stability boundary as the organism deforms or dissolves;
  - log mass ratio: 0 where mass is conserved, negative where the organism
    dissolves, positive where it grows / explodes.

The Orbium base point (mu=0.15, sigma=0.014) is marked on both. Writes a CSV
(one row per grid point) and the figure.
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
from holons.lenia_stability import measure_stability
from holons.orbium import place


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Lenia stability boundary via the centroid temporal defect.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mu-min", type=float, default=0.10)
    p.add_argument("--mu-max", type=float, default=0.20)
    p.add_argument("--sigma-min", type=float, default=0.006)
    p.add_argument("--sigma-max", type=float, default=0.022)
    p.add_argument("--resolution", type=int, default=25, help="grid points per axis")
    p.add_argument("--grid-size", type=int, default=64, help="square Lenia field side")
    p.add_argument("--settle", type=int, default=50, help="relaxation steps before tracking")
    p.add_argument("--window", type=int, default=150, help="centroid-tracking steps")
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args(argv)


def run(args):
    mus = np.linspace(args.mu_min, args.mu_max, args.resolution)
    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.resolution)

    # The kernel depends only on the fixed Orbium kernel shape and grid, not on
    # (mu, sigma), so precompute it and the seed once.
    spectrum = kernel_fft(ORBIUM, args.grid_size)
    seed = place(args.grid_size)

    defect = np.empty((args.resolution, args.resolution))
    log_mass = np.empty((args.resolution, args.resolution))
    rows = []
    total = args.resolution * args.resolution
    done = 0
    for i, sigma in enumerate(sigmas):
        for j, mu in enumerate(mus):
            result = measure_stability(
                seed, spectrum, ORBIUM.dt, mu, sigma, args.settle, args.window
            )
            defect[i, j] = result.centroid_defect
            log_mass[i, j] = result.log_mass_ratio
            rows.append(
                {
                    "mu": mu,
                    "sigma": sigma,
                    "centroid_defect": result.centroid_defect,
                    "log_mass_ratio": result.log_mass_ratio,
                }
            )
            done += 1
        print("  swept sigma row %d/%d" % (i + 1, args.resolution), file=sys.stderr)
    return mus, sigmas, defect, log_mass, rows


def write_csv(rows, csv_path):
    fields = ["mu", "sigma", "centroid_defect", "log_mass_ratio"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot(mus, sigmas, defect, log_mass, png_path):
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt

    extent = [mus.min(), mus.max(), sigmas.min(), sigmas.max()]
    fig, (ax_d, ax_m) = plt.subplots(1, 2, figsize=(14, 6))

    # The centroid defect is only meaningful where an organism persists: a
    # dissolved (near-uniform) field has a trivially uniform centroid, so its
    # low defect is degenerate with a coherent glider's. Gate the defect by
    # survival (mass conserved within ~2x) and grey out the rest.
    survives = np.abs(log_mass) <= 0.3
    defect_clipped = np.clip(defect, 0.0, 2.0)
    defect_masked = np.ma.masked_where(~survives, defect_clipped)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#cccccc")
    im_d = ax_d.imshow(
        defect_masked, origin="lower", extent=extent, aspect="auto", cmap=cmap
    )
    ax_d.set_title("Temporal defect within the survival corridor (grey = no organism)")
    ax_d.set_xlabel("growth centre  $\\mu$")
    ax_d.set_ylabel("growth width  $\\sigma$")
    fig.colorbar(im_d, ax=ax_d, label="centroid defect (clipped at 2)")

    # Survival: diverging around 0 (mass conserved).
    span = float(np.nanmax(np.abs(log_mass))) or 1.0
    im_m = ax_m.imshow(
        log_mass, origin="lower", extent=extent, aspect="auto",
        cmap="RdBu_r", vmin=-span, vmax=span,
    )
    ax_m.set_title("Survival: $\\log_{10}$(final mass / initial mass)")
    ax_m.set_xlabel("growth centre  $\\mu$")
    ax_m.set_ylabel("growth width  $\\sigma$")
    fig.colorbar(im_m, ax=ax_m, label="log mass ratio (0 = conserved)")

    # Mark the Orbium base point on both panels.
    for ax in (ax_d, ax_m):
        ax.plot(ORBIUM.mu, ORBIUM.sigma, marker="*", color="white",
                markersize=15, markeredgecolor="black")
        ax.annotate("Orbium", (ORBIUM.mu, ORBIUM.sigma),
                    textcoords="offset points", xytext=(8, 6),
                    color="white", fontsize=9)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)


def main(argv=None):
    args = parse_args(argv)
    mus, sigmas, defect, log_mass, rows = run(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "lenia_stability.csv"
    png_path = args.output_dir / "lenia_stability.png"
    write_csv(rows, csv_path)
    plot(mus, sigmas, defect, log_mass, png_path)

    # Report the defect at the Orbium base point (nearest grid node).
    i = int(np.argmin(np.abs(sigmas - ORBIUM.sigma)))
    j = int(np.argmin(np.abs(mus - ORBIUM.mu)))
    print("Wrote " + str(csv_path))
    print("Wrote " + str(png_path))
    print(
        "Orbium base point (mu=%.3f, sigma=%.3f): defect=%.3f, log_mass=%.3f"
        % (mus[j], sigmas[i], defect[i, j], log_mass[i, j])
    )
    island = int((defect < 0.5).sum())
    print("Low-defect points (defect < 0.5): %d of %d" % (island, defect.size))


if __name__ == "__main__":
    main()
