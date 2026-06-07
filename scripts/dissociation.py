#!/usr/bin/env python3
"""Dissociation figure: structural vs temporal emergence over time.

For a single glider collision (fixed separation) this measures both
emergence axes at each measurement time tau:

  temporal:   delta^t_tau(beta)        -- defect.py  (the given-model defect)
  structural: support breadth |L|      -- structural.py

and renders a two-panel figure sharing the tau axis. The structural verdict
(|L| >= 2) holds throughout, while the temporal defect is zero during free
flight and positive only from the collision onward: the Definition 13 /
Definition 18 dissociation rendered as data.

CAVEAT (stated plainly): the temporal defect here is the GIVEN-MODEL defect
of defect.py -- the failure of the independent-translation coarse model. A
positive value means the two gliders interacted; it is an interaction
detector, not evidence of intrinsic irreducibility. This figure turns the
paper's analytic Def 13 / Def 18 distinction into a picture; it does not add
evidence beyond that. The matching-metric and closure experiments are the
steps that make the temporal axis carry information the separation does not
already give for free.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holons.defect import defect
from holons.geometry import recommended_grid_size, setup_collision
from holons.life import evolve
from holons.patterns import (
    GLIDER_NW_PHASE0,
    GLIDER_NW_VELOCITY,
    GLIDER_PERIOD,
    GLIDER_SE_PHASE0,
    GLIDER_SE_VELOCITY,
)
from holons.structural import STRUCTURAL_FLOOR, support_breadths


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# The two catalogued gliders of the collision and their per-period velocities.
VELOCITIES = {
    GLIDER_SE_PHASE0: GLIDER_SE_VELOCITY,
    GLIDER_NW_PHASE0: GLIDER_NW_VELOCITY,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Structural-vs-temporal emergence dissociation figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--separation",
        type=int,
        default=8,
        help="vacuum cells between the two glider bounding boxes at t=0",
    )
    p.add_argument(
        "--tau-max",
        type=int,
        default=80,
        help="largest measurement time tau (a multiple of the glider period)",
    )
    p.add_argument(
        "--grid-size",
        type=int,
        default=0,
        help="square grid side; 0 means use the recommended safe size",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="directory for the .csv and .png outputs",
    )
    return p.parse_args(argv)


def measure(separation, tau_max, grid_size):
    """Return per-tau measurement rows for one collision.

    Each row is a dict with the measurement time, the temporal defect, the
    object count, and the min/mean/all support breadths at that time.
    """
    if grid_size <= 0:
        grid_size = recommended_grid_size(separation, tau_max)
    beta = setup_collision(grid_size, separation)

    rows = []
    for tau in range(GLIDER_PERIOD, tau_max + 1, GLIDER_PERIOD):
        temporal = defect(beta, tau, VELOCITIES, GLIDER_PERIOD)
        breadths = support_breadths(evolve(beta, tau))
        n_objects = len(breadths)
        min_support = min(breadths) if breadths else 0
        mean_support = float(np.mean(breadths)) if breadths else 0.0
        rows.append(
            {
                "tau": tau,
                "defect": temporal,
                "n_objects": n_objects,
                "min_support": min_support,
                "mean_support": mean_support,
                "supports": ";".join(str(b) for b in sorted(breadths)),
            }
        )
    return rows


def write_csv(rows, csv_path):
    fields = ["tau", "defect", "n_objects", "min_support", "mean_support", "supports"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def first_contact_tau(rows):
    """Return the smallest tau at which the temporal defect is positive.

    Returns None when no contact occurs within the measured window.
    """
    for row in rows:
        if row["defect"] > 0:
            return row["tau"]
    return None


def plot(rows, separation, png_path):
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt

    taus = [row["tau"] for row in rows]
    defects = [row["defect"] for row in rows]
    min_support = [row["min_support"] for row in rows]
    mean_support = [row["mean_support"] for row in rows]
    contact = first_contact_tau(rows)

    fig, (ax_t, ax_s) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    # Top panel: temporal emergence (the defect).
    ax_t.step(taus, defects, where="mid", color="C3", marker="o", markersize=4)
    ax_t.fill_between(taus, defects, step="mid", color="C3", alpha=0.15)
    ax_t.set_ylabel("temporal defect  $\\delta^t_\\tau$")
    ax_t.set_title(
        "Structural vs temporal emergence  (glider collision, separation = "
        + str(separation)
        + ")"
    )
    ax_t.grid(True, alpha=0.3)
    ax_t.set_ylim(bottom=0)
    ax_t.text(
        0.02,
        0.92,
        "temporal verdict: 0 in free flight, > 0 at contact",
        transform=ax_t.transAxes,
        fontsize=8,
        va="top",
    )

    # Bottom panel: structural emergence (the support breadth).
    ax_s.plot(taus, min_support, color="C0", marker="o", markersize=4, label="min $|L|$")
    ax_s.plot(
        taus, mean_support, color="C0", marker=".", linestyle="--", alpha=0.6,
        label="mean $|L|$",
    )
    ax_s.axhline(
        STRUCTURAL_FLOOR, color="gray", linestyle=":", linewidth=1.2,
        label="structural floor $|L| = " + str(STRUCTURAL_FLOOR) + "$",
    )
    ax_s.set_ylabel("support breadth  $|L|$")
    ax_s.set_xlabel("fine time  $\\tau$")
    ax_s.grid(True, alpha=0.3)
    ax_s.set_ylim(bottom=0)
    ax_s.legend(fontsize=8, loc="upper right")
    ax_s.text(
        0.02,
        0.10,
        "structural verdict: persistent objects $|L| \\geq 2$; "
        "only the one-tick reaction splash dips to $|L| = 1$",
        transform=ax_s.transAxes,
        fontsize=8,
        va="bottom",
    )

    # Mark the collision onset on both panels.
    if contact is not None:
        for ax in (ax_t, ax_s):
            ax.axvline(contact, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax_t.text(
            contact,
            ax_t.get_ylim()[1] * 0.98,
            " contact $\\tau = " + str(contact) + "$",
            fontsize=8,
            va="top",
            ha="left",
        )

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)


def main(argv=None):
    args = parse_args(argv)
    rows = measure(args.separation, args.tau_max, args.grid_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "dissociation_sep" + format(args.separation, "02d")
    csv_path = args.output_dir / (stem + ".csv")
    png_path = args.output_dir / (stem + ".png")

    write_csv(rows, csv_path)
    plot(rows, args.separation, png_path)

    contact = first_contact_tau(rows)
    print("Wrote " + str(csv_path))
    print("Wrote " + str(png_path))
    if contact is None:
        print("No contact within tau <= " + str(args.tau_max))
    else:
        print("Contact onset at tau = " + str(contact))


if __name__ == "__main__":
    main()
