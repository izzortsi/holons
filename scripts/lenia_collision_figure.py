#!/usr/bin/env python3
"""Presentation figure for the Lenia Orbium collision.

Top row: snapshots of the actual joint evolution evolve(A+B) at three times --
approaching, collision (defect peak), and after. Bottom: the superposition
defect delta(tau) = ||evolve(A+B) - (evolve(A)+evolve(B))|| / mass, which is
~0 while the two solitons are independent and spikes at contact: structurally
emergent throughout, temporally emergent at the collision.

This is the single-collision companion to the (separation, tau) spectrum in
lenia_collision.py; both use the same superposition defect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the project importable without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holons.lenia import ORBIUM, kernel_fft, step, total_mass
from holons.lenia_collision import orbium_velocity, setup_collision


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Snapshot + defect figure for one Lenia Orbium collision.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--grid-size", type=int, default=128)
    p.add_argument("--separation", type=float, default=55.0)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--contact-threshold", type=float, default=0.05)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "lenia_collision_figure.png")
    return p.parse_args(argv)


def run(args):
    velocity = orbium_velocity(ORBIUM, args.grid_size)
    spectrum = kernel_fft(ORBIUM, args.grid_size)
    field_a, field_b = setup_collision(args.grid_size, args.separation, ORBIUM, velocity)
    mass0 = total_mass(field_a) + total_mass(field_b)

    together = field_a + field_b
    a_alone = field_a.copy()
    b_alone = field_b.copy()

    frames = []
    defects = []
    for _ in range(args.steps + 1):
        frames.append(together.copy())
        # Same superposition defect as lenia_collision.superposition_defect_trajectory.
        defects.append(float(np.abs(together - (a_alone + b_alone)).sum() / mass0))
        together = step(together, spectrum, ORBIUM.mu, ORBIUM.sigma, ORBIUM.dt)
        a_alone = step(a_alone, spectrum, ORBIUM.mu, ORBIUM.sigma, ORBIUM.dt)
        b_alone = step(b_alone, spectrum, ORBIUM.mu, ORBIUM.sigma, ORBIUM.dt)

    defects = np.array(defects)
    peak = int(np.argmax(defects))
    over = defects > args.contact_threshold
    contact = int(np.argmax(over)) if over.any() else -1
    return frames, defects, peak, contact


def plot(frames, defects, peak, contact, steps, output):
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(2, 3, height_ratios=[1, 1], hspace=0.28, wspace=0.2)

    snap_times = [0, peak, steps]
    titles = [
        "t=0 (approaching)",
        "t=%d (collision)" % peak,
        "t=%d (after)" % steps,
    ]
    for i, (t, title) in enumerate(zip(snap_times, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames[t], cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Lenia collision: two Orbia head-on (continuous temporal-emergence spike)",
        fontsize=13,
        y=0.98,
    )

    ax_d = fig.add_subplot(gs[1, :])
    ax_d.plot(np.arange(len(defects)), defects, color="crimson", lw=2)
    if contact >= 0:
        ax_d.axvline(contact, color="gray", ls="--", lw=1)
        ax_d.text(
            contact + 2, defects.max() * 0.85, "contact t=%d" % contact,
            color="gray", fontsize=9,
        )
    ax_d.set_xlabel(r"steps $\tau$")
    ax_d.set_ylabel(r"temporal defect $\delta$ (field $L_1$, mass-normalised)")
    ax_d.set_title(
        "$\\delta \\approx 0$ while the solitons are independent, spikes at contact "
        "and stays nonzero after\n"
        "(structurally emergent throughout; temporally emergent at the collision)",
        fontsize=10.5,
    )
    ax_d.grid(alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")


def main(argv=None):
    args = parse_args(argv)
    frames, defects, peak, contact = run(args)

    plot(frames, defects, peak, contact, args.steps, args.output)

    print("Wrote " + str(args.output))
    print(
        "pre-contact mean delta (first 20) = %.4f; peak = %.3f at t=%d; contact t=%s"
        % (
            defects[:20].mean(),
            defects.max(),
            peak,
            "none" if contact < 0 else str(contact),
        )
    )
    print("final mass = %.1f (0 = mutual annihilation)" % total_mass(frames[-1]))


if __name__ == "__main__":
    main()
