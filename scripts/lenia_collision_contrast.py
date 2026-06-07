#!/usr/bin/env python3
"""Contrast figure: annihilation vs surviving-product Lenia collisions.

Two head-on Orbium collisions with different outcomes -- one separation that
mutually annihilates, one that leaves a persistent product -- shown with their
field snapshots and their superposition defect curves overlaid. Both defects
are 0 before contact and spike at the collision (the dissociation), but they
settle to DIFFERENT plateaus: the plateau encodes the reaction product, the
continuous-substrate counterpart of the Game-of-Life parity plateau (defect 2
for annihilation vs 3 for a surviving block).
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

OUTCOME_COLOUR = {"annihilation": "crimson", "survivor": "teal"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Annihilation-vs-survivor Lenia collision contrast figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--grid-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--annihilation-sep", type=float, default=55.0)
    p.add_argument("--survivor-sep", type=float, default=38.0)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "lenia_collision_contrast.png")
    return p.parse_args(argv)


def run_collision(separation, grid_size, steps, spectrum, velocity):
    field_a, field_b = setup_collision(grid_size, separation, ORBIUM, velocity)
    mass0 = total_mass(field_a) + total_mass(field_b)
    together = field_a + field_b
    a_alone = field_a.copy()
    b_alone = field_b.copy()

    frames = []
    defects = []
    for _ in range(steps + 1):
        frames.append(together.copy())
        defects.append(float(np.abs(together - (a_alone + b_alone)).sum() / mass0))
        together = step(together, spectrum, ORBIUM.mu, ORBIUM.sigma, ORBIUM.dt)
        a_alone = step(a_alone, spectrum, ORBIUM.mu, ORBIUM.sigma, ORBIUM.dt)
        b_alone = step(b_alone, spectrum, ORBIUM.mu, ORBIUM.sigma, ORBIUM.dt)

    defects = np.array(defects)
    peak = int(np.argmax(defects))
    final_ratio = total_mass(frames[-1]) / mass0
    return frames, defects, peak, final_ratio


def plot(outcomes, steps, output):
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13, 11))
    gs = GridSpec(3, 3, height_ratios=[1, 1, 1.15], hspace=0.3, wspace=0.1)

    phases = ["approaching", "collision", "after"]
    for row, oc in enumerate(outcomes):
        times = [0, oc["peak"], steps]
        for col, (t, phase) in enumerate(zip(times, phases)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(oc["frames"][t], cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("%s  (t=%d)" % (phase, t), fontsize=9)
            if col == 0:
                ax.set_ylabel(
                    "%s\nsep %d" % (oc["label"], int(oc["separation"])),
                    fontsize=11, color=OUTCOME_COLOUR[oc["label"]],
                )

    ax_d = fig.add_subplot(gs[2, :])
    for oc in outcomes:
        ax_d.plot(
            np.arange(len(oc["defects"])), oc["defects"],
            color=OUTCOME_COLOUR[oc["label"]], lw=2,
            label="sep %d: %s (final mass %.2f x)"
            % (int(oc["separation"]), oc["label"], oc["final_ratio"]),
        )
    ax_d.set_xlabel(r"steps $\tau$")
    ax_d.set_ylabel(r"superposition defect $\delta$ (mass-normalised)")
    ax_d.set_title(
        "Both defects are 0 before contact and spike at the collision; "
        "the plateau encodes the product\n"
        "(the continuous counterpart of the Game-of-Life parity plateau)",
        fontsize=10.5,
    )
    ax_d.legend(fontsize=9)
    ax_d.grid(alpha=0.3)

    fig.suptitle(
        "Lenia collisions: annihilation vs surviving product",
        fontsize=14, y=0.995,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")


def main(argv=None):
    args = parse_args(argv)
    velocity = orbium_velocity(ORBIUM, args.grid_size)
    spectrum = kernel_fft(ORBIUM, args.grid_size)

    outcomes = []
    for separation, label in [
        (args.annihilation_sep, "annihilation"),
        (args.survivor_sep, "survivor"),
    ]:
        frames, defects, peak, final_ratio = run_collision(
            separation, args.grid_size, args.steps, spectrum, velocity
        )
        outcomes.append(
            {
                "separation": separation,
                "label": label,
                "frames": frames,
                "defects": defects,
                "peak": peak,
                "final_ratio": final_ratio,
            }
        )
        print(
            "sep %d (%s): peak delta %.3f at t=%d, final mass %.2f x"
            % (int(separation), label, defects.max(), peak, final_ratio)
        )

    plot(outcomes, args.steps, args.output)
    print("Wrote " + str(args.output))


if __name__ == "__main__":
    main()
