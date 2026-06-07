#!/usr/bin/env python3
"""Experiment 1: glider-glider collision defect spectrum.

Sweeps (separation, tau) and writes a CSV with one row per measurement.
Each row records the temporal-emergence defect delta^t_tau for two
head-on gliders (SE + NW) initialised at the given diagonal separation,
after tau fine steps.

tau is constrained to be a positive multiple of the glider period (4).
Separation is the literal gap, in empty cells, between the SE glider's
3x3 bounding box and the NW glider's 3x3 bounding box along each axis
at t = 0. The two gliders are placed so they head directly at each
other along the main diagonal.

The catalogued velocities are SE -> (+1, +1) and NW -> (-1, -1).
Any other connected component that appears (post-collision debris:
blocks, blinkers, escaping spaceships, etc.) is treated as stationary
under the coarse dynamics, which is the section 7 default. Those
components show up in the defect as long as they differ from the
two-translated-gliders prediction.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make the project importable without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holons.defect import defect
from holons.geometry import recommended_grid_size, setup_collision
from holons.patterns import (
    GLIDER_NW_PHASE0,
    GLIDER_NW_VELOCITY,
    GLIDER_PERIOD,
    GLIDER_SE_PHASE0,
    GLIDER_SE_VELOCITY,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Glider collision defect spectrum experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="path to write the CSV result file",
    )
    p.add_argument(
        "--grid-size",
        required=True,
        type=int,
        help="size N of the (N x N) square Life grid",
    )
    p.add_argument(
        "--tau-max",
        required=True,
        type=int,
        help=f"maximum tau in fine steps; must be a multiple of {GLIDER_PERIOD}",
    )
    p.add_argument(
        "--separation-min",
        required=True,
        type=int,
        help="minimum separation in empty cells (inclusive)",
    )
    p.add_argument(
        "--separation-max",
        required=True,
        type=int,
        help="maximum separation in empty cells (inclusive)",
    )
    p.add_argument(
        "--tau-step",
        type=int,
        default=GLIDER_PERIOD,
        help=f"tau increment in fine steps; must be a multiple of {GLIDER_PERIOD}",
    )
    p.add_argument(
        "--separation-step",
        type=int,
        default=1,
        help="separation increment in cells",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress output to stderr",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.tau_max <= 0 or args.tau_max % GLIDER_PERIOD != 0:
        print(
            f"error: --tau-max must be a positive multiple of "
            f"{GLIDER_PERIOD}, got {args.tau_max}",
            file=sys.stderr,
        )
        return 2
    if args.tau_step <= 0 or args.tau_step % GLIDER_PERIOD != 0:
        print(
            f"error: --tau-step must be a positive multiple of "
            f"{GLIDER_PERIOD}, got {args.tau_step}",
            file=sys.stderr,
        )
        return 2
    if args.separation_min < 1:
        print(
            f"error: --separation-min must be >= 1, got {args.separation_min}",
            file=sys.stderr,
        )
        return 2
    if args.separation_max < args.separation_min:
        print(
            f"error: --separation-max ({args.separation_max}) must be >= "
            f"--separation-min ({args.separation_min})",
            file=sys.stderr,
        )
        return 2
    if args.separation_step <= 0:
        print(
            f"error: --separation-step must be positive, "
            f"got {args.separation_step}",
            file=sys.stderr,
        )
        return 2

    needed = recommended_grid_size(args.separation_max, args.tau_max)
    if args.grid_size < needed:
        print(
            f"warning: --grid-size {args.grid_size} is below the "
            f"recommended {needed} for separation_max="
            f"{args.separation_max} tau_max={args.tau_max}; "
            f"late-tau measurements may include boundary artefacts",
            file=sys.stderr,
        )

    velocities = {
        GLIDER_SE_PHASE0: GLIDER_SE_VELOCITY,
        GLIDER_NW_PHASE0: GLIDER_NW_VELOCITY,
    }

    separations = list(
        range(args.separation_min, args.separation_max + 1, args.separation_step)
    )
    taus = list(range(args.tau_step, args.tau_max + 1, args.tau_step))
    total = len(separations) * len(taus)

    if total == 0:
        print("error: empty sweep (no valid (separation, tau) pairs)", file=sys.stderr)
        return 2

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    progress_every = max(1, total // 20)

    with args.output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["separation", "tau", "defect"])
        i = 0
        for separation in separations:
            grid = setup_collision(args.grid_size, separation)
            for tau in taus:
                d = defect(grid, tau, velocities, GLIDER_PERIOD)
                writer.writerow([separation, tau, d])
                f.flush()
                i += 1
                if not args.quiet and i % progress_every == 0:
                    print(
                        f"  [{i}/{total}] separation={separation} "
                        f"tau={tau} defect={d}",
                        file=sys.stderr,
                    )

    if not args.quiet:
        print(f"\nWrote {total} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
