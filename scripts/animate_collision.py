#!/usr/bin/env python3
"""Animate a GoL collision synchronised with the defect heatmap.

For a chosen separation, evolves two head-on gliders fine-step by fine-step
and renders, on each frame:

- left:  the GoL state (auto-cropped to the action region)
- right: the defect heatmap with a red crosshair at
         (current tau, chosen separation)

The defect is measured only at multiples of the glider period (4); on
intermediate steps the title shows the last measured value. Output is a
GIF by default; pass an output path ending in `.mp4` to produce an MP4
instead (requires `ffmpeg` on PATH).

Typical use:

    python3 scripts/animate_collision.py \\
        --csv results/collision_spectrum.csv \\
        --separation 8 \\
        --grid-size 80 \\
        --tau-max 80 \\
        --output results/collision_separation_08.gif
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make the project importable without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from holons.geometry import MARGIN, setup_collision
from holons.life import step
from holons.patterns import GLIDER_PERIOD


def load_spectrum(path: Path):
    """Load the spectrum CSV into a `(separations, taus, H, lookup)` tuple.

    `H` is a 2D array shaped `(n_separations, n_taus)` of defect values
    (NaN for missing cells), suitable for `imshow`. `lookup` is a dict
    `(separation, tau) -> defect` for point queries.
    """
    rows: list[tuple[int, int, int]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                (
                    int(row["separation"]),
                    int(row["tau"]),
                    int(row["defect"]),
                )
            )
    if not rows:
        raise ValueError(f"no rows in {path}")
    separations = sorted({s for s, _, _ in rows})
    taus = sorted({t for _, t, _ in rows})
    sep_idx = {s: i for i, s in enumerate(separations)}
    tau_idx = {t: j for j, t in enumerate(taus)}
    H = np.full((len(separations), len(taus)), np.nan)
    lookup: dict[tuple[int, int], int] = {}
    for s, t, d in rows:
        H[sep_idx[s], tau_idx[t]] = d
        lookup[(s, t)] = d
    return separations, taus, H, lookup


def crop_window(
    grid_size: int, separation: int, tau_max: int
) -> tuple[int, int, int, int]:
    """Return `(r0, r1, c0, c1)` slice bounds for the action region.

    The window is the union of the gliders' initial bounding boxes and
    their motion envelopes over `tau_max` fine steps, centred on the
    midpoint of the initial configuration, clipped to the grid.
    """
    periods_max = max(1, tau_max // GLIDER_PERIOD)
    se_center = MARGIN + 1
    nw_center = MARGIN + 3 + separation + 1
    center = (se_center + nw_center) // 2
    half_extent = max(periods_max, separation) + 5
    r0 = max(0, center - half_extent)
    r1 = min(grid_size, center + half_extent + 1)
    return r0, r1, r0, r1  # square window


def heatmap_extent(separations: list[int], taus: list[int]) -> list[float]:
    """Return `[left, right, bottom, top]` for the heatmap `imshow`.

    Bottom > top (data coords) because we use `origin='upper'` so that
    small separation appears at the top of the figure, matching the
    existing static figure.
    """
    tau_step = taus[1] - taus[0] if len(taus) > 1 else 1
    sep_step = (
        separations[1] - separations[0] if len(separations) > 1 else 1
    )
    return [
        taus[0] - tau_step / 2,
        taus[-1] + tau_step / 2,
        separations[-1] + sep_step / 2,
        separations[0] - sep_step / 2,
    ]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Animate a GoL glider collision synchronised with the "
            "defect spectrum heatmap."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="path to the collision spectrum CSV",
    )
    p.add_argument(
        "--separation",
        required=True,
        type=int,
        help="separation in cells to animate; must be present in the CSV",
    )
    p.add_argument(
        "--grid-size",
        required=True,
        type=int,
        help="GoL grid size N; must match the experiment that produced the CSV",
    )
    p.add_argument(
        "--tau-max",
        required=True,
        type=int,
        help="last fine step to animate (number of frames = tau_max + 1)",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output animation path (.gif or .mp4)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=6,
        help="animation frames per second",
    )
    p.add_argument(
        "--no-crop",
        action="store_true",
        help="show the full grid instead of auto-cropping to the action region",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.tau_max <= 0:
        print(
            f"error: --tau-max must be positive, got {args.tau_max}",
            file=sys.stderr,
        )
        return 2
    if args.fps <= 0:
        print(f"error: --fps must be positive, got {args.fps}", file=sys.stderr)
        return 2

    separations, taus, H, lookup = load_spectrum(args.csv)
    if args.separation not in separations:
        print(
            f"error: separation {args.separation} not in CSV; "
            f"available range {separations[0]}..{separations[-1]}",
            file=sys.stderr,
        )
        return 2

    grid = setup_collision(args.grid_size, args.separation)

    if args.no_crop:
        r0, r1, c0, c1 = 0, args.grid_size, 0, args.grid_size
    else:
        r0, r1, c0, c1 = crop_window(
            args.grid_size, args.separation, args.tau_max
        )

    fig, (ax_grid, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))

    im_grid = ax_grid.imshow(
        grid[r0:r1, c0:c1],
        cmap="binary",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_grid.set_aspect("equal")
    ax_grid.set_title(f"GoL state (separation = {args.separation})")
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])

    extent = heatmap_extent(separations, taus)
    im_heat = ax_heat.imshow(
        H,
        aspect="auto",
        cmap="viridis",
        extent=extent,
        origin="upper",
    )
    ax_heat.set_xlabel("tau")
    ax_heat.set_ylabel("separation")
    ax_heat.set_title("Defect spectrum  (red crosshair = current frame)")
    cbar = plt.colorbar(im_heat, ax=ax_heat)
    cbar.set_label("defect")

    ax_heat.axhline(args.separation, color="red", linewidth=1.2, alpha=0.6)
    vline = ax_heat.axvline(0, color="red", linewidth=1.8, alpha=0.9)

    title = fig.suptitle("tau = 0     defect = 0", fontsize=14)

    state = {"grid": grid, "tau": 0}

    def update(frame: int):
        if frame > 0:
            state["grid"] = step(state["grid"])
            state["tau"] += 1
        im_grid.set_data(state["grid"][r0:r1, c0:c1])
        vline.set_xdata([state["tau"], state["tau"]])

        t = state["tau"]
        if t == 0:
            d_text = "0  (initial state)"
        elif t % GLIDER_PERIOD == 0:
            d = lookup.get((args.separation, t))
            d_text = str(d) if d is not None else "(not in CSV)"
        else:
            t_last = (t // GLIDER_PERIOD) * GLIDER_PERIOD
            d_last = lookup.get((args.separation, t_last))
            d_text = (
                f"(off-period; last measured at tau={t_last}: {d_last})"
            )
        title.set_text(f"tau = {t}     defect = {d_text}")
        return im_grid, vline, title

    n_frames = args.tau_max + 1
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / args.fps,
        blit=False,
        repeat=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()
    if suffix == ".gif":
        writer = PillowWriter(fps=args.fps)
        anim.save(args.output, writer=writer)
    elif suffix == ".mp4":
        try:
            from matplotlib.animation import FFMpegWriter

            writer = FFMpegWriter(fps=args.fps)
            anim.save(args.output, writer=writer)
        except Exception as exc:
            print(
                f"error: MP4 output requires ffmpeg on PATH; "
                f"install it or use a .gif output. ({exc})",
                file=sys.stderr,
            )
            return 2
    else:
        print(
            f"error: --output must end in .gif or .mp4, got '{suffix}'",
            file=sys.stderr,
        )
        return 2
    plt.close(fig)
    print(f"Wrote animation to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
