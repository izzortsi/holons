#!/usr/bin/env python3
"""Small-multiples dissociation figure across separations.

Renders the structural-vs-temporal dissociation (see dissociation.py) for
several collision separations side by side, one column per separation, so the
separation-parity structure of the reaction is visible alongside the
dissociation:

  - odd separations annihilate: the temporal defect plateaus at delta = 2
    and the structural curve empties (no surviving object);
  - even separations leave a still-life product: delta = 3 and the structural
    breadth holds at |L| >= 2;
  - the contact time grows with separation.

The measurement is reused verbatim from dissociation.py (single source of
truth). The same given-model-defect caveat applies: a positive temporal
defect means the gliders interacted, not that the product is intrinsically
irreducible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the scripts directory so the sibling measurement module is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dissociation import RESULTS_DIR, first_contact_tau, measure
from holons.structural import STRUCTURAL_FLOOR


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Small-multiples structural-vs-temporal dissociation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--separations",
        type=int,
        nargs="+",
        default=[5, 6, 7, 8, 9, 10],
        help="collision separations to render, one column each",
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
        "--output-path",
        type=Path,
        default=RESULTS_DIR / "dissociation_grid.png",
        help="path for the output PNG",
    )
    return p.parse_args(argv)


def plot_grid(per_sep, separations, png_path):
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt

    ncols = len(separations)
    fig, axes = plt.subplots(
        2, ncols, figsize=(3.0 * ncols, 6.0), sharex="col", sharey="row",
        squeeze=False,
    )

    for j, sep in enumerate(separations):
        rows = per_sep[sep]
        taus = [row["tau"] for row in rows]
        defects = [row["defect"] for row in rows]
        min_support = [row["min_support"] for row in rows]
        contact = first_contact_tau(rows)

        ax_t = axes[0][j]
        ax_s = axes[1][j]

        # Top row: temporal defect.
        ax_t.step(taus, defects, where="mid", color="C3", marker="o", markersize=3)
        ax_t.fill_between(taus, defects, step="mid", color="C3", alpha=0.15)
        ax_t.set_title("sep = " + str(sep), fontsize=10)
        ax_t.grid(True, alpha=0.3)
        ax_t.set_ylim(bottom=0)

        # Bottom row: structural support breadth (verdict-relevant min |L|).
        ax_s.plot(taus, min_support, color="C0", marker="o", markersize=3)
        ax_s.axhline(STRUCTURAL_FLOOR, color="gray", linestyle=":", linewidth=1.0)
        ax_s.grid(True, alpha=0.3)
        ax_s.set_ylim(bottom=0)
        ax_s.set_xlabel("$\\tau$", fontsize=9)

        # Mark contact on both panels of this column.
        if contact is not None:
            for ax in (ax_t, ax_s):
                ax.axvline(
                    contact, color="black", linestyle="--", linewidth=0.9, alpha=0.7
                )

    axes[0][0].set_ylabel("temporal defect  $\\delta^t_\\tau$")
    axes[1][0].set_ylabel("min support  $|L|$")
    fig.suptitle(
        "Structural vs temporal emergence across separations "
        "(odd: annihilation; even: surviving block)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=150)


def main(argv=None):
    args = parse_args(argv)

    per_sep = {}
    for sep in args.separations:
        per_sep[sep] = measure(sep, args.tau_max, args.grid_size)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_grid(per_sep, args.separations, args.output_path)

    # Print a compact parity summary so the figure is interpretable from stdout.
    print("Wrote " + str(args.output_path))
    print("sep  contact  plateau_delta  final_objects  final_min|L|")
    for sep in args.separations:
        rows = per_sep[sep]
        contact = first_contact_tau(rows)
        last = rows[-1]
        contact_str = "none" if contact is None else str(contact)
        print(
            format(sep, "3d")
            + "  " + format(contact_str, ">7s")
            + "  " + format(last["defect"], "13d")
            + "  " + format(last["n_objects"], "13d")
            + "  " + format(last["min_support"], "12d")
        )


if __name__ == "__main__":
    main()
