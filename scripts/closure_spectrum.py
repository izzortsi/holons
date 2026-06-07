#!/usr/bin/env python3
"""Experiment 2: closure defect across Life-like rules.

Measures the intrinsic closure defect (closure.measure_closure) for every
rule in the catalogue (rules.CATALOGUE) and asks whether it separates the
literature regime labels (trivial / ordered / complex / chaotic). Unlike the
glider collision (Experiment 1), this defect uses no object catalogue and no
hand-built coarse dynamics: it measures whether a coarse rule EXISTS under
the block coarse-graining.

Writes a CSV (one row per rule) and a two-panel figure: the conditional
entropy H(successor | neighbourhood) and the ambiguous-sample fraction, both
sorted ascending and coloured by regime.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make the project importable without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holons.closure import measure_closure
from holons.rules import CATALOGUE, format_rule


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Regime -> colour for the bars.
REGIME_COLOUR = {
    "trivial": "#9e9e9e",
    "ordered": "#4c9a2a",
    "complex": "#1f77b4",
    "chaotic": "#d62728",
}
REGIME_ORDER = ["trivial", "ordered", "complex", "chaotic"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Closure-defect spectrum across Life-like rules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--block", type=int, default=2, help="supercell side b")
    p.add_argument(
        "--time-steps", type=int, default=0,
        help="fine steps per coarse tick T; 0 means use the block size b",
    )
    p.add_argument("--n-soups", type=int, default=200, help="random soups per rule")
    p.add_argument(
        "--grid-blocks", type=int, default=48,
        help="supercells per side; fine grid is grid_blocks * block",
    )
    p.add_argument("--seed", type=int, default=12345, help="rng seed")
    p.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR,
        help="directory for the .csv and .png outputs",
    )
    return p.parse_args(argv)


def run(args):
    rows = []
    for rule in CATALOGUE:
        result = measure_closure(
            rule.birth,
            rule.survive,
            block=args.block,
            time_steps=args.time_steps,
            n_soups=args.n_soups,
            grid_blocks=args.grid_blocks,
            seed=args.seed,
        )
        rows.append(
            {
                "name": rule.name,
                "rule": format_rule(rule.birth, rule.survive),
                "regime": rule.regime,
                "cond_entropy": result.cond_entropy,
                "ambiguous_fraction": result.ambiguous_fraction,
                "n_samples": result.n_samples,
                "n_keys": result.n_keys,
            }
        )
    rows.sort(key=lambda r: r["cond_entropy"])
    return rows


def write_csv(rows, csv_path):
    fields = [
        "name", "rule", "regime", "cond_entropy",
        "ambiguous_fraction", "n_samples", "n_keys",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot(rows, png_path):
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    labels = [r["name"] + "\n" + r["rule"] for r in rows]
    entropy = [r["cond_entropy"] for r in rows]
    ambiguous = [r["ambiguous_fraction"] for r in rows]
    colours = [REGIME_COLOUR[r["regime"]] for r in rows]
    x = range(len(rows))

    fig, (ax_e, ax_a) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax_e.bar(x, entropy, color=colours)
    ax_e.set_ylabel("closure defect\n$H(\\mathrm{successor}\\mid\\mathrm{neighbourhood})$  (bits)")
    ax_e.set_title(
        "Closure defect across Life-like rules "
        "(0 = a coarse rule exists; sorted ascending)"
    )
    ax_e.grid(True, axis="y", alpha=0.3)

    ax_a.bar(x, ambiguous, color=colours)
    ax_a.set_ylabel("ambiguous-sample fraction")
    ax_a.grid(True, axis="y", alpha=0.3)
    ax_a.set_xticks(list(x))
    ax_a.set_xticklabels(labels, fontsize=8)

    legend_handles = [
        Patch(facecolor=REGIME_COLOUR[r], label=r) for r in REGIME_ORDER
    ]
    ax_e.legend(handles=legend_handles, title="regime (literature prior)", fontsize=9)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)


def main(argv=None):
    args = parse_args(argv)
    rows = run(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "closure_spectrum.csv"
    png_path = args.output_dir / "closure_spectrum.png"
    write_csv(rows, csv_path)
    plot(rows, png_path)

    print("Wrote " + str(csv_path))
    print("Wrote " + str(png_path))
    print()
    print(
        format("rule", "<18s") + format("regime", "<10s")
        + format("H(s|nbhd)", ">11s") + format("ambig_frac", ">12s")
        + format("n_keys", ">8s")
    )
    for row in rows:
        print(
            format(row["name"] + " " + row["rule"], "<18s")
            + format(row["regime"], "<10s")
            + format(row["cond_entropy"], ">11.4f")
            + format(row["ambiguous_fraction"], ">12.4f")
            + format(row["n_keys"], ">8d")
        )


if __name__ == "__main__":
    main()
