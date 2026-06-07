#!/usr/bin/env python3
"""Experiment 2b: closure defect vs relaxation (burn-in).

The first-cut closure spectrum (closure_spectrum.py) measured closure on raw
random soup, where every rule is transiently turbulent, so the regime labels
did not separate. Here we sweep a burn-in K: each soup is evolved K fine steps
before the closure measurement, so closure is sampled on the rule's relaxed
distribution. The hypothesis is that ordered rules freeze (their coarse update
becomes the identity, so the defect decays toward 0) while complex rules
plateau and chaotic rules stay saturated.

Writes a CSV (one row per rule x K) and a figure of defect vs K, one line per
rule, coloured by regime.
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

REGIME_COLOUR = {
    "trivial": "#9e9e9e",
    "ordered": "#4c9a2a",
    "complex": "#1f77b4",
    "chaotic": "#d62728",
}
REGIME_ORDER = ["trivial", "ordered", "complex", "chaotic"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Closure defect vs relaxation (burn-in) across rules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--burn-ins", type=int, nargs="+",
        default=[0, 2, 4, 8, 16, 32, 64, 100],
        help="burn-in step counts K to sweep",
    )
    p.add_argument("--block", type=int, default=2, help="supercell side b")
    p.add_argument("--n-soups", type=int, default=60, help="random soups per measurement")
    p.add_argument(
        "--grid-blocks", type=int, default=36,
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
        for k in args.burn_ins:
            result = measure_closure(
                rule.birth,
                rule.survive,
                block=args.block,
                burn_in=k,
                n_soups=args.n_soups,
                grid_blocks=args.grid_blocks,
                seed=args.seed,
            )
            rows.append(
                {
                    "name": rule.name,
                    "rule": format_rule(rule.birth, rule.survive),
                    "regime": rule.regime,
                    "burn_in": k,
                    "cond_entropy": result.cond_entropy,
                    "ambiguous_fraction": result.ambiguous_fraction,
                    "n_samples": result.n_samples,
                    "n_keys": result.n_keys,
                }
            )
    return rows


def write_csv(rows, csv_path):
    fields = [
        "name", "rule", "regime", "burn_in", "cond_entropy",
        "ambiguous_fraction", "n_samples", "n_keys",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot(rows, burn_ins, png_path):
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    by_rule = {}
    for row in rows:
        by_rule.setdefault(row["name"], []).append(row)

    fig, ax = plt.subplots(figsize=(11, 7))
    x_max = max(burn_ins)
    for name, rule_rows in by_rule.items():
        rule_rows.sort(key=lambda r: r["burn_in"])
        ks = [r["burn_in"] for r in rule_rows]
        ys = [r["cond_entropy"] for r in rule_rows]
        regime = rule_rows[0]["regime"]
        ax.plot(ks, ys, marker="o", markersize=4, color=REGIME_COLOUR[regime], alpha=0.85)
        # Label each line at its right end.
        ax.annotate(
            name,
            xy=(ks[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=REGIME_COLOUR[regime],
            va="center",
        )

    ax.set_xlabel("burn-in  $K$  (fine steps before measurement)")
    ax.set_ylabel("closure defect  $H(\\mathrm{successor}\\mid\\mathrm{neighbourhood})$  (bits)")
    ax.set_title("Closure defect vs relaxation: does the coarse rule emerge as rules settle?")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_max * 1.18)
    ax.set_ylim(bottom=0)

    legend_handles = [Patch(facecolor=REGIME_COLOUR[r], label=r) for r in REGIME_ORDER]
    ax.legend(handles=legend_handles, title="regime (literature prior)", fontsize=9)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)


def main(argv=None):
    args = parse_args(argv)
    rows = run(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "closure_relaxation.csv"
    png_path = args.output_dir / "closure_relaxation.png"
    write_csv(rows, csv_path)
    plot(rows, args.burn_ins, png_path)

    print("Wrote " + str(csv_path))
    print("Wrote " + str(png_path))
    print()
    # Compact table: defect at K=0 vs the largest K, per rule.
    k_lo = min(args.burn_ins)
    k_hi = max(args.burn_ins)
    print(
        format("rule", "<18s") + format("regime", "<10s")
        + format("H(K=%d)" % k_lo, ">10s") + format("H(K=%d)" % k_hi, ">10s")
        + format("change", ">10s")
    )
    by_rule = {}
    for row in rows:
        by_rule.setdefault(row["name"], {})[row["burn_in"]] = row
    for rule in CATALOGUE:
        lo = by_rule[rule.name][k_lo]["cond_entropy"]
        hi = by_rule[rule.name][k_hi]["cond_entropy"]
        print(
            format(rule.name + " " + format_rule(rule.birth, rule.survive), "<18s")
            + format(rule.regime, "<10s")
            + format(lo, ">10.4f")
            + format(hi, ">10.4f")
            + format(hi - lo, ">+10.4f")
        )


if __name__ == "__main__":
    main()
