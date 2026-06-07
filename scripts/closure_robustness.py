#!/usr/bin/env python3
"""Experiment 2c: robustness of the closure regime separation.

The relaxation result (closure_relaxation.py) used one block size, one
projection, one seed. This checks whether the regime ordering (trivial <
ordered < complex < chaotic in closure defect) survives changes to those
choices. For each configuration -- block size, projection/threshold -- and
each of several seeds, it measures the relaxed closure defect (fixed burn-in)
for every rule and scores the separation two ways:

  - Spearman rho between regime rank (trivial=0..chaotic=3) and defect: how
    well the defect orders the regimes (1.0 = perfect monotone separation);
  - margin = min(chaotic defect) - max(ordered defect): positive means the
    ordered band sits entirely below the chaotic band.

Writes a per-(config, seed, rule) CSV and a two-panel figure: Spearman rho
per configuration (with seed spread) and the defect-by-regime spread for the
baseline configuration.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Make the project importable without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holons.closure import measure_closure
from holons.coarse_block import project_subsample
from holons.rules import CATALOGUE, format_rule


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

REGIME_RANK = {"trivial": 0, "ordered": 1, "complex": 2, "chaotic": 3}
REGIME_COLOUR = {
    "trivial": "#9e9e9e",
    "ordered": "#4c9a2a",
    "complex": "#1f77b4",
    "chaotic": "#d62728",
}
REGIME_ORDER = ["trivial", "ordered", "complex", "chaotic"]


def configurations():
    """Return the list of (label, measure_closure kwargs) to compare.

    Spans two block sizes, three count thresholds (OR / majority / AND) and a
    structurally different decimation projection.
    """
    return [
        ("b2 majority", {"block": 2, "threshold": 2}),
        ("b2 OR", {"block": 2, "threshold": 1}),
        ("b2 AND", {"block": 2, "threshold": 4}),
        ("b2 subsample", {"block": 2, "project": lambda g: project_subsample(g, 2, (0, 0))}),
        ("b3 majority", {"block": 3, "threshold": 5}),
    ]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Robustness of the closure regime separation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--burn-in", type=int, default=64, help="relaxation steps K")
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3, 4],
        help="rng seeds (one measurement each, for the spread)",
    )
    p.add_argument("--n-soups", type=int, default=30, help="random soups per measurement")
    p.add_argument(
        "--grid-blocks", type=int, default=36,
        help="supercells per side; fine grid is grid_blocks * block",
    )
    p.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR,
        help="directory for the .csv and .png outputs",
    )
    return p.parse_args(argv)


def _rankdata(values):
    """Return 1-based ranks of `values`, averaging ties."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    order = a.argsort()
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom == 0.0:
        return 0.0
    return float((xm * ym).sum() / denom)


def spearman(x, y):
    return _pearson(_rankdata(x), _rankdata(y))


def run(args):
    rows = []
    for label, kwargs in configurations():
        for seed in args.seeds:
            for rule in CATALOGUE:
                result = measure_closure(
                    rule.birth,
                    rule.survive,
                    burn_in=args.burn_in,
                    n_soups=args.n_soups,
                    grid_blocks=args.grid_blocks,
                    seed=seed,
                    **kwargs,
                )
                rows.append(
                    {
                        "config": label,
                        "seed": seed,
                        "name": rule.name,
                        "rule": format_rule(rule.birth, rule.survive),
                        "regime": rule.regime,
                        "cond_entropy": result.cond_entropy,
                    }
                )
    return rows


def summarise(rows, seeds):
    """Per config: Spearman rho and margin, as mean +/- std over seeds."""
    configs = []
    for label, _ in configurations():
        rhos = []
        margins = []
        for seed in seeds:
            sel = [r for r in rows if r["config"] == label and r["seed"] == seed]
            defects = [r["cond_entropy"] for r in sel]
            ranks = [REGIME_RANK[r["regime"]] for r in sel]
            rhos.append(spearman(ranks, defects))
            chaotic = [r["cond_entropy"] for r in sel if r["regime"] == "chaotic"]
            ordered = [r["cond_entropy"] for r in sel if r["regime"] == "ordered"]
            margins.append(min(chaotic) - max(ordered))
        configs.append(
            {
                "config": label,
                "rho_mean": float(np.mean(rhos)),
                "rho_std": float(np.std(rhos)),
                "margin_mean": float(np.mean(margins)),
                "margin_std": float(np.std(margins)),
            }
        )
    return configs


def write_csv(rows, csv_path):
    fields = ["config", "seed", "name", "rule", "regime", "cond_entropy"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot(rows, summary, png_path):
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, (ax_rho, ax_strip) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Spearman rho per config, error bars = seed std.
    labels = [c["config"] for c in summary]
    rho = [c["rho_mean"] for c in summary]
    err = [c["rho_std"] for c in summary]
    x = range(len(summary))
    ax_rho.bar(x, rho, yerr=err, capsize=4, color="#555555")
    ax_rho.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax_rho.set_xticks(list(x))
    ax_rho.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_rho.set_ylabel("Spearman $\\rho$ (regime rank vs defect)")
    ax_rho.set_title("Does closure defect order the regimes?  (1.0 = perfect)")
    ax_rho.set_ylim(0, 1.05)
    ax_rho.grid(True, axis="y", alpha=0.3)

    # Right: defect by regime for the baseline config, points over seeds.
    baseline = "b2 majority"
    ax_strip.set_title("Closure defect by regime  (" + baseline + ")")
    for regime in REGIME_ORDER:
        pts = [
            r["cond_entropy"]
            for r in rows
            if r["config"] == baseline and r["regime"] == regime
        ]
        xpos = REGIME_RANK[regime]
        jitter = np.linspace(-0.16, 0.16, len(pts)) if len(pts) > 1 else [0.0]
        ax_strip.scatter(
            [xpos + j for j in jitter], pts,
            color=REGIME_COLOUR[regime], alpha=0.7, s=22,
        )
    ax_strip.set_xticks([REGIME_RANK[r] for r in REGIME_ORDER])
    ax_strip.set_xticklabels(REGIME_ORDER)
    ax_strip.set_ylabel("closure defect (bits)")
    ax_strip.set_xlabel("regime (literature prior)")
    ax_strip.grid(True, axis="y", alpha=0.3)
    legend_handles = [Patch(facecolor=REGIME_COLOUR[r], label=r) for r in REGIME_ORDER]
    ax_strip.legend(handles=legend_handles, fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)


def main(argv=None):
    args = parse_args(argv)
    rows = run(args)
    summary = summarise(rows, args.seeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "closure_robustness.csv"
    png_path = args.output_dir / "closure_robustness.png"
    write_csv(rows, csv_path)
    plot(rows, summary, png_path)

    print("Wrote " + str(csv_path))
    print("Wrote " + str(png_path))
    print()
    print(
        format("config", "<16s") + format("spearman_rho", ">16s")
        + format("margin", ">16s")
    )
    for c in summary:
        print(
            format(c["config"], "<16s")
            + format("%.3f +/- %.3f" % (c["rho_mean"], c["rho_std"]), ">16s")
            + format("%+.3f +/- %.3f" % (c["margin_mean"], c["margin_std"]), ">16s")
        )


if __name__ == "__main__":
    main()
