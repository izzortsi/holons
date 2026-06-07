#!/usr/bin/env python3
"""Sanity checks for the holons library.

Verifies framework invariants on simple configurations:

1. The SE glider returns to its phase-0 shape translated by `(+1, +1)` after
   4 fine steps.
2. The NW glider returns to its phase-0 shape translated by `(-1, -1)` after
   4 fine steps.
3. Coarse-graining an isolated glider gives a single placed object whose
   pattern and anchor are exactly the catalogued ones.
4. The defect of an isolated glider is 0 at every measured τ.
5. The defect of two well-separated parallel SE gliders is 0 at every
   measured τ (no interaction).

Exits 0 on success, 1 on any failure. Prints what was checked.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from holons.coarse import coarse_state
from holons.defect import defect
from holons.life import evolve
from holons.patterns import (
    GLIDER_NW_PHASE0,
    GLIDER_NW_VELOCITY,
    GLIDER_PERIOD,
    GLIDER_SE_PHASE0,
    GLIDER_SE_VELOCITY,
    make_grid,
    stamp,
)


def check_se_glider_periodicity() -> None:
    grid = make_grid(20)
    anchor = (5, 5)
    stamp(grid, GLIDER_SE_PHASE0, anchor)

    after = evolve(grid, GLIDER_PERIOD)
    expected = make_grid(20)
    expected_anchor = (
        anchor[0] + GLIDER_SE_VELOCITY[0],
        anchor[1] + GLIDER_SE_VELOCITY[1],
    )
    stamp(expected, GLIDER_SE_PHASE0, expected_anchor)

    if not np.array_equal(after, expected):
        diff = np.argwhere(after != expected)
        raise AssertionError(
            f"SE glider periodicity failed: after 4 steps the grid does not "
            f"match the phase-0 glider at {expected_anchor}. "
            f"Differing cells: {diff.tolist()[:10]}"
        )
    print("  OK: SE glider returns to phase 0 shifted by (+1,+1) after 4 steps")


def check_nw_glider_periodicity() -> None:
    grid = make_grid(20)
    anchor = (10, 10)
    stamp(grid, GLIDER_NW_PHASE0, anchor)

    after = evolve(grid, GLIDER_PERIOD)
    expected = make_grid(20)
    expected_anchor = (
        anchor[0] + GLIDER_NW_VELOCITY[0],
        anchor[1] + GLIDER_NW_VELOCITY[1],
    )
    stamp(expected, GLIDER_NW_PHASE0, expected_anchor)

    if not np.array_equal(after, expected):
        diff = np.argwhere(after != expected)
        raise AssertionError(
            f"NW glider periodicity failed: after 4 steps the grid does not "
            f"match the phase-0 glider at {expected_anchor}. "
            f"Differing cells: {diff.tolist()[:10]}"
        )
    print("  OK: NW glider returns to phase 0 shifted by (-1,-1) after 4 steps")


def check_isolated_glider_is_one_object() -> None:
    grid = make_grid(20)
    stamp(grid, GLIDER_SE_PHASE0, (5, 5))
    coarse = coarse_state(grid)
    if len(coarse) != 1:
        raise AssertionError(
            f"isolated glider should produce 1 placed object, got {len(coarse)}"
        )
    pattern, anchor = next(iter(coarse))
    if pattern != GLIDER_SE_PHASE0:
        raise AssertionError(
            f"isolated glider's canonical pattern should be GLIDER_SE_PHASE0, "
            f"got {set(pattern)}"
        )
    if anchor != (5, 5):
        raise AssertionError(
            f"isolated glider's anchor should be (5,5), got {anchor}"
        )
    print("  OK: isolated SE glider coarse-grains to (GLIDER_SE_PHASE0, (5,5))")


def check_isolated_glider_zero_defect() -> None:
    grid = make_grid(40)
    stamp(grid, GLIDER_SE_PHASE0, (10, 10))
    velocities = {GLIDER_SE_PHASE0: GLIDER_SE_VELOCITY}
    for tau in (4, 8, 12, 16, 20):
        d = defect(grid, tau, velocities, GLIDER_PERIOD)
        if d != 0:
            raise AssertionError(
                f"isolated glider should have defect 0 at tau={tau}, got {d}"
            )
    print("  OK: isolated SE glider has defect 0 at tau in {4, 8, 12, 16, 20}")


def check_parallel_gliders_zero_defect() -> None:
    grid = make_grid(60)
    stamp(grid, GLIDER_SE_PHASE0, (5, 5))
    stamp(grid, GLIDER_SE_PHASE0, (5, 30))
    velocities = {GLIDER_SE_PHASE0: GLIDER_SE_VELOCITY}
    for tau in (4, 8, 12, 16, 20):
        d = defect(grid, tau, velocities, GLIDER_PERIOD)
        if d != 0:
            raise AssertionError(
                f"two parallel SE gliders should have defect 0 at tau={tau}, "
                f"got {d}"
            )
    print("  OK: two parallel SE gliders have defect 0 at tau in {4, 8, 12, 16, 20}")


def main() -> int:
    checks = [
        ("SE glider periodicity", check_se_glider_periodicity),
        ("NW glider periodicity", check_nw_glider_periodicity),
        (
            "Isolated glider coarse-grains correctly",
            check_isolated_glider_is_one_object,
        ),
        ("Isolated glider has zero defect", check_isolated_glider_zero_defect),
        ("Parallel gliders have zero defect", check_parallel_gliders_zero_defect),
    ]
    failures = 0
    for name, check in checks:
        print(f"[{name}]")
        try:
            check()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failures += 1
    if failures:
        print(f"\n{failures} of {len(checks)} checks failed.")
        return 1
    print(f"\nAll {len(checks)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
