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
from holons.defect import defect, defect_matched, matched_distance
from holons.geometry import setup_collision
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


# A 2x2 still-life block, used as a distinct object type in the grading check.
BLOCK_PHASE0 = frozenset([(0, 0), (0, 1), (1, 0), (1, 1)])


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


def check_matched_defect_grades_displacement() -> None:
    # One predicted glider at the origin, compared against three outcomes.
    predicted = frozenset([(GLIDER_SE_PHASE0, (0, 0))])
    survived_shifted = frozenset([(GLIDER_SE_PHASE0, (1, 1))])  # same type, 1 cell off
    annihilated = frozenset()                                   # nothing survives
    replaced = frozenset([(BLOCK_PHASE0, (0, 0))])              # glider -> block

    shifted = matched_distance(predicted, survived_shifted)
    gone = matched_distance(predicted, annihilated)
    swapped = matched_distance(predicted, replaced)

    # The graded metric must order survived-shifted < annihilated < replaced.
    if not (0.0 < shifted < gone < swapped):
        raise AssertionError(
            f"matched defect ordering wrong: shifted={shifted}, "
            f"annihilated={gone}, replaced={swapped} "
            f"(expected 0 < shifted < annihilated < replaced)"
        )

    # The symmetric-difference metric gets it backwards: a surviving but
    # shifted glider (2) scores worse than a clean annihilation (1).
    sym_shifted = len(predicted.symmetric_difference(survived_shifted))
    sym_gone = len(predicted.symmetric_difference(annihilated))
    if not (sym_shifted == 2 and sym_gone == 1):
        raise AssertionError(
            f"symmetric-difference baseline changed: shifted={sym_shifted}, "
            f"annihilated={sym_gone} (expected 2 and 1)"
        )
    print(
        "  OK: matched defect grades shifted(%.2f) < annihilated(%.2f) < "
        "replaced(%.2f); symdiff gets shifted(%d) > annihilated(%d) backwards"
        % (shifted, gone, swapped, sym_shifted, sym_gone)
    )


def check_matched_defect_matches_baseline_on_collision() -> None:
    # On a symmetric head-on collision no object survives displaced, so the
    # graded metric must reduce to the integer symmetric-difference defect.
    grid = setup_collision(60, 8)
    velocities = {
        GLIDER_SE_PHASE0: GLIDER_SE_VELOCITY,
        GLIDER_NW_PHASE0: GLIDER_NW_VELOCITY,
    }
    for tau in (4, 8, 12, 16, 20, 24, 28, 32, 36, 40):
        base = defect(grid, tau, velocities, GLIDER_PERIOD)
        graded = defect_matched(grid, tau, velocities, GLIDER_PERIOD)
        if graded != float(base):
            raise AssertionError(
                f"matched defect {graded} != baseline {base} at tau={tau} "
                f"on the symmetric collision (they must agree here)"
            )
    print(
        "  OK: matched defect equals the integer baseline at every tau on the "
        "sep=8 symmetric collision"
    )


def check_vanish_rule_is_closed() -> None:
    # B/S kills every cell in one step, so the coarse field is constantly
    # dead: the coarse update is the constant 0 function, perfectly closed.
    from holons.closure import measure_closure

    result = measure_closure(
        frozenset(), frozenset(), n_soups=40, grid_blocks=24, seed=7
    )
    if result.cond_entropy != 0.0 or result.ambiguous_fraction != 0.0:
        raise AssertionError(
            f"Vanish rule should be perfectly closed (defect 0), got "
            f"cond_entropy={result.cond_entropy}, "
            f"ambiguous_fraction={result.ambiguous_fraction}"
        )
    print("  OK: Vanish B/S has closure defect 0 (a coarse rule exists)")


def check_life_rule_is_not_closed() -> None:
    # Life under a 2x2 majority block has no exact coarse rule: the same
    # coarse neighbourhood yields different successors, so the defect is > 0.
    from holons.closure import measure_closure

    result = measure_closure(
        frozenset({3}), frozenset({2, 3}), n_soups=40, grid_blocks=24, seed=7
    )
    if result.cond_entropy <= 0.0:
        raise AssertionError(
            f"Life rule should be non-closed (defect > 0), got "
            f"cond_entropy={result.cond_entropy}"
        )
    print(
        "  OK: Life B3/S23 has closure defect %.3f > 0 (no exact coarse rule)"
        % result.cond_entropy
    )


def check_lenia_kernel_normalised() -> None:
    from holons.lenia import ORBIUM, kernel_fft

    spectrum = kernel_fft(ORBIUM, 64)
    # The DC term of the kernel's rfft2 is exactly the kernel's spatial sum.
    dc = float(spectrum[0, 0].real)
    if abs(dc - 1.0) > 1e-9:
        raise AssertionError(f"Lenia kernel should sum to 1, got {dc}")
    print("  OK: Lenia kernel is normalised to 1 (DC term = %.6f)" % dc)


def check_lenia_shift_equivariance() -> None:
    # Circular convolution makes the update shift-equivariant on the torus:
    # stepping a rolled field equals rolling the stepped field.
    from holons.lenia import ORBIUM, kernel_fft, step

    rng = np.random.default_rng(0)
    field = rng.random((64, 64))
    spectrum = kernel_fft(ORBIUM, 64)
    shift = (7, -3)
    a = step(np.roll(field, shift, axis=(0, 1)), spectrum, ORBIUM.mu, ORBIUM.sigma, ORBIUM.dt)
    b = np.roll(
        step(field, spectrum, ORBIUM.mu, ORBIUM.sigma, ORBIUM.dt), shift, axis=(0, 1)
    )
    err = float(np.abs(a - b).max())
    if err > 1e-9:
        raise AssertionError(f"Lenia step not shift-equivariant, max err {err}")
    print("  OK: Lenia step is shift-equivariant on the torus (max err %.1e)" % err)


def check_orbium_glides() -> None:
    # The decisive seed test: the Orbium must persist (mass neither vanishes
    # nor explodes) and translate (its centroid moves) over 100 fine steps.
    from holons.lenia import ORBIUM, center_of_mass, evolve, kernel_fft, total_mass
    from holons.orbium import place

    n = 64
    field = place(n)
    spectrum = kernel_fft(ORBIUM, n)
    m0 = total_mass(field)
    c0 = center_of_mass(field)

    after = evolve(field, spectrum, ORBIUM.mu, ORBIUM.sigma, ORBIUM.dt, 100)
    m1 = total_mass(after)
    c1 = center_of_mass(after)

    ratio = m1 / m0
    if not (0.5 <= ratio <= 2.0):
        raise AssertionError(
            f"Orbium mass ratio {ratio:.3f} outside [0.5, 2.0]: it dissolved "
            f"or exploded, so the seed/params are not a stable organism"
        )
    # Torus displacement of the centroid.
    dr = (c1[0] - c0[0] + n / 2) % n - n / 2
    dc = (c1[1] - c0[1] + n / 2) % n - n / 2
    dist = (dr * dr + dc * dc) ** 0.5
    if dist < 2.0:
        raise AssertionError(
            f"Orbium centroid moved only {dist:.2f} cells in 100 steps: not gliding"
        )
    print(
        "  OK: Orbium glides (mass ratio %.2f, centroid moved %.1f cells in 100 steps)"
        % (ratio, dist)
    )


def check_lenia_collision_is_local() -> None:
    # The dynamics is local: two Orbia still separated (well before contact)
    # evolve independently, so evolve(A+B) == evolve(A) + evolve(B) and the
    # superposition defect is zero to machine precision.
    from holons.lenia import ORBIUM, kernel_fft
    from holons.lenia_collision import setup_collision, superposition_defect_trajectory

    n = 96
    spectrum = kernel_fft(ORBIUM, n)
    field_a, field_b = setup_collision(n, 60.0, ORBIUM)
    _, defects = superposition_defect_trajectory(field_a, field_b, spectrum, ORBIUM, 10, 1)
    worst = max(defects)
    if worst > 1e-9:
        raise AssertionError(
            f"separated Orbia should have ~0 superposition defect before "
            f"contact, got {worst:g}"
        )
    print(
        "  OK: separated Orbia have superposition defect %.1e before contact (locality)"
        % worst
    )


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
        (
            "Matched defect grades displacement",
            check_matched_defect_grades_displacement,
        ),
        (
            "Matched defect matches baseline on collision",
            check_matched_defect_matches_baseline_on_collision,
        ),
        ("Vanish rule is closed", check_vanish_rule_is_closed),
        ("Life rule is not closed", check_life_rule_is_not_closed),
        ("Lenia kernel is normalised", check_lenia_kernel_normalised),
        ("Lenia step is shift-equivariant", check_lenia_shift_equivariance),
        ("Orbium glides", check_orbium_glides),
        ("Lenia collision is local", check_lenia_collision_is_local),
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
