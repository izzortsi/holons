"""Temporal-emergence defect δ^t_τ.

For a fine state β and time τ, the defect compares two coarse descriptions:

- actual:    `coarse(Φ_τ(β))` — coarse-grain the τ-evolved fine state
- predicted: `Φ^+_{τ/period}(coarse(β))` — evolve the initial coarse state
             by Φ^+, which translates each placed object by its
             catalogued velocity scaled to τ

The defect is the cardinality of the symmetric difference of these sets
(the discrete pseudometric of §6, instantiated on the §7 shared host
`O × Z^2`).

Placed objects whose pattern is not in the `velocities` map are treated as
stationary, which is the §7 default for uncatalogued reaction products. To
catalogue a new pattern (a still life, an oscillator, a new spaceship),
add it to the `velocities` map passed in by the caller.
"""
from __future__ import annotations

import numpy as np

from .coarse import CoarseState, coarse_state
from .life import evolve
from .patterns import Pattern, Velocity


def coarse_evolve(
    coarse: CoarseState,
    velocities: dict[Pattern, Velocity],
    n_periods: int,
) -> CoarseState:
    """Apply Φ^+ for `n_periods` coarse ticks to `coarse`.

    Each placed object whose pattern is a key in `velocities` is
    translated by `velocities[pattern] * n_periods`. Placed objects whose
    pattern is not in `velocities` are treated as stationary.
    """
    if n_periods < 0:
        raise ValueError(f"n_periods must be >= 0, got {n_periods}")
    result: set[tuple[Pattern, tuple[int, int]]] = set()
    for pattern, (r, c) in coarse:
        v = velocities.get(pattern, (0, 0))
        result.add(
            (pattern, (r + v[0] * n_periods, c + v[1] * n_periods))
        )
    return frozenset(result)


def defect(
    beta: np.ndarray,
    tau: int,
    velocities: dict[Pattern, Velocity],
    period: int,
) -> int:
    """Compute `δ^t_τ(β)` as the symmetric-difference cardinality.

    `tau` must be a positive multiple of `period` so that every
    catalogued object returns to its phase-0 canonical form at the
    measurement time. This is the §7 simplification; a phase-aware
    extension is straightforward.

    `velocities` maps the phase-0 canonical pattern of each catalogued
    object class to its per-period velocity.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if tau % period != 0:
        raise ValueError(
            f"tau must be a multiple of period {period}, got {tau}"
        )
    n_periods = tau // period
    initial_coarse = coarse_state(beta)
    predicted = coarse_evolve(initial_coarse, velocities, n_periods)
    evolved = evolve(beta, tau)
    actual = coarse_state(evolved)
    return len(predicted.symmetric_difference(actual))
