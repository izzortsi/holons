"""Temporal-emergence defect δ^t_τ.

For a fine state β and time τ, the defect compares two coarse descriptions:

- actual:    `coarse(Φ_τ(β))` — coarse-grain the τ-evolved fine state
- predicted: `Φ^+_{τ/period}(coarse(β))` — evolve the initial coarse state
             by Φ^+, which translates each placed object by its
             catalogued velocity scaled to τ

Two pseudometrics on these object-sets are provided:

- `defect` — the cardinality of the symmetric difference (the discrete
  pseudometric of §6 on the §7 shared host `O × Z^2`). An integer. It scores
  a survivor displaced by one cell (two distinct `(pattern, anchor)` keys)
  identically to a clean annihilation: both contribute 2. This conflates
  displacement with creation/annihilation.

- `defect_matched` — a graded distance via min-cost object matching
  (`matching.py`). Same-pattern objects are paired at a small bounded
  displacement cost; an unmatched (created or destroyed) object pays a fixed
  cost. So "survived but shifted" grades strictly below "destroyed", which
  grades below "destroyed and replaced". On a symmetric head-on collision,
  where no object survives displaced, `defect_matched` reduces to the same
  integers as `defect`; the grading only separates them when an object
  persists off its predicted slot (asymmetric collisions, downstream work).

Both share the same given-model Φ^+: a positive defect means the
independent-translation coarse model mispredicted, i.e. the objects
interacted. It is an interaction detector, not a measure of intrinsic
irreducibility.

Placed objects whose pattern is not in the `velocities` map are treated as
stationary, which is the §7 default for uncatalogued reaction products. To
catalogue a new pattern (a still life, an oscillator, a new spaceship),
add it to the `velocities` map passed in by the caller.
"""
from __future__ import annotations

import numpy as np

from .coarse import CoarseState, coarse_state
from .life import evolve
from .matching import min_cost_assignment
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


def predicted_and_actual(
    beta: np.ndarray,
    tau: int,
    velocities: dict[Pattern, Velocity],
    period: int,
) -> tuple[CoarseState, CoarseState]:
    """Return `(predicted, actual)` coarse states at time `tau`.

    `predicted` is the initial coarse state pushed forward by Φ^+ (catalogued
    objects translated by their velocity); `actual` is the coarse-graining of
    the τ-evolved fine state. Shared by `defect` and `defect_matched`.

    `tau` must be a positive multiple of `period` so that every catalogued
    object returns to its phase-0 canonical form at the measurement time.
    This is the §7 simplification; a phase-aware extension is straightforward.
    `velocities` maps the phase-0 canonical pattern of each catalogued object
    class to its per-period velocity.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if tau % period != 0:
        raise ValueError(
            f"tau must be a multiple of period {period}, got {tau}"
        )
    n_periods = tau // period
    predicted = coarse_evolve(coarse_state(beta), velocities, n_periods)
    actual = coarse_state(evolve(beta, tau))
    return predicted, actual


def defect(
    beta: np.ndarray,
    tau: int,
    velocities: dict[Pattern, Velocity],
    period: int,
) -> int:
    """Compute `δ^t_τ(β)` as the symmetric-difference cardinality.

    See the module docstring for the conflation this metric has (a displaced
    survivor scores like an annihilation); `defect_matched` grades them apart.
    """
    predicted, actual = predicted_and_actual(beta, tau, velocities, period)
    return len(predicted.symmetric_difference(actual))


def matched_distance(
    predicted: CoarseState,
    actual: CoarseState,
    unmatched_cost: float = 1.0,
    disp_rate: float = 0.25,
    disp_cap: float = 0.5,
    type_change_cost: float = -1.0,
) -> float:
    """Graded distance between two coarse states via min-cost matching.

    Each predicted object is matched to at most one actual object:

    - same pattern: cost `min(disp_rate * L1(anchor_p, anchor_a), disp_cap)`,
      a small bounded displacement penalty;
    - different pattern: forbidden when `type_change_cost < 0` (the default,
      so "same object" means same pattern); otherwise
      `type_change_cost + displacement`, treating it as a transformation.

    Every unmatched object (a prediction that did not occur, or an actual
    object that was not predicted) pays `unmatched_cost`. The returned value
    is the optimal total. With the defaults a survivor displaced by one cell
    costs 0.25, an annihilation 1.0, and a destroy-and-replace 2.0.
    """
    pred_list = list(predicted)
    act_list = list(actual)
    m = len(pred_list)
    n = len(act_list)
    if m + n == 0:
        return 0.0

    # A finite stand-in for "forbidden", strictly above any all-unmatched
    # solution so the solver never chooses a forbidden pairing.
    big = (m + n) * unmatched_cost + 1.0

    # Square cost matrix of size (m + n). Columns [0, n) are actual objects,
    # columns [n, n + m) are per-predicted "destroyed" slots. Rows [0, m) are
    # predicted objects, rows [m, m + n) are per-actual "created" slots.
    size = m + n
    cost = np.full((size, size), big, dtype=float)
    for i in range(m):
        pattern_p, anchor_p = pred_list[i]
        for j in range(n):
            pattern_a, anchor_a = act_list[j]
            displacement = min(
                disp_rate
                * (
                    abs(anchor_p[0] - anchor_a[0])
                    + abs(anchor_p[1] - anchor_a[1])
                ),
                disp_cap,
            )
            if pattern_p == pattern_a:
                cost[i, j] = displacement
            elif type_change_cost >= 0.0:
                cost[i, j] = type_change_cost + displacement
            # else leave as `big` (forbidden cross-type match)
        cost[i, n + i] = unmatched_cost  # predicted i destroyed
    for j in range(n):
        cost[m + j, j] = unmatched_cost  # actual j created
        for l in range(m):
            cost[m + j, n + l] = 0.0  # ghost-to-ghost filler

    _, total = min_cost_assignment(cost)
    return total


def defect_matched(
    beta: np.ndarray,
    tau: int,
    velocities: dict[Pattern, Velocity],
    period: int,
    unmatched_cost: float = 1.0,
    disp_rate: float = 0.25,
    disp_cap: float = 0.5,
    type_change_cost: float = -1.0,
) -> float:
    """Compute the matched (graded) defect `δ^t_τ(β)`.

    Same predicted/actual construction as `defect`, scored by
    `matched_distance` instead of the symmetric difference. See the module
    docstring and `matched_distance` for the cost model.
    """
    predicted, actual = predicted_and_actual(beta, tau, velocities, period)
    return matched_distance(
        predicted,
        actual,
        unmatched_cost,
        disp_rate,
        disp_cap,
        type_change_cost,
    )
