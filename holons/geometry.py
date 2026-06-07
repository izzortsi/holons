"""Shared collision geometry.

The collision experiment and the animation script both need the same glider
placement and grid-size recommendation logic. Putting them here makes the
geometry the single source of truth: both consumers see the same MARGIN,
the same anchor formula, and the same safe-grid-size bound.
"""
from __future__ import annotations

import numpy as np

from .patterns import (
    GLIDER_NW_PHASE0,
    GLIDER_PERIOD,
    GLIDER_SE_PHASE0,
    make_grid,
    stamp,
)


# Cells of vacuum kept between the SE glider's bounding box and the grid's
# upper-left corner at t = 0. The NW glider is placed symmetrically on the
# other end.
MARGIN: int = 5


def setup_collision(grid_size: int, separation: int) -> np.ndarray:
    """Return a fresh grid with an SE glider and an NW glider.

    The SE glider's anchor is at `(MARGIN, MARGIN)`. The NW glider's anchor
    is at `(MARGIN + 3 + separation, MARGIN + 3 + separation)`, placing
    `separation` cells of vacuum between the two 3x3 bounding boxes along
    each axis. Both gliders move along the main diagonal toward each other.
    """
    grid = make_grid(grid_size)
    se_anchor = (MARGIN, MARGIN)
    nw_anchor = (MARGIN + 3 + separation, MARGIN + 3 + separation)
    stamp(grid, GLIDER_SE_PHASE0, se_anchor)
    stamp(grid, GLIDER_NW_PHASE0, nw_anchor)
    return grid


def recommended_grid_size(separation_max: int, tau_max: int) -> int:
    """Return a conservative lower bound on a safe grid size.

    Accommodates both gliders' bounding boxes at `t = 0`, plus motion of at
    most `tau_max // GLIDER_PERIOD` cells per glider toward and away from
    the centre. Adds the same `MARGIN` on the far side as on the near side
    so that post-collision debris has room to move before hitting the
    boundary.
    """
    periods_max = tau_max // GLIDER_PERIOD
    return 2 * MARGIN + 6 + separation_max + 2 * periods_max
