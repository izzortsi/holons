"""Canonical Life patterns.

Coordinates are `(row, col)` with row increasing downward and column
increasing rightward.

The canonical SE-moving glider at phase 0 has live cells
`{(0,1), (1,2), (2,0), (2,1), (2,2)}`:

    . X .
    . . X
    X X X

After 4 B3/S23 steps it returns to phase 0 with the same shape translated
by `(+1, +1)`.

The canonical NW-moving glider at phase 0 is the 180° rotation of the SE
glider, with live cells `{(0,0), (0,1), (0,2), (1,0), (2,1)}`:

    X X X
    X . .
    . X .

After 4 steps it returns to phase 0 translated by `(-1, -1)`.
"""
from __future__ import annotations

import numpy as np


# Type aliases. `Pattern` is the canonical shape of an object: a frozenset of
# `(row, col)` offsets with minimum corner at `(0, 0)`. `Anchor` is the
# position of the minimum corner on the grid. `Velocity` is the per-period
# translation of the anchor under the catalogued coarse dynamics.
Pattern = frozenset[tuple[int, int]]
Anchor = tuple[int, int]
Velocity = tuple[int, int]


GLIDER_SE_PHASE0: Pattern = frozenset(
    [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
)
GLIDER_SE_VELOCITY: Velocity = (1, 1)

GLIDER_NW_PHASE0: Pattern = frozenset(
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 1)]
)
GLIDER_NW_VELOCITY: Velocity = (-1, -1)

GLIDER_PERIOD: int = 4


def make_grid(size: int) -> np.ndarray:
    """Return a `size`-by-`size` zero grid of dtype uint8."""
    if size < 3:
        raise ValueError(f"grid size must be at least 3, got {size}")
    return np.zeros((size, size), dtype=np.uint8)


def stamp(grid: np.ndarray, pattern: Pattern, anchor: Anchor) -> None:
    """Set the cells of `pattern` placed at `anchor` to live in `grid`.

    `anchor` is the `(row, col)` offset added to each cell of `pattern`.
    Raises `ValueError` if any stamped cell falls outside `grid`.
    """
    r0, c0 = anchor
    rows, cols = grid.shape
    for dr, dc in pattern:
        r, c = r0 + dr, c0 + dc
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(
                f"pattern cell ({r},{c}) out of grid {grid.shape}; "
                f"increase grid size or move anchor"
            )
        grid[r, c] = 1
