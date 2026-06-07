"""Coarse-graining: liveness grid -> set of placed objects.

The catalogue `O` of §7 is all finite Life patterns up to translation,
represented as the canonical frozenset of `(row, col)` offsets with minimum
corner at `(0, 0)`. A placed object is a `(pattern, anchor)` pair.

The coarse-graining sends a fine state to the frozenset of placed objects
obtained from its Moore-connected (8-neighbour) live-cell components. The
clustering rule is defined for every fine state and lands in `O × Z^2`
by construction, so reaction products are catalogued rather than
exceptional (annihilation being the empty set).
"""
from __future__ import annotations

import numpy as np

from .patterns import Anchor, Pattern


PlacedObject = tuple[Pattern, Anchor]
CoarseState = frozenset[PlacedObject]


def moore_components(beta: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return the Moore-connected components of live cells in `beta`.

    Each component is a list of `(row, col)` tuples. The order of
    components and of cells within a component is unspecified; downstream
    code must not depend on it.
    """
    visited = np.zeros(beta.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    rows, cols = beta.shape
    for r0 in range(rows):
        for c0 in range(cols):
            if beta[r0, c0] == 0 or visited[r0, c0]:
                continue
            component: list[tuple[int, int]] = []
            stack: list[tuple[int, int]] = [(r0, c0)]
            while stack:
                r, c = stack.pop()
                if not (0 <= r < rows and 0 <= c < cols):
                    continue
                if visited[r, c] or beta[r, c] == 0:
                    continue
                visited[r, c] = True
                component.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        stack.append((r + dr, c + dc))
            components.append(component)
    return components


def canonicalise(cells: list[tuple[int, int]]) -> PlacedObject:
    """Return `(canonical pattern, anchor)` for a non-empty list of cells.

    The anchor is `(min row, min col)` over the cells. The canonical
    pattern is the frozenset of cells shifted so that the anchor maps to
    `(0, 0)`. Two cell lists that are translates of each other produce
    the same canonical pattern.
    """
    if not cells:
        raise ValueError("cannot canonicalise empty component")
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    pattern: Pattern = frozenset(
        (r - min_r, c - min_c) for r, c in cells
    )
    return pattern, (min_r, min_c)


def coarse_state(beta: np.ndarray) -> CoarseState:
    """Coarse-grain a fine liveness grid to a frozenset of placed objects."""
    components = moore_components(beta)
    return frozenset(canonicalise(c) for c in components if c)
