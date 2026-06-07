"""Structural-emergence support breadth |L|.

Definition 13 (structural emergence) holds for a placed object when its
centroid valuation's essential support meets >= 2 distinct cell-parts. For
the object-scale coarse-graining of Section 6 the centroid is the finite
average over L (the object's live cells), so its essential support is
exactly L and the support breadth is

    |L| = number of live cells in the component.

An object is structurally emergent iff |L| >= 2; a lone live cell (|L| = 1)
has singleton support (its "centroid" is that one cell) and is therefore
not structurally emergent.

This is the structural counterpart to defect.py (the temporal axis). The
two together exhibit the Definition 13 / Definition 18 dissociation: the
support breadth of a glider stays >= 2 throughout free flight and collision,
while the temporal defect is zero in free flight and positive only at
contact.
"""
from __future__ import annotations

import numpy as np

from .coarse import coarse_state


# |L| >= STRUCTURAL_FLOOR  <=>  the object is structurally emergent.
STRUCTURAL_FLOOR: int = 2


def support_breadths(beta: np.ndarray) -> list[int]:
    """Return the support breadth |L| of each placed object in `beta`.

    Each value is the live-cell count of one Moore-connected component,
    i.e. the cardinality of the essential support of that object's centroid
    valuation. The list order is unspecified; callers must not depend on it.
    Returns an empty list when `beta` has no live cells.
    """
    return [len(pattern) for pattern, _ in coarse_state(beta)]


def min_support_breadth(beta: np.ndarray) -> int:
    """Return the smallest support breadth over the placed objects in `beta`.

    Returns 0 when `beta` has no objects (no live cells). A return value
    >= STRUCTURAL_FLOOR means every object present is structurally emergent.
    The empty-state 0 is distinguishable from a genuine |L| = 1 fragment by
    also checking the object count (see `support_breadths`).
    """
    breadths = support_breadths(beta)
    if not breadths:
        return 0
    return min(breadths)
