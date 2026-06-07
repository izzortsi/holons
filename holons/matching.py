"""Min-cost assignment (Hungarian algorithm), numpy-only.

Used by defect.py to grade the temporal defect. The symmetric-difference
count scores a displaced survivor (a glider one cell off its prediction)
identically to a clean annihilation; min-cost matching instead pairs
predicted and actual objects, charging a small bounded cost for displacement
and a larger fixed cost for each unmatched (created or destroyed) object.

`min_cost_assignment` solves the square assignment problem in O(n^3) by the
standard potentials method. Disallowed pairings are encoded by the caller as
large finite costs, never infinities, so the solver does no infinity
arithmetic on the cost entries.
"""
from __future__ import annotations

import numpy as np


def min_cost_assignment(cost: np.ndarray) -> tuple[list[int], float]:
    """Solve the square min-cost assignment problem.

    `cost` is an n-by-n array of finite costs. Returns `(assignment, total)`
    where `assignment[i]` is the column matched to row `i` and `total` is the
    summed cost. An empty (0-by-0) matrix returns `([], 0.0)`.
    """
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError(f"cost must be a square 2D array, got shape {cost.shape}")
    n = cost.shape[0]
    if n == 0:
        return [], 0.0

    sentinel = float("inf")
    # 1-indexed potentials (u, v), column->row matching (p) and augmenting
    # back-pointers (way), per the standard O(n^3) formulation.
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [sentinel] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = sentinel
            j1 = -1
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    total = float(sum(cost[i, assignment[i]] for i in range(n)))
    return assignment, total
