"""B3/S23 cellular automaton on Z^2 with open (zero-padded) boundary.

Pure numpy. The grid is a 2D uint8 array with values in {0, 1}. The boundary
is treated as permanently dead, which is correct for finite patterns kept
sufficiently far from the grid edges. The caller is responsible for choosing
a grid large enough that the patterns of interest do not reach the boundary
within the simulation window.
"""
from __future__ import annotations

import numpy as np


def step(beta: np.ndarray) -> np.ndarray:
    """Apply one B3/S23 update to liveness grid `beta`.

    Returns a new uint8 array of the same shape. The implementation pads
    `beta` with zeros, computes Moore-neighbour counts via eight slice
    additions, and applies the standard birth/survive rules.
    """
    if beta.dtype != np.uint8:
        raise TypeError(f"beta must be uint8, got {beta.dtype}")
    if beta.ndim != 2:
        raise ValueError(f"beta must be 2D, got shape {beta.shape}")
    if beta.shape[0] < 1 or beta.shape[1] < 1:
        raise ValueError(f"beta must be non-empty, got shape {beta.shape}")

    # Pad with dead cells (open boundary). int8 keeps the neighbour sum well
    # below overflow (max 8).
    padded = np.zeros(
        (beta.shape[0] + 2, beta.shape[1] + 2),
        dtype=np.int8,
    )
    padded[1:-1, 1:-1] = beta

    # Count live Moore neighbours via eight slice additions.
    n = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )

    interior = padded[1:-1, 1:-1]
    birth = (interior == 0) & (n == 3)
    survive = (interior == 1) & ((n == 2) | (n == 3))
    return (birth | survive).astype(np.uint8)


def evolve(beta: np.ndarray, steps: int) -> np.ndarray:
    """Apply `step()` `steps` times to `beta`.

    Returns a new array; the input is not modified.
    """
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    result = beta
    for _ in range(steps):
        result = step(result)
    return result
