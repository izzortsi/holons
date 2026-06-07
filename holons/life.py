"""B3/S23 cellular automaton on Z^2 with open (zero-padded) boundary.

Pure numpy. The grid is a 2D uint8 array with values in {0, 1}. The boundary
is treated as permanently dead, which is correct for finite patterns kept
sufficiently far from the grid edges. The caller is responsible for choosing
a grid large enough that the patterns of interest do not reach the boundary
within the simulation window.
"""
from __future__ import annotations

import numpy as np


# Conway's Life as a Life-like (outer-totalistic) rule: birth on exactly 3
# live neighbours, survive on 2 or 3. Used as the default for `step`.
LIFE_BIRTH: frozenset[int] = frozenset({3})
LIFE_SURVIVE: frozenset[int] = frozenset({2, 3})


def _neighbour_lut(counts: frozenset[int]) -> np.ndarray:
    """Return a length-9 bool lookup table marking the live-neighbour counts.

    `lut[k]` is True iff `k` is in `counts`. Indexed by Moore-neighbour
    counts, which range over 0..8.
    """
    lut = np.zeros(9, dtype=bool)
    for k in counts:
        if not (0 <= k <= 8):
            raise ValueError(f"neighbour count {k} out of range 0..8")
        lut[k] = True
    return lut


def step_rule(
    beta: np.ndarray,
    birth: frozenset[int],
    survive: frozenset[int],
) -> np.ndarray:
    """Apply one outer-totalistic (Life-like) update to liveness grid `beta`.

    `birth` and `survive` are sets of live-neighbour counts (each a subset of
    0..8): a dead cell becomes live iff its neighbour count is in `birth`, a
    live cell stays live iff its count is in `survive`. Returns a new uint8
    array of the same shape, with an open (zero-padded) boundary.

    B0 rules (0 in `birth`) are rejected: with a zero-padded boundary they
    would spontaneously birth the infinite dead exterior, which this finite,
    open-boundary model does not represent.
    """
    if beta.dtype != np.uint8:
        raise TypeError(f"beta must be uint8, got {beta.dtype}")
    if beta.ndim != 2:
        raise ValueError(f"beta must be 2D, got shape {beta.shape}")
    if beta.shape[0] < 1 or beta.shape[1] < 1:
        raise ValueError(f"beta must be non-empty, got shape {beta.shape}")
    if 0 in birth:
        raise ValueError("B0 rules are not supported on an open boundary")

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
    birth_lut = _neighbour_lut(birth)
    survive_lut = _neighbour_lut(survive)
    born = (interior == 0) & birth_lut[n]
    survived = (interior == 1) & survive_lut[n]
    return (born | survived).astype(np.uint8)


def step(beta: np.ndarray) -> np.ndarray:
    """Apply one B3/S23 (Conway's Life) update to liveness grid `beta`.

    Thin wrapper over `step_rule` with Life's birth/survive masks. Returns a
    new uint8 array of the same shape.
    """
    return step_rule(beta, LIFE_BIRTH, LIFE_SURVIVE)


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
