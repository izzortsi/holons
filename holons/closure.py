"""Intrinsic closure defect of a Life-like rule under block coarse-graining.

This is the rule-agnostic defect of the framework's temporal axis. There is
no object catalogue and no hand-built coarse dynamics. Instead we ask whether
a coarse dynamics EXISTS at all: under the block coarse-graining (coarse_block)
with supercell size b and time-rescaling T, is the coarse successor of a
supercell a well-defined function of its coarse neighbourhood?

For a radius-1 fine rule, after T fine steps a supercell's successor depends
on the fine cells within Chebyshev distance T of its b-by-b footprint. With
T <= b that influence region falls inside the 3-by-3 coarse neighbourhood of
the supercell. So the closure question is: does the 3-by-3 coarse
neighbourhood at t=0 determine the coarse cell at t=T?

We estimate this over random soups. For every interior supercell we record
the pair (3x3 coarse neighbourhood, coarse successor). If the SAME
neighbourhood ever yields DIFFERENT successors the coarse update is not a
function -- that ambiguity is the defect:

  cond_entropy      = E_neighbourhood[ H(successor | neighbourhood) ]  in bits
  ambiguous_fraction = fraction of samples whose neighbourhood is ambiguous

cond_entropy is 0 iff the coarse update is closed (a deterministic coarse
rule exists at this b, T, threshold), and at most 1 bit since the successor
is binary. T must not exceed b, or the 3x3 neighbourhood would not capture
the full influence region and the measured ambiguity would be inflated by
out-of-window cells rather than genuine non-closure.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import numpy as np

from .coarse_block import majority_threshold, project_blocks
from .life import step_rule


class ClosureResult(NamedTuple):
    cond_entropy: float        # E[H(successor | neighbourhood)], bits, 0 == closed
    ambiguous_fraction: float  # fraction of samples in ambiguous neighbourhoods
    n_samples: int             # total (neighbourhood, successor) pairs observed
    n_keys: int                # distinct neighbourhoods observed (of 512)


# The 3x3 coarse neighbourhood is encoded as a 9-bit key; bit (di,dj) ->
# (di+1)*3 + (dj+1), with di,dj in {-1,0,1}.
_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    ((di + 1) * 3 + (dj + 1), di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
)
_N_KEYS = 512  # 2 ** 9


def _neighbourhood_keys(coarse0: np.ndarray) -> np.ndarray:
    """Return the 9-bit neighbourhood key for every interior supercell.

    `coarse0` is a binary (G, G) supercell grid. The result has shape
    (G-2, G-2): entry (i, j) is the key of the 3x3 window centred on
    `coarse0[i+1, j+1]`.
    """
    gr, gc = coarse0.shape
    key = np.zeros((gr - 2, gc - 2), dtype=np.int64)
    for bit, di, dj in _OFFSETS:
        plane = coarse0[1 + di : gr - 1 + di, 1 + dj : gc - 1 + dj]
        key |= plane.astype(np.int64) << bit
    return key


def measure_closure(
    birth: frozenset[int],
    survive: frozenset[int],
    block: int = 2,
    time_steps: int = 0,
    burn_in: int = 0,
    threshold: int = 0,
    project: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_soups: int = 200,
    grid_blocks: int = 48,
    densities: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    seed: int = 12345,
) -> ClosureResult:
    """Measure the intrinsic closure defect of a Life-like rule.

    `time_steps` defaults to `block` (the natural T = b rescaling) and must
    not exceed it. `burn_in` evolves each random soup that many fine steps
    BEFORE the measurement, so closure is sampled on the rule's relaxed
    distribution (near its attractor) rather than on raw soup; with
    `burn_in = 0` the soup is measured directly. `project` is the fine->binary
    coarse projection; when omitted it is the block-majority projection at
    `threshold` (default the block majority). Soups are drawn at the given
    densities, cycled across `n_soups` samples, on a `grid_blocks * block`
    square fine grid.
    """
    if time_steps <= 0:
        time_steps = block
    if time_steps > block:
        raise ValueError(
            f"time_steps {time_steps} exceeds block {block}; the 3x3 coarse "
            f"neighbourhood would not capture the full influence region"
        )
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")
    if threshold <= 0:
        threshold = majority_threshold(block)
    if grid_blocks < 3:
        raise ValueError(f"grid_blocks must be >= 3, got {grid_blocks}")

    projector = project
    if projector is None:
        def projector(grid: np.ndarray) -> np.ndarray:
            return project_blocks(grid, block, threshold)

    rng = np.random.default_rng(seed)
    fine_side = grid_blocks * block

    # counts[key, successor] over all soups.
    counts = np.zeros((_N_KEYS, 2), dtype=np.int64)
    for s in range(n_soups):
        density = densities[s % len(densities)]
        soup = (rng.random((fine_side, fine_side)) < density).astype(np.uint8)

        # Relax the soup toward the rule's attractor before measuring.
        beta0 = soup
        for _ in range(burn_in):
            beta0 = step_rule(beta0, birth, survive)

        betaT = beta0
        for _ in range(time_steps):
            betaT = step_rule(betaT, birth, survive)

        coarse0 = projector(beta0)
        coarseT = projector(betaT)

        keys = _neighbourhood_keys(coarse0).ravel()
        successors = coarseT[1:-1, 1:-1].ravel().astype(np.int64)
        flat = np.bincount(keys * 2 + successors, minlength=_N_KEYS * 2)
        counts += flat.reshape(_N_KEYS, 2)

    return _summarise(counts)


def _summarise(counts: np.ndarray) -> ClosureResult:
    """Reduce a (512, 2) count table to a ClosureResult."""
    per_key = counts.sum(axis=1)
    grand_total = int(per_key.sum())
    observed = per_key > 0
    n_keys = int(observed.sum())
    if grand_total == 0:
        return ClosureResult(0.0, 0.0, 0, 0)

    # Per-key conditional entropy H(successor | key), in bits.
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(per_key[:, None] > 0, counts / per_key[:, None], 0.0)
        log_term = np.where(prob > 0, np.log2(prob), 0.0)
    key_entropy = -(prob * log_term).sum(axis=1)

    weight = per_key / grand_total
    cond_entropy = float((weight * key_entropy).sum())

    # A key is ambiguous when it maps to both successors.
    ambiguous = (counts[:, 0] > 0) & (counts[:, 1] > 0)
    ambiguous_fraction = float(per_key[ambiguous].sum() / grand_total)

    return ClosureResult(cond_entropy, ambiguous_fraction, grand_total, n_keys)
