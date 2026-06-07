"""Block coarse-graining: a fine liveness grid to a binary supercell grid.

Unlike the connected-component coarse-graining of coarse.py (which is tuned
to isolated gliders), the block transform is rule-agnostic: it partitions the
fine grid into disjoint b-by-b blocks and projects each block to a single
binary supercell that is live iff at least `threshold` of its b^2 fine cells
are live (majority by default). This is the coarse-graining used by the
closure experiment, where no object catalogue exists.

The fine grid side length must be a multiple of the block size `b`.
"""
from __future__ import annotations

import numpy as np


def majority_threshold(block: int) -> int:
    """Return the majority threshold (ceil(b^2 / 2)) for a `block`-by-`block` cell."""
    return (block * block + 1) // 2


def project_blocks(beta: np.ndarray, block: int, threshold: int) -> np.ndarray:
    """Project `beta` to a binary supercell grid by b-by-b block majority.

    `beta` is a 2D uint8 liveness grid whose side lengths are multiples of
    `block`. Returns a uint8 array of shape `(rows // block, cols // block)`
    whose entry is 1 iff its block contains at least `threshold` live cells.
    """
    if beta.ndim != 2:
        raise ValueError(f"beta must be 2D, got shape {beta.shape}")
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    rows, cols = beta.shape
    if rows % block != 0 or cols % block != 0:
        raise ValueError(
            f"grid shape {beta.shape} not divisible by block size {block}"
        )
    gr, gc = rows // block, cols // block
    # Sum each b-by-b block via a reshape over the two block axes.
    block_sums = beta.reshape(gr, block, gc, block).sum(axis=(1, 3))
    return (block_sums >= threshold).astype(np.uint8)


def project_subsample(beta: np.ndarray, block: int, offset: tuple[int, int]) -> np.ndarray:
    """Project `beta` to supercells by reading one fixed cell per block.

    `offset` is the `(row, col)` position within each block to sample (each in
    0..block-1). This is a structurally different projection from
    `project_blocks` (a decimation rather than a vote), used to check that the
    closure separation is not an artefact of the majority projection.
    """
    if beta.ndim != 2:
        raise ValueError(f"beta must be 2D, got shape {beta.shape}")
    rows, cols = beta.shape
    if rows % block != 0 or cols % block != 0:
        raise ValueError(
            f"grid shape {beta.shape} not divisible by block size {block}"
        )
    r, c = offset
    if not (0 <= r < block and 0 <= c < block):
        raise ValueError(f"offset {offset} out of range for block {block}")
    return beta[r::block, c::block].astype(np.uint8)
