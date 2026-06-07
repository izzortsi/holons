"""Lenia organism collision via the superposition defect.

The framework's defect asks whether the whole reduces to the non-interacting
aggregate of its parts. In Lenia this is computable directly on the field,
with no object catalogue and no segmentation, because the dynamics is local
(kernel radius R): two organisms farther than ~2R apart evolve independently,
so

    delta(tau) = || evolve(A + B, tau) - ( evolve(A, tau) + evolve(B, tau) ) ||

is exactly 0 while they are separated and becomes positive only once they
interact. The norm is the L1 field difference normalised by the organisms'
total mass -- the fraction of mass that the non-interacting prediction
misplaces. This is the continuous-substrate counterpart of the Game-of-Life
collision defect (Experiment 1).

A is the Orbium gliding along its natural velocity; B is the 180-degree
rotation of the Orbium, which (the Lenia dynamics being symmetric under
lattice rotations) glides along the reversed velocity, so the two head toward
each other.
"""
from __future__ import annotations

import numpy as np

from .lenia import ORBIUM, LeniaParams, center_of_mass, kernel_fft, step, total_mass
from .orbium import ORBIUM_CELLS


def orbium_velocity(
    params: LeniaParams = ORBIUM,
    grid_size: int = 64,
    settle: int = 20,
    steps: int = 40,
) -> tuple[float, float]:
    """Return the Orbium's per-step velocity vector (rows, cols) after settling."""
    spectrum = kernel_fft(params, grid_size)
    field = _place_centroid(grid_size, ORBIUM_CELLS, (grid_size / 2, grid_size / 2))
    for _ in range(settle):
        field = step(field, spectrum, params.mu, params.sigma, params.dt)
    c0 = center_of_mass(field)
    for _ in range(steps):
        field = step(field, spectrum, params.mu, params.sigma, params.dt)
    c1 = center_of_mass(field)

    def wrap(d):
        return (d + grid_size / 2) % grid_size - grid_size / 2

    return (wrap(c1[0] - c0[0]) / steps, wrap(c1[1] - c0[1]) / steps)


def _place_centroid(grid_size: int, cells: np.ndarray, target) -> np.ndarray:
    """Return a field with `cells` placed so its mass centroid sits at `target`.

    Placement wraps around the torus, so the organism may straddle the
    boundary. The seed must be no larger than the grid.
    """
    h, w = cells.shape
    if h > grid_size or w > grid_size:
        raise ValueError(f"seed {cells.shape} larger than grid {grid_size}")
    cm_r = float((cells.sum(axis=1) * np.arange(h)).sum() / cells.sum())
    cm_c = float((cells.sum(axis=0) * np.arange(w)).sum() / cells.sum())
    r0 = int(round(target[0] - cm_r))
    c0 = int(round(target[1] - cm_c))
    base = np.zeros((grid_size, grid_size), dtype=np.float64)
    base[:h, :w] = cells
    return np.roll(base, (r0, c0), axis=(0, 1))


def setup_collision(
    grid_size: int,
    separation: float,
    params: LeniaParams = ORBIUM,
    velocity=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two Orbium fields (A, B) on a head-on collision course.

    `separation` is the centre-to-centre distance, in cells, along the glide
    axis at t=0. A glides toward the centre along +velocity; B is the
    180-degree-rotated Orbium placed `separation` cells ahead along that axis,
    gliding back along -velocity. The two fields are returned separately so the
    caller can evolve each alone for the superposition defect.
    """
    if velocity is None:
        velocity = orbium_velocity(params, grid_size)
    v = np.array(velocity, dtype=float)
    speed = float(np.linalg.norm(v))
    if speed == 0.0:
        raise ValueError("organism velocity is zero; cannot set up a collision")
    unit = v / speed
    centre = np.array([grid_size / 2.0, grid_size / 2.0])

    field_a = _place_centroid(grid_size, ORBIUM_CELLS, centre - (separation / 2.0) * unit)
    cells_b = np.rot90(ORBIUM_CELLS, 2)
    field_b = _place_centroid(grid_size, cells_b, centre + (separation / 2.0) * unit)
    return field_a, field_b


def superposition_defect_trajectory(
    field_a: np.ndarray,
    field_b: np.ndarray,
    kernel_spectrum: np.ndarray,
    params: LeniaParams,
    tau_max: int,
    tau_step: int = 1,
) -> tuple[list[int], list[float]]:
    """Return (taus, defects) for the superposition defect up to `tau_max`.

    Evolves A alone, B alone, and A+B together in lockstep, sampling the
    normalised L1 difference `||together - (A + B)|| / total_mass` every
    `tau_step` steps. The defect is 0 at tau=0 and while the organisms are
    separated, rising once they interact.
    """
    mass0 = total_mass(field_a) + total_mass(field_b)
    a = field_a
    b = field_b
    together = field_a + field_b

    taus = [0]
    defects = [0.0]
    for t in range(1, tau_max + 1):
        a = step(a, kernel_spectrum, params.mu, params.sigma, params.dt)
        b = step(b, kernel_spectrum, params.mu, params.sigma, params.dt)
        together = step(together, kernel_spectrum, params.mu, params.sigma, params.dt)
        if t % tau_step == 0:
            defect = float(np.abs(together - (a + b)).sum() / mass0)
            taus.append(t)
            defects.append(defect)
    return taus, defects
