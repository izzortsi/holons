"""Stability of a Lenia organism through the framework's temporal defect.

Coarse-grain the organism to its mass centroid and take the coarse dynamics
Phi^+ to be uniform translation. The temporal-emergence defect is then the
failure of the centroid trajectory to be constant-velocity: the RMS distance
of the trajectory from its best-fit straight line, in cells. A coherent
glider moves at constant velocity, so its defect is ~0 (its coarse motion is
reducible to translation, like a lone Game-of-Life glider); a destabilising
organism deforms, accelerates or dissolves, so its centroid trajectory bends
and the defect grows.

`measure_stability` returns both the centroid defect and the log mass ratio
(survival), so a sweep over the growth parameters (mu, sigma) maps the
low-defect glider island and the dissolution / explosion regions around it.
The Lenia kernel does not depend on (mu, sigma), so a caller sweeping
parameters should precompute the kernel spectrum once and pass it in.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .lenia import center_of_mass, step, total_mass


class StabilityResult(NamedTuple):
    centroid_defect: float   # RMS deviation of the centroid path from uniform motion, cells
    log_mass_ratio: float    # log10(final mass / initial mass); 0 == conserved


def _unwrap_periodic(values: np.ndarray, period: int) -> np.ndarray:
    """Unwrap a periodic position sequence so a gliding centroid is monotone.

    Wraps the step-to-step differences into (-period/2, period/2] and
    accumulates them, removing torus jumps from an otherwise smooth path.
    """
    values = np.asarray(values, dtype=float)
    diffs = np.diff(values)
    diffs = (diffs + period / 2.0) % period - period / 2.0
    return np.concatenate([[values[0]], values[0] + np.cumsum(diffs)])


def _uniform_motion_residual(rows: np.ndarray, cols: np.ndarray) -> float:
    """RMS distance of a 2D trajectory from its best constant-velocity fit."""
    t = np.arange(len(rows), dtype=float)
    design = np.vstack([np.ones_like(t), t]).T
    coef_r, _, _, _ = np.linalg.lstsq(design, rows, rcond=None)
    coef_c, _, _, _ = np.linalg.lstsq(design, cols, rcond=None)
    resid_r = rows - design @ coef_r
    resid_c = cols - design @ coef_c
    return float(np.sqrt(np.mean(resid_r ** 2 + resid_c ** 2)))


def measure_stability(
    seed_field: np.ndarray,
    kernel_spectrum: np.ndarray,
    dt: float,
    mu: float,
    sigma: float,
    settle: int,
    window: int,
) -> StabilityResult:
    """Measure an organism's centroid defect and survival at growth (mu, sigma).

    The seed is first evolved `settle` steps (to relax onto the organism or to
    begin dissolving), then its centroid is tracked for `window` further steps.
    The defect is the trajectory's deviation from uniform motion; the survival
    is the log10 mass ratio over the whole run.
    """
    grid_size = seed_field.shape[0]
    initial_mass = total_mass(seed_field)

    field = seed_field
    for _ in range(settle):
        field = step(field, kernel_spectrum, mu, sigma, dt)

    rows = np.empty(window + 1)
    cols = np.empty(window + 1)
    rows[0], cols[0] = center_of_mass(field)
    for i in range(1, window + 1):
        field = step(field, kernel_spectrum, mu, sigma, dt)
        rows[i], cols[i] = center_of_mass(field)

    final_mass = total_mass(field)
    log_mass_ratio = float(np.log10(max(final_mass, 1e-9) / max(initial_mass, 1e-9)))

    rows_u = _unwrap_periodic(rows, grid_size)
    cols_u = _unwrap_periodic(cols, grid_size)
    defect = _uniform_motion_residual(rows_u, cols_u)
    return StabilityResult(defect, log_mass_ratio)
