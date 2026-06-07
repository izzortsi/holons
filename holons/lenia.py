"""Lenia: a continuous cellular automaton (Chan 2019), pure numpy.

The state A(x) is a float field in [0, 1] on a periodic (torus) grid. One
update is

    A <- clip(A + dt * G(K * A), 0, 1)

where K is a radial kernel (a Gaussian shell), `*` is circular convolution
(computed via the FFT, so the boundary is periodic), and G is the growth
function G(u) = 2 * bell(u, mu, sigma) - 1, a Gaussian bump mapped to
[-1, 1]. The canonical Orbium organism lives at the parameters in `ORBIUM`.

This module is the continuous-substrate counterpart to life.py. It supplies
the dynamics and the mass/centroid observables used to coarse-grain a
localized organism to a moving point; the organism seed itself is supplied by
the caller (no seed is fabricated here).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class LeniaParams(NamedTuple):
    radius: int        # kernel radius R in cells
    dt: float          # time step (1 / T)
    mu: float          # growth centre
    sigma: float       # growth width
    kernel_peak: float    # radial position of the kernel shell peak, in [0, 1]
    kernel_width: float   # radial width of the kernel shell


# Orbium bicaudatus parameters (Chan's Lenia), as supplied with the seed by
# the project author. The seed cell array lives in orbium.py; these are only
# the dynamics parameters.
ORBIUM: LeniaParams = LeniaParams(
    radius=13, dt=0.1, mu=0.15, sigma=0.014, kernel_peak=0.5, kernel_width=0.15
)


def bell(x: np.ndarray, mean: float, width: float) -> np.ndarray:
    """Return the unnormalised Gaussian bump exp(-((x-mean)/width)^2 / 2)."""
    return np.exp(-0.5 * ((x - mean) / width) ** 2)


def kernel_fft(params: LeniaParams, grid_size: int) -> np.ndarray:
    """Return the rfft2 of the normalised Lenia kernel for a square grid.

    The kernel is a radial Gaussian shell of radius `params.radius`, placed on
    a `grid_size`-by-`grid_size` field, normalised to sum 1, and rolled so its
    centre sits at index (0, 0) ready for circular convolution via the FFT.
    """
    if grid_size < 2 * params.radius + 1:
        raise ValueError(
            f"grid_size {grid_size} too small for kernel radius {params.radius}"
        )
    axis = np.arange(grid_size) - grid_size // 2
    xx, yy = np.meshgrid(axis, axis)
    distance = np.sqrt(xx ** 2 + yy ** 2) / params.radius
    shell = (distance < 1.0) * bell(distance, params.kernel_peak, params.kernel_width)
    total = shell.sum()
    if total == 0.0:
        raise ValueError("kernel is identically zero; check parameters")
    shell = shell / total
    # Move the kernel centre (grid_size//2, grid_size//2) to the origin.
    shell = np.roll(shell, (-(grid_size // 2), -(grid_size // 2)), axis=(0, 1))
    return np.fft.rfft2(shell)


def growth(potential: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Return the Lenia growth G(u) = 2 * bell(u, mu, sigma) - 1, in [-1, 1]."""
    return 2.0 * bell(potential, mu, sigma) - 1.0


def step(
    field: np.ndarray,
    kernel_spectrum: np.ndarray,
    mu: float,
    sigma: float,
    dt: float,
) -> np.ndarray:
    """Apply one Lenia update to `field` with a precomputed kernel spectrum.

    `kernel_spectrum` is `kernel_fft(params, grid_size)` for the same grid.
    Returns a new float64 field clipped to [0, 1]; the input is not modified.
    """
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError(f"field must be a square 2D array, got {field.shape}")
    potential = np.fft.irfft2(np.fft.rfft2(field) * kernel_spectrum, field.shape)
    return np.clip(field + dt * growth(potential, mu, sigma), 0.0, 1.0)


def evolve(
    field: np.ndarray,
    kernel_spectrum: np.ndarray,
    mu: float,
    sigma: float,
    dt: float,
    steps: int,
) -> np.ndarray:
    """Apply `step` `steps` times. Returns a new field; input unmodified."""
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    result = field
    for _ in range(steps):
        result = step(result, kernel_spectrum, mu, sigma, dt)
    return result


def total_mass(field: np.ndarray) -> float:
    """Return the total activation mass (sum of the field)."""
    return float(field.sum())


def center_of_mass(field: np.ndarray) -> tuple[float, float]:
    """Return the mass-weighted (row, col) centroid on the torus.

    Uses a circular mean per axis so the centroid is correct even when the
    organism straddles the periodic boundary. Returns (0.0, 0.0) for an empty
    field.
    """
    mass = field.sum()
    if mass == 0.0:
        return (0.0, 0.0)
    rows, cols = field.shape
    return (
        _circular_mean(field.sum(axis=1), rows),
        _circular_mean(field.sum(axis=0), cols),
    )


def _circular_mean(marginal: np.ndarray, n: int) -> float:
    """Circular mean position of a 1D mass marginal over a period-n axis."""
    angle = 2.0 * np.pi * np.arange(n) / n
    c = float((marginal * np.cos(angle)).sum())
    s = float((marginal * np.sin(angle)).sum())
    mean_angle = np.arctan2(s, c) % (2.0 * np.pi)
    return mean_angle * n / (2.0 * np.pi)
