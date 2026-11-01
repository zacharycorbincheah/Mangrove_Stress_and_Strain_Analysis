"""Strain calculation from gridded displacement fields.

Displacement gradients are estimated by a local least-squares plane fit over a
square window of grid points (pointwise least squares, Pan et al. 2009). The fit
is done with normalised convolution, so missing (NaN) points are simply
excluded from each window rather than contaminating their neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class StrainField:
    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray   # tensorial shear strain (engineering shear = 2 * exy)
    e1: np.ndarray    # major principal strain
    e2: np.ndarray    # minor principal strain
    von_mises: np.ndarray
    dudx: np.ndarray
    dudy: np.ndarray
    dvdx: np.ndarray
    dvdy: np.ndarray


def plane_fit_gradient(field: np.ndarray, spacing: float, window: int, min_fraction: float = 0.6):
    """Return (d/dx, d/dy) of a gridded field via local weighted plane fits.

    field:   (ny, nx) values, NaN where missing.
    spacing: grid pitch in pixels (or physical units).
    window:  odd number of grid points per side of the fitting window.
    """
    if window % 2 == 0 or window < 3:
        raise ValueError("strain window must be an odd integer >= 3")
    h = window // 2
    k = np.arange(-h, h + 1, dtype=np.float64) * spacing
    ky, kx = np.meshgrid(k, k, indexing="ij")

    w = np.isfinite(field).astype(np.float64)
    z = np.where(w > 0, field, 0.0)

    def corr(a, kern):
        return ndimage.correlate(a, kern, mode="constant", cval=0.0)

    one = np.ones_like(kx)
    s1, sx, sy = corr(w, one), corr(w, kx), corr(w, ky)
    sxx, sxy, syy = corr(w, kx * kx), corr(w, kx * ky), corr(w, ky * ky)
    sz, sxz, syz = corr(z, one), corr(z, kx), corr(z, ky)

    A = np.stack(
        [np.stack([s1, sx, sy], -1), np.stack([sx, sxx, sxy], -1), np.stack([sy, sxy, syy], -1)], -2
    )
    rhs = np.stack([sz, sxz, syz], -1)

    enough = (s1 >= min_fraction * window * window) & np.isfinite(field)
    det = np.linalg.det(A)
    enough &= np.abs(det) > 1e-9 * np.maximum(np.abs(A).max(axis=(-1, -2)), 1) ** 3

    gx = np.full(field.shape, np.nan)
    gy = np.full(field.shape, np.nan)
    if np.any(enough):
        sol = np.linalg.solve(A[enough], rhs[enough][..., None])[..., 0]
        gx[enough] = sol[:, 1]
        gy[enough] = sol[:, 2]
    return gx, gy


def compute_strain(
    u: np.ndarray, v: np.ndarray, spacing: float, window: int = 5, kind: str = "green-lagrange"
) -> StrainField:
    """Strain tensor fields from displacement fields u, v on a regular grid.

    kind: "green-lagrange" (finite strain, reference configuration) or
          "engineering" (infinitesimal / Cauchy strain).
    """
    dudx, dudy = plane_fit_gradient(u, spacing, window)
    dvdx, dvdy = plane_fit_gradient(v, spacing, window)

    if kind == "green-lagrange":
        exx = dudx + 0.5 * (dudx**2 + dvdx**2)
        eyy = dvdy + 0.5 * (dudy**2 + dvdy**2)
        exy = 0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy)
    elif kind == "engineering":
        exx, eyy, exy = dudx, dvdy, 0.5 * (dudy + dvdx)
    else:
        raise ValueError(f"unknown strain kind: {kind!r}")

    centre = 0.5 * (exx + eyy)
    radius = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy**2)
    e1, e2 = centre + radius, centre - radius
    # Von Mises equivalent strain assuming incompressibility (e3 = -(e1 + e2)).
    von_mises = (2.0 / np.sqrt(3.0)) * np.sqrt(e1**2 + e2**2 + e1 * e2)
    return StrainField(exx, eyy, exy, e1, e2, von_mises, dudx, dudy, dvdx, dvdy)
