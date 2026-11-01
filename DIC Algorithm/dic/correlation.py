"""Subset-based 2D DIC correlation engine.

Implements the inverse-compositional Gauss-Newton (IC-GN) algorithm with a
first-order shape function and the zero-normalised sum of squared differences
(ZNSSD) criterion, vectorised across all subsets at once. Grey levels at
sub-pixel positions come from bicubic B-spline interpolation.

Shape-function parameters per subset: p = [u, du/dx, du/dy, v, dv/dx, dv/dy].
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

# 5-tap central-difference kernel (4th-order accurate) for image gradients.
_DERIV_KERNEL = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0


class SplineImage:
    """Bicubic B-spline interpolant of a greyscale image."""

    def __init__(self, img: np.ndarray):
        self.img = np.asarray(img, dtype=np.float64)
        self.shape = self.img.shape
        self._coeffs = ndimage.spline_filter(self.img, order=3, output=np.float64, mode="mirror")

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        coords = np.stack([np.ravel(y), np.ravel(x)])
        vals = ndimage.map_coordinates(self._coeffs, coords, order=3, prefilter=False, mode="mirror")
        return vals.reshape(np.shape(x))


class ReferenceImage:
    """Reference image with interpolants for intensity and its gradients."""

    def __init__(self, img: np.ndarray):
        img = np.asarray(img, dtype=np.float64)
        gx = ndimage.correlate1d(img, _DERIV_KERNEL, axis=1, mode="mirror")
        gy = ndimage.correlate1d(img, _DERIV_KERNEL, axis=0, mode="mirror")
        self.img = img
        self.shape = img.shape
        self.f = SplineImage(img)
        self.fx = SplineImage(gx)
        self.fy = SplineImage(gy)


@dataclass
class CorrelationResult:
    p: np.ndarray          # (N, 6) shape-function parameters
    zncc: np.ndarray       # (N,) zero-normalised cross-correlation, 1 = perfect
    iterations: np.ndarray  # (N,) IC-GN iterations used
    valid: np.ndarray      # (N,) bool


def subset_offsets(subset_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Local (dx, dy) coordinates of every pixel in a square subset."""
    if subset_size % 2 == 0 or subset_size < 5:
        raise ValueError("subset_size must be an odd integer >= 5")
    h = subset_size // 2
    d = np.arange(-h, h + 1, dtype=np.float64)
    dy, dx = np.meshgrid(d, d, indexing="ij")
    return dx.ravel(), dy.ravel()


def compose_inverse(p: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """IC-GN warp update W(p) <- W(p) o W(dp)^-1, batched over rows."""

    def to_mat(q):
        m = np.zeros((q.shape[0], 3, 3))
        m[:, 0, 0] = 1 + q[:, 1]
        m[:, 0, 1] = q[:, 2]
        m[:, 0, 2] = q[:, 0]
        m[:, 1, 0] = q[:, 4]
        m[:, 1, 1] = 1 + q[:, 5]
        m[:, 1, 2] = q[:, 3]
        m[:, 2, 2] = 1
        return m

    w = to_mat(p) @ np.linalg.inv(to_mat(dp))
    return np.stack(
        [w[:, 0, 2], w[:, 0, 0] - 1, w[:, 0, 1], w[:, 1, 2], w[:, 1, 0], w[:, 1, 1] - 1], axis=1
    )


def integer_search(
    ref_img: np.ndarray,
    cur_img: np.ndarray,
    centers: np.ndarray,
    subset_size: int,
    search_radius: int,
    guess: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-level displacement by normalised cross-correlation template matching.

    Returns (uv (N, 2), peak_ncc (N,)). Points whose template or search window
    falls outside the image get NaN.
    """
    h = subset_size // 2
    H, W = cur_img.shape
    ref32 = ref_img.astype(np.float32)
    cur32 = cur_img.astype(np.float32)
    n = len(centers)
    uv = np.full((n, 2), np.nan)
    peak = np.full(n, np.nan)
    if guess is None:
        guess = np.zeros((n, 2))
    for i, (cx, cy) in enumerate(centers):
        if not (np.isfinite(cx) and np.isfinite(cy)) or not np.all(np.isfinite(guess[i])):
            continue
        xi, yi = int(round(cx)), int(round(cy))
        if xi - h < 0 or yi - h < 0 or xi + h >= W or yi + h >= H:
            continue
        tmpl = ref32[yi - h : yi + h + 1, xi - h : xi + h + 1]
        gx, gy = int(round(guess[i, 0])), int(round(guess[i, 1]))
        x0 = max(xi + gx - h - search_radius, 0)
        y0 = max(yi + gy - h - search_radius, 0)
        x1 = min(xi + gx + h + search_radius + 1, W)
        y1 = min(yi + gy + h + search_radius + 1, H)
        win = cur32[y0:y1, x0:x1]
        if win.shape[0] < subset_size or win.shape[1] < subset_size:
            continue
        res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, maxloc = cv2.minMaxLoc(res)
        uv[i, 0] = (x0 + maxloc[0] + h) - xi
        uv[i, 1] = (y0 + maxloc[1] + h) - yi
        peak[i] = maxval
    return uv, peak


def icgn(
    ref: ReferenceImage,
    cur: SplineImage,
    centers: np.ndarray,
    p0: np.ndarray,
    subset_size: int,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> CorrelationResult:
    """Sub-pixel refinement of all subsets simultaneously with IC-GN / ZNSSD.

    centers: (N, 2) subset centres (x, y) in the reference image (may be sub-pixel).
    p0:      (N, 6) initial guess of shape-function parameters.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    dx, dy = subset_offsets(subset_size)
    h = subset_size // 2
    Hh, Ww = cur.shape

    p = np.array(p0, dtype=np.float64, copy=True)
    valid = np.all(np.isfinite(centers), axis=1) & np.all(np.isfinite(p), axis=1)
    zncc = np.full(n, np.nan)
    iters = np.zeros(n, dtype=int)

    # Reference subsets must lie fully inside the reference image.
    cx, cy = centers[:, 0], centers[:, 1]
    inside = (cx - h >= 0) & (cy - h >= 0) & (cx + h <= ref.shape[1] - 1) & (cy + h <= ref.shape[0] - 1)
    valid &= np.nan_to_num(inside, nan=False).astype(bool)
    idx_all = np.flatnonzero(valid)
    if idx_all.size == 0:
        return CorrelationResult(np.full((n, 6), np.nan), zncc, iters, valid)

    # --- Pre-computation on the reference subsets (the "inverse" in IC-GN) ---
    X = centers[idx_all, 0, None] + dx
    Y = centers[idx_all, 1, None] + dy
    f = ref.f(X, Y)
    fx = ref.fx(X, Y)
    fy = ref.fy(X, Y)
    fc = f - f.mean(axis=1, keepdims=True)
    df = np.sqrt(np.sum(fc**2, axis=1, keepdims=True))
    J = np.stack([fx, fx * dx, fx * dy, fy, fy * dx, fy * dy], axis=-1)  # (n, M, 6)
    Hess = np.einsum("nmi,nmj->nij", J, J)

    # Reject texture-less subsets (flat intensity or ill-conditioned Hessian).
    cond = np.linalg.cond(Hess)
    ok = (df[:, 0] > 1e-6) & np.isfinite(cond) & (cond < 1e12)
    Hinv = np.zeros_like(Hess)
    Hinv[ok] = np.linalg.inv(Hess[ok])

    local = np.arange(idx_all.size)
    active = ok.copy()
    loc_valid = ok.copy()
    scale = np.array([1.0, h, h, 1.0, h, h])

    def sample_current(li):
        q = p[idx_all[li]]
        xw = centers[idx_all[li], 0, None] + q[:, 0, None] + (1 + q[:, 1, None]) * dx + q[:, 2, None] * dy
        yw = centers[idx_all[li], 1, None] + q[:, 3, None] + q[:, 4, None] * dx + (1 + q[:, 5, None]) * dy
        in_img = (xw.min(1) >= 0) & (yw.min(1) >= 0) & (xw.max(1) <= Ww - 1) & (yw.max(1) <= Hh - 1)
        return cur(xw, yw), in_img

    for it in range(1, max_iter + 1):
        li = local[active]
        if li.size == 0:
            break
        g, in_img = sample_current(li)
        # Subsets warped out of the image are abandoned.
        if not np.all(in_img):
            gone = li[~in_img]
            loc_valid[gone] = False
            active[gone] = False
            li, g = li[in_img], g[in_img]
            if li.size == 0:
                break
        gc = g - g.mean(axis=1, keepdims=True)
        dg = np.sqrt(np.sum(gc**2, axis=1, keepdims=True))
        dg = np.where(dg < 1e-12, np.nan, dg)
        resid = fc[li] - (df[li] / dg) * gc
        b = np.einsum("nmi,nm->ni", J[li], resid)
        dp = -np.einsum("nij,nj->ni", Hinv[li], b)
        p[idx_all[li]] = compose_inverse(p[idx_all[li]], dp)
        iters[idx_all[li]] = it

        step = np.sqrt(np.sum((dp * scale) ** 2, axis=1))
        bad = ~np.isfinite(step)
        loc_valid[li[bad]] = False
        active[li[bad]] = False
        active[li[step < tol]] = False

    # --- Final correlation quality at converged parameters ---
    li = local[loc_valid]
    if li.size:
        g, in_img = sample_current(li)
        gc = g - g.mean(axis=1, keepdims=True)
        dg = np.sqrt(np.sum(gc**2, axis=1, keepdims=True))
        znssd = np.sum((fc[li] / df[li] - gc / np.where(dg < 1e-12, np.nan, dg)) ** 2, axis=1)
        z = 1 - 0.5 * znssd
        zncc[idx_all[li]] = np.where(in_img, z, np.nan)
        loc_valid[li[~in_img | ~np.isfinite(z)]] = False
    # Points still "active" hit max_iter without converging.
    loc_valid[active] = False

    valid[idx_all] = loc_valid
    p[~valid] = np.nan
    zncc[~valid] = np.nan
    return CorrelationResult(p, zncc, iters, valid)
