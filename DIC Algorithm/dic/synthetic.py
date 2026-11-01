"""Synthetic speckle images with known deformation, for validating the DIC.

Speckles are analytic Gaussian blobs, so a deformed image is rendered exactly
(no interpolation of a reference bitmap), giving an unbiased ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

DisplacementFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class SpecklePattern:
    def __init__(self, width: int, height: int, radius: float = 2.5, coverage: float = 0.55,
                 margin: int = 60, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.width, self.height, self.radius = width, height, radius
        area = (width + 2 * margin) * (height + 2 * margin)
        n = int(coverage * area / (np.pi * radius**2))
        self.xy = np.column_stack([
            rng.uniform(-margin, width + margin, n),
            rng.uniform(-margin, height + margin, n),
        ])
        self.r = radius * rng.uniform(0.7, 1.3, n)
        self.a = rng.uniform(0.6, 1.0, n)
        # Bin speckles into cells so each pixel only sums nearby speckles.
        self.cell = 3.5 * radius * 1.3
        self.x_min, self.y_min = -margin, -margin
        cx = ((self.xy[:, 0] - self.x_min) // self.cell).astype(int)
        cy = ((self.xy[:, 1] - self.y_min) // self.cell).astype(int)
        self.ncx, self.ncy = cx.max() + 1, cy.max() + 1
        cid = cy * self.ncx + cx
        order = np.argsort(cid)
        counts = np.bincount(cid, minlength=self.ncx * self.ncy)
        K = counts.max()
        self.table = np.full((self.ncy * self.ncx, K), -1, dtype=int)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for c in np.flatnonzero(counts):
            self.table[c, : counts[c]] = order[starts[c] : starts[c] + counts[c]]

    def intensity(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Pattern intensity in [0, ~1] at reference-configuration coordinates."""
        shp = X.shape
        X, Y = X.ravel(), Y.ravel()
        out = np.zeros(X.size)
        cx = ((X - self.x_min) // self.cell).astype(int)
        cy = ((Y - self.y_min) // self.cell).astype(int)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                ncx, ncy = cx + ox, cy + oy
                ok = (ncx >= 0) & (ncy >= 0) & (ncx < self.ncx) & (ncy < self.ncy)
                cells = np.where(ok, ncy * self.ncx + ncx, 0)
                ids = self.table[cells]                       # (P, K)
                ids = np.where(ok[:, None], ids, -1)
                valid = ids >= 0
                ids_c = np.where(valid, ids, 0)
                d2 = (X[:, None] - self.xy[ids_c, 0]) ** 2 + (Y[:, None] - self.xy[ids_c, 1]) ** 2
                out += np.sum(np.where(valid, self.a[ids_c] * np.exp(-d2 / self.r[ids_c] ** 2), 0.0), axis=1)
        return np.clip(out, 0, 1.2).reshape(shp)

    def render(self, displacement: DisplacementFn | None = None, noise: float = 0.0,
               seed: int | None = None, bit_depth: int = 8) -> np.ndarray:
        """Image of the pattern after deformation x = X + u(X).

        displacement(X, Y) -> (u, v) is defined in reference coordinates; it is
        inverted by fixed-point iteration to evaluate the pattern at each pixel.
        """
        y, x = np.mgrid[0 : self.height, 0 : self.width].astype(np.float64)
        X, Y = x.copy(), y.copy()
        if displacement is not None:
            for _ in range(100):
                u, v = displacement(X, Y)
                Xn, Yn = x - u, y - v
                delta = max(np.abs(Xn - X).max(), np.abs(Yn - Y).max())
                X, Y = Xn, Yn
                if delta < 1e-10:
                    break
        img = 20 + 200 * self.intensity(X, Y)
        if noise > 0:
            img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
        levels = 2**bit_depth - 1
        return np.clip(np.round(img), 0, levels).astype(np.uint8 if bit_depth <= 8 else np.uint16)


def uniaxial(strain_x: float, poisson: float = 0.3, center=(0.0, 0.0), shift=(0.0, 0.0)) -> DisplacementFn:
    """Homogeneous uniaxial stretch about `center` plus rigid shift (engineering strain)."""
    def fn(X, Y):
        return (strain_x * (X - center[0]) + shift[0], -poisson * strain_x * (Y - center[1]) + shift[1])
    return fn


def write_sequence(folder: str | Path, frames: list[np.ndarray]) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(folder / f"frame_{i:04d}.png"), f)
    return folder


def write_video(path: str | Path, frames: list[np.ndarray], fps: float = 10.0) -> Path:
    """Write frames to a (lossy) mp4 — handy for trying the video workflow."""
    path = Path(path)
    h, w = frames[0].shape
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)
    for f in frames:
        vw.write(f)
    vw.release()
    return path
