"""Frame-by-frame DIC tracking over a video or image sequence."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .correlation import ReferenceImage, SplineImage, icgn, integer_search
from .strain import StrainField, compute_strain
from .video import FrameSource


@dataclass
class DICSettings:
    subset_size: int = 31          # odd, pixels per side of each correlation subset
    step: int = 10                 # grid spacing between subset centres (pixels)
    search_radius: int = 30        # pixel search range for initial guesses
    max_iter: int = 50             # IC-GN iteration limit
    tol: float = 1e-3              # IC-GN convergence tolerance on |dp|
    zncc_threshold: float = 0.7    # points below this are rejected (NaN)
    reference: str = "auto"        # "first", "auto" (update when correlation degrades), "incremental"
    update_threshold: float = 0.9  # mean ZNCC that triggers a reference update in "auto" mode
    prefilter_sigma: float = 0.0   # Gaussian pre-smoothing of frames (pixels), 0 = off
    strain_window: int = 5         # odd, grid points per side of the strain plane fit
    strain_kind: str = "green-lagrange"
    pixel_size: float = 1.0        # physical units per pixel (e.g. mm/px) for displacement output
    units: str = "px"


@dataclass
class DICResult:
    settings: DICSettings
    grid_x: np.ndarray             # (nx,) subset-centre x positions in the first frame (px)
    grid_y: np.ndarray             # (ny,)
    frames: np.ndarray             # (F,) source frame indices
    times: np.ndarray              # (F,) seconds (NaN if unknown)
    u: np.ndarray                  # (F, ny, nx) displacement in x (px), relative to first frame
    v: np.ndarray                  # (F, ny, nx)
    zncc: np.ndarray               # (F, ny, nx)
    reference_frames: list = field(default_factory=list)
    _strain_cache: dict = field(default_factory=dict, repr=False)

    @property
    def shape(self):
        return self.u.shape[1:]

    def strain(self, k: int, window: int | None = None, kind: str | None = None) -> StrainField:
        """Strain field for the k-th processed frame (cached)."""
        window = window or self.settings.strain_window
        kind = kind or self.settings.strain_kind
        key = (k, window, kind)
        if key not in self._strain_cache:
            self._strain_cache[key] = compute_strain(self.u[k], self.v[k], self.settings.step, window, kind)
        return self._strain_cache[key]

    def strain_history(self, window: int | None = None, kind: str | None = None) -> dict[str, np.ndarray]:
        """Stack all frames: dict of (F, ny, nx) arrays for exx, eyy, exy, e1, e2, von_mises."""
        names = ["exx", "eyy", "exy", "e1", "e2", "von_mises"]
        out = {n: np.empty_like(self.u) for n in names}
        for k in range(len(self.frames)):
            s = self.strain(k, window, kind)
            for n in names:
                out[n][k] = getattr(s, n)
        return out

    def save(self, path: str | Path) -> None:
        strains = self.strain_history()
        np.savez_compressed(
            path,
            settings=np.array([asdict(self.settings)], dtype=object),
            grid_x=self.grid_x, grid_y=self.grid_y, frames=self.frames, times=self.times,
            u=self.u, v=self.v, zncc=self.zncc,
            reference_frames=np.array(self.reference_frames),
            **strains,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DICResult":
        d = np.load(path, allow_pickle=True)
        return cls(
            DICSettings(**d["settings"][0]), d["grid_x"], d["grid_y"], d["frames"], d["times"],
            d["u"], d["v"], d["zncc"], list(d["reference_frames"]),
        )


def make_grid(shape, roi, subset_size, step, mask=None):
    """Subset centres on a regular grid inside roi=(x, y, w, h)."""
    H, W = shape
    x, y, w, h = roi if roi is not None else (0, 0, W, H)
    half = subset_size // 2
    x0, y0 = max(x, half), max(y, half)
    x1, y1 = min(x + w - 1, W - 1 - half), min(y + h - 1, H - 1 - half)
    gx = np.arange(x0, x1 + 1, step, dtype=np.float64)
    gy = np.arange(y0, y1 + 1, step, dtype=np.float64)
    if gx.size == 0 or gy.size == 0:
        raise ValueError("ROI too small for the chosen subset size")
    XX, YY = np.meshgrid(gx, gy)
    active = np.ones(XX.shape, bool)
    if mask is not None:
        active = mask[YY.astype(int), XX.astype(int)].astype(bool)
    return gx, gy, XX, YY, active


def _fill_nearest(field2d: np.ndarray) -> np.ndarray:
    """Replace NaNs by the nearest valid grid value."""
    bad = ~np.isfinite(field2d)
    if not bad.any() or bad.all():
        return field2d
    _, (iy, ix) = ndimage.distance_transform_edt(bad, return_indices=True)
    return field2d[iy, ix]


def _prep(img, sigma):
    return ndimage.gaussian_filter(img, sigma) if sigma > 0 else img


def run_dic(
    source: FrameSource,
    settings: DICSettings | None = None,
    roi: tuple[int, int, int, int] | None = None,
    mask: np.ndarray | None = None,
    verbose: bool = True,
) -> DICResult:
    s = settings or DICSettings()
    frames_iter = iter(source)
    try:
        f0_idx, f0 = next(frames_iter)
    except StopIteration:
        raise ValueError("no frames to process") from None
    f0 = _prep(f0, s.prefilter_sigma)

    gx, gy, XX, YY, active = make_grid(f0.shape, roi, s.subset_size, s.step, mask)
    ny, nx = XX.shape
    X0 = np.column_stack([XX.ravel(), YY.ravel()])
    X0[~active.ravel()] = np.nan
    N = len(X0)

    ref = ReferenceImage(f0)
    ref_raw, ref_idx = f0, f0_idx
    U_ref = np.zeros((N, 2))
    ref_centers = X0.copy()
    p_guess = np.zeros((N, 6))
    need_search = np.ones(N, bool)

    zero = np.where(active, 0.0, np.nan)
    us, vs, zs = [zero.copy()], [zero.copy()], [np.where(active, 1.0, np.nan)]
    frame_ids, ref_frames = [f0_idx], [f0_idx]
    t_start = time.perf_counter()

    for idx, img in frames_iter:
        img = _prep(img, s.prefilter_sigma)
        cur = SplineImage(img)

        def solve(points, p_init, search, radius):
            p0 = p_init.copy()
            if search.any():
                uv, _ = integer_search(ref_raw, img, ref_centers[points][search], s.subset_size,
                                       radius, p0[search][:, [0, 3]])
                p0[np.flatnonzero(search), :] = 0.0
                p0[np.flatnonzero(search), 0] = uv[:, 0]
                p0[np.flatnonzero(search), 3] = uv[:, 1]
            return icgn(ref, cur, ref_centers[points], p0, s.subset_size, s.max_iter, s.tol)

        all_pts = np.arange(N)
        res = solve(all_pts, p_guess, need_search, s.search_radius)
        p, zncc = res.p, res.zncc
        bad = ~res.valid | ~(zncc >= s.zncc_threshold)
        bad &= np.all(np.isfinite(ref_centers), axis=1)

        # Retry failures with a fresh pixel-level search around the last good guess.
        if bad.any():
            pts = np.flatnonzero(bad)
            guess = np.nan_to_num(p_guess[pts])
            retry = solve(pts, guess, np.ones(len(pts), bool), 2 * s.search_radius)
            better = retry.valid & (retry.zncc >= s.zncc_threshold)
            p[pts[better]] = retry.p[better]
            zncc[pts[better]] = retry.zncc[better]
            bad[pts[better]] = False

        p[bad] = np.nan
        zncc[bad] = np.nan
        U = U_ref + p[:, [0, 3]]
        us.append(U[:, 0].reshape(ny, nx))
        vs.append(U[:, 1].reshape(ny, nx))
        zs.append(zncc.reshape(ny, nx))
        frame_ids.append(idx)

        good = ~bad
        p_guess[good] = p[good]
        need_search = ~np.all(np.isfinite(p_guess), axis=1)

        mean_z = np.nanmean(zncc) if np.any(np.isfinite(zncc)) else 0.0
        update = s.reference == "incremental" or (s.reference == "auto" and mean_z < s.update_threshold)
        if update:
            Uf = np.column_stack([
                _fill_nearest(U[:, 0].reshape(ny, nx)).ravel(),
                _fill_nearest(U[:, 1].reshape(ny, nx)).ravel(),
            ])
            Uf[~active.ravel()] = np.nan
            U_ref = Uf
            ref_centers = X0 + U_ref
            ref, ref_raw, ref_idx = ReferenceImage(img), img, idx
            p_guess = np.zeros((N, 6))
            need_search = ~np.all(np.isfinite(ref_centers), axis=1)
            ref_frames.append(idx)

        if verbose:
            n_ok = int(np.sum(np.isfinite(zncc)))
            print(f"frame {idx:5d}  valid {n_ok}/{int(active.sum())}  mean ZNCC {mean_z:.4f}"
                  f"{'  [reference updated]' if update and s.reference == 'auto' else ''}"
                  f"  ({time.perf_counter() - t_start:.1f}s)", flush=True)

    frames = np.array(frame_ids)
    times = frames / source.fps if source.fps else np.full(len(frames), np.nan)
    return DICResult(s, gx, gy, frames, times, np.array(us), np.array(vs), np.array(zs), ref_frames)
