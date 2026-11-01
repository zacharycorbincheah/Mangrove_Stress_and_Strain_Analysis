#!/usr/bin/env python3
"""Synthetic tensile test: generate a speckle video with a known necking strain
field, run the DIC pipeline on it, and report the error against ground truth.

    python examples/synthetic_tensile_demo.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dic.pipeline import DICSettings, run_dic  # noqa: E402
from dic.synthetic import SpecklePattern, write_sequence, write_video  # noqa: E402
from dic.video import FrameSource  # noqa: E402
from dic.visualize import save_summary_figures, write_overlay_video, write_summary_csv  # noqa: E402

W, H = 480, 320
CX, CY = W / 2, H / 2
N_FRAMES = 31
NU = 0.4
SIGMA = 45.0  # width of the neck (px)


def neck(eps_far: float, eps_peak: float):
    """u, v and exact displacement gradients of a localised tensile field."""
    A = eps_peak - eps_far
    k = SIGMA * np.sqrt(np.pi) / 2

    def disp(X, Y):
        exx = eps_far + A * np.exp(-((X - CX) / SIGMA) ** 2)
        u = eps_far * (X - CX) + A * k * erf((X - CX) / SIGMA)
        v = -NU * exx * (Y - CY)
        return u, v

    def grads(X, Y):
        g = np.exp(-((X - CX) / SIGMA) ** 2)
        dudx = eps_far + A * g
        dvdy = -NU * dudx
        dvdx = -NU * A * g * (-2 * (X - CX) / SIGMA**2) * (Y - CY)
        return dudx, np.zeros_like(X), dvdx, dvdy

    return disp, grads


def schedule(i):
    t = i / (N_FRAMES - 1)
    return 0.08 * t, 0.08 * t + 0.17 * t**2  # far-field and peak engineering strain


def main():
    out = Path(__file__).resolve().parent / "synthetic_output"
    out.mkdir(exist_ok=True)
    sp = SpecklePattern(W, H, radius=2.2, seed=3)
    print("Rendering synthetic frames...")
    frames = [sp.render(neck(*schedule(i))[0], noise=1.5, seed=i) for i in range(N_FRAMES)]
    folder = write_sequence(out / "frames", frames)
    write_video(out / "tensile.mp4", frames, fps=10)

    settings = DICSettings(subset_size=29, step=8, strain_window=5, reference="auto")
    src = FrameSource(folder)
    roi = (60, 40, W - 120, H - 80)
    result = run_dic(src, settings, roi=roi)

    # Compare against analytic Green-Lagrange strain at the tracked points.
    XX, YY = np.meshgrid(result.grid_x, result.grid_y)
    print("\nframe   true peak exx   DIC peak exx   RMS err exx   RMS err eyy   disp RMS err (px)")
    for k in range(0, N_FRAMES, 5):
        disp, grads = neck(*schedule(k))
        ux, uy, vx, vy = grads(XX, YY)
        exx_t = ux + 0.5 * (ux**2 + vx**2)
        eyy_t = vy + 0.5 * (uy**2 + vy**2)
        u_t, v_t = disp(XX, YY)
        s = result.strain(k)
        ok = np.isfinite(s.exx)
        derr = np.sqrt(np.nanmean((result.u[k] - u_t) ** 2 + (result.v[k] - v_t) ** 2))
        print(f"{k:5d}   {exx_t.max():12.4f}   {np.nanmax(s.exx):12.4f}   "
              f"{np.sqrt(np.mean((s.exx[ok] - exx_t[ok]) ** 2)):11.2e}   "
              f"{np.sqrt(np.mean((s.eyy[ok] - eyy_t[ok]) ** 2)):11.2e}   {derr:10.4f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result.save(out / "results.npz")
        write_summary_csv(result, out / "strain_summary.csv")
        save_summary_figures(result, src, out)
        write_overlay_video(result, src, out / "overlay_exx.mp4", "exx", fps=10)
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
