#!/usr/bin/env python3
"""Run 2D DIC strain analysis on a video file or image-sequence folder.

Examples:
    python run_dic.py test.mp4 --select-roi
    python run_dic.py test.mp4 --roi 100 50 400 300 --subset 31 --step 8 --pixel-size 0.02 --units mm
    python run_dic.py frames/ --roi 100 50 400 300 --reference first --component eyy
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np

from dic.pipeline import DICSettings, run_dic
from dic.video import FrameSource, select_roi
from dic.visualize import STRAINS, save_summary_figures, write_overlay_video, write_summary_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="video file or folder of images")
    ap.add_argument("-o", "--out", default=None, help="output folder (default: <input>_dic)")
    g = ap.add_argument_group("region of interest")
    g.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    g.add_argument("--select-roi", action="store_true", help="draw the ROI on the first frame")
    g.add_argument("--mask", help="binary image; grid points on black pixels are ignored")
    g = ap.add_argument_group("frames")
    g.add_argument("--start", type=int, default=0)
    g.add_argument("--stop", type=int, default=None)
    g.add_argument("--every", type=int, default=1, help="process every n-th frame")
    g = ap.add_argument_group("correlation")
    d = DICSettings()
    g.add_argument("--subset", type=int, default=d.subset_size, help="subset size in px (odd)")
    g.add_argument("--step", type=int, default=d.step, help="grid spacing in px")
    g.add_argument("--search", type=int, default=d.search_radius, help="initial-guess search radius in px")
    g.add_argument("--reference", choices=["auto", "first", "incremental"], default=d.reference)
    g.add_argument("--update-threshold", type=float, default=d.update_threshold)
    g.add_argument("--zncc-threshold", type=float, default=d.zncc_threshold)
    g.add_argument("--prefilter", type=float, default=d.prefilter_sigma, help="Gaussian sigma (px)")
    g = ap.add_argument_group("strain")
    g.add_argument("--strain-window", type=int, default=d.strain_window, help="grid points per side (odd)")
    g.add_argument("--strain-kind", choices=["green-lagrange", "engineering"], default=d.strain_kind)
    g.add_argument("--pixel-size", type=float, default=1.0, help="physical size of one pixel")
    g.add_argument("--units", default="px", help="units of --pixel-size, e.g. mm")
    g = ap.add_argument_group("output")
    g.add_argument("--component", default="exx", choices=list(STRAINS) + ["u", "v", "zncc"],
                   help="field shown in the overlay video")
    g.add_argument("--clim", nargs=2, type=float, help="colour limits for the overlay video")
    g.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    src = FrameSource(args.input, args.start, args.stop, args.every)
    first = src.read(src.indices()[0])
    roi = tuple(args.roi) if args.roi else (select_roi(first) if args.select_roi else None)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) > 0 if args.mask else None
    print(f"{src.n_frames} frames, {first.shape[1]}x{first.shape[0]} px, fps={src.fps}, ROI={roi}")

    settings = DICSettings(
        subset_size=args.subset, step=args.step, search_radius=args.search, reference=args.reference,
        update_threshold=args.update_threshold, zncc_threshold=args.zncc_threshold,
        prefilter_sigma=args.prefilter, strain_window=args.strain_window, strain_kind=args.strain_kind,
        pixel_size=args.pixel_size, units=args.units,
    )
    result = run_dic(src, settings, roi=roi, mask=mask)

    out = Path(args.out) if args.out else Path(str(Path(args.input).with_suffix("")) + "_dic")
    out.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices in empty frames
        result.save(out / "results.npz")
        write_summary_csv(result, out / "strain_summary.csv")
        figs = save_summary_figures(result, src, out)
        if not args.no_video:
            write_overlay_video(result, src, out / f"overlay_{args.component}.mp4", args.component,
                                clim=tuple(args.clim) if args.clim else None)

    hist = result.strain_history()
    print(f"\nFinal frame mean strains: exx={np.nanmean(hist['exx'][-1]):.5f}  "
          f"eyy={np.nanmean(hist['eyy'][-1]):.5f}  exy={np.nanmean(hist['exy'][-1]):.5f}")
    print(f"Reference frames used: {result.reference_frames}")
    print(f"Results written to {out}/ ({', '.join(p.name for p in figs)}, results.npz, strain_summary.csv"
          + ("" if args.no_video else f", overlay_{args.component}.mp4") + ")")


if __name__ == "__main__":
    main()
