"""Plots and overlay videos of DIC results."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .pipeline import DICResult, _fill_nearest  # noqa: E402
from .video import FrameSource  # noqa: E402

LABELS = {
    "u": "u", "v": "v", "zncc": "ZNCC",
    "exx": r"$\varepsilon_{xx}$", "eyy": r"$\varepsilon_{yy}$", "exy": r"$\varepsilon_{xy}$",
    "e1": r"$\varepsilon_1$", "e2": r"$\varepsilon_2$", "von_mises": r"$\varepsilon_{vM}$",
}
STRAINS = ("exx", "eyy", "exy", "e1", "e2", "von_mises")


def field_values(result: DICResult, k: int, component: str) -> np.ndarray:
    if component in ("u", "v"):
        return getattr(result, component)[k] * result.settings.pixel_size
    if component == "zncc":
        return result.zncc[k]
    return getattr(result.strain(k), component)


def deformed_grid(result: DICResult, k: int):
    """Current-frame positions of the grid points (NaNs filled for plotting)."""
    XX, YY = np.meshgrid(result.grid_x, result.grid_y)
    return XX + _fill_nearest(result.u[k]), YY + _fill_nearest(result.v[k])


def _symmetric(comp):
    return comp in ("u", "v", "exx", "eyy", "exy", "e1", "e2")


def auto_clim(result: DICResult, component: str, frames=None):
    frames = range(len(result.frames)) if frames is None else frames
    vals = np.concatenate([field_values(result, k, component).ravel() for k in frames])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1, 1)
    if _symmetric(component):
        m = np.percentile(np.abs(vals), 99) or 1e-6
        return (-m, m)
    return (np.percentile(vals, 1), np.percentile(vals, 99))


def plot_frame(result: DICResult, source: FrameSource, k: int, component: str, ax=None,
               clim=None, alpha=1.0):
    """Field overlaid on the current (deformed) frame."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    img = source.read(int(result.frames[k]))
    ax.imshow(img, cmap="gray")
    xd, yd = deformed_grid(result, k)
    val = np.ma.masked_invalid(field_values(result, k, component))
    clim = clim or auto_clim(result, component, [k])
    cmap = "RdBu_r" if _symmetric(component) else "viridis"
    mesh = ax.pcolormesh(xd, yd, val, cmap=cmap, vmin=clim[0], vmax=clim[1], alpha=alpha, shading="gouraud",
                         rasterized=True)
    unit = f" ({result.settings.units})" if component in ("u", "v") else ""
    plt.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02, label=LABELS.get(component, component) + unit)
    t = result.times[k]
    ax.set_title(f"{LABELS.get(component, component)} — frame {result.frames[k]}"
                 + (f", t = {t:.3f} s" if np.isfinite(t) else ""))
    ax.set_axis_off()
    return ax


def save_summary_figures(result: DICResult, source: FrameSource, out_dir: str | Path, k: int = -1) -> list[Path]:
    out_dir = Path(out_dir)
    k = k % len(result.frames)
    paths = []

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, comp in zip(axes.ravel(), ["u", "v", "exx", "eyy", "exy", "von_mises"]):
        plot_frame(result, source, k, comp, ax=ax)
    fig.tight_layout()
    p = out_dir / f"fields_frame{result.frames[k]}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    hist = result.strain_history()
    x = result.times if np.all(np.isfinite(result.times)) else result.frames
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for comp in ("exx", "eyy", "exy", "e1", "e2"):
        a1.plot(x, np.nanmean(hist[comp], axis=(1, 2)), label=LABELS[comp])
    a1.set_xlabel("time (s)" if x is result.times else "frame")
    a1.set_ylabel("mean strain over ROI")
    a1.grid(alpha=0.3)
    a1.legend()
    a2.plot(x, np.nanmean(result.zncc, axis=(1, 2)), label="mean")
    a2.plot(x, np.nanmin(result.zncc, axis=(1, 2)), label="min", alpha=0.7)
    valid_frac = np.mean(np.isfinite(result.zncc), axis=(1, 2))
    a2b = a2.twinx()
    a2b.plot(x, valid_frac * 100, "k--", alpha=0.4, label="valid %")
    a2b.set_ylabel("valid points (%)")
    a2b.set_ylim(0, 105)
    a2.set_xlabel(a1.get_xlabel())
    a2.set_ylabel("ZNCC")
    a2.grid(alpha=0.3)
    a2.legend(loc="lower left")
    fig.tight_layout()
    p = out_dir / "strain_history.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths


def _colorbar_strip(cmap, clim, height, label, width=90):
    strip = np.full((height, width, 3), 255, np.uint8)
    top, bot = 30, height - 30
    ramp = np.linspace(1, 0, bot - top)[:, None]
    colors = (cmap(ramp)[..., :3] * 255).astype(np.uint8)[..., ::-1]
    strip[top:bot, 10:30] = np.repeat(colors, 20, axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for frac in (0, 0.5, 1):
        y = int(top + (1 - frac) * (bot - top - 1))
        val = clim[0] + frac * (clim[1] - clim[0])
        cv2.putText(strip, f"{val:.3g}", (34, y + 4), font, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(strip, label, (5, 20), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return strip


def write_overlay_video(result: DICResult, source: FrameSource, path: str | Path, component: str = "exx",
                        clim=None, alpha: float = 0.6, fps: float | None = None) -> Path:
    """MP4 with the chosen field painted over each frame at the tracked (deformed) positions."""
    path = Path(path)
    clim = clim or auto_clim(result, component)
    cmap = matplotlib.colormaps["RdBu_r" if _symmetric(component) else "viridis"]
    fps = fps or (source.fps / max(source.step, 1) if source.fps else 10.0)
    frame_pos = {int(f): k for k, f in enumerate(result.frames)}
    names = {"exx": "exx", "eyy": "eyy", "exy": "exy", "e1": "e1", "e2": "e2", "von_mises": "e_vM",
             "u": f"u ({result.settings.units})", "v": f"v ({result.settings.units})", "zncc": "ZNCC"}

    writer = None
    for idx, img in source:
        if idx not in frame_pos:
            continue
        k = frame_pos[idx]
        gray = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        paint = base.copy()
        painted = np.zeros(gray.shape, np.uint8)
        val = field_values(result, k, component)
        xd, yd = deformed_grid(result, k)
        norm = np.clip((val - clim[0]) / (clim[1] - clim[0] + 1e-30), 0, 1)
        col = (cmap(norm)[..., :3] * 255).astype(np.uint8)[..., ::-1]
        ny, nx = val.shape
        # Paint each grid cell as a quad coloured by the mean of its corner values.
        for j in range(ny - 1):
            for i in range(nx - 1):
                corners = val[j:j + 2, i:i + 2]
                if not np.all(np.isfinite(corners)):
                    continue
                c = col[j:j + 2, i:i + 2].reshape(-1, 3).mean(0)
                quad = np.array([[xd[j, i], yd[j, i]], [xd[j, i + 1], yd[j, i + 1]],
                                 [xd[j + 1, i + 1], yd[j + 1, i + 1]], [xd[j + 1, i], yd[j + 1, i]]])
                pts = np.round(quad * 16).astype(np.int32)
                cv2.fillPoly(paint, [pts], tuple(int(x) for x in c), lineType=cv2.LINE_AA, shift=4)
                cv2.fillPoly(painted, [pts], 255, shift=4)
        blend = cv2.addWeighted(paint, alpha, base, 1 - alpha, 0)
        out = np.where(painted[..., None] > 0, blend, base)
        t = result.times[k]
        label = f"frame {idx}" + (f"  t={t:.3f}s" if np.isfinite(t) else "")
        cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        out = np.hstack([out, _colorbar_strip(cmap, clim, out.shape[0], names.get(component, component))])
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (out.shape[1], out.shape[0]))
        writer.write(out)
    if writer is not None:
        writer.release()
    return path


def write_summary_csv(result: DICResult, path: str | Path) -> Path:
    hist = result.strain_history()
    cols = {
        "frame": result.frames,
        "time_s": result.times,
        f"mean_u_{result.settings.units}": np.nanmean(result.u, axis=(1, 2)) * result.settings.pixel_size,
        f"mean_v_{result.settings.units}": np.nanmean(result.v, axis=(1, 2)) * result.settings.pixel_size,
    }
    for c in STRAINS:
        cols[f"mean_{c}"] = np.nanmean(hist[c], axis=(1, 2))
    for c in ("e1", "von_mises"):
        cols[f"max_{c}"] = np.nanmax(hist[c], axis=(1, 2))
    cols["mean_zncc"] = np.nanmean(result.zncc, axis=(1, 2))
    cols["valid_fraction"] = np.mean(np.isfinite(result.zncc), axis=(1, 2))
    data = np.column_stack(list(cols.values()))
    np.savetxt(path, data, delimiter=",", header=",".join(cols), comments="", fmt="%.8g")
    return Path(path)
