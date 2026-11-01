import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dic.correlation import ReferenceImage, SplineImage, compose_inverse, icgn, integer_search  # noqa: E402
from dic.pipeline import DICSettings, DICResult, run_dic  # noqa: E402
from dic.strain import compute_strain, plane_fit_gradient  # noqa: E402
from dic.synthetic import SpecklePattern, uniaxial, write_sequence  # noqa: E402
from dic.video import FrameSource  # noqa: E402


@pytest.fixture(scope="module")
def pattern():
    return SpecklePattern(300, 220, seed=7)


def grid(x0, x1, y0, y1, step):
    XX, YY = np.meshgrid(np.arange(x0, x1, step, dtype=float), np.arange(y0, y1, step, dtype=float))
    return XX, YY, np.column_stack([XX.ravel(), YY.ravel()])


def test_compose_inverse_identity():
    p = np.array([[1.2, 0.01, -0.02, -0.5, 0.03, 0.005]])
    assert np.allclose(compose_inverse(p, np.zeros((1, 6))), p)
    assert np.allclose(compose_inverse(p, p), 0, atol=1e-12)


@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.37, -0.81), (5.5, 2.25)])
def test_rigid_translation(pattern, shift):
    f = pattern.render().astype(float)
    g = pattern.render(lambda X, Y: (np.full_like(X, shift[0]), np.full_like(Y, shift[1]))).astype(float)
    _, _, C = grid(40, 261, 40, 181, 20)
    uv, _ = integer_search(f, g, C, 31, 10)
    p0 = np.zeros((len(C), 6))
    p0[:, 0], p0[:, 3] = uv[:, 0], uv[:, 1]
    r = icgn(ReferenceImage(f), SplineImage(g), C, p0, 31)
    assert r.valid.all()
    assert np.abs(r.p[:, 0] - shift[0]).max() < 0.02
    assert np.abs(r.p[:, 3] - shift[1]).max() < 0.02
    assert np.nanmin(r.zncc) > 0.99


def test_uniform_strain(pattern):
    d = uniaxial(0.03, 0.3, center=(150, 110), shift=(1.3, 0.4))
    f, g = pattern.render().astype(float), pattern.render(d).astype(float)
    XX, YY, C = grid(30, 271, 30, 191, 10)
    uv, _ = integer_search(f, g, C, 25, 15)
    p0 = np.zeros((len(C), 6))
    p0[:, 0], p0[:, 3] = uv[:, 0], uv[:, 1]
    r = icgn(ReferenceImage(f), SplineImage(g), C, p0, 25)
    assert r.valid.all()
    assert abs(np.mean(r.p[:, 1]) - 0.03) < 2e-4
    assert abs(np.mean(r.p[:, 5]) + 0.009) < 2e-4
    s = compute_strain(r.p[:, 0].reshape(XX.shape), r.p[:, 3].reshape(XX.shape), 10, 5, "engineering")
    assert abs(np.nanmean(s.exx) - 0.03) < 2e-4
    assert abs(np.nanmean(s.eyy) + 0.009) < 2e-4
    assert abs(np.nanmean(s.exy)) < 2e-4


def test_plane_fit_exact_with_nans():
    y, x = np.mgrid[0:20, 0:25].astype(float) * 5
    u = 0.01 * x - 0.02 * y + 3
    u[7, 9] = np.nan
    gx, gy = plane_fit_gradient(u, 5.0, 5)
    ok = np.isfinite(gx)
    assert ok.sum() > 0.8 * ok.size
    assert np.allclose(gx[ok], 0.01) and np.allclose(gy[ok], -0.02)


def test_green_lagrange_large_stretch():
    y, x = np.mgrid[0:15, 0:15].astype(float)
    s = compute_strain(0.2 * x, 0 * y, 1.0, 5, "green-lagrange")
    assert np.allclose(np.nanmean(s.exx), 0.2 + 0.5 * 0.04)


def test_pipeline_sequence(tmp_path, pattern):
    """Full video pipeline: growing strain, lossless frames, save/load round trip."""
    eps = np.linspace(0, 0.06, 7)
    frames = [pattern.render(uniaxial(e, 0.35, center=(150, 110)), noise=1.0, seed=i) for i, e in enumerate(eps)]
    folder = write_sequence(tmp_path / "frames", frames)
    res = run_dic(FrameSource(folder), DICSettings(subset_size=25, step=10), roi=(30, 30, 240, 160), verbose=False)
    exx = [np.nanmean(res.strain(k, kind="engineering").exx) for k in range(len(eps))]
    assert np.allclose(exx, eps, atol=5e-4)
    assert np.all(np.isfinite(res.u[-1]))
    res.save(tmp_path / "r.npz")
    back = DICResult.load(tmp_path / "r.npz")
    assert np.allclose(back.u, res.u, equal_nan=True)
    assert back.settings.subset_size == 25
