import numpy as np
import pytest

from anatobind.data_engine.fastmri_knee import (
    VIEWS, degrade, equispaced_mask, noise_sigma, reconstruct_rss,
)


def _kspace(seed=0, coils=4, ny=64, nx=40):
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(coils, ny, nx)) + 1j * rng.normal(size=(coils, ny, nx))
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def test_the_seven_views_are_the_same_as_leg_one():
    assert VIEWS == ("clean", "noise_q1", "noise_q2", "noise_q3", "us4", "us8", "us16")


def test_reconstruction_centre_crops_to_the_requested_size():
    out = reconstruct_rss(_kspace(), size=32)
    assert out.shape == (32, 32) and out.dtype == np.float32 and (out >= 0).all()


def test_the_clean_view_is_the_reconstruction_itself():
    k = _kspace()
    np.testing.assert_allclose(reconstruct_rss(degrade(k, "clean", seed=0), size=32),
                               reconstruct_rss(k, size=32), rtol=0, atol=0)


def test_noise_grows_with_the_quality_level():
    k = _kspace()
    ref = reconstruct_rss(k, size=32)
    errs = [np.linalg.norm(reconstruct_rss(degrade(k, v, seed=0), size=32) - ref) for v in
            ("noise_q1", "noise_q2", "noise_q3")]
    assert errs[0] < errs[1] < errs[2]
    assert noise_sigma(k, 2.0) == pytest.approx(2 * noise_sigma(k, 1.0))


def test_the_undersampling_masks_are_nested_and_keep_the_centre():
    m4 = equispaced_mask(100, 4, 0.08, seed=0)
    m8 = equispaced_mask(100, 8, 0.04, seed=0)
    m16 = equispaced_mask(100, 16, 0.04, seed=0)
    assert m4.sum() > m8.sum() > m16.sum()
    assert (m16 <= m8).all() and (m8 <= m4).all()          # nested
    assert m16[48:52].all()                                 # the centre lines survive every level
    assert equispaced_mask(100, 4, 0.08, seed=0).tolist() == equispaced_mask(100, 4, 0.08, seed=0).tolist()


def test_undersampling_zeroes_the_unsampled_phase_encode_lines():
    k = _kspace()
    out = degrade(k, "us8", seed=0)
    kept = np.abs(out).sum(axis=(0, 1)) > 0
    assert kept.sum() < k.shape[-1]
    np.testing.assert_allclose(out[..., kept], k[..., kept])
