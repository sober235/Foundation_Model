import numpy as np

from anatobind.data_engine.skmtea_recon import (
    add_complex_noise, adjoint_sense, embed_poisson, noise_sigma, undersample,
)


def _forward(image, maps):
    """Test-only forward model matching the module's convention."""
    coil_imgs = image[..., None] * maps
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(coil_imgs, axes=(1, 2)), axes=(1, 2), norm="ortho"), axes=(1, 2))


def _case(seed=0, shape=(6, 8, 4), coils=3):
    rng = np.random.default_rng(seed)
    image = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    maps = rng.standard_normal(shape + (coils,)) + 1j * rng.standard_normal(shape + (coils,))
    maps /= np.sqrt((np.abs(maps) ** 2).sum(-1, keepdims=True))  # sum_c |S_c|^2 == 1
    return image, maps


def test_adjoint_sense_inverts_fully_sampled_forward_model():
    image, maps = _case()
    rec = adjoint_sense(_forward(image, maps), maps)
    assert rec.shape == image.shape
    assert np.allclose(rec, image, atol=1e-6)


def test_embed_poisson_centres_the_mask():
    m = np.ones((416, 80), dtype=bool)
    e = embed_poisson(m, ky=512, kz=160)
    assert e.shape == (512, 160) and e.dtype == bool
    assert e[48:464, 40:120].all() and not e[:48].any() and not e[464:].any() and not e[:, :40].any() and not e[:, 120:].any()


def test_undersample_zeroes_unsampled_lines_for_all_x_and_coils():
    k = np.ones((6, 8, 4, 3), dtype=np.complex64)
    mask = np.zeros((8, 4), dtype=bool)
    mask[2, 1] = True
    u = undersample(k, mask)
    assert u[:, 2, 1, :].all() and u.sum() == 6 * 3


def test_noise_sigma_is_fraction_of_rms_over_nonzero_entries():
    k = np.zeros((2, 4, 4, 1), dtype=np.complex64)
    k[0, 0, 0, 0] = 3 + 4j  # magnitude 5
    assert noise_sigma(k, fraction=0.5) == 2.5


def test_add_complex_noise_has_requested_std_and_zero_is_identity():
    k = np.zeros((32, 32, 32, 2), dtype=np.complex64)
    n = add_complex_noise(k, sigma=1.0, rng=np.random.default_rng(0))
    assert abs(np.sqrt((np.abs(n) ** 2).mean()) - 1.0) < 0.05
    assert np.array_equal(add_complex_noise(k, sigma=0.0, rng=np.random.default_rng(0)), k)
