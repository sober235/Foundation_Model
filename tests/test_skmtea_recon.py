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


def test_noise_sigma_is_fraction_of_rms_renormalised_to_eight_coils():
    k = np.zeros((2, 4, 4, 8), dtype=np.complex64)
    k[0, 0, 0, 0] = 3 + 4j  # magnitude 5; C == 8 is the reference, so the factor is 1
    assert noise_sigma(k, fraction=0.5) == 2.5
    k16 = np.zeros((2, 4, 4, 16), dtype=np.complex64)
    k16[0, 0, 0, 0] = 3 + 4j
    assert abs(noise_sigma(k16, fraction=0.5) - 2.5 * np.sqrt(2.0)) < 1e-6


def test_add_complex_noise_has_requested_std_on_the_support_and_zero_is_identity():
    k = np.zeros((32, 32, 32, 2), dtype=np.complex64)
    k[:, 4:28, 4:28, :] = 1e-6  # the acquired support
    n = add_complex_noise(k, sigma=1.0, rng=np.random.default_rng(0))
    d = n[:, 4:28, 4:28, :] - k[:, 4:28, 4:28, :]
    assert abs(np.sqrt((np.abs(d) ** 2).mean()) - 1.0) < 0.05
    assert np.array_equal(add_complex_noise(k, sigma=0.0, rng=np.random.default_rng(0)), k)


def test_noise_is_added_only_where_kspace_was_acquired():
    """Only 38.5% of the ky-kz grid is acquired; noise off that support is energy the scanner never measured."""
    k = np.zeros((3, 8, 6, 2), dtype=np.complex64)
    k[:, 2:5, 1:4, :] = 1 + 1j
    n = add_complex_noise(k, sigma=1.0, rng=np.random.default_rng(0))
    off = np.ones((8, 6), dtype=bool)
    off[2:5, 1:4] = False
    assert not n[:, off].any()
    assert np.abs(n[:, 2:5, 1:4, :] - k[:, 2:5, 1:4, :]).max() > 0


def test_image_domain_noise_level_does_not_depend_on_coil_count():
    """Per-coil k-space RMS scales as 1/sqrt(C), so the 8 sixteen-coil scans got a 1.39x weaker ladder."""
    levels = []
    for coils in (8, 16):
        image, maps = _case(seed=1, shape=(4, 16, 12), coils=coils)
        k = _forward(image, maps)
        rec = adjoint_sense(add_complex_noise(k, noise_sigma(k, 0.5), np.random.default_rng(0)), maps)
        levels.append(float(np.abs(rec - image).std()))
    assert abs(levels[1] / levels[0] - 1.0) < 0.1, levels
