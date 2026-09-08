"""SKM-TEA hybrid k-space: adjoint SENSE, Poisson-disc mask embedding, undersampling, noise.

Layout (verified on MTR_001): kspace (X, KY, KZ, C) with the readout axis X already in the
image domain; maps (X, Y, Z, C). Reconstruction = ifftshift -> ifftn over (KY, KZ) -> fftshift,
orthonormal, then coil combination sum_c conj(S_c) * x_c.
"""
import numpy as np


def adjoint_sense(kspace_hybrid, maps):
    x = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace_hybrid, axes=(1, 2)), axes=(1, 2), norm="ortho"), axes=(1, 2))
    return (np.conj(maps) * x).sum(axis=-1)


def embed_poisson(mask, ky, kz):
    """Centre the (416, 80) acquisition-grid mask inside the (ky, kz) reconstruction grid."""
    out = np.zeros((ky, kz), dtype=bool)
    oy, oz = (ky - mask.shape[0]) // 2, (kz - mask.shape[1]) // 2
    out[oy:oy + mask.shape[0], oz:oz + mask.shape[1]] = mask.astype(bool)
    return out


def undersample(kspace_hybrid, mask_kykz):
    return kspace_hybrid * mask_kykz[None, :, :, None]


NOISE_COILS_REF = 8


def acquired_support(kspace_hybrid):
    """The (ky, kz) lines the scanner actually sampled; identical across readout and coils."""
    return (kspace_hybrid != 0).any(axis=(0, -1))


def noise_sigma(kspace_hybrid, fraction, coils_ref=NOISE_COILS_REF):
    """Per-sample noise level, renormalised to an 8-coil reference.

    The RMS is a per-coil quantity and scales as 1/sqrt(C), while the coil-combined image
    noise equals sigma exactly because sum_c |S_c|^2 == 1. Without the sqrt(C/ref) factor
    the eight 16-coil scans get a 1.39x weaker ladder than the other 147 (measured).
    """
    mag = np.abs(kspace_hybrid)
    nz = mag[mag > 0]
    return float(fraction * np.sqrt((nz ** 2).mean()) * np.sqrt(kspace_hybrid.shape[-1] / coils_ref))


def add_complex_noise(kspace_hybrid, sigma, rng):
    """Add noise on the acquired support only; unsampled lines stay exactly zero.

    A noisier scan measures the same lines more noisily, it does not put energy where nothing
    was sampled. Restricting to the support also keeps the noise draw 2.6x smaller than the array.
    """
    if sigma == 0.0:
        return kspace_hybrid
    support = acquired_support(kspace_hybrid)
    out = kspace_hybrid.copy()
    sub = out[:, support]
    n = rng.standard_normal(sub.shape) + 1j * rng.standard_normal(sub.shape)
    out[:, support] = sub + (sigma / np.sqrt(2.0)) * n
    return out
