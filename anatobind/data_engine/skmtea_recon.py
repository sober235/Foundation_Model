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


def noise_sigma(kspace_hybrid, fraction):
    mag = np.abs(kspace_hybrid)
    nz = mag[mag > 0]
    return float(fraction * np.sqrt((nz ** 2).mean()))


def add_complex_noise(kspace_hybrid, sigma, rng):
    if sigma == 0.0:
        return kspace_hybrid
    n = rng.standard_normal(kspace_hybrid.shape) + 1j * rng.standard_normal(kspace_hybrid.shape)
    return (kspace_hybrid + (sigma / np.sqrt(2.0)) * n).astype(kspace_hybrid.dtype)
