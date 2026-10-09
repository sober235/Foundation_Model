"""Crops in (z, y, x) order with padding, validity, physical coordinates and intensity handling (spec N4, §6).

Volumes are read as (x, y, z) arrays (nibabel) and turned to (z, y, x) so that the anisotropic slice axis is the
model's D axis, where the (2, 4, 4) patch stem expects it. A crop may run outside the volume: the outside is filled and
marked invalid (M_valid); nothing is resized. Coordinates are physical (mm, the volume's own frame) for the RoPE and
normalised to [-1, 1] over the volume's extent for PE_local (plan §7)."""
import numpy as np
from scipy import ndimage

from anatobind.anatomy.simulate import bias_field

CROP = (128, 160, 160)          # (z, y, x) voxels, spec §6
PATCH = (2, 4, 4)


def to_zyx(arr):
    return np.ascontiguousarray(np.transpose(np.asarray(arr), (2, 1, 0)))


def spacing_zyx(spacing):
    return (float(spacing[2]), float(spacing[1]), float(spacing[0]))


def crop_window(shape, crop, rng, centre=None):
    """[(start, stop)] per axis in volume voxel coordinates (may leave the volume). Without a centre the window is
    uniformly random inside the volume where the volume is larger than the crop; where it is smaller the crop is
    centred on the volume. With a centre the window is placed on it and then pushed back inside the volume where it
    fits."""
    out = []
    for n, c, ctr in zip(shape, crop, centre if centre is not None else (None, None, None)):
        if n <= c:
            start = -((c - n) // 2)
        elif ctr is None:
            start = int(rng.integers(0, n - c + 1))
        else:
            start = min(max(int(round(ctr)) - c // 2, 0), n - c)
        out.append((start, start + c))
    return out


def extract(arr, window, fill=0):
    """(arr inside the window with the outside filled, boolean validity of the same shape)."""
    crop = np.full([b - a for a, b in window], fill, dtype=arr.dtype)
    valid = np.zeros(crop.shape, bool)
    src = tuple(slice(max(a, 0), min(b, n)) for (a, b), n in zip(window, arr.shape))
    dst = tuple(slice(s.start - a, s.stop - a) for s, (a, b) in zip(src, window))
    if all(s.stop > s.start for s in src):
        crop[dst] = arr[src]
        valid[dst] = True
    return crop, valid


def coordinates_mm(window, spacing=None, affine=None):
    """Physical RAS coordinates from an xyz NIfTI affine, returned as zyx channels.

    NIfTI voxel indices represent voxel centres; no extra 0.5 offset applies.
    The spacing fallback exists only for older synthetic fixtures.
    """
    if affine is not None:
        A = np.asarray(affine, dtype=np.float64)
        if A.shape != (4, 4) or not np.isfinite(A).all():
            raise ValueError("invalid NIfTI affine")
        zz, yy, xx = np.meshgrid(*(np.arange(a, b, dtype=np.float64) for a, b in window), indexing="ij")
        vox = np.stack((xx, yy, zz))
        world = np.einsum("ij,jdhw->idhw", A[:3, :3], vox) + A[:3, 3, None, None, None]
        return np.ascontiguousarray(world[[2, 1, 0]], dtype=np.float32)
    if spacing is None:
        raise ValueError("physical coordinates require affine")
    axes = [(np.arange(a, b, dtype=np.float32) + 0.5) * s for (a, b), s in zip(window, spacing)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), 0).astype(np.float32)


def local_coordinates(window, shape):
    """(3, D, H, W) float32 in [-1, 1]: position relative to the volume's extent (PE_local); outside voxels run beyond."""
    axes = [((np.arange(a, b, dtype=np.float32) + 0.5) / n) * 2 - 1 for (a, b), n in zip(window, shape)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), 0).astype(np.float32)


def normalise(image):
    """Whole-volume robust scaling: the 0.5th–99.5th percentiles of the non-zero voxels to [-1, 1]; zeros stay at -1."""
    img = np.asarray(image, np.float32)
    vals = img[img != 0]
    if vals.size == 0:
        return np.full(img.shape, -1.0, np.float32)
    lo, hi = np.percentile(vals, [0.5, 99.5])
    hi = hi if hi > lo else lo + 1.0
    out = (np.clip(img, lo, hi) - lo) / (hi - lo) * 2 - 1
    out[img == 0] = -1.0
    return out.astype(np.float32)


def augment(image, rng, p=0.8):
    """Intensity augmentation only, no mirroring (S4 A17): a second-order bias field, gamma, in-plane blur and Gaussian
    noise, each with probability p, on an image in [-1, 1]; invalid voxels (exactly -1) keep their value."""
    img = np.asarray(image, np.float32).copy()
    fg = img > -1.0
    if rng.random() < p:
        field = bias_field(img.shape, rng.uniform(-1, 1, 6), amp=0.2).astype(np.float32)
        img = np.where(fg, (img + 1.0) * field - 1.0, img)
    if rng.random() < p:
        g = float(rng.uniform(0.7, 1.4))
        img = np.where(fg, np.power(np.clip((img + 1.0) / 2.0, 0.0, 1.0), g) * 2.0 - 1.0, img)
    if rng.random() < p * 0.5:
        img = np.where(fg, ndimage.gaussian_filter(img, sigma=(0.0, float(rng.uniform(0.3, 0.8)), float(rng.uniform(0.3, 0.8)))), img)
    if rng.random() < p:
        img = np.where(fg, img + rng.normal(0.0, float(rng.uniform(0.0, 0.05)), img.shape).astype(np.float32), img)
    return img.astype(np.float32)


def rotate_inplane(arrays, angle_deg, orders):
    """Rotate every array about the z axis (the in-plane rotation, axes (1, 2) in (z, y, x)) by the same angle, each
    with its own interpolation order (1 for images, 0 for labels and masks); shapes are kept."""
    out = []
    for a, order in zip(arrays, orders):
        r = ndimage.rotate(a, angle_deg, axes=(1, 2), reshape=False, order=order, mode="constant", cval=0, prefilter=False)
        out.append(r.astype(a.dtype) if order == 0 else r)
    return out
