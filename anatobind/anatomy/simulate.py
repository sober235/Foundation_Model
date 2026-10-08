"""Slab simulation: a 1 mm isotropic RAS volume -> a fastMRI-style FLAIR stack (spec 2026-10-02 §5, A5-A8).

Geometry follows the measurements of docs/verification/2026-10-02/s4_probe2 (16 x 5 mm slices positioned from the
vertex down, 2-3 empty slices above the brain, 65 mm of brain in the stack) and the array frame of the fastMRI RSS
NIfTIs (docs/verification/2026-10-02/s4_probe3_frame: flip axis 0 of a RAS array). Intensities are augmented, not
simulated from sequence physics (A7). Labels are voted over the five 1 mm slices of each slab; lesion voxels become
the ignore label (A8). Every function is deterministic given its random generator."""
import math

import numpy as np
from scipy import ndimage

from anatobind.anatomy.labels import IGNORE, N_CLASSES
from anatobind.data_engine.fastmri import rss_affine

SLICE_MM = 5.0
# empty 5 mm slices above the brain in the fastMRI stacks (s4_probe2: 1 / 19 / 157 / 191 / 63 / 15 of 447 stacks have 0 / 1 / 2 / 3 / 4 / 5+)
EMPTY_TOP = ((1, 0.045), (2, 0.35), (3, 0.43), (4, 0.14), (5, 0.035))
N_SLICES = ((16, 0.9), (14, 0.1))
INPLANE_MM = ((0.6875, 0.82), (0.86, 0.18))
MATRICES = (((320, 320), 0.6), ((260, 320), 0.2), ((276, 276), 0.2))
TILT_LR_DEG, TILT_AP_DEG = 10.0, 5.0
BIAS_AMP, GAMMA, BLUR_SIGMA, NOISE = 0.2, (0.7, 1.4), (0.0, 0.7), (0.01, 0.04)
OUT_RANGE = 1000.0
MIN_AREA_MM2 = 500.0                            # 5 cm2: a slice with less brain is "empty" (s4_probe2)


def to_fastmri_frame(ras):
    """RAS array (axis 0 towards the right) -> fastMRI array frame (patient left at high axis-0 indices): flip axis 0."""
    return np.ascontiguousarray(np.asarray(ras)[::-1])


def _choose(rng, options):
    items, weights = zip(*options)
    return items[int(rng.choice(len(items), p=np.asarray(weights) / sum(weights)))]


def sample_params(rng):
    return {"theta_lr": float(rng.uniform(-TILT_LR_DEG, TILT_LR_DEG)), "theta_ap": float(rng.uniform(-TILT_AP_DEG, TILT_AP_DEG)),
            "empty_top": int(_choose(rng, EMPTY_TOP)), "n_slices": int(_choose(rng, N_SLICES)), "inplane_mm": float(_choose(rng, INPLANE_MM)),
            "matrix": tuple(_choose(rng, MATRICES)), "gamma": float(rng.uniform(*GAMMA)), "blur": float(rng.uniform(*BLUR_SIGMA)),
            "noise": float(rng.uniform(*NOISE)), "bias": [float(v) for v in rng.uniform(-1, 1, 6)]}


def rotation_matrix(theta_lr_deg, theta_ap_deg):
    """Rotation about axis 0 (left-right: nodding, tilts the slab in the sagittal plane) then about axis 1 (anterior-
    posterior: roll), for arrays ordered (left-right, anterior-posterior, inferior-superior)."""
    a, b = math.radians(theta_lr_deg), math.radians(theta_ap_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    return ry @ rx


def rotate(volume, theta_lr_deg, theta_ap_deg, center, order):
    """Rotate about `center` (voxel coordinates); order 1 for images, 0 for labels and masks. Outside -> 0."""
    m = rotation_matrix(theta_lr_deg, theta_ap_deg)
    center = np.asarray(center, float)
    offset = center - m @ center           # affine_transform maps output -> input: in = m @ out + offset
    return ndimage.affine_transform(np.asarray(volume), m, offset=offset, order=order, mode="constant", cval=0.0,
                                    output=np.asarray(volume).dtype if order == 0 else np.float32)


def area_profile(labels):
    """Brain cross-section (voxels, = mm2 at 1 mm) of every axial slice: labels > 0 and not ignore."""
    brain = (np.asarray(labels) > 0) & (np.asarray(labels) != IGNORE)
    return brain.sum(axis=(0, 1))


def top_index(profile, min_area_mm2=MIN_AREA_MM2):
    """The highest 1 mm slice with more than min_area of brain; -1 when there is none."""
    idx = np.flatnonzero(np.asarray(profile) > min_area_mm2)
    return int(idx.max()) if len(idx) else -1


def bottom_index(profile, n_slices, empty_top, thickness=int(SLICE_MM)):
    """Stacks are positioned from the vertex down (s4_probe2: 65 mm of brain under 2-3 empty slices): the stack ends
    empty_top slices above the last slice with brain, and its bottom lies n_slices x 5 mm lower, never below 0."""
    top = top_index(profile)
    return max(0, top + 1 + empty_top * thickness - n_slices * thickness)


def slab_groups(z0, n_slices, n_z, thickness=int(SLICE_MM)):
    """[(start, stop)] of the 1 mm slices averaged into each output slice; stop <= n_z, empty when start >= n_z."""
    return [(min(z0 + k * thickness, n_z), min(z0 + (k + 1) * thickness, n_z)) for k in range(n_slices)]


def average_slices(volume, groups):
    v = np.asarray(volume, np.float32)
    out = np.zeros(v.shape[:2] + (len(groups),), np.float32)
    for k, (a, b) in enumerate(groups):
        if b > a:
            out[:, :, k] = v[:, :, a:b].mean(axis=2)
    return out


def vote_labels(labels, groups, n_classes=N_CLASSES):
    """Majority vote of the compact labels over each group's slices; a tie goes to the slice nearest the group's
    centre; an empty group is background."""
    lab = np.asarray(labels)
    nx, ny = lab.shape[:2]
    out = np.zeros((nx, ny, len(groups)), np.uint8)
    ii, jj = np.arange(nx)[:, None], np.arange(ny)[None, :]
    for k, (a, b) in enumerate(groups):
        if b <= a:
            continue
        block = lab[:, :, a:b]
        counts = np.zeros((nx, ny, n_classes), np.int16)
        for z in range(block.shape[2]):
            np.add.at(counts, (ii, jj, block[:, :, z]), 1)
        best = counts.max(axis=2)
        centre = (a + b - 1) / 2.0
        result = np.full((nx, ny), -1, np.int16)
        for z in sorted(range(block.shape[2]), key=lambda z: abs(a + z - centre)):
            l = block[:, :, z].astype(np.int64)
            take = (result < 0) & (counts[ii, jj, l] == best)
            result[take] = l[take]
        out[:, :, k] = result
    return out


def ignore_any(mask, groups):
    """A slab voxel is lesion when any of its 1 mm slices is (A8)."""
    m = np.asarray(mask) > 0
    out = np.zeros(m.shape[:2] + (len(groups),), bool)
    for k, (a, b) in enumerate(groups):
        if b > a:
            out[:, :, k] = m[:, :, a:b].any(axis=2)
    return out


def fit_to_matrix(stack, matrix, center_xy):
    """Crop / zero-pad the first two axes to `matrix`, centred on center_xy (in the stack's own pixels)."""
    s = np.asarray(stack)
    out = np.zeros((matrix[0], matrix[1]) + s.shape[2:], s.dtype)
    x0 = int(math.floor(center_xy[0] - matrix[0] / 2.0 + 0.5))        # round half up, not to even
    y0 = int(math.floor(center_xy[1] - matrix[1] / 2.0 + 0.5))
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + matrix[0], s.shape[0]), min(y0 + matrix[1], s.shape[1])
    if sx1 > sx0 and sy1 > sy0:
        out[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0] = s[sx0:sx1, sy0:sy1]
    return out


def resample_inplane(stack, inplane_mm, matrix, center_xy_mm, order):
    """1 mm slab stack -> target in-plane spacing (axis 2 untouched) and matrix, centred on a point given in mm of the
    1 mm grid."""
    f = 1.0 / inplane_mm
    z = ndimage.zoom(np.asarray(stack), (f, f, 1.0), order=order, mode="constant", cval=0.0, grid_mode=False)
    return fit_to_matrix(z, matrix, (center_xy_mm[0] * f, center_xy_mm[1] * f))


def bias_field(shape, coeffs, amp=BIAS_AMP):
    """1 + a smooth second-order field whose largest deviation is amp; coeffs: six numbers in [-1, 1]."""
    x, y, z = [np.linspace(-1, 1, n) for n in shape]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    c = np.asarray(coeffs, float)
    field = c[0] * X + c[1] * Y + c[2] * Z + c[3] * X * X + c[4] * Y * Y + c[5] * X * Y
    peak = np.abs(field).max()
    return 1.0 + (amp * field / peak if peak > 0 else 0.0)


def intensity_augment(stack, brain, rng, params):
    """Bias field, gamma on the brain's robust range, in-plane blur, Rician noise, then the brain's 1st-99th
    percentiles mapped to [0, OUT_RANGE]. brain: boolean mask of the same shape (where the labels are not background)."""
    img = np.asarray(stack, np.float32) * bias_field(stack.shape, params["bias"]).astype(np.float32)
    vals = img[brain] if brain.any() else img.ravel()
    lo, hi = np.percentile(vals, [1, 99])
    hi = hi if hi > lo else lo + 1.0
    norm = np.clip((img - lo) / (hi - lo), 0, None)
    img = np.power(norm, params["gamma"]).astype(np.float32)
    if params["blur"] > 0:
        img = ndimage.gaussian_filter(img, sigma=(params["blur"], params["blur"], 0.0))
    sigma = params["noise"] * float(np.median(img[brain])) if brain.any() else params["noise"]
    n1, n2 = rng.normal(0, sigma, img.shape).astype(np.float32), rng.normal(0, sigma, img.shape).astype(np.float32)
    img = np.sqrt((img + n1) ** 2 + n2 ** 2)
    vals = img[brain] if brain.any() else img.ravel()
    lo, hi = np.percentile(vals, [1, 99])
    hi = hi if hi > lo else lo + 1.0
    return ((img - lo) / (hi - lo) * OUT_RANGE).astype(np.float32)


def simulate(flair_ras, student_ras, lesion_ras, rng, params=None):
    """flair_ras: 1 mm RAS image; student_ras: compact labels (0..14) on the same grid; lesion_ras: mask or None.
    Returns (image stack float32 (nx, ny, n_slices), label stack uint8 with IGNORE on lesion voxels, params)."""
    params = dict(params) if params is not None else sample_params(rng)
    img, lab = to_fastmri_frame(flair_ras), to_fastmri_frame(student_ras)
    mask = to_fastmri_frame(lesion_ras) if lesion_ras is not None else np.zeros(lab.shape, np.uint8)
    if img.shape != lab.shape or mask.shape != lab.shape:
        raise ValueError(f"image {img.shape}, labels {lab.shape} and mask {mask.shape} differ")
    brain = np.argwhere(lab > 0)
    center = brain.mean(axis=0) if len(brain) else (np.asarray(lab.shape) - 1) / 2.0
    img = rotate(img, params["theta_lr"], params["theta_ap"], center, order=1)
    lab = rotate(lab, params["theta_lr"], params["theta_ap"], center, order=0)
    mask = rotate(mask, params["theta_lr"], params["theta_ap"], center, order=0)
    profile = area_profile(lab)
    z0 = bottom_index(profile, params["n_slices"], params["empty_top"])
    groups = slab_groups(z0, params["n_slices"], lab.shape[2])
    stack, labels = average_slices(img, groups), vote_labels(lab, groups)
    labels[ignore_any(mask, groups)] = IGNORE
    stack = resample_inplane(stack, params["inplane_mm"], params["matrix"], center[:2], order=1)
    labels = resample_inplane(labels, params["inplane_mm"], params["matrix"], center[:2], order=0)
    stack = intensity_augment(stack, (labels > 0) & (labels != IGNORE), rng, params)
    params["bottom_slice_1mm"] = int(z0)
    params["bottom_area_share"] = round(float(profile[z0] / profile.max()), 3) if profile.max() > 0 else None     # compare with s4_probe2 (median 0.83)
    return stack, labels, params


def stack_affine(inplane_mm, shape):
    """The affine the fastMRI RSS NIfTIs carry (anatobind.data_engine.fastmri.rss_affine), for a (col, row, slice) array."""
    return rss_affine(inplane_mm, inplane_mm, SLICE_MM, shape)
