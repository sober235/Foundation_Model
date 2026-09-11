"""Which axis swap and flips put a segmentation NIfTI array on the h5 (x, y, z) grid.

A frame is (transpose_xy, flip_x, flip_y, flip_z): optionally swap the first two axes, then flip
along the h5 axes. The dicom-track set needs "T---" for "LR" scans and "T--z" for the 18 "RL"
scans (docs/data_engine_skmtea.md). The raw-data-track set is a different product, so its frame is
measured, not assumed.

Score: the frame gate of scripts/verify_skmtea_frames.py -- on echo 1, the smaller of patellar (1)
and femoral (2) cartilage mean intensity divided by the mean outside all labels. Cartilage is bright
on echo 1, so the frame that puts the labels on the tissue maximises it.
"""
import itertools

import numpy as np

FRAMES = [(t, fx, fy, fz) for t in (False, True) for fx, fy, fz in itertools.product((False, True), repeat=3)]


def frame_name(frame):
    return "".join(c if on else "-" for c, on in zip("Txyz", frame))


def parse_frame_name(name):
    assert len(name) == 4 and all(ch in "-Txyz" for ch in name), name
    return tuple(ch != "-" for ch in name)


def apply_frame(arr, frame):
    t, fx, fy, fz = frame
    out = np.transpose(arr, (1, 0, 2)) if t else arr
    for axis, flip in enumerate((fx, fy, fz)):
        if flip:
            out = np.flip(out, axis=axis)
    return out


def store_in_frame(arr_h5, frame):
    """Inverse of apply_frame: the stored array that apply_frame(., frame) maps onto arr_h5."""
    t, fx, fy, fz = frame
    out = arr_h5
    for axis, flip in enumerate((fx, fy, fz)):
        if flip:
            out = np.flip(out, axis=axis)
    return np.transpose(out, (1, 0, 2)) if t else out


def zooms_in_h5_frame(zooms, frame):
    """Voxel sizes permute with the array; flips do not change them."""
    z = tuple(float(v) for v in zooms[:3])
    return (z[1], z[0], z[2]) if frame[0] else z


def cartilage_score(mag, seg):
    outside = mag[seg == 0].mean()
    vals = [mag[seg == l].mean() / outside for l in (1, 2) if (seg == l).any()]
    return float(min(vals)) if vals else float("nan")


def rank_frames(mag, seg_stored):
    """[(frame, score)], best first, over the frames whose result has the image's shape."""
    scored = []
    for fr in FRAMES:
        s = apply_frame(seg_stored, fr)
        if s.shape == mag.shape:
            scored.append((fr, cartilage_score(mag, s)))
    return sorted(scored, key=lambda kv: -kv[1])


def overlap_stats(seg_a, seg_b, spacing):
    """{label: (dice, centroid shift in mm)} for labels present in both maps on the same grid."""
    sp = np.asarray(spacing, dtype=float)
    out = {}
    for label in range(1, 7):
        a, b = seg_a == label, seg_b == label
        if not a.any() or not b.any():
            continue
        dice = 2.0 * np.logical_and(a, b).sum() / (a.sum() + b.sum())
        shift = np.linalg.norm((np.argwhere(a).mean(0) - np.argwhere(b).mean(0)) * sp)
        out[label] = (float(dice), float(shift))
    return out
