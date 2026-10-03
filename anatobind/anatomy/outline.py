"""Brain outline for skull-stripping fastMRI FLAIR stacks (spec 2026-10-02 §6, A10).

Training labels come from the SynthSeg map of the same stack: labels > 0, holes filled slice by slice, the largest
three-dimensional component kept. Only the slices where that outline is trustworthy supervise the model: from slice
LOW up to the slice below the last one with more than MIN_AREA_MM2 of brain; every other slice is the ignore label.
Inference keeps the largest component of the predicted mask and fills its holes slice by slice."""
import numpy as np
from scipy import ndimage

LABELS_JSON = {"background": 0, "brain": 1, "ignore": 2}
IGNORE = 2
LOW = 2                        # the lowest two slices never supervise (s4_probe: SynthSeg misses parts of the brain there)
TOP_MARGIN = 1                 # the last slice with brain is left out too (the vertex is covered only in part)
MIN_AREA_MM2 = 500.0           # 5 cm2
MIN_VOLUME_ML = 300.0          # a stack whose outline holds less brain is a SynthSeg failure and is not used (s4_probe: 14 of 447)


def fill_and_keep_largest(mask):
    """Boolean (x, y, z) mask -> holes filled in every slice, then the largest 26-connected component."""
    m = np.asarray(mask).astype(bool)
    out = np.zeros_like(m)
    for k in range(m.shape[2]):
        out[:, :, k] = ndimage.binary_fill_holes(m[:, :, k])
    comp, n = ndimage.label(out, structure=np.ones((3, 3, 3)))
    if n == 0:
        return out
    sizes = ndimage.sum(out, comp, index=np.arange(1, n + 1))
    return comp == (int(np.argmax(sizes)) + 1)


def supervised_slices(outline, voxel_area_mm2, low=LOW, top_margin=TOP_MARGIN, min_area_mm2=MIN_AREA_MM2):
    """range(low, top) of the slices that supervise: top = the last slice with more than min_area of brain, minus
    the margin, plus one; empty when the stack holds too little brain."""
    areas = np.asarray(outline).astype(bool).sum(axis=(0, 1)) * float(voxel_area_mm2)
    with_brain = np.flatnonzero(areas > min_area_mm2)
    if len(with_brain) == 0:
        return range(0, 0)
    top = int(with_brain.max()) - top_margin + 1
    return range(low, max(low, top))


def outline_label(seg, voxel_area_mm2, voxel_volume_mm3):
    """Training label of one stack: 1 inside the outline, 0 outside, IGNORE on every unsupervised slice.
    Returns (label uint8, info) with info = {volume_ml, supervised: [first, last] or None, usable}."""
    outline = fill_and_keep_largest(np.asarray(seg) > 0)
    volume_ml = float(outline.sum()) * float(voxel_volume_mm3) / 1000.0
    sup = supervised_slices(outline, voxel_area_mm2)
    label = outline.astype(np.uint8)
    for k in range(label.shape[2]):
        if k not in sup:
            label[:, :, k] = IGNORE
    info = {"volume_ml": round(volume_ml, 1), "supervised": [sup.start, sup.stop - 1] if len(sup) else None,
            "usable": volume_ml >= MIN_VOLUME_ML and len(sup) > 0}
    return label, info


def postprocess(pred):
    """Predicted mask (any array, > 0 = brain) -> largest component with holes filled, as uint8."""
    return fill_and_keep_largest(np.asarray(pred) > 0).astype(np.uint8)


def dice(a, b):
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * float((a & b).sum()) / float(s)
