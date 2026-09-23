"""Predicted lesion boxes from an nnU-Net lesion label map (spec 2026-09-23 §3.3, §4.1).

One box per 26-connected component of each family, in the export (X, Y, Z) frame. The score is the
mean predicted probability of that family inside the component (nnU-Net --npz / --save_probabilities);
without probabilities the score is n / (n + MIN_VOXELS), which only orders components by size.
"""
import nibabel as nib
import numpy as np
from scipy import ndimage

from anatobind.nnunet.lesion_labels import FAMILY_OF_LABEL

STRUCTURE = np.ones((3, 3, 3), bool)     # 26-connectivity
MIN_VOXELS = 27                          # a 3x3x3 block; smaller blobs are decoder noise
AGREEMENT_MIN = 0.99


def decode_boxes(label_map, probs=None, min_voxels=MIN_VOXELS):
    out = []
    for label, family in FAMILY_OF_LABEL.items():
        comp, n = ndimage.label(label_map == label, structure=STRUCTURE)
        if n == 0:
            continue
        for k, sl in enumerate(ndimage.find_objects(comp), start=1):
            if sl is None:
                continue
            mask = comp[sl] == k
            nv = int(mask.sum())
            if nv < min_voxels:
                continue
            score = float(probs[label][sl][mask].mean()) if probs is not None else nv / (nv + min_voxels)
            out.append({"family": family, "label": int(label),
                        "box": (sl[0].start, sl[1].start, sl[2].start, sl[0].stop, sl[1].stop, sl[2].stop),
                        "score": float(score), "n_voxels": nv})
    return sorted(out, key=lambda d: -d["score"])


def load_label_map(nii_path):
    return np.ascontiguousarray(np.asanyarray(nib.load(str(nii_path)).dataobj)).astype(np.uint8)


def load_nnunet_probabilities(npz_path, label_map_xyz):
    """nnU-Net stores 'probabilities' as (C, Z, Y, X) (SimpleITK array order); return (C, X, Y, Z)."""
    with np.load(str(npz_path)) as z:
        p = z["probabilities"]
    p = np.ascontiguousarray(p.transpose(0, 3, 2, 1)).astype(np.float32)
    agree = float((p.argmax(0) == label_map_xyz).mean())
    if agree < AGREEMENT_MIN:
        raise ValueError(f"{npz_path}: probabilities disagree with the label map on {1 - agree:.1%} of voxels; axis order?")
    return p
