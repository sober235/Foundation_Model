"""Which structure a lesion sits in, from its mask and a SynthSeg label map on the same grid (spec 2026-09-29 §6).

Host classes are the seven of anatobind.eval.geometry (sides merged; ventricles and CSF are landmarks, never hosts).
Everything this module returns rests on a pseudo-label: NOT_EVIDENCE."""
import numpy as np
from scipy import ndimage

from anatobind.eval.geometry import CLASS_NAMES, HOST_CLASSES, LEFT_LABELS, MIDLINE_SHARE, RIGHT_LABELS, host_class_map


def side_of(labels):
    """left / right by the majority of the sided labels; bilateral when both reach MIDLINE_SHARE; midline without any."""
    left, right = int(np.isin(labels, LEFT_LABELS).sum()), int(np.isin(labels, RIGHT_LABELS).sum())
    if left + right == 0:
        return "midline"
    if min(left, right) / (left + right) >= MIDLINE_SHARE:
        return "bilateral"
    return "left" if left > right else "right"


class BrainBinder:
    def __init__(self, seg, spacing):
        self.seg = np.asarray(seg)
        self.spacing = tuple(float(s) for s in spacing)
        self.classes = host_class_map(self.seg)
        self._nearest = None

    def _nearest_host(self):
        """(distance in mm to the nearest host voxel, SynthSeg label of that voxel); (None, None) without any host."""
        if self._nearest is None:
            host = self.classes > 0
            if not host.any():
                self._nearest = (None, None)
            else:
                dist, idx = ndimage.distance_transform_edt(~host, sampling=self.spacing, return_indices=True)
                self._nearest = (dist, self.seg[tuple(idx)])
        return self._nearest

    def bind(self, sl, mask):
        """sl: the lesion box as slices; mask: the lesion's voxels inside that box.

        side is counted over the whole lesion, host_side over the lesion's voxels inside the main structure, and
        host_sides likewise for every structure of host_fractions: a lesion of the left thalamus that reaches into
        the right white matter is bilateral, its main structure is the left thalamus, the white matter it involves is
        the right one. host_distance_mm is 0 when the lesion overlaps a structure, else the distance to the nearest
        one."""
        if self.seg[sl].shape != mask.shape:
            raise ValueError(f"lesion mask {mask.shape} does not fit the anatomy grid {self.seg.shape}")
        labels = self.seg[sl][mask]
        counts = np.bincount(self.classes[sl][mask].astype(np.int64), minlength=len(CLASS_NAMES) + 1)[1:]
        if counts.sum() > 0:
            host, rule, distance = CLASS_NAMES[int(np.argmax(counts))], "overlap", 0.0
            fractions = {CLASS_NAMES[i]: round(float(c) / float(counts.sum()), 4) for i, c in enumerate(counts) if c}
            sides = {name: side_of(labels[np.isin(labels, HOST_CLASSES[name])]) for name in fractions}
        else:
            dist, near = self._nearest_host()
            if dist is None:
                return {"host": None, "host_rule": None, "host_fractions": {}, "side": "midline", "host_side": "midline",
                        "host_sides": {}, "host_distance_mm": None}
            j = int(np.argmin(dist[sl][mask]))
            labels = near[sl][mask][j:j + 1]
            host, rule, fractions = CLASS_NAMES[int(host_class_map(labels)[0]) - 1], "nearest", {}
            sides, distance = {host: side_of(labels)}, round(float(dist[sl][mask][j]), 2)
        return {"host": host, "host_rule": rule, "host_fractions": fractions, "side": side_of(labels),
                "host_side": sides[host], "host_sides": sides, "host_distance_mm": distance}
