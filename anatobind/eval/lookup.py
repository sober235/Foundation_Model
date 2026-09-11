"""B0: class-aware seg-then-lookup on predicted geometry (the fair baseline of the 09-09 audit, V1).

Only predictions enter: a predicted label map, a predicted box and a predicted class. The class
restricts the candidate structures (meniscal tear -> menisci, cartilage lesion -> the four
cartilages); the candidate with the largest overlap inside the box wins; if every candidate has zero
overlap, the nearest candidate in millimetres wins. An effusion has no host and a ligament tear's
host is outside the ontology.
"""
import numpy as np

CANDIDATES = {0: (5, 6), 1: (1, 2, 3, 4)}
EFFUSION, LIGAMENT = 2, 3
NONE, UNKNOWN = "none", "unknown"
FAMILY_OF_LABEL = {1: "patellar", 2: "femoral", 3: "tibial", 4: "tibial", 5: "meniscus", 6: "meniscus"}
FAMILY_OF_TISSUE = {5: "patellar", 4: "femoral", 6: "tibial", 1: "meniscus"}


class LabelIndex:
    """Voxel coordinates of every label, computed once per map for the nearest-structure fallback."""

    def __init__(self, label_map, spacing):
        self.label_map = label_map
        self.spacing = np.asarray(spacing, dtype=float)
        self.points = {label: np.argwhere(label_map == label) for label in range(1, 7)}

    def present(self, label):
        return len(self.points[label]) > 0

    def distance_mm(self, box, label):
        pts = self.points[label]
        if not len(pts):
            return np.inf
        lo, hi = np.asarray(box[:3]), np.asarray(box[3:])
        d = np.maximum(np.maximum(lo - pts, pts - (hi - 1)), 0) * self.spacing
        return float(np.sqrt((d ** 2).sum(1)).min())


def clip_box(box, shape):
    shape = np.asarray(shape)
    lo = np.clip(np.floor(np.asarray(box[:3], dtype=float)).astype(int), 0, shape)
    hi = np.clip(np.ceil(np.asarray(box[3:], dtype=float)).astype(int), 0, shape)
    return np.concatenate([lo, hi])


def b0_host(index, box, cls):
    if cls == EFFUSION:
        return NONE
    if cls == LIGAMENT:
        return UNKNOWN
    b = clip_box(box, index.label_map.shape)
    cands = [label for label in CANDIDATES[cls] if index.present(label)]
    if not cands:
        return None
    sub = index.label_map[b[0]:b[3], b[1]:b[4], b[2]:b[5]]
    vol = max(sub.size, 1)
    ioa = {label: float((sub == label).sum()) / vol for label in cands}
    best = max(cands, key=lambda label: ioa[label])
    if ioa[best] > 0:
        return best
    return min(cands, key=lambda label: index.distance_mm(b, label))
