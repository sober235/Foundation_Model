"""Box-filled lesion label maps for the knee lesion detector (spec 2026-09-23 §3.1, decision N1/Q9).

Every annotated 3D box is painted into a (X, Y, Z) uint8 map with its family label, in the export
frame of derived/skmtea/m1r (the frame of seg.nii.gz and of the x0..z1 columns of boxes.csv).
Painting order decides the 3.3% of box voxels that lie in boxes of two families (measured
2026-09-23 on all 465 boxes): effusion boxes go first because they are huge (median 99 mL) and
box-shaped, then the rest from largest to smallest, so a smaller box overwrites a larger one.
"""
import csv

import numpy as np

from anatobind.train.dataset_v2 import CLASSES as LOOKUP_CLASS_OF_FAMILY  # noqa: F401  (single source of the lookup ids)

FAMILY_LABELS = {"Cartilage Lesion": 1, "Meniscal Tear": 2, "Ligament Tear": 3, "Effusion": 4}
LABELS = {"background": 0, "cartilage_lesion": 1, "meniscal_tear": 2, "ligament_tear": 3, "effusion": 4}
FAMILY_OF_LABEL = {v: k for k, v in FAMILY_LABELS.items()}
NAME_OF_LABEL = {v: k for k, v in LABELS.items() if v}
EFFUSION_FAMILY = "Effusion"


def read_boxes_xyz(path):
    """Every row of a boxes.csv with its box as ints in the export (X, Y, Z) frame."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            host = r["host_label"].strip()
            rows.append({
                "ann_id": int(r["ann_id"]), "supercategory": r["supercategory"], "layer": r["layer"],
                "tissue_id": int(r["tissue_id"]), "host_label": int(host) if host else None,
                "host_side": r["host_side"],
                "box": tuple(int(r[k]) for k in ("x0", "y0", "z0", "x1", "y1", "z1")),
            })
    return rows


def box_volume(box):
    x0, y0, z0, x1, y1, z1 = box
    return max(x1 - x0, 0) * max(y1 - y0, 0) * max(z1 - z0, 0)


def paint_order(rows):
    """Row indices in painting order: effusion first, then the others from largest to smallest."""
    return sorted(range(len(rows)),
                  key=lambda i: (rows[i]["supercategory"] != EFFUSION_FAMILY, -box_volume(rows[i]["box"])))


def _clip(box, shape):
    x0, y0, z0, x1, y1, z1 = box
    return (max(x0, 0), max(y0, 0), max(z0, 0), min(x1, shape[0]), min(y1, shape[1]), min(z1, shape[2]))


def boxes_to_label_map(rows, shape):
    lab = np.zeros(tuple(int(s) for s in shape), np.uint8)
    for i in paint_order(rows):
        x0, y0, z0, x1, y1, z1 = _clip(rows[i]["box"], lab.shape)
        if x1 > x0 and y1 > y0 and z1 > z0:
            lab[x0:x1, y0:y1, z0:z1] = FAMILY_LABELS[rows[i]["supercategory"]]
    return lab
