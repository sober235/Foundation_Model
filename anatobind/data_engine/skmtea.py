"""SKM-TEA annotations, inclusion rule D5, segmentation frame and host-side resolution."""
import json
from pathlib import Path

import numpy as np

SEG_LABELS = {
    1: "patellar_cartilage", 2: "femoral_cartilage",
    3: "tibial_cartilage_medial", 4: "tibial_cartilage_lateral",
    5: "meniscus_medial", 6: "meniscus_lateral",
}
TISSUE_NAMES = {1: "Meniscus", 2: "ACL", 3: "PCL", 4: "Femoral Cartilage", 5: "Patellar Cartilage",
                6: "Tibial Cartilage", -1: "none"}
TISSUE_TO_SEG = {1: (5, 6), 4: (2,), 5: (1,), 6: (3, 4), 2: (), 3: (), -1: ()}
SUPER_TO_TISSUES = {"Meniscal Tear": {1}, "Ligament Tear": {2, 3}, "Cartilage Lesion": {4, 5, 6}, "Effusion": {-1}}


def layer_of(tissue_id):
    if tissue_id in (1, 4, 5, 6):
        return "in_seg"
    if tissue_id == -1:
        return "effusion"
    return "ligament"


def normalise_box(bbox):
    """[x, y, z, w, h, d] -> integer (x0, y0, z0, x1, y1, z1) with sorted endpoints."""
    x, y, z, w, h, d = (int(round(float(v))) for v in bbox)
    lo = (min(x, x + w), min(y, y + h), min(z, z + d))
    hi = (max(x, x + w), max(y, y + h), max(z, z + d))
    return lo + hi


def screen_annotation(ann, image, supercategory):
    """Apply rule D5 to one record; the box is returned in h5 array coordinates."""
    w, h, d = (float(v) for v in ann["bbox"][3:6])
    shape = tuple(int(v) for v in image["matrix_shape"])
    if w <= 0 or h <= 0 or d <= 0:
        return {"keep": False, "reason": "non-positive extent", "box": None}
    box = normalise_box(ann["bbox"])
    if min(box[:3]) < 0 or any(box[3 + i] > shape[i] for i in range(3)):
        return {"keep": False, "reason": "out of bounds", "box": None}
    if ann["tissue_id"] not in SUPER_TO_TISSUES.get(supercategory, set()):
        return {"keep": False, "reason": "category-tissue mismatch", "box": None}
    return {"keep": True, "reason": "", "box": box}


def screen_split(json_path):
    json_path = Path(json_path)
    doc = json.loads(json_path.read_text())
    supercat = {c["id"]: c["supercategory"] for c in doc["categories"]}
    images = {im["id"]: im for im in doc["images"]}
    rows = []
    for ann in doc["annotations"]:
        im = images[ann["image_id"]]
        res = screen_annotation(ann, im, supercat[ann["category_id"]])
        box = res["box"] if res["box"] is not None else normalise_box(ann["bbox"])
        rows.append({
            "split": json_path.stem, "ann_id": ann["id"], "scan_id": im["scan_id"], "image_id": ann["image_id"],
            "category_id": ann["category_id"], "supercategory": supercat[ann["category_id"]],
            "tissue_id": ann["tissue_id"], "layer": layer_of(ann["tissue_id"]),
            "keep": res["keep"], "reason": res["reason"],
            "x0": box[0], "y0": box[1], "z0": box[2], "x1": box[3], "y1": box[4], "z1": box[5],
            "depth": int(im["matrix_shape"][2]),
        })
    return rows
