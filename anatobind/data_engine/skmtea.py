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


# --- segmentation frame and loaders -------------------------------------------------------------
import h5py  # noqa: E402
import nibabel as nib  # noqa: E402


def seg_nifti_to_h5_frame(arr, orientation=("SI", "AP", "LR")):
    """The dicom-track NIfTI stores (y, x, z) relative to the h5 (x, y, z) grid.

    Scans whose annotation metadata lists the third axis as "RL" (18 of 155) are
    additionally mirrored along that axis; verified on all 18 (2026-09-07).
    """
    out = np.transpose(arr, (1, 0, 2))
    if tuple(orientation)[2] == "RL":
        out = np.flip(out, axis=2)
    return out


def load_seg_h5_frame(nii_path, orientation=("SI", "AP", "LR")):
    return np.ascontiguousarray(seg_nifti_to_h5_frame(np.asarray(nib.load(str(nii_path)).dataobj), orientation)).astype(np.uint8)


def load_orientations(annotation_dir):
    """scan_id -> orientation tuple from the three split JSON files."""
    out = {}
    for split in ("train", "val", "test"):
        doc = json.loads((Path(annotation_dir) / f"{split}.json").read_text())
        for im in doc["images"]:
            out[im["scan_id"]] = tuple(im["orientation"])
    return out


def load_target_magnitude(h5_path, echo):
    with h5py.File(h5_path, "r") as f:
        return np.abs(f["target"][:, :, :, echo, 0]).astype(np.float32)


def tissue_contrast(magnitude, seg_h5):
    inside = seg_h5 > 0
    return float(magnitude[inside].mean() / magnitude[~inside].mean())


# --- host-side resolution -------------------------------------------------------------------------
def host_seg_label(seg_h5, box, tissue_id, pad=4, ambiguous=(0.4, 0.6)):
    """Which segmentation label hosts a lesion box (medial/lateral resolved by overlap)."""
    labels = TISSUE_TO_SEG.get(tissue_id, ())
    if not labels:
        return {"label": None, "side": "none", "ratio": None, "n_voxels": 0}
    x0, y0, z0, x1, y1, z1 = box
    sub = seg_h5[max(x0 - pad, 0):x1 + pad, max(y0 - pad, 0):y1 + pad, max(z0 - pad, 0):z1 + pad]
    counts = {l: int((sub == l).sum()) for l in labels}
    n = sum(counts.values())
    if n == 0:
        return {"label": None, "side": "none", "ratio": None, "n_voxels": 0}
    if len(labels) == 1:
        return {"label": labels[0], "side": "single", "ratio": 1.0, "n_voxels": n}
    medial, lateral = labels  # (5, 6) or (3, 4): medial label listed first
    ratio = counts[medial] / n
    if ratio >= ambiguous[1]:
        return {"label": medial, "side": "medial", "ratio": ratio, "n_voxels": n}
    if ratio <= ambiguous[0]:
        return {"label": lateral, "side": "lateral", "ratio": ratio, "n_voxels": n}
    return {"label": None, "side": "ambiguous", "ratio": ratio, "n_voxels": n}
