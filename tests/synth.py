"""A tiny export shaped like derived/skmtea/m1r: (X, Y, Z) NIfTIs at 0.625 x 0.625 x 0.8 mm."""
import csv
import json

import nibabel as nib
import numpy as np

SHAPE_XYZ = (64, 64, 30)          # model frame (30, 64, 64); depth pads to 32
VIEW_NAMES = ["image_clean_e1.nii.gz", "image_noise_q1_e1.nii.gz", "image_noise_q2_e1.nii.gz",
              "image_noise_q3_e1.nii.gz", "image_us4_e1.nii.gz", "image_us8_e1.nii.gz", "image_us16_e1.nii.gz"]
BOX_FIELDS = ["ann_id", "split", "layer", "supercategory", "category_id", "tissue_id", "host_label", "host_side",
              "host_ratio", "x0", "y0", "z0", "x1", "y1", "z1", "x0_full", "y0_full", "z0_full", "x1_full",
              "y1_full", "z1_full"]
MANIFEST_FIELDS = ["scan_id", "out_dir", "n_boxes_kept", "n_ambiguous", "n_no_overlap", "n_unresolved",
                   "spacing", "nrmse", "files", "status", "seconds"]
DEFAULT_BOXES = [
    {"ann_id": 1, "layer": "in_seg", "supercategory": "Cartilage Lesion", "tissue_id": 4, "host_label": 2,
     "host_side": "single", "x0": 10, "y0": 10, "z0": 3, "x1": 16, "y1": 20, "z1": 8},
    {"ann_id": 2, "layer": "in_seg", "supercategory": "Meniscal Tear", "tissue_id": 1, "host_label": 5,
     "host_side": "medial", "x0": 32, "y0": 24, "z0": 5, "x1": 40, "y1": 34, "z1": 9},
    {"ann_id": 3, "layer": "effusion", "supercategory": "Effusion", "tissue_id": -1, "host_label": "",
     "host_side": "none", "x0": 46, "y0": 40, "z0": 12, "x1": 56, "y1": 54, "z1": 20},
    {"ann_id": 4, "layer": "ligament", "supercategory": "Ligament Tear", "tissue_id": 2, "host_label": "",
     "host_side": "none", "x0": 20, "y0": 44, "z0": 13, "x1": 26, "y1": 52, "z1": 17},
]


def synthetic_seg(shape=SHAPE_XYZ):
    seg = np.zeros(shape, np.uint8)
    seg[8:20, 8:40, 2:12] = 2      # femoral cartilage
    seg[30:44, 20:50, 4:10] = 5    # medial meniscus
    seg[30:44, 20:50, 18:26] = 6   # lateral meniscus
    return seg


def write_boxes(path, boxes):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BOX_FIELDS)
        w.writeheader()
        for r in boxes:
            w.writerow({**{k: "" for k in BOX_FIELDS}, **r})


def write_synthetic_scan(export_root, scan, boxes, shape=SHAPE_XYZ, seed=0):
    d = export_root / scan
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    affine = np.diag([0.625, 0.625, 0.8, 1.0])
    seg = synthetic_seg(shape)
    nib.save(nib.Nifti1Image(seg, affine), str(d / "seg.nii.gz"))
    for name in VIEW_NAMES:
        img = (rng.gamma(2.0, 1e6, size=shape) * (1.0 + seg)).astype(np.float32)
        nib.save(nib.Nifti1Image(img, affine), str(d / name))
    write_boxes(d / "boxes.csv", boxes)
    return d


def write_synthetic_export(root, scans, folds, boxes=DEFAULT_BOXES):
    root.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(scans):
        write_synthetic_scan(root, s, boxes, seed=i)
    with open(root / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for s in scans:
            w.writerow({"scan_id": s, "out_dir": str(root / s), "status": "ok"})
    (root / "splits.json").write_text(json.dumps({"seed": 0, "folds": folds}))
