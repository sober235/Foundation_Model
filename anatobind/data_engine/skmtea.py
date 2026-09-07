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


# --- per-scan export for M1 -----------------------------------------------------------------------
import csv  # noqa: E402

from anatobind.data_engine.resample import downsample2_inplane_image, downsample2_inplane_labels, scale_box_inplane  # noqa: E402
from anatobind.data_engine.skmtea_recon import add_complex_noise, adjoint_sense, embed_poisson, noise_sigma, undersample  # noqa: E402

SPACING_FULL = (0.3125, 0.3125, 0.8)
BOX_FIELDS = ["ann_id", "split", "layer", "supercategory", "category_id", "tissue_id", "host_label", "host_side",
              "host_ratio", "x0", "y0", "z0", "x1", "y1", "z1", "x0_full", "y0_full", "z0_full", "x1_full", "y1_full", "z1_full"]


def _affine(spacing):
    return np.diag([spacing[0], spacing[1], spacing[2], 1.0])


def _save(vol, spacing, path):
    img = nib.Nifti1Image(np.ascontiguousarray(vol), _affine(spacing))
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))


def export_scan(h5_path, seg_nii_path, box_rows, out_dir, conditions, rng_seed, orientation=("SI", "AP", "LR")):
    """Write the M1 files for one scan; returns the manifest row."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spacing = (SPACING_FULL[0] * 2, SPACING_FULL[1] * 2, SPACING_FULL[2])
    files = []
    with h5py.File(h5_path, "r") as f:
        for echo in (0, 1):
            mag = np.abs(f["target"][:, :, :, echo, 0]).astype(np.float32)
            p = out_dir / f"image_clean_e{echo + 1}.nii.gz"
            _save(downsample2_inplane_image(mag), spacing, p)
            files.append(p.name)
        k = f["kspace"][:, :, :, 0, :]
        maps = f["maps"][:, :, :, :, 0]
        rng = np.random.default_rng(rng_seed)
        for q, frac in enumerate(conditions["noise"], start=1):
            rec = np.abs(adjoint_sense(add_complex_noise(k, noise_sigma(k, frac), rng), maps)).astype(np.float32)
            p = out_dir / f"image_noise_q{q}_e1.nii.gz"
            _save(downsample2_inplane_image(rec), spacing, p)
            files.append(p.name)
        for r in conditions["us"]:
            mask = embed_poisson(f[f"masks/poisson_{r}.0x"][()], ky=k.shape[1], kz=k.shape[2])
            rec = np.abs(adjoint_sense(undersample(k, mask), maps)).astype(np.float32)
            p = out_dir / f"image_us{r}_e1.nii.gz"
            _save(downsample2_inplane_image(rec), spacing, p)
            files.append(p.name)
    seg_h5 = load_seg_h5_frame(seg_nii_path, orientation)
    _save(downsample2_inplane_labels(seg_h5), spacing, out_dir / "seg.nii.gz")
    files.append("seg.nii.gz")
    rows, n_amb = [], 0
    for b in box_rows:
        if not b["keep"]:
            continue
        full = (b["x0"], b["y0"], b["z0"], b["x1"], b["y1"], b["z1"])
        host = host_seg_label(seg_h5, full, b["tissue_id"])
        n_amb += host["side"] == "ambiguous"
        half = scale_box_inplane(full)
        rows.append({
            "ann_id": b["ann_id"], "split": b["split"], "layer": b["layer"], "supercategory": b["supercategory"],
            "category_id": b["category_id"], "tissue_id": b["tissue_id"],
            "host_label": "" if host["label"] is None else host["label"], "host_side": host["side"],
            "host_ratio": "" if host["ratio"] is None else f"{host['ratio']:.3f}",
            "x0": half[0], "y0": half[1], "z0": half[2], "x1": half[3], "y1": half[4], "z1": half[5],
            "x0_full": full[0], "y0_full": full[1], "z0_full": full[2], "x1_full": full[3], "y1_full": full[4], "z1_full": full[5],
        })
    with open(out_dir / "boxes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BOX_FIELDS)
        w.writeheader()
        w.writerows(rows)
    files.append("boxes.csv")
    return {"scan_id": Path(h5_path).name[:-3], "out_dir": str(out_dir), "n_boxes_kept": len(rows),
            "n_ambiguous": n_amb, "files": files}
