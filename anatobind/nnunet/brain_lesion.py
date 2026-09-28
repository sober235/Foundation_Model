"""nnU-Net v2 dataset for the brain small-lesion detector (spec 2026-09-28 §2).

Cases are the 165 Gate 0.5 registry volumes (1297 small lesions) and the 88 FLAIR volumes whose every fastMRI+ row
is "Normal for age". Each registry lesion's per-slice member boxes (RSS frame, the Level R export path) are painted
into a binary map on the raw RSS grid (col, row, slice); normal volumes are all zero. Lesion patients keep the Level R
fold table; normal patients are dealt into five folds with seed 0; the two sets must not overlap. nnU-Net's custom
splits make its cross-validation outputs out-of-fold predictions.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ID = 903
DATASET_NAME = f"Dataset{DATASET_ID}_FastMRIBrainSmallLesion"
TRAINER = "nnUNetTrainer_250epochs"
LABELS = {"background": 0, "small_lesion": 1}
FAMILY = "small_lesion"
FAMILIES = {1: FAMILY}
NORMAL_LABEL = "Normal for age"


def normal_files(csv_rows):
    labels = {}
    for r in csv_rows:
        if "AXFLAIR" in r["file"]:
            labels.setdefault(r["file"], set()).add(r["label"].strip())
    return sorted(f for f, s in labels.items() if s == {NORMAL_LABEL})


def check_members_inside(members_of, shape):
    nc, nr, ns = shape
    for lid, members in members_of.items():
        for m in members:
            if not (0 <= m["x"] and m["x"] + m["width"] <= nc and 0 <= m["y"] and m["y"] + m["height"] <= nr
                    and 0 <= m["slice"] < ns):
                raise ValueError(f"lesion {lid}: member box {m} leaves the grid {shape}")


def paint_members(members_of, shape):
    lab = np.zeros(shape, np.uint8)
    for members in members_of.values():
        for m in members:
            lab[m["x"]:m["x"] + m["width"], m["y"]:m["y"] + m["height"], m["slice"]] = 1
    return lab


def gt_boxes(registry_rows):
    """Registry boxes as (x0, y0, z0, x1, y1, z1 + 1) on the (col, row, slice) grid; x1 / y1 are already exclusive."""
    return [{"lesion_id": r["lesion_id"], "family": FAMILY, "box": (r["x0"], r["y0"], r["z0"], r["x1"], r["y1"], r["z1"] + 1)}
            for r in registry_rows]


def assign_normal_folds(patients, k=5, seed=0):
    ids = sorted(set(patients))
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {ids[i]: int(j % k) for j, i in enumerate(perm)}


def case_folds(lesion_cases, normal_cases, lesion_patient_fold, normal_patient_fold):
    overlap = sorted(set(lesion_cases.values()) & set(normal_cases.values()))
    if overlap:
        raise ValueError(f"patients {overlap} are both lesion and normal patients")
    out = {c: int(lesion_patient_fold[p]) for c, p in lesion_cases.items()}
    out.update({c: int(normal_patient_fold[p]) for c, p in normal_cases.items()})
    return out


def make_splits(case_fold, k=5):
    cases = sorted(case_fold)
    return [{"train": [c for c in cases if case_fold[c] != f], "val": [c for c in cases if case_fold[c] == f]} for f in range(k)]


def write_label(label, image_path, out_path):
    img = nib.load(str(image_path))
    if tuple(img.shape) != tuple(label.shape):
        raise ValueError(f"label shape {label.shape} != image shape {img.shape} ({image_path})")
    out = nib.Nifti1Image(np.asarray(label, np.uint8), img.affine)
    out.set_data_dtype("uint8")
    out.header.set_xyzt_units("mm")
    nib.save(out, str(out_path))


def build_raw(raw_root, cases, write_case):
    base = Path(raw_root) / DATASET_NAME
    if base.exists():
        raise FileExistsError(f"{base} exists; the dataset is never rebuilt in place")
    (base / "imagesTr").mkdir(parents=True)
    (base / "labelsTr").mkdir()
    for c in cases:
        write_case(c, base / "imagesTr" / f"{c}_0000.nii.gz", base / "labelsTr" / f"{c}.nii.gz")
    meta = {"channel_names": {"0": "FLAIR"}, "labels": LABELS, "numTraining": len(cases), "file_ending": ".nii.gz"}
    (base / "dataset.json").write_text(json.dumps(meta, indent=1))
    return base


def write_splits(preprocessed_root, case_fold, k=5):
    d = Path(preprocessed_root) / DATASET_NAME
    d.mkdir(parents=True, exist_ok=True)
    p = d / "splits_final.json"
    if p.exists():
        raise FileExistsError(f"{p} exists")
    p.write_text(json.dumps(make_splits(case_fold, k), indent=1))
    return p


def validation_path(results_root, config, fold, case):
    return Path(results_root) / DATASET_NAME / f"{TRAINER}__nnUNetPlans__{config}" / f"fold_{fold}" / "validation" / f"{case}.nii.gz"


def validation_npz_path(results_root, config, fold, case):
    return validation_path(results_root, config, fold, case).with_suffix("").with_suffix(".npz")
