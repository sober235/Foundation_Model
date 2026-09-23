"""nnU-Net v2 dataset for the knee lesion detector (spec 2026-09-23 §3.1, decision N1).

Same images, views, folds and case names as Dataset901 (anatobind/nnunet/prepare.py): each scan
contributes its six degraded views once and its clean view six times, all hard-linked. The label of
every case of a scan is the same box-filled map (anatobind/nnunet/lesion_labels.py), written once
and hard-linked eleven times.
"""
import json
from pathlib import Path

import nibabel as nib

from anatobind.nnunet.lesion_labels import LABELS, boxes_to_label_map, read_boxes_xyz
from anatobind.nnunet.prepare import TRAINER_DIR, _link, make_splits, train_cases, view_of_case
from anatobind.train.cache import VIEW_FILES

DATASET_ID = 902
DATASET_NAME = f"Dataset{DATASET_ID}_SKMTEAlesion"


def write_label_map(export_dir, out_path):
    seg = nib.load(str(Path(export_dir) / "seg.nii.gz"))
    lab = boxes_to_label_map(read_boxes_xyz(Path(export_dir) / "boxes.csv"), seg.shape)
    img = nib.Nifti1Image(lab, seg.affine)
    img.set_data_dtype("uint8")
    img.header.set_xyzt_units("mm")
    nib.save(img, str(out_path))


def build_raw(export_root, raw_root, scans):
    base = Path(raw_root) / DATASET_NAME
    images, labels = base / "imagesTr", base / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    n = 0
    for scan in scans:
        src = Path(export_root) / scan
        cases = train_cases(scan)
        first = labels / f"{cases[0]}.nii.gz"
        if not first.exists():
            write_label_map(src, first)
        for case in cases:
            _link(src / VIEW_FILES[view_of_case(case)], images / f"{case}_0000.nii.gz")
            _link(first, labels / f"{case}.nii.gz")
            n += 1
    meta = {"channel_names": {"0": "qDESS_E1"}, "labels": LABELS, "numTraining": n, "file_ending": ".nii.gz"}
    (base / "dataset.json").write_text(json.dumps(meta, indent=1))
    return n


def write_splits(preprocessed_root, folds, n_folds=5):
    d = Path(preprocessed_root) / DATASET_NAME
    d.mkdir(parents=True, exist_ok=True)
    (d / "splits_final.json").write_text(json.dumps(make_splits(folds, n_folds), indent=1))


def _case(scan, view):
    return f"{scan}_clean0" if view == "clean" else f"{scan}_{view}"


def validation_path(results_root, fold, scan, view):
    return Path(results_root) / DATASET_NAME / TRAINER_DIR / f"fold_{fold}" / "validation" / f"{_case(scan, view)}.nii.gz"


def validation_npz_path(results_root, fold, scan, view):
    return validation_path(results_root, fold, scan, view).with_suffix("").with_suffix(".npz")
