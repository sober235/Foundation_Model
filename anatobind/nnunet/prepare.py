"""nnU-Net v2 dataset for the G2 reference segmenter (RESEARCH_PLAN v2.2 13.6).

Same views, folds and clean share as the upstream: each scan contributes its six degraded views
once and its clean view six times (hard links), so clean images are half of the training cases.
Each fold's validation list holds the held-out scans' seven distinct views, so nnU-Net's own
end-of-training validation, which runs with the final weights, writes exactly the held-out
segmentations G2 needs to fold_<f>/validation/<case>.nii.gz.
"""
import json
import os
from pathlib import Path

from anatobind.train.cache import VIEW_FILES

DATASET_ID = 901
DATASET_NAME = f"Dataset{DATASET_ID}_SKMTEAm1r"
TRAINER = "nnUNetTrainer_250epochs"   # user decision 2026-09-12: 1000 epochs cost 14-18 days per fold on a shared GPU
TRAINER_DIR = f"{TRAINER}__nnUNetPlans__3d_fullres"
LABELS = {"background": 0, "patellar_cartilage": 1, "femoral_cartilage": 2, "tibial_cartilage_medial": 3,
          "tibial_cartilage_lateral": 4, "meniscus_medial": 5, "meniscus_lateral": 6}
CLEAN_COPIES = 6
DEGRADED = [v for v in VIEW_FILES if v != "clean"]


def train_cases(scan):
    return [f"{scan}_clean{i}" for i in range(CLEAN_COPIES)] + [f"{scan}_{v}" for v in DEGRADED]


def val_cases(scan):
    return [f"{scan}_clean0"] + [f"{scan}_{v}" for v in DEGRADED]


def view_of_case(case):
    tail = case.split("_", 2)[2]
    return "clean" if tail.startswith("clean") else tail


def _link(src, dst):
    if not dst.exists():
        os.link(src, dst)


def build_raw(export_root, raw_root, scans):
    base = Path(raw_root) / DATASET_NAME
    images, labels = base / "imagesTr", base / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    n = 0
    for scan in scans:
        src = Path(export_root) / scan
        for case in train_cases(scan):
            _link(src / VIEW_FILES[view_of_case(case)], images / f"{case}_0000.nii.gz")
            _link(src / "seg.nii.gz", labels / f"{case}.nii.gz")
            n += 1
    meta = {"channel_names": {"0": "qDESS_E1"}, "labels": LABELS, "numTraining": n, "file_ending": ".nii.gz"}
    (base / "dataset.json").write_text(json.dumps(meta, indent=1))
    return n


def make_splits(folds, n_folds=5):
    """folds: scan -> fold index. Returns nnU-Net's list of {"train": [...], "val": [...]}."""
    scans = sorted(folds)
    return [{"train": [c for s in scans if folds[s] != f for c in train_cases(s)],
             "val": [c for s in scans if folds[s] == f for c in val_cases(s)]} for f in range(n_folds)]


def write_splits(preprocessed_root, folds, n_folds=5):
    d = Path(preprocessed_root) / DATASET_NAME
    d.mkdir(parents=True, exist_ok=True)
    (d / "splits_final.json").write_text(json.dumps(make_splits(folds, n_folds), indent=1))


def validation_path(results_root, fold, scan, view):
    case = f"{scan}_clean0" if view == "clean" else f"{scan}_{view}"
    return Path(results_root) / DATASET_NAME / TRAINER_DIR / f"fold_{fold}" / "validation" / f"{case}.nii.gz"
