import json
import os

import nibabel as nib
import numpy as np
from synth import write_synthetic_export

from anatobind.nnunet.lesion_labels import LABELS
from anatobind.nnunet.prepare import make_splits
from anatobind.nnunet.prepare_lesion import (
    DATASET_NAME, build_raw, validation_npz_path, validation_path, write_splits,
)

FOLDS = {"MTR_001": 0, "MTR_002": 1}


def test_the_raw_dataset_links_images_and_shares_one_label_map_per_scan(tmp_path):
    exp = tmp_path / "m1r"
    write_synthetic_export(exp, list(FOLDS), FOLDS)
    assert build_raw(exp, tmp_path / "raw", list(FOLDS)) == 24
    base = tmp_path / "raw" / DATASET_NAME
    assert len(list((base / "imagesTr").iterdir())) == 24 and len(list((base / "labelsTr").iterdir())) == 24
    ino = lambda p: os.stat(p).st_ino  # noqa: E731
    assert ino(base / "imagesTr/MTR_001_us16_0000.nii.gz") == ino(exp / "MTR_001/image_us16_e1.nii.gz")
    assert ino(base / "labelsTr/MTR_001_clean0.nii.gz") == ino(base / "labelsTr/MTR_001_noise_q3.nii.gz")
    assert ino(base / "labelsTr/MTR_001_clean0.nii.gz") != ino(base / "labelsTr/MTR_002_clean0.nii.gz")
    lab = nib.load(str(base / "labelsTr/MTR_002_us4.nii.gz"))
    seg = nib.load(str(exp / "MTR_002/seg.nii.gz"))
    assert lab.shape == seg.shape and np.allclose(lab.affine, seg.affine) and lab.get_data_dtype() == np.uint8
    arr = np.asanyarray(lab.dataobj)
    assert arr[12, 15, 5] == 1 and arr[35, 30, 7] == 2 and arr[22, 48, 15] == 3 and arr[50, 45, 15] == 4
    assert arr[0, 0, 0] == 0
    ds = json.loads((base / "dataset.json").read_text())
    assert ds == {"channel_names": {"0": "qDESS_E1"}, "labels": LABELS, "numTraining": 24, "file_ending": ".nii.gz"}


def test_splits_and_validation_paths_mirror_dataset901(tmp_path):
    write_splits(tmp_path / "pre", FOLDS, n_folds=2)
    assert json.loads((tmp_path / "pre" / DATASET_NAME / "splits_final.json").read_text()) == make_splits(FOLDS, 2)
    p = validation_path(tmp_path, 3, "MTR_001", "clean")
    assert p == tmp_path / DATASET_NAME / "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres" / "fold_3" / "validation" / "MTR_001_clean0.nii.gz"
    assert validation_npz_path(tmp_path, 0, "MTR_001", "us8").name == "MTR_001_us8.npz"
