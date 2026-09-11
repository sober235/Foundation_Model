import json
import os

from synth import write_synthetic_export

from anatobind.nnunet.prepare import (
    CLEAN_COPIES, DATASET_NAME, LABELS, build_raw, make_splits, train_cases, val_cases, validation_path,
    view_of_case, write_splits,
)

FOLDS = {"MTR_001": 0, "MTR_002": 1}


def test_case_names_carry_the_view():
    assert train_cases("MTR_001")[:2] == ["MTR_001_clean0", "MTR_001_clean1"]
    assert len(train_cases("MTR_001")) == CLEAN_COPIES + 6 == 12
    assert val_cases("MTR_001") == ["MTR_001_clean0", "MTR_001_noise_q1", "MTR_001_noise_q2", "MTR_001_noise_q3",
                                    "MTR_001_us4", "MTR_001_us8", "MTR_001_us16"]
    assert view_of_case("MTR_001_clean3") == "clean" and view_of_case("MTR_001_noise_q2") == "noise_q2"


def test_the_raw_dataset_hard_links_images_and_labels(tmp_path):
    exp = tmp_path / "m1r"
    write_synthetic_export(exp, list(FOLDS), FOLDS)
    assert build_raw(exp, tmp_path / "raw", list(FOLDS)) == 24
    base = tmp_path / "raw" / DATASET_NAME
    assert len(list((base / "imagesTr").iterdir())) == 24 and len(list((base / "labelsTr").iterdir())) == 24
    ino = lambda p: os.stat(p).st_ino  # noqa: E731
    assert ino(base / "imagesTr/MTR_001_us16_0000.nii.gz") == ino(exp / "MTR_001/image_us16_e1.nii.gz")
    assert ino(base / "imagesTr/MTR_001_clean4_0000.nii.gz") == ino(exp / "MTR_001/image_clean_e1.nii.gz")
    assert ino(base / "labelsTr/MTR_002_us4.nii.gz") == ino(exp / "MTR_002/seg.nii.gz")
    ds = json.loads((base / "dataset.json").read_text())
    assert ds["numTraining"] == 24 and ds["labels"] == LABELS and ds["file_ending"] == ".nii.gz"


def test_splits_keep_every_case_of_a_scan_in_its_fold(tmp_path):
    splits = make_splits(FOLDS, n_folds=2)
    assert splits[0] == {"train": train_cases("MTR_002"), "val": val_cases("MTR_001")}
    assert splits[1] == {"train": train_cases("MTR_001"), "val": val_cases("MTR_002")}
    write_splits(tmp_path / "pre", FOLDS, n_folds=2)
    assert json.loads((tmp_path / "pre" / DATASET_NAME / "splits_final.json").read_text()) == splits


def test_the_validation_path_points_at_nnunets_final_validation_output(tmp_path):
    p = validation_path(tmp_path, 3, "MTR_001", "clean")
    assert p == tmp_path / DATASET_NAME / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_3" / "validation" / "MTR_001_clean0.nii.gz"
    assert validation_path(tmp_path, 0, "MTR_001", "us8").name == "MTR_001_us8.nii.gz"
