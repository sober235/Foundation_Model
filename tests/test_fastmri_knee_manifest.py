import csv
import json

import h5py
import numpy as np
import pytest

from anatobind.data_engine.fastmri import TRANSFORM_VERSION
from anatobind.data_engine.fastmri_knee import (
    LESION_FIELDS, MANIFEST_FIELDS, LegacyBoxConvention, assert_folds_by_patient, load_folds, load_lesions,
    load_manifest, manifest_row, patient_of, write_lesions, write_manifest,
)

HEADER = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding>'
    "<encodedSpace><matrixSize><x>640</x><y>368</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>280</x><y>161.42</y><z>4.5</z></fieldOfView_mm></encodedSpace>"
    "<reconSpace><matrixSize><x>320</x><y>320</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>140</x><y>140</y><z>3</z></fieldOfView_mm></reconSpace>"
    "</encoding></ismrmrdHeader>"
)


def _write_h5(path, patient="p1", rows=320, cols=320, slices=3, with_patient=True):
    with h5py.File(path, "w") as h:
        h.create_dataset("kspace", data=np.zeros((slices, 2, rows * 2, cols + 48), np.complex64))
        h.create_dataset("reconstruction_rss", data=np.zeros((slices, rows, cols), np.float32))
        h.create_dataset("ismrmrd_header", data=np.bytes_(HEADER.encode()))
        if with_patient:
            h.attrs["patient_id"] = patient
    return path


def _export(root, files, patients, version=TRANSFORM_VERSION):
    root.mkdir(parents=True, exist_ok=True)
    write_manifest(root / "manifest.csv",
                   [{"file": f, "patient_id": p, "transform_version": version, "status": "ok"} for f, p in zip(files, patients)])
    write_lesions(root / "lesions.csv", [{"file": files[0], "family": "meniscus", "z0": 1, "z1": 2,
                                          "x0": 4, "y0": 5, "x1": 12, "y1": 15, "n_boxes": 2, "members": []}])
    (root / "folds.json").write_text(json.dumps({"folds": {f: i % 2 for i, f in enumerate(files)}}))


def test_manifest_row_records_patient_grid_spacing_and_convention(tmp_path):
    h5 = _write_h5(tmp_path / "file1.h5")
    row = manifest_row("file1", h5, tmp_path / "out", n_lesions=2, status="ok")
    assert row["patient_id"] == "p1" and row["n_rows"] == 320 and row["n_cols"] == 320 and row["slices"] == 3
    assert row["spacing_slice_mm"] == pytest.approx(3.0) and row["spacing_row_mm"] == pytest.approx(0.4375)
    assert row["transform_version"] == TRANSFORM_VERSION and row["box_coordinate_convention"] == "rss_rows_from_top"
    assert set(row) == set(MANIFEST_FIELDS)


def test_write_lesions_numbers_them_and_drops_members(tmp_path):
    write_lesions(tmp_path / "lesions.csv", [
        {"file": "b", "family": "meniscus", "z0": 1, "z1": 2, "x0": 4, "y0": 5, "x1": 12, "y1": 15, "n_boxes": 2, "members": [1]},
        {"file": "a", "family": "effusion", "z0": 0, "z1": 0, "x0": 1, "y0": 1, "x1": 5, "y1": 5, "n_boxes": 1, "members": [1]}])
    with open(tmp_path / "lesions.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == LESION_FIELDS
    assert [(r["lesion_id"], r["file"]) for r in rows] == [("0", "a"), ("1", "b")]


def test_a_manifest_without_transform_version_is_refused(tmp_path):
    root = tmp_path / "leg2"
    root.mkdir()
    (root / "manifest.csv").write_text("file,out_dir,slices,n_lesions,status\nfile1,x,3,1,ok\n")   # the 2026-09-14 layout
    with pytest.raises(LegacyBoxConvention, match="transform_version"):
        load_manifest(root)


def test_version_one_is_refused_and_version_two_loads_with_integers(tmp_path):
    _export(tmp_path / "v1", ["file1"], ["p1"], version=1)
    with pytest.raises(LegacyBoxConvention):
        load_lesions(tmp_path / "v1")
    _export(tmp_path / "v2", ["file1"], ["p1"])
    lesions = load_lesions(tmp_path / "v2")
    assert lesions[0]["y0"] == 5 and isinstance(lesions[0]["z1"], int) and lesions[0]["n_boxes"] == 2


def test_folds_that_split_a_patient_are_rejected():
    with pytest.raises(ValueError, match="patient-disjoint"):
        assert_folds_by_patient({"a": 0, "b": 1}, {"a": "p", "b": "p"})
    assert_folds_by_patient({"a": 0, "b": 0, "c": 1}, {"a": "p", "b": "p", "c": "q"})   # no raise


def test_load_folds_checks_patients_from_the_manifest(tmp_path):
    _export(tmp_path / "ok", ["file1", "file2"], ["p1", "p2"])
    assert load_folds(tmp_path / "ok") == {"file1": 0, "file2": 1}
    _export(tmp_path / "bad", ["file1", "file2"], ["p1", "p1"])          # one patient, two folds
    with pytest.raises(ValueError, match="patient-disjoint"):
        load_folds(tmp_path / "bad")


def test_patient_of_refuses_a_file_without_patient_id(tmp_path):
    with_id = _write_h5(tmp_path / "a.h5", patient="pa")
    without = _write_h5(tmp_path / "b.h5", with_patient=False)
    assert patient_of({"a": with_id}) == {"a": "pa"}
    with pytest.raises(KeyError, match="patient_id"):
        patient_of({"b": without})
