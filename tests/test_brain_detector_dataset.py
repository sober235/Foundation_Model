import json

import nibabel as nib
import numpy as np
import pytest

from anatobind.nnunet.brain_lesion import (
    DATASET_NAME, assign_normal_folds, build_raw, case_folds, check_members_inside, gt_boxes, make_splits,
    normal_files, paint_members, validation_npz_path, validation_path, write_label, write_splits,
)


def _m(x, y, w, h, s):
    return {"x": x, "y": y, "width": w, "height": h, "slice": s}


def test_normal_files_need_every_row_to_be_normal_and_flair():
    rows = [{"file": "file_brain_AXFLAIR_1", "label": "Normal for age"},
            {"file": "file_brain_AXFLAIR_2", "label": "Normal for age"}, {"file": "file_brain_AXFLAIR_2", "label": "Mass "},
            {"file": "file_brain_AXT1_3", "label": "Normal for age"}, {"file": "file_brain_AXFLAIR_4", "label": " Normal for age "}]
    assert normal_files(rows) == ["file_brain_AXFLAIR_1", "file_brain_AXFLAIR_4"]


def test_paint_and_inside_check():
    members = {0: [_m(2, 3, 4, 2, 1), _m(2, 3, 4, 3, 2)], 1: [_m(10, 10, 2, 2, 0)]}
    lab = paint_members(members, (16, 16, 4))
    assert lab.dtype == np.uint8 and lab.sum() == 8 + 12 + 4 and lab[2:6, 3:5, 1].all() and lab[10:12, 10:12, 0].all()
    check_members_inside(members, (16, 16, 4))
    with pytest.raises(ValueError, match="lesion 7"):
        check_members_inside({7: [_m(14, 0, 4, 2, 0)]}, (16, 16, 4))
    with pytest.raises(ValueError, match="lesion 8"):
        check_members_inside({8: [_m(0, 0, 2, 2, 4)]}, (16, 16, 4))


def test_gt_boxes_close_the_inclusive_slice_range():
    rows = [{"lesion_id": 5, "x0": 2, "y0": 3, "x1": 6, "y1": 6, "z0": 1, "z1": 2}]
    assert gt_boxes(rows) == [{"lesion_id": 5, "family": "small_lesion", "box": (2, 3, 1, 6, 6, 3)}]


def test_folds_by_patient_and_overlap_refused():
    nf = assign_normal_folds(["n3", "n1", "n2", "n4", "n5", "n6"], 5, 0)
    assert set(nf) == {"n1", "n2", "n3", "n4", "n5", "n6"} and set(nf.values()) <= set(range(5)) and len(set(nf.values())) == 5
    assert assign_normal_folds(["n1", "n2", "n3", "n4", "n5", "n6"], 5, 0) == nf
    cf = case_folds({"a": "p1", "b": "p2"}, {"c": "n1"}, {"p1": 0, "p2": 3}, nf)
    assert cf == {"a": 0, "b": 3, "c": nf["n1"]}
    with pytest.raises(ValueError, match="p1"):
        case_folds({"a": "p1"}, {"c": "p1"}, {"p1": 0}, {"p1": 1})
    sp = make_splits({"a": 0, "b": 1, "c": 0}, 2)
    assert sp == [{"train": ["b"], "val": ["a", "c"]}, {"train": ["a", "c"], "val": ["b"]}]


def test_build_raw_writes_layout_and_refuses_existing(tmp_path):
    def write_case(case, img, lab):
        img.write_text("i")
        lab.write_text("l")
    base = build_raw(tmp_path, ["c1", "c2"], write_case)
    assert base.name == DATASET_NAME and (base / "imagesTr" / "c1_0000.nii.gz").exists() and (base / "labelsTr" / "c2.nii.gz").exists()
    meta = json.loads((base / "dataset.json").read_text())
    assert meta == {"channel_names": {"0": "FLAIR"}, "labels": {"background": 0, "small_lesion": 1}, "numTraining": 2, "file_ending": ".nii.gz"}
    with pytest.raises(FileExistsError):
        build_raw(tmp_path, ["c1"], write_case)


def test_write_label_shares_affine_and_shape_and_splits_refuse_overwrite(tmp_path):
    aff = np.diag([0.5, 0.5, 5.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 3), np.float32), aff), str(tmp_path / "img.nii.gz"))
    lab = np.zeros((8, 8, 3), np.uint8)
    lab[1:3, 1:3, 1] = 1
    write_label(lab, tmp_path / "img.nii.gz", tmp_path / "lab.nii.gz")
    back = nib.load(str(tmp_path / "lab.nii.gz"))
    assert np.allclose(back.affine, aff) and back.get_data_dtype() == np.uint8 and np.array_equal(np.asanyarray(back.dataobj), lab)
    with pytest.raises(ValueError):
        write_label(np.zeros((8, 8, 4), np.uint8), tmp_path / "img.nii.gz", tmp_path / "lab2.nii.gz")
    p = write_splits(tmp_path, {"a": 0, "b": 1}, 2)
    assert json.loads(p.read_text())[0]["val"] == ["a"]
    with pytest.raises(FileExistsError):
        write_splits(tmp_path, {"a": 0, "b": 1}, 2)


def test_validation_paths():
    p = validation_path("/r", "2d", 3, "case")
    assert str(p) == f"/r/{DATASET_NAME}/nnUNetTrainer_250epochs__nnUNetPlans__2d/fold_3/validation/case.nii.gz"
    assert str(validation_npz_path("/r", "2d", 3, "case")).endswith("validation/case.npz")
