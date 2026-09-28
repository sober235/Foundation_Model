# tests/test_nndet_task.py
import json

import nibabel as nib
import numpy as np
import pytest

from anatobind.nndet.brain_task import (
    DATASET_META, TASK_NAME, build_task, changed_boxes, check_same_support, instances_json, paint_instances,
    read_splits_pkl, splits_payload, write_instance_label, write_splits_pkl,
)


def _box(x, y, w, h, s):
    return {"x": x, "y": y, "width": w, "height": h, "slice": s}


def test_largest_first_keeps_a_small_lesion_inside_a_big_one():
    members_of = {7: [_box(2, 2, 10, 10, 1)], 3: [_box(4, 4, 2, 2, 1)]}       # lesion 3 sits inside lesion 7
    inst, instance_of = paint_instances(members_of, (20, 20, 3))
    assert instance_of == {7: 1, 3: 2}
    assert inst.dtype == np.uint16
    assert int((inst == 2).sum()) == 4 and int((inst == 1).sum()) == 100 - 4
    assert changed_boxes(members_of, inst, instance_of) == []                 # 7 keeps its corners


def test_ties_paint_the_smaller_lesion_id_first():
    members_of = {5: [_box(0, 0, 2, 2, 0)], 2: [_box(10, 10, 2, 2, 0)]}
    _, instance_of = paint_instances(members_of, (20, 20, 1))
    assert instance_of == {2: 1, 5: 2}


def test_a_lesion_covered_by_later_instances_is_refused():
    members_of = {1: [_box(0, 0, 2, 2, 0)], 2: [_box(0, 0, 1, 2, 0)], 3: [_box(1, 0, 1, 2, 0)]}
    with pytest.raises(ValueError, match="lesion 1 "):
        paint_instances(members_of, (4, 4, 1))


def test_changed_boxes_lists_a_lesion_whose_edge_was_overwritten():
    members_of = {7: [_box(2, 2, 4, 4, 0)], 3: [_box(2, 2, 1, 4, 0)]}        # 3 covers 7's whole first column
    inst, instance_of = paint_instances(members_of, (10, 10, 1))
    assert changed_boxes(members_of, inst, instance_of) == [7]


def test_support_must_equal_the_binary_label():
    inst = np.zeros((4, 4, 1), np.uint16)
    inst[1, 1, 0] = 1
    binary = (inst > 0).astype(np.uint8)
    check_same_support(inst, binary, "c")
    binary[2, 2, 0] = 1
    with pytest.raises(ValueError, match="^c:"):
        check_same_support(inst, binary, "c")


def test_instances_json_maps_every_instance_to_class_zero():
    assert instances_json({7: 1, 3: 2}) == {"instances": {"1": 0, "2": 0}}
    assert instances_json({}) == {"instances": {}}


def test_instance_label_shares_the_image_affine_and_refuses_a_shape_mismatch(tmp_path):
    aff = np.diag([0.6875, 0.6875, 5.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 2), np.float32), aff), str(tmp_path / "img.nii.gz"))
    inst = np.zeros((4, 5, 2), np.uint16)
    inst[1, 2, 1] = 3
    write_instance_label(inst, tmp_path / "img.nii.gz", tmp_path / "lab.nii.gz")
    lab = nib.load(str(tmp_path / "lab.nii.gz"))
    assert np.allclose(lab.affine, aff) and lab.get_data_dtype() == np.uint16
    assert np.asarray(lab.dataobj)[1, 2, 1] == 3
    with pytest.raises(ValueError, match="shape"):
        write_instance_label(np.zeros((4, 4, 2), np.uint16), tmp_path / "img.nii.gz", tmp_path / "lab2.nii.gz")


def test_build_task_layout_and_refusal(tmp_path):
    seen = []

    def write_case(case, img, labels_dir):
        seen.append((case, img.name, labels_dir.name))

    base = build_task(tmp_path, ["a", "b"], write_case)
    assert base == tmp_path / TASK_NAME
    assert (base / "raw_splitted/imagesTr").is_dir() and (base / "raw_splitted/labelsTr").is_dir()
    assert seen == [("a", "a_0000.nii.gz", "labelsTr"), ("b", "b_0000.nii.gz", "labelsTr")]
    assert json.loads((base / "dataset.json").read_text()) == DATASET_META
    assert DATASET_META["dim"] == 3 and DATASET_META["labels"] == {"0": "small_lesion"}
    with pytest.raises(FileExistsError):
        build_task(tmp_path, ["a"], write_case)


def test_splits_pickle_is_plain_lists_and_refuses_overwrite(tmp_path):
    sj = [{"train": ["b", "c"], "val": ["a"]}, {"train": ["a"], "val": ["b", "c"]}]
    with pytest.raises(FileNotFoundError):
        write_splits_pkl(tmp_path / "missing", sj)
    p = write_splits_pkl(tmp_path, sj)
    assert b"numpy" not in p.read_bytes()
    assert read_splits_pkl(p) == splits_payload(sj) == sj
    with pytest.raises(FileExistsError):
        write_splits_pkl(tmp_path, sj)
