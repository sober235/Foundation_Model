# tests/test_brain_anatomy_labels.py
import numpy as np
import pytest

import anatobind.anatomy.labels as L
from anatobind.eval.geometry import CLASS_NAMES, HOST_CLASSES, LANDMARKS, host_class_map


def test_the_label_space_is_consistent_with_the_host_classes():
    assert L.check_consistency()
    assert L.N_CLASSES == 16 and L.IGNORE == 15 and len(L.STUDENT) == 15
    assert L.LABELS_JSON["background"] == 0 and L.LABELS_JSON["ignore"] == 15 and L.LABELS_JSON["ventricles"] == 14
    assert L.HOST_CLASS_OF[9] == "brainstem" and L.SIDE_OF[9] is None and L.HOST_CLASS_OF[14] is None
    assert len(L.HOST_IDS) == 13 and 9 in L.HOST_IDS and 14 not in L.HOST_IDS
    # every SynthSeg host label is covered exactly once
    covered = [l for _, _, labels, _ in L.STUDENT for l in labels]
    assert len(covered) == len(set(covered))
    assert set(covered) >= {l for labels in HOST_CLASSES.values() for l in labels}


def test_to_student_and_back_keep_the_host_class_and_drop_csf():
    seg = np.array([[0, 2, 41, 3, 42], [10, 49, 12, 58, 16], [8, 47, 18, 60, 24], [4, 43, 14, 15, 5]], np.int16)
    stu = L.to_student(seg)
    assert stu.dtype == np.uint8
    assert stu.tolist() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 0], [14, 14, 14, 14, 14]]   # CSF 24 -> background
    back = L.to_synthseg(stu)
    assert back.dtype == np.int16
    # the representative values fall in the same host class as the original labels
    assert np.array_equal(host_class_map(back), host_class_map(seg))
    assert all(v in LANDMARKS for v in back[3]) and back[2, 4] == 0
    assert CLASS_NAMES[int(host_class_map(back)[1, 2]) - 1] == "basal_ganglia"


def test_ignore_marks_lesion_voxels_and_is_refused_on_the_way_back():
    stu = np.array([[1, 2], [3, 0]], np.uint8)
    lab = L.with_ignore(stu, np.array([[0, 1], [0, 2]]))
    assert lab.tolist() == [[1, 15], [3, 15]] and stu[0, 1] == 2          # the input is not changed
    with pytest.raises(ValueError, match="student labels"):
        L.to_synthseg(lab)
    with pytest.raises(ValueError, match="outside"):
        L.to_student(np.array([300]))
