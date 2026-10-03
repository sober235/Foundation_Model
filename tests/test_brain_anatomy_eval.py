# tests/test_brain_anatomy_eval.py
import numpy as np
import pytest

import anatobind.eval.brain_anatomy as E
from anatobind.anatomy.labels import to_student


def _maps():
    seg = np.zeros((30, 30, 8), np.int16)
    seg[2:28, 2:28, 1:7] = 2                 # left white matter block (RAS-agnostic here: just labels)
    seg[15:28, 2:28, 1:7] = 41               # right white matter on the far half
    seg[10:20, 10:20, 3:5] = 10              # left thalamus
    return seg


def test_class_dice_and_summary_skip_classes_absent_on_both_sides():
    seg = _maps()
    stu = to_student(seg)
    pred = stu.copy()
    pred[10:20, 10:15, 3:5] = 1              # half the thalamus called white matter
    d = E.class_dice(pred, stu, slices=range(1, 7))
    assert d[5] == pytest.approx(2 * 100 / (100 + 200)) and d[2] == 1.0 and d[6] is None and d[9] is None
    s = E.summarize_dice([d, E.class_dice(stu, stu)])
    assert s["per_class"]["thalamus_left"] == pytest.approx((d[5] + 1.0) / 2) and s["per_class"]["thalamus_right"] is None
    assert s["n_cases"] == 2 and 0.9 < s["mean_host_dice"] <= 1.0


def test_reliable_slices_follow_the_outline_rule():
    seg = _maps()
    r = E.reliable_slices(seg, voxel_area_mm2=4.0)      # 26 x 26 x 4 mm2 = 2704 mm2 > 5 cm2 in slices 1..6 -> top 6, margin 1
    assert list(r) == [2, 3, 4, 5]
    assert list(E.reliable_slices(np.zeros((5, 5, 3)), 4.0)) == []


def test_host_agreement_evaluates_boxes_inside_the_reliable_slices_only():
    seg = _maps()
    stu = to_student(seg)
    wrong = stu.copy()
    wrong[10:20, 10:20, 3:5] = 1             # the student misses the thalamus entirely
    rows = [{"lesion_id": 1, "x0": 11, "x1": 14, "y0": 11, "y1": 14, "z0": 3, "z1": 4},      # inside the thalamus, reliable slices
            {"lesion_id": 2, "x0": 3, "x1": 6, "y0": 3, "y1": 6, "z0": 2, "z1": 2},         # left white matter
            {"lesion_id": 3, "x0": 3, "x1": 6, "y0": 3, "y1": 6, "z0": 0, "z1": 1}]         # touches slice 0: not reliable
    rel = E.reliable_slices(seg, 4.0)
    out = E.host_agreement(wrong, seg, (2.0, 2.0, 5.0), rows, rel)
    assert out["n_lesions"] == 3 and out["n_evaluated"] == 2 and out["n_outside_reliable"] == 1
    assert out["n_agree"] == 1 and out["rate"] == 0.5
    assert out["pairs"][0] == {"lesion_id": 1, "student": "white_matter", "synthseg": "thalamus"}
    same = E.host_agreement(stu, seg, (2.0, 2.0, 5.0), rows, rel)
    assert same["rate"] == 1.0
    pooled = E.pool_agreement([out, same])
    assert pooled == {"n_lesions": 6, "n_evaluated": 4, "n_agree": 3, "n_outside_reliable": 2, "rate": 0.75}


def test_outline_dice_and_verdict():
    seg = _maps()
    mask = (seg > 0).astype(np.uint8)
    assert E.outline_dice(mask, seg, 4.0) == 1.0
    mask[2:10] = 0
    assert 0.8 < E.outline_dice(mask, seg, 4.0) < 1.0
    assert E.outline_dice(mask, np.zeros_like(seg), 4.0) is None
    v = E.verdict(0.95, 0.85, 0.98)
    assert v["pass"] and v["gates"] == {"host_agreement": 0.9, "mean_host_dice": 0.8, "outline_dice": 0.97}
    assert not E.verdict(0.89, 0.85, 0.98)["pass"] and not E.verdict(0.95, None, 0.98)["pass"]
    assert E.verdict(0.90, 0.80, 0.97)["pass"]                                                 # the gate values pass
