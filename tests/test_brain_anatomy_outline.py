# tests/test_brain_anatomy_outline.py
import numpy as np
import pytest

import anatobind.anatomy.outline as O


def _seg():
    seg = np.zeros((40, 40, 16), np.int16)
    for k in range(13):                                   # brain in slices 0..12, widest in the middle
        r = 6 + k if k < 6 else 18 - k
        x, y = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
        seg[:, :, k][(x - 20) ** 2 + (y - 20) ** 2 <= r ** 2] = 2
    seg[20, 20, 6] = 0                                    # a hole inside the brain
    seg[2, 2, 6] = 3                                      # a speck away from the brain
    return seg


def test_fill_and_largest_component_removes_holes_and_specks():
    seg = _seg()
    out = O.fill_and_keep_largest(seg > 0)
    assert out[20, 20, 6] and not out[2, 2, 6] and out.dtype == bool
    assert O.fill_and_keep_largest(np.zeros((4, 4, 2), bool)).sum() == 0


def test_supervised_slices_leave_out_the_lowest_two_and_the_top():
    seg = _seg()
    outline = O.fill_and_keep_largest(seg > 0)
    sup = O.supervised_slices(outline, voxel_area_mm2=25.0)          # 5 x 5 mm pixels: areas 25 x count
    assert sup.start == 2 and sup.stop == 12                          # brain up to slice 12 (area > 5 cm2), margin 1 -> last supervised 11
    assert len(O.supervised_slices(np.zeros((4, 4, 3), bool), 1.0)) == 0
    tiny = np.zeros((40, 40, 16), bool)
    tiny[18:22, 18:22, :] = True                                      # 16 voxels x 25 mm2 = 400 mm2 < 5 cm2 everywhere
    assert len(O.supervised_slices(tiny, 25.0)) == 0


def test_outline_label_marks_unsupervised_slices_ignore_and_reports_volume():
    seg = _seg()
    label, info = O.outline_label(seg, voxel_area_mm2=25.0, voxel_volume_mm3=125.0)
    assert label.dtype == np.uint8 and set(np.unique(label)) == {0, 1, 2}
    assert (label[:, :, 0] == O.IGNORE).all() and (label[:, :, 1] == O.IGNORE).all() and (label[:, :, 12] == O.IGNORE).all()
    assert (label[:, :, 15] == O.IGNORE).all() and label[20, 20, 6] == 1 and label[0, 0, 6] == 0 and label[2, 2, 6] == 0
    assert info["supervised"] == [2, 11] and info["usable"] is True and info["volume_ml"] > 300
    small, info2 = O.outline_label(seg[:, :, :3], 25.0, 125.0)
    assert info2["usable"] is False and info2["supervised"] is None and (small == O.IGNORE).all()


def test_postprocess_and_dice():
    pred = np.zeros((10, 10, 3), np.float32)
    pred[2:8, 2:8, :] = 0.9
    pred[5, 5, 1] = 0.0
    pred[0, 0, 0] = 0.7
    m = O.postprocess(pred)
    assert m.dtype == np.uint8 and m[5, 5, 1] == 1 and m[0, 0, 0] == 0
    assert O.dice(m, pred > 0.5) == pytest.approx(2 * 108 / (108 + 108), abs=0.02)
    assert O.dice(np.zeros(3), np.zeros(3)) == 1.0 and O.dice([1, 0], [0, 1]) == 0.0
