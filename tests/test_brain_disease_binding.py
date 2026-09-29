# tests/test_brain_disease_binding.py
import numpy as np
import pytest

from anatobind.bind.brain_lookup import BrainBinder

# SynthSeg labels: 2 / 41 left / right white matter, 3 / 42 left / right cortex, 11 left caudate, 16 brainstem,
# 4 left lateral ventricle (a landmark, never a host), 0 background


def _seg():
    seg = np.zeros((20, 10, 4), np.int16)
    seg[0:10, :, :] = 2            # left white matter
    seg[10:20, :, :] = 41          # right white matter
    seg[0:4, 0:4, :] = 3           # left cortex
    seg[8:10, 4:6, :] = 4          # left lateral ventricle
    seg[9:11, 8:10, :] = 16        # brainstem across the midline
    return seg


def _box(x0, x1, y0, y1, z0, z1):
    sl = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
    return sl, np.ones((x1 - x0, y1 - y0, z1 - z0), bool)


def test_main_structure_fractions_and_side_by_overlap():
    b = BrainBinder(_seg(), (1.0, 1.0, 1.0))
    out = b.bind(*_box(2, 6, 0, 4, 0, 1))          # 8 cortex voxels + 8 white matter voxels, all on the left
    assert out["host_rule"] == "overlap" and out["side"] == "left"
    assert out["host_fractions"] == {"white_matter": 0.5, "cortex": 0.5}
    assert out["host"] == "white_matter"           # a tie goes to the lower class id
    out = b.bind(*_box(12, 16, 0, 4, 0, 2))
    assert out == {"host": "white_matter", "host_rule": "overlap", "host_fractions": {"white_matter": 1.0}, "side": "right"}


def test_both_sides_above_forty_percent_is_bilateral():
    b = BrainBinder(_seg(), (1.0, 1.0, 1.0))
    assert b.bind(*_box(8, 12, 0, 2, 0, 1))["side"] == "bilateral"        # 4 left + 4 right voxels
    assert b.bind(*_box(5, 12, 0, 1, 0, 1))["side"] == "left"             # 5 left + 2 right: 2 / 7 < 0.4


def test_a_structure_without_a_side_is_midline():
    out = BrainBinder(_seg(), (1.0, 1.0, 1.0)).bind(*_box(9, 11, 8, 10, 0, 1))
    assert out["host"] == "brainstem" and out["side"] == "midline"


def test_a_lesion_on_a_landmark_takes_the_nearest_host():
    out = BrainBinder(_seg(), (1.0, 1.0, 1.0)).bind(*_box(8, 10, 4, 6, 1, 2))   # inside the left lateral ventricle
    assert out == {"host": "white_matter", "host_rule": "nearest", "host_fractions": {}, "side": "left"}


def test_nearest_uses_millimetres_not_voxels():
    seg = np.zeros((9, 1, 9), np.int16)
    seg[0, 0, 4] = 3               # left cortex, 4 voxels away along x
    seg[4, 0, 8] = 41              # right white matter, 4 voxels away along z
    sl, mask = (slice(4, 5), slice(0, 1), slice(4, 5)), np.ones((1, 1, 1), bool)
    assert BrainBinder(seg, (1.0, 1.0, 5.0)).bind(sl, mask)["host"] == "cortex"          # 4 mm against 20 mm
    assert BrainBinder(seg, (5.0, 1.0, 1.0)).bind(sl, mask)["host"] == "white_matter"    # 20 mm against 4 mm


def test_no_host_anywhere_and_a_mask_off_the_grid():
    seg = np.zeros((4, 4, 4), np.int16)
    seg[0, 0, 0] = 4
    b = BrainBinder(seg, (1.0, 1.0, 1.0))
    assert b.bind(*_box(1, 2, 1, 2, 1, 2)) == {"host": None, "host_rule": None, "host_fractions": {}, "side": "midline"}
    with pytest.raises(ValueError, match="does not fit"):
        b.bind((slice(2, 6), slice(0, 1), slice(0, 1)), np.ones((4, 1, 1), bool))
