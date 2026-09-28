import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nndet")
from nndet.io.transforms.instances import instances_to_boxes_np  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

RUNNER = Path(__file__).resolve().parents[1] / "scripts/nndet_runner.py"
spec = importlib.util.spec_from_file_location("nndet_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _original():
    inst = np.zeros((6, 40, 30), np.int32)          # original array axes (slice, row, col)
    inst[2, 10:12, 5:7] = 1                          # single-slice 2 x 2 lesion
    inst[1:4, 20:25, 12:20] = 2                      # three-slice lesion
    return inst


def _tight(inst, k):
    s, r, c = np.nonzero(inst == k)
    return np.array([s.min(), r.min(), s.max() + 1, r.max() + 1, c.min(), c.max() + 1], float)


def test_roundtrip_with_crop_and_no_resampling_is_exact():
    inst = _original()
    crop = [(1, 5), (5, 35), (2, 28)]
    boxes, ids = instances_to_boxes_np(inst[1:5, 5:35, 2:28])
    props = {"original_spacing": np.array([5.0, 0.6875, 0.6875]),
             "spacing_after_resampling": np.array([5.0, 0.6875, 0.6875]), "crop_bbox": crop}
    got = runner.to_original(boxes, props, [0, 1, 2])
    for row, k in zip(got, ids):
        assert np.allclose(row, _tight(inst, k))


def test_roundtrip_with_transpose_and_resampling_is_within_one_voxel():
    inst = _original()
    tf = [2, 0, 1]
    tb = [int(i) for i in np.argsort(tf)]
    orig_sp = np.array([5.0, 0.86, 0.86])            # original axis order
    target_sp = np.array([5.0, 0.6875, 0.6875])
    res = zoom(inst.transpose(tf), orig_sp[tf] / target_sp[tf], order=0)
    boxes, ids = instances_to_boxes_np(res)
    props = {"original_spacing": orig_sp, "spacing_after_resampling": target_sp[tf],
             "crop_bbox": [(0, 6), (0, 40), (0, 30)]}
    got = runner.to_original(boxes, props, tb)
    assert sorted(int(k) for k in ids) == [1, 2]
    for row, k in zip(got, ids):
        assert np.all(np.abs(row - _tight(inst, k)) <= 1.0 + 1e-9)
