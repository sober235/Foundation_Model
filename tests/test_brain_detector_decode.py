import numpy as np

from anatobind.eval.lesion_boxes import BRAIN_MIN_VOXELS, MIN_VOXELS, decode_boxes


def _single_slice_lesion():
    lab = np.zeros((20, 20, 4), np.uint8)
    lab[5:9, 6:10, 2] = 1                      # 4 x 4 x 1 = 16 voxels
    probs = np.zeros((2,) + lab.shape, np.float32)
    probs[1][lab == 1] = 0.8
    return lab, probs


def test_brain_minimum_keeps_a_single_slice_4x4_lesion_and_the_knee_default_is_unchanged():
    lab, probs = _single_slice_lesion()
    assert MIN_VOXELS == 27 and BRAIN_MIN_VOXELS == 9
    assert decode_boxes(lab, probs, families={1: "small_lesion"}) == []           # 16 < 27
    out = decode_boxes(lab, probs, min_voxels=BRAIN_MIN_VOXELS, families={1: "small_lesion"})
    assert len(out) == 1 and out[0]["family"] == "small_lesion" and out[0]["n_voxels"] == 16
    assert out[0]["box"] == (5, 6, 2, 9, 10, 3) and abs(out[0]["score"] - 0.8) < 1e-6


def test_default_families_are_the_knee_ones():
    lab = np.zeros((10, 10, 10), np.uint8)
    lab[1:4, 1:4, 1:4] = 2
    assert decode_boxes(lab)[0]["family"] == "Meniscal Tear"
