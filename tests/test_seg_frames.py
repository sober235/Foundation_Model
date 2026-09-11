import numpy as np
import pytest

from anatobind.data_engine.seg_frames import (
    FRAMES, apply_frame, frame_name, overlap_stats, parse_frame_name, rank_frames, store_in_frame,
    zooms_in_h5_frame,
)


def _scene(shape=(12, 12, 8)):
    """Two asymmetric cartilage blocks that are bright; no frame but the true one maps both onto the bright voxels."""
    seg = np.zeros(shape, np.uint8)
    seg[1:4, 6:10, 1:3] = 1
    seg[7:11, 2:5, 5:7] = 2
    mag = np.ones(shape, np.float32)
    mag[seg > 0] = 5.0
    return mag, seg


def test_sixteen_frames_with_unique_names():
    assert len(FRAMES) == 16 and len({frame_name(f) for f in FRAMES}) == 16


def test_names_round_trip():
    for f in FRAMES:
        assert parse_frame_name(frame_name(f)) == f


def test_the_dicom_track_rules_have_the_expected_names():
    assert frame_name((True, False, False, False)) == "T---"
    assert frame_name((True, False, False, True)) == "T--z"


def test_storing_then_applying_is_the_identity():
    a = np.arange(3 * 3 * 2).reshape(3, 3, 2)
    for f in FRAMES:
        assert np.array_equal(apply_frame(store_in_frame(a, f), f), a)


@pytest.mark.parametrize("frame", FRAMES, ids=frame_name)
def test_ranking_recovers_the_frame_the_array_was_stored_in(frame):
    mag, seg = _scene()
    ranked = rank_frames(mag, store_in_frame(seg, frame))
    assert ranked[0][0] == frame
    assert ranked[0][1] == pytest.approx(5.0) and ranked[1][1] < 5.0


def test_zooms_permute_only_with_the_transpose():
    assert zooms_in_h5_frame((0.3, 0.4, 0.8), (True, False, True, False)) == (0.4, 0.3, 0.8)
    assert zooms_in_h5_frame((0.3, 0.4, 0.8), (False, True, True, True)) == (0.3, 0.4, 0.8)


def test_overlap_stats_on_identical_and_shifted_maps():
    _, seg = _scene()
    same = overlap_stats(seg, seg, (0.5, 0.5, 1.0))
    assert same[1] == (1.0, 0.0) and same[2] == (1.0, 0.0)
    dice, shift = overlap_stats(seg, np.roll(seg, 1, axis=0), (0.5, 0.5, 1.0))[1]
    assert dice == pytest.approx(2 / 3) and shift == pytest.approx(0.5)
