import numpy as np
import pytest

from anatobind.eval.lookup import NONE, UNKNOWN, LabelIndex, b0_host
from anatobind.eval.matching import CORRECT, MISS, WRONG_CLASS, WRONG_HOST, bucket, detected, iou3d, match

SP = (0.8, 0.625, 0.625)


def _map():
    m = np.zeros((20, 20, 20), np.uint8)
    m[0:5] = 2                      # femoral cartilage slab, z 0-4
    m[10:14, 2:8, 2:8] = 5          # medial meniscus
    m[10:14, 12:18, 12:18] = 6      # lateral meniscus
    m[16:20, 2:8, 2:8] = 1          # patellar cartilage
    return m


def test_an_effusion_has_no_host_and_a_ligament_tear_an_unknown_one():
    idx = LabelIndex(_map(), SP)
    assert b0_host(idx, (0, 0, 0, 5, 5, 5), 2) == NONE
    assert b0_host(idx, (0, 0, 0, 5, 5, 5), 3) == UNKNOWN


def test_a_cartilage_lesion_takes_the_cartilage_it_overlaps_most():
    # femoral voxels in two slices, patellar in one, medial meniscus is not a cartilage candidate
    assert b0_host(LabelIndex(_map(), SP), (3, 2, 2, 17, 8, 8), 1) == 2


def test_the_class_restricts_the_candidates_before_overlap_is_counted():
    # a meniscal-tear box on femoral cartilage only: femoral is no candidate, the nearest meniscus wins
    assert b0_host(LabelIndex(_map(), SP), (0, 2, 2, 4, 8, 8), 0) == 5


def test_zero_overlap_falls_back_to_the_nearest_candidate_in_millimetres():
    assert b0_host(LabelIndex(_map(), SP), (10, 10, 10, 14, 11, 11), 0) == 6


def test_no_candidate_in_the_map_returns_none():
    m = np.zeros((10, 10, 10), np.uint8)
    m[:3] = 2
    assert b0_host(LabelIndex(m, SP), (0, 0, 0, 3, 3, 3), 0) is None


def test_iou_of_identical_disjoint_and_half_overlapping_boxes():
    a = np.array([[0, 0, 0, 2, 2, 2]])
    b = np.array([[0, 0, 0, 2, 2, 2], [5, 5, 5, 6, 6, 6], [1, 0, 0, 3, 2, 2]])
    assert np.allclose(iou3d(a, b), [[1.0, 0.0, 1 / 3]])


def test_a_query_is_a_detection_unless_no_object_is_most_probable():
    p = np.array([[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.6]])
    assert detected(p).tolist() == [0]


def test_matching_is_one_to_one_and_drops_pairs_below_the_threshold():
    gt = np.array([[0, 0, 0, 2, 2, 2], [10, 10, 10, 12, 12, 12]])
    pred = np.array([[0, 0, 0, 2, 2, 2], [0, 0, 0, 2, 2, 1], [30, 30, 30, 31, 31, 31]])
    assert match(gt, pred, 0.1) == {0: 0}
    assert match(gt, np.zeros((0, 6)), 0.1) == {} and match(np.zeros((0, 6)), pred, 0.1) == {}


def test_buckets_at_the_tissue_family_level():
    assert bucket(0, 1, None, None) == MISS
    assert bucket(0, 1, 1, 2) == WRONG_CLASS
    assert bucket(1, 4, 1, None) == WRONG_HOST          # no candidate structure in the predicted map
    assert bucket(1, 4, 1, 1) == WRONG_HOST             # patellar label for a femoral-cartilage lesion
    assert bucket(1, 4, 1, 2) == CORRECT
    assert bucket(0, 1, 0, 6) == CORRECT                # the lateral meniscus is still the meniscus family

from anatobind.eval.lookup import HOST_NAMES, SIDE_OF_LABEL, describe_host, host_fractions  # noqa: E402


def test_host_fractions_cover_only_the_class_candidates_present_in_the_map():
    idx = LabelIndex(_map(), SP)
    f = host_fractions(idx, (3, 2, 2, 17, 8, 8), 1)      # cartilage: femoral z 3-4 (2 of 14 slices) and patellar z 16
    assert set(f) == {1, 2} and f[2] == pytest.approx(2 * 36 / (14 * 36)) and f[1] == pytest.approx(36 / (14 * 36))
    assert host_fractions(idx, (10, 10, 10, 14, 11, 11), 0) == {5: 0.0, 6: 0.0}
    assert host_fractions(idx, (0, 0, 0, 5, 5, 5), 2) == {} and host_fractions(idx, (0, 0, 0, 5, 5, 5), 3) == {}


def test_describe_host_agrees_with_b0_host_and_names_the_side():
    idx = LabelIndex(_map(), SP)
    d = describe_host(idx, (10, 2, 2, 14, 8, 8), 0)
    assert d["host_label"] == b0_host(idx, (10, 2, 2, 14, 8, 8), 0) == 5
    assert d["host_name"] == "meniscus_medial" and d["side"] == "medial" and d["host_fractions"][5] == 1.0
    assert describe_host(idx, (0, 0, 0, 5, 5, 5), 2) == {"host_label": None, "host_name": "none", "side": "-", "host_fractions": {}}
    assert describe_host(idx, (0, 0, 0, 5, 5, 5), 3)["host_name"] == "unknown"
    empty = np.zeros((10, 10, 10), np.uint8)
    assert describe_host(LabelIndex(empty, SP), (0, 0, 0, 3, 3, 3), 0) == {"host_label": None, "host_name": "", "side": "-", "host_fractions": {}}
    assert SIDE_OF_LABEL[3] == "medial" and SIDE_OF_LABEL[6] == "lateral" and HOST_NAMES[2] == "femoral_cartilage"
