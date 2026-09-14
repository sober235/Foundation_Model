import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.eval.detect3d import aggregate_to_3d, failure_labels


def _det(slice_, y0, x0, y1, x1, family="meniscus", score=0.9):
    return {"slice": slice_, "family": family, "score": score,
            "y0": y0, "x0": x0, "y1": y1, "x1": x1, "embed": np.zeros(4, np.float32)}


def _les(z0, z1, y0, x0, y1, x1, family="meniscus"):
    return {"family": family, "z0": z0, "z1": z1, "y0": y0, "x0": x0, "y1": y1, "x1": x1}


def test_adjacent_overlapping_detections_become_one_lesion():
    dets = [_det(3, 10, 10, 30, 30), _det(4, 11, 10, 31, 30), _det(5, 12, 11, 32, 31)]
    out = aggregate_to_3d(dets)
    assert len(out) == 1 and (out[0]["z0"], out[0]["z1"]) == (3, 5)
    assert out[0]["score"] == pytest.approx(0.9)               # the aggregate keeps the peak score


def test_separate_detections_stay_separate():
    dets = [_det(3, 10, 10, 30, 30), _det(3, 200, 200, 220, 220)]
    assert len(aggregate_to_3d(dets)) == 2


def test_a_perfect_prediction_labels_everything_correct_and_the_scan_clean():
    gt = [_les(3, 5, 10, 10, 30, 30)]
    pred = [{"family": "meniscus", "score": 0.9, "z0": 3, "z1": 5,
             "y0": 10, "x0": 10, "y1": 30, "x1": 30, "embed": np.zeros(4, np.float32)}]
    per, scan = failure_labels(gt, pred)
    assert [p["correct"] for p in per] == [1] and scan == 0


def test_a_wrong_family_is_an_error_even_when_the_box_matches():
    gt = [_les(3, 5, 10, 10, 30, 30)]
    pred = [{"family": "cartilage", "score": 0.9, "z0": 3, "z1": 5,
             "y0": 10, "x0": 10, "y1": 30, "x1": 30, "embed": np.zeros(4, np.float32)}]
    per, scan = failure_labels(gt, pred)
    assert [p["correct"] for p in per] == [0] and scan == 1


def test_a_miss_makes_the_scan_label_one_with_no_per_lesion_row():
    gt = [_les(3, 5, 10, 10, 30, 30)]
    per, scan = failure_labels(gt, [])
    assert per == [] and scan == 1


def test_a_detection_matching_nothing_makes_the_scan_label_one():
    pred = [{"family": "meniscus", "score": 0.9, "z0": 3, "z1": 5,
             "y0": 10, "x0": 10, "y1": 30, "x1": 30, "embed": np.zeros(4, np.float32)}]
    per, scan = failure_labels([], pred)
    assert [p["correct"] for p in per] == [0] and scan == 1


def test_a_false_positive_alongside_a_correct_detection_still_fails_the_scan():
    gt = [_les(3, 5, 10, 10, 30, 30)]
    pred = [
        {"family": "meniscus", "score": 0.9, "z0": 3, "z1": 5,
         "y0": 10, "x0": 10, "y1": 30, "x1": 30, "embed": np.zeros(4, np.float32)},
        {"family": "cartilage", "score": 0.8, "z0": 20, "z1": 21,
         "y0": 200, "x0": 200, "y1": 220, "x1": 220, "embed": np.zeros(4, np.float32)},
    ]
    per, scan = failure_labels(gt, pred)
    assert sorted(p["correct"] for p in per) == [0, 1]
    assert scan == 1                       # the lesion was found, but the spurious detection is an error
