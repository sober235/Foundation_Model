# tests/test_fastmri_knee_detection_eval.py
import pytest

from anatobind.eval.detection_metrics import operating_point, sweep
from anatobind.eval.fastmri_knee_detection import (
    SHARED_FAMILIES, THRESHOLDS, box_volume_ml, corners, fp_per_normal_scan, hit_geometry, patient_coverage,
    scan_record, size_terciles,
)

SP = (3.0, 0.5, 0.5)


def _lesion(family, z0=2, z1=4, y0=10, x0=10, y1=30, x1=30):
    return {"family": family, "z0": z0, "z1": z1, "y0": y0, "x0": x0, "y1": y1, "x1": x1}


def _det(family, score, **kw):
    return {**_lesion(family, **kw), "score": score}


def test_corners_make_z1_exclusive_like_failure_labels():
    assert corners(_lesion("meniscus")) == [2, 10, 10, 5, 30, 30]


def test_scan_record_keeps_only_the_requested_families_for_truth_and_detections():
    gt = [_lesion("bone"), _lesion("meniscus", x0=100, x1=120)]
    dets = [_det("bone", 0.9), _det("cartilage", 0.8, x0=100, x1=120)]
    s = scan_record("f", gt, dets, normal=False, families=SHARED_FAMILIES)
    assert [g["family"] for g in s["gt"]] == ["meniscus"] and [d["family"] for d in s["dets"]] == ["cartilage"]
    assert scan_record("f", gt, dets, normal=False)["gt"][0]["family"] == "bone"


def test_box_volume_in_ml_uses_the_spacing():
    assert box_volume_ml([0, 0, 0, 2, 10, 10], SP) == pytest.approx(2 * 3.0 * 10 * 0.5 * 10 * 0.5 / 1000)


def test_hit_geometry_reports_centre_error_in_mm_and_iou():
    s = scan_record("f", [_lesion("meniscus")], [_det("meniscus", 0.9, y0=12, y1=32)], normal=False)   # shifted 2 rows
    hits = hit_geometry(s, thr=0.5, spacing=SP)
    assert len(hits) == 1 and hits[0]["centre_error_mm"] == pytest.approx(2 * 0.5) and 0.6 < hits[0]["iou"] < 1.0
    assert hits[0]["pred_family"] == "meniscus"


def test_size_terciles_split_each_family_into_three_bins_and_count_hits():
    scans = [scan_record(f"s{i}", [_lesion("meniscus", x1=10 + 5 * (i + 1))], [_det("meniscus", 0.9, x1=10 + 5 * (i + 1))] if i % 2 else [], normal=False)
             for i in range(6)]
    t = size_terciles(scans, {f"s{i}": SP for i in range(6)}, thr=0.5)
    assert [b["n"] for b in t["meniscus"]] == [2, 2, 2] and sum(b["hit"] for b in t["meniscus"]) == 3


def test_patient_coverage_counts_patients_with_a_family_correct_hit():
    scans = [scan_record("a", [_lesion("meniscus")], [_det("meniscus", 0.9)], normal=False),
             scan_record("b", [_lesion("meniscus")], [_det("cartilage", 0.9)], normal=False),     # localised, wrong family
             scan_record("c", [], [], normal=True)]
    cov = patient_coverage(scans, {"a": "p1", "b": "p2", "c": "p3"}, thr=0.5)
    assert cov == {"n_patients": 2, "covered": 1, "coverage": 0.5}


def test_fp_per_normal_scan_counts_only_scans_the_radiologist_called_normal():
    scans = [scan_record("n1", [], [_det("bone", 0.9), _det("bone", 0.3)], normal=True),
             scan_record("a", [], [_det("bone", 0.9)], normal=False)]     # annotated volume whose boxes were all dropped
    assert fp_per_normal_scan(scans, thr=0.5) == {"n_normal": 1, "fp_per_normal_scan": 1.0}


def test_sweep_thresholds_are_fine_enough_for_a_poorly_calibrated_detector():
    assert THRESHOLDS[0] == 0.01 and len(THRESHOLDS) == 99
    s = scan_record("f", [_lesion("meniscus")], [_det("meniscus", 0.03)], normal=False)
    rows = sweep([s], THRESHOLDS)
    assert operating_point(rows)["sensitivity_family"] == 1.0
