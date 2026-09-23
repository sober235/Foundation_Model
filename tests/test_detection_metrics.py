import pytest

from anatobind.eval.detection_metrics import (
    FP_MAX, GATE_SENSITIVITY, gate, match_scan, operating_point, per_family, sweep,
)


def _det(box, family, score):
    return {"family": family, "box": box, "score": score, "label": 1, "n_voxels": 100}


def _gt(box, family):
    return {"box": box, "family": family}


A, B = (0, 0, 0, 10, 10, 10), (20, 20, 20, 30, 30, 30)
SCANS = [
    {"scan": "s1", "gt": [_gt(A, "Cartilage Lesion"), _gt(B, "Effusion")],
     "dets": [_det(A, "Cartilage Lesion", 0.9), _det(B, "Meniscal Tear", 0.6), _det((40, 40, 40, 45, 45, 45), "Effusion", 0.3)]},
    {"scan": "s2", "gt": [_gt(A, "Meniscal Tear")], "dets": [_det((0, 0, 0, 10, 10, 3), "Meniscal Tear", 0.5)]},
    {"scan": "s3", "gt": [], "dets": [_det(B, "Effusion", 0.2)]},
]


def test_matching_is_one_to_one_at_iou_0_1_regardless_of_family():
    assert match_scan(SCANS[0]["gt"], SCANS[0]["dets"]) == {0: 0, 1: 1}
    assert match_scan(SCANS[1]["gt"], SCANS[1]["dets"]) == {0: 0}      # IoU 0.3 >= 0.1
    assert match_scan([], SCANS[2]["dets"]) == {}


def test_sweep_counts_hits_family_hits_and_false_positives_per_threshold():
    rows = sweep(SCANS, thresholds=[0.1, 0.55, 0.95])
    r = rows[0]
    assert (r["n_gt"], r["n_hit"], r["n_hit_family"], r["n_fp"], r["n_scans"]) == (3, 3, 2, 2, 3)
    assert r["sensitivity"] == 1.0 and r["sensitivity_family"] == pytest.approx(2 / 3) and r["fp_per_scan"] == pytest.approx(2 / 3)
    r = rows[1]   # only scores >= 0.55 survive: A-hit (0.9, right family), B-hit (0.6, wrong family)
    assert (r["n_hit"], r["n_hit_family"], r["n_fp"]) == (2, 1, 0)
    assert rows[2]["n_hit"] == 0 and rows[2]["sensitivity"] == 0.0


def test_operating_point_is_the_best_family_sensitivity_under_the_fp_budget():
    rows = [{"thr": 0.1, "sensitivity_family": 0.9, "fp_per_scan": 3.0},
            {"thr": 0.3, "sensitivity_family": 0.7, "fp_per_scan": 1.5},
            {"thr": 0.5, "sensitivity_family": 0.7, "fp_per_scan": 0.5},
            {"thr": 0.9, "sensitivity_family": 0.2, "fp_per_scan": 0.0}]
    assert operating_point(rows)["thr"] == 0.5
    assert operating_point(rows, fp_max=0.1)["thr"] == 0.9
    assert operating_point([{"thr": 0.1, "sensitivity_family": 1.0, "fp_per_scan": 9.0}]) is None


def test_per_family_reports_each_family_separately():
    f = per_family(SCANS, 0.1)
    assert f["Cartilage Lesion"] == {"n_gt": 1, "n_hit": 1, "n_hit_family": 1, "sensitivity": 1.0, "sensitivity_family": 1.0}
    assert f["Effusion"]["n_hit"] == 1 and f["Effusion"]["n_hit_family"] == 0
    assert f["Meniscal Tear"]["sensitivity_family"] == 1.0 and "Ligament Tear" not in f


def test_gate_reads_the_operating_point():
    assert FP_MAX == 2.0 and GATE_SENSITIVITY == 0.5
    g = gate(sweep(SCANS, thresholds=[0.1, 0.55]))
    assert g == {"pass": True, "thr": 0.1, "sensitivity_family": pytest.approx(2 / 3), "fp_per_scan": pytest.approx(2 / 3)}
    assert gate([{"thr": 0.1, "sensitivity_family": 0.4, "fp_per_scan": 0.0}])["pass"] is False
    assert gate([])["pass"] is False and gate([])["thr"] is None
