import numpy as np
import pytest

from anatobind.eval.brain_detector import collect, evaluate, normal_fp_per_scan, scan_record, strata_sensitivity


def _det(box, score):
    return {"family": "small_lesion", "box": box, "score": score}


def _scans():
    gt_a = [{"lesion_id": 0, "family": "small_lesion", "box": (0, 0, 0, 4, 4, 1)},
            {"lesion_id": 1, "family": "small_lesion", "box": (10, 10, 0, 14, 14, 1)}]
    return [{"case": "a", "gt": gt_a, "dets": [_det((0, 0, 0, 4, 4, 1), 0.9), _det((20, 20, 0, 24, 24, 1), 0.3)]},
            {"case": "n", "gt": [], "dets": [_det((5, 5, 0, 8, 8, 1), 0.6)]}]


def test_scan_record_decodes_with_the_brain_minimum():
    lab = np.zeros((20, 20, 3), np.uint8)
    lab[2:6, 2:6, 1] = 1
    probs = np.zeros((2,) + lab.shape, np.float32)
    probs[1][lab == 1] = 0.7
    rec = scan_record("c", [], lab, probs)
    assert rec["case"] == "c" and len(rec["dets"]) == 1 and rec["dets"][0]["box"] == (2, 2, 1, 6, 6, 2)


def test_evaluate_gate_normal_fp_and_strata():
    scans = _scans()
    out = evaluate(scans, {"n"})
    op = out["gate"]
    # sensitivity is 0.5 at every threshold up to 0.9 (only lesion 0 is found); the operating point is the highest such threshold
    assert op["sensitivity_family"] == 0.5 and op["pass"] is True
    assert normal_fp_per_scan(scans, {"n"}, 0.5) == 1.0 and normal_fp_per_scan(scans, {"n"}, 0.7) == 0.0
    st = strata_sensitivity(scans, 0.5, {0: "small", 1: "large"})
    assert st == {"small": {"n_gt": 1, "n_hit": 1, "sensitivity": 1.0}, "large": {"n_gt": 1, "n_hit": 0, "sensitivity": 0.0}}


def test_collect_refuses_a_missing_validation_case(tmp_path):
    with pytest.raises(FileNotFoundError, match="case_x"):
        collect(tmp_path, "2d", [{"train": [], "val": ["case_x"]}], {"case_x": []})
