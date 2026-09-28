# tests/test_nndet_eval.py
import nibabel as nib
import numpy as np
import pytest

from anatobind.eval.brain_detector import strata_maps
from anatobind.eval.brain_nndet import (
    gt_check, gt_of_cases, nndet_scans, nnunet_scans, paired_table, rule_a,
)
from anatobind.nnunet.brain_lesion import DATASET_NAME, TRAINER


def _reg(lid, file, x0, y0, z0, x1, y1, z1, **kw):
    return {"lesion_id": lid, "file": file, "x0": x0, "y0": y0, "z0": z0, "x1": x1, "y1": y1, "z1": z1, **kw}


def test_strata_maps_follow_the_s2_definitions():
    reg = [{"lesion_id": i, "band": b, "n_slices": n, "inplane_mm": mm, "stratum_geometry": g}
           for i, (b, n, mm, g) in enumerate([("0", 1, 1.0, "a"), ("0-2", 2, 2.0, "a"), ("2-4", 1, 3.0, "b"),
                                              (">4", 3, 4.0, "b"), ("0", 1, 5.0, "c"), ("0", 1, 6.0, "c")])]
    m = strata_maps(reg)
    assert m["inplane_tertile"] == {0: "tertile_1", 1: "tertile_1", 2: "tertile_1", 3: "tertile_2", 4: "tertile_2",
                                    5: "tertile_3"}
    assert m["n_slices"] == {0: "1", 1: ">1", 2: "1", 3: ">1", 4: "1", 5: "1"}
    assert m["band"][3] == ">4" and m["stratum_geometry"][5] == "c"


def test_gt_of_cases_uses_registry_boxes_and_refuses_a_lesion_case_without_rows():
    reg = [_reg(0, "A", 5, 5, 10, 10, 10, 10)]
    g = gt_of_cases({"A": "lesion", "N": "normal"}, reg)
    assert g == {"A": [{"lesion_id": 0, "family": "small_lesion", "box": (5, 5, 10, 10, 10, 11)}], "N": []}
    with pytest.raises(ValueError, match="B"):
        gt_of_cases({"A": "lesion", "B": "lesion"}, reg)


def test_nndet_scans_refuse_a_missing_case():
    with pytest.raises(KeyError, match="N"):
        nndet_scans({"A": {"boxes": [], "scores": []}}, ["A", "N"], {"A": [], "N": []})


def _rows(*triples):
    return [{"thr": t, "n_hit": h, "sensitivity_family": h / 280, "fp_per_scan": fp} for t, h, fp in triples]


def test_rule_a_needs_ceil_margin_times_lesions_more_hits_than_the_baseline():
    base = _rows((0.3, 99, 2.5), (0.6, 92, 1.5))                 # operating point: 92 hits at <= 2 FP
    assert rule_a(_rows((0.5, 105, 1.0)), base, 280)["pass"] is False
    r = rule_a(_rows((0.2, 130, 3.0), (0.5, 106, 1.9)), base, 280)
    assert r["pass"] is True and r["required"] == 106 and r["nndet_hits"] == 106 and r["baseline_thr"] == 0.6
    assert rule_a(_rows((0.5, 150, 2.5)), base, 280)["pass"] is False   # no nnDetection threshold at <= 2 FP


def _scan(case, lids, found, thr_score=0.9):
    gt = [{"lesion_id": l, "family": "small_lesion", "box": (10 * l, 0, 0, 10 * l + 5, 5, 1)} for l in lids]
    dets = [{"box": (10 * l, 0, 0, 10 * l + 5, 5, 1), "score": thr_score, "family": "small_lesion"} for l in found]
    return {"case": case, "gt": gt, "dets": dets}


def test_paired_table_counts_the_four_groups():
    a = [_scan("A", [0, 1, 2, 3], [0, 1])]
    b = [_scan("A", [0, 1, 2, 3], [1, 2])]
    assert paired_table(a, 0.5, b, 0.5) == {"both": 1, "only_a": 1, "only_b": 1, "neither": 1, "n_gt": 4}
    with pytest.raises(ValueError):
        paired_table(a, 0.5, [_scan("A", [0, 1], [])], 0.5)


def test_gt_check_reports_iou_lost_and_below_threshold_instances():
    gt_of_case = {"A": [{"lesion_id": 10, "family": "small_lesion", "box": (5, 6, 1, 9, 10, 2)},
                        {"lesion_id": 11, "family": "small_lesion", "box": (20, 20, 0, 24, 24, 1)},
                        {"lesion_id": 12, "family": "small_lesion", "box": (40, 40, 0, 44, 44, 1)}],
                  "N": []}
    cases = {"A": {"boxes": [[1, 6, 2, 10, 5, 9], [0, 50, 1, 54, 50, 54]], "scores": [1.0, 1.0], "instances": [1, 2]},
             "N": {"boxes": [], "scores": [], "instances": []}}
    res = gt_check(cases, {"A": {"1": 10, "2": 11, "3": 12}, "N": {}}, gt_of_case)
    assert res["iou"][10] == pytest.approx(1.0) and res["iou"][11] == 0.0
    assert res["lost"] == [12] and res["below_iou"] == [11]
    with pytest.raises(ValueError, match="instance map"):
        gt_check({"A": {"boxes": [[1, 6, 2, 10, 5, 9]], "scores": [1.0], "instances": [4]}},
                 {"A": {"1": 10, "2": 11, "3": 12}}, {"A": gt_of_case["A"]})
    with pytest.raises(ValueError, match="registry"):
        gt_check(cases, {"A": {"1": 10, "2": 11}, "N": {}}, gt_of_case)


def test_nnunet_scans_read_only_the_requested_folds(tmp_path):
    splits = [{"train": ["B"], "val": ["A"]}, {"train": ["A"], "val": ["B"]}]
    shape = (32, 32, 4)
    d = tmp_path / DATASET_NAME / f"{TRAINER}__nnUNetPlans__2d" / "fold_1" / "validation"
    d.mkdir(parents=True)
    lab = np.zeros(shape, np.uint8)
    lab[5:10, 5:10, 2] = 1
    probs = np.zeros((2,) + shape, np.float32)
    probs[1][lab == 1] = 0.8
    nib.save(nib.Nifti1Image(lab, np.eye(4)), str(d / "B.nii.gz"))
    np.savez_compressed(d / "B.npz", probabilities=np.ascontiguousarray(probs.transpose(0, 3, 2, 1)))
    gt = {"A": [], "B": [{"lesion_id": 3, "family": "small_lesion", "box": (5, 5, 2, 10, 10, 3)}]}
    scans = nnunet_scans(tmp_path, "2d", splits, [1], gt)
    assert [s["case"] for s in scans] == ["B"] and len(scans[0]["dets"]) == 1
    with pytest.raises(FileNotFoundError, match="fold 0 case A"):
        nnunet_scans(tmp_path, "2d", splits, [0], gt)
