# tests/test_brain_disease_crossrun.py
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from anatobind.eval.lesion_components import component_rows, components
from anatobind.infer.brain_disease import detections


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/brain_disease_crossrun.py"
    spec = importlib.util.spec_from_file_location("brain_disease_crossrun", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_only_the_three_pairs_with_shared_sequences_exist():
    assert set(_load().CROSS) == {("infarct", "glioma"), ("metastasis", "glioma"), ("glioma", "metastasis")}


def _volumes():
    gt = np.zeros((20, 20, 6), np.uint8)
    gt[2:6, 2:6, 1:4] = 1                 # 48 voxels: a counted lesion, touched by a detection
    gt[12:16, 12:16, 1:4] = 1             # 48 voxels: a counted lesion, untouched
    gt[18, 18, 5] = 1                     # 1 voxel: ignored
    pred = np.zeros((20, 20, 6), np.uint8)
    probs = np.zeros((2, 20, 20, 6), np.float32)
    for sl, p in (((slice(4, 8), slice(4, 8), slice(1, 4)), 0.875),      # overlaps the first lesion
                  ((slice(8, 11), slice(0, 3), slice(3, 6)), 0.75),      # on no ground truth
                  ((slice(16, 19), slice(2, 5), slice(0, 3)), 0.625)):   # on no ground truth, under the threshold
        pred[sl] = 1
        probs[1][sl] = p
    return gt, pred, probs


def test_overlap_counts_by_hand():
    c = _load()
    gt, pred, probs = _volumes()
    dets, det_comp = detections(pred, probs, 1.0, "infarct")
    gt_comp, n = components(gt)
    rows = component_rows(gt_comp, n, 1.0, "tumor")
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.7) == {"n_det": 2, "n_det_on_gt": 1, "n_gt": 2, "n_gt_claimed": 1}
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.5) == {"n_det": 3, "n_det_on_gt": 1, "n_gt": 2, "n_gt_claimed": 1}
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.9) == {"n_det": 0, "n_det_on_gt": 0, "n_gt": 2, "n_gt_claimed": 0}


def test_summary_rates_and_empty_inputs():
    c = _load()
    s = c.summarise({"a": {"n_det": 2, "n_det_on_gt": 1, "n_gt": 2, "n_gt_claimed": 1},
                     "b": {"n_det": 0, "n_det_on_gt": 0, "n_gt": 1, "n_gt_claimed": 0}})
    assert s == {"n_scans": 2, "n_det": 2, "n_det_on_gt": 1, "n_gt": 3, "n_gt_claimed": 1, "det_per_scan": 1.0,
                 "scans_with_any_detection": 1, "share_of_detections_on_gt": 0.5, "share_of_gt_claimed": pytest.approx(1 / 3)}
    assert c.summarise({"a": {"n_det": 0, "n_det_on_gt": 0, "n_gt": 0, "n_gt_claimed": 0}})["share_of_detections_on_gt"] is None


def test_an_impossible_pair_and_an_existing_output_are_refused(tmp_path):
    c = _load()
    with pytest.raises(SystemExit, match="no cross run"):
        c.main(["--model", "infarct", "--data", "metastasis", "--threshold", "0.5", "--gpu", "0",
                "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    (tmp_path / "o").mkdir()
    with pytest.raises(FileExistsError):
        c.main(["--model", "infarct", "--data", "glioma", "--threshold", "0.5", "--gpu", "0",
                "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    assert not (tmp_path / "w").exists()
