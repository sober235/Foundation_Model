# tests/test_brain_disease_crossrun.py
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from anatobind.eval.lesion_components import component_rows, components
from anatobind.infer.brain_disease import detections
from anatobind.nnunet.brain_disease import DISEASES, fold_dir


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
    # a score equal to the threshold is kept
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.75) == {"n_det": 2, "n_det_on_gt": 1, "n_gt": 2, "n_gt_claimed": 1}
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.875)["n_det"] == 1


def test_summary_rates_and_empty_inputs():
    c = _load()
    s = c.summarise({"a": {"n_det": 2, "n_det_on_gt": 1, "n_gt": 2, "n_gt_claimed": 1},
                     "b": {"n_det": 0, "n_det_on_gt": 0, "n_gt": 1, "n_gt_claimed": 0}})
    assert s == {"n_scans": 2, "n_det": 2, "n_det_on_gt": 1, "n_gt": 3, "n_gt_claimed": 1, "det_per_scan": 1.0,
                 "scans_with_any_detection": 1, "share_of_detections_on_gt": 0.5, "share_of_gt_claimed": pytest.approx(1 / 3)}
    empty = c.summarise({"a": {"n_det": 0, "n_det_on_gt": 0, "n_gt": 0, "n_gt_claimed": 0}})
    assert empty["share_of_detections_on_gt"] is None and empty["share_of_gt_claimed"] is None


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


def test_the_channels_of_each_pair_follow_the_model_by_meaning():
    c = _load()
    assert DISEASES["infarct"]["channels"] == ("DWI", "ADC") and c.CROSS[("infarct", "glioma")] == ("DWI", "ADC")
    assert DISEASES["metastasis"]["channels"] == ("T1pre", "T1post", "FLAIR")
    assert c.CROSS[("metastasis", "glioma")] == ("T1", "T1c", "FLAIR")                  # glioma's names for the same three
    assert DISEASES["glioma"]["channels"] == ("T1", "T1c", "T2", "FLAIR")
    assert c.CROSS[("glioma", "metastasis")] == ("T1pre", "T1post", "T2Synth", "FLAIR")   # the T2 is synthetic
    assert set(c.NOTES) == {("glioma", "metastasis")}


def test_overlap_counts_each_object_once_and_a_fragment_is_lesion_tissue():
    c = _load()
    gt = np.zeros((24, 20, 6), np.uint8)
    gt[2:6, 2:6, 1:4] = 1                 # lesion A, 48 voxels
    gt[2:6, 10:14, 1:4] = 1               # lesion B, 48 voxels
    gt[20, 16, 4] = 1                     # a fragment of 1 voxel: ignored
    pred = np.zeros((24, 20, 6), np.uint8)
    probs = np.zeros((2, 24, 20, 6), np.float32)
    for sl, p in (((slice(2, 4), slice(2, 4), slice(1, 4)), 0.875),       # on A
                  ((slice(5, 8), slice(4, 8), slice(1, 4)), 0.875),       # on A again (touches its corner)
                  ((slice(19, 22), slice(15, 18), slice(3, 6)), 0.75)):   # on the fragment only
        pred[sl] = 1
        probs[1][sl] = p
    dets, det_comp = detections(pred, probs, 1.0, "infarct")
    gt_comp, n = components(gt)
    rows = component_rows(gt_comp, n, 1.0, "tumor")
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.5) == {"n_det": 3, "n_det_on_gt": 3, "n_gt": 2, "n_gt_claimed": 1}
    wide = np.zeros((24, 20, 6), np.uint8)
    wide[3:5, 3:12, 1:4] = 1              # one detection across A and B
    wp = np.zeros((2, 24, 20, 6), np.float32)
    wp[1][wide == 1] = 0.875
    dets, det_comp = detections(wide, wp, 1.0, "infarct")
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.5) == {"n_det": 1, "n_det_on_gt": 1, "n_gt": 2, "n_gt_claimed": 2}


def test_a_later_detection_whose_box_covers_an_earlier_one_does_not_erase_its_claim():
    c = _load()
    gt = np.zeros((12, 12, 6), np.uint8)
    gt[2:6, 2:6, 1:4] = 1                 # lesion A, 48 voxels
    pred = np.zeros((12, 12, 6), np.uint8)
    probs = np.zeros((2, 12, 12, 6), np.float32)
    pred[2:4, 2:4, 1:4] = 1               # detection on A
    probs[1][2:4, 2:4, 1:4] = 0.875
    pred[7, 2:8, 1:4] = 1                 # an L-shaped detection off A whose box [2:8, 2:8, 1:4] contains A's box;
    pred[2:8, 7, 1:4] = 1                 # it comes after the first one in label order and in score order
    probs[1][7, 2:8, 1:4] = 0.75
    probs[1][2:8, 7, 1:4] = 0.75
    dets, det_comp = detections(pred, probs, 1.0, "infarct")
    assert len(dets) == 2
    gt_comp, n = components(gt)
    rows = component_rows(gt_comp, n, 1.0, "tumor")
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.5) == {"n_det": 2, "n_det_on_gt": 1, "n_gt": 1, "n_gt_claimed": 1}


def test_nothing_is_written_when_a_model_fold_or_a_channel_is_missing(tmp_path):
    c = _load()
    nn = tmp_path / "derived/nnunet"
    args = ["--model", "infarct", "--data", "glioma", "--threshold", "0.5", "--gpu", "0", "--folds", "0",
            "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")]
    with patch.object(c, "FM", tmp_path), patch.object(c, "NNUNET", nn):
        with pytest.raises(FileNotFoundError, match="the infarct model has no trained fold 0"):
            c.main(args)
        ckpt = fold_dir(nn / "results", "infarct", 0) / "checkpoint_final.pth"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_text("")
        host = DISEASES["glioma"]["name"]
        for sub in ("raw", "preprocessed"):
            (nn / sub / host).mkdir(parents=True)
        (nn / "raw" / host / "cases.json").write_text(json.dumps({"UCSF-PDGM-0004": {"voxel_mm3": 1.0}}))
        (nn / "preprocessed" / host / "splits_final.json").write_text(json.dumps([{"train": [], "val": ["UCSF-PDGM-0004"]}]))
        with pytest.raises(FileNotFoundError, match="UCSF-PDGM-0004: missing .*DWI"):
            c.main(args)
    assert not (tmp_path / "w").exists() and not (tmp_path / "o").exists()
