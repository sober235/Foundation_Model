# tests/test_brain_disease_eval_script.py
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from anatobind.nnunet.brain_disease import fold_dir

SHAPE = (24, 24, 8)
BIG = (slice(2, 6), slice(2, 6), slice(2, 5))          # 48 voxels, left side
SHIFTED = (slice(3, 7), slice(2, 6), slice(2, 5))      # box IoU 0.6 with BIG
OTHER = (slice(16, 20), slice(4, 8), slice(2, 5))      # 48 voxels, right side
STRAY = (slice(8, 11), slice(16, 19), slice(5, 8))     # 27 voxels
# fold 0 = {100101A (BIG found at 0.875, prior surgery), 100101B (OTHER missed)}, fold 1 = {100102A (no lesion, a stray at 0.75)}
CASES = {"100101A": ([BIG], [(SHIFTED, 0.875)], 0), "100101B": ([OTHER], [], 0), "100102A": ([], [(STRAY, 0.75)], 1)}


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/eval_brain_disease.py"
    spec = importlib.util.spec_from_file_location("eval_brain_disease", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _tree(root, cases=None):
    nn, name = root / "derived/nnunet", "Dataset905_BMSRMetastasis"
    info, dice = {}, {0: [], 1: []}
    for case, (gt, pred, fold) in (cases or CASES).items():
        lab, out = np.zeros(SHAPE, np.uint8), np.zeros(SHAPE, np.uint8)
        probs = np.zeros((2,) + SHAPE, np.float32)
        probs[0] = 1.0
        for sl in gt:
            lab[sl] = 1
        for sl, p in pred:
            out[sl] = 1
            probs[1][sl], probs[0][sl] = p, 1.0 - p
        seg = np.zeros(SHAPE, np.int16)
        seg[:12], seg[12:] = 2, 41
        _save(nn / "raw" / name / "labelsTr" / f"{case}.nii.gz", lab)
        v = fold_dir(nn / "results", "metastasis", fold) / "validation"
        _save(v / f"{case}.nii.gz", out)
        np.savez(v / f"{case}.npz", probabilities=np.ascontiguousarray(probs.transpose(0, 3, 2, 1)))
        _save(root / "derived/synthseg/bmsr/seg_native" / f"{case}_T1pre_seg.nii.gz", seg)
        info[case] = {"patient": case[:-1], "voxel_mm3": 1.0, "n_label_voxels": int(lab.sum()), "prior_surgery": case == "100101A"}
        dice[fold].append({"metrics": {"1": {"Dice": 0.5 if gt and pred else (0.0 if gt else float("nan")), "n_ref": int(lab.sum())}}})
    for fold, cases in dice.items():
        (fold_dir(nn / "results", "metastasis", fold) / "validation" / "summary.json").write_text(json.dumps({"metric_per_case": cases}))
    (nn / "raw" / name / "cases.json").write_text(json.dumps(info))
    (nn / "preprocessed" / name).mkdir(parents=True)
    (nn / "preprocessed" / name / "splits_final.json").write_text(json.dumps(
        [{"train": ["100102A"], "val": ["100101A", "100101B"]}, {"train": ["100101A", "100101B"], "val": ["100102A"]}]))


def test_fold_report_records_and_refusals(tmp_path):
    mod = _load()
    _tree(tmp_path)
    with patch.object(mod, "FM", tmp_path), patch.object(mod, "NNUNET", tmp_path / "derived/nnunet"):
        out, rec = tmp_path / "rep", tmp_path / "records"
        mod.main(["--disease", "metastasis", "--folds", "0", "--out", str(out), "--records", str(rec), "--workers", "1"])
        v = json.loads((out / "verdict.json").read_text())
        # fold 0: two lesions, BIG found (score 0.875) up to threshold 0.85, no false positive
        assert v == {"kind": "early_reading", "folds": [0], "pass": None, "operating_point": True, "sensitivity": 0.5,
                     "thr": 0.85, "fp_per_scan": 0.0, "beyond_budget": None, "stop_remaining_folds": False,
                     "early_stop_undecided": False}
        rep = (out / "REPORT.md").read_text()
        assert "NOT the gate" in rep and "NOT_EVIDENCE" in rep
        assert "| <5 | 2 | 1 | 0.5000 |" in rep                                   # 48 mm3 is a 4.5 mm sphere
        assert "| no | 1 | 0 | 0.0000 |" in rep and "| yes | 1 | 1 | 1.0000 |" in rep
        assert '"n_cases": 2' in rep and '"mean": 0.25' in rep                    # Dice 0.5 and 0.0
        assert (out / "froc.csv").read_text().splitlines()[17] == "0.85,1,0.500000,0.000000"
        assert "Scans: 2; lesions counted: 2; ignored: 0" in (out / "output.txt").read_text()
        a = json.loads((rec / "100101A.json").read_text())
        assert a["impression"] == "疑似脑转移瘤" and a["lesions"][0]["side"] == "left" and a["lesions"][0]["score"] == 0.875
        assert a["sentence"] == "左侧大脑白质存在转移瘤样异常，体积约 48 mm³。疑似脑转移瘤。"
        assert a["threshold"] == 0.85 and a["model_folds"] == [0] and "NOT_EVIDENCE" in a["anatomy_source"]
        assert json.loads((rec / "100101B.json").read_text())["impression"] == "未检出相关异常"
        assert '"n_scans_over_budget": 0' in rep and '"nearest_rate": 0.0' in rep
        assert '"host_side_agreement": 1.0' in rep and '"unlocated_rate": 0.0' in rep
        assert a["lesions"][0]["host_side"] == "left" and a["lesions"][0]["host_sides"] == {"white_matter": "left"}
        assert sorted(p.name for p in rec.iterdir()) == ["100101A.json", "100101B.json"]
        with pytest.raises(FileExistsError):
            mod.main(["--disease", "metastasis", "--folds", "0", "--out", str(out), "--workers", "1"])
        with pytest.raises(FileExistsError):
            mod.main(["--disease", "metastasis", "--folds", "0", "--out", str(tmp_path / "rep2"), "--records", str(rec), "--workers", "1"])
        assert not (tmp_path / "rep2").exists()


def test_both_folds_count_the_stray_and_a_missing_prediction_is_named(tmp_path):
    mod = _load()
    _tree(tmp_path)
    with patch.object(mod, "FM", tmp_path), patch.object(mod, "NNUNET", tmp_path / "derived/nnunet"):
        mod.main(["--disease", "metastasis", "--folds", "0", "1", "--out", str(tmp_path / "rep"), "--workers", "1"])
        lines = (tmp_path / "rep" / "froc.csv").read_text().splitlines()
        assert lines[15] == "0.75,1,0.500000,0.333333" and lines[16] == "0.80,1,0.500000,0.000000"
        (fold_dir(tmp_path / "derived/nnunet/results", "metastasis", 1) / "validation" / "100102A.npz").rename(tmp_path / "moved.npz")
        with pytest.raises(FileNotFoundError, match="fold 1 case 100102A"):
            mod.main(["--disease", "metastasis", "--folds", "1", "--out", str(tmp_path / "rep3"), "--workers", "1"])


def test_no_operating_point_is_said_plainly_and_writes_no_records(tmp_path):
    mod = _load()
    strays = [((slice(8, 11), slice(16, 19), slice(5, 8)), 0.96875), ((slice(14, 17), slice(16, 19), slice(0, 3)), 0.96875),
              ((slice(20, 23), slice(18, 21), slice(4, 7)), 0.96875)]
    # three confident strays in each fold 0 scan: 3 false positives per scan at every threshold of the grid
    _tree(tmp_path, {"100101A": ([BIG], strays, 0), "100101B": ([OTHER], strays, 0), "100102A": ([], [], 1)})
    with patch.object(mod, "FM", tmp_path), patch.object(mod, "NNUNET", tmp_path / "derived/nnunet"):
        out, rec = tmp_path / "rep", tmp_path / "records"
        mod.main(["--disease", "metastasis", "--folds", "0", "--out", str(out), "--records", str(rec), "--workers", "1"])
    assert json.loads((out / "verdict.json").read_text()) == {
        "kind": "early_reading", "folds": [0], "pass": None, "operating_point": False, "sensitivity": None, "thr": None,
        "fp_per_scan": None, "beyond_budget": {"thr": 0.95, "sensitivity": 0.0, "fp_per_scan": 3.0, "n_hit": 0, "n_gt": 2,
                                               "out_of_reach": False},
        "stop_remaining_folds": False, "early_stop_undecided": True}
    rep = (out / "REPORT.md").read_text()
    assert "there is no operating point" in rep and "does not stop the remaining folds" in rep
    assert "decisive all the same" not in rep          # two lesions in the fold: nothing is out of reach of 0.3
    assert "no operating point" in (out / "output.txt").read_text() and not rec.exists()
    assert (out / "froc.csv").read_text().splitlines()[19] == "0.95,0,0.000000,3.000000"


def test_a_failure_while_the_report_is_computed_leaves_no_folder(tmp_path):
    mod = _load()
    _tree(tmp_path)
    with patch.object(mod, "FM", tmp_path), patch.object(mod, "NNUNET", tmp_path / "derived/nnunet"), \
            patch.object(mod, "binding_agreement", side_effect=RuntimeError("late failure")):
        with pytest.raises(RuntimeError, match="late failure"):
            mod.main(["--disease", "metastasis", "--folds", "0", "--out", str(tmp_path / "rep"),
                      "--records", str(tmp_path / "records"), "--workers", "1"])
    assert not (tmp_path / "rep").exists() and not (tmp_path / "records").exists()


def test_records_hold_only_detections_at_the_operating_threshold(tmp_path):
    mod = _load()
    # fold 0: BIG found at 0.875 (the operating threshold becomes 0.85), one stray at 0.625 in each scan
    _tree(tmp_path, {"100101A": ([BIG], [(SHIFTED, 0.875), (STRAY, 0.625)], 0), "100101B": ([OTHER], [(STRAY, 0.625)], 0),
                     "100102A": ([], [], 1)})
    with patch.object(mod, "FM", tmp_path), patch.object(mod, "NNUNET", tmp_path / "derived/nnunet"):
        mod.main(["--disease", "metastasis", "--folds", "0", "--out", str(tmp_path / "rep"),
                  "--records", str(tmp_path / "records"), "--workers", "1"])
    a = json.loads((tmp_path / "records" / "100101A.json").read_text())
    b = json.loads((tmp_path / "records" / "100101B.json").read_text())
    assert a["threshold"] == 0.85 and [l["score"] for l in a["lesions"]] == [0.875]      # the stray at 0.625 is left out
    assert b["lesions"] == [] and b["sentence"] == "本模型未检出转移瘤样异常（阈值 0.85）。"


def test_a_reading_without_an_operating_point_says_when_it_is_decisive_all_the_same(tmp_path):
    mod = _load()
    strays = [((slice(8, 11), slice(16, 19), slice(5, 8)), 0.96875), ((slice(14, 17), slice(16, 19), slice(0, 3)), 0.96875),
              ((slice(20, 23), slice(18, 21), slice(4, 7)), 0.96875)]
    seven = [(slice(x, x + 2), slice(0, 2), slice(0, 3)) for x in range(0, 21, 3)]      # 7 lesions of 12 mm3, none found
    _tree(tmp_path, {"100101A": (seven, strays, 0), "100101B": ([], strays, 0), "100102A": ([], [], 1)})
    with patch.object(mod, "FM", tmp_path), patch.object(mod, "NNUNET", tmp_path / "derived/nnunet"):
        mod.main(["--disease", "metastasis", "--folds", "0", "--out", str(tmp_path / "rep"), "--workers", "1"])
    v = json.loads((tmp_path / "rep" / "verdict.json").read_text())
    assert v["operating_point"] is False and v["early_stop_undecided"] is True and v["stop_remaining_folds"] is False
    assert v["beyond_budget"] == {"thr": 0.95, "sensitivity": 0.0, "fp_per_scan": 3.0, "n_hit": 0, "n_gt": 7, "out_of_reach": True}
    rep = (tmp_path / "rep" / "REPORT.md").read_text()
    assert "decisive all the same" in rep and "finds 0 of 7 lesions" in rep
