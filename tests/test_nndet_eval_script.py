# tests/test_nndet_eval_script.py
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from anatobind.nndet.boxes import LAYOUT
from anatobind.nnunet.brain_lesion import DATASET_NAME, TRAINER

REG = [
    {"lesion_id": 0, "file": "A", "band": "0", "n_slices": 1, "inplane_mm": 2.5, "stratum_geometry": "g1",
     "x0": 5, "y0": 5, "x1": 10, "y1": 10, "z0": 10, "z1": 10},
    {"lesion_id": 1, "file": "A", "band": ">4", "n_slices": 2, "inplane_mm": 3.5, "stratum_geometry": "g2",
     "x0": 20, "y0": 20, "x1": 25, "y1": 25, "z0": 15, "z1": 16},
    {"lesion_id": 2, "file": "B", "band": "2-4", "n_slices": 1, "inplane_mm": 4.5, "stratum_geometry": "g1",
     "x0": 30, "y0": 30, "x1": 35, "y1": 35, "z0": 8, "z1": 8},
]
# fold 0 = {A (lesions 0, 1), N (normal)}, fold 1 = {B}
SPLITS = [{"train": ["B"], "val": ["A", "N"]}, {"train": ["A", "N"], "val": ["B"]}]
INFO = {"A": {"patient_id": "p1", "kind": "lesion"}, "B": {"patient_id": "p2", "kind": "lesion"},
        "N": {"patient_id": "p3", "kind": "normal"}}
# runner layout [s_lo, r_lo, s_hi, r_hi, c_lo, c_hi]
BOX0, BOX1, BOX2 = [10, 5, 11, 10, 5, 10], [15, 20, 17, 25, 20, 25], [8, 30, 9, 35, 30, 35]


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/eval_brain_nndet.py"
    spec = importlib.util.spec_from_file_location("eval_brain_nndet", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runner_json(path, cases):
    path.write_text(json.dumps({"layout": LAYOUT, "source": {}, "cases": cases}))
    return path


def _nnunet_outputs(results, config):
    """fold 0 validation: A has only lesion 0 predicted (prob 0.8); N has nothing."""
    d = results / DATASET_NAME / f"{TRAINER}__nnUNetPlans__{config}" / "fold_0" / "validation"
    d.mkdir(parents=True)
    shape = (64, 64, 20)
    for case in ("A", "N"):
        lab = np.zeros(shape, np.uint8)
        probs = np.zeros((2,) + shape, np.float32)
        if case == "A":
            lab[5:10, 5:10, 10] = 1
            probs[1][5:10, 5:10, 10] = 0.8
        nib.save(nib.Nifti1Image(lab, np.eye(4)), str(d / f"{case}.nii.gz"))
        np.savez_compressed(d / f"{case}.npz", probabilities=np.ascontiguousarray(probs.transpose(0, 3, 2, 1)))


def _patched(mod, tmp_path, instance_map=None):
    (tmp_path / "cases.json").write_text(json.dumps(INFO))
    (tmp_path / "splits.json").write_text(json.dumps(SPLITS))
    (tmp_path / "instances.json").write_text(json.dumps(instance_map or {"A": {"1": 0, "2": 1}, "B": {"1": 2}, "N": {}}))
    return [patch.object(mod, "CASES_JSON", tmp_path / "cases.json"),
            patch.object(mod, "SPLITS_JSON", tmp_path / "splits.json"),
            patch.object(mod, "NNUNET_RESULTS", tmp_path / "results"),
            patch.object(mod, "INSTANCE_MAP", tmp_path / "instances.json"),
            patch.object(mod, "EXPECTED_N_GT", 3), patch.object(mod, "EXPECTED_N_SCANS", 3),
            patch.object(mod, "load_registry", return_value=REG)]


def _enter(ps):
    for p in ps:
        p.start()


def _exit(ps):
    for p in ps:
        p.stop()


def test_fold_report_rule_a_paired_table_and_refusals(tmp_path):
    mod = _load()
    _nnunet_outputs(tmp_path / "results", "2d")
    _nnunet_outputs(tmp_path / "results", "3d_fullres")
    # nnDetection: lesion 0 at 0.9, lesion 1 at 0.6, one false positive on N at 0.3
    dets = _runner_json(tmp_path / "d.json", {"A": {"boxes": [BOX0, BOX1], "scores": [0.9, 0.6], "labels": [0, 0]},
                                              "N": {"boxes": [[5, 40, 6, 45, 40, 45]], "scores": [0.3], "labels": [0]}})
    ps = _patched(mod, tmp_path)
    _enter(ps)
    try:
        out = tmp_path / "rep"
        mod.main(["--dets", str(dets), "--swept", str(dets), "--folds", "0", "--out", str(out)])
        ra = json.loads((out / "rule_a.json").read_text())
        # nnDetection: sensitivity 1.0 for every threshold <= 0.60 with 0 FP above 0.30 -> operating threshold 0.60
        # nnU-Net 2d: sensitivity 0.5 for thresholds <= 0.80 -> operating threshold 0.80, 1 hit
        # required = 1 + ceil(0.05 * 2) = 2
        assert ra == {"pass": True, "nndet_hits": 2, "baseline_hits": 1, "required": 2, "n_gt": 2, "margin": 0.05,
                      "nndet_thr": 0.6, "baseline_thr": 0.8}
        rep = (out / "REPORT.md").read_text()
        assert "NOT the D1 gate" in rep
        assert "| both | 1 |" in rep and "| only nnDetection | 1 |" in rep and "| only nnU-Net 2d | 0 |" in rep
        assert "| neither | 0 |" in rep
        assert "NOT_GATE" in rep and "3d_fullres" in rep
        assert "| >4 | 1 | 1 |" in rep                                   # band stratum of lesion 1
        assert "Normal FP per volume at 0.60: 0.0000" in (out / "output.txt").read_text()
        assert (out / "froc.csv").read_text().splitlines()[0] == "thr,n_hit,sensitivity,fp_per_scan"
        with pytest.raises(FileExistsError):
            mod.main(["--dets", str(dets), "--folds", "0", "--out", str(out)])
        short = _runner_json(tmp_path / "short.json", {"A": {"boxes": [], "scores": []}})
        with pytest.raises(KeyError, match="N"):
            mod.main(["--dets", str(short), "--folds", "0", "--out", str(tmp_path / "rep2")])
    finally:
        _exit(ps)


def test_gt_check_passes_on_exact_boxes_and_fails_on_a_displaced_one(tmp_path):
    mod = _load()
    good = _runner_json(tmp_path / "gt.json", {
        "A": {"boxes": [BOX0, BOX1], "scores": [1.0, 1.0], "instances": [1, 2]},
        "B": {"boxes": [BOX2], "scores": [1.0], "instances": [1]},
        "N": {"boxes": [], "scores": [], "instances": []}})
    ps = _patched(mod, tmp_path)
    _enter(ps)
    try:
        mod.main(["--gt-check", str(good), "--out", str(tmp_path / "ok")])
        res = json.loads((tmp_path / "ok" / "gt_check.json").read_text())
        assert res["pass"] is True and res["lost"] == [] and res["below_iou"] == [] and res["n_fp"] == 0
        assert res["n_present"] == 3 and res["min_iou"] == pytest.approx(1.0)
        bad = _runner_json(tmp_path / "bad.json", {
            "A": {"boxes": [BOX0, BOX1], "scores": [1.0, 1.0], "instances": [1, 2]},
            "B": {"boxes": [[0, 50, 1, 55, 50, 55]], "scores": [1.0], "instances": [1]},
            "N": {"boxes": [], "scores": [], "instances": []}})
        with pytest.raises(SystemExit):
            mod.main(["--gt-check", str(bad), "--out", str(tmp_path / "no")])
        assert json.loads((tmp_path / "no" / "gt_check.json").read_text())["below_iou"] == [2]
    finally:
        _exit(ps)


def test_gt_check_fails_when_the_margin_is_not_undone(tmp_path):
    """Every box one voxel larger at its low ends (the +-1 expansion left in): IoU stays above 0.1 for all three
    lesions (25 / 72 = 0.35 for a 5x5x1 box), so the old criterion passed; the exact-IoU share must fail it."""
    mod = _load()

    def grown(b):
        return [b[0] - 1, b[1] - 1, b[2], b[3], b[4] - 1, b[5]]

    big = _runner_json(tmp_path / "big.json", {
        "A": {"boxes": [grown(BOX0), grown(BOX1)], "scores": [1.0, 1.0], "instances": [1, 2]},
        "B": {"boxes": [grown(BOX2)], "scores": [1.0], "instances": [1]},
        "N": {"boxes": [], "scores": [], "instances": []}})
    ps = _patched(mod, tmp_path)
    _enter(ps)
    try:
        with pytest.raises(SystemExit):
            mod.main(["--gt-check", str(big), "--out", str(tmp_path / "big")])
        res = json.loads((tmp_path / "big" / "gt_check.json").read_text())
        assert res["pass"] is False and res["below_iou"] == [] and res["n_fp"] == 0
        assert res["n_present"] == 3 and res["n_iou_ge_0_99"] == 0 and res["exact_share_required"] == 0.8
        assert 0.1 < res["min_iou"] < 0.99
    finally:
        _exit(ps)
