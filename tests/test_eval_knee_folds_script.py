import csv
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.nnunet import prepare as anat
from anatobind.nnunet import prepare_lesion as les
from anatobind.nnunet.lesion_labels import boxes_to_label_map, read_boxes_xyz
from anatobind.train.cache import VIEWS


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/eval_knee_folds.py"
    spec = importlib.util.spec_from_file_location("eval_knee_folds", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _perfect_results(export, results, scans):
    """Dataset901 validation = the export seg; Dataset902 validation = the box-filled map, one-hot npz."""
    folds = json.loads((export / "splits.json").read_text())["folds"]
    for scan in scans:
        seg_img = nib.load(str(export / scan / "seg.nii.gz"))
        seg = np.asanyarray(seg_img.dataobj).astype(np.uint8)
        lab = boxes_to_label_map(read_boxes_xyz(export / scan / "boxes.csv"), seg.shape)
        probs = np.zeros((5,) + seg.shape, np.float32)
        for c in range(5):
            probs[c] = lab == c
        for view in VIEWS:
            a, l = anat.validation_path(results, folds[scan], scan, view), les.validation_path(results, folds[scan], scan, view)
            a.parent.mkdir(parents=True, exist_ok=True)
            l.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(seg, seg_img.affine), str(a))
            nib.save(nib.Nifti1Image(lab, seg_img.affine), str(l))
            np.savez_compressed(les.validation_npz_path(results, folds[scan], scan, view),
                                probabilities=np.ascontiguousarray(probs.transpose(0, 3, 2, 1)).astype(np.float16))


def test_perfect_outputs_pass_the_gate_and_bind_every_in_seg_lesion(synthetic_m1r, tmp_path):
    export, _, scans = synthetic_m1r
    _perfect_results(export, tmp_path / "results", scans)
    out = tmp_path / "eval"
    rc = _load_script().main(["--out", str(out), "--export-root", str(export), "--results-root", str(tmp_path / "results"),
                              "--views", "clean,noise_q1"])
    assert rc == 0
    g = json.loads((out / "gate.json").read_text())
    assert g["pass"] is True and g["view"] == "clean" and g["sensitivity_family"] == 1.0 and g["fp_per_scan"] == 0.0
    assert g["n_scans"] == 5 and g["n_gt"] == 20
    with open(out / "records.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    clean = [r for r in rows if r["view"] == "clean"]
    assert len(clean) == 20 and all(r["matched"] == "1" for r in clean)
    cart = next(r for r in clean if r["gt_family"] == "Cartilage Lesion")
    assert cart["bucket"] == "correct" and cart["pred_host_label"] == "2" and cart["given_bucket"] == "correct"
    men = next(r for r in clean if r["gt_family"] == "Meniscal Tear")
    assert men["pred_side"] == "medial" and men["side_correct"] == "1" and json.loads(men["pred_host_fractions"])["5"] == 1.0
    eff = next(r for r in clean if r["gt_family"] == "Effusion")
    assert eff["bucket"] == "" and eff["pred_host_label"] == "" and eff["pred_side"] == "-"
    froc = list(csv.DictReader(open(out / "froc_clean.csv", newline="")))
    assert len(froc) == 19 and froc[0]["thr"] == "0.05" and float(froc[-1]["sensitivity"]) == 1.0
    text = (out / "summary.md").read_text()
    assert "| clean |" in text and "GATE: PASS" in text and "Dice" in text


def test_missing_validation_files_fail_loudly_unless_allowed(synthetic_m1r, tmp_path):
    export, _, scans = synthetic_m1r
    _perfect_results(export, tmp_path / "results", scans)
    folds = json.loads((export / "splits.json").read_text())["folds"]
    les.validation_npz_path(tmp_path / "results", folds[scans[0]], scans[0], "clean").unlink()
    mod = _load_script()
    args = ["--out", str(tmp_path / "e1"), "--export-root", str(export), "--results-root", str(tmp_path / "results"), "--views", "clean"]
    assert mod.main(args) == 2
    assert mod.main(args + ["--allow-missing"]) == 0
    assert json.loads((tmp_path / "e1" / "gate.json").read_text())["n_scans"] == 4
