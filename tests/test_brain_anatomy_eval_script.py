# tests/test_brain_anatomy_eval_script.py
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from anatobind.anatomy.labels import to_synthseg


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/eval_brain_anatomy.py"
    spec = importlib.util.spec_from_file_location("eval_brain_anatomy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthseg():
    seg = np.zeros((30, 30, 8), np.int16)
    seg[2:28, 2:28, 1:7] = 2
    seg[15:28, 2:28, 1:7] = 41
    seg[10:20, 10:20, 3:5] = 10
    return seg


def _write(path, data, zooms=(2.0, 2.0, 5.0)):
    nib.save(nib.Nifti1Image(data, np.diag(list(zooms) + [1.0])), str(path))
    return path


def test_stem_metrics_and_summary(tmp_path):
    mod = _load()
    seg = _synthseg()
    student = mod.student_of(seg)
    wrong = student.copy()
    wrong[10:20, 10:20, 3:5] = 1                                      # the thalamus is called white matter
    mask = (seg > 0).astype(np.uint8)
    rows = [{"lesion_id": 1, "x0": 11, "x1": 14, "y0": 11, "y1": 14, "z0": 3, "z1": 4}]
    m = mod.stem_metrics(wrong, seg, mask, (2.0, 2.0, 5.0), rows, is_test=True)
    assert m["reliable"] == [2, 5] and m["agreement"]["rate"] == 0.0 and m["outline_dice"] == 1.0
    assert m["dice"][5] == 0.0 and m["dice"][1] < 1.0 and m["dice"][6] is None
    assert m["low_slice_area_cm2"] == {0: 0.0, 1: pytest.approx(26 * 26 * 4 / 100.0, abs=0.1)}
    exact = mod.stem_metrics(student, seg, mask, (2.0, 2.0, 5.0), rows, is_test=False)
    assert exact["agreement"]["rate"] == 1.0 and exact["outline_dice"] is None and exact["dice"][5] == 1.0
    per_stem = {"a": {"split": "test", **m}, "b": {"split": "train", **exact}}
    sim = {"s1": {**{c: 1.0 for c in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)}, 14: 0.5}}
    s = mod.summarize(per_stem, sim)
    assert s["host_agreement"]["n_evaluated"] == 2 and s["host_agreement"]["rate"] == 0.5 and s["outline"]["n_test_stacks"] == 1
    assert s["simulated_test"]["ventricles"] == 0.5 and s["simulated_test"]["mean_host_dice"] == 1.0
    assert s["verdict"]["pass"] is False and s["verdict"]["outline_dice"] is True and s["n_stacks"] == 2
    out = tmp_path / "rep"
    out.mkdir()
    mod.report(s, per_stem, out, "cmd", "abc1234")
    text = (out / "REPORT.md").read_text()
    assert "**fail**" in text and "Code: commit abc1234" in text and "Level R" in text and "thalamus_left" in text
    assert json.loads((out / "verdict.json").read_text())["verdict"]["pass"] is False
    assert (out / "per_stem.csv").read_text().splitlines()[1].startswith("a,test,2,5,")


def test_main_runs_the_two_models_once_each_and_refuses_existing_dirs(tmp_path, monkeypatch):
    mod = _load()
    raw = tmp_path / "nnunet" / "raw"
    seg = _synthseg()
    seg_dir = tmp_path / "segs"
    seg_dir.mkdir()
    for folder, stems in (("imagesTr", ["file_brain_AXFLAIR_200_1"]), ("imagesTs", ["file_brain_AXFLAIR_200_2"])):
        (raw / "Dataset908_FastMRIBrainOutline" / folder).mkdir(parents=True)
        for s in stems:
            _write(raw / "Dataset908_FastMRIBrainOutline" / folder / f"{s}_0000.nii.gz", np.where(seg > 0, 500.0, 100.0).astype(np.float32))
            _write(seg_dir / f"{s}_seg.nii.gz", seg)
    (raw / "Dataset907_BrainAnatomyFLAIR" / "imagesTs").mkdir(parents=True)
    _write(raw / "Dataset907_BrainAnatomyFLAIR" / "imagesTs" / "SIM_s0_0000.nii.gz", np.zeros((30, 30, 8), np.float32))
    sim_work = tmp_path / "work"
    (sim_work / "sim" / "test").mkdir(parents=True)
    _write(sim_work / "sim" / "test" / "SIM_s0.nii.gz", mod.student_of(seg))
    monkeypatch.setattr(mod, "NNUNET_ROOT", tmp_path / "nnunet")
    monkeypatch.setattr(mod, "load_registry", lambda: [{"file": "file_brain_AXFLAIR_200_1", "lesion_id": 1, "x0": 11, "x1": 14, "y0": 11, "y1": 14, "z0": 3, "z1": 4}])
    monkeypatch.setattr(mod, "code_version", lambda repo: "deadbee")
    calls = []

    def predict(dataset_id, config, in_dir, out_dir, folds, gpu):
        calls.append((dataset_id, config, in_dir.name, folds, gpu))
        out_dir.mkdir(parents=True)
        for p in sorted(in_dir.glob("*_0000.nii.gz")):
            name = p.name[:-len("_0000.nii.gz")]
            img = nib.load(str(p))
            data = np.asarray(img.dataobj)
            lab = (data > 300).astype(np.uint8) if dataset_id == 908 else mod.student_of(seg)
            nib.save(nib.Nifti1Image(lab, img.affine), str(out_dir / f"{name}.nii.gz"))

    out, work = tmp_path / "out", tmp_path / "pred"
    summary = mod.main(["--out", str(out), "--work", str(work), "--gpu", "6", "--sim-work", str(sim_work), "--seg-dir", str(seg_dir)], predict=predict)
    assert [c[:2] for c in calls] == [(908, "2d"), (907, "3d_fullres"), (907, "3d_fullres")] and calls[0][2] == "stacks" and calls[1][2] == "stripped"
    assert summary["n_stacks"] == 2 and summary["verdict"]["pass"] is True and summary["host_agreement"]["n_evaluated"] == 1
    assert summary["outline"]["n_test_stacks"] == 1 and summary["simulated_test"]["mean_host_dice"] == 1.0
    assert (out / "REPORT.md").exists() and (work / "stripped" / "file_brain_AXFLAIR_200_1_mask.nii.gz").exists()
    with pytest.raises(FileExistsError):
        mod.main(["--out", str(out), "--work", str(tmp_path / "pred2"), "--gpu", "6"], predict=predict)
