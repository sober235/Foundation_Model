# tests/test_nndet_infer.py
import json

import nibabel as nib
import numpy as np
import pytest

import anatobind.infer.brain_nndet as B
from anatobind.nndet.boxes import LAYOUT


def _fake_rss(h5, out, pad_to_slices):
    nib.save(nib.Nifti1Image(np.zeros((16, 12, 3), np.float32), np.eye(4)), str(out))


def _predict_with(cases):
    def fake(image, train_dir, work, out_json, gpu):
        out_json.write_text(json.dumps({"layout": LAYOUT, "source": {}, "cases": cases}))
    return fake


def test_run_writes_rows_from_runner_json_and_refuses_an_existing_out(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "rss_h5_to_nifti", _fake_rss)
    fake = _predict_with({"case": {"boxes": [[1, 2, 2, 5, 3, 6]], "scores": [0.7], "labels": [0]}})
    rows = B.run(tmp_path / "x.h5", tmp_path / "out", tmp_path / "train", 0, predict=fake)
    assert rows == [{"z0": 1, "z1": 1, "score": 0.7, "boxes": {"1": [[2, 5, 3, 6]]}}]
    assert json.loads((tmp_path / "out" / "lesions.json").read_text()) == rows
    with pytest.raises(FileExistsError):
        B.run(tmp_path / "x.h5", tmp_path / "out", tmp_path / "train", 0, predict=fake)


def test_run_handles_no_detections_and_refuses_other_case_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "rss_h5_to_nifti", _fake_rss)
    assert B.run(tmp_path / "x.h5", tmp_path / "o1", tmp_path / "t", 0,
                 predict=_predict_with({"case": {"boxes": [], "scores": [], "labels": []}})) == []
    with pytest.raises(ValueError, match="expected"):
        B.run(tmp_path / "x.h5", tmp_path / "o2", tmp_path / "t", 0,
              predict=_predict_with({"other": {"boxes": [], "scores": []}}))


def test_run_runner_pins_the_gpu_and_calls_predict(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(B.subprocess, "run", lambda cmd, check: calls.append(cmd))
    B.run_runner(tmp_path / "a b.nii.gz", tmp_path / "t", tmp_path / "w", tmp_path / "p.json", 5)
    cmd = calls[0]
    assert cmd[:2] == ["bash", "-c"]
    assert "CUDA_VISIBLE_DEVICES=5 nice -n 19 python" in cmd[2] and "nndet_runner.py predict" in cmd[2]
    assert f"--image '{tmp_path / 'a b.nii.gz'}'" in cmd[2]
