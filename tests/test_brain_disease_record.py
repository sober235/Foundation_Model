# tests/test_brain_disease_record.py
import json

import nibabel as nib
import numpy as np
import pytest

import anatobind.infer.brain_disease as B
from anatobind.bind.brain_lookup import BrainBinder


def _row(score, mm3, host="white_matter", side="right", fractions=None, box=(1, 2, 3, 4, 5, 6)):
    return {"component": 1, "family": "tumor", "box": box, "n_voxels": 1, "mm3": mm3, "score": score, "host": host,
            "host_rule": "overlap" if host else None, "host_fractions": fractions if fractions is not None else {host: 1.0},
            "side": side}


def test_detections_drop_small_components_and_score_by_mean_probability():
    pred = np.zeros((12, 12, 6), np.uint8)
    pred[1:4, 1:4, 1:4] = 1                       # 27 voxels
    pred[8, 8, 1] = 1                             # one voxel: 1 mm3, under the floor
    pred[6:8, 9:12, 3:5] = 1                      # 12 voxels
    probs = np.zeros((2, 12, 12, 6), np.float32)
    probs[1][1:4, 1:4, 1:4] = 0.9
    probs[1][6:8, 9:12, 3:5] = 0.6
    probs[1][8, 8, 1] = 0.99
    rows, comp = B.detections(pred, probs, 1.0, "tumor")
    assert [(r["n_voxels"], round(r["score"], 4), r["box"]) for r in rows] == [
        (27, 0.9, (1, 1, 1, 4, 4, 4)), (12, 0.6, (6, 9, 3, 8, 12, 5))]
    assert "ignore" not in rows[0] and comp.shape == pred.shape
    assert [r["n_voxels"] for r in B.detections(pred, probs, 8.0, "tumor")[0]] == [27, 12]   # 8 mm3 voxel: 1 voxel is still under
    assert [r["n_voxels"] for r in B.detections(pred, probs, 12.0, "tumor")[0]] == [1, 27, 12]


def test_bind_rows_attaches_the_structure_of_each_component():
    seg = np.zeros((12, 12, 6), np.int16)
    seg[:6] = 2
    seg[6:] = 41
    pred = np.zeros((12, 12, 6), np.uint8)
    pred[1:4, 1:4, 1:4] = 1
    probs = np.zeros((2, 12, 12, 6), np.float32)
    probs[1][pred == 1] = 0.8
    rows, comp = B.detections(pred, probs, 1.0, "tumor")
    B.bind_rows(rows, comp, BrainBinder(seg, (1.0, 1.0, 1.0)))
    assert rows[0]["host"] == "white_matter" and rows[0]["side"] == "left" and rows[0]["host_fractions"] == {"white_matter": 1.0}


def test_volume_text_switches_units():
    assert B.volume_text(81750.0) == "约 82 mL" and B.volume_text(2880.0) == "约 2.9 mL" and B.volume_text(47.6) == "约 48 mm³"


def test_record_and_sentence_for_one_large_tumour():
    rows = [_row(0.93, 81750.0, fractions={"white_matter": 0.61, "cortex": 0.30, "basal_ganglia": 0.09}),
            _row(0.40, 500.0)]
    rec = B.study_record("s1", "glioma", 0.55, rows)
    assert rec["impression"] == "疑似胶质瘤" and rec["disease_model"] == "glioma" and rec["threshold"] == 0.55
    assert len(rec["lesions"]) == 1 and rec["lesions"][0] == {
        "type": "tumor", "score": 0.93, "box": [1, 2, 3, 4, 5, 6], "volume_mm3": 81750.0, "host": "white_matter",
        "host_rule": "overlap", "host_fractions": {"white_matter": 0.61, "cortex": 0.30, "basal_ganglia": 0.09}, "side": "right"}
    assert rec["sentence"] == "右侧大脑白质存在肿瘤样异常，体积约 82 mL，累及大脑皮层。疑似胶质瘤。"
    json.dumps(rec)


def test_sentence_lists_five_lesions_and_counts_the_rest():
    rows = [_row(0.9 - 0.01 * i, 100.0, host="cortex", side="left") for i in range(7)]
    rec = B.study_record("s2", "metastasis", 0.5, rows)
    assert len(rec["lesions"]) == 7 and [l["score"] for l in rec["lesions"]] == sorted((l["score"] for l in rec["lesions"]), reverse=True)
    assert rec["sentence"].count("左侧大脑皮层存在转移瘤样异常") == 5
    assert rec["sentence"].endswith("；另有 2 处同类异常。疑似脑转移瘤。")


def test_midline_unlocated_and_empty_records():
    rec = B.study_record("s3", "infarct", 0.5, [_row(0.8, 300.0, host="brainstem", side="midline"),
                                                 _row(0.7, 200.0, host=None, side="midline", fractions={})])
    assert rec["sentence"] == "脑干存在梗死样异常，体积约 300 mm³；未能定位的区域存在梗死样异常，体积约 200 mm³。疑似缺血性梗死。"
    empty = B.study_record("s4", "infarct", 0.5, [_row(0.3, 300.0)])
    assert empty["lesions"] == [] and empty["impression"] == "未见相关异常" and empty["sentence"] == "未见梗死样异常。"


def _write(path, data, affine=np.diag([2.0, 2.0, 2.0, 1.0])):
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def test_run_predicts_binds_and_refuses_an_existing_out(tmp_path):
    seg = np.zeros((10, 10, 6), np.int16)
    seg[:5] = 2
    seg[5:] = 41
    anatomy = _write(tmp_path / "seg.nii.gz", seg)
    images = [_write(tmp_path / f"{k}.nii.gz", np.ones((10, 10, 6), np.float32)) for k in ("dwi", "adc")]
    seen = {}

    def fake_predict(dataset_id, in_dir, out_dir, folds, gpu):
        seen.update(id=dataset_id, inputs=sorted(p.name for p in in_dir.iterdir()), folds=folds, gpu=gpu,
                    links=all(p.is_symlink() for p in in_dir.iterdir()))
        out_dir.mkdir(parents=True)
        lab = np.zeros((10, 10, 6), np.uint8)
        lab[6:8, 2:4, 1:3] = 1                                  # 8 voxels of 8 mm3 on the right
        _write(out_dir / "case.nii.gz", lab)
        p = np.zeros((2, 6, 10, 10), np.float32)                # nnU-Net order (C, Z, Y, X)
        p[0] = 1.0
        p[1][1:3, 2:4, 6:8] = 0.75
        p[0][1:3, 2:4, 6:8] = 0.25
        np.savez(out_dir / "case.npz", probabilities=p)

    rec = B.run("infarct", images, anatomy, tmp_path / "out", [0, 1], 3, 0.5, predict=fake_predict)
    assert seen == {"id": 906, "inputs": ["case_0000.nii.gz", "case_0001.nii.gz"], "folds": [0, 1], "gpu": 3, "links": True}
    assert rec["impression"] == "疑似缺血性梗死" and len(rec["lesions"]) == 1
    assert rec["lesions"][0]["volume_mm3"] == 64.0 and rec["lesions"][0]["side"] == "right" and rec["lesions"][0]["score"] == 0.75
    assert json.loads((tmp_path / "out" / "record.json").read_text()) == rec
    with pytest.raises(FileExistsError):
        B.run("infarct", images, anatomy, tmp_path / "out", [0], 3, 0.5, predict=fake_predict)
    with pytest.raises(ValueError, match="2 channels are needed, 1 given"):
        B.run("infarct", images[:1], anatomy, tmp_path / "out2", [0], 3, 0.5, predict=fake_predict)


def test_run_handles_no_detection_and_a_grid_mismatch(tmp_path):
    anatomy = _write(tmp_path / "seg.nii.gz", np.full((10, 10, 6), 2, np.int16))
    images = [_write(tmp_path / f"{k}.nii.gz", np.ones((10, 10, 6), np.float32)) for k in ("dwi", "adc")]

    def nothing(dataset_id, in_dir, out_dir, folds, gpu, shape=(10, 10, 6)):
        out_dir.mkdir(parents=True)
        _write(out_dir / "case.nii.gz", np.zeros(shape, np.uint8))
        p = np.zeros((2,) + shape[::-1], np.float32)
        p[0] = 1.0
        np.savez(out_dir / "case.npz", probabilities=p)

    rec = B.run("infarct", images, anatomy, tmp_path / "o1", [0], 0, 0.5, predict=nothing)
    assert rec["lesions"] == [] and rec["sentence"] == "未见梗死样异常。"
    with pytest.raises(ValueError, match="different grids"):
        B.run("infarct", images, anatomy, tmp_path / "o2", [0], 0, 0.5,
              predict=lambda *a: nothing(*a, shape=(10, 10, 5)))
