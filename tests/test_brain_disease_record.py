# tests/test_brain_disease_record.py
import json

import nibabel as nib
import numpy as np
import pytest

import anatobind.infer.brain_disease as B
from anatobind.bind.brain_lookup import BrainBinder


def _row(score, mm3, host="white_matter", side="right", fractions=None, box=(1, 2, 3, 4, 5, 6), host_side=None, sides=None):
    """sides: the side of the involved structures that differ from the main structure's side."""
    fractions = fractions if fractions is not None else {host: 1.0}
    host_side = side if host_side is None else host_side
    return {"component": 1, "family": "tumor", "box": box, "n_voxels": 1, "mm3": mm3, "score": score, "host": host,
            "host_rule": "overlap" if host else None, "host_fractions": fractions, "side": side, "host_side": host_side,
            "host_sides": {h: (sides or {}).get(h, host_side) for h in fractions}, "host_distance_mm": 0.0 if host else None}


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
        "host_rule": "overlap", "host_fractions": {"white_matter": 0.61, "cortex": 0.30, "basal_ganglia": 0.09}, "side": "right",
        "host_side": "right", "host_sides": {"white_matter": "right", "cortex": "right", "basal_ganglia": "right"},
        "host_distance_mm": 0.0}
    assert rec["sentence"] == "右侧大脑白质存在肿瘤样异常，体积约 82 mL，累及大脑皮层。疑似胶质瘤。"
    json.dumps(rec)


def test_sentence_lists_five_lesions_and_counts_the_rest():
    # seven lesions; the smaller a lesion, the higher its score, as on real outputs
    rows = [_row(0.9 - 0.01 * i, 100.0 * (i + 1), host="cortex", side="left") for i in range(7)]
    rec = B.study_record("s2", "metastasis", 0.5, rows)
    assert len(rec["lesions"]) == 7 and [l["score"] for l in rec["lesions"]] == sorted((l["score"] for l in rec["lesions"]), reverse=True)
    assert rec["sentence"].count("左侧大脑皮层存在转移瘤样异常") == 5
    assert rec["sentence"].endswith("；另有 2 处同类异常。疑似脑转移瘤。")
    # the sentence names the five largest, largest first; the two it leaves out are the smallest
    assert [c.split("体积")[1] for c in rec["sentence"].split("；")[:5]] == [
        "约 700 mm³", "约 600 mm³", "约 500 mm³", "约 400 mm³", "约 300 mm³"]
    assert [l["volume_mm3"] for l in rec["lesions"][:2]] == [100.0, 200.0]      # the record stays in the order of the scores


def test_midline_unlocated_and_empty_records():
    rec = B.study_record("s3", "infarct", 0.5, [_row(0.8, 300.0, host="brainstem", side="midline"),
                                                 _row(0.7, 200.0, host=None, side="midline", fractions={})])
    assert rec["sentence"] == "脑干存在梗死样异常，体积约 300 mm³；未能定位的区域存在梗死样异常，体积约 200 mm³。疑似缺血性梗死。"
    empty = B.study_record("s4", "infarct", 0.5, [_row(0.3, 300.0)])
    assert empty["lesions"] == [] and empty["impression"] == "未检出相关异常"
    assert empty["sentence"] == "本模型未检出梗死样异常（阈值 0.50）。"


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
    assert rec["lesions"] == [] and rec["sentence"] == "本模型未检出梗死样异常（阈值 0.50）。" and rec["model_folds"] == [0]
    with pytest.raises(ValueError, match="different grids"):
        B.run("infarct", images, anatomy, tmp_path / "o2", [0], 0, 0.5,
              predict=lambda *a: nothing(*a, shape=(10, 10, 5)))


def test_run_refuses_channels_off_the_anatomy_grid_before_anything_is_written(tmp_path):
    anatomy = _write(tmp_path / "seg.nii.gz", np.full((10, 10, 6), 2, np.int16))
    ones = np.ones((10, 10, 6), np.float32)
    dwi = _write(tmp_path / "dwi.nii.gz", ones)
    thick = _write(tmp_path / "thick.nii.gz", ones, affine=np.diag([2.0, 2.0, 5.0, 1.0]))      # same shape, other voxel size
    moved = np.diag([2.0, 2.0, 2.0, 1.0])
    moved[0, 3] = 4.0                                                                          # same voxels, shifted by 4 mm
    moved = _write(tmp_path / "moved.nii.gz", ones, affine=moved)
    small = _write(tmp_path / "small.nii.gz", np.ones((10, 10, 5), np.float32))
    called = []
    for k, bad in enumerate((thick, moved, small)):
        with pytest.raises(ValueError, match="is not on the anatomy's grid"):
            B.run("infarct", [dwi, bad], anatomy, tmp_path / f"o{k}", [0], 0, 0.5, predict=lambda *a: called.append(a))
        assert not (tmp_path / f"o{k}").exists()
    with pytest.raises(FileNotFoundError):
        B.run("infarct", [dwi, tmp_path / "missing.nii.gz"], anatomy, tmp_path / "o9", [0], 0, 0.5,
              predict=lambda *a: called.append(a))
    assert called == [] and not (tmp_path / "o9").exists()


def test_a_rounding_difference_of_the_affine_is_the_same_grid_and_links_need_every_file(tmp_path):
    anatomy = nib.load(str(_write(tmp_path / "seg.nii.gz", np.full((4, 4, 4), 2, np.int16))))
    near = np.diag([2.0, 2.0, 2.0, 1.0])
    near[:3] += 1e-5
    a = _write(tmp_path / "a.nii.gz", np.ones((4, 4, 4), np.float32), affine=near)
    B.check_grid([a], anatomy)                                                                 # no error
    with pytest.raises(FileNotFoundError):
        B.link_inputs(tmp_path / "in", "case", [a, tmp_path / "missing.nii.gz"], 2)
    assert not (tmp_path / "in").exists()


def test_the_score_is_the_mean_and_a_foreign_probability_map_is_refused():
    pred = np.zeros((12, 12, 6), np.uint8)
    pred[2:6, 2:6, 2:3] = 1                                       # one component of 16 voxels
    probs = np.zeros((2, 12, 12, 6), np.float32)
    probs[1][2:4, 2:6, 2:3], probs[1][4:6, 2:6, 2:3] = 0.875, 0.625
    rows, _ = B.detections(pred, probs, 1.0, "tumor")
    assert [(r["n_voxels"], r["score"]) for r in rows] == [(16, 0.75)]
    with pytest.raises(ValueError, match="do not belong to this label map"):
        B.detections(pred, probs.transpose(0, 2, 1, 3)[:, ::-1].copy(), 1.0, "tumor")   # the same values on other axes
    small = np.zeros((12, 12, 6), np.uint8)
    small[9, 9, 4] = 1                                            # under the floor: dropped, but checked all the same
    with pytest.raises(ValueError, match="do not belong to this label map"):
        B.detections(small, np.zeros((2, 12, 12, 6), np.float32), 1.0, "tumor")


def test_the_sentence_says_next_to_for_the_nearest_rule_and_no_side_for_the_brainstem():
    near = dict(_row(0.8, 64.0, side="left"), host_rule="nearest", host_fractions={}, host_distance_mm=10.0)
    stem = _row(0.7, 300.0, host="brainstem", side="left", host_side="midline",
                fractions={"brainstem": 0.9, "cerebellum": 0.1}, sides={"cerebellum": "left"})
    rec = B.study_record("s5", "metastasis", 0.5, [near, stem], folds=[3, 1])
    # the larger lesion comes first in the sentence, the higher score first in the record
    assert rec["sentence"] == ("脑干存在转移瘤样异常，体积约 300 mm³，累及左侧小脑；"
                               "邻近左侧大脑白质（未与任何结构重叠）存在转移瘤样异常，体积约 64 mm³。疑似脑转移瘤。")
    assert [l["score"] for l in rec["lesions"]] == [0.8, 0.7]
    assert rec["lesions"][1]["side"] == "left" and rec["lesions"][1]["host_side"] == "midline"   # the record keeps both
    assert rec["model_folds"] == [1, 3] and "NOT_EVIDENCE" in rec["anatomy_source"]


def test_the_side_before_the_structure_is_the_structure_s_own_and_a_far_lesion_is_not_located():
    across = _row(0.9, 900.0, host="thalamus", side="right", host_side="left",
                  fractions={"thalamus": 0.38, "white_matter": 0.21, "cortex": 0.21, "basal_ganglia": 0.10, "brainstem": 0.10},
                  sides={"white_matter": "right", "cortex": "bilateral", "brainstem": "midline"})
    far = dict(_row(0.8, 64.0, side="right"), host_rule="nearest", host_fractions={}, host_distance_mm=10.01)
    rec = B.study_record("s7", "glioma", 0.5, [across, far])
    # one involved structure lies on another side: every one gets its side, the basal ganglia on the main structure's
    # side too, and the brainstem, which has none, comes first so that no side word is read on to it
    assert rec["sentence"] == ("左侧丘脑存在肿瘤样异常，体积约 900 mm³，累及脑干、右侧大脑白质、双侧大脑皮层、左侧基底节；"
                               "未能定位的区域存在肿瘤样异常，体积约 64 mm³。疑似胶质瘤。")
    assert rec["lesions"][0]["host_sides"] == {"thalamus": "left", "white_matter": "right", "cortex": "bilateral",
                                               "basal_ganglia": "left", "brainstem": "midline"}
    # every involved structure on the main structure's side, or without a side: the short form
    same = _row(0.9, 900.0, host="cerebellum", side="left", fractions={"cerebellum": 0.6, "white_matter": 0.2, "brainstem": 0.2},
                sides={"brainstem": "midline"})
    assert B.lesion_clause(B.study_record("s8", "infarct", 0.5, [same])["lesions"][0]) == (
        "左侧小脑存在梗死样异常，体积约 900 mm³，累及大脑白质、脑干")
    assert rec["lesions"][1]["host"] == "white_matter" and rec["lesions"][1]["host_distance_mm"] == 10.01
    assert B.NEAR_MM == 10.0


def test_a_score_equal_to_the_threshold_enters_the_record():
    rec = B.study_record("s6", "glioma", 0.75, [_row(0.75, 100.0), _row(0.7499, 100.0)])
    assert [l["score"] for l in rec["lesions"]] == [0.75] and rec["model_folds"] is None


def test_the_lesions_that_are_only_counted_still_name_their_places():
    big = [_row(0.6, 1000.0 - i, host="cortex", side="left") for i in range(5)]
    rest = [_row(0.9, 90.0, host="cortex", side="right"), _row(0.9, 80.0, host="thalamus", side="left"),
            _row(0.9, 70.0, host="cortex", side="left"), _row(0.9, 60.0, host="cortex", side="right"),
            dict(_row(0.9, 50.0, side="left"), host_rule="nearest", host_fractions={}, host_distance_mm=3.0),
            _row(0.9, 40.0, host=None, side="midline", fractions={}),
            _row(0.9, 30.0, host="other_deep_grey", side="left")]
    rec = B.study_record("s9", "infarct", 0.5, big + rest)
    # inside the bracket the deep grey matter is written without its own bracket
    assert rec["sentence"].endswith(
        "；另有 7 处同类异常（还见于右侧大脑皮层、左侧丘脑、邻近左侧大脑白质、未能定位的区域、左侧深部灰质）。疑似缺血性梗死。")
    alone = B.study_record("s11", "infarct", 0.5, [_row(0.9, 30.0, host="other_deep_grey", side="left")])
    assert alone["sentence"] == "左侧深部灰质（海马、杏仁核等）存在梗死样异常，体积约 30 mm³。疑似缺血性梗死。"
    same = B.study_record("s10", "infarct", 0.5, big + [_row(0.9, 70.0, host="cortex", side="left")])
    assert same["sentence"].endswith("；另有 1 处同类异常。疑似缺血性梗死。")          # nothing new to name


def test_the_places_of_the_named_lesions_are_compared_in_their_short_form():
    # the fifth-named lesion is deep grey matter; a smaller deep-grey lesion on the same side adds no new place, so the
    # count must come without a bracket (the long name "左侧深部灰质（海马、杏仁核等）" must not be compared with the short one)
    big = [_row(0.6, 1000.0 - i, host="cortex", side="left") for i in range(4)] + [_row(0.6, 900.0, host="other_deep_grey", side="left")]
    rec = B.study_record("s12", "infarct", 0.5, big + [_row(0.9, 30.0, host="other_deep_grey", side="left")])
    assert rec["sentence"].endswith("；另有 1 处同类异常。疑似缺血性梗死。") and "还见于" not in rec["sentence"]
