# tests/test_aur_scripts_t12.py
"""The evaluation and inference entry points (SSL-first plan T12; spec §8): the model rebuilt from an export, the
outputs written back in the input's orientation, the structured record and its sentences, the reference jobs."""
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

import anatobind.aur.infer as I
import anatobind.aur.train as T
from anatobind.aur.labels import HOST_NAMES, NO_HOST
from anatobind.aur.model import AnatoBindBrain

TINY = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4), d_model=16, n_events=4,
            mask_dim=4, pixel_dim=8, dec_layers=1, dec_heads=2, rel_layers=1, rel_heads=2)


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _export(tmp_path, stage="III"):
    torch.manual_seed(0)
    model = AnatoBindBrain(**TINY, use_checkpoint=False)
    meta = {"stage": stage, "config": {"model": TINY, "crop": [8, 16, 16]}, "code_sha": "test", "step": 1, "seen_crops": 2}
    return T.export_model(tmp_path / f"aur_stage{stage}.pt", model, meta)


def _lps_volume(tmp_path, shape=(40, 36, 12)):
    """A volume stored LPS (x and y flipped relative to RAS) with a bright blob."""
    rng = np.random.default_rng(0)
    arr = rng.normal(0, 5, shape).astype(np.float32)
    arr[8:32, 6:30, 2:10] += 300.0
    aff = np.diag([-1.0, -1.0, 2.0, 1.0])
    aff[0, 3], aff[1, 3] = shape[0] - 1, shape[1] - 1
    p = tmp_path / "in_lps.nii.gz"
    nib.save(nib.Nifti1Image(arr, aff), str(p))
    assert "".join(nib.aff2axcodes(aff)) == "LPS"
    return p


def test_model_from_export_rebuilds_the_configured_model(tmp_path):
    path = _export(tmp_path)
    model, meta = I.model_from_export(path, torch.device("cpu"))
    assert isinstance(model, AnatoBindBrain) and model.cfg["embed"] == 32 and model.events.M == 4 and meta["stage"] == "III"
    assert not model.training
    ssl = tmp_path / "ssl.pt"
    torch.save({"backbone_state_dict": {}, "meta": {"stage": "I"}}, ssl)
    with pytest.raises(ValueError, match="Stage II / III"):
        I.model_from_export(ssl, torch.device("cpu"))


def test_to_original_orientation_round_trips():
    rng = np.random.default_rng(1)
    for codes in ("LPS", "LAS", "RAS", "PIR"):
        aff = nib.orientations.inv_ornt_aff(nib.orientations.axcodes2ornt(codes), (5, 6, 7))
        arr = rng.integers(0, 9, (5, 6, 7)).astype(np.int16)
        img = nib.Nifti1Image(arr, aff)
        canon = np.asarray(nib.as_closest_canonical(img).dataobj)
        back = I.to_original(canon, img.affine)
        assert back.shape == arr.shape and np.array_equal(back, arr), codes


def test_record_and_sentences_follow_the_spec():
    left_wm, brainstem = HOST_NAMES.index("white_matter_left"), HOST_NAMES.index("brainstem")
    rows = [{"instance": 1, "box": [0, 0, 0, 2, 2, 2], "n_voxels": 8, "volume_mm3": 12000.0, "score": 0.9},
            {"instance": 2, "box": [4, 4, 4, 6, 6, 6], "n_voxels": 8, "volume_mm3": 800.0, "score": 0.8},
            {"instance": 3, "box": [8, 8, 8, 9, 9, 9], "n_voxels": 1, "volume_mm3": 40.0, "score": 0.7}]
    probs = np.zeros(NO_HOST + 1, np.float32)
    bound = [{"instance": 1, "host": left_wm, "host_probs": probs, "query": 0, "query_iou": 0.5, "in_crop_share": 1.0},
             {"instance": 2, "host": brainstem, "host_probs": probs, "query": 1, "query_iou": 0.4, "in_crop_share": 1.0},
             {"instance": 3, "host": NO_HOST, "host_probs": probs, "query": 2, "query_iou": 0.1, "in_crop_share": 1.0}]
    lesions = I.lesion_records(rows, bound)
    assert [l["host"] for l in lesions] == ["white_matter_left", "brainstem", "no_host"] and lesions[0]["host_side"] == "left"
    assert lesions[1]["host_side"] is None and lesions[0]["host_tissue"] == "white_matter" and len(lesions[2]["host_probs"]) == NO_HOST + 1
    s = I.sentences(lesions, max_lesions=2)
    assert s == ["左侧大脑白质存在异常，体积约 12 mL", "脑干存在异常，体积约 800 mm³", "另有 1 处异常未逐一列出"]
    assert I.sentences(lesions)[-1] == "未能定位的区域存在异常，体积约 40 mm³" and I.sentences([]) == ["未见异常"]


def test_infer_script_writes_anatomy_lesions_and_record_in_the_input_orientation(tmp_path):
    export = _export(tmp_path)
    nifti = _lps_volume(tmp_path)
    out = tmp_path / "pred"
    mod = _load("infer_anatobind_brain")
    rc = mod.main(["--nifti", str(nifti), "--checkpoint", str(export), "--out", str(out), "--crop", "8", "16", "16", "--cpu", "--seq", "FLAIR"])
    assert rc == 0
    orig = nib.load(str(nifti))
    anat, les = nib.load(str(out / "anatomy.nii.gz")), nib.load(str(out / "lesions.nii.gz"))
    assert anat.shape == orig.shape == les.shape and np.allclose(anat.affine, orig.affine) and np.allclose(les.affine, orig.affine)
    assert np.asarray(anat.dataobj).dtype == np.int16 and np.asarray(les.dataobj).dtype == np.int32
    record = json.loads((out / "record.json").read_text())
    assert record["sequence"] == "FLAIR" and record["sequence_source"] == "given" and record["evidence"] == "NOT_EVIDENCE"
    assert record["anatomy_source"] == "AnatoBind A head (NOT_EVIDENCE)" and record["stage"] == "III" and record["n_lesions"] == len(record["lesions"])
    assert record["operating_point"]["instance_threshold"] == 0.3 and record["box_frame"]["box_input"].startswith("lesions.nii.gz")
    written = np.asarray(les.dataobj)
    for l in record["lesions"]:
        b = l["box_input"]
        assert b is not None and (written[b[0]:b[3], b[1]:b[4], b[2]:b[5]] == l["instance"]).any()
    assert isinstance(record["sentences"], list) and record["sentences"] and {"input", "checkpoint", "seq_probs", "n_windows"} <= set(record)
    for l in record["lesions"]:
        assert {"instance", "host", "host_side", "host_tissue", "host_probs", "volume_mm3", "box", "score"} <= set(l)
    # the canonical prediction and the written map agree voxel for voxel after reorientation
    canon = np.asarray(nib.as_closest_canonical(anat).dataobj)
    direct = I.predict_volume(*I.model_from_export(export, torch.device("cpu"))[:1], *I.load_input(nifti)[:2], crop=(8, 16, 16), device=torch.device("cpu"))
    from anatobind.aur.labels import entity_to_synthseg
    assert np.array_equal(canon, entity_to_synthseg(direct["entity"]).transpose(2, 1, 0))
    with pytest.raises(FileExistsError):
        mod.main(["--nifti", str(nifti), "--checkpoint", str(export), "--out", str(out), "--cpu"])
    guessed = tmp_path / "pred2"
    mod.main(["--nifti", str(nifti), "--checkpoint", str(export), "--out", str(guessed), "--crop", "8", "16", "16", "--cpu"])
    rec2 = json.loads((guessed / "record.json").read_text())
    assert rec2["sequence_source"] == "predicted" and rec2["sequence"] in ("T1", "T1c", "T2", "FLAIR", "DWI", "ADC")


def test_eval_script_contract_and_reference_jobs(tmp_path, monkeypatch):
    mod = _load("aur_eval")
    a = mod.parser().parse_args(["--checkpoint", "c.pt", "--samples", "s.json", "--out", "o", "--split", "test", "--cpu", "--u-threshold", "pdgm=0.4"])
    assert a.split == "test" and a.limit is None and a.sources is None and a.no_reference is False and a.u_threshold == ["pdgm=0.4"]
    with pytest.raises(SystemExit):
        mod.parser().parse_args(["--checkpoint", "c.pt", "--samples", "s.json", "--out", "o"])
    import anatobind.aur.eval as E
    calls = []
    monkeypatch.setattr(E, "case_scan", lambda job: calls.append(job) or {"case": job[0], "gt": [], "dets": []})
    root = tmp_path / "nnunet"
    (root / "preprocessed/Dataset905_BMSRMetastasis").mkdir(parents=True)
    (root / "preprocessed/Dataset905_BMSRMetastasis/splits_final.json").write_text(json.dumps([{"train": [], "val": ["100102A"]}, {"train": [], "val": ["100104A"]}]))
    label_dir = root / "raw/Dataset905_BMSRMetastasis/labelsTr"
    label_dir.mkdir(parents=True)
    for case in ("100102A", "100104A"):
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.diag([1.0, 1.0, 2.0, 1.0])), str(label_dir / f"{case}.nii.gz"))
    scans = E.reference_scans("bmsr", ["100104A", "100102A"], nnunet_root=root, anatomy_of=lambda disease, case: tmp_path / f"{case}_seg.nii.gz")
    assert [s["case"] for s in scans] == ["100102A", "100104A"] and len(calls) == 2
    case, disease, label, nii, npz, anatomy, voxel_mm3 = calls[0]
    assert disease == "metastasis" and label == label_dir / "100102A.nii.gz" and nii.name == "100102A.nii.gz" and npz.name == "100102A.npz"
    assert "fold_0" in str(nii) and "fold_1" in str([c[3] for c in calls if c[0] == "100104A"][0]) and voxel_mm3 == pytest.approx(2.0)
    with pytest.raises(KeyError):
        E.reference_scans("sibbms", ["x"], nnunet_root=root, anatomy_of=lambda d, c: tmp_path)


def test_eval_script_runs_end_to_end_without_the_reference(tmp_path):
    export = _export(tmp_path)
    rows = []
    for name, source, lesion in (("t0", "pdgm", True), ("t1", "pdgm", False), ("t2", "bmsr", True)):
        shape, spacing = (40, 36, 12), (1.0, 1.0, 2.0)
        affine = np.diag(list(spacing) + [1.0])
        seg = np.zeros(shape, np.int16)
        seg[:20] = 2
        seg[20:] = 41
        img = np.where(seg > 0, 300.0, 0.0).astype(np.float32)
        les = np.zeros(shape, np.uint8)
        if lesion:
            les[4:7, 4:7, 3:5] = 1
        paths = {}
        for key, arr in (("image", img), ("anatomy", seg), ("lesion", les)):
            p = tmp_path / f"{name}_{key}.nii.gz"
            nib.save(nib.Nifti1Image(arr, affine), str(p))
            paths[key] = str(p)
        rows.append({"case": name, "source": source, "patient": name, "sequence": "FLAIR", "source_sequence": "FLAIR", "image": paths["image"],
                     "anatomy": paths["anatomy"], "lesion": paths["lesion"] if lesion else None, "u_supervised": lesion, "u_values": [1] if lesion else [],
                     "a_ignore_values": [1] if lesion else [], "split": "test", "a_supervised": True, "r_supervised": True})
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps(rows))
    mod = _load("aur_eval")
    sel = mod.select_rows(rows, "test", None, ["pdgm"], 1)
    assert [r["case"] for r in sel] == ["t0"]
    out = tmp_path / "eval"
    rc = mod.main(["--checkpoint", str(export), "--samples", str(samples), "--out", str(out), "--cpu", "--no-reference", "--crop", "8", "16", "16", "--no-e2e"])
    assert rc == 0
    agg = json.loads((out / "aggregate.json").read_text())
    assert agg["n_results"] == 3 and agg["run"]["n_rows"] == 3 and agg["u"]["gate"]["sources_judged"] == 0 and "pdgm/FLAIR" in agg["u"]["per_source_sequence"]
    assert (out / "REPORT.md").is_file() and (out / "reference.json").read_text().strip() == "{}" and (out / "level_r_sheet.csv").is_file()
