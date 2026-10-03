# tests/test_brain_anatomy_prepare.py
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from anatobind.anatomy.labels import IGNORE, LABELS_JSON


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/brain_anatomy_prepare.py"
    spec = importlib.util.spec_from_file_location("brain_anatomy_prepare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _brain(shape=(50, 56, 60)):
    x, y, z = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    rad = ((x - 24.5) / 20) ** 2 + ((y - 27.5) / 24) ** 2 + ((z - 24) / 22) ** 2
    seg = np.zeros(shape, np.int16)
    seg[(rad <= 1) & (x < 24.5)] = 2
    seg[(rad <= 1) & (x >= 24.5)] = 41
    seg[rad <= 0.05] = 4
    img = np.where(seg > 0, 300.0, 0.0).astype(np.float32)
    img[seg == 4] = 80.0
    return img, seg


def _write(path, data, zooms=(1.0, 1.0, 1.0)):
    aff = np.diag(list(zooms) + [1.0])
    nib.save(nib.Nifti1Image(data, aff), str(path))
    return path


def test_load_1mm_ras_resamples_non_isotropic_files(tmp_path):
    mod = _load()
    seg = np.zeros((10, 12, 6), np.int16)
    seg[2:8, 2:10, 1:5] = 41
    p = _write(tmp_path / "seg.nii.gz", seg, (1.0, 1.0, 2.0))
    out, zooms = mod.load_1mm_ras(p, 0)
    assert out.shape == (10, 12, 12) and zooms == (1.0, 1.0, 2.0) and set(np.unique(out)) == {0, 41}
    img, _ = mod.load_1mm_ras(_write(tmp_path / "img.nii.gz", seg.astype(np.float32) * 2, (1.0, 1.0, 2.0)), 1)
    assert img.shape == (10, 12, 12) and img.dtype == np.float32 and img.max() == pytest.approx(82.0, abs=1e-3)
    same, _ = mod.load_1mm_ras(_write(tmp_path / "iso.nii.gz", seg), 0)
    assert np.array_equal(same, seg)


def test_simulate_case_writes_stacks_labels_and_params_deterministically(tmp_path):
    mod = _load()
    img, seg = _brain()
    lesion = np.zeros(seg.shape, np.int16)
    lesion[20:30, 20:30, 20:30] = 2                                # value 2 of a (1, 2, 4) tumour mask
    rec = {"source": "pdgm", "patient": "P1", "flair": str(_write(tmp_path / "f.nii.gz", img)),
           "anatomy": str(_write(tmp_path / "a.nii.gz", seg)), "lesion": str(_write(tmp_path / "l.nii.gz", lesion)), "lesion_values": [1, 2, 4]}
    out = tmp_path / "sim"
    out.mkdir()
    rows = mod.simulate_case(("CASE", rec, out, [0, 1]))
    assert [r["sample"] for r in rows] == ["CASE_s0", "CASE_s1"] and rows[0]["source"] == "pdgm" and rows[0]["ignore_voxels"] > 0
    stack, labels = nib.load(str(out / "CASE_s0_0000.nii.gz")), nib.load(str(out / "CASE_s0.nii.gz"))
    assert stack.shape == labels.shape and stack.shape[2] in (14, 16) and tuple(stack.shape[:2]) == tuple(rows[0]["matrix"])
    assert np.allclose(np.diag(stack.affine)[:3], [-rows[0]["inplane_mm"], -rows[0]["inplane_mm"], 5.0])
    lab = np.asarray(labels.dataobj)
    assert lab.dtype == np.uint8 and lab.max() <= IGNORE and (lab == IGNORE).any()
    assert json.loads((out / "CASE_s1.json").read_text())["sample"] == "CASE_s1"
    again = tmp_path / "again"
    again.mkdir()
    rows2 = mod.simulate_case(("CASE", rec, again, [0]))
    assert np.array_equal(np.asarray(nib.load(str(again / "CASE_s0.nii.gz")).dataobj), lab) and rows2[0]["empty_top"] == rows[0]["empty_top"]
    assert mod.sample_seed("CASE", 0) != mod.sample_seed("CASE", 1)
    bad = dict(rec, lesion=str(_write(tmp_path / "bad.nii.gz", np.full(seg.shape, 7, np.int16))))
    with pytest.raises(ValueError, match="unexpected values"):
        mod.simulate_case(("CASE", bad, again, [1]))
    # a teacher map or a lesion mask on another grid is refused before anything is resampled
    shifted = tmp_path / "shifted.nii.gz"
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    aff[0, 3] = 2.0
    nib.save(nib.Nifti1Image(seg, aff), str(shifted))
    with pytest.raises(ValueError, match="not on the grid"):
        mod.simulate_case(("CASE", dict(rec, anatomy=str(shifted)), again, [2]))
    with pytest.raises(ValueError, match="not on the grid"):
        mod.simulate_case(("CASE", dict(rec, lesion=str(_write(tmp_path / "small.nii.gz", lesion[:-1]))), again, [2]))


def test_dataset907_links_training_samples_and_test_images_and_refuses_a_rebuild(tmp_path):
    mod = _load()
    work = tmp_path / "work"
    for split, samples in (("train", ["A_s0", "A_s1"]), ("test", ["B_s0"])):
        d = work / "sim" / split
        d.mkdir(parents=True)
        for s in samples:
            _write(d / f"{s}_0000.nii.gz", np.zeros((4, 4, 2), np.float32))
            _write(d / f"{s}.nii.gz", np.zeros((4, 4, 2), np.uint8))
    (work / "cases.json").write_text(json.dumps({"A": {"split": "train"}, "B": {"split": "test"}}))
    rows = [{"sample": "A_s0", "case": "A", "source": "pdgm", "patient": "pA"}, {"sample": "A_s1", "case": "A", "source": "pdgm", "patient": "pA"},
            {"sample": "B_s0", "case": "B", "source": "sibbms", "patient": "pB"}]
    (work / "sim" / "manifest.json").write_text(json.dumps(rows))
    base = mod.stage_dataset907(tmp_path / "raw", work)
    meta = json.loads((base / "dataset.json").read_text())
    assert meta["labels"] == LABELS_JSON and meta["labels"]["ignore"] == 15 and meta["numTraining"] == 2 and meta["channel_names"] == {"0": "FLAIR"}
    assert (base / "imagesTr" / "A_s0_0000.nii.gz").is_symlink() and (base / "labelsTr" / "A_s1.nii.gz").is_symlink()
    assert (base / "imagesTs" / "B_s0_0000.nii.gz").is_symlink() and not (base / "labelsTr" / "B_s0.nii.gz").exists()
    assert json.loads((base / "cases.json").read_text())["B_s0"] == {"case": "B", "patient": "pB", "source": "sibbms", "split": "test"}
    with pytest.raises(FileExistsError):
        mod.stage_dataset907(tmp_path / "raw", work)


def test_dataset908_places_usable_stacks_by_patient_and_lists_the_unusable(tmp_path):
    mod = _load()
    seg_dir = tmp_path / "segs"
    seg_dir.mkdir()
    full = np.zeros((40, 40, 16), np.int16)
    for k in range(13):
        r = 6 + k if k < 6 else 18 - k
        x, y = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
        full[:, :, k][(x - 20) ** 2 + (y - 20) ** 2 <= r ** 2] = 2
    stems = [f"file_brain_AXFLAIR_200_{i}" for i in range(10)]
    for i, s in enumerate(stems):
        data = full if i != 3 else (full * 0).astype(np.int16)        # stem 3: SynthSeg failed (empty map)
        _write(seg_dir / f"{s}_seg.nii.gz", data, (5.0, 5.0, 5.0))      # 5 mm pixels: areas large enough, volume 300+ mL
    patient = {s: f"p{i // 2}" for i, s in enumerate(stems)}            # five patients of two stacks each

    def convert(stem, out):
        _write(out, np.ones((40, 40, 16), np.float32), (5.0, 5.0, 5.0))

    base = mod.stage_dataset908(tmp_path / "raw", tmp_path / "work", seg_dir=seg_dir, convert=convert, patient=patient)
    info, excluded = json.loads((base / "cases.json").read_text()), json.loads((base / "excluded.json").read_text())
    assert set(excluded) == {stems[3]} and stems[3] not in info and not (base / "imagesTr" / f"{stems[3]}_0000.nii.gz").exists()
    test_patients = {v["patient"] for v in info.values() if v["split"] == "test"}
    assert len(test_patients) == 1 and all((v["patient"] in test_patients) == (v["split"] == "test") for v in info.values())
    for s, v in info.items():
        folder = "imagesTr" if v["split"] == "train" else "imagesTs"
        assert (base / folder / f"{s}_0000.nii.gz").exists() and (base / "labelsTr" / f"{s}.nii.gz").exists() == (v["split"] == "train")
        assert v["supervised"] == [2, 11]
    meta = json.loads((base / "dataset.json").read_text())
    assert meta["labels"] == {"background": 0, "brain": 1, "ignore": 2} and meta["numTraining"] == sum(1 for v in info.values() if v["split"] == "train")
    lab = np.asarray(nib.load(str(base / "labelsTr" / f"{[s for s, v in info.items() if v['split'] == 'train'][0]}.nii.gz")).dataobj)
    assert (lab[:, :, 0] == 2).all() and set(np.unique(lab[:, :, 5])) == {0, 1}
    # an RSS image that is not on its SynthSeg map's grid stops the stage (same shape, shifted affine)
    def shifted(stem, out):
        aff = np.diag([5.0, 5.0, 5.0, 1.0])
        aff[1, 3] = 10.0
        nib.save(nib.Nifti1Image(np.ones((40, 40, 16), np.float32), aff), str(out))

    with pytest.raises(ValueError, match="not on the grid of its SynthSeg map"):
        mod.stage_dataset908(tmp_path / "raw_shifted", tmp_path / "work", seg_dir=seg_dir, convert=shifted, patient=patient)


def test_patient_splits_and_split_files(tmp_path):
    mod = _load()
    info = {f"c{i}": {"patient": f"p{i // 3}"} for i in range(30)}       # ten patients of three cases
    fold = mod.patient_splits(info, k=5, seed=0)
    assert sorted(set(fold.values())) == [0, 1, 2, 3, 4] and all(fold[f"c{i}"] == fold[f"c{i - i % 3}"] for i in range(30))
    d = tmp_path / "pre" / "Dataset907_BrainAnatomyFLAIR"
    d.mkdir(parents=True)
    p, sizes = mod.write_splits(tmp_path / "pre", "Dataset907_BrainAnatomyFLAIR", fold)
    splits = json.loads(p.read_text())
    assert len(splits) == 5 and sum(sizes) == 30 and all(len(set(s["train"]) & set(s["val"])) == 0 for s in splits)
    with pytest.raises(FileExistsError):
        mod.write_splits(tmp_path / "pre", "Dataset907_BrainAnatomyFLAIR", fold)
    with pytest.raises(FileNotFoundError):
        mod.write_splits(tmp_path / "pre", "Dataset908_FastMRIBrainOutline", fold)
    # the stage writes no split file unless both datasets are preprocessed
    raw, pre = tmp_path / "raw_s", tmp_path / "pre_s"
    for d in ("Dataset907_BrainAnatomyFLAIR", "Dataset908_FastMRIBrainOutline"):
        (raw / d).mkdir(parents=True)
        (raw / d / "cases.json").write_text(json.dumps({f"c{i}": {"patient": f"p{i}", "split": "train" if i else "test"} for i in range(11)}))
    (pre / "Dataset907_BrainAnatomyFLAIR").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Dataset908"):
        mod.stage_splits(raw, pre)
    assert not (pre / "Dataset907_BrainAnatomyFLAIR" / "splits_final.json").exists()
    (pre / "Dataset908_FastMRIBrainOutline").mkdir()
    mod.stage_splits(raw, pre)
    s907 = json.loads((pre / "Dataset907_BrainAnatomyFLAIR" / "splits_final.json").read_text())
    assert sum(len(f["val"]) for f in s907) == 10 and "c0" not in {c for f in s907 for c in f["val"]}        # the test case is in no fold
