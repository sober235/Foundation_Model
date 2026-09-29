# tests/test_brain_disease_dataset.py
import json
import os

import nibabel as nib
import numpy as np
import pytest

from anatobind.nnunet.brain_disease import (
    DISEASES, anatomy_path, binary_label, build_raw, case_folds, channel_path, check_same_grid, fold_dir, label_path,
    patient_of, write_binary_label, write_splits,
)

AFF = np.diag([0.86, 0.86, 1.5, 1.0])


def _nii(path, data, affine=AFF):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def test_three_datasets_have_distinct_ids_and_native_channels():
    assert {k: (v["id"], v["channels"]) for k, v in DISEASES.items()} == {
        "glioma": (904, ("T1", "T1c", "T2", "FLAIR")), "metastasis": (905, ("T1pre", "T1post", "FLAIR")),
        "infarct": (906, ("DWI", "ADC"))}
    assert DISEASES["glioma"]["label_values"] == (1, 2, 4)


def test_patient_of_groups_repeat_scans_and_follow_ups():
    assert patient_of("metastasis", "100202A") == patient_of("metastasis", "100202C") == "100202"
    assert patient_of("glioma", "UCSF-PDGM-0391_FU016d") == patient_of("glioma", "UCSF-PDGM-0391") == "UCSF-PDGM-0391"
    assert patient_of("infarct", "sub-strokecase0007") == "sub-strokecase0007"
    for disease, bad in (("metastasis", "100202"), ("glioma", "PDGM-1"), ("infarct", "strokecase1")):
        with pytest.raises(ValueError, match=bad):
            patient_of(disease, bad)


def test_source_paths_follow_each_dataset_layout(tmp_path):
    assert channel_path("glioma", "UCSF-PDGM-0004", "T1c", tmp_path) == (
        tmp_path / "UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5/UCSF-PDGM-0004_nifti/UCSF-PDGM-0004_T1c.nii.gz")
    assert label_path("glioma", "UCSF-PDGM-0004", tmp_path).name == "UCSF-PDGM-0004_tumor_segmentation.nii.gz"
    assert channel_path("metastasis", "100101A", "T1post", tmp_path) == (
        tmp_path / "UCSF-BMSR_cbb/UCSF-BMSR/UCSF_BrainMetastases_TRAIN/100101A/100101A_T1post.nii.gz")
    assert channel_path("infarct", "sub-strokecase0001", "ADC", tmp_path) == (
        tmp_path / "ISLES_ltr/ISLES-2022/sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_adc.nii.gz")
    assert label_path("infarct", "sub-strokecase0001", tmp_path) == (
        tmp_path / "ISLES_ltr/ISLES-2022/derivatives/sub-strokecase0001/ses-0001/sub-strokecase0001_ses-0001_msk.nii.gz")
    assert anatomy_path("metastasis", "100101A", tmp_path) == tmp_path / "derived/synthseg/bmsr/seg_native/100101A_T1pre_seg.nii.gz"
    assert anatomy_path("infarct", "sub-strokecase0001", tmp_path).name == "sub-strokecase0001_ses-0001_dwi_seg.nii.gz"
    assert anatomy_path("glioma", "UCSF-PDGM-0391_FU016d", tmp_path).name == "UCSF-PDGM-0391_FU016d_T1_seg.nii.gz"


def test_binary_label_merges_the_allowed_values_and_refuses_others():
    a = np.array([[0, 1, 2], [4, 0, 1]])
    assert binary_label(a, (1, 2, 4), "c").tolist() == [[0, 1, 1], [1, 0, 1]]
    assert binary_label(a, (1, 2, 4), "c").dtype == np.uint8
    with pytest.raises(ValueError, match=r"c: label holds unexpected values \[2, 4\]"):
        binary_label(a, (1,), "c")


def test_written_label_keeps_the_header_geometry(tmp_path):
    src = _nii(tmp_path / "lab.nii.gz", np.array([[[0, 2], [4, 0]]], np.int16))
    n, vox = write_binary_label(src, tmp_path / "out.nii.gz", (1, 2, 4), "c")
    out = nib.load(str(tmp_path / "out.nii.gz"))
    assert n == 2 and vox == pytest.approx(0.86 * 0.86 * 1.5)
    assert np.allclose(out.affine, AFF) and out.get_data_dtype() == np.uint8
    assert np.asarray(out.dataobj).tolist() == [[[0, 1], [1, 0]]]
    check_same_grid("c", [src], tmp_path / "out.nii.gz")


def test_grid_check_names_the_case_and_the_channel(tmp_path):
    lab = _nii(tmp_path / "lab.nii.gz", np.zeros((4, 5, 6), np.uint8))
    good = _nii(tmp_path / "a.nii.gz", np.zeros((4, 5, 6), np.float32))
    shape = _nii(tmp_path / "b.nii.gz", np.zeros((4, 5, 7), np.float32))
    spacing = _nii(tmp_path / "s.nii.gz", np.zeros((4, 5, 6), np.float32), np.diag([0.86, 0.86, 1.6, 1.0]))
    check_same_grid("c", [good], lab)
    with pytest.raises(ValueError, match="c: b.nii.gz is not on the label's grid"):
        check_same_grid("c", [good, shape], lab)
    with pytest.raises(ValueError, match="c: s.nii.gz"):
        check_same_grid("c", [spacing], lab)
    with pytest.raises(FileNotFoundError, match="c: missing"):
        check_same_grid("c", [tmp_path / "nope.nii.gz"], lab)


def _sources(root, cases):
    out = {}
    for c in cases:
        lab = np.zeros((6, 6, 4), np.int16)
        lab[1:3, 1:3, 1] = 1
        out[c] = {"channels": [_nii(root / c / f"{c}_{k}.nii.gz", np.ones((6, 6, 4), np.float32)) for k in ("T1pre", "T1post", "FLAIR")],
                  "label": _nii(root / c / f"{c}_seg.nii.gz", lab)}
    return out


def test_build_raw_links_images_writes_labels_and_refuses_to_rebuild(tmp_path):
    srcs = _sources(tmp_path / "src", ["100101A", "100101B", "100102A"])
    base = build_raw(tmp_path / "raw", "metastasis", srcs, extra={"100101A": {"prior_surgery": True}})
    assert base == tmp_path / "raw" / "Dataset905_BMSRMetastasis"
    link = base / "imagesTr" / "100101B_0001.nii.gz"
    assert link.is_symlink() and os.path.realpath(link) == str(srcs["100101B"]["channels"][1].resolve())
    assert np.asarray(nib.load(str(base / "labelsTr" / "100102A.nii.gz")).dataobj).sum() == 4
    meta = json.loads((base / "dataset.json").read_text())
    assert meta == {"channel_names": {"0": "T1pre", "1": "T1post", "2": "FLAIR"}, "labels": {"background": 0, "metastasis": 1},
                    "numTraining": 3, "file_ending": ".nii.gz"}
    info = json.loads((base / "cases.json").read_text())
    assert info["100101A"] == {"patient": "100101", "voxel_mm3": pytest.approx(0.86 * 0.86 * 1.5), "n_label_voxels": 4,
                               "prior_surgery": True}
    assert "prior_surgery" not in info["100102A"]
    with pytest.raises(FileExistsError):
        build_raw(tmp_path / "raw", "metastasis", srcs)


def test_build_raw_checks_every_case_before_writing_anything(tmp_path):
    srcs = _sources(tmp_path / "src", ["100101A", "100102A"])
    _nii(srcs["100102A"]["channels"][2], np.ones((6, 6, 5), np.float32))
    with pytest.raises(ValueError, match="100102A: 100102A_FLAIR.nii.gz"):
        build_raw(tmp_path / "raw", "metastasis", srcs)
    assert not (tmp_path / "raw").exists()


def test_folds_keep_a_patient_in_one_fold_and_cover_all_cases(tmp_path):
    cases = [f"{100100 + i}{s}" for i in range(12) for s in ("A", "B")]
    cf = case_folds("metastasis", cases)
    assert all(cf[f"{100100 + i}A"] == cf[f"{100100 + i}B"] for i in range(12))
    assert sorted(set(cf.values())) == [0, 1, 2, 3, 4] and cf == case_folds("metastasis", list(reversed(cases)))
    with pytest.raises(FileNotFoundError, match="plan_and_preprocess"):
        write_splits(tmp_path, "metastasis", cf)
    (tmp_path / "Dataset905_BMSRMetastasis").mkdir()
    p = write_splits(tmp_path, "metastasis", cf)
    splits = json.loads(p.read_text())
    assert len(splits) == 5 and sorted(c for s in splits for c in s["val"]) == sorted(cases)
    assert all(set(s["train"]).isdisjoint(s["val"]) and len(s["train"]) + len(s["val"]) == 24 for s in splits)
    with pytest.raises(FileExistsError):
        write_splits(tmp_path, "metastasis", cf)


def test_fold_dir_follows_the_nnunet_layout(tmp_path):
    assert fold_dir(tmp_path, "glioma", 3) == tmp_path / "Dataset904_PDGMGlioma" / "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres" / "fold_3"
    assert fold_dir(tmp_path, "infarct", 0, "nnUNetTrainer_5epochs").parent.name == "nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres"
