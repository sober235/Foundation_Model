# Brain Multi-Disease Detection (S7, stage A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one nnU-Net detector per brain disease (glioma on UCSF-PDGM, metastasis on UCSF-BMSR, infarct on ISLES 2022) on its native sequences, judge each against the D1-style gate on five-fold out-of-fold predictions, bind every detected lesion to a host structure and side with the existing SynthSeg pseudo-labels, and write one structured record and sentence per study.

**Architecture:** Each source dataset becomes an nnU-Net v2 dataset (904, 905, 906): images are symlinks to the read-only originals, labels are rewritten as binary maps that keep the original header, folds are by patient. A small queue starts the fifteen trainings on whatever GPUs are idle. Predicted masks are decoded into scored 26-connected components with a 10 mm³ volume floor; the existing detection metrics score them, with a new optional ignore flag for ground-truth fragments under the floor. Binding is a lookup of the lesion mask in the SynthSeg label map (seven host classes, patient left / right).

**Tech Stack:** Python 3.11, numpy, scipy.ndimage, nibabel, SimpleITK, nnU-Net v2 2.8.0 (`nnUNetv2_*` in env nvgen), pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-brain-multidisease-design.md` (decisions M1–M14). Read it before any task.

**Plan dry run (2026-09-29, throwaway):** every code and test file of Tasks 1–10 was written verbatim into a scratch export of this branch (22d5127) and run: the 84 new tests pass, the full suite gives 831 passed, 1 skipped. The dataset module was run read-only against the real sources (501 / 495, 461 / 314, 250 / 250 scans / patients; every channel, label and SynthSeg file present). Mini datasets of six real cases per disease passed `nnUNetv2_plan_and_preprocess --verify_dataset_integrity` (symlinked images, rewritten labels), and the queue ran 5-epoch trainings on them on the two cards that were idle (the third job waited for a card, as designed). Epoch times on an A800: glioma 67–73 s, metastasis 62–67 s, infarct 23 s (the first epoch of each is slower), so 250 epochs come to about 4.7 h, 4.5 h and 1.6 h plus validation. The evaluation script then read those real nnU-Net outputs end to end (axis order, grids, binding, records). With such undertrained models the infarct reading had no operating point at all, which is why the script states that case plainly (Task 9). Implementers still run every step themselves; these numbers are a hint, not evidence.

## Global Constraints

- Tests: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest <paths> -q -p no:cacheprovider`; tests never read `/data2`. Baseline before Task 1: 747 passed, 1 skipped.
- nnU-Net env: `source scripts/nnunet_env.sh` (raw / preprocessed / results under `/data2/congcong/data/FM_data/derived/nnunet/`). The Bash tool runs zsh: wrap anything that sources a script in `bash -c '…'`.
- Data is read only from `/data2/congcong/data/FM_data`; the original datasets are never modified. New outputs go under `/data2/congcong/data/FM_data/derived/` (`nnunet/` for datasets and models, `brain_disease/` for the subject table, records and cross runs).
- Datasets: `Dataset904_PDGMGlioma` (channels T1, T1c, T2, FLAIR; label `tumor_segmentation > 0`), `Dataset905_BMSRMetastasis` (T1pre, T1post, FLAIR; `seg > 0`), `Dataset906_ISLESInfarct` (DWI, ADC; `msk > 0`). Plain (not bias-corrected) PDGM images; no synthetic T2 and no subtraction image for BMSR.
- Folds by patient: metastasis patient = scan id without its trailing letter; a glioma follow-up (`_FU…`) follows its patient; sorted patients, seed 0, five folds.
- Training: `nnUNetv2_train <id> 3d_fullres <fold> -tr nnUNetTrainer_250epochs --npz`, `nnUNet_n_proc_DA=6`, at most 6 trainings at once, nice 19, CPU threads ≤ 48 in total.
- GPUs (M6, the user's words: "后续整个服务器的算力都要优先该任务使用，但是前提是只能是考虑或者占用空的 GPU"): any idle GPU may be used; a GPU is idle when it has < 1000 MiB in use and no compute process; a card somebody else uses is never touched; nobody else's process is ever signalled. The nnDetection fold 0 training of this project may still run on GPU 3: never disturb it.
- A lesion is a 26-connected component. Components under 10 mm³ are ignored in the ground truth (not in the denominator; a detection matched to one is neither a hit nor a false positive) and dropped from the predictions.
- Gate per disease (M2, M4): box IoU ≥ 0.1 one-to-one, operating point at ≤ 2 false positives per scan, lesion sensitivity ≥ 0.5, judged on all five folds. A fold subset gives an early reading that is never called the gate. Fold 0 sensitivity < 0.3 stops the remaining folds of that disease. A failed gate is reported, never tuned (M5).
- Binding numbers are NOT_EVIDENCE: the anatomy is a SynthSeg pseudo-label.
- Nothing is overwritten or deleted: every output directory or file that exists is refused. Records are evidence: never append monitoring lines to a file that is committed, never edit a committed record afterwards.
- Commits: repository-local author, English messages, no Co-Authored-By or any AI trace (the user's rule overrides the harness reminder). No push, no merge. `logs/` is never added.

## Review Focus

1. A channel that is missing or off the label's grid — expected: the build names the case and the file and writes nothing (Task 1 `test_build_raw_checks_every_case_before_writing_anything`, `test_grid_check_names_the_case_and_the_channel`).
2. Repeat scans of one patient — expected: all of them land in one fold (Task 1 `test_folds_keep_a_patient_in_one_fold_and_cover_all_cases`).
3. A GPU that runs one of our jobs, or a full queue — expected: no second job on that card, never more than the cap (Task 3 `test_launches_take_only_free_idle_gpus_up_to_the_cap`).
4. A detection on a ground-truth fragment under 10 mm³, and a real lesion next to such a fragment — expected: the first is excused, the second still wins its detection (Task 5 `test_ignored_rows_leave_the_denominator_and_excuse_their_detections`, `test_a_real_lesion_wins_a_detection_over_an_ignored_neighbour`).
5. A lesion that lies on a ventricle or outside every host structure — expected: the nearest host in millimetres, or an explicit "no host" (Task 6 `test_a_lesion_on_a_landmark_takes_the_nearest_host`, `test_nearest_uses_millimetres_not_voxels`, `test_no_host_anywhere_and_a_mask_off_the_grid`).

---

### Task 1: Dataset module `anatobind/nnunet/brain_disease.py`

**Files:**
- Create: `anatobind/nnunet/brain_disease.py`
- Test: `tests/test_brain_disease_dataset.py`

**Interfaces:**
- Consumes: `anatobind.nnunet.brain_lesion.assign_normal_folds(patients, k=5, seed=0) -> {patient: fold}`, `make_splits(case_fold, k=5) -> [{"train", "val"}]`.
- Produces:
  - constants `FM`, `TRAINER = "nnUNetTrainer_250epochs"`, `CONFIG = "3d_fullres"`, `DISEASES = {disease: {"id", "name", "type", "channels", "label_values", "impression"}}` with keys `glioma`, `metastasis`, `infarct`
  - `patient_of(disease, case) -> str` (ValueError on an unexpected name)
  - `channel_path(disease, case, channel, root=FM) -> Path`, `label_path(disease, case, root=FM)`, `anatomy_path(disease, case, root=FM)`, `list_cases(disease, root=FM) -> [case]`, `sources(disease, root=FM) -> {case: {"channels": [Path], "label": Path}}`
  - `geometry(path)`, `check_same_grid(case, channel_paths, label)` (FileNotFoundError / ValueError naming the case)
  - `binary_label(values, allowed, case) -> uint8`, `write_binary_label(src, dst, allowed, case) -> (n foreground voxels, voxel mm³)`
  - `build_raw(raw_root, disease, srcs, extra=None) -> Path` (checks every case before writing; refuses an existing dataset; writes `dataset.json`, `cases.json`)
  - `case_folds(disease, cases, k=5, seed=0) -> {case: fold}`, `write_splits(preprocessed_root, disease, case_fold, k=5) -> Path`
  - `fold_dir(results_root, disease, fold, trainer=TRAINER) -> Path`

- [ ] **Step 1: Write the failing test**

````python
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
````

- [ ] **Step 2: Run to verify it fails** — `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_disease_dataset.py -q -p no:cacheprovider` → `ModuleNotFoundError: No module named 'anatobind.nnunet.brain_disease'`.

- [ ] **Step 3: Implement** `anatobind/nnunet/brain_disease.py`:

````python
"""nnU-Net v2 datasets for the brain multi-disease detectors (spec 2026-09-29 §3).

One dataset per disease, each on the native sequences of its source: glioma (UCSF-PDGM), metastasis (UCSF-BMSR),
infarct (ISLES 2022). Images are symlinks to the read-only originals. Labels are rewritten as binary maps that keep
the original label's header, so SimpleITK reads the same geometry for them as for the images. Folds are by patient.
"""
import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from anatobind.nnunet.brain_lesion import assign_normal_folds, make_splits

FM = Path("/data2/congcong/data/FM_data")
TRAINER = "nnUNetTrainer_250epochs"
CONFIG = "3d_fullres"
DISEASES = {
    "glioma": {"id": 904, "name": "Dataset904_PDGMGlioma", "type": "tumor", "channels": ("T1", "T1c", "T2", "FLAIR"),
               "label_values": (1, 2, 4), "impression": "疑似胶质瘤"},
    "metastasis": {"id": 905, "name": "Dataset905_BMSRMetastasis", "type": "metastasis",
                   "channels": ("T1pre", "T1post", "FLAIR"), "label_values": (1,), "impression": "疑似脑转移瘤"},
    "infarct": {"id": 906, "name": "Dataset906_ISLESInfarct", "type": "infarct", "channels": ("DWI", "ADC"),
                "label_values": (1,), "impression": "疑似缺血性梗死"},
}
PATIENT_PATTERN = {"glioma": r"(UCSF-PDGM-\d+)(_FU\d+d)?", "metastasis": r"(\d+)[A-Z]", "infarct": r"(sub-strokecase\d+)"}


def patient_of(disease, case):
    m = re.fullmatch(PATIENT_PATTERN[disease], case)
    if not m:
        raise ValueError(f"unexpected {disease} case name {case!r}")
    return m.group(1)


def channel_path(disease, case, channel, root=FM):
    """Path of one sequence of one case in the source dataset (any sequence the source holds, not only the model's)."""
    root = Path(root)
    if disease == "glioma":
        return root / "UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5" / f"{case}_nifti" / f"{case}_{channel}.nii.gz"
    if disease == "metastasis":
        return root / "UCSF-BMSR_cbb/UCSF-BMSR/UCSF_BrainMetastases_TRAIN" / case / f"{case}_{channel}.nii.gz"
    if disease == "infarct":
        return root / "ISLES_ltr/ISLES-2022" / case / "ses-0001" / "dwi" / f"{case}_ses-0001_{channel.lower()}.nii.gz"
    raise KeyError(disease)


def label_path(disease, case, root=FM):
    root = Path(root)
    if disease == "glioma":
        return channel_path(disease, case, "tumor_segmentation", root)
    if disease == "metastasis":
        return channel_path(disease, case, "seg", root)
    if disease == "infarct":
        return root / "ISLES_ltr/ISLES-2022/derivatives" / case / "ses-0001" / f"{case}_ses-0001_msk.nii.gz"
    raise KeyError(disease)


def anatomy_path(disease, case, root=FM):
    """The SynthSeg pseudo-label of this case (derived/synthseg, native grid)."""
    stem = {"glioma": f"{case}_T1", "metastasis": f"{case}_T1pre", "infarct": f"{case}_ses-0001_dwi"}[disease]
    folder = {"glioma": "pdgm", "metastasis": "bmsr", "infarct": "isles"}[disease]
    return Path(root) / "derived/synthseg" / folder / "seg_native" / f"{stem}_seg.nii.gz"


def list_cases(disease, root=FM):
    root = Path(root)
    if disease == "glioma":
        base = root / "UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5"
        return sorted(d.name[:-len("_nifti")] for d in base.glob("UCSF-PDGM-*_nifti") if d.is_dir())
    if disease == "metastasis":
        base = root / "UCSF-BMSR_cbb/UCSF-BMSR/UCSF_BrainMetastases_TRAIN"
        return sorted(d.name for d in base.iterdir() if d.is_dir())
    if disease == "infarct":
        return sorted(d.name for d in (root / "ISLES_ltr/ISLES-2022").glob("sub-strokecase*") if d.is_dir())
    raise KeyError(disease)


def sources(disease, root=FM):
    return {c: {"channels": [channel_path(disease, c, k, root) for k in DISEASES[disease]["channels"]],
                "label": label_path(disease, c, root)} for c in list_cases(disease, root)}


def geometry(path):
    """Size, spacing, origin and direction as SimpleITK reads them: what nnU-Net's integrity check compares."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return tuple(r.GetSize()), np.array(r.GetSpacing()), np.array(r.GetOrigin()), np.array(r.GetDirection())


def check_same_grid(case, channel_paths, label):
    for p in (*channel_paths, label):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{case}: missing {p}")
    ref = geometry(label)
    for p in channel_paths:
        g = geometry(p)
        if g[0] != ref[0] or not all(np.allclose(a, b) for a, b in zip(g[1:], ref[1:])):
            raise ValueError(f"{case}: {Path(p).name} is not on the label's grid")


def binary_label(values, allowed, case):
    v = np.rint(np.asarray(values)).astype(np.int16)
    extra = sorted(set(np.unique(v).tolist()) - {0, *allowed})
    if extra:
        raise ValueError(f"{case}: label holds unexpected values {extra}")
    return (v > 0).astype(np.uint8)


def write_binary_label(src, dst, allowed, case):
    """Binary copy of a label that keeps its header; returns (foreground voxels, voxel volume in mm3)."""
    lab = nib.load(str(src))
    b = binary_label(lab.dataobj, allowed, case)
    hdr = lab.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image(b, lab.affine, hdr), str(dst))
    return int(b.sum()), float(np.prod(lab.header.get_zooms()[:3]))


def build_raw(raw_root, disease, srcs, extra=None):
    """<raw_root>/<dataset>/{imagesTr (symlinks), labelsTr, dataset.json, cases.json}; never rebuilds a dataset."""
    spec = DISEASES[disease]
    base = Path(raw_root) / spec["name"]
    if base.exists():
        raise FileExistsError(f"{base} exists; the dataset is never rebuilt in place")
    for case, s in srcs.items():
        patient_of(disease, case)
        check_same_grid(case, s["channels"], s["label"])
    (base / "imagesTr").mkdir(parents=True)
    (base / "labelsTr").mkdir()
    info = {}
    for i, (case, s) in enumerate(srcs.items(), start=1):
        for k, p in enumerate(s["channels"]):
            os.symlink(Path(p).resolve(), base / "imagesTr" / f"{case}_{k:04d}.nii.gz")
        n, vox = write_binary_label(s["label"], base / "labelsTr" / f"{case}.nii.gz", spec["label_values"], case)
        info[case] = {"patient": patient_of(disease, case), "voxel_mm3": vox, "n_label_voxels": n, **(extra or {}).get(case, {})}
        print(f"{i}/{len(srcs)} {case} {n} label voxels", flush=True)
    meta = {"channel_names": {str(k): n for k, n in enumerate(spec["channels"])}, "labels": {"background": 0, spec["type"]: 1},
            "numTraining": len(srcs), "file_ending": ".nii.gz"}
    (base / "dataset.json").write_text(json.dumps(meta, indent=1))
    (base / "cases.json").write_text(json.dumps(info, indent=1))
    return base


def case_folds(disease, cases, k=5, seed=0):
    patients = {c: patient_of(disease, c) for c in cases}
    fold_of = assign_normal_folds(patients.values(), k, seed)
    return {c: fold_of[p] for c, p in patients.items()}


def write_splits(preprocessed_root, disease, case_fold, k=5):
    d = Path(preprocessed_root) / DISEASES[disease]["name"]
    if not d.is_dir():
        raise FileNotFoundError(f"{d} missing: run nnUNetv2_plan_and_preprocess first")
    p = d / "splits_final.json"
    if p.exists():
        raise FileExistsError(f"{p} exists")
    splits = make_splits(case_fold, k)
    patients = [{patient_of(disease, c) for c in s["val"]} for s in splits]
    if sorted(c for s in splits for c in s["val"]) != sorted(case_fold) or sum(len(x) for x in patients) != len(set().union(*patients)):
        raise ValueError("folds do not partition the cases by patient")
    p.write_text(json.dumps(splits, indent=1))
    return p


def fold_dir(results_root, disease, fold, trainer=TRAINER):
    return Path(results_root) / DISEASES[disease]["name"] / f"{trainer}__nnUNetPlans__{CONFIG}" / f"fold_{int(fold)}"
````

- [ ] **Step 4: Run** `tests/test_brain_disease_dataset.py tests/test_brain_detector_dataset.py` → all pass (10 new).

- [ ] **Step 5: Commit** — `git add anatobind/nnunet/brain_disease.py tests/test_brain_disease_dataset.py && git commit -m "Brain disease datasets: three nnU-Net datasets on native sequences, symlinked images, binary labels that keep the header, folds by patient"`

---

### Task 2: Prepare script, real build, preprocessing and folds

**Files:**
- Create: `scripts/brain_disease_prepare.py`
- Test: `tests/test_brain_disease_prepare.py`
- Create (from the runs): `docs/verification/2026-09-29/brain_multidisease/build/{bmsr_subjects.txt, glioma_raw.txt, metastasis_raw.txt, infarct_raw.txt, plans.txt, splits.txt}`

**Interfaces:**
- Consumes: Task 1.
- Produces: `surgery_flags(csv_path) -> {case: {"prior_surgery": bool}}`, `EXPECTED`, `BMSR_SUBJECTS`; the three datasets under `/data2/congcong/data/FM_data/derived/nnunet/{raw,preprocessed}/`, each with `cases.json` (raw) and `splits_final.json` (preprocessed); `/data2/congcong/data/FM_data/derived/brain_disease/bmsr_subjects.csv`.

- [ ] **Step 1: Write the failing test**

````python
# tests/test_brain_disease_prepare.py
import importlib.util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/brain_disease_prepare.py"
    spec = importlib.util.spec_from_file_location("brain_disease_prepare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_surgery_flags_read_yes_and_no_and_refuse_anything_else(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("SubjectID,Sex,Prior Craniotomy/Biopsy/Resection\n100101A,Male,No\n100202C,Female,Yes\n")
    assert _load().surgery_flags(p) == {"100101A": {"prior_surgery": False}, "100202C": {"prior_surgery": True}}
    p.write_text("SubjectID,Prior Craniotomy/Biopsy/Resection\n100101A,maybe\n")
    with pytest.raises(ValueError, match="100101A: unexpected surgery value 'maybe'"):
        _load().surgery_flags(p)


def test_expected_counts_match_the_spec():
    assert _load().EXPECTED == {"glioma": (501, 495), "metastasis": (461, 314), "infarct": (250, 250)}
````

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_brain_disease_prepare.py -q -p no:cacheprovider` → FileNotFoundError on the script path.

- [ ] **Step 3: Implement** `scripts/brain_disease_prepare.py`:

````python
#!/usr/bin/env python
# scripts/brain_disease_prepare.py
"""Build one brain disease dataset for nnU-Net (spec 2026-09-29 §3), or write its folds.

  source scripts/nnunet_env.sh
  PYTHONPATH=. nice -n 19 python scripts/brain_disease_prepare.py --disease glioma --stage raw
  nice -n 19 nnUNetv2_plan_and_preprocess -d 904 -c 3d_fullres --verify_dataset_integrity -np 4 -npfp 4
  PYTHONPATH=. python scripts/brain_disease_prepare.py --disease glioma --stage splits
"""
import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.nnunet.brain_disease import DISEASES, FM, build_raw, case_folds, sources, write_splits  # noqa: E402

BMSR_SUBJECTS = FM / "derived/brain_disease/bmsr_subjects.csv"
EXPECTED = {"glioma": (501, 495), "metastasis": (461, 314), "infarct": (250, 250)}     # (scans, patients)


def surgery_flags(path):
    """{case: {"prior_surgery": bool}} from the subject table exported to CSV (columns SubjectID and
    'Prior Craniotomy/Biopsy/Resection')."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        v = r["Prior Craniotomy/Biopsy/Resection"].strip()
        if v not in ("Yes", "No"):
            raise ValueError(f"{r['SubjectID']}: unexpected surgery value {v!r}")
        out[r["SubjectID"].strip()] = {"prior_surgery": v == "Yes"}
    return out


def stage_raw(disease, raw_root):
    srcs = sources(disease, FM)
    extra = surgery_flags(BMSR_SUBJECTS) if disease == "metastasis" else None
    if extra is not None and set(extra) != set(srcs):
        raise ValueError(f"the subject table and the scans differ: {sorted(set(extra) ^ set(srcs))[:5]}")
    base = build_raw(raw_root, disease, srcs, extra)
    info = json.loads((base / "cases.json").read_text())
    n, p = len(info), len({v["patient"] for v in info.values()})
    print(f"wrote {base}: {n} cases, {p} patients, {sum(1 for v in info.values() if not v['n_label_voxels'])} cases without label voxels")
    if (n, p) != EXPECTED[disease]:
        raise AssertionError(f"expected {EXPECTED[disease]} (scans, patients), got {(n, p)}")


def stage_splits(disease, raw_root, preprocessed_root):
    info = json.loads((Path(raw_root) / DISEASES[disease]["name"] / "cases.json").read_text())
    cf = case_folds(disease, sorted(info))
    p = write_splits(preprocessed_root, disease, cf)
    print(f"wrote {p}: cases per fold {sorted(collections.Counter(cf.values()).items())}")


def main():
    ap = argparse.ArgumentParser(description="Build a brain disease dataset for nnU-Net or write its folds")
    ap.add_argument("--disease", choices=sorted(DISEASES), required=True)
    ap.add_argument("--stage", choices=("raw", "splits"), required=True)
    a = ap.parse_args()
    if a.stage == "raw":
        stage_raw(a.disease, os.environ["nnUNet_raw"])
    else:
        stage_splits(a.disease, os.environ["nnUNet_raw"], os.environ["nnUNet_preprocessed"])


if __name__ == "__main__":
    main()
````

- [ ] **Step 4: Run** the test → 2 passed.

- [ ] **Step 5: Export the BMSR subject table to CSV.** nvgen has no openpyxl; the base conda python has (3.0.10). Nothing is installed.

```bash
mkdir -p docs/verification/2026-09-29/brain_multidisease/build
test ! -e /data2/congcong/data/FM_data/derived/brain_disease/bmsr_subjects.csv && mkdir -p /data2/congcong/data/FM_data/derived/brain_disease && PYTHONNOUSERSITE=1 ~/anaconda3/bin/python - <<'EOF' | tee docs/verification/2026-09-29/brain_multidisease/build/bmsr_subjects.txt
import csv
import openpyxl
src = "/data2/congcong/data/FM_data/UCSF-BMSR_cbb/UCSF-BMSR/TableS1_UCSF_BrainMetastases_SubjectInfo.xlsx"
dst = "/data2/congcong/data/FM_data/derived/brain_disease/bmsr_subjects.csv"
ws = openpyxl.load_workbook(src, read_only=True).worksheets[0]
rows = [r for r in ws.iter_rows(values_only=True) if any(v is not None for v in r)]
hdr = [str(h) if h is not None else f"column_{i}" for i, h in enumerate(rows[0])]
with open(dst, "x", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(hdr)
    w.writerows(rows[1:])
print(len(rows) - 1, "subjects;", "columns:", hdr)
EOF
```

Expected: `461 subjects; columns: ['SubjectID', 'Sex', 'Age', 'CancerType', …, 'Prior Craniotomy/Biopsy/Resection', …]`.

- [ ] **Step 6: Build the three datasets** (each a few minutes; run in the background and poll if a call would exceed the tool's limit). The raw roots must not hold them yet: `ls /data2/congcong/data/FM_data/derived/nnunet/raw` shows only 901, 902, 903.

```bash
for d in glioma metastasis infarct; do bash -c "source scripts/nnunet_env.sh && PYTHONPATH=. nice -n 19 python scripts/brain_disease_prepare.py --disease $d --stage raw" 2>&1 | tee docs/verification/2026-09-29/brain_multidisease/build/${d}_raw.txt; done
```

Expected last lines: `wrote …/Dataset904_PDGMGlioma: 501 cases, 495 patients, 0 cases without label voxels`, `wrote …/Dataset905_BMSRMetastasis: 461 cases, 314 patients, 0 cases without label voxels`, `wrote …/Dataset906_ISLESInfarct: 250 cases, 250 patients, 3 cases without label voxels`. Any exception: stop and report; a partly built dataset directory stays as it is (name its path for the controller, never delete it).

- [ ] **Step 7: Plan and preprocess** (CPU; one dataset after the other; logs under the untracked `logs/brain_disease/`):

```bash
mkdir -p logs/brain_disease
for id in 904 905 906; do bash -c "source scripts/nnunet_env.sh && nice -n 19 nnUNetv2_plan_and_preprocess -d $id -c 3d_fullres --verify_dataset_integrity -np 8 -npfp 8" > logs/brain_disease/prep_$id.log 2>&1; grep -c "verify_dataset_integrity Done" logs/brain_disease/prep_$id.log; done
bash -c 'source scripts/nnunet_env.sh && python - <<"PY"
import json, os
for n in ("Dataset904_PDGMGlioma", "Dataset905_BMSRMetastasis", "Dataset906_ISLESInfarct"):
    c = json.load(open(os.path.join(os.environ["nnUNet_preprocessed"], n, "nnUNetPlans.json")))["configurations"]["3d_fullres"]
    print(n, "spacing", c["spacing"], "patch", c["patch_size"], "batch", c["batch_size"], "normalization", c["normalization_schemes"], "mask_for_norm", c["use_mask_for_norm"])
PY' | tee docs/verification/2026-09-29/brain_multidisease/build/plans.txt
```

Expected: each grep prints `1`; no traceback in the logs; three plan lines with `ZScoreNormalization` and `mask_for_norm` all true (the dry run on six cases per disease gave spacing 1 mm isotropic / patch 128×160×112 for glioma, 1.5 × 0.86 × 0.86 mm / 80×192×160 for metastasis, 2 mm isotropic / 80×80×80 for infarct; the full datasets may plan differently).

- [ ] **Step 8: Write the folds**

```bash
for d in glioma metastasis infarct; do bash -c "source scripts/nnunet_env.sh && PYTHONPATH=. python scripts/brain_disease_prepare.py --disease $d --stage splits"; done | tee docs/verification/2026-09-29/brain_multidisease/build/splits.txt
```

Expected cases per fold: glioma `[(0, 100), (1, 101), (2, 100), (3, 99), (4, 101)]`, metastasis `[(0, 105), (1, 86), (2, 87), (3, 106), (4, 77)]`, infarct `[(0, 50), (1, 50), (2, 50), (3, 50), (4, 50)]`.

- [ ] **Step 9: Commit** — `git add scripts/brain_disease_prepare.py tests/test_brain_disease_prepare.py docs/verification/2026-09-29/brain_multidisease/build && git commit -m "Brain disease datasets built on real data: glioma 501, metastasis 461, infarct 250 scans; nnU-Net plans and patient folds"`

---

### Task 3: GPU queue `scripts/gpu_queue.py` (the launch itself is the controller's)

**Files:**
- Create: `scripts/gpu_queue.py`
- Test: `tests/test_gpu_queue.py`

**Interfaces:**
- Consumes: `scripts/brain_detector_train.py::{idle_gpus, query_nvidia_smi, query_busy_pids}`; Task 1's `CONFIG, DISEASES, TRAINER, fold_dir`; `anatobind.infer.knee.NNUNET_ROOT`.
- Produces: `job_list(diseases, folds)`, `log_path(log_dir, job, trainer)`, `refusal(results_root, log_dir, job, trainer)`, `command(job, gpu, trainer, log, env_sh, n_proc_da=6)` (one brace group, the redirect on the group), `skipped_diseases(log_dir)`, `plan_launches(pending, running_gpus, idle, max_jobs)`, `idle_now(gpus) -> [gpu] | None` (None when nvidia-smi cannot be queried: nothing starts in that round), `main(argv=None)`; constants `LOG_DIR`, `MAX_JOBS = 6`, `N_PROC_DA = 6`, `POLL_SECONDS = 60`.

**Scope for the implementer: Steps 1–5 only.** Never run the queue for real and never start anything on a GPU; the controller launches it (Steps 6–7) after the reviews of Tasks 1–3.

- [ ] **Step 1: Write the failing test**

````python
# tests/test_gpu_queue.py
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("gpu_queue", REPO / "scripts/gpu_queue.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_jobs_go_out_fold_by_fold():
    q = _load()
    assert q.job_list(["glioma", "metastasis", "infarct"], [0, 1]) == [
        ("glioma", 0), ("metastasis", 0), ("infarct", 0), ("glioma", 1), ("metastasis", 1), ("infarct", 1)]


def test_launches_take_only_free_idle_gpus_up_to_the_cap():
    q = _load()
    pending = q.job_list(["glioma", "metastasis", "infarct"], [0, 1])
    assert q.plan_launches(pending, set(), [1, 2, 4], 6) == [(("glioma", 0), 1), (("metastasis", 0), 2), (("infarct", 0), 4)]
    assert q.plan_launches(pending, {1, 2}, [1, 2, 4, 5], 6) == [(("glioma", 0), 4), (("metastasis", 0), 5)]   # ours are not reused
    assert q.plan_launches(pending, {0, 1, 2, 4, 5}, [6, 7], 6) == [(("glioma", 0), 6)]                           # one slot left
    assert q.plan_launches(pending, {0, 1, 2, 4, 5, 6}, [7], 6) == []                                              # at the cap
    assert q.plan_launches(pending, set(), [], 6) == [] and q.plan_launches([], set(), [1], 6) == []


def test_refusal_names_existing_outputs(tmp_path):
    q = _load()
    job = ("infarct", 2)
    assert q.refusal(tmp_path / "res", tmp_path / "logs", job, "nnUNetTrainer_250epochs") is None
    log = q.log_path(tmp_path / "logs", job, "nnUNetTrainer_250epochs")
    assert log.name == "Dataset906_ISLESInfarct_nnUNetTrainer_250epochs_fold2.log"
    log.parent.mkdir()
    log.write_text("")
    assert "log" in q.refusal(tmp_path / "res", tmp_path / "logs", job, "nnUNetTrainer_250epochs")
    (tmp_path / "res/Dataset906_ISLESInfarct/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_2").mkdir(parents=True)
    assert "result folder" in q.refusal(tmp_path / "res", tmp_path / "logs", job, "nnUNetTrainer_250epochs")
    assert q.refusal(tmp_path / "res", tmp_path / "logs", job, "nnUNetTrainer_5epochs") is None


def test_command_pins_the_gpu_sets_the_workers_and_quotes_the_log():
    cmd = _load().command(("glioma", 3), 5, "nnUNetTrainer_250epochs", Path("/tmp/a b/x.log"), Path("/r/scripts/nnunet_env.sh"))
    assert cmd[:2] == ["bash", "-c"]
    # the redirect covers the whole group, so a failing `source` is written to the job's log too
    assert cmd[2] == ("{ source /r/scripts/nnunet_env.sh && export nnUNet_n_proc_DA=6 && CUDA_VISIBLE_DEVICES=5 nice -n 19 "
                      "nnUNetv2_train 904 3d_fullres 3 -tr nnUNetTrainer_250epochs --npz; } > '/tmp/a b/x.log' 2>&1")


def test_skip_files_name_diseases_by_dataset_id(tmp_path):
    q = _load()
    assert q.skipped_diseases(tmp_path) == set()
    (tmp_path / "skip_905").write_text("fold 0 sensitivity 0.21 < 0.3")
    assert q.skipped_diseases(tmp_path) == {"metastasis"}


def test_dry_run_prints_the_first_round_and_launches_nothing(tmp_path, monkeypatch, capsys):
    q = _load()
    monkeypatch.setattr(q, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))
    monkeypatch.setattr(q, "query_nvidia_smi", lambda: "0, 30000\n1, 14\n2, 14\n3, 18000\n")
    monkeypatch.setattr(q, "query_busy_pids", lambda: {0: [11], 3: [22]})
    monkeypatch.setattr(q.subprocess, "Popen", lambda *a, **k: pytest.fail("a dry run must not launch"))
    assert q.main(["--diseases", "glioma", "metastasis", "infarct", "--folds", "0", "--gpus", "0", "1", "2", "3", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would launch ('glioma', 0) on GPU 1" in out and "would launch ('metastasis', 0) on GPU 2" in out
    assert "('infarct', 0) on GPU" not in out                   # GPUs 0 and 3 are busy: no card for the third job


def test_a_failing_gpu_query_costs_one_round_not_the_queue(tmp_path, monkeypatch, capsys):
    q = _load()

    def boom():
        raise subprocess.CalledProcessError(9, ["nvidia-smi"])

    monkeypatch.setattr(q, "query_nvidia_smi", boom)
    monkeypatch.setattr(q, "query_busy_pids", lambda: {})
    assert q.idle_now([0, 1]) is None
    assert "GPU query failed (CalledProcessError" in capsys.readouterr().out
    monkeypatch.setattr(q, "query_nvidia_smi", lambda: "0, 14\n1, not-a-number\n")
    assert q.idle_now([0, 1]) is None and "GPU query failed (ValueError" in capsys.readouterr().out
    monkeypatch.setattr(q, "query_nvidia_smi", lambda: "0, 14\n1, 30000\n")
    assert q.idle_now([0, 1]) == [0]
    # the loop itself: a failing query starts nothing and does not raise
    monkeypatch.setattr(q, "query_nvidia_smi", boom)
    monkeypatch.setattr(q, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))
    assert q.main(["--diseases", "infarct", "--folds", "0", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "nothing starts in this round" in out and "would launch" not in out


def test_the_job_log_receives_a_failure_of_the_environment_script(tmp_path):
    q = _load()
    log = tmp_path / "job.log"
    cmd = q.command(("infarct", 0), 0, "nnUNetTrainer_250epochs", log, tmp_path / "no_such_env.sh")
    done = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert done.returncode != 0 and "no_such_env.sh" in log.read_text()


class _Proc:
    """A training that ends with `code` at the n-th poll."""
    pid = 4242

    def __init__(self, code, polls):
        self.code, self.polls, self.returncode = code, polls, None

    def poll(self):
        self.polls -= 1
        if self.polls <= 0:
            self.returncode = self.code
        return self.returncode


def _loop(q, tmp_path, monkeypatch, outcomes, clock_step=1.0, on_launch=None):
    """Run main() for the three diseases' fold 0 and 1 on two pretend cards; outcomes: job -> (exit code, polls)."""
    launched, now = [], [0.0]
    monkeypatch.setattr(q, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))
    monkeypatch.setattr(q, "idle_now", lambda gpus: [0, 1])

    def tick():
        now[0] += clock_step
        return now[0]

    def popen(cmd, **kw):
        job = next(j for j in q.job_list(["glioma", "metastasis", "infarct"], [0, 1])
                   if f"nnUNetv2_train {q.DISEASES[j[0]]['id']} 3d_fullres {j[1]} " in cmd[2])
        launched.append(job)
        if on_launch:
            on_launch(job)
        return _Proc(*outcomes.get(job, (0, 2)))

    monkeypatch.setattr(q, "time", SimpleNamespace(time=tick, sleep=lambda s: None))   # the queue's own clock only
    monkeypatch.setattr(q.subprocess, "Popen", popen)
    code = q.main(["--diseases", "glioma", "metastasis", "infarct", "--folds", "0", "1", "--gpus", "0", "1"])
    return code, launched


def test_the_loop_runs_every_job_and_counts_them(tmp_path, monkeypatch, capsys):
    q = _load()
    code, launched = _loop(q, tmp_path, monkeypatch, {})
    assert code == 0 and launched == q.job_list(["glioma", "metastasis", "infarct"], [0, 1])
    assert "done; succeeded 6, failed 0 [], refused 0 [], never started 0 []" in capsys.readouterr().out


def test_a_skip_file_drops_a_disease_and_a_stop_file_everything_that_has_not_started(tmp_path, monkeypatch, capsys):
    q = _load()
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "skip_905").write_text("fold 0 sensitivity 0.21 < 0.3")
    code, launched = _loop(q, tmp_path, monkeypatch, {})
    assert launched == [("glioma", 0), ("infarct", 0), ("glioma", 1), ("infarct", 1)] and code == 1
    assert "never started 2 [('metastasis', 0), ('metastasis', 1)]" in capsys.readouterr().out
    q2, other = _load(), tmp_path / "second"
    code, launched = _loop(q2, other, monkeypatch, {}, on_launch=lambda job: (
        (other / "logs" / "stop").write_text("stop") if job == ("metastasis", 0) else None))
    assert launched == [("glioma", 0), ("metastasis", 0)] and code == 1
    out = capsys.readouterr().out
    assert "stop file found: 4 jobs will not start" in out and "succeeded 2, failed 0 [], refused 0 [], never started 4" in out


def test_a_training_that_fails_at_once_stops_new_starts_and_a_late_failure_does_not(tmp_path, monkeypatch, capsys):
    q = _load()
    code, launched = _loop(q, tmp_path, monkeypatch, {("glioma", 0): (3, 1)})
    assert launched == [("glioma", 0), ("metastasis", 0)] and code == 1
    out = capsys.readouterr().out
    assert "finished ('glioma', 0) on GPU 0 with exit code 3" in out
    assert "failed within 10 min of its start: nothing new starts; 4 jobs will not start" in out
    assert "succeeded 1, failed 1 [('glioma', 0)], refused 0 [], never started 4" in out
    q2 = _load()
    code, launched = _loop(q2, tmp_path / "second", monkeypatch, {("glioma", 0): (3, 1)}, clock_step=700.0)
    assert len(launched) == 6 and code == 1                      # after 700 s a failure is that training's own
    assert "succeeded 5, failed 1 [('glioma', 0)], refused 0 [], never started 0 []" in capsys.readouterr().out


def test_a_refused_job_is_counted_and_the_others_run(tmp_path, monkeypatch, capsys):
    q = _load()
    (tmp_path / "logs").mkdir()
    q.log_path(tmp_path / "logs", ("infarct", 0), "nnUNetTrainer_250epochs").write_text("an earlier start")
    code, launched = _loop(q, tmp_path, monkeypatch, {})
    assert ("infarct", 0) not in launched and len(launched) == 5 and code == 1
    assert "refused 1 [('infarct', 0)]" in capsys.readouterr().out
````

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_gpu_queue.py -q -p no:cacheprovider` → FileNotFoundError on the script path.

- [ ] **Step 3: Implement** `scripts/gpu_queue.py`:

````python
#!/usr/bin/env python
# scripts/gpu_queue.py
"""Run the brain multi-disease nnU-Net trainings on whatever GPUs are idle (spec 2026-09-29 §4, M6, M14).

Jobs go out fold by fold (fold 0 of every disease first). Every POLL_SECONDS the queue looks for GPUs with no compute
process and almost no memory in use, and starts the next jobs there, one per GPU, never more than MAX_JOBS at once.
A card somebody else uses is never touched. Control files in the log directory: `skip_<dataset id>` drops the jobs of
that dataset that have not started; `stop` lets the running jobs finish and starts nothing new.

A training that fails within FAST_FAILURE_SECONDS of its start points at a fault that would hit every start (the
environment, a card that another job filled at the same moment): the queue then starts nothing new, because every
failed start leaves a log that blocks the job's next start. The last line counts what succeeded, failed, was refused
and never started; the return code is 1 unless every job succeeded.

Every launch writes its own log, so that the record of an earlier launch is never overwritten:

  cd <worktree> && PYTHONNOUSERSITE=1 PYTHONPATH=. setsid nohup ~/anaconda3/envs/nvgen/bin/python scripts/gpu_queue.py \
      --diseases glioma metastasis infarct --folds 0 1 2 3 4 \
      > logs/brain_disease/queue_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain_detector_train import idle_gpus, query_busy_pids, query_nvidia_smi  # noqa: E402
from anatobind.infer.knee import NNUNET_ROOT  # noqa: E402
from anatobind.nnunet.brain_disease import CONFIG, DISEASES, TRAINER, fold_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "logs" / "brain_disease"
MAX_JOBS = 6
N_PROC_DA = 6
POLL_SECONDS = 60
FAST_FAILURE_SECONDS = 600


def job_list(diseases, folds):
    """Fold-major order: fold 0 of every disease, then fold 1, ..."""
    return [(d, int(f)) for f in folds for d in diseases]


def log_path(log_dir, job, trainer):
    return Path(log_dir) / f"{DISEASES[job[0]]['name']}_{trainer}_fold{job[1]}.log"


def refusal(results_root, log_dir, job, trainer):
    """Why this job must not start (its outputs exist and would be overwritten), or None."""
    r, l = fold_dir(results_root, job[0], job[1], trainer), log_path(log_dir, job, trainer)
    if r.exists():
        return f"result folder {r} exists"
    if l.exists():
        return f"log {l} exists"
    return None


def command(job, gpu, trainer, log, env_sh, n_proc_da=N_PROC_DA):
    cmd = ["nnUNetv2_train", str(DISEASES[job[0]]["id"]), CONFIG, str(int(job[1])), "-tr", trainer, "--npz"]
    inner = (f"{{ source {shlex.quote(str(env_sh))} && export nnUNet_n_proc_DA={int(n_proc_da)} && "
             f"CUDA_VISIBLE_DEVICES={int(gpu)} nice -n 19 {' '.join(shlex.quote(c) for c in cmd)}; }} "
             f"> {shlex.quote(str(log))} 2>&1")
    return ["bash", "-c", inner]


def skipped_diseases(log_dir):
    return {d for d, spec in DISEASES.items() if (Path(log_dir) / f"skip_{spec['id']}").exists()}


def plan_launches(pending, running_gpus, idle, max_jobs):
    """[(job, gpu)] to start now: pending order is kept, a GPU that runs one of our jobs is never reused, and at most
    max_jobs run at once."""
    free = [g for g in idle if g not in running_gpus]
    room = max(0, max_jobs - len(running_gpus))
    return list(zip(pending, free))[:room]


def say(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def idle_now(gpus):
    """Idle GPUs among gpus, or None when nvidia-smi cannot be queried or read right now: the queue then starts
    nothing in this round and asks again at the next poll."""
    try:
        return idle_gpus(query_nvidia_smi(), gpus, query_busy_pids())
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        say(f"GPU query failed ({type(e).__name__}: {e}); nothing starts in this round")
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Queue nnU-Net trainings of the brain disease detectors on idle GPUs")
    ap.add_argument("--diseases", nargs="+", choices=sorted(DISEASES), required=True)
    ap.add_argument("--folds", nargs="+", type=int, required=True)
    ap.add_argument("--trainer", default=TRAINER)
    ap.add_argument("--gpus", nargs="+", type=int, default=list(range(8)), help="GPUs the queue may consider")
    ap.add_argument("--max-jobs", type=int, default=MAX_JOBS)
    ap.add_argument("--dry-run", action="store_true", help="print the first round of launches and exit")
    a = ap.parse_args(argv)

    results_root = Path(os.environ.get("nnUNet_results") or NNUNET_ROOT / "results")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pending, running = job_list(a.diseases, a.folds), {}
    succeeded, failed, refused, dropped = [], [], [], []
    say(f"queue of {len(pending)} jobs: {pending}")
    while True:
        for gpu, (job, proc, started) in list(running.items()):
            if proc.poll() is not None:
                took = time.time() - started
                say(f"finished {job} on GPU {gpu} with exit code {proc.returncode} after {took / 60:.1f} min")
                (succeeded if proc.returncode == 0 else failed).append(job)
                del running[gpu]
                if proc.returncode != 0 and took < FAST_FAILURE_SECONDS and pending:
                    say(f"{job} failed within {FAST_FAILURE_SECONDS // 60} min of its start: nothing new starts; "
                        f"{len(pending)} jobs will not start: {pending}")
                    dropped, pending = dropped + pending, []
        if (LOG_DIR / "stop").exists() and pending:
            say(f"stop file found: {len(pending)} jobs will not start: {pending}")
            dropped, pending = dropped + pending, []
        skip = skipped_diseases(LOG_DIR)
        if any(j[0] in skip for j in pending):
            say(f"skip file found: dropping {[j for j in pending if j[0] in skip]}")
            dropped += [j for j in pending if j[0] in skip]
            pending = [j for j in pending if j[0] not in skip]
        if not pending and not running:
            say(f"queue empty, nothing running: done; succeeded {len(succeeded)}, failed {len(failed)} {failed}, "
                f"refused {len(refused)} {refused}, never started {len(dropped)} {dropped}")
            return 0 if a.dry_run or not (failed or refused or dropped) else 1
        if pending and len(running) < a.max_jobs:
            for job, gpu in plan_launches(pending, set(running), idle_now(a.gpus) or [], a.max_jobs):
                pending.remove(job)
                why = refusal(results_root, LOG_DIR, job, a.trainer)
                if why:
                    say(f"refused {job}: {why}")
                    refused.append(job)
                    continue
                cmd = command(job, gpu, a.trainer, log_path(LOG_DIR, job, a.trainer), REPO / "scripts/nnunet_env.sh")
                if a.dry_run:
                    say(f"would launch {job} on GPU {gpu}: {cmd[2]}")
                    continue
                proc = subprocess.Popen(cmd, cwd=str(REPO), start_new_session=True, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                running[gpu] = (job, proc, time.time())
                say(f"launched {job} on GPU {gpu} (pid {proc.pid})")
        if a.dry_run:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
````

- [ ] **Step 4: Run** `tests/test_gpu_queue.py tests/test_brain_detector_train.py tests/test_nndet_train.py` → all pass (12 new; two of them were added after the task review of 2026-09-29: a failing GPU query costs one round, and the job log receives a failure of the environment script; four after the whole-branch review: the loop itself with a skip file, a stop file, a refused job, a training that fails at once and one that fails late).

- [ ] **Step 5: Commit** — `git add scripts/gpu_queue.py tests/test_gpu_queue.py && git commit -m "GPU queue: start the brain disease trainings fold by fold on idle cards, one per card, with skip and stop files"`

- [ ] **Step 6 (controller): launch.** Dry run first, then the real queue:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
bash -c 'source scripts/nnunet_env.sh && PYTHONPATH=. python scripts/gpu_queue.py --diseases glioma metastasis infarct --folds 0 1 2 3 4 --dry-run'
mkdir -p logs/brain_disease
bash -c 'source scripts/nnunet_env.sh && PYTHONPATH=. setsid nohup python scripts/gpu_queue.py --diseases glioma metastasis infarct --folds 0 1 2 3 4 > logs/brain_disease/queue_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &'   # the launch of 2026-09-29 13:46 wrote queue.log; every later launch gets its own name
```

- [ ] **Step 7 (controller): timing (M5 of the S2 line, N9-style rule).** After about 15 minutes read the epoch times of the running jobs: `grep "Epoch time" logs/brain_disease/Dataset90*_fold0.log | tail -20`. Projection per job = 250 × mean epoch time + validation (the dry run measured 67–73 s, 62–67 s and 23 s per epoch for glioma, metastasis and infarct). Write `docs/verification/2026-09-29/brain_multidisease/launch.md` (nvidia-smi snapshot, the queue's first lines, epoch times, projection per disease, how many cards were idle). A job projected beyond 24 h: tell the user and ask; the queue keeps running meanwhile. Commit the record: `git add docs/verification/2026-09-29/brain_multidisease/launch.md && git commit -m "Brain disease trainings queued: idle cards, epoch times and projection"`.

---

### Task 4: Lesion components `anatobind/eval/lesion_components.py`

**Files:**
- Create: `anatobind/eval/lesion_components.py`
- Test: `tests/test_brain_disease_components.py`

**Interfaces:**
- Consumes: `anatobind.eval.lesion_boxes.STRUCTURE` (26-connectivity).
- Produces: `MIN_MM3 = 10.0`, `STRATA = ("<5", "5-10", ">=10")`, `min_voxels_for(voxel_mm3, min_mm3=MIN_MM3) -> int`, `equivalent_diameter_mm(mm3)`, `size_stratum(mm3)`, `components(mask) -> (component map, n)`, `component_rows(comp, n, voxel_mm3, family, min_mm3=MIN_MM3) -> [{"component", "family", "box", "n_voxels", "mm3", "ignore"}]`, `component_mask(comp, row) -> (slices, bool mask)`.

- [ ] **Step 1: Write the failing test**

````python
# tests/test_brain_disease_components.py
import numpy as np
import pytest

from anatobind.eval.lesion_components import (
    MIN_MM3, component_mask, component_rows, components, equivalent_diameter_mm, min_voxels_for, size_stratum,
)


def test_volume_floor_in_voxels_for_the_three_voxel_sizes():
    assert MIN_MM3 == 10.0
    assert min_voxels_for(1.0) == 10            # PDGM, 1 mm isotropic: 10 voxels = 10 mm3
    assert min_voxels_for(0.859375 * 0.859375 * 1.5) == 10   # BMSR, 1.108 mm3: 9 voxels = 9.97 mm3 is under the floor
    assert min_voxels_for(8.0) == 2             # ISLES, 2 mm isotropic: one voxel = 8 mm3 is under the floor
    assert min_voxels_for(20.0) == 1            # a voxel larger than the floor always counts
    assert min_voxels_for(2.5) == 4             # exactly 10 mm3


def test_size_strata_use_the_equivalent_diameter():
    assert equivalent_diameter_mm(523.5988) == pytest.approx(10.0, abs=1e-3)
    assert size_stratum(65.0) == "<5" and size_stratum(66.0) == "5-10"      # 5 mm sphere = 65.45 mm3
    assert size_stratum(523.0) == "5-10" and size_stratum(524.0) == ">=10"  # 10 mm sphere = 523.6 mm3


def _mask():
    m = np.zeros((12, 12, 6), np.uint8)
    m[1:3, 1:3, 1:3] = 1          # 8 voxels
    m[3, 3, 3] = 1                # touches the block only by a corner: same component with 26-connectivity
    m[8:11, 8:11, 2:5] = 1        # 27 voxels
    m[0, 11, 5] = 1               # a single voxel
    return m


def test_rows_use_26_connectivity_and_flag_small_components():
    comp, n = components(_mask())
    rows = sorted(component_rows(comp, n, 1.0, "tumor"), key=lambda r: r["n_voxels"])
    assert n == 3 and [r["n_voxels"] for r in rows] == [1, 9, 27]
    assert [r["ignore"] for r in rows] == [True, True, False]       # 1 mm3 and 9 mm3 are under 10 mm3
    assert rows[1]["box"] == (1, 1, 1, 4, 4, 4) and rows[2]["box"] == (8, 8, 2, 11, 11, 5)
    assert rows[2]["mm3"] == 27.0 and rows[2]["family"] == "tumor"
    assert sorted(r["component"] for r in rows) == [1, 2, 3]
    big = sorted(component_rows(comp, n, 8.0, "infarct"), key=lambda r: r["n_voxels"])   # 2 mm isotropic voxels
    assert [r["ignore"] for r in big] == [True, False, False] and big[1]["mm3"] == 72.0


def test_component_mask_selects_only_its_own_component():
    m = np.zeros((6, 6, 1), np.uint8)
    m[0, 0:5, 0] = 1
    m[0:5, 0, 0] = 1              # an L
    m[3, 3, 0] = 1                # a separate voxel inside the L's box
    comp, n = components(m)
    rows = sorted(component_rows(comp, n, 20.0, "x"), key=lambda r: -r["n_voxels"])
    sl, mask = component_mask(comp, rows[0])
    assert n == 2 and mask.shape == (5, 5, 1) and int(mask.sum()) == 9 and not mask[3, 3, 0]
    assert sl == (slice(0, 5), slice(0, 5), slice(0, 1))


def test_an_empty_mask_has_no_rows():
    comp, n = components(np.zeros((4, 4, 4), np.uint8))
    assert n == 0 and component_rows(comp, n, 1.0, "tumor") == []


def test_the_floor_is_the_same_for_two_float32_headers_of_one_grid():
    assert min_voxels_for(0.5 * 0.5 * 2.0) == 20 and min_voxels_for(0.49999997 * 0.5 * 2.0) == 20    # scan 100201B
    assert min_voxels_for(0.5 * 1.0 * 0.5) == 40 and min_voxels_for(0.5 * 0.99999994 * 0.5) == 40    # scan 100203A
    assert min_voxels_for(0.4999) == 21                        # a voxel that is smaller by more than a rounding
````

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_brain_disease_components.py -q -p no:cacheprovider` → `ModuleNotFoundError: No module named 'anatobind.eval.lesion_components'`.

- [ ] **Step 3: Implement** `anatobind/eval/lesion_components.py`:

````python
"""Lesions as 26-connected components of a binary mask, with a physical volume floor (spec 2026-09-29 M7, §5).

A component under MIN_MM3 is kept but flagged "ignore": it never enters the denominator, and a detection matched to it
is neither a hit nor a false positive. The floor is a volume, not a voxel count, because the three datasets' voxels
differ eightfold (1 mm3 to 8 mm3)."""
import math

import numpy as np
from scipy import ndimage

from anatobind.eval.lesion_boxes import STRUCTURE

MIN_MM3 = 10.0
STRATA = ("<5", "5-10", ">=10")


def min_voxels_for(voxel_mm3, min_mm3=MIN_MM3):
    """Fewest voxels whose volume reaches min_mm3. The relative tolerance absorbs float32 headers: two headers of one
    grid (0.5 mm and 0.49999997 mm) give one floor."""
    return max(1, math.ceil(min_mm3 / float(voxel_mm3) * (1.0 - 1e-6)))


def equivalent_diameter_mm(mm3):
    return (6.0 * float(mm3) / math.pi) ** (1.0 / 3.0)


def size_stratum(mm3):
    d = equivalent_diameter_mm(mm3)
    return "<5" if d < 5.0 else ("5-10" if d < 10.0 else ">=10")


def components(mask):
    """(component map, number of components), 26-connectivity."""
    return ndimage.label(np.asarray(mask) > 0, structure=STRUCTURE)


def component_rows(comp, n, voxel_mm3, family, min_mm3=MIN_MM3):
    """One row per component: half-open box (x0, y0, z0, x1, y1, z1), voxel count, volume, and the ignore flag."""
    floor = min_voxels_for(voxel_mm3, min_mm3)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    rows = []
    for k, sl in enumerate(ndimage.find_objects(comp), start=1):
        if sl is None:
            continue
        nv = int(sizes[k])
        rows.append({"component": k, "family": family,
                     "box": (sl[0].start, sl[1].start, sl[2].start, sl[0].stop, sl[1].stop, sl[2].stop),
                     "n_voxels": nv, "mm3": nv * float(voxel_mm3), "ignore": bool(nv < floor)})
    return rows


def component_mask(comp, row):
    """(slices of the row's box, boolean mask of the component inside the box)."""
    b = row["box"]
    sl = tuple(slice(b[i], b[i + 3]) for i in range(3))
    return sl, comp[sl] == row["component"]
````

- [ ] **Step 4: Run** the test → 6 passed (the last test and the relative tolerance of the floor were added after the whole-branch review of 2026-09-29: two float32 headers of one grid, 0.5 mm and 0.49999997 mm, gave floors of 20 and 21 voxels).

- [ ] **Step 5: Commit** — `git add anatobind/eval/lesion_components.py tests/test_brain_disease_components.py && git commit -m "Lesion components: 26-connected components with a 10 mm3 volume floor and size strata"`

---

### Task 5: Ignore flag in `anatobind/eval/detection_metrics.py`

**Files:**
- Modify: `anatobind/eval/detection_metrics.py` (replace `_count`, add `scan_matches` above it, replace `per_family`; nothing else changes)
- Test: `tests/test_detection_metrics_ignore.py`

**Interfaces:**
- Produces: `scan_matches(s, thr, iou=IOU) -> (hits {gt index: detection index}, n false positives, detections at thr)`; `sweep`, `gate`, `operating_point`, `per_family` unchanged in signature; ground-truth rows may carry `"ignore": True`, and every counting function of the module leaves them out.

- [ ] **Step 1: Write the failing test**

````python
# tests/test_detection_metrics_ignore.py
import copy

from anatobind.eval.detection_metrics import gate, per_family, scan_matches, sweep


def _gt(box, ignore=None):
    r = {"family": "tumor", "box": box}
    if ignore is not None:
        r["ignore"] = ignore
    return r


def _det(box, score):
    return {"family": "tumor", "box": box, "score": score}


A, B, C, D = (0, 0, 0, 4, 4, 4), (10, 10, 10, 14, 14, 14), (20, 20, 20, 22, 22, 22), (30, 30, 30, 34, 34, 34)


def _scan():
    # ground truth: A and B real, C ignored (a fragment); detections: A found, C found, one stray at D; B missed
    return {"case": "s", "gt": [_gt(A, False), _gt(B, False), _gt(C, True)],
            "dets": [_det(A, 0.9), _det(C, 0.8), _det(D, 0.7)]}


def test_ignored_rows_leave_the_denominator_and_excuse_their_detections():
    hits, fp, dets = scan_matches(_scan(), 0.5)
    assert hits == {0: 0} and fp == 1 and len(dets) == 3          # only the stray at D is a false positive
    row = sweep([_scan()], thresholds=(0.5,))[0]
    assert (row["n_gt"], row["n_hit"], row["n_fp"]) == (2, 1, 1)
    assert row["sensitivity"] == 0.5 and row["fp_per_scan"] == 1.0
    high = sweep([_scan()], thresholds=(0.85,))[0]                # only the detection on A survives
    assert (high["n_gt"], high["n_hit"], high["n_fp"]) == (2, 1, 0)


def test_a_real_lesion_wins_a_detection_over_an_ignored_neighbour():
    # one detection overlaps both a real lesion and an ignored fragment more strongly
    real, frag, det = (0, 0, 0, 4, 4, 4), (1, 1, 1, 4, 4, 4), (1, 1, 1, 4, 4, 4)
    s = {"case": "s", "gt": [_gt(frag, True), _gt(real, False)], "dets": [_det(det, 0.9)]}
    hits, fp, _ = scan_matches(s, 0.5)
    assert hits == {1: 0} and fp == 0


def test_without_flags_the_counts_are_the_plain_ones():
    s = _scan()
    plain = copy.deepcopy(s)
    for r in plain["gt"]:
        r.pop("ignore")
    row = sweep([plain], thresholds=(0.5,))[0]
    assert (row["n_gt"], row["n_hit"], row["n_fp"]) == (3, 2, 1)   # C counts and is found; the stray stays
    false_flags = copy.deepcopy(s)
    for r in false_flags["gt"]:
        r["ignore"] = False
    assert sweep([false_flags], thresholds=(0.5,))[0] == row
    assert gate(sweep([plain]))["pass"] is True                    # 2 of 3 at 1 false positive per scan


def test_per_family_leaves_ignored_rows_out():
    assert per_family([_scan()], 0.5) == {"tumor": {"n_gt": 2, "n_hit": 1, "n_hit_family": 1, "sensitivity": 0.5,
                                                    "sensitivity_family": 0.5}}
    only_a_fragment = {"case": "s", "gt": [_gt(C, True)], "dets": [_det(C, 0.9)]}
    assert per_family([only_a_fragment], 0.5) == {}                # a detection on the fragment is no hit
    plain = copy.deepcopy(_scan())
    for r in plain["gt"]:
        r.pop("ignore")
    assert per_family([plain], 0.5)["tumor"] == {"n_gt": 3, "n_hit": 2, "n_hit_family": 2, "sensitivity": 2 / 3,
                                                 "sensitivity_family": 2 / 3}


def test_a_fragment_excuses_one_detection_and_empty_scans_count_every_detection():
    twice = {"case": "s", "gt": [_gt(C, True)], "dets": [_det(C, 0.9), _det(C, 0.8), _det(D, 0.7)]}
    hits, fp, _ = scan_matches(twice, 0.5)
    assert hits == {} and fp == 2                                  # the second detection on C and the stray at D
    row = sweep([twice, _scan()], thresholds=(0.5,))[0]            # a scan with fragments only adds nothing to n_gt
    assert (row["n_gt"], row["n_hit"], row["n_fp"], row["n_scans"]) == (2, 1, 3, 2)
    empty = {"case": "n", "gt": [], "dets": [_det(A, 0.9), _det(B, 0.6)]}
    assert scan_matches(empty, 0.5)[:2] == ({}, 2)
````

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_detection_metrics_ignore.py -q -p no:cacheprovider` → `ImportError: cannot import name 'scan_matches'`.

- [ ] **Step 3: Implement** — in `anatobind/eval/detection_metrics.py` replace the whole function `_count` by:

````python
def scan_matches(s, thr, iou=IOU):
    """Matches of one scan at a score threshold: (hits {gt index: detection index}, false positives, detections kept).

    Ground-truth rows flagged "ignore" (spec 2026-09-29 M7) never count: the other rows are matched first; detections
    left over are then matched against the ignored rows and excused. Without any flag this is plain matching."""
    dets = [d for d in s["dets"] if d["score"] >= thr]
    real = [g for g, r in enumerate(s["gt"]) if not r.get("ignore")]
    ignored = [g for g, r in enumerate(s["gt"]) if r.get("ignore")]
    pairs = match_scan([s["gt"][g] for g in real], dets, iou)
    hits = {real[g]: p for g, p in pairs.items()}
    used = set(pairs.values())
    rest = [d for i, d in enumerate(dets) if i not in used]
    excused = match_scan([s["gt"][g] for g in ignored], rest, iou) if ignored and rest else {}
    return hits, len(dets) - len(pairs) - len(excused), dets


def _count(scans, thr, iou):
    n_gt = n_hit = n_fam = n_fp = 0
    for s in scans:
        hits, fp, dets = scan_matches(s, thr, iou)
        n_gt += sum(1 for r in s["gt"] if not r.get("ignore"))
        n_hit += len(hits)
        n_fam += sum(1 for g, p in hits.items() if dets[p]["family"] == s["gt"][g]["family"])
        n_fp += fp
    return n_gt, n_hit, n_fam, n_fp
````

and replace the whole function `per_family` by:

````python
def per_family(scans, thr, iou=IOU):
    """Counts per family over the ground truth that is not ignored, with the matches of scan_matches."""
    out = {}
    for s in scans:
        hits, _, dets = scan_matches(s, thr, iou)
        for g, r in enumerate(s["gt"]):
            if r.get("ignore"):
                continue
            f = out.setdefault(r["family"], {"n_gt": 0, "n_hit": 0, "n_hit_family": 0})
            f["n_gt"] += 1
            if g in hits:
                f["n_hit"] += 1
                f["n_hit_family"] += int(dets[hits[g]]["family"] == r["family"])
    for f in out.values():
        f["sensitivity"] = _rate(f["n_hit"], f["n_gt"])
        f["sensitivity_family"] = _rate(f["n_hit_family"], f["n_gt"])
    return out
````

- [ ] **Step 4: Run** `tests/test_detection_metrics_ignore.py tests/test_detection_metrics.py tests/test_brain_detector_eval.py tests/test_brain_detector_eval_script.py tests/test_nndet_eval.py tests/test_nndet_eval_script.py tests/test_eval_knee_folds_script.py` → all pass (5 new; the old ones unchanged). The last two tests and the new `per_family` were added after the task review of 2026-09-29: `per_family` counted ignored rows and hits on them, and three corners (two detections on one ignored row, a scan whose ground truth is ignored entirely, a scan without ground truth) had no test. Without any flag the old and the new `per_family` agree (8000 random comparisons in the dry-run copy).

- [ ] **Step 5: Commit** — `git add anatobind/eval/detection_metrics.py tests/test_detection_metrics_ignore.py && git commit -m "Detection metrics: optional ignore flag on ground-truth rows, real rows matched first; unchanged without the flag"`

---

### Task 6: Binding `anatobind/bind/brain_lookup.py`

**Files:**
- Create: `anatobind/bind/__init__.py`, `anatobind/bind/brain_lookup.py`
- Test: `tests/test_brain_disease_binding.py`

**Interfaces:**
- Consumes: `anatobind.eval.geometry.{CLASS_NAMES, LEFT_LABELS, RIGHT_LABELS, MIDLINE_SHARE, host_class_map}`.
- Produces: `BrainBinder(seg, spacing).bind(sl, mask) -> {"host": name | None, "host_rule": "overlap" | "nearest" | None, "host_fractions": {name: share}, "side": "left" | "right" | "bilateral" | "midline"}`; `sl` = the lesion box as a tuple of slices, `mask` = the lesion's voxels inside it (what `component_mask` returns).

- [ ] **Step 1: Write the failing test**

````python
# tests/test_brain_disease_binding.py
import numpy as np
import pytest

from anatobind.bind.brain_lookup import BrainBinder

# SynthSeg labels: 2 / 41 left / right white matter, 3 / 42 left / right cortex, 11 left caudate, 16 brainstem,
# 4 left lateral ventricle (a landmark, never a host), 0 background


def _seg():
    seg = np.zeros((20, 10, 4), np.int16)
    seg[0:10, :, :] = 2            # left white matter
    seg[10:20, :, :] = 41          # right white matter
    seg[0:4, 0:4, :] = 3           # left cortex
    seg[8:10, 4:6, :] = 4          # left lateral ventricle
    seg[9:11, 8:10, :] = 16        # brainstem across the midline
    return seg


def _box(x0, x1, y0, y1, z0, z1):
    sl = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
    return sl, np.ones((x1 - x0, y1 - y0, z1 - z0), bool)


def test_main_structure_fractions_and_side_by_overlap():
    b = BrainBinder(_seg(), (1.0, 1.0, 1.0))
    out = b.bind(*_box(2, 6, 0, 4, 0, 1))          # 8 cortex voxels + 8 white matter voxels, all on the left
    assert out["host_rule"] == "overlap" and out["side"] == "left"
    assert out["host_fractions"] == {"white_matter": 0.5, "cortex": 0.5}
    assert out["host"] == "white_matter"           # a tie goes to the lower class id
    out = b.bind(*_box(12, 16, 0, 4, 0, 2))
    assert out == {"host": "white_matter", "host_rule": "overlap", "host_fractions": {"white_matter": 1.0}, "side": "right",
                   "host_side": "right", "host_sides": {"white_matter": "right"}, "host_distance_mm": 0.0}


def test_both_sides_above_forty_percent_is_bilateral():
    b = BrainBinder(_seg(), (1.0, 1.0, 1.0))
    assert b.bind(*_box(8, 12, 0, 2, 0, 1))["side"] == "bilateral"        # 4 left + 4 right voxels
    assert b.bind(*_box(5, 12, 0, 1, 0, 1))["side"] == "left"             # 5 left + 2 right: 2 / 7 < 0.4


def test_a_structure_without_a_side_is_midline():
    out = BrainBinder(_seg(), (1.0, 1.0, 1.0)).bind(*_box(9, 11, 8, 10, 0, 1))
    assert out["host"] == "brainstem" and out["side"] == "midline" and out["host_side"] == "midline"


def test_a_lesion_on_a_landmark_takes_the_nearest_host():
    out = BrainBinder(_seg(), (1.0, 1.0, 1.0)).bind(*_box(8, 10, 4, 6, 1, 2))   # inside the left lateral ventricle
    assert out == {"host": "white_matter", "host_rule": "nearest", "host_fractions": {}, "side": "left",
                   "host_side": "left", "host_sides": {"white_matter": "left"}, "host_distance_mm": 1.0}


def test_nearest_uses_millimetres_not_voxels():
    seg = np.zeros((9, 1, 9), np.int16)
    seg[0, 0, 4] = 3               # left cortex, 4 voxels away along x
    seg[4, 0, 8] = 41              # right white matter, 4 voxels away along z
    sl, mask = (slice(4, 5), slice(0, 1), slice(4, 5)), np.ones((1, 1, 1), bool)
    assert BrainBinder(seg, (1.0, 1.0, 5.0)).bind(sl, mask)["host"] == "cortex"          # 4 mm against 20 mm
    assert BrainBinder(seg, (5.0, 1.0, 1.0)).bind(sl, mask)["host"] == "white_matter"    # 20 mm against 4 mm


def test_no_host_anywhere_and_a_mask_off_the_grid():
    seg = np.zeros((4, 4, 4), np.int16)
    seg[0, 0, 0] = 4
    b = BrainBinder(seg, (1.0, 1.0, 1.0))
    assert b.bind(*_box(1, 2, 1, 2, 1, 2)) == {"host": None, "host_rule": None, "host_fractions": {}, "side": "midline",
                                               "host_side": "midline", "host_sides": {}, "host_distance_mm": None}
    with pytest.raises(ValueError, match="does not fit"):
        b.bind((slice(2, 6), slice(0, 1), slice(0, 1)), np.ones((4, 1, 1), bool))


def test_the_side_of_the_main_structure_is_counted_on_its_own_voxels():
    seg = np.zeros((20, 10, 4), np.int16)
    seg[0:10] = 10                  # left thalamus
    seg[10:20] = 41                 # right white matter
    seg[9:11, 8:10, :] = 16         # brainstem
    b = BrainBinder(seg, (1.0, 1.0, 1.0))
    out = b.bind(*_box(5, 14, 0, 1, 0, 1))            # 5 voxels of the left thalamus, 4 of the right white matter
    assert out["host"] == "thalamus" and out["side"] == "bilateral" and out["host_side"] == "left"
    assert out["host_sides"] == {"white_matter": "right", "thalamus": "left"}
    out = b.bind(*_box(7, 10, 8, 10, 0, 1))           # 4 voxels of the left thalamus, 2 of the brainstem
    assert out["host"] == "thalamus" and out["host_side"] == "left"
    out = b.bind(*_box(8, 11, 8, 10, 0, 1))           # 4 brainstem voxels, 2 of the left thalamus
    assert out["host"] == "brainstem" and out["side"] == "left" and out["host_side"] == "midline"
    assert out["host_sides"] == {"thalamus": "left", "brainstem": "midline"}


def test_the_distance_of_the_nearest_rule_is_in_millimetres():
    seg = np.zeros((30, 1, 3), np.int16)
    seg[0, 0, 0] = 3                # left cortex: the only structure
    b = BrainBinder(seg, (1.0, 1.0, 5.0))
    sl, mask = (slice(12, 14), slice(0, 1), slice(0, 1)), np.ones((2, 1, 1), bool)
    assert b.bind(sl, mask)["host_distance_mm"] == 12.0          # the nearer of the lesion's two voxels
    sl = (slice(0, 1), slice(0, 1), slice(2, 3))
    out = b.bind(sl, np.ones((1, 1, 1), bool))
    assert out["host_distance_mm"] == 10.0 and out["host_side"] == "left"   # two slices of 5 mm
````

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_brain_disease_binding.py -q -p no:cacheprovider` → `ModuleNotFoundError: No module named 'anatobind.bind'`.

- [ ] **Step 3: Implement.** `anatobind/bind/__init__.py`:

````python
"""Binding a lesion to anatomy (spec 2026-09-29 §6)."""
````

`anatobind/bind/brain_lookup.py`:

````python
"""Which structure a lesion sits in, from its mask and a SynthSeg label map on the same grid (spec 2026-09-29 §6).

Host classes are the seven of anatobind.eval.geometry (sides merged; ventricles and CSF are landmarks, never hosts).
Everything this module returns rests on a pseudo-label: NOT_EVIDENCE."""
import numpy as np
from scipy import ndimage

from anatobind.eval.geometry import CLASS_NAMES, HOST_CLASSES, LEFT_LABELS, MIDLINE_SHARE, RIGHT_LABELS, host_class_map


def side_of(labels):
    """left / right by the majority of the sided labels; bilateral when both reach MIDLINE_SHARE; midline without any."""
    left, right = int(np.isin(labels, LEFT_LABELS).sum()), int(np.isin(labels, RIGHT_LABELS).sum())
    if left + right == 0:
        return "midline"
    if min(left, right) / (left + right) >= MIDLINE_SHARE:
        return "bilateral"
    return "left" if left > right else "right"


class BrainBinder:
    def __init__(self, seg, spacing):
        self.seg = np.asarray(seg)
        self.spacing = tuple(float(s) for s in spacing)
        self.classes = host_class_map(self.seg)
        self._nearest = None

    def _nearest_host(self):
        """(distance in mm to the nearest host voxel, SynthSeg label of that voxel); (None, None) without any host."""
        if self._nearest is None:
            host = self.classes > 0
            if not host.any():
                self._nearest = (None, None)
            else:
                dist, idx = ndimage.distance_transform_edt(~host, sampling=self.spacing, return_indices=True)
                self._nearest = (dist, self.seg[tuple(idx)])
        return self._nearest

    def bind(self, sl, mask):
        """sl: the lesion box as slices; mask: the lesion's voxels inside that box.

        side is counted over the whole lesion, host_side over the lesion's voxels inside the main structure, and
        host_sides likewise for every structure of host_fractions: a lesion of the left thalamus that reaches into
        the right white matter is bilateral, its main structure is the left thalamus, the white matter it involves is
        the right one. host_distance_mm is 0 when the lesion overlaps a structure, else the distance to the nearest
        one."""
        if self.seg[sl].shape != mask.shape:
            raise ValueError(f"lesion mask {mask.shape} does not fit the anatomy grid {self.seg.shape}")
        labels = self.seg[sl][mask]
        counts = np.bincount(self.classes[sl][mask].astype(np.int64), minlength=len(CLASS_NAMES) + 1)[1:]
        if counts.sum() > 0:
            host, rule, distance = CLASS_NAMES[int(np.argmax(counts))], "overlap", 0.0
            fractions = {CLASS_NAMES[i]: round(float(c) / float(counts.sum()), 4) for i, c in enumerate(counts) if c}
            sides = {name: side_of(labels[np.isin(labels, HOST_CLASSES[name])]) for name in fractions}
        else:
            dist, near = self._nearest_host()
            if dist is None:
                return {"host": None, "host_rule": None, "host_fractions": {}, "side": "midline", "host_side": "midline",
                        "host_sides": {}, "host_distance_mm": None}
            j = int(np.argmin(dist[sl][mask]))
            labels = near[sl][mask][j:j + 1]
            host, rule, fractions = CLASS_NAMES[int(host_class_map(labels)[0]) - 1], "nearest", {}
            sides, distance = {host: side_of(labels)}, round(float(dist[sl][mask][j]), 2)
        return {"host": host, "host_rule": rule, "host_fractions": fractions, "side": side_of(labels),
                "host_side": sides[host], "host_sides": sides, "host_distance_mm": distance}
````

- [ ] **Step 4: Run** the test → 8 passed (after the re-review of 2026-09-29 the binder also returns `host_side`, the side counted over the lesion's voxels inside the main structure, and `host_distance_mm`).

- [ ] **Step 5: Commit** — `git add anatobind/bind/__init__.py anatobind/bind/brain_lookup.py tests/test_brain_disease_binding.py && git commit -m "Brain binding: main structure, shares and patient side of a lesion mask from the SynthSeg label map"`

---

### Task 7: Inference, structured record and sentence

**Files:**
- Create: `anatobind/infer/brain_disease.py`, `scripts/infer_brain_disease.py`
- Test: `tests/test_brain_disease_record.py`

**Interfaces:**
- Consumes: Tasks 1, 4, 6; `anatobind.eval.lesion_boxes.{load_label_map, load_nnunet_probabilities}`; `anatobind.infer.knee.nnunet_env`.
- Produces: `detections(pred, probs, voxel_mm3, family) -> (rows sorted by score, component map)` (rows as `component_rows` without `ignore`, plus `score`), `bind_rows(rows, comp, binder) -> rows`, `volume_text(mm3)`, `lesion_clause(lesion)`, `study_record(study, disease, threshold, rows) -> dict`, `run_nnunet(dataset_id, in_dir, out_dir, folds, gpu)`, `GRID_TOL = 1e-3`, `check_grid(images, seg_img)` (every channel on the anatomy's grid: same shape, affines equal within `GRID_TOL`; raises before anything is written), `link_inputs(in_dir, case, images, n_channels)` (checks every file before it creates the folder), `run(disease, images, anatomy, out_dir, folds, gpu, threshold, predict=run_nnunet) -> record`.
- The spec (§5) says decoding reuses `decode_boxes`; `detections` replaces it here because binding needs the component of every detection, which `decode_boxes` does not return. Scores and boxes are computed the same way.

- [ ] **Step 1: Write the failing test**

````python
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
            _row(0.9, 40.0, host=None, side="midline", fractions={})]
    rec = B.study_record("s9", "infarct", 0.5, big + rest)
    assert rec["sentence"].endswith("；另有 6 处同类异常（还见于右侧大脑皮层、左侧丘脑、邻近左侧大脑白质、未能定位的区域）。疑似缺血性梗死。")
    same = B.study_record("s10", "infarct", 0.5, big + [_row(0.9, 70.0, host="cortex", side="left")])
    assert same["sentence"].endswith("；另有 1 处同类异常。疑似缺血性梗死。")          # nothing new to name
````

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_brain_disease_record.py -q -p no:cacheprovider` → `ModuleNotFoundError: No module named 'anatobind.infer.brain_disease'`.

- [ ] **Step 3: Implement.** `anatobind/infer/brain_disease.py`:

````python
"""Brain multi-disease inference and the per-study structured record (spec 2026-09-29 §6, §7).

A study's sequences (one disease model's native channels, one grid) and a SynthSeg label map on that grid go in; scored
lesions with their host structure and side, a disease impression and one sentence come out. The impression is the
model's disease: this entry detects one kind of lesion, it does not tell diseases apart (spec M13)."""
import json
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.bind.brain_lookup import BrainBinder
from anatobind.eval.lesion_boxes import load_label_map, load_nnunet_probabilities
from anatobind.eval.lesion_components import component_mask, component_rows, components
from anatobind.infer.knee import nnunet_env
from anatobind.nnunet.brain_disease import CONFIG, DISEASES, TRAINER

SIDE_ZH = {"left": "左侧", "right": "右侧", "bilateral": "双侧", "midline": ""}
HOST_ZH = {"white_matter": "大脑白质", "cortex": "大脑皮层", "thalamus": "丘脑", "basal_ganglia": "基底节",
           "brainstem": "脑干", "cerebellum": "小脑", "other_deep_grey": "深部灰质（海马、杏仁核等）"}
TYPE_ZH = {"tumor": "肿瘤样异常", "metastasis": "转移瘤样异常", "infarct": "梗死样异常"}
NOWHERE_ZH = "未能定位的区域"
NEAR_MM = 10.0
ANATOMY_SOURCE = "SynthSeg pseudo-label, lookup by voxel count (NOT_EVIDENCE)"
MAX_SENTENCE_LESIONS = 5
INVOLVED_MIN = 0.10
GRID_TOL = 1e-3
PROB_TOL = 1e-3


def detections(pred, probs, voxel_mm3, family):
    """Scored components of the predicted label map, highest score first, and the component map they index.
    Components under the volume floor are dropped; the score is the mean foreground probability in the component.
    Every voxel of the argmax map has a foreground probability above 0.5: a component that holds a lower one shows
    that the probabilities do not belong to this label map, and nothing is scored."""
    comp, n = components(np.asarray(pred) == 1)
    rows = []
    for r in component_rows(comp, n, voxel_mm3, family):
        sl, m = component_mask(comp, r)
        p = probs[1][sl][m]
        if float(p.min()) < 0.5 - PROB_TOL:
            raise ValueError(f"component {r['component']} holds a voxel with foreground probability {float(p.min()):.3f}: "
                             "the probabilities do not belong to this label map (axis order?)")
        if not r["ignore"]:
            rows.append({**{k: v for k, v in r.items() if k != "ignore"}, "score": float(p.mean())})
    return sorted(rows, key=lambda r: -r["score"]), comp


def bind_rows(rows, comp, binder):
    """Attach the binder's fields (host, host_rule, host_fractions, side, host_side, host_sides, host_distance_mm) to
    every row, in place; returns rows."""
    for r in rows:
        r.update(binder.bind(*component_mask(comp, r)))
    return rows


def volume_text(mm3):
    if mm3 >= 10000:
        return f"约 {mm3 / 1000:.0f} mL"
    if mm3 >= 1000:
        return f"约 {mm3 / 1000:.1f} mL"
    return f"约 {mm3:.0f} mm³"


def place(lesion):
    """Where a lesion lies, in words: side and main structure, "邻近…" for the nearest rule within NEAR_MM, not
    located beyond it or without any structure."""
    if not lesion["host"] or (lesion["host_rule"] == "nearest" and lesion["host_distance_mm"] > NEAR_MM):
        return NOWHERE_ZH
    where = SIDE_ZH[lesion["host_side"]] + HOST_ZH[lesion["host"]]
    return f"邻近{where}" if lesion["host_rule"] == "nearest" else where


def lesion_clause(lesion):
    """One lesion in words. The side written before the main structure is the side of the lesion's voxels inside
    that structure (none for the brainstem). When every involved structure lies on the main structure's side, no side
    is written after "累及"; when one does not, every involved structure gets its side, so that a lesion across the
    midline reads as one and no side word is read on to the next structure; the brainstem, which has no side, then
    comes first. A lesion that overlaps no structure is said to lie next to the nearest one when that is at most
    NEAR_MM away (user, 2026-09-29), else it is not located."""
    where = place(lesion) + ("（未与任何结构重叠）" if lesion["host_rule"] == "nearest" and place(lesion) != NOWHERE_ZH else "")
    sides = lesion["host_sides"]
    involved = [h for h, f in sorted(lesion["host_fractions"].items(), key=lambda kv: -kv[1])
                if h != lesion["host"] and f >= INVOLVED_MIN]
    if any(sides[h] not in ("midline", lesion["host_side"]) for h in involved):
        involved = [h for h in involved if sides[h] == "midline"] + [h for h in involved if sides[h] != "midline"]
        words = [SIDE_ZH[sides[h]] + HOST_ZH[h] for h in involved]
    else:
        words = [HOST_ZH[h] for h in involved]
    text = f"{where}存在{TYPE_ZH[lesion['type']]}，体积{volume_text(lesion['volume_mm3'])}"
    return text + (f"，累及{'、'.join(words)}" if words else "")


def study_record(study, disease, threshold, rows, folds=None):
    """rows: bound detections (any score); only those at or above the threshold enter the record, highest score
    first. The sentence names the MAX_SENTENCE_LESIONS largest lesions, largest first: the score is a mean
    probability, which is highest for the smallest components, and a reader looks for the largest lesion first.
    The others are counted, and their places are named where no named lesion lies there.
    folds: the folds of the model that predicted. Without a lesion the sentence says that this model detected nothing
    at this threshold; that is no negative finding."""
    spec = DISEASES[disease]
    lesions = [{"type": spec["type"], "score": round(float(r["score"]), 4), "box": [int(v) for v in r["box"]],
                "volume_mm3": round(float(r["mm3"]), 1), "host": r["host"], "host_rule": r["host_rule"],
                "host_fractions": r["host_fractions"], "side": r["side"], "host_side": r["host_side"],
                "host_sides": r["host_sides"], "host_distance_mm": r["host_distance_mm"]}
               for r in sorted(rows, key=lambda r: -r["score"]) if r["score"] >= threshold]
    if lesions:
        rest = len(lesions) - MAX_SENTENCE_LESIONS
        by_volume = sorted(lesions, key=lambda l: -l["volume_mm3"])
        largest, others = by_volume[:MAX_SENTENCE_LESIONS], by_volume[MAX_SENTENCE_LESIONS:]
        named = {place(l) for l in largest}
        also = list(dict.fromkeys(place(l) for l in others if place(l) not in named))
        sentence = "；".join(lesion_clause(l) for l in largest)
        if rest > 0:
            sentence += f"；另有 {rest} 处同类异常" + (f"（还见于{'、'.join(also)}）" if also else "")
        sentence += f"。{spec['impression']}。"
        impression = spec["impression"]
    else:
        sentence, impression = f"本模型未检出{TYPE_ZH[spec['type']]}（阈值 {float(threshold):.2f}）。", "未检出相关异常"
    return {"study": study, "disease_model": disease, "model_folds": None if folds is None else sorted(int(f) for f in folds),
            "threshold": float(threshold), "impression": impression, "lesions": lesions, "sentence": sentence,
            "anatomy_source": ANATOMY_SOURCE}


def run_nnunet(dataset_id, in_dir, out_dir, folds, gpu):
    cmd = ["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir), "-d", str(dataset_id), "-c", CONFIG, "-tr", TRAINER,
           "-f", *[str(f) for f in folds], "-npp", "2", "-nps", "2", "--disable_progress_bar", "--save_probabilities"]
    subprocess.run(["nice", "-n", "19", *cmd], check=True, env=nnunet_env(gpu))


def check_grid(images, seg_img):
    """Every channel must be on the anatomy's grid (same shape, affines equal within GRID_TOL): lesion volumes and the
    nearest structure are computed with the anatomy's voxel size."""
    for p in images:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
        img = nib.load(str(p))
        dev = float(np.abs(img.affine - seg_img.affine).max())
        if img.shape != seg_img.shape or dev > GRID_TOL:
            raise ValueError(f"{p} is not on the anatomy's grid: shape {img.shape} against {seg_img.shape}, "
                             f"largest affine difference {dev:.3g}")


def link_inputs(in_dir, case, images, n_channels):
    """Nothing is created unless the channel count is right and every file exists."""
    if len(images) != n_channels:
        raise ValueError(f"{n_channels} channels are needed, {len(images)} given")
    for p in images:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    Path(in_dir).mkdir(parents=True)
    for k, p in enumerate(images):
        os.symlink(Path(p).resolve(), Path(in_dir) / f"{case}_{k:04d}.nii.gz")


def run(disease, images, anatomy, out_dir, folds, gpu, threshold, predict=run_nnunet):
    """images: one NIfTI per channel, in the order of DISEASES[disease]["channels"], all on the anatomy's grid."""
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    spec = DISEASES[disease]
    seg_img = nib.load(str(anatomy))
    check_grid(images, seg_img)
    link_inputs(out / "input", "case", images, len(spec["channels"]))
    predict(spec["id"], out / "input", out / "pred", folds, gpu)
    pred = load_label_map(out / "pred" / "case.nii.gz")
    if pred.shape != seg_img.shape:
        raise ValueError(f"prediction {pred.shape} and anatomy {seg_img.shape} are on different grids")
    zooms = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
    rows, comp = detections(pred, load_nnunet_probabilities(out / "pred" / "case.npz", pred), float(np.prod(zooms)), spec["type"])
    bind_rows(rows, comp, BrainBinder(np.asarray(seg_img.dataobj), zooms))
    record = study_record(Path(images[0]).name, disease, threshold, rows, folds)
    (out / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    return record
````

`scripts/infer_brain_disease.py`:

````python
#!/usr/bin/env python
# scripts/infer_brain_disease.py
"""Brain disease inference: a study's sequences and its SynthSeg label map -> the structured record (spec 2026-09-29 §7).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_disease.py --disease infarct \
      --images DWI.nii.gz ADC.nii.gz --anatomy DWI_seg.nii.gz --threshold 0.55 --out DIR --folds 0 1 2 3 4 --gpu 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.infer.brain_disease import run  # noqa: E402
from anatobind.nnunet.brain_disease import DISEASES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Brain disease inference with binding and a structured record")
    ap.add_argument("--disease", choices=sorted(DISEASES), required=True)
    ap.add_argument("--images", type=Path, nargs="+", required=True, help="one NIfTI per channel, in the model's channel order")
    ap.add_argument("--anatomy", type=Path, required=True, help="SynthSeg label map on the images' grid")
    ap.add_argument("--threshold", type=float, required=True, help="the disease's operating threshold (verdict.json)")
    ap.add_argument("--out", type=Path, required=True, help="output directory (must not exist)")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    print(f"channels expected: {DISEASES[a.disease]['channels']}")
    rec = run(a.disease, a.images, a.anatomy, a.out, a.folds, a.gpu, a.threshold)
    print(f"{len(rec['lesions'])} lesions -> {a.out / 'record.json'}")
    print(rec["sentence"])


if __name__ == "__main__":
    main()
````

- [ ] **Step 4: Run** `tests/test_brain_disease_record.py tests/test_brain_detector_infer.py` → all pass (15 new). Do not run the CLI for real (the smoke run is Task 12). The side written before a structure is `host_side`; when an involved structure lies on another side than the main structure, every involved structure gets its side (`host_sides`) and the brainstem comes first, else none is written; the sentence names the five largest lesions, largest first, and after the count of the others the places among them where no named lesion lies (the record keeps the order of the scores; both recommended to the user on 2026-09-29, open until confirmed); a lesion bound by the nearest rule farther than `NEAR_MM = 10` mm from every structure is written as not located (user, 2026-09-29). After the whole-branch review of 2026-09-29: a lesion that overlaps no structure is written as next to the nearest one; the side is written only before structures with a left and a right half; an empty record says that this model detected nothing at this threshold; records name their folds and the anatomy's source; `detections` refuses a probability map that does not belong to the label map. `check_grid`, the file check of `link_inputs` before its `mkdir` and the last two tests were added after the task review of 2026-09-29: `run()` compared only array shapes, so an anatomy of the same shape on another grid gave a wrong volume, structure and side without an error, and a mistyped image path left a partial output folder that blocked the next try. On the real data all 3887 channel files have exactly the affine of their case's SynthSeg map.

- [ ] **Step 5: Commit** — `git add anatobind/infer/brain_disease.py scripts/infer_brain_disease.py tests/test_brain_disease_record.py && git commit -m "Brain disease inference: scored components, binding, structured record and sentence per study"`

---

### Task 8: Evaluation module `anatobind/eval/brain_disease.py`

**Files:**
- Create: `anatobind/eval/brain_disease.py`
- Test: `tests/test_brain_disease_eval.py`

**Interfaces:**
- Consumes: Tasks 1, 4, 5, 6, 7.
- Produces: `EARLY_STOP = 0.3`, `N_FOLDS = 5`, `case_scan(job) -> {"case", "gt", "dets"}` with `job = (case, disease, label path, prediction nii, prediction npz, anatomy path, voxel_mm3)`, `jobs(results_root, raw_root, disease, splits, folds, info, anatomy_of) -> [job]`, `evaluate(scans) -> {"rows", "gate", "n_scans", "n_gt", "n_ignored"}`, `verdict(result, folds) -> {"kind", "folds", "pass", "operating_point", "sensitivity", "thr", "fp_per_scan", "stop_remaining_folds"}` (without an operating point `sensitivity` is None and `stop_remaining_folds` is False), `strata(scans, thr, key)`, `binding_agreement(scans, thr)`, `dice_summary(results_root, disease, folds)`.

- [ ] **Step 1: Write the failing test**

````python
# tests/test_brain_disease_eval.py
import json

import nibabel as nib
import numpy as np
import pytest

from anatobind.eval.brain_disease import (
    beyond_budget, binding_agreement, case_scan, code_version, dice_summary, evaluate, false_positive_spread, jobs, strata,
    verdict,
)
from anatobind.eval.lesion_components import size_stratum
from anatobind.nnunet.brain_disease import fold_dir

AFF = np.diag([1.0, 1.0, 1.0, 1.0])
SHAPE = (24, 24, 8)


def _save(path, data, affine=AFF):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def _case(tmp_path, name, gt_blocks, pred_blocks):
    """blocks: [(slices, probability)]; the anatomy is left white matter for x < 12 and right for x >= 12.
    Probabilities are binary fractions (0.875, 0.75, 0.625) so that float32 means are exact."""
    lab, pred = np.zeros(SHAPE, np.uint8), np.zeros(SHAPE, np.uint8)
    probs = np.zeros((2,) + SHAPE, np.float32)
    probs[0] = 1.0
    for sl, _ in gt_blocks:
        lab[sl] = 1
    for sl, p in pred_blocks:
        pred[sl] = 1
        probs[1][sl], probs[0][sl] = p, 1.0 - p
    seg = np.zeros(SHAPE, np.int16)
    seg[:12], seg[12:] = 2, 41
    d = tmp_path / name
    d.mkdir(parents=True)
    np.savez(d / "p.npz", probabilities=np.ascontiguousarray(probs.transpose(0, 3, 2, 1)))   # nnU-Net order (C, Z, Y, X)
    return (name, "glioma", _save(d / "lab.nii.gz", lab), _save(d / "pred.nii.gz", pred), d / "p.npz", _save(d / "seg.nii.gz", seg), 1.0)


BIG = (slice(2, 6), slice(2, 6), slice(2, 5))          # 48 voxels, left
SHIFTED = (slice(3, 7), slice(2, 6), slice(2, 5))      # box IoU with BIG: 36 / 60 = 0.6
SMALL = (slice(16, 18), slice(16, 18), slice(2, 4))    # 8 voxels = 8 mm3, right: under the floor
OTHER = (slice(16, 20), slice(4, 8), slice(2, 5))      # 48 voxels, right
STRAY = (slice(8, 11), slice(16, 19), slice(5, 8))     # 27 voxels, left


def test_case_scan_gives_bound_ground_truth_and_detections(tmp_path):
    s = case_scan(_case(tmp_path, "c", [(BIG, 1), (SMALL, 1)], [(SHIFTED, 0.875), (STRAY, 0.625)]))
    assert [(r["n_voxels"], r["ignore"], r["side"], r["host"]) for r in s["gt"]] == [
        (48, False, "left", "white_matter"), (8, True, "right", "white_matter")]
    assert [(r["n_voxels"], r["score"], r["side"]) for r in s["dets"]] == [(48, 0.875, "left"), (27, 0.625, "left")]


def test_case_scan_refuses_grids_that_differ(tmp_path):
    job = list(_case(tmp_path, "c", [(BIG, 1)], [(BIG, 0.875)]))
    job[5] = _save(tmp_path / "other_seg.nii.gz", np.zeros((24, 24, 9), np.int16))
    with pytest.raises(ValueError, match="c: label"):
        case_scan(tuple(job))


def _scans(tmp_path):
    return [case_scan(_case(tmp_path, "a", [(BIG, 1), (SMALL, 1)], [(SHIFTED, 0.875), (SMALL, 0.875), (STRAY, 0.625)])),
            case_scan(_case(tmp_path, "b", [(OTHER, 1)], [])),
            case_scan(_case(tmp_path, "n", [], [(STRAY, 0.75)]))]


def test_evaluate_counts_by_hand(tmp_path):
    r = evaluate(_scans(tmp_path))
    # Ground truth that counts: BIG (a) and OTHER (b) = 2; SMALL (8 mm3) is ignored. The prediction on SMALL is itself
    # under the volume floor and is dropped before matching. BIG is found by SHIFTED (score 0.875, box IoU 0.6) up to
    # threshold 0.85; OTHER is never found. False positives: the strays at 0.625 (a) and 0.75 (n).
    assert (r["n_scans"], r["n_gt"], r["n_ignored"]) == (3, 2, 1)
    by = {round(x["thr"], 2): x for x in r["rows"]}
    assert (by[0.05]["n_hit"], by[0.05]["n_fp"]) == (1, 2) and by[0.6]["n_fp"] == 2
    assert (by[0.65]["n_fp"], by[0.75]["n_fp"], by[0.8]["n_fp"]) == (1, 1, 0)
    assert (by[0.85]["n_hit"], by[0.9]["n_hit"]) == (1, 0)
    assert r["gate"] == {"pass": True, "thr": 0.85, "sensitivity_family": 0.5, "fp_per_scan": 0.0}


def test_verdict_is_a_gate_only_with_five_folds(tmp_path):
    r = evaluate(_scans(tmp_path))
    # the row just beyond the budget does not exist here: the strays never exceed 2 per scan
    assert verdict(r, [0]) == {"kind": "early_reading", "folds": [0], "pass": None, "operating_point": True,
                               "sensitivity": 0.5, "thr": 0.85, "fp_per_scan": 0.0, "beyond_budget": None,
                               "stop_remaining_folds": False, "early_stop_undecided": False}
    assert verdict(r, [4, 3, 2, 1, 0])["kind"] == "gate" and verdict(r, [0, 1, 2, 3, 4])["pass"] is True
    low = {"gate": {"pass": False, "thr": 0.55, "sensitivity_family": 0.29, "fp_per_scan": 1.0}}
    assert verdict(low, [0])["stop_remaining_folds"] is True and verdict(low, [0, 1, 2, 3, 4])["stop_remaining_folds"] is False


def test_strata_and_binding_agreement(tmp_path):
    scans = _scans(tmp_path)
    st = strata(scans, 0.5, lambda s, r: size_stratum(r["mm3"]))
    assert st == {"<5": {"n_gt": 2, "n_hit": 1, "sensitivity": 0.5}}            # 48 mm3 is a 4.5 mm sphere
    by_case = strata(scans, 0.5, lambda s, r: s["case"])
    assert by_case == {"a": {"n_gt": 1, "n_hit": 1, "sensitivity": 1.0}, "b": {"n_gt": 1, "n_hit": 0, "sensitivity": 0.0}}
    assert binding_agreement(scans, 0.5) == {"n_pairs": 1, "host_agreement": 1.0, "side_agreement": 1.0,
                                             "host_side_agreement": 1.0, "n_detections": 3, "no_host_rate": 0.0,
                                             "nearest_rate": 0.0, "unlocated_rate": 0.0}
    assert false_positive_spread(scans, 0.5) == {"n_scans": 3, "median": 1.0, "max": 1, "n_scans_over_budget": 0}
    assert false_positive_spread(scans, 0.7) == {"n_scans": 3, "median": 0.0, "max": 1, "n_scans_over_budget": 0}
    assert binding_agreement(scans, 0.95)["host_agreement"] is None


def test_jobs_name_a_missing_validation_file(tmp_path):
    splits = [{"train": ["b"], "val": ["a"]}, {"train": ["a"], "val": ["b"]}]
    info = {"a": {"voxel_mm3": 1.0}, "b": {"voxel_mm3": 2.0}}
    v = fold_dir(tmp_path / "res", "glioma", 1) / "validation"
    v.mkdir(parents=True)
    (v / "b.nii.gz").write_text("")
    with pytest.raises(FileNotFoundError, match="fold 1 case b: missing .*b.npz"):
        jobs(tmp_path / "res", tmp_path / "raw", "glioma", splits, [1], info, lambda c: tmp_path / f"{c}_seg.nii.gz")
    (v / "b.npz").write_text("")
    out = jobs(tmp_path / "res", tmp_path / "raw", "glioma", splits, [1], info, lambda c: tmp_path / f"{c}_seg.nii.gz")
    assert out == [("b", "glioma", tmp_path / "raw" / "Dataset904_PDGMGlioma" / "labelsTr" / "b.nii.gz", v / "b.nii.gz",
                    v / "b.npz", tmp_path / "b_seg.nii.gz", 2.0)]


def test_dice_summary_skips_cases_without_ground_truth(tmp_path):
    for f, cases in ((0, [(0.8, 10), (float("nan"), 0)]), (1, [(0.6, 5)])):
        v = fold_dir(tmp_path, "infarct", f) / "validation"
        v.mkdir(parents=True)
        (v / "summary.json").write_text(json.dumps({"metric_per_case": [{"metrics": {"1": {"Dice": d, "n_ref": n}}} for d, n in cases]}))
    assert dice_summary(tmp_path, "infarct", [0, 1]) == {"n_cases": 2, "mean": pytest.approx(0.7), "median": pytest.approx(0.7)}
    assert dice_summary(tmp_path, "infarct", [1])["n_cases"] == 1


def test_verdict_without_an_operating_point_measures_nothing_and_stops_nothing():
    none = {"gate": {"pass": False, "thr": None, "sensitivity_family": 0.0, "fp_per_scan": None}}   # what gate() returns
    assert verdict(none, [0]) == {"kind": "early_reading", "folds": [0], "pass": None, "operating_point": False,
                                  "sensitivity": None, "thr": None, "fp_per_scan": None, "beyond_budget": None,
                                  "stop_remaining_folds": False, "early_stop_undecided": True}
    full = verdict(none, [0, 1, 2, 3, 4])
    assert full["kind"] == "gate" and full["pass"] is False and full["sensitivity"] is None
    edge = {"gate": {"pass": False, "thr": 0.9, "sensitivity_family": 0.3, "fp_per_scan": 2.0}}
    assert verdict(edge, [0])["stop_remaining_folds"] is False      # the rule is "below 0.3"


def test_jobs_refuse_folds_given_twice_or_out_of_range(tmp_path):
    splits = [{"train": [], "val": ["a"]}] * 5
    for folds in ([0, 0], [5], [-1, 0]):
        with pytest.raises(ValueError, match="folds must be distinct and within 0..4"):
            jobs(tmp_path / "res", tmp_path / "raw", "glioma", splits, folds, {}, lambda c: tmp_path / c)


def _rows(*triples, n_gt=100):
    """(threshold, lesions found, false positives per scan) of a fold with n_gt counted lesions."""
    return [{"thr": t, "n_gt": n_gt, "n_hit_family": h, "sensitivity_family": h / n_gt, "fp_per_scan": f} for t, h, f in triples]


def test_an_early_stop_needs_the_row_beyond_the_budget_to_be_out_of_reach_too():
    low = {"pass": False, "thr": 0.95, "sensitivity_family": 0.27, "fp_per_scan": 1.6}
    # 0.90 exceeds the budget and finds 34 of 100: a threshold between the two rows may reach 0.3
    near = verdict({"gate": low, "rows": _rows((0.85, 40, 3.1), (0.90, 34, 2.4), (0.95, 27, 1.6))}, [0])
    assert near["beyond_budget"] == {"thr": 0.90, "sensitivity": 0.34, "fp_per_scan": 2.4, "n_hit": 34, "n_gt": 100,
                                     "out_of_reach": False}
    assert near["stop_remaining_folds"] is False and near["early_stop_undecided"] is True
    # 28 of 100 is two lesions short of 0.3: the matching is redone per threshold, so this is still within reach
    close = verdict({"gate": low, "rows": _rows((0.90, 28, 2.4), (0.95, 27, 1.6))}, [0])
    assert close["stop_remaining_folds"] is False and close["early_stop_undecided"] is True
    # 27 of 100 is three lesions short: out of reach, the reading is clear
    far = verdict({"gate": low, "rows": _rows((0.90, 27, 2.4), (0.95, 27, 1.6))}, [0])
    assert far["beyond_budget"]["out_of_reach"] is True
    assert far["stop_remaining_folds"] is True and far["early_stop_undecided"] is False
    # the budget is never exceeded: the operating point is the lowest threshold, nothing lies beyond it
    alone = verdict({"gate": dict(low, thr=0.05), "rows": _rows((0.05, 27, 1.6), (0.95, 20, 0.4))}, [0])
    assert alone["beyond_budget"] is None and alone["stop_remaining_folds"] is True
    full = verdict({"gate": low, "rows": _rows((0.90, 34, 2.4), (0.95, 27, 1.6))}, [0, 1, 2, 3, 4])
    assert full["stop_remaining_folds"] is False and full["early_stop_undecided"] is False and full["pass"] is False
    assert beyond_budget(_rows((0.5, 90, 2.0))) is None         # exactly the budget is within the budget
    # no operating point: always undecided; out_of_reach tells whether the reading is decisive in substance
    none = {"pass": False, "thr": None, "sensitivity_family": 0.0, "fp_per_scan": None}
    flood = verdict({"gate": none, "rows": _rows((0.90, 30, 4.0), (0.95, 20, 2.5))}, [0])
    assert flood["early_stop_undecided"] is True and flood["stop_remaining_folds"] is False
    assert flood["beyond_budget"]["thr"] == 0.95 and flood["beyond_budget"]["out_of_reach"] is True


def test_case_scan_counts_in_millimetres_on_a_grid_that_is_not_isotropic(tmp_path):
    aff = np.diag([1.0, 1.0, 2.5, 1.0])                          # 2.5 mm3 per voxel: the floor is 4 voxels
    lab, pred = np.zeros(SHAPE, np.uint8), np.zeros(SHAPE, np.uint8)
    lab[2:7, 2, 2] = 1                                           # 5 voxels = 12.5 mm3: counted
    lab[2:5, 10, 2] = 1                                          # 3 voxels = 7.5 mm3: ignored
    pred[2:7, 2, 2] = 1                                          # found
    pred[2:5, 20, 6] = 1                                         # 3 voxels: dropped
    pred[10:14, 12, 4] = 1                                       # 4 voxels = 10 mm3, on no structure
    probs = np.zeros((2,) + SHAPE, np.float32)
    probs[0] = 1.0
    probs[1][pred == 1], probs[0][pred == 1] = 0.875, 0.125
    seg = np.zeros(SHAPE, np.int16)
    seg[:8, :8] = 2                                              # left white matter around the lesion
    seg[10:14, 12, 1] = 3                                        # left cortex 3 voxels below the last detection: 7.5 mm
    seg[10:14, 17, 4] = 41                                       # right white matter 5 voxels beside it: 5 mm
    d = tmp_path / "c"
    np.savez(_save(d / "lab.nii.gz", lab, aff).parent / "p.npz", probabilities=np.ascontiguousarray(probs.transpose(0, 3, 2, 1)))
    s = case_scan(("c", "infarct", d / "lab.nii.gz", _save(d / "pred.nii.gz", pred, aff), d / "p.npz",
                   _save(d / "seg.nii.gz", seg, aff), 2.5))
    assert [(r["n_voxels"], r["mm3"], r["ignore"]) for r in s["gt"]] == [(5, 12.5, False), (3, 7.5, True)]
    assert sorted((r["n_voxels"], r["mm3"], r["host"], r["host_rule"], r["side"]) for r in s["dets"]) == [
        (4, 10.0, "white_matter", "nearest", "right"), (5, 12.5, "white_matter", "overlap", "left")]
    far = next(r for r in s["dets"] if r["host_rule"] == "nearest")
    assert far["host_distance_mm"] == 5.0 and far["host_sides"] == {"white_matter": "right"}
    # one of the two detections is bound by the nearest rule, 5 mm away: near enough to be located
    assert binding_agreement([s], 0.5)["nearest_rate"] == 0.5 and binding_agreement([s], 0.5)["unlocated_rate"] == 0.0


def test_the_two_side_agreements_are_counted_apart():
    box = (2, 2, 2, 6, 6, 5)
    gt = {"box": box, "family": "tumor", "host": "thalamus", "side": "right", "host_side": "left"}
    det = {"box": box, "family": "tumor", "score": 0.875, "host": "thalamus", "host_rule": "overlap", "side": "right",
           "host_side": "right", "host_distance_mm": 0.0}
    out = binding_agreement([{"case": "c", "gt": [gt], "dets": [det]}], 0.5)
    assert (out["n_pairs"], out["side_agreement"], out["host_side_agreement"]) == (1, 1.0, 0.0)


def test_the_code_version_names_the_commit_and_marks_changed_files(tmp_path):
    import subprocess
    assert code_version(tmp_path / "no_such_folder") == "unknown"
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q"], cwd=repo, check=True)
    (repo / "anatobind").mkdir()
    (repo / "anatobind" / "x.py").write_text("one\n")
    (repo / "STATUS.md").write_text("one\n")
    subprocess.run(git + ["add", "anatobind/x.py", "STATUS.md"], cwd=repo, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "one"], cwd=repo, check=True)
    clean = code_version(repo)
    assert len(clean) >= 7 and not clean.endswith("+")
    (repo / "anatobind" / "untracked.py").write_text("x\n")
    (repo / "STATUS.md").write_text("two\n")
    assert code_version(repo) == clean                         # untracked files and documents do not count
    (repo / "anatobind" / "x.py").write_text("two\n")
    assert code_version(repo) == clean + "+"
````

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_brain_disease_eval.py -q -p no:cacheprovider` → `ModuleNotFoundError: No module named 'anatobind.eval.brain_disease'`.

- [ ] **Step 3: Implement** `anatobind/eval/brain_disease.py`:

````python
"""Out-of-fold evaluation of the brain multi-disease detectors (spec 2026-09-29 §5, §6): ground-truth and predicted
components with their binding, the D1-style gate per disease, size strata, nnU-Net's Dice, and the agreement of the
binding between matched pairs (NOT_EVIDENCE)."""
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.bind.brain_lookup import BrainBinder
from anatobind.eval.detection_metrics import FP_MAX, gate, scan_matches, sweep
from anatobind.eval.lesion_boxes import load_label_map, load_nnunet_probabilities
from anatobind.eval.lesion_components import component_rows, components
from anatobind.infer.brain_disease import NEAR_MM, bind_rows, detections
from anatobind.nnunet.brain_disease import DISEASES, fold_dir

EARLY_STOP = 0.3
REACH_LESIONS = 2
N_FOLDS = 5


def code_version(repo):
    """The commit a report was made with: the short hash, with "+" when tracked code (anatobind, scripts) differs from
    it; "unknown" where git cannot tell. Read-only: git takes no lock and refreshes no index."""
    def git(*args):
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=str(repo), capture_output=True, text=True,
                              check=True, timeout=30).stdout.strip()
    try:
        head = git("rev-parse", "--short", "HEAD")
        changed = git("status", "--porcelain", "--untracked-files=no", "--", "anatobind", "scripts")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return head + ("+" if changed else "")


def case_scan(job):
    """(case, disease, label, prediction nii, prediction npz, anatomy, voxel_mm3) -> {"case", "gt", "dets"}; every
    row carries its binding."""
    case, disease, label_p, nii, npz, anatomy, voxel_mm3 = job
    family = DISEASES[disease]["type"]
    label, pred = load_label_map(label_p), load_label_map(nii)
    seg_img = nib.load(str(anatomy))
    if label.shape != pred.shape or label.shape != seg_img.shape:
        raise ValueError(f"{case}: label {label.shape}, prediction {pred.shape} and anatomy {seg_img.shape} differ")
    binder = BrainBinder(np.asarray(seg_img.dataobj), seg_img.header.get_zooms()[:3])
    comp, n = components(label)
    gt = bind_rows(component_rows(comp, n, voxel_mm3, family), comp, binder)
    dets, dcomp = detections(pred, load_nnunet_probabilities(npz, pred), voxel_mm3, family)
    return {"case": case, "gt": gt, "dets": bind_rows(dets, dcomp, binder)}


def jobs(results_root, raw_root, disease, splits, folds, info, anatomy_of):
    if len(set(folds)) != len(folds) or not set(folds) <= set(range(N_FOLDS)):
        raise ValueError(f"folds must be distinct and within 0..{N_FOLDS - 1}: {list(folds)}")
    out = []
    for f in folds:
        d = fold_dir(results_root, disease, f) / "validation"
        for case in splits[f]["val"]:
            nii, npz = d / f"{case}.nii.gz", d / f"{case}.npz"
            for p in (nii, npz):
                if not p.exists():
                    raise FileNotFoundError(f"fold {f} case {case}: missing {p}")
            out.append((case, disease, Path(raw_root) / DISEASES[disease]["name"] / "labelsTr" / f"{case}.nii.gz", nii, npz,
                        anatomy_of(case), info[case]["voxel_mm3"]))
    return out


def evaluate(scans):
    rows = sweep(scans)
    return {"rows": rows, "gate": gate(rows), "n_scans": len(scans), "n_gt": rows[0]["n_gt"],
            "n_ignored": sum(1 for s in scans for r in s["gt"] if r.get("ignore"))}


def beyond_budget(rows, fp_max=FP_MAX):
    """The row with the highest threshold whose false positives exceed the budget, or None. The grid is coarse: the
    threshold that just meets the budget lies between this row and the operating point."""
    over = [r for r in rows if r["fp_per_scan"] > fp_max]
    return max(over, key=lambda r: r["thr"]) if over else None


def within_reach(row):
    """Could a threshold near this row reach EARLY_STOP? Yes when the row reaches it or misses it by at most
    REACH_LESIONS lesions: the matching is redone at every threshold, so the hits need not fall as the threshold
    rises, and a threshold between two rows of the grid can find a lesion or two more than the lower row."""
    return row["n_hit_family"] + REACH_LESIONS >= EARLY_STOP * row["n_gt"]


def verdict(result, folds):
    """The gate when all five folds were scored; otherwise an early reading with the early-stop flag (spec M4).

    Without an operating point (no threshold keeps the false positives within the budget) no sensitivity was measured:
    it is None and the gate fails. An early reading stops the remaining folds only when it is clear: the sensitivity
    at the operating point is under EARLY_STOP, and the row just beyond the budget is out of reach of it too (see
    within_reach). Every other low reading is undecided: the remaining folds go on and the user decides. A reading
    without an operating point is always undecided; out_of_reach then says whether it is decisive in substance."""
    full = sorted(folds) == list(range(N_FOLDS))
    g = result["gate"]
    found = g["thr"] is not None
    sens = g["sensitivity_family"] if found else None
    b = beyond_budget(result.get("rows", []))
    low = found and sens < EARLY_STOP
    reach = b is not None and within_reach(b)
    return {"kind": "gate" if full else "early_reading", "folds": sorted(folds), "pass": g["pass"] if full else None,
            "operating_point": found, "sensitivity": sens, "thr": g["thr"], "fp_per_scan": g["fp_per_scan"],
            "beyond_budget": None if b is None else {"thr": b["thr"], "sensitivity": b["sensitivity_family"],
                                                     "fp_per_scan": b["fp_per_scan"], "n_hit": b["n_hit_family"],
                                                     "n_gt": b["n_gt"], "out_of_reach": not reach},
            "stop_remaining_folds": bool(not full and low and not reach),
            "early_stop_undecided": bool(not full and (not found or (low and reach)))}


def strata(scans, thr, key):
    """Sensitivity per stratum over the ground truth that is not ignored; key(scan, row) names the stratum."""
    out = {}
    for s in scans:
        hits, _, _ = scan_matches(s, thr)
        for g, r in enumerate(s["gt"]):
            if r.get("ignore"):
                continue
            e = out.setdefault(key(s, r), {"n_gt": 0, "n_hit": 0})
            e["n_gt"] += 1
            e["n_hit"] += int(g in hits)
    for e in out.values():
        e["sensitivity"] = e["n_hit"] / e["n_gt"]
    return out


def false_positive_spread(scans, thr):
    """How the false positives at thr spread over the scans: the budget is a mean, a few scans can carry it."""
    fps = [scan_matches(s, thr)[1] for s in scans]
    return {"n_scans": len(fps), "median": float(np.median(fps)) if fps else None, "max": int(max(fps)) if fps else None,
            "n_scans_over_budget": sum(1 for f in fps if f > FP_MAX)}


def binding_agreement(scans, thr):
    """NOT_EVIDENCE. Over matched (ground truth, detection) pairs: the share with the same main structure, with the
    same side of the whole lesion and with the same side of the main structure (the one the sentence writes); over all
    detections at thr: the share without any host, the share that overlaps no structure and was bound to the nearest
    one, and the share that the sentence calls not located (no host, or the nearest one beyond NEAR_MM)."""
    n = same_host = same_side = same_host_side = n_det = no_host = nearest = unlocated = 0
    for s in scans:
        hits, _, dets = scan_matches(s, thr)
        n_det += len(dets)
        no_host += sum(1 for d in dets if d["host"] is None)
        nearest += sum(1 for d in dets if d["host_rule"] == "nearest")
        unlocated += sum(1 for d in dets if d["host"] is None
                         or (d["host_rule"] == "nearest" and d["host_distance_mm"] > NEAR_MM))
        for g, p in hits.items():
            n += 1
            same_host += int(s["gt"][g]["host"] == dets[p]["host"])
            same_side += int(s["gt"][g]["side"] == dets[p]["side"])
            same_host_side += int(s["gt"][g]["host_side"] == dets[p]["host_side"])
    return {"n_pairs": n, "host_agreement": same_host / n if n else None, "side_agreement": same_side / n if n else None,
            "host_side_agreement": same_host_side / n if n else None,
            "n_detections": n_det, "no_host_rate": no_host / n_det if n_det else None,
            "nearest_rate": nearest / n_det if n_det else None, "unlocated_rate": unlocated / n_det if n_det else None}


def dice_summary(results_root, disease, folds):
    """Foreground Dice of the validation cases that have ground truth, from nnU-Net's own summary.json."""
    vals = []
    for f in folds:
        s = json.loads((fold_dir(results_root, disease, f) / "validation" / "summary.json").read_text())
        vals += [c["metrics"]["1"]["Dice"] for c in s["metric_per_case"] if c["metrics"]["1"]["n_ref"] > 0]
    return {"n_cases": len(vals), "mean": float(np.mean(vals)) if vals else None,
            "median": float(np.median(vals)) if vals else None}
````

- [ ] **Step 4: Run** the test → 13 passed (after the whole-branch review of 2026-09-29 the early stop also needs the row just beyond the budget to be under 0.3, every other low reading is `early_stop_undecided`; `false_positive_spread` and `nearest_rate` are report-only; the verdict's `operating_point`, the rule that a reading without an operating point stops no folds, and the check of the folds in `jobs` were added after the task review of 2026-09-29: `gate()` fills sensitivity 0.0 when no threshold keeps the false positives within budget, and the early reading took that filler for a measurement). The probabilities in the fixtures are binary fractions (0.875, 0.75, 0.625) on purpose: float32 means of other values land a hair above or below the threshold grid.

- [ ] **Step 5: Commit** — `git add anatobind/eval/brain_disease.py tests/test_brain_disease_eval.py && git commit -m "Brain disease evaluation: bound ground truth and detections per case, gate or early reading, strata, Dice and binding agreement"`

---

### Task 9: Evaluation script `scripts/eval_brain_disease.py`

**Files:**
- Create: `scripts/eval_brain_disease.py`
- Test: `tests/test_brain_disease_eval_script.py`

**Interfaces:**
- Consumes: Tasks 1, 4, 7, 8.
- Produces: `main(argv=None)`; module constants `FM`, `NNUNET`; writes `REPORT.md`, `verdict.json`, `froc.csv`, `output.txt` into `--out`, and with `--records DIR` one `<case>.json` per scored case.

- [ ] **Step 1: Write the failing test**

````python
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
        assert rep.rstrip().splitlines()[-1].startswith("Code: commit ")
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
````

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_brain_disease_eval_script.py -q -p no:cacheprovider` → FileNotFoundError on the script path.

- [ ] **Step 3: Implement** `scripts/eval_brain_disease.py`:

````python
#!/usr/bin/env python
# scripts/eval_brain_disease.py
"""Out-of-fold evaluation of one brain disease detector (spec 2026-09-29 §5, §6).

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_disease.py \
      --disease glioma --folds 0 --out docs/verification/2026-09-30/brain_multidisease/glioma_fold0
  ... --disease glioma --folds 0 1 2 3 4 --out DIR --records /data2/.../derived/brain_disease/glioma/records
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.brain_disease import (  # noqa: E402
    binding_agreement, case_scan, code_version, dice_summary, evaluate, false_positive_spread, jobs, strata, verdict,
)
from anatobind.eval.lesion_components import MIN_MM3, STRATA, size_stratum  # noqa: E402
from anatobind.infer.brain_disease import study_record  # noqa: E402
from anatobind.nnunet.brain_disease import DISEASES, FM, anatomy_path  # noqa: E402

NNUNET = FM / "derived/nnunet"


def table(title, st, order):
    out = [f"### {title}\n\n| stratum | N (GT) | N (hit) | sensitivity |\n|---|---|---|---|\n"]
    out += [f"| {k} | {st[k]['n_gt']} | {st[k]['n_hit']} | {st[k]['sensitivity']:.4f} |\n" for k in order if k in st]
    return out + ["\n"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Out-of-fold evaluation of one brain disease detector")
    ap.add_argument("--disease", choices=sorted(DISEASES), required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True, help="report directory (must not exist)")
    ap.add_argument("--records", type=Path, help="directory for one structured record per case (must not exist)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    for p in (a.out, a.records):
        if p is not None and p.exists():
            raise FileExistsError(f"{p} already exists")
    name = DISEASES[a.disease]["name"]
    info = json.loads((NNUNET / "raw" / name / "cases.json").read_text())
    splits = json.loads((NNUNET / "preprocessed" / name / "splits_final.json").read_text())
    val = sorted(c for s in splits for c in s["val"])
    if val != sorted(info):
        raise AssertionError(f"{name}: the folds do not cover the {len(info)} cases exactly once")
    todo = jobs(NNUNET / "results", NNUNET / "raw", a.disease, splits, a.folds, info, lambda c: anatomy_path(a.disease, c, FM))
    if a.workers > 1:
        with Pool(a.workers) as pool:
            scans = pool.map(case_scan, todo, chunksize=2)
    else:
        scans = [case_scan(j) for j in todo]
    result = evaluate(scans)
    if result["n_scans"] != sum(len(splits[f]["val"]) for f in a.folds):
        raise AssertionError(f"scored {result['n_scans']} scans, the folds hold {sum(len(splits[f]['val']) for f in a.folds)}")
    v = verdict(result, a.folds)
    thr = v["thr"]
    dice = dice_summary(NNUNET / "results", a.disease, a.folds)

    L = [f"# Brain multi-disease detector: {a.disease} ({name}), folds {sorted(a.folds)}\n\n",
         ("All five folds: the line below is the gate (spec M2).\n\n" if v["kind"] == "gate" else
          "Fold subset: an early reading, NOT the gate (spec M4).\n\n"),
         "## Verdict\n\n```json\n", json.dumps(v, indent=1), "\n```\n\n",
         ("" if thr is not None else "No threshold of the grid keeps the false positives at or below 2 per scan: there is no "
          "operating point, so no sensitivity was measured and strata, binding agreement and records are not produced. "
          "With all five folds this fails the gate; an early reading without an operating point does not stop the "
          "remaining folds by itself (the rule of spec M4 needs a measured sensitivity): that decision is the user's.\n\n"),
         ("" if not (thr is None and v["beyond_budget"] and v["beyond_budget"]["out_of_reach"]) else
          f"In substance this reading is decisive all the same: the row at threshold {v['beyond_budget']['thr']:.2f} still "
          f"exceeds the budget ({v['beyond_budget']['fp_per_scan']:.4f} false positives per scan) and finds "
          f"{v['beyond_budget']['n_hit']} of {v['beyond_budget']['n_gt']} lesions; a threshold that meets the budget lies "
          "above it and is not expected to find more.\n\n"),
         ("" if not (v["early_stop_undecided"] and thr is not None) else
          f"The sensitivity at the operating point is under 0.3, but the row just beyond the budget (threshold "
          f"{v['beyond_budget']['thr']:.2f}: {v['beyond_budget']['n_hit']} of {v['beyond_budget']['n_gt']} lesions at "
          f"{v['beyond_budget']['fp_per_scan']:.4f} false positives per scan) reaches 0.3 or misses it by at most two "
          "lesions, and the matching is redone at every threshold: a threshold between the two rows of the grid may reach "
          "0.3. The early stop is undecided, the remaining folds go on, the decision is the user's.\n\n"),
         f"Scans {result['n_scans']}; ground-truth lesions counted {result['n_gt']}; ignored (< {MIN_MM3:g} mm3) "
         f"{result['n_ignored']}.\n\n",
         "Scores are mean foreground probabilities over components of the argmax map, so they exceed 0.5 by construction: "
         "the rows up to threshold 0.50 are identical and the budget of 2 false positives per scan may not be reached.\n\n",
         "## FROC\n\n| thr | n_hit | sensitivity | FP per scan |\n|---|---|---|---|\n"]
    L += [f"| {r['thr']:.2f} | {r['n_hit']} | {r['sensitivity']:.4f} | {r['fp_per_scan']:.4f} |\n" for r in result["rows"]]
    L += ["\n## Dice (report only, nnU-Net summary.json, cases with ground truth)\n\n```json\n", json.dumps(dice, indent=1), "\n```\n\n"]
    if thr is not None:
        L += ["## False positives per scan at the operating threshold (report only)\n\n```json\n",
              json.dumps(false_positive_spread(scans, thr), indent=1), "\n```\n\n"]
        L += ["## Strata at the operating threshold (report only)\n\n"]
        L += table("Equivalent diameter (mm)", strata(scans, thr, lambda s, r: size_stratum(r["mm3"])), STRATA)
        if a.disease == "metastasis":
            L += table("Prior craniotomy, biopsy or resection",
                       strata(scans, thr, lambda s, r: "yes" if info[s["case"]].get("prior_surgery") else "no"), ("no", "yes"))
        L += ["## Binding agreement (NOT_EVIDENCE: the anatomy is a SynthSeg pseudo-label)\n\n```json\n",
              json.dumps(binding_agreement(scans, thr), indent=1), "\n```\n\n"]
    L += ["## Command\n\n```\n", " ".join(sys.argv), "\n```\n\nCode: commit ", code_version(Path(__file__).resolve().parents[1]), "\n"]
    fold_of = {c: f for f in a.folds for c in splits[f]["val"]}
    records = {}
    if a.records is not None and thr is not None:
        records = {s["case"]: json.dumps(study_record(s["case"], a.disease, thr, s["dets"], [fold_of[s["case"]]]),
                                         ensure_ascii=False, indent=1) for s in scans}
    a.out.mkdir(parents=True)                                   # everything is computed: only now is anything written
    (a.out / "REPORT.md").write_text("".join(L))
    (a.out / "verdict.json").write_text(json.dumps(v, indent=1))
    (a.out / "froc.csv").write_text("thr,n_hit,sensitivity,fp_per_scan\n" + "".join(
        f"{r['thr']:.2f},{r['n_hit']},{r['sensitivity']:.6f},{r['fp_per_scan']:.6f}\n" for r in result["rows"]))
    out = [f"Disease: {a.disease}; folds {sorted(a.folds)}; kind {v['kind']}\n",
           f"Scans: {result['n_scans']}; lesions counted: {result['n_gt']}; ignored: {result['n_ignored']}\n",
           (f"Sensitivity {v['sensitivity']:.4f} at threshold {thr} with {v['fp_per_scan']} FP per scan\n" if thr is not None
            else "No threshold keeps the false positives at or below 2 per scan: no operating point\n"),
           f"Pass: {v['pass']}; stop remaining folds: {v['stop_remaining_folds']}; "
           f"early stop undecided: {v['early_stop_undecided']}\n",
           f"Dice mean {dice['mean']} over {dice['n_cases']} cases\n"]
    (a.out / "output.txt").write_text("".join(out))
    if records:
        a.records.mkdir(parents=True)
        for case, text in records.items():
            (a.records / f"{case}.json").write_text(text)
    print("".join(out), end="")


if __name__ == "__main__":
    main()
````

- [ ] **Step 4: Run** `tests/test_brain_disease_eval_script.py tests/test_brain_disease_eval.py` → all pass (6 new). The report folder is created only when everything is computed (after the task review of 2026-09-29: a failure between the `mkdir` and the first file would have left an empty folder, which cannot be deleted here and blocks the next try). While trainings run, pass `--workers 2`: the default of 8 does not fit the CPU rule beside six trainings.

- [ ] **Step 5: Commit** — `git add scripts/eval_brain_disease.py tests/test_brain_disease_eval_script.py && git commit -m "Brain disease evaluation script: verdict, FROC, strata, Dice, binding agreement and per-study records"`

---

### Task 10: Cross false-alarm script `scripts/brain_disease_crossrun.py`

**Files:**
- Create: `scripts/brain_disease_crossrun.py`
- Test: `tests/test_brain_disease_crossrun.py`

**Interfaces:**
- Consumes: Tasks 1, 4, 7.
- Produces: `CROSS = {(model, data): channels}` (three pairs), `NOTES`, `overlap_counts(dets, det_comp, gt_rows, gt_comp, thr) -> {"n_det", "n_det_on_gt", "n_gt", "n_gt_claimed"}`, `summarise(per_case)`, `main(argv=None)` (checks the model folds' `checkpoint_final.pth` and every channel file before it creates `--work`; the summary names its cases).

- [ ] **Step 1: Write the failing test**

````python
# tests/test_brain_disease_crossrun.py
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from anatobind.eval.lesion_components import component_rows, components
from anatobind.infer.brain_disease import detections
from anatobind.nnunet.brain_disease import DISEASES, fold_dir


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/brain_disease_crossrun.py"
    spec = importlib.util.spec_from_file_location("brain_disease_crossrun", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_only_the_three_pairs_with_shared_sequences_exist():
    assert set(_load().CROSS) == {("infarct", "glioma"), ("metastasis", "glioma"), ("glioma", "metastasis")}


def _volumes():
    gt = np.zeros((20, 20, 6), np.uint8)
    gt[2:6, 2:6, 1:4] = 1                 # 48 voxels: a counted lesion, touched by a detection
    gt[12:16, 12:16, 1:4] = 1             # 48 voxels: a counted lesion, untouched
    gt[18, 18, 5] = 1                     # 1 voxel: ignored
    pred = np.zeros((20, 20, 6), np.uint8)
    probs = np.zeros((2, 20, 20, 6), np.float32)
    for sl, p in (((slice(4, 8), slice(4, 8), slice(1, 4)), 0.875),      # overlaps the first lesion
                  ((slice(8, 11), slice(0, 3), slice(3, 6)), 0.75),      # on no ground truth
                  ((slice(16, 19), slice(2, 5), slice(0, 3)), 0.625)):   # on no ground truth, under the threshold
        pred[sl] = 1
        probs[1][sl] = p
    return gt, pred, probs


def test_overlap_counts_by_hand():
    c = _load()
    gt, pred, probs = _volumes()
    dets, det_comp = detections(pred, probs, 1.0, "infarct")
    gt_comp, n = components(gt)
    rows = component_rows(gt_comp, n, 1.0, "tumor")
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.7) == {"n_det": 2, "n_det_on_gt": 1, "n_gt": 2, "n_gt_claimed": 1}
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.5) == {"n_det": 3, "n_det_on_gt": 1, "n_gt": 2, "n_gt_claimed": 1}
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.9) == {"n_det": 0, "n_det_on_gt": 0, "n_gt": 2, "n_gt_claimed": 0}
    # a score equal to the threshold is kept
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.75) == {"n_det": 2, "n_det_on_gt": 1, "n_gt": 2, "n_gt_claimed": 1}
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.875)["n_det"] == 1


def test_summary_rates_and_empty_inputs():
    c = _load()
    s = c.summarise({"a": {"n_det": 2, "n_det_on_gt": 1, "n_gt": 2, "n_gt_claimed": 1},
                     "b": {"n_det": 0, "n_det_on_gt": 0, "n_gt": 1, "n_gt_claimed": 0}})
    assert s == {"n_scans": 2, "n_det": 2, "n_det_on_gt": 1, "n_gt": 3, "n_gt_claimed": 1, "det_per_scan": 1.0,
                 "scans_with_any_detection": 1, "share_of_detections_on_gt": 0.5, "share_of_gt_claimed": pytest.approx(1 / 3)}
    empty = c.summarise({"a": {"n_det": 0, "n_det_on_gt": 0, "n_gt": 0, "n_gt_claimed": 0}})
    assert empty["share_of_detections_on_gt"] is None and empty["share_of_gt_claimed"] is None


def test_an_impossible_pair_and_an_existing_output_are_refused(tmp_path):
    c = _load()
    with pytest.raises(SystemExit, match="no cross run"):
        c.main(["--model", "infarct", "--data", "metastasis", "--threshold", "0.5", "--gpu", "0",
                "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    (tmp_path / "o").mkdir()
    with pytest.raises(FileExistsError):
        c.main(["--model", "infarct", "--data", "glioma", "--threshold", "0.5", "--gpu", "0",
                "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    assert not (tmp_path / "w").exists()


def test_the_channels_of_each_pair_follow_the_model_by_meaning():
    c = _load()
    assert DISEASES["infarct"]["channels"] == ("DWI", "ADC") and c.CROSS[("infarct", "glioma")] == ("DWI", "ADC")
    assert DISEASES["metastasis"]["channels"] == ("T1pre", "T1post", "FLAIR")
    assert c.CROSS[("metastasis", "glioma")] == ("T1", "T1c", "FLAIR")                  # glioma's names for the same three
    assert DISEASES["glioma"]["channels"] == ("T1", "T1c", "T2", "FLAIR")
    assert c.CROSS[("glioma", "metastasis")] == ("T1pre", "T1post", "T2Synth", "FLAIR")   # the T2 is synthetic
    assert set(c.NOTES) == {("glioma", "metastasis")}


def test_overlap_counts_each_object_once_and_a_fragment_is_lesion_tissue():
    c = _load()
    gt = np.zeros((24, 20, 6), np.uint8)
    gt[2:6, 2:6, 1:4] = 1                 # lesion A, 48 voxels
    gt[2:6, 10:14, 1:4] = 1               # lesion B, 48 voxels
    gt[20, 16, 4] = 1                     # a fragment of 1 voxel: ignored
    pred = np.zeros((24, 20, 6), np.uint8)
    probs = np.zeros((2, 24, 20, 6), np.float32)
    for sl, p in (((slice(2, 4), slice(2, 4), slice(1, 4)), 0.875),       # on A
                  ((slice(5, 8), slice(4, 8), slice(1, 4)), 0.875),       # on A again (touches its corner)
                  ((slice(19, 22), slice(15, 18), slice(3, 6)), 0.75)):   # on the fragment only
        pred[sl] = 1
        probs[1][sl] = p
    dets, det_comp = detections(pred, probs, 1.0, "infarct")
    gt_comp, n = components(gt)
    rows = component_rows(gt_comp, n, 1.0, "tumor")
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.5) == {"n_det": 3, "n_det_on_gt": 3, "n_gt": 2, "n_gt_claimed": 1}
    wide = np.zeros((24, 20, 6), np.uint8)
    wide[3:5, 3:12, 1:4] = 1              # one detection across A and B
    wp = np.zeros((2, 24, 20, 6), np.float32)
    wp[1][wide == 1] = 0.875
    dets, det_comp = detections(wide, wp, 1.0, "infarct")
    assert c.overlap_counts(dets, det_comp, rows, gt_comp, 0.5) == {"n_det": 1, "n_det_on_gt": 1, "n_gt": 2, "n_gt_claimed": 2}


def test_nothing_is_written_when_a_model_fold_or_a_channel_is_missing(tmp_path):
    c = _load()
    nn = tmp_path / "derived/nnunet"
    args = ["--model", "infarct", "--data", "glioma", "--threshold", "0.5", "--gpu", "0", "--folds", "0",
            "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")]
    with patch.object(c, "FM", tmp_path), patch.object(c, "NNUNET", nn):
        with pytest.raises(FileNotFoundError, match="the infarct model has no trained fold 0"):
            c.main(args)
        ckpt = fold_dir(nn / "results", "infarct", 0) / "checkpoint_final.pth"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_text("")
        host = DISEASES["glioma"]["name"]
        for sub in ("raw", "preprocessed"):
            (nn / sub / host).mkdir(parents=True)
        (nn / "raw" / host / "cases.json").write_text(json.dumps({"UCSF-PDGM-0004": {"voxel_mm3": 1.0}}))
        (nn / "preprocessed" / host / "splits_final.json").write_text(json.dumps([{"train": [], "val": ["UCSF-PDGM-0004"]}]))
        with pytest.raises(FileNotFoundError, match="UCSF-PDGM-0004: missing .*DWI"):
            c.main(args)
    assert not (tmp_path / "w").exists() and not (tmp_path / "o").exists()
````

- [ ] **Step 2: Run to verify it fails** — `… -m pytest tests/test_brain_disease_crossrun.py -q -p no:cacheprovider` → FileNotFoundError on the script path.

- [ ] **Step 3: Implement** `scripts/brain_disease_crossrun.py`:

````python
#!/usr/bin/env python
# scripts/brain_disease_crossrun.py
"""Cross false-alarm check (spec 2026-09-29 M11, report only): run one disease's detector on another disease's data.

Only three pairs have the sequences: infarct model on glioma data (PDGM has DWI and ADC), metastasis model on glioma
data (T1, T1c, FLAIR), glioma model on metastasis data (BMSR's T2 is synthetic). The cases are the fold 0 validation
cases of the data's own split; the model never saw that dataset.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_disease_crossrun.py \
      --model infarct --data glioma --threshold 0.55 --gpu 4 \
      --work /data2/congcong/data/FM_data/derived/brain_disease/crossrun/infarct_on_glioma \
      --out docs/verification/2026-09-30/brain_multidisease/crossrun/infarct_on_glioma
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.brain_disease import code_version  # noqa: E402
from anatobind.eval.lesion_boxes import load_label_map, load_nnunet_probabilities  # noqa: E402
from anatobind.eval.lesion_components import component_mask, component_rows, components  # noqa: E402
from anatobind.infer.brain_disease import detections, run_nnunet  # noqa: E402
from anatobind.nnunet.brain_disease import DISEASES, FM, channel_path, fold_dir  # noqa: E402

NNUNET = FM / "derived/nnunet"
CROSS = {("infarct", "glioma"): ("DWI", "ADC"),
         ("metastasis", "glioma"): ("T1", "T1c", "FLAIR"),
         ("glioma", "metastasis"): ("T1pre", "T1post", "T2Synth", "FLAIR")}
NOTES = {("glioma", "metastasis"): "channel 2 (T2) is BMSR's synthetic T2"}


def overlap_counts(dets, det_comp, gt_rows, gt_comp, thr):
    """Detections at thr against another disease's ground truth: how many detections touch any ground-truth voxel, and
    how many counted ground-truth lesions are touched by any detection.

    A labelled fragment under the volume floor is lesion tissue: a detection on it is on ground truth. It is no counted
    lesion, so it is never in n_gt and cannot be claimed."""
    kept = [d for d in dets if d["score"] >= thr]
    gt_any = gt_comp > 0
    det_any = np.zeros(gt_comp.shape, bool)
    touching = 0
    for d in kept:
        sl, m = component_mask(det_comp, d)
        det_any[sl] |= m
        touching += int((gt_any[sl] & m).any())
    real = [r for r in gt_rows if not r["ignore"]]
    claimed = 0
    for r in real:
        sl, m = component_mask(gt_comp, r)
        claimed += int((det_any[sl] & m).any())
    return {"n_det": len(kept), "n_det_on_gt": touching, "n_gt": len(real), "n_gt_claimed": claimed}


def summarise(per_case):
    tot = {k: sum(c[k] for c in per_case.values()) for k in ("n_det", "n_det_on_gt", "n_gt", "n_gt_claimed")}
    n = len(per_case)
    return {"n_scans": n, **tot, "det_per_scan": tot["n_det"] / n if n else None,
            "scans_with_any_detection": sum(1 for c in per_case.values() if c["n_det"]),
            "share_of_detections_on_gt": tot["n_det_on_gt"] / tot["n_det"] if tot["n_det"] else None,
            "share_of_gt_claimed": tot["n_gt_claimed"] / tot["n_gt"] if tot["n_gt"] else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one disease's detector on another disease's data (report only)")
    ap.add_argument("--model", choices=sorted(DISEASES), required=True)
    ap.add_argument("--data", choices=sorted(DISEASES), required=True)
    ap.add_argument("--threshold", type=float, required=True, help="the model's operating threshold")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="model folds to ensemble")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--work", type=Path, required=True, help="inputs and predictions (must not exist)")
    ap.add_argument("--out", type=Path, required=True, help="report directory (must not exist)")
    a = ap.parse_args(argv)
    if (a.model, a.data) not in CROSS:
        raise SystemExit(f"no cross run for the {a.model} model on {a.data} data; possible: {sorted(CROSS)}")
    for p in (a.work, a.out):
        if p.exists():
            raise FileExistsError(f"{p} already exists")
    for f in a.folds:
        ckpt = fold_dir(NNUNET / "results", a.model, f) / "checkpoint_final.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"the {a.model} model has no trained fold {f}: missing {ckpt}")
    host = DISEASES[a.data]["name"]
    info = json.loads((NNUNET / "raw" / host / "cases.json").read_text())
    cases = json.loads((NNUNET / "preprocessed" / host / "splits_final.json").read_text())[0]["val"]
    srcs = {case: [channel_path(a.data, case, ch, FM) for ch in CROSS[(a.model, a.data)]] for case in cases}
    for case, paths in srcs.items():
        for src in paths:
            if not src.is_file():
                raise FileNotFoundError(f"{case}: missing {src}")
    (a.work / "input").mkdir(parents=True)
    for case, paths in srcs.items():
        for k, src in enumerate(paths):
            os.symlink(src.resolve(), a.work / "input" / f"{case}_{k:04d}.nii.gz")
    run_nnunet(DISEASES[a.model]["id"], a.work / "input", a.work / "pred", a.folds, a.gpu)
    per_case = {}
    for case in cases:
        pred = load_label_map(a.work / "pred" / f"{case}.nii.gz")
        gt = load_label_map(NNUNET / "raw" / host / "labelsTr" / f"{case}.nii.gz")
        if pred.shape != gt.shape:
            raise ValueError(f"{case}: prediction {pred.shape} and ground truth {gt.shape} differ")
        vox = info[case]["voxel_mm3"]
        dets, det_comp = detections(pred, load_nnunet_probabilities(a.work / "pred" / f"{case}.npz", pred), vox, DISEASES[a.model]["type"])
        gt_comp, n = components(gt)
        per_case[case] = overlap_counts(dets, det_comp, component_rows(gt_comp, n, vox, DISEASES[a.data]["type"]), gt_comp, a.threshold)
    s = {"model": a.model, "data": a.data, "cases": f"fold 0 validation cases of {host}",
         "channels": list(CROSS[(a.model, a.data)]), "note": NOTES.get((a.model, a.data)),
         "threshold": a.threshold, "model_folds": sorted(a.folds), **summarise(per_case)}
    version = code_version(Path(__file__).resolve().parents[1])
    a.out.mkdir(parents=True)
    (a.out / "crossrun.json").write_text(json.dumps({"summary": s, "per_case": per_case}, indent=1))
    (a.out / "REPORT.md").write_text("".join([
        f"# Cross false-alarm check: {a.model} model on {a.data} data (report only, spec M11)\n\n",
        "The model never saw this dataset. Resolution, preprocessing and scanners differ from its training data, so these "
        "numbers describe this pair of datasets, not the diseases in general.\n\n",
        "Counting: a detection is on ground truth when it shares a voxel with any labelled voxel, fragments under "
        "10 mm3 included; a ground-truth lesion is claimed when a detection shares a voxel with it, and only lesions of "
        "at least 10 mm3 are counted. These are counts of overlap: not a sensitivity, not a precision and not a "
        "false-positive rate.\n\n"
        "Threshold: it is the model's operating threshold, measured on single-fold models (every case predicted by the "
        "fold that held it out). Here the model's folds are averaged; the behaviour of the averaged model at this "
        "threshold was not measured on its own data.\n\n"
        "```json\n", json.dumps(s, indent=1),
        "\n```\n\n## Command\n\n```\n", " ".join(sys.argv), "\n```\n\nCode: commit ", version, "\n"]))
    print(json.dumps(s))


if __name__ == "__main__":
    main()
````

- [ ] **Step 4: Run** the test → 7 passed; then the full suite → 831 passed, 1 skipped. (After the task review of 2026-09-29: the channel tuples are pinned by a test, the report states its counting rule and its cases, and every check comes before the first `mkdir`, because a partial folder cannot be deleted here.)

- [ ] **Step 5: Commit** — `git add scripts/brain_disease_crossrun.py tests/test_brain_disease_crossrun.py && git commit -m "Brain disease cross runs: one disease's detector on another disease's data, report only"`

---

### Task 11: Fold 0 early reading (controller, once per disease)

**Precondition:** the disease's fold 0 has finished: `<results>/<Dataset>/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation/summary.json` exists and the queue log shows `finished ('<disease>', 0) … exit code 0`.

- [ ] **Step 1:** `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_disease.py --disease <disease> --folds 0 --workers 2 --out docs/verification/2026-09-29/brain_multidisease/<disease>_fold0` (`--workers 2` while trainings run)
- [ ] **Step 2:** read `verdict.json`. If `early_stop_undecided` is true (no operating point, or a sensitivity under 0.3 at the operating point while the row just beyond the budget reaches 0.3), write no skip file: report the FROC table and the two rows to the user, who decides whether the remaining folds go on (they go on meanwhile). If `stop_remaining_folds` is true: `test ! -e logs/brain_disease/skip_<id> && echo "fold 0 sensitivity <value> < 0.3 ($(date '+%F %T'))" > logs/brain_disease/skip_<id>`; folds of that disease already running are left to finish (nobody's process is signalled, ours included, unless the user says so).
- [ ] **Step 3:** commit the reading — `git add docs/verification/2026-09-29/brain_multidisease/<disease>_fold0 && git commit -m "Brain disease <disease>: fold 0 early reading (not the gate)"` — and report it to the user: sensitivity, threshold, false positives per scan, Dice, the size strata, and the line "this is an early reading, not the gate".

---

### Task 12: Five-fold evaluation, records, cross runs, inference smoke, summary

**Precondition:** the queue log ends with `queue empty, nothing running: done`; it holds a line `finished … with exit code 0` for every fold that is evaluated, and every fold that is missing is named in the README with the reason; every trained fold has `validation/summary.json`. At the first non-zero exit code during the run the controller reads that job's log, and creates `logs/brain_disease/stop` if the cause could hit every start.

**Files:**
- Create: `docs/verification/2026-09-29/brain_multidisease/{glioma,metastasis,infarct}/`, `crossrun/<model>_on_<data>/`, `infer_smoke.md`, `README.md`

- [ ] **Step 1: Evaluate each disease on its five folds** (a disease stopped by M4 is evaluated on the folds it has, which the report labels an early reading):

```bash
for d in glioma metastasis infarct; do PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_disease.py --disease $d --folds 0 1 2 3 4 --out docs/verification/2026-09-29/brain_multidisease/$d --records /data2/congcong/data/FM_data/derived/brain_disease/$d/records; done
```

Expected scans: 501, 461, 250. The gate line of each `verdict.json` is the result, pass or fail (M5: no tuning).

- [ ] **Step 2: Cross runs** (one idle GPU each, picked with nvidia-smi right before; thresholds from each model's `verdict.json`):

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_disease_crossrun.py --model infarct --data glioma --threshold <infarct thr> --gpu <idle> --work /data2/congcong/data/FM_data/derived/brain_disease/crossrun/infarct_on_glioma --out docs/verification/2026-09-29/brain_multidisease/crossrun/infarct_on_glioma
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_disease_crossrun.py --model metastasis --data glioma --threshold <metastasis thr> --gpu <idle> --work /data2/congcong/data/FM_data/derived/brain_disease/crossrun/metastasis_on_glioma --out docs/verification/2026-09-29/brain_multidisease/crossrun/metastasis_on_glioma
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_disease_crossrun.py --model glioma --data metastasis --threshold <glioma thr> --gpu <idle> --work /data2/congcong/data/FM_data/derived/brain_disease/crossrun/glioma_on_metastasis --out docs/verification/2026-09-29/brain_multidisease/crossrun/glioma_on_metastasis
```

A model of a disease stopped by M4 runs with `--folds` set to the folds it has.

- [ ] **Step 3: Inference smoke**, one fold 0 validation case per disease with the fold 0 model only (the first case of `splits_final.json[0]["val"]`), e.g. for infarct:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_disease.py --disease infarct --images <case DWI> <case ADC> --anatomy <case SynthSeg seg> --threshold <infarct thr> --out /data2/congcong/data/FM_data/derived/brain_disease/infer_smoke/infarct --folds 0 --gpu <idle>
```

Write `infer_smoke.md`: the three commands, the printed lines, each `record.json` sentence, and a check that the record's lesions equal the same case's out-of-fold record from Step 1 when thresholded the same way (same fold 0 model, so boxes and scores must agree to 1e-4).

- [ ] **Step 4: `README.md`** of the folder: the verdict of each disease in one table beside the small-lesion detectors (nnU-Net 2d 0.366 at 1.64 FP per volume from `docs/verification/2026-09-28/brain_detector/README.md`; nnDetection fold 0 from `docs/verification/2026-09-29/brain_nndet/fold0/` if it exists); the cross-run summaries; the binding agreement with its NOT_EVIDENCE label; known deviations (cases without label voxels, ignored fragments per disease, the synthetic T2 of the glioma-on-metastasis run); training and evaluation wall clock read from the logs. Every number is copied from the files of this folder, none typed from memory.

- [ ] **Step 5: Commit** — `git add docs/verification/2026-09-29/brain_multidisease && git commit -m "Brain disease detectors: five-fold verdicts, cross runs, inference smoke and summary"`

- [ ] **Step 6 (controller):** report the three verdicts to the user. A failed disease stops there; whether it gets an nnDetection second arm is the user's decision (M5).

---

### Task 13: Documentation, final review, merge

- [ ] **Step 1:** full suite → record the count.
- [ ] **Step 2:** `CLAUDE.md`: add a code-map paragraph after the S2 one: "`anatobind/nnunet/brain_disease.py`（S7 三个病种的 nnU-Net 数据集 904/905/906）、`anatobind/eval/lesion_components.py`（连通块与 10 mm³ 规则）、`anatobind/eval/brain_disease.py`（折外评估、达标线、分层、绑定一致率）、`anatobind/bind/brain_lookup.py`（主结构、占比、侧别）、`anatobind/infer/brain_disease.py`（推理与结构化记录）；脚本 `scripts/brain_disease_{prepare,crossrun}.py`、`gpu_queue.py`、`eval_brain_disease.py`、`infer_brain_disease.py`；规格 `docs/superpowers/specs/2026-09-29-brain-multidisease-design.md`。" Update the status sentence and the test count. Change nothing else.
- [ ] **Step 3:** `STATUS.md` rewritten with the five fixed sections.
- [ ] **Step 4:** commit — `git add CLAUDE.md STATUS.md && git commit -m "Docs: brain multi-disease detectors status and code map"`.
- [ ] **Step 5:** whole-branch review on the most capable model; fix; re-review; the controller merges to main and tags `handoff/<date>-brain-multidisease` (no push).

---

## Self-review notes

- Spec coverage: §3 → Tasks 1–2; §4 → Task 3 (timing read from the first epochs of the real trainings instead of a separate 5-epoch run; the spec is amended in this commit); §5 → Tasks 4, 5, 8, 9, 10, 11, 12; §6 → Tasks 6, 7, 8, 9, 12; §7 → Tasks 7, 12; §8 layout → all; §9 tests → Tasks 1–10; §10–§11 → Task 13; M4 early stop → Tasks 3, 11; M11 → Tasks 10, 12; M14 → Task 3 (cap fixed at 6).
- Types: boxes are half-open `(x0, y0, z0, x1, y1, z1)` in voxel indices of the case's own grid everywhere; rows of ground truth and detections share the keys `component, family, box, n_voxels, mm3`; ground truth adds `ignore`, detections add `score`; both get `host, host_rule, host_fractions, side` from `bind_rows`. `fold_dir(results_root, disease, fold, trainer)` is used by Tasks 3, 8, 9.
- Review Focus lines each have a test in the task that owns the code.
