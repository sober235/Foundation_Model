# S4 Brain Anatomy on fastMRI-style FLAIR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate the two models of S4 — a brain-outline model that strips the skull of fastMRI FLAIR stacks and a student anatomy model that outputs the 14 sided host classes plus ventricles that the binding of S7 uses — from simulated fastMRI-style stacks of three skull-stripped 1 mm FLAIR sources with SynthSeg teacher labels, and measure their agreement with SynthSeg on the 447 fastMRI stacks (NOT_EVIDENCE until the Level R reader labels exist).

**Architecture:** Pure, tested modules under `anatobind/anatomy/` (label space, sources and patient split, slab simulation, outline rule), an evaluation module and an inference chain; thin scripts build two nnU-Net v2 datasets (Dataset907_BrainAnatomyFLAIR, 3d_fullres; Dataset908_FastMRIBrainOutline, 2d), launch the two fold-0 trainings on idle GPUs, evaluate and infer. Simulation is offline (K = 4 stacks per training case); nnU-Net's `ignore` label carries lesion regions and unsupervised outline slices. Everything the models output is a pseudo-label: NOT_EVIDENCE.

**Tech Stack:** Python 3.11 (`~/anaconda3/envs/nvgen`), numpy, scipy.ndimage, nibabel, nnU-Net v2.8.0 (`scripts/nnunet_env.sh`), matplotlib (montages), pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-brain-anatomy-flair-design.md` (decisions A1–A16; §3 data, §4 label space, §5 simulation, §6 outline model, §7 evaluation, §8 inference). Probes it rests on: `docs/verification/2026-10-02/s4_probe2/` (stack geometry and coverage), `docs/verification/2026-10-02/s4_probe3_frame/` (array frames), `docs/verification/2026-10-02/s4_sibbms_synthseg/` (teacher maps, exclusions).

## Global Constraints

- Work on `main` in the main checkout `/data0/congcong/code/Project_Doing/foundation_model` (no parallel line is open). Commit per task; `git add` only the files named in the task (the checkout holds untracked user files under `docs/`; never `git add -A`). Commit author = the repository's local config (Congcong Liu); English messages; no Co-Authored-By, no "Generated with", no AI trace. No push.
- Python: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python …` for every command; nnU-Net only through `bash -c 'source scripts/nnunet_env.sh && …'` or `anatobind.infer.knee.nnunet_env`. nnU-Net v2.8.0: the `ignore` label must be the highest label id (student 15, outline 2).
- Never delete, move, shorten or overwrite a file anywhere (the pre-tool hook blocks the usual deletion calls, and a Python file that contains such a call cannot even be written). Every output directory is checked before `mkdir` and refused when it exists; a rerun uses a new name. Deletions are listed for the user.
- Data: read only under `/data2/congcong/data/FM_data`; write only under `/data2/congcong/data/FM_data/derived/{brain_anatomy,nnunet}`, the repository's `docs/` and `logs/`. `/data0/congcong/data/FM_Data` is a cold backup: never read it.
- Compute: CPU jobs `nice -n 19`, at most 48 threads in total on the machine; GPU only on cards that `nvidia-smi` shows idle (no compute process, under 1000 MiB used), one training per card; a job projected above 24 h is reported to the user before it starts; never signal a process this session did not start.
- Array frame: the student is trained and applied in the fastMRI RSS array frame of `anatobind.data_engine.fastmri.rss_h5_to_nifti` (axis 0 towards the patient's left, axis 1 with the cerebellum at low indices, axis 2 upwards); a RAS array enters it by flipping axis 0 only (`s4_probe3_frame`). Left/right follow the headers (spec M10).
- Label space: `anatobind.anatomy.labels.STUDENT` (16 compact classes) is the single source of truth; inference writes SynthSeg representative values so that `BrainBinder` and `host_class_map` read the output like a SynthSeg map.
- Gates (A11): host agreement ≥ 0.90, mean sided-host Dice ≥ 0.80 on the reliable fastMRI slices, outline Dice ≥ 0.97 on the outline model's test stacks; no tuning to pass them; the lowest two slices are reported, not judged; the final judgement waits for Level R (A12). Every record carries the command and `Code: commit <hash>`.
- Tests: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider` (839 passed, 1 skipped before this plan; 872 after it); unit tests never read `/data2`.

## Review Focus

1. A source case whose FLAIR and teacher map are not on one grid after the 1 mm resampling (BMSR has 1.5 mm slices): `case_arrays` must refuse it with the file name, never silently vote labels onto the wrong voxels — pinned by the two `not on the grid` cases of `test_simulate_case_writes_stacks_labels_and_params_deterministically` (Task 7; `check_one_grid` compares shape and affine from the headers) and by `test_simulate_end_to_end_gives_a_fastmri_shaped_stack` (`ValueError: differ`, Task 3).
2. A brain shorter than the 80 mm stack, or a stack whose top lies above the volume: the bottom clamps to 0 and the slabs above the volume come out empty (background), never an index error — `test_bottom_slice_and_slab_groups` (clamping, empty groups, Task 3).
3. A fastMRI stack whose SynthSeg map is empty or holds under 300 mL (14 of 447): it must not train the outline model, it is listed in `excluded.json`, and the evaluation must not count it as agreement — `test_dataset908_places_usable_stacks_by_patient_and_lists_the_unusable` (Task 7) and `test_reliable_slices_follow_the_outline_rule` (empty map, no reliable slice, Task 5).
4. A toy geometry (2 mm pixels, 8 slices) or a real one of 14 or 12 slices at 0.86 mm: the reliable-slice rule and the outline rule must apply by area in mm², not by voxel count — `test_reliable_slices_follow_the_outline_rule` (Task 5), `test_supervised_slices_leave_out_the_lowest_two_and_the_top` (Task 4).
5. A patient with several sessions or follow-ups (SibBMS `ses-*`, PDGM `_FU*`, BMSR letter suffixes, fastMRI `patient_id`) must never sit in both the training and the test set — `test_split_keeps_every_patient_on_one_side_per_source_and_is_deterministic` and `test_pdgm_and_bmsr_cases_carry_lesion_masks_and_patients` (Task 2), `test_patient_splits_and_split_files` (Task 7).

---

### Task 1: Label space of the student

**Files:**
- Create: `anatobind/anatomy/__init__.py`
- Create: `anatobind/anatomy/labels.py`
- Test: `tests/test_brain_anatomy_labels.py`

**Interfaces:**
- Consumes: `anatobind.eval.geometry.HOST_CLASSES`, `LANDMARKS`, `LEFT_LABELS`, `RIGHT_LABELS`, `host_class_map`.
- Produces: `STUDENT` (tuple of (compact id, name, SynthSeg labels, representative value)), `IGNORE = 15`, `N_CLASSES = 16`, `NAMES`, `LABELS_JSON` (nnU-Net dataset.json labels), `HOST_CLASS_OF`, `SIDE_OF`, `HOST_IDS` (the 13 sided host ids), `to_student(seg) -> uint8`, `to_synthseg(student) -> int16`, `with_ignore(student, mask) -> uint8`, `check_consistency()`.

`anatobind/anatomy/__init__.py` is an empty file.

- [ ] **Step 1: Write the failing test**

`tests/test_brain_anatomy_labels.py`:

```python
# tests/test_brain_anatomy_labels.py
import numpy as np
import pytest

import anatobind.anatomy.labels as L
from anatobind.eval.geometry import CLASS_NAMES, HOST_CLASSES, LANDMARKS, host_class_map


def test_the_label_space_is_consistent_with_the_host_classes():
    assert L.check_consistency()
    assert L.N_CLASSES == 16 and L.IGNORE == 15 and len(L.STUDENT) == 15
    assert L.LABELS_JSON["background"] == 0 and L.LABELS_JSON["ignore"] == 15 and L.LABELS_JSON["ventricles"] == 14
    assert L.HOST_CLASS_OF[9] == "brainstem" and L.SIDE_OF[9] is None and L.HOST_CLASS_OF[14] is None
    assert len(L.HOST_IDS) == 13 and 9 in L.HOST_IDS and 14 not in L.HOST_IDS
    # every SynthSeg host label is covered exactly once
    covered = [l for _, _, labels, _ in L.STUDENT for l in labels]
    assert len(covered) == len(set(covered))
    assert set(covered) >= {l for labels in HOST_CLASSES.values() for l in labels}


def test_to_student_and_back_keep_the_host_class_and_drop_csf():
    seg = np.array([[0, 2, 41, 3, 42], [10, 49, 12, 58, 16], [8, 47, 18, 60, 24], [4, 43, 14, 15, 5]], np.int16)
    stu = L.to_student(seg)
    assert stu.dtype == np.uint8
    assert stu.tolist() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 0], [14, 14, 14, 14, 14]]   # CSF 24 -> background
    back = L.to_synthseg(stu)
    assert back.dtype == np.int16
    # the representative values fall in the same host class as the original labels
    assert np.array_equal(host_class_map(back), host_class_map(seg))
    assert all(v in LANDMARKS for v in back[3]) and back[2, 4] == 0
    assert CLASS_NAMES[int(host_class_map(back)[1, 2]) - 1] == "basal_ganglia"


def test_ignore_marks_lesion_voxels_and_is_refused_on_the_way_back():
    stu = np.array([[1, 2], [3, 0]], np.uint8)
    lab = L.with_ignore(stu, np.array([[0, 1], [0, 2]]))
    assert lab.tolist() == [[1, 15], [3, 15]] and stu[0, 1] == 2          # the input is not changed
    with pytest.raises(ValueError, match="student labels"):
        L.to_synthseg(lab)
    with pytest.raises(ValueError, match="outside"):
        L.to_student(np.array([300]))
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_labels.py -q -p no:cacheprovider`
Expected: errors at import or collection (the module does not exist yet).

- [ ] **Step 3: Write `anatobind/anatomy/__init__.py`**

```python
```

- [ ] **Step 4: Write `anatobind/anatomy/labels.py`**

```python
"""Label space of the S4 student anatomy model (spec 2026-10-02 §4).

Sixteen compact classes for nnU-Net: background, the seven host classes of anatobind.eval.geometry split into left
and right (the brainstem has no side), the ventricles as one landmark class, and an ignore class that marks lesion
voxels in training labels (nnU-Net leaves them out of the loss). to_synthseg maps a compact map back to SynthSeg label
values so that BrainBinder and host_class_map read it like a SynthSeg map. Everything is a pseudo-label: NOT_EVIDENCE."""
import numpy as np

from anatobind.eval.geometry import HOST_CLASSES, LANDMARKS, LEFT_LABELS, RIGHT_LABELS

# (compact id, name, SynthSeg labels, representative SynthSeg value written at inference)
STUDENT = (
    (0, "background", (), 0),
    (1, "white_matter_left", (2,), 2), (2, "white_matter_right", (41,), 41),
    (3, "cortex_left", (3,), 3), (4, "cortex_right", (42,), 42),
    (5, "thalamus_left", (10,), 10), (6, "thalamus_right", (49,), 49),
    (7, "basal_ganglia_left", (11, 12, 13, 26), 11), (8, "basal_ganglia_right", (50, 51, 52, 58), 50),
    (9, "brainstem", (16,), 16),
    (10, "cerebellum_left", (7, 8), 7), (11, "cerebellum_right", (46, 47), 46),
    (12, "other_deep_grey_left", (17, 18, 28), 17), (13, "other_deep_grey_right", (53, 54, 60), 53),
    (14, "ventricles", (4, 43, 5, 44, 14, 15), 4),
)
IGNORE = 15
N_CLASSES = 16                                   # 0..14 are predicted, 15 is the training-only ignore label
NAMES = tuple(name for _, name, _, _ in STUDENT)
LABELS_JSON = {**{name: cid for cid, name, _, _ in STUDENT}, "ignore": IGNORE}      # nnU-Net dataset.json
HOST_CLASS_OF = {cid: (name.rsplit("_left", 1)[0].rsplit("_right", 1)[0] if cid not in (0, 14) else None)
                 for cid, name, _, _ in STUDENT}
SIDE_OF = {cid: ("left" if name.endswith("_left") else "right" if name.endswith("_right") else None)
           for cid, name, _, _ in STUDENT}
HOST_IDS = tuple(cid for cid in range(1, 14) if HOST_CLASS_OF[cid])                  # the 13 sided host classes
_MAX_SYNTHSEG = 255

_TO_STUDENT = np.zeros(_MAX_SYNTHSEG + 1, np.uint8)
for _cid, _, _labels, _ in STUDENT:
    for _l in _labels:
        _TO_STUDENT[_l] = _cid
_TO_SYNTHSEG = np.zeros(N_CLASSES, np.int16)
for _cid, _, _, _rep in STUDENT:
    _TO_SYNTHSEG[_cid] = _rep


def to_student(seg):
    """SynthSeg label map -> compact student labels (uint8). Labels not listed in STUDENT (CSF, background, the
    small structures the host classes leave out) become background."""
    s = np.asarray(seg)
    if s.min() < 0 or s.max() > _MAX_SYNTHSEG:
        raise ValueError(f"label values outside 0..{_MAX_SYNTHSEG}: {int(s.min())}..{int(s.max())}")
    return _TO_STUDENT[s.astype(np.int64)]


def to_synthseg(student):
    """Compact student labels (ignore not allowed) -> representative SynthSeg values (int16)."""
    s = np.asarray(student)
    if s.min() < 0 or s.max() >= IGNORE:
        raise ValueError(f"student labels must be 0..{IGNORE - 1}, got {int(s.min())}..{int(s.max())}")
    return _TO_SYNTHSEG[s.astype(np.int64)]


def with_ignore(student, lesion_mask):
    """Training label: lesion voxels (any mask value > 0) are set to IGNORE."""
    out = np.array(student, np.uint8, copy=True)
    out[np.asarray(lesion_mask) > 0] = IGNORE
    return out


def check_consistency():
    """Every listed SynthSeg label belongs to the host class and side its student class names; the ventricle labels
    are landmarks; representative values map back to their own class. Raises AssertionError otherwise."""
    for cid, name, labels, rep in STUDENT:
        host, side = HOST_CLASS_OF[cid], SIDE_OF[cid]
        if host:
            assert set(labels) <= set(HOST_CLASSES[host]), name
            sided = LEFT_LABELS if side == "left" else RIGHT_LABELS if side == "right" else ()
            assert all(l in sided for l in labels) if side else not any(l in LEFT_LABELS + RIGHT_LABELS for l in labels), name
        elif cid == 14:
            assert set(labels) <= set(LANDMARKS) and 24 not in labels, name
        if cid:
            assert int(_TO_STUDENT[rep]) == cid, name
    assert sorted(LABELS_JSON.values()) == list(range(N_CLASSES))
    return True
```

- [ ] **Step 5: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_labels.py -q -p no:cacheprovider`
Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add anatobind/anatomy/__init__.py anatobind/anatomy/labels.py tests/test_brain_anatomy_labels.py
git commit -m "S4 anatomy: student label space (16 compact classes, SynthSeg mapping, ignore)"
```

---

### Task 2: The three sources and the patient-level split

**Files:**
- Create: `anatobind/anatomy/sources.py`
- Test: `tests/test_brain_anatomy_sources.py`

**Interfaces:**
- Consumes: `anatobind.nnunet.brain_disease.{FM, anatomy_path, channel_path, label_path, list_cases, patient_of}`.
- Produces: `sibbms_cases(root)`, `pdgm_cases(root)`, `bmsr_cases(root)`, `all_cases(root) -> {case: {source, patient, flair, anatomy, lesion, lesion_values}}`, `EXCLUDED_SIBBMS`, `split_by_patient(cases, test_share=0.2, seed=0) -> {case: 'train' | 'test'}`, `check_split`, `TEST_SHARE`, `SOURCES`.

- [ ] **Step 1: Write the failing test**

`tests/test_brain_anatomy_sources.py`:

```python
# tests/test_brain_anatomy_sources.py
from pathlib import Path

import pytest

import anatobind.anatomy.sources as S


def _sibbms_tree(root, sessions, segs):
    for cohort, sub, ses in sessions:
        d = root / S.SIBBMS_OUTPUT / cohort / sub / ses / "anat"
        d.mkdir(parents=True)
        (d / f"{sub}_{ses}_FLAIR.nii.gz").write_bytes(b"")
        (d / f"{sub}_{ses}_T1w.nii.gz").write_bytes(b"")
        (d / f"{sub}_{ses}_ce-GAD_T1w.nii.gz").write_bytes(b"")
    (root / S.SIBBMS_SEG).mkdir(parents=True)
    for case in segs:
        (root / S.SIBBMS_SEG / f"{case}_T1w_seg.nii.gz").write_bytes(b"")


def test_sibbms_cases_follow_the_cohort_prefix_and_the_exclusion_list(tmp_path):
    sessions = [("MS", "sub-001", "ses-001"), ("MS", "sub-001", "ses-002"), ("Norm", "sub-001", "ses-001"), ("MS", "sub-011", "ses-001"),
                ("Annotation", "sub-047", "ses-001")]
    _sibbms_tree(tmp_path, sessions, ["MS_sub-001_ses-001", "MS_sub-001_ses-002", "Norm_sub-001_ses-001"])
    cases = S.sibbms_cases(tmp_path)
    assert sorted(cases) == ["MS_sub-001_ses-001", "MS_sub-001_ses-002", "Norm_sub-001_ses-001"]   # excluded + Annotation left out
    assert cases["MS_sub-001_ses-002"]["patient"] == "MS_sub-001" and cases["Norm_sub-001_ses-001"]["patient"] == "Norm_sub-001"
    assert cases["MS_sub-001_ses-001"]["flair"].name == "sub-001_ses-001_FLAIR.nii.gz"
    assert cases["MS_sub-001_ses-001"]["anatomy"] == tmp_path / S.SIBBMS_SEG / "MS_sub-001_ses-001_T1w_seg.nii.gz"
    assert cases["MS_sub-001_ses-001"]["lesion"] is None and cases["MS_sub-001_ses-001"]["lesion_values"] == ()
    _sibbms_tree(tmp_path / "b", [("MS", "sub-002", "ses-001")], [])
    with pytest.raises(FileNotFoundError, match="without a teacher map"):
        S.sibbms_cases(tmp_path / "b")


def test_pdgm_and_bmsr_cases_carry_lesion_masks_and_patients(tmp_path):
    base = tmp_path / "UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5"
    for c in ("UCSF-PDGM-0004", "UCSF-PDGM-0004_FU007d"):
        (base / f"{c}_nifti").mkdir(parents=True)
    bm = tmp_path / "UCSF-BMSR_cbb/UCSF-BMSR/UCSF_BrainMetastases_TRAIN"
    for c in ("100101A", "100101B"):
        (bm / c).mkdir(parents=True)
    p = S.pdgm_cases(tmp_path)
    assert sorted(p) == ["UCSF-PDGM-0004", "UCSF-PDGM-0004_FU007d"] and {r["patient"] for r in p.values()} == {"UCSF-PDGM-0004"}
    assert p["UCSF-PDGM-0004"]["flair"].name == "UCSF-PDGM-0004_FLAIR.nii.gz" and p["UCSF-PDGM-0004"]["lesion_values"] == (1, 2, 4)
    assert p["UCSF-PDGM-0004"]["lesion"].name == "UCSF-PDGM-0004_tumor_segmentation.nii.gz"
    b = S.bmsr_cases(tmp_path)
    assert {r["patient"] for r in b.values()} == {"100101"} and b["100101A"]["lesion"].name == "100101A_seg.nii.gz"
    assert b["100101A"]["anatomy"].name == "100101A_T1pre_seg.nii.gz" and b["100101A"]["lesion_values"] == (1,)


def test_split_keeps_every_patient_on_one_side_per_source_and_is_deterministic():
    cases = {}
    for i in range(10):
        for s in ("ses-001", "ses-002"):
            cases[f"MS_sub-{i:03d}_{s}"] = {"source": "sibbms", "patient": f"MS_sub-{i:03d}"}
    for i in range(10):
        cases[f"UCSF-PDGM-{i:04d}"] = {"source": "pdgm", "patient": f"UCSF-PDGM-{i:04d}"}
    split = S.split_by_patient(cases, 0.2, seed=0)
    assert split == S.split_by_patient(cases, 0.2, seed=0)
    for src, n_test in (("sibbms", 2), ("pdgm", 2)):
        test_patients = {cases[c]["patient"] for c, v in split.items() if v == "test" and cases[c]["source"] == src}
        assert len(test_patients) == n_test                              # ceil(0.2 x 10) patients per source
    sib_test = [c for c, v in split.items() if v == "test" and c.startswith("MS_")]
    assert len(sib_test) == 4 and all(c.replace("ses-001", "ses-002") in sib_test or c.replace("ses-002", "ses-001") in sib_test for c in sib_test)
    assert S.check_split(cases, split)
    with pytest.raises(ValueError, match="mixes patients"):
        S.check_split(cases, {**split, "MS_sub-000_ses-001": "test", "MS_sub-000_ses-002": "train"})
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_sources.py -q -p no:cacheprovider`
Expected: errors at import or collection (the module does not exist yet).

- [ ] **Step 3: Write `anatobind/anatomy/sources.py`**

```python
"""The three training sources of the S4 anatomy model and their patient-level split (spec 2026-10-02 §3, A2, A3).

A case is one FLAIR volume with a SynthSeg teacher map on the same grid: SibBMS sessions (healthy and MS), UCSF-PDGM
and UCSF-BMSR scans. Lesion masks (PDGM tumour, BMSR metastases) become the ignore region of the training label."""
import math
import random
from pathlib import Path

from anatobind.nnunet.brain_disease import FM, anatomy_path, channel_path, label_path, list_cases, patient_of

SIBBMS_OUTPUT = "SibBMS_ms/sibbms/Output"
SIBBMS_SEG = "derived/synthseg/sibbms/seg_native"
# s4_sibbms_synthseg/README.md: eight maps SynthSeg could not segment (near-empty inputs) and one two-dimensional T1w file
EXCLUDED_SIBBMS = ("MS_sub-011_ses-001", "MS_sub-027_ses-005", "MS_sub-057_ses-001", "MS_sub-070_ses-003", "MS_sub-070_ses-004",
                   "MS_sub-070_ses-005", "Norm_sub-010_ses-001", "Norm_sub-020_ses-001", "Norm_sub-037_ses-001")
LESION_VALUES = {"pdgm": (1, 2, 4), "bmsr": (1,)}        # label values that mean lesion in the source masks
TEST_SHARE = 0.2
SOURCES = ("sibbms", "pdgm", "bmsr")


def sibbms_cases(root=FM):
    """{case: record} for every SibBMS session with a FLAIR and a teacher map, excluded sessions left out.
    case = '<Cohort>_sub-XXX_ses-YYY', patient = '<Cohort>_sub-XXX' (MS and Norm both count subjects from sub-001)."""
    root = Path(root)
    out = {}
    for flair in sorted((root / SIBBMS_OUTPUT).glob("*/sub-*/ses-*/anat/sub-*_ses-???_FLAIR.nii.gz")):
        cohort, sub, ses = flair.relative_to(root / SIBBMS_OUTPUT).parts[:3]
        case = f"{cohort}_{sub}_{ses}"
        if cohort not in ("MS", "Norm") or case in EXCLUDED_SIBBMS:
            continue
        seg = root / SIBBMS_SEG / f"{case}_T1w_seg.nii.gz"
        if not seg.is_file():
            raise FileNotFoundError(f"{case}: FLAIR without a teacher map ({seg}); add it to EXCLUDED_SIBBMS or run SynthSeg")
        out[case] = {"source": "sibbms", "patient": f"{cohort}_{sub}", "flair": flair, "anatomy": seg, "lesion": None, "lesion_values": ()}
    return out


def pdgm_cases(root=FM):
    return {c: {"source": "pdgm", "patient": patient_of("glioma", c), "flair": channel_path("glioma", c, "FLAIR", root),
                "anatomy": anatomy_path("glioma", c, root), "lesion": label_path("glioma", c, root), "lesion_values": LESION_VALUES["pdgm"]}
            for c in list_cases("glioma", root)}


def bmsr_cases(root=FM):
    return {c: {"source": "bmsr", "patient": patient_of("metastasis", c), "flair": channel_path("metastasis", c, "FLAIR", root),
                "anatomy": anatomy_path("metastasis", c, root), "lesion": label_path("metastasis", c, root),
                "lesion_values": LESION_VALUES["bmsr"]} for c in list_cases("metastasis", root)}


def all_cases(root=FM):
    """Every case of the three sources; case ids are disjoint by construction (prefixes MS_/Norm_, UCSF-PDGM-, digits)."""
    out = {}
    for fn in (sibbms_cases, pdgm_cases, bmsr_cases):
        part = fn(root)
        if set(part) & set(out):
            raise ValueError(f"case ids collide: {sorted(set(part) & set(out))[:5]}")
        out.update(part)
    return out


def split_by_patient(cases, test_share=TEST_SHARE, seed=0):
    """{case: 'train' | 'test'}: per source, ceil(test_share x patients) patients drawn with the seed go to the test set
    with all their cases; the rest train. No patient is in both sets."""
    out = {}
    for source in sorted({r["source"] for r in cases.values()}):
        patients = sorted({r["patient"] for r in cases.values() if r["source"] == source})
        rng = random.Random(f"{seed}:{source}")
        rng.shuffle(patients)
        test = set(patients[:math.ceil(test_share * len(patients))])
        for c, r in cases.items():
            if r["source"] == source:
                out[c] = "test" if r["patient"] in test else "train"
    check_split(cases, out)
    return out


def check_split(cases, split):
    by_patient = {}
    for c, s in split.items():
        by_patient.setdefault(cases[c]["patient"], set()).add(s)
    mixed = sorted(p for p, s in by_patient.items() if len(s) > 1)
    if mixed or set(split) != set(cases):
        raise ValueError(f"split mixes patients {mixed[:5]} or misses cases")
    return True
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_sources.py -q -p no:cacheprovider`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add anatobind/anatomy/sources.py tests/test_brain_anatomy_sources.py
git commit -m "S4 anatomy: SibBMS, PDGM and BMSR sources with lesion masks and a patient-level split"
```

---

### Task 3: Slab simulation

**Files:**
- Create: `anatobind/anatomy/simulate.py`
- Test: `tests/test_brain_anatomy_simulate.py`

**Interfaces:**
- Consumes: Task 1 (`IGNORE`, `N_CLASSES`), `anatobind.data_engine.fastmri.rss_affine`.
- Produces: `to_fastmri_frame(ras)`, `sample_params(rng) -> dict(theta_lr, theta_ap, empty_top, n_slices, inplane_mm, matrix, gamma, blur, noise, bias)`, `rotate`, `area_profile`, `top_index`, `bottom_index(profile, n_slices, empty_top)`, `slab_groups`, `average_slices`, `vote_labels`, `ignore_any`, `fit_to_matrix`, `resample_inplane`, `bias_field`, `intensity_augment`, `simulate(flair_ras, student_ras, lesion_ras, rng, params=None) -> (stack float32, labels uint8, params)` (params gains `bottom_slice_1mm` and `bottom_area_share`), `stack_affine(inplane_mm, shape)`.

- [ ] **Step 1: Write the failing test**

`tests/test_brain_anatomy_simulate.py`:

```python
# tests/test_brain_anatomy_simulate.py
import numpy as np
import pytest

import anatobind.anatomy.simulate as SIM
from anatobind.anatomy.labels import IGNORE, to_student


def _brain(shape=(60, 70, 80)):
    """A 1 mm RAS 'brain': an ellipsoid of left (2 / 3) and right (41 / 42) white matter and cortex with ventricles (4) in the
    middle, sitting on the lower two thirds of the volume; the FLAIR image follows the labels."""
    x, y, z = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    cx, cy, cz = 29.5, 34.5, 32.0
    rad = ((x - cx) / 25) ** 2 + ((y - cy) / 30) ** 2 + ((z - cz) / 28) ** 2
    seg = np.zeros(shape, np.int16)
    inside, shell = rad <= 1.0, (rad <= 1.0) & (rad > 0.7)
    seg[inside & (x < cx)] = 2             # RAS: axis 0 grows towards the patient's right, so the left half has low indices
    seg[inside & (x >= cx)] = 41
    seg[shell & (x < cx)] = 3
    seg[shell & (x >= cx)] = 42
    seg[(rad <= 0.05)] = 4
    img = np.zeros(shape, np.float32)
    img[np.isin(seg, (2, 41))] = 300.0
    img[np.isin(seg, (3, 42))] = 450.0
    img[seg == 4] = 80.0
    return img, seg


def test_the_fastmri_frame_flips_left_and_right_only():
    img, seg = _brain()
    f = SIM.to_fastmri_frame(seg)
    assert f.shape == seg.shape and np.array_equal(f, seg[::-1])
    left = np.argwhere(f == 2)[:, 0].mean()
    right = np.argwhere(f == 41)[:, 0].mean()
    assert left > right                                         # patient left now at high axis-0 indices, as in fastMRI


def test_parameters_are_deterministic_and_inside_the_measured_ranges():
    a, b = SIM.sample_params(np.random.default_rng(3)), SIM.sample_params(np.random.default_rng(3))
    assert a == b
    tops = [SIM.sample_params(np.random.default_rng(i))["empty_top"] for i in range(400)]
    assert set(tops) <= {1, 2, 3, 4, 5} and 2 <= float(np.median(tops)) <= 3
    for i in range(50):
        p = SIM.sample_params(np.random.default_rng(i))
        assert p["n_slices"] in (14, 16) and p["inplane_mm"] in (0.6875, 0.86) and p["matrix"] in ((320, 320), (260, 320), (276, 276))
        assert abs(p["theta_lr"]) <= 10 and abs(p["theta_ap"]) <= 5 and 0.7 <= p["gamma"] <= 1.4


def test_bottom_slice_and_slab_groups():
    profile = [0] * 20 + [600] * 70 + [0] * 10              # brain in 1 mm slices 20..89: top index 89
    assert SIM.top_index(profile) == 89 and SIM.top_index([0, 100]) == -1
    assert SIM.bottom_index(profile, 16, 2) == 89 + 1 + 10 - 80          # 20: the stack ends 2 slices above the brain
    assert SIM.bottom_index(profile, 16, 3) == 25 and SIM.bottom_index(profile, 14, 3) == 35
    assert SIM.bottom_index([600] * 30, 16, 2) == 0                       # a short brain: the bottom is clamped
    groups = SIM.slab_groups(4, 16, 30)
    assert len(groups) == 16 and groups[0] == (4, 9) and groups[5] == (29, 30) and groups[6] == (30, 30)   # beyond the top: empty
    assert SIM.slab_groups(0, 14, 100)[-1] == (65, 70)


def test_votes_average_and_ignore_on_a_tiny_stack():
    lab = np.zeros((2, 2, 7), np.uint8)
    lab[0, 0, :] = [1, 1, 2, 2, 2, 0, 0]          # 2 wins the first group (3 of 5)
    lab[0, 1, :] = [1, 1, 2, 2, 9, 0, 0]          # tie 1 vs 2 in slices 0-4: the centre slice (2) holds 2 -> 2
    lab[1, 0, :] = [3, 0, 0, 0, 3, 14, 14]        # 0 wins (3 of 5); second group (5, 7) -> 14
    img = np.zeros((2, 2, 7), np.float32)
    img[1, 1, :] = [10, 20, 30, 40, 50, 100, 200]
    groups = SIM.slab_groups(0, 3, 7)
    assert groups == [(0, 5), (5, 7), (7, 7)]
    v = SIM.vote_labels(lab, groups)
    assert v[0, 0, 0] == 2 and v[0, 1, 0] == 2 and v[1, 0, 0] == 0 and v[1, 0, 1] == 14 and v[:, :, 2].max() == 0
    a = SIM.average_slices(img, groups)
    assert a[1, 1, 0] == pytest.approx(30.0) and a[1, 1, 1] == pytest.approx(150.0) and a[1, 1, 2] == 0.0
    mask = np.zeros((2, 2, 7), np.uint8)
    mask[0, 0, 4] = 1                             # one lesion slice is enough
    ig = SIM.ignore_any(mask, groups)
    assert ig[0, 0, 0] and not ig[0, 0, 1] and not ig[0, 1, 0]


def test_fit_to_matrix_crops_and_pads_around_the_centre():
    s = np.arange(5 * 6 * 2, dtype=np.float32).reshape(5, 6, 2)
    out = SIM.fit_to_matrix(s, (3, 10), (2.0, 3.0))
    assert out.shape == (3, 10, 2) and out[0, 0, 0] == 0.0 and out[1, 2, 0] == s[2, 0, 0]       # x: rows 1..3 kept; y padded by 2
    back = SIM.fit_to_matrix(out, (5, 6), (1.5, 5.0))
    assert back.shape == (5, 6, 2) and back[2, 0, 0] == s[2, 0, 0] and back[0, :, :].max() == 0.0


def test_rotation_by_zero_is_identity_and_small_tilts_keep_the_brain():
    img, seg = _brain()
    stu = to_student(seg)
    c = np.argwhere(stu > 0).mean(axis=0)
    assert np.array_equal(SIM.rotate(stu, 0.0, 0.0, c, order=0), stu)
    r = SIM.rotate(stu, 10.0, -5.0, c, order=0)
    assert r.dtype == stu.dtype and abs(int((r > 0).sum()) - int((stu > 0).sum())) < 0.03 * (stu > 0).sum()
    assert np.allclose(np.argwhere(r > 0).mean(axis=0), c, atol=1.5)


def test_simulate_end_to_end_gives_a_fastmri_shaped_stack():
    img, seg = _brain()
    stu = to_student(seg)
    lesion = np.zeros(seg.shape, np.uint8)
    lesion[30:36, 30:40, 20:40] = 1
    params = {"theta_lr": 4.0, "theta_ap": -2.0, "empty_top": 3, "n_slices": 16, "inplane_mm": 0.6875, "matrix": (260, 320),
              "gamma": 1.0, "blur": 0.3, "noise": 0.02, "bias": [0.2, -0.1, 0.3, 0.0, 0.1, -0.2]}
    stack, labels, out = SIM.simulate(img, stu, lesion, np.random.default_rng(0), params)
    assert stack.shape == (260, 320, 16) and labels.shape == (260, 320, 16) and stack.dtype == np.float32 and labels.dtype == np.uint8
    assert set(np.unique(labels)) <= set(range(16)) and (labels == IGNORE).any()
    assert out["bottom_slice_1mm"] >= 0 and labels[:, :, -1].max() == 0                      # the top slices are above the brain
    present = [k for k in range(16) if (labels[:, :, k] > 0).any()]
    assert 11 <= len(present) <= 13 and present[0] <= 1 and 16 - 1 - present[-1] in (3, 4)          # 3 empty slabs above the vertex (4 when the top slab is thin)
    assert out["bottom_area_share"] is None or 0.0 <= out["bottom_area_share"] <= 1.0
    wl, wr = np.argwhere(labels == 1)[:, 0].mean(), np.argwhere(labels == 2)[:, 0].mean()
    assert wl > wr                                                                            # fastMRI frame: left at high indices
    brain = (labels > 0) & (labels != IGNORE)
    assert np.isfinite(stack).all() and 900 < np.percentile(stack[brain], 99) <= 1000.5 and np.percentile(stack[brain], 1) >= -0.5
    aff = SIM.stack_affine(0.6875, stack.shape)
    assert np.allclose(np.diag(aff)[:3], [-0.6875, -0.6875, 5.0])
    stack2, labels2, _ = SIM.simulate(img, stu, lesion, np.random.default_rng(0), params)
    assert np.array_equal(labels, labels2) and np.allclose(stack, stack2)
    with pytest.raises(ValueError, match="differ"):
        SIM.simulate(img[:-1], stu, None, np.random.default_rng(0), params)
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_simulate.py -q -p no:cacheprovider`
Expected: errors at import or collection (the module does not exist yet).

- [ ] **Step 3: Write `anatobind/anatomy/simulate.py`**

```python
"""Slab simulation: a 1 mm isotropic RAS volume -> a fastMRI-style FLAIR stack (spec 2026-10-02 §5, A5-A8).

Geometry follows the measurements of docs/verification/2026-10-02/s4_probe2 (16 x 5 mm slices positioned from the
vertex down, 2-3 empty slices above the brain, 65 mm of brain in the stack) and the array frame of the fastMRI RSS
NIfTIs (docs/verification/2026-10-02/s4_probe3_frame: flip axis 0 of a RAS array). Intensities are augmented, not
simulated from sequence physics (A7). Labels are voted over the five 1 mm slices of each slab; lesion voxels become
the ignore label (A8). Every function is deterministic given its random generator."""
import math

import numpy as np
from scipy import ndimage

from anatobind.anatomy.labels import IGNORE, N_CLASSES
from anatobind.data_engine.fastmri import rss_affine

SLICE_MM = 5.0
# empty 5 mm slices above the brain in the fastMRI stacks (s4_probe2: 1 / 19 / 157 / 191 / 63 / 14 of 447 stacks have 0 / 1 / 2 / 3 / 4 / 5+)
EMPTY_TOP = ((1, 0.045), (2, 0.35), (3, 0.43), (4, 0.14), (5, 0.035))
N_SLICES = ((16, 0.9), (14, 0.1))
INPLANE_MM = ((0.6875, 0.82), (0.86, 0.18))
MATRICES = (((320, 320), 0.6), ((260, 320), 0.2), ((276, 276), 0.2))
TILT_LR_DEG, TILT_AP_DEG = 10.0, 5.0
BIAS_AMP, GAMMA, BLUR_SIGMA, NOISE = 0.2, (0.7, 1.4), (0.0, 0.7), (0.01, 0.04)
OUT_RANGE = 1000.0
MIN_AREA_MM2 = 500.0                            # 5 cm2: a slice with less brain is "empty" (s4_probe2)


def to_fastmri_frame(ras):
    """RAS array (axis 0 towards the right) -> fastMRI array frame (patient left at high axis-0 indices): flip axis 0."""
    return np.ascontiguousarray(np.asarray(ras)[::-1])


def _choose(rng, options):
    items, weights = zip(*options)
    return items[int(rng.choice(len(items), p=np.asarray(weights) / sum(weights)))]


def sample_params(rng):
    return {"theta_lr": float(rng.uniform(-TILT_LR_DEG, TILT_LR_DEG)), "theta_ap": float(rng.uniform(-TILT_AP_DEG, TILT_AP_DEG)),
            "empty_top": int(_choose(rng, EMPTY_TOP)), "n_slices": int(_choose(rng, N_SLICES)), "inplane_mm": float(_choose(rng, INPLANE_MM)),
            "matrix": tuple(_choose(rng, MATRICES)), "gamma": float(rng.uniform(*GAMMA)), "blur": float(rng.uniform(*BLUR_SIGMA)),
            "noise": float(rng.uniform(*NOISE)), "bias": [float(v) for v in rng.uniform(-1, 1, 6)]}


def rotation_matrix(theta_lr_deg, theta_ap_deg):
    """Rotation about axis 0 (left-right: nodding, tilts the slab in the sagittal plane) then about axis 1 (anterior-
    posterior: roll), for arrays ordered (left-right, anterior-posterior, inferior-superior)."""
    a, b = math.radians(theta_lr_deg), math.radians(theta_ap_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    return ry @ rx


def rotate(volume, theta_lr_deg, theta_ap_deg, center, order):
    """Rotate about `center` (voxel coordinates); order 1 for images, 0 for labels and masks. Outside -> 0."""
    m = rotation_matrix(theta_lr_deg, theta_ap_deg)
    center = np.asarray(center, float)
    offset = center - m @ center           # affine_transform maps output -> input: in = m @ out + offset
    return ndimage.affine_transform(np.asarray(volume), m, offset=offset, order=order, mode="constant", cval=0.0,
                                    output=np.asarray(volume).dtype if order == 0 else np.float32)


def area_profile(labels):
    """Brain cross-section (voxels, = mm2 at 1 mm) of every axial slice: labels > 0 and not ignore."""
    brain = (np.asarray(labels) > 0) & (np.asarray(labels) != IGNORE)
    return brain.sum(axis=(0, 1))


def top_index(profile, min_area_mm2=MIN_AREA_MM2):
    """The highest 1 mm slice with more than min_area of brain; -1 when there is none."""
    idx = np.flatnonzero(np.asarray(profile) > min_area_mm2)
    return int(idx.max()) if len(idx) else -1


def bottom_index(profile, n_slices, empty_top, thickness=int(SLICE_MM)):
    """Stacks are positioned from the vertex down (s4_probe2: 65 mm of brain under 2-3 empty slices): the stack ends
    empty_top slices above the last slice with brain, and its bottom lies n_slices x 5 mm lower, never below 0."""
    top = top_index(profile)
    return max(0, top + 1 + empty_top * thickness - n_slices * thickness)


def slab_groups(z0, n_slices, n_z, thickness=int(SLICE_MM)):
    """[(start, stop)] of the 1 mm slices averaged into each output slice; stop <= n_z, empty when start >= n_z."""
    return [(min(z0 + k * thickness, n_z), min(z0 + (k + 1) * thickness, n_z)) for k in range(n_slices)]


def average_slices(volume, groups):
    v = np.asarray(volume, np.float32)
    out = np.zeros(v.shape[:2] + (len(groups),), np.float32)
    for k, (a, b) in enumerate(groups):
        if b > a:
            out[:, :, k] = v[:, :, a:b].mean(axis=2)
    return out


def vote_labels(labels, groups, n_classes=N_CLASSES):
    """Majority vote of the compact labels over each group's slices; a tie goes to the slice nearest the group's
    centre; an empty group is background."""
    lab = np.asarray(labels)
    nx, ny = lab.shape[:2]
    out = np.zeros((nx, ny, len(groups)), np.uint8)
    ii, jj = np.arange(nx)[:, None], np.arange(ny)[None, :]
    for k, (a, b) in enumerate(groups):
        if b <= a:
            continue
        block = lab[:, :, a:b]
        counts = np.zeros((nx, ny, n_classes), np.int16)
        for z in range(block.shape[2]):
            np.add.at(counts, (ii, jj, block[:, :, z]), 1)
        best = counts.max(axis=2)
        centre = (a + b - 1) / 2.0
        result = np.full((nx, ny), -1, np.int16)
        for z in sorted(range(block.shape[2]), key=lambda z: abs(a + z - centre)):
            l = block[:, :, z].astype(np.int64)
            take = (result < 0) & (counts[ii, jj, l] == best)
            result[take] = l[take]
        out[:, :, k] = result
    return out


def ignore_any(mask, groups):
    """A slab voxel is lesion when any of its 1 mm slices is (A8)."""
    m = np.asarray(mask) > 0
    out = np.zeros(m.shape[:2] + (len(groups),), bool)
    for k, (a, b) in enumerate(groups):
        if b > a:
            out[:, :, k] = m[:, :, a:b].any(axis=2)
    return out


def fit_to_matrix(stack, matrix, center_xy):
    """Crop / zero-pad the first two axes to `matrix`, centred on center_xy (in the stack's own pixels)."""
    s = np.asarray(stack)
    out = np.zeros((matrix[0], matrix[1]) + s.shape[2:], s.dtype)
    x0 = int(math.floor(center_xy[0] - matrix[0] / 2.0 + 0.5))        # round half up, not to even
    y0 = int(math.floor(center_xy[1] - matrix[1] / 2.0 + 0.5))
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + matrix[0], s.shape[0]), min(y0 + matrix[1], s.shape[1])
    if sx1 > sx0 and sy1 > sy0:
        out[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0] = s[sx0:sx1, sy0:sy1]
    return out


def resample_inplane(stack, inplane_mm, matrix, center_xy_mm, order):
    """1 mm slab stack -> target in-plane spacing (axis 2 untouched) and matrix, centred on a point given in mm of the
    1 mm grid."""
    f = 1.0 / inplane_mm
    z = ndimage.zoom(np.asarray(stack), (f, f, 1.0), order=order, mode="constant", cval=0.0, grid_mode=False)
    return fit_to_matrix(z, matrix, (center_xy_mm[0] * f, center_xy_mm[1] * f))


def bias_field(shape, coeffs, amp=BIAS_AMP):
    """1 + a smooth second-order field whose largest deviation is amp; coeffs: six numbers in [-1, 1]."""
    x, y, z = [np.linspace(-1, 1, n) for n in shape]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    c = np.asarray(coeffs, float)
    field = c[0] * X + c[1] * Y + c[2] * Z + c[3] * X * X + c[4] * Y * Y + c[5] * X * Y
    peak = np.abs(field).max()
    return 1.0 + (amp * field / peak if peak > 0 else 0.0)


def intensity_augment(stack, brain, rng, params):
    """Bias field, gamma on the brain's robust range, in-plane blur, Rician noise, then the brain's 1st-99th
    percentiles mapped to [0, OUT_RANGE]. brain: boolean mask of the same shape (where the labels are not background)."""
    img = np.asarray(stack, np.float32) * bias_field(stack.shape, params["bias"]).astype(np.float32)
    vals = img[brain] if brain.any() else img.ravel()
    lo, hi = np.percentile(vals, [1, 99])
    hi = hi if hi > lo else lo + 1.0
    norm = np.clip((img - lo) / (hi - lo), 0, None)
    img = np.power(norm, params["gamma"]).astype(np.float32)
    if params["blur"] > 0:
        img = ndimage.gaussian_filter(img, sigma=(params["blur"], params["blur"], 0.0))
    sigma = params["noise"] * float(np.median(img[brain])) if brain.any() else params["noise"]
    n1, n2 = rng.normal(0, sigma, img.shape).astype(np.float32), rng.normal(0, sigma, img.shape).astype(np.float32)
    img = np.sqrt((img + n1) ** 2 + n2 ** 2)
    vals = img[brain] if brain.any() else img.ravel()
    lo, hi = np.percentile(vals, [1, 99])
    hi = hi if hi > lo else lo + 1.0
    return ((img - lo) / (hi - lo) * OUT_RANGE).astype(np.float32)


def simulate(flair_ras, student_ras, lesion_ras, rng, params=None):
    """flair_ras: 1 mm RAS image; student_ras: compact labels (0..14) on the same grid; lesion_ras: mask or None.
    Returns (image stack float32 (nx, ny, n_slices), label stack uint8 with IGNORE on lesion voxels, params)."""
    params = dict(params) if params is not None else sample_params(rng)
    img, lab = to_fastmri_frame(flair_ras), to_fastmri_frame(student_ras)
    mask = to_fastmri_frame(lesion_ras) if lesion_ras is not None else np.zeros(lab.shape, np.uint8)
    if img.shape != lab.shape or mask.shape != lab.shape:
        raise ValueError(f"image {img.shape}, labels {lab.shape} and mask {mask.shape} differ")
    brain = np.argwhere(lab > 0)
    center = brain.mean(axis=0) if len(brain) else (np.asarray(lab.shape) - 1) / 2.0
    img = rotate(img, params["theta_lr"], params["theta_ap"], center, order=1)
    lab = rotate(lab, params["theta_lr"], params["theta_ap"], center, order=0)
    mask = rotate(mask, params["theta_lr"], params["theta_ap"], center, order=0)
    profile = area_profile(lab)
    z0 = bottom_index(profile, params["n_slices"], params["empty_top"])
    groups = slab_groups(z0, params["n_slices"], lab.shape[2])
    stack, labels = average_slices(img, groups), vote_labels(lab, groups)
    labels[ignore_any(mask, groups)] = IGNORE
    stack = resample_inplane(stack, params["inplane_mm"], params["matrix"], center[:2], order=1)
    labels = resample_inplane(labels, params["inplane_mm"], params["matrix"], center[:2], order=0)
    stack = intensity_augment(stack, (labels > 0) & (labels != IGNORE), rng, params)
    params["bottom_slice_1mm"] = int(z0)
    params["bottom_area_share"] = round(float(profile[z0] / profile.max()), 3) if profile.max() > 0 else None     # compare with s4_probe2 (median 0.83)
    return stack, labels, params


def stack_affine(inplane_mm, shape):
    """The affine the fastMRI RSS NIfTIs carry (anatobind.data_engine.fastmri.rss_affine), for a (col, row, slice) array."""
    return rss_affine(inplane_mm, inplane_mm, SLICE_MM, shape)
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_simulate.py -q -p no:cacheprovider`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add anatobind/anatomy/simulate.py tests/test_brain_anatomy_simulate.py
git commit -m "S4 anatomy: slab simulation (vertex-anchored stacks, slice averaging, label voting, intensity augmentation)"
```

---

### Task 4: Brain outline labels and post-processing

**Files:**
- Create: `anatobind/anatomy/outline.py`
- Test: `tests/test_brain_anatomy_outline.py`

**Interfaces:**
- Consumes: scipy.ndimage only.
- Produces: `LABELS_JSON = {background: 0, brain: 1, ignore: 2}`, `IGNORE`, `LOW = 2`, `TOP_MARGIN = 1`, `MIN_AREA_MM2 = 500`, `MIN_VOLUME_ML = 300`, `fill_and_keep_largest(mask)`, `supervised_slices(outline, voxel_area_mm2) -> range`, `outline_label(seg, voxel_area_mm2, voxel_volume_mm3) -> (label uint8, info)`, `postprocess(pred) -> uint8`, `dice(a, b)`.

- [ ] **Step 1: Write the failing test**

`tests/test_brain_anatomy_outline.py`:

```python
# tests/test_brain_anatomy_outline.py
import numpy as np
import pytest

import anatobind.anatomy.outline as O


def _seg():
    seg = np.zeros((40, 40, 16), np.int16)
    for k in range(13):                                   # brain in slices 0..12, widest in the middle
        r = 6 + k if k < 6 else 18 - k
        x, y = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
        seg[:, :, k][(x - 20) ** 2 + (y - 20) ** 2 <= r ** 2] = 2
    seg[20, 20, 6] = 0                                    # a hole inside the brain
    seg[2, 2, 6] = 3                                      # a speck away from the brain
    return seg


def test_fill_and_largest_component_removes_holes_and_specks():
    seg = _seg()
    out = O.fill_and_keep_largest(seg > 0)
    assert out[20, 20, 6] and not out[2, 2, 6] and out.dtype == bool
    assert O.fill_and_keep_largest(np.zeros((4, 4, 2), bool)).sum() == 0


def test_supervised_slices_leave_out_the_lowest_two_and_the_top():
    seg = _seg()
    outline = O.fill_and_keep_largest(seg > 0)
    sup = O.supervised_slices(outline, voxel_area_mm2=25.0)          # 5 x 5 mm pixels: areas 25 x count
    assert sup.start == 2 and sup.stop == 12                          # brain up to slice 12 (area > 5 cm2), margin 1 -> last supervised 11
    assert len(O.supervised_slices(np.zeros((4, 4, 3), bool), 1.0)) == 0
    tiny = np.zeros((40, 40, 16), bool)
    tiny[18:22, 18:22, :] = True                                      # 16 voxels x 25 mm2 = 400 mm2 < 5 cm2 everywhere
    assert len(O.supervised_slices(tiny, 25.0)) == 0


def test_outline_label_marks_unsupervised_slices_ignore_and_reports_volume():
    seg = _seg()
    label, info = O.outline_label(seg, voxel_area_mm2=25.0, voxel_volume_mm3=125.0)
    assert label.dtype == np.uint8 and set(np.unique(label)) == {0, 1, 2}
    assert (label[:, :, 0] == O.IGNORE).all() and (label[:, :, 1] == O.IGNORE).all() and (label[:, :, 12] == O.IGNORE).all()
    assert (label[:, :, 15] == O.IGNORE).all() and label[20, 20, 6] == 1 and label[0, 0, 6] == 0 and label[2, 2, 6] == 0
    assert info["supervised"] == [2, 11] and info["usable"] is True and info["volume_ml"] > 300
    small, info2 = O.outline_label(seg[:, :, :3], 25.0, 125.0)
    assert info2["usable"] is False and info2["supervised"] is None and (small == O.IGNORE).all()


def test_postprocess_and_dice():
    pred = np.zeros((10, 10, 3), np.float32)
    pred[2:8, 2:8, :] = 0.9
    pred[5, 5, 1] = 0.0
    pred[0, 0, 0] = 0.7
    m = O.postprocess(pred)
    assert m.dtype == np.uint8 and m[5, 5, 1] == 1 and m[0, 0, 0] == 0
    assert O.dice(m, pred > 0.5) == pytest.approx(2 * 108 / (108 + 108), abs=0.02)
    assert O.dice(np.zeros(3), np.zeros(3)) == 1.0 and O.dice([1, 0], [0, 1]) == 0.0
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_outline.py -q -p no:cacheprovider`
Expected: errors at import or collection (the module does not exist yet).

- [ ] **Step 3: Write `anatobind/anatomy/outline.py`**

```python
"""Brain outline for skull-stripping fastMRI FLAIR stacks (spec 2026-10-02 §6, A10).

Training labels come from the SynthSeg map of the same stack: labels > 0, holes filled slice by slice, the largest
three-dimensional component kept. Only the slices where that outline is trustworthy supervise the model: from slice
LOW up to the slice below the last one with more than MIN_AREA_MM2 of brain; every other slice is the ignore label.
Inference keeps the largest component of the predicted mask and fills its holes slice by slice."""
import numpy as np
from scipy import ndimage

LABELS_JSON = {"background": 0, "brain": 1, "ignore": 2}
IGNORE = 2
LOW = 2                        # the lowest two slices never supervise (s4_probe: SynthSeg misses parts of the brain there)
TOP_MARGIN = 1                 # the last slice with brain is left out too (the vertex is covered only in part)
MIN_AREA_MM2 = 500.0           # 5 cm2
MIN_VOLUME_ML = 300.0          # a stack whose outline holds less brain is a SynthSeg failure and is not used (s4_probe: 14 of 447)


def fill_and_keep_largest(mask):
    """Boolean (x, y, z) mask -> holes filled in every slice, then the largest 26-connected component."""
    m = np.asarray(mask).astype(bool)
    out = np.zeros_like(m)
    for k in range(m.shape[2]):
        out[:, :, k] = ndimage.binary_fill_holes(m[:, :, k])
    comp, n = ndimage.label(out, structure=np.ones((3, 3, 3)))
    if n == 0:
        return out
    sizes = ndimage.sum(out, comp, index=np.arange(1, n + 1))
    return comp == (int(np.argmax(sizes)) + 1)


def supervised_slices(outline, voxel_area_mm2, low=LOW, top_margin=TOP_MARGIN, min_area_mm2=MIN_AREA_MM2):
    """range(low, top) of the slices that supervise: top = the last slice with more than min_area of brain, minus
    the margin, plus one; empty when the stack holds too little brain."""
    areas = np.asarray(outline).astype(bool).sum(axis=(0, 1)) * float(voxel_area_mm2)
    with_brain = np.flatnonzero(areas > min_area_mm2)
    if len(with_brain) == 0:
        return range(0, 0)
    top = int(with_brain.max()) - top_margin + 1
    return range(low, max(low, top))


def outline_label(seg, voxel_area_mm2, voxel_volume_mm3):
    """Training label of one stack: 1 inside the outline, 0 outside, IGNORE on every unsupervised slice.
    Returns (label uint8, info) with info = {volume_ml, supervised: [first, last] or None, usable}."""
    outline = fill_and_keep_largest(np.asarray(seg) > 0)
    volume_ml = float(outline.sum()) * float(voxel_volume_mm3) / 1000.0
    sup = supervised_slices(outline, voxel_area_mm2)
    label = outline.astype(np.uint8)
    for k in range(label.shape[2]):
        if k not in sup:
            label[:, :, k] = IGNORE
    info = {"volume_ml": round(volume_ml, 1), "supervised": [sup.start, sup.stop - 1] if len(sup) else None,
            "usable": volume_ml >= MIN_VOLUME_ML and len(sup) > 0}
    return label, info


def postprocess(pred):
    """Predicted mask (any array, > 0 = brain) -> largest component with holes filled, as uint8."""
    return fill_and_keep_largest(np.asarray(pred) > 0).astype(np.uint8)


def dice(a, b):
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * float((a & b).sum()) / float(s)
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_outline.py -q -p no:cacheprovider`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add anatobind/anatomy/outline.py tests/test_brain_anatomy_outline.py
git commit -m "S4 anatomy: brain outline labels with supervised slices and mask post-processing"
```

---

### Task 5: Evaluation module

**Files:**
- Create: `anatobind/eval/brain_anatomy.py`
- Test: `tests/test_brain_anatomy_eval.py`

**Interfaces:**
- Consumes: Task 1 (`HOST_IDS`, `NAMES`, `to_synthseg`), Task 4 (`fill_and_keep_largest`, `supervised_slices`, `dice`), `anatobind.eval.lookup.{BRAIN_PARENCHYMA, BrainLookup}`, `anatobind.eval.geometry.{CLASS_NAMES, host_class_map}`.
- Produces: `GATES`, `reliable_slices(synthseg, voxel_area_mm2)`, `class_dice(pred, ref, classes=HOST_IDS, slices=None)`, `summarize_dice(per_case)`, `rects_of(row)`, `box_host(lookup, rects)`, `host_agreement(student, synthseg, spacing, rows, reliable)`, `pool_agreement(parts)`, `outline_dice(pred_mask, synthseg, voxel_area_mm2)`, `verdict(host_rate, mean_host_dice, outline)`.

- [ ] **Step 1: Write the failing test**

`tests/test_brain_anatomy_eval.py`:

```python
# tests/test_brain_anatomy_eval.py
import numpy as np
import pytest

import anatobind.eval.brain_anatomy as E
from anatobind.anatomy.labels import to_student


def _maps():
    seg = np.zeros((30, 30, 8), np.int16)
    seg[2:28, 2:28, 1:7] = 2                 # left white matter block (RAS-agnostic here: just labels)
    seg[15:28, 2:28, 1:7] = 41               # right white matter on the far half
    seg[10:20, 10:20, 3:5] = 10              # left thalamus
    return seg


def test_class_dice_and_summary_skip_classes_absent_on_both_sides():
    seg = _maps()
    stu = to_student(seg)
    pred = stu.copy()
    pred[10:20, 10:15, 3:5] = 1              # half the thalamus called white matter
    d = E.class_dice(pred, stu, slices=range(1, 7))
    assert d[5] == pytest.approx(2 * 100 / (100 + 200)) and d[2] == 1.0 and d[6] is None and d[9] is None
    s = E.summarize_dice([d, E.class_dice(stu, stu)])
    assert s["per_class"]["thalamus_left"] == pytest.approx((d[5] + 1.0) / 2) and s["per_class"]["thalamus_right"] is None
    assert s["n_cases"] == 2 and 0.9 < s["mean_host_dice"] <= 1.0


def test_reliable_slices_follow_the_outline_rule():
    seg = _maps()
    r = E.reliable_slices(seg, voxel_area_mm2=4.0)      # 26 x 26 x 4 mm2 = 2704 mm2 > 5 cm2 in slices 1..6 -> top 6, margin 1
    assert list(r) == [2, 3, 4, 5]
    assert list(E.reliable_slices(np.zeros((5, 5, 3)), 4.0)) == []


def test_host_agreement_evaluates_boxes_inside_the_reliable_slices_only():
    seg = _maps()
    stu = to_student(seg)
    wrong = stu.copy()
    wrong[10:20, 10:20, 3:5] = 1             # the student misses the thalamus entirely
    rows = [{"lesion_id": 1, "x0": 11, "x1": 14, "y0": 11, "y1": 14, "z0": 3, "z1": 4},      # inside the thalamus, reliable slices
            {"lesion_id": 2, "x0": 3, "x1": 6, "y0": 3, "y1": 6, "z0": 2, "z1": 2},         # left white matter
            {"lesion_id": 3, "x0": 3, "x1": 6, "y0": 3, "y1": 6, "z0": 0, "z1": 1}]         # touches slice 0: not reliable
    rel = E.reliable_slices(seg, 4.0)
    out = E.host_agreement(wrong, seg, (2.0, 2.0, 5.0), rows, rel)
    assert out["n_lesions"] == 3 and out["n_evaluated"] == 2 and out["n_outside_reliable"] == 1
    assert out["n_agree"] == 1 and out["rate"] == 0.5
    assert out["pairs"][0] == {"lesion_id": 1, "student": "white_matter", "synthseg": "thalamus"}
    same = E.host_agreement(stu, seg, (2.0, 2.0, 5.0), rows, rel)
    assert same["rate"] == 1.0
    pooled = E.pool_agreement([out, same])
    assert pooled == {"n_lesions": 6, "n_evaluated": 4, "n_agree": 3, "n_outside_reliable": 2, "rate": 0.75}


def test_outline_dice_and_verdict():
    seg = _maps()
    mask = (seg > 0).astype(np.uint8)
    assert E.outline_dice(mask, seg, 4.0) == 1.0
    mask[2:10] = 0
    assert 0.8 < E.outline_dice(mask, seg, 4.0) < 1.0
    assert E.outline_dice(mask, np.zeros_like(seg), 4.0) is None
    v = E.verdict(0.95, 0.85, 0.98)
    assert v["pass"] and v["gates"] == {"host_agreement": 0.9, "mean_host_dice": 0.8, "outline_dice": 0.97}
    assert not E.verdict(0.89, 0.85, 0.98)["pass"] and not E.verdict(0.95, None, 0.98)["pass"]
    assert E.verdict(0.90, 0.80, 0.97)["pass"]                                                 # the gate values pass
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_eval.py -q -p no:cacheprovider`
Expected: errors at import or collection (the module does not exist yet).

- [ ] **Step 3: Write `anatobind/eval/brain_anatomy.py`**

```python
"""Evaluation of the S4 anatomy model (spec 2026-10-02 §7, A11, A13).

All numbers are agreement with SynthSeg pseudo-labels, NOT_EVIDENCE. On the fastMRI stacks only the reliable slices
count (the same slices that supervise the outline model: from slice 2 up to the slice below the last one with brain);
the host agreement is the share of fastMRI+ lesion boxes for which the box lookup on the student's anatomy names the
same host class as the lookup on SynthSeg's anatomy."""
import numpy as np

from anatobind.anatomy.labels import HOST_IDS, NAMES, to_synthseg
from anatobind.anatomy.outline import dice as mask_dice, fill_and_keep_largest, supervised_slices
from anatobind.eval.geometry import CLASS_NAMES, host_class_map
from anatobind.eval.lookup import BRAIN_PARENCHYMA, BrainLookup

GATES = {"host_agreement": 0.90, "mean_host_dice": 0.80, "outline_dice": 0.97}      # A11 (1), (2), (3)


def reliable_slices(synthseg, voxel_area_mm2):
    """The slices of a fastMRI stack whose SynthSeg labels are trusted: outline rule of the skull-strip model."""
    return supervised_slices(fill_and_keep_largest(np.asarray(synthseg) > 0), voxel_area_mm2)


def class_dice(pred, ref, classes=HOST_IDS, slices=None):
    """{class id: Dice} over the given slices (all when None); None for a class empty in both maps."""
    p, r = np.asarray(pred), np.asarray(ref)
    if slices is not None:
        p, r = p[:, :, list(slices)], r[:, :, list(slices)]
    out = {}
    for c in classes:
        a, b = p == c, r == c
        s = int(a.sum()) + int(b.sum())
        out[int(c)] = None if s == 0 else 2.0 * float((a & b).sum()) / s
    return out


def summarize_dice(per_case):
    """per_case: list of {class id: Dice | None}. Mean per class over the cases where the class occurs, and the mean of
    those per-class means over HOST_IDS (mean_host_dice)."""
    per_class = {}
    for c in HOST_IDS:
        vals = [d[c] for d in per_case if d.get(c) is not None]
        per_class[NAMES[c]] = float(np.mean(vals)) if vals else None
    host = [v for v in per_class.values() if v is not None]
    return {"per_class": per_class, "mean_host_dice": float(np.mean(host)) if host else None, "n_cases": len(per_case)}


def rects_of(row):
    """Registry row (x0, x1 half-open columns; y0, y1 half-open rows; z0..z1 inclusive slices) -> lookup rectangles."""
    return [(int(row["x0"]), int(row["x1"]), int(row["y0"]), int(row["y1"]), s) for s in range(int(row["z0"]), int(row["z1"]) + 1)]


def box_host(lookup, rects):
    """Host class name of a box by argmax overlap over the parenchyma labels, nearest when no overlap; None off the grid."""
    label, _ = lookup.host(rects)
    if label is None:
        return None
    cls = int(host_class_map(np.array([label]))[0])
    return CLASS_NAMES[cls - 1] if cls > 0 else None


def host_agreement(student, synthseg, spacing, rows, reliable):
    """student: compact labels; synthseg: SynthSeg map on the same grid; rows: registry rows of this stack.
    Boxes with a slice outside `reliable` are counted but not evaluated."""
    a = BrainLookup(to_synthseg(student), spacing, BRAIN_PARENCHYMA)
    b = BrainLookup(np.asarray(synthseg), spacing, BRAIN_PARENCHYMA)
    n_eval = n_agree = n_outside = 0
    pairs = []
    for row in rows:
        if any(s not in reliable for s in range(int(row["z0"]), int(row["z1"]) + 1)):
            n_outside += 1
            continue
        rects = rects_of(row)
        hs, hr = box_host(a, rects), box_host(b, rects)
        n_eval += 1
        n_agree += int(hs == hr)
        pairs.append({"lesion_id": int(row["lesion_id"]), "student": hs, "synthseg": hr})
    return {"n_lesions": len(rows), "n_evaluated": n_eval, "n_agree": n_agree, "n_outside_reliable": n_outside,
            "rate": (n_agree / n_eval) if n_eval else None, "pairs": pairs}


def pool_agreement(parts):
    n_eval = sum(p["n_evaluated"] for p in parts)
    n_agree = sum(p["n_agree"] for p in parts)
    return {"n_lesions": sum(p["n_lesions"] for p in parts), "n_evaluated": n_eval, "n_agree": n_agree,
            "n_outside_reliable": sum(p["n_outside_reliable"] for p in parts), "rate": (n_agree / n_eval) if n_eval else None}


def outline_dice(pred_mask, synthseg, voxel_area_mm2):
    """Dice of a predicted brain mask against the SynthSeg outline on the reliable slices."""
    ref = fill_and_keep_largest(np.asarray(synthseg) > 0)
    sl = list(reliable_slices(synthseg, voxel_area_mm2))
    if not sl:
        return None
    return mask_dice(np.asarray(pred_mask)[:, :, sl] > 0, ref[:, :, sl])


def verdict(host_rate, mean_host_dice, outline):
    """A11: the three lines must all pass; a missing number fails its line."""
    checks = {"host_agreement": host_rate is not None and host_rate >= GATES["host_agreement"],
              "mean_host_dice": mean_host_dice is not None and mean_host_dice >= GATES["mean_host_dice"],
              "outline_dice": outline is not None and outline >= GATES["outline_dice"]}
    return {"pass": all(checks.values()), **checks, "values": {"host_agreement": host_rate, "mean_host_dice": mean_host_dice,
                                                                "outline_dice": outline}, "gates": dict(GATES)}
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_eval.py -q -p no:cacheprovider`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add anatobind/eval/brain_anatomy.py tests/test_brain_anatomy_eval.py
git commit -m "S4 anatomy: evaluation (Dice on reliable slices, box host agreement, outline Dice, A11 verdict)"
```

---

### Task 6: Inference chain

**Files:**
- Create: `anatobind/infer/brain_anatomy.py`
- Test: `tests/test_brain_anatomy_infer.py`

**Interfaces:**
- Consumes: Task 1 (`NAMES`, `to_synthseg`), Task 4 (`postprocess`), Task 5 (`reliable_slices`), `anatobind.bind.brain_lookup.BrainBinder`, `anatobind.data_engine.fastmri.rss_h5_to_nifti`, `anatobind.eval.lesion_boxes.load_label_map`, `anatobind.infer.knee.nnunet_env`.
- Produces: `OUTLINE = {id: 908, config: '2d', trainer: 'nnUNetTrainer_250epochs'}`, `STUDENT = {id: 907, config: '3d_fullres', trainer: 'nnUNetTrainer_250epochs_NoMirroring'}` (spec A17), `TRAINERS` (by dataset id), `FOLDS = [0]`, `run_nnunet(dataset_id, config, in_dir, out_dir, folds, gpu)`, `stage_input` (float32 staging), `check_box(box, shape)`, `run(out_dir, gpu, h5=None, nifti=None, box=None, predict=run_nnunet) -> record` (a missing input is refused before anything is created; a stage that fails raises `RuntimeError` naming the stage and the directory that keeps the partial output).

- [ ] **Step 1: Write the failing test**

`tests/test_brain_anatomy_infer.py`:

```python
# tests/test_brain_anatomy_infer.py
import json

import nibabel as nib
import numpy as np
import pytest

import anatobind.infer.brain_anatomy as I
from anatobind.data_engine.fastmri import rss_affine


def _stack(tmp_path):
    data = np.zeros((40, 40, 8), np.float32)
    data[5:35, 5:35, :] = 200.0                        # "head"
    data[10:30, 10:30, :] = 400.0                      # "brain"
    aff = rss_affine(2.0, 2.0, 5.0, data.shape)          # 2 mm pixels: the toy brain covers more than 5 cm2 per slice
    p = tmp_path / "stack.nii.gz"
    nib.save(nib.Nifti1Image(data, aff), str(p))
    return p


def _fake_predict(calls):
    def predict(dataset_id, config, in_dir, out_dir, folds, gpu):
        img = nib.load(str(in_dir / "case_0000.nii.gz"))
        data = np.asarray(img.dataobj)
        out_dir.mkdir(parents=True)
        calls.append((dataset_id, config, folds, gpu, float(data.max()), float(data[7, 7, 0])))
        if dataset_id == 908:
            lab = (data > 300).astype(np.uint8)                      # the brain; a speck outside
            lab[0, 0, 0] = 1
        else:
            lab = np.zeros(data.shape, np.uint8)
            lab[10:20, 10:30, :] = 1                                 # white matter left
            lab[20:30, 10:30, :] = 2                                 # white matter right
            lab[14:16, 14:16, 3:5] = 5                               # thalamus left
        nib.save(nib.Nifti1Image(lab, img.affine), str(out_dir / "case.nii.gz"))
    return predict


def test_the_chain_strips_the_skull_binds_a_box_and_refuses_an_existing_out(tmp_path):
    p = _stack(tmp_path)
    calls = []
    rec = I.run(tmp_path / "out", 2, nifti=p, box=(14, 14, 3, 16, 16, 5), predict=_fake_predict(calls))
    assert [c[:4] for c in calls] == [(908, "2d", [0], 2), (907, "3d_fullres", [0], 2)]
    assert calls[1][5] == 0.0 and calls[0][5] == 200.0                 # the student saw the head set to zero
    anatomy = np.asarray(nib.load(rec["anatomy"]).dataobj)
    assert anatomy.dtype == np.int16 and set(np.unique(anatomy)) == {0, 2, 41, 10}
    mask = np.asarray(nib.load(rec["brain_mask"]).dataobj)
    assert mask[0, 0, 0] == 0 and mask[15, 15, 2] == 1                 # speck removed, brain kept
    assert rec["class_volumes_ml"]["thalamus_left"] == pytest.approx(8 * 2.0 * 2.0 * 5 / 1000, abs=1e-3)
    assert rec["binding"]["host"] == "thalamus" and rec["binding"]["host_rule"] == "overlap" and rec["box"] == [14, 14, 3, 16, 16, 5]
    assert rec["reliable_slices"] == [2, 6] and "NOT_EVIDENCE" in rec["anatomy_source"]
    assert json.loads((tmp_path / "out" / "record.json").read_text())["brain_ml"] == rec["brain_ml"]
    with pytest.raises(FileExistsError, match="use a new output directory"):
        I.run(tmp_path / "out", 2, nifti=p, predict=_fake_predict(calls))
    with pytest.raises(ValueError, match="exactly one"):
        I.run(tmp_path / "out2", 2, predict=_fake_predict(calls))


def test_nothing_is_created_for_a_missing_input_and_a_failed_stage_names_itself(tmp_path):
    p = _stack(tmp_path)
    with pytest.raises(FileNotFoundError, match="is missing"):
        I.run(tmp_path / "a", 0, nifti=tmp_path / "nope.nii.gz", predict=_fake_predict([]))
    with pytest.raises(FileNotFoundError):
        I.run(tmp_path / "a", 0, h5=tmp_path / "nope.h5", predict=_fake_predict([]))
    assert not (tmp_path / "a").exists()                            # nothing staged: the corrected rerun may use the same name

    def broken(dataset_id, config, in_dir, out_dir, folds, gpu):
        if dataset_id == 907:
            raise OSError("no GPU")
        _fake_predict([])(dataset_id, config, in_dir, out_dir, folds, gpu)

    with pytest.raises(RuntimeError, match="the student prediction failed; the partial output stays in .*rerun into a new output directory"):
        I.run(tmp_path / "b", 0, nifti=p, predict=broken)
    assert (tmp_path / "b" / "brain_mask.nii.gz").exists() and not (tmp_path / "b" / "anatomy.nii.gz").exists()
    with pytest.raises(RuntimeError, match="checking the box failed"):
        I.run(tmp_path / "c", 0, nifti=p, box=(5, 5, 0, 5, 9, 2), predict=_fake_predict([]))        # empty along x
    assert I.check_box((0, 0, 0, 40, 40, 8), (40, 40, 8)) == (0, 0, 0, 40, 40, 8)
    with pytest.raises(ValueError, match="outside the grid"):
        I.check_box((0, 0, 0, 41, 40, 8), (40, 40, 8))


def test_the_nnunet_command_line_and_the_float32_staging(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(I.subprocess, "run", lambda cmd, check, env: seen.update(cmd=cmd, check=check, gpu=env["CUDA_VISIBLE_DEVICES"]))
    I.run_nnunet(908, "2d", tmp_path / "in", tmp_path / "o", [0], 3)
    assert seen["cmd"] == ["nice", "-n", "19", "nnUNetv2_predict", "-i", str(tmp_path / "in"), "-o", str(tmp_path / "o"), "-d", "908", "-c", "2d",
                           "-tr", "nnUNetTrainer_250epochs", "-f", "0", "-npp", "2", "-nps", "2", "--disable_progress_bar"]
    assert seen["check"] is True and seen["gpu"] == "3"
    I.run_nnunet(907, "3d_fullres", tmp_path / "in", tmp_path / "o", [0], 3)       # sided classes: the trainer without mirroring
    assert seen["cmd"][seen["cmd"].index("-tr") + 1] == "nnUNetTrainer_250epochs_NoMirroring"
    ints = tmp_path / "int16.nii.gz"
    nib.save(nib.Nifti1Image(np.full((4, 4, 4), 7, np.int16), np.eye(4)), str(ints))
    img = I.stage_input(None, ints, tmp_path / "staged")
    assert img.get_data_dtype() == np.float32 and float(np.asarray(img.dataobj).max()) == 7.0
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_infer.py -q -p no:cacheprovider`
Expected: errors at import or collection (the module does not exist yet).

- [ ] **Step 3: Write `anatobind/infer/brain_anatomy.py`**

```python
"""S4 inference chain (spec 2026-10-02 §8): a fastMRI FLAIR stack -> brain outline -> skull-stripped stack -> student
anatomy -> a label map in SynthSeg values that BrainBinder reads like a SynthSeg map; optional binding of one box.

Inputs: a fastMRI h5 (RSS reconstruction, the frame of S2's `rss_h5_to_nifti`) or a NIfTI stack already in that frame.
The output directory must not exist, and a run that fails half-way leaves its partial output there (nothing is ever
deleted): the error says so and the rerun takes a new directory. The student sees the stack with everything outside
the outline set to zero; its labels are not clipped to the outline afterwards. Everything downstream of the two nnU-Net
models is a pseudo-label: NOT_EVIDENCE."""
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.anatomy.labels import NAMES, to_synthseg
from anatobind.anatomy.outline import postprocess
from anatobind.bind.brain_lookup import BrainBinder
from anatobind.data_engine.fastmri import rss_h5_to_nifti
from anatobind.eval.brain_anatomy import reliable_slices
from anatobind.eval.lesion_boxes import load_label_map
from anatobind.infer.knee import nnunet_env

OUTLINE = {"id": 908, "config": "2d", "trainer": "nnUNetTrainer_250epochs"}
# The student's classes have a side. nnU-Net's default mirroring flips the image and the labels together, so a left
# structure appears on either side under the same label and the sides cannot be learned: the student is trained and
# run without mirroring (spec A17). The outline has no side and keeps the default.
STUDENT = {"id": 907, "config": "3d_fullres", "trainer": "nnUNetTrainer_250epochs_NoMirroring"}
TRAINERS = {m["id"]: m["trainer"] for m in (OUTLINE, STUDENT)}
FOLDS = [0]


def run_nnunet(dataset_id, config, in_dir, out_dir, folds, gpu):
    cmd = ["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir), "-d", str(dataset_id), "-c", config, "-tr", TRAINERS[dataset_id],
           "-f", *[str(f) for f in folds], "-npp", "2", "-nps", "2", "--disable_progress_bar"]
    subprocess.run(["nice", "-n", "19", *cmd], check=True, env=nnunet_env(gpu))


def stage_input(h5, nifti, out):
    """The RSS stack as float32 in <out>/input_outline/case_0000.nii.gz; returns the loaded image."""
    d = out / "input_outline"
    d.mkdir(parents=True)
    if h5 is not None:
        rss_h5_to_nifti(h5, d / "case_0000.nii.gz", pad_to_slices=0)
    else:
        src = nib.load(str(nifti))
        nib.save(nib.Nifti1Image(np.asarray(src.dataobj).astype(np.float32), src.affine), str(d / "case_0000.nii.gz"))
    return nib.load(str(d / "case_0000.nii.gz"))


def _stage(name, out, fn):
    """Run one stage; a failure is re-raised with the stage's name and the fact that the partial output stays."""
    try:
        return fn()
    except Exception as e:
        raise RuntimeError(f"{name} failed; the partial output stays in {out} (nothing is deleted here): "
                           f"rerun into a new output directory") from e


def check_box(box, shape):
    """(x0, y0, z0, x1, y1, z1) as ints; refuses a box that is empty or leaves the grid."""
    x0, y0, z0, x1, y1, z1 = [int(v) for v in box]
    if not (0 <= x0 < x1 <= shape[0] and 0 <= y0 < y1 <= shape[1] and 0 <= z0 < z1 <= shape[2]):
        raise ValueError(f"box {[x0, y0, z0, x1, y1, z1]} is empty or outside the grid {tuple(shape)}")
    return x0, y0, z0, x1, y1, z1


def run(out_dir, gpu, h5=None, nifti=None, box=None, predict=run_nnunet):
    """box: (x0, y0, z0, x1, y1, z1), half-open, on the stack's (col, row, slice) grid; bound with BrainBinder."""
    if (h5 is None) == (nifti is None):
        raise ValueError("give exactly one of h5 and nifti")
    source = Path(h5 if h5 is not None else nifti)
    if not source.is_file():
        raise FileNotFoundError(f"{source} is missing")
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; nothing is deleted or overwritten here, use a new output directory")
    img = _stage("staging the input", out, lambda: stage_input(h5, nifti, out))
    data = np.asarray(img.dataobj).astype(np.float32)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if box is not None:
        box = _stage("checking the box", out, lambda: check_box(box, data.shape))
    _stage("the outline prediction", out,
           lambda: predict(OUTLINE["id"], OUTLINE["config"], out / "input_outline", out / "pred_outline", FOLDS, gpu))
    mask = postprocess(load_label_map(out / "pred_outline" / "case.nii.gz"))
    if mask.shape != data.shape:
        raise ValueError(f"outline {mask.shape} and stack {data.shape} differ")
    nib.save(nib.Nifti1Image(mask, img.affine), str(out / "brain_mask.nii.gz"))
    (out / "input_student").mkdir()
    nib.save(nib.Nifti1Image(data * mask, img.affine), str(out / "input_student" / "case_0000.nii.gz"))
    _stage("the student prediction", out,
           lambda: predict(STUDENT["id"], STUDENT["config"], out / "input_student", out / "pred_student", FOLDS, gpu))
    student = load_label_map(out / "pred_student" / "case.nii.gz")
    if student.shape != data.shape:
        raise ValueError(f"student {student.shape} and stack {data.shape} differ")
    anatomy = to_synthseg(student)
    nib.save(nib.Nifti1Image(anatomy, img.affine), str(out / "anatomy.nii.gz"))
    voxel_ml = float(np.prod(zooms)) / 1000.0
    reliable = reliable_slices(anatomy, zooms[0] * zooms[1])
    record = {"input": str(h5 if h5 is not None else nifti), "shape": list(data.shape), "spacing_mm": list(zooms),
              "brain_ml": round(float(mask.sum()) * voxel_ml, 1),
              "class_volumes_ml": {NAMES[c]: round(float((student == c).sum()) * voxel_ml, 3) for c in range(1, 15)},
              "reliable_slices": [reliable.start, reliable.stop - 1] if len(reliable) else None,
              "anatomy": str(out / "anatomy.nii.gz"), "brain_mask": str(out / "brain_mask.nii.gz"),
              "anatomy_source": "S4 student on FLAIR, trained on SynthSeg pseudo-labels (NOT_EVIDENCE)"}
    if box is not None:
        x0, y0, z0, x1, y1, z1 = box
        sl = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
        record["box"] = [x0, y0, z0, x1, y1, z1]
        record["binding"] = BrainBinder(anatomy, zooms).bind(sl, np.ones((x1 - x0, y1 - y0, z1 - z0), bool))
    (out / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    return record
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_infer.py -q -p no:cacheprovider`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add anatobind/infer/brain_anatomy.py tests/test_brain_anatomy_infer.py
git commit -m "S4 anatomy: inference chain (RSS, outline, stripped stack, student, SynthSeg-valued map, box binding)"
```

---

### Task 7: Preparation script and the data build

**Files:**
- Create: `scripts/brain_anatomy_prepare.py`
- Create: `docs/verification/2026-10-02/brain_anatomy_flair/checks/sim_montage.py`
- Test: `tests/test_brain_anatomy_prepare.py`

**Interfaces:**
- Consumes: Tasks 1–4, `anatobind.nnunet.brain_disease.{FM, binary_label}`, `anatobind.nnunet.brain_lesion.{assign_normal_folds, make_splits}`, `anatobind.data_engine.fastmri.rss_h5_to_nifti`, `anatobind.data_engine.fastmri_knee.volume_geometry`, `anatobind.infer.knee.NNUNET_ROOT`.
- Produces: `scripts/brain_anatomy_prepare.py --stage sources | simulate | dataset907 | dataset908 | splits [--work] [--workers]`; functions `load_1mm_ras`, `check_one_grid`, `case_arrays`, `sample_seed`, `simulate_case`, `stage_sources`, `stage_simulate`, `stage_dataset907`, `h5_of`, `fastmri_stems`, `outline_from_seg`, `write_outline_case(stem, image_path, label, seg, base, split)` (refuses an RSS image off its SynthSeg map's grid: shape and affine), `stage_dataset908(raw_root, work, seg_dir, convert=None, patient=None)`, `build_dataset908`, `patient_splits`, `write_splits`, `stage_splits`; constants `WORK`, `K_TRAIN = 4`, `DATASET907`, `DATASET908`.
- Data layout produced: `<work>/cases.json`; `<work>/sim/{train,test}/<case>_s<k>_0000.nii.gz`, `<case>_s<k>.nii.gz`, `<case>_s<k>.json`; `<work>/sim/manifest.json`; nnU-Net raw `Dataset907_BrainAnatomyFLAIR/{imagesTr, labelsTr, imagesTs, dataset.json, cases.json}` and `Dataset908_FastMRIBrainOutline/{imagesTr, labelsTr, imagesTs, dataset.json, cases.json, excluded.json}`; `splits_final.json` of both.

The controller runs the data build after the task review (Steps 8–13); the implementer stops after Step 7.

- [ ] **Step 1: Write the failing test**

`tests/test_brain_anatomy_prepare.py`:

```python
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_prepare.py -q -p no:cacheprovider`
Expected: errors at import or collection (the module does not exist yet).

- [ ] **Step 3: Write `scripts/brain_anatomy_prepare.py`**

```python
#!/usr/bin/env python
# scripts/brain_anatomy_prepare.py
"""Build the data of the S4 anatomy model (spec 2026-10-02 §3, §5, §6; decisions A2-A10). Nothing is ever rebuilt in
place: every stage refuses an output that exists, and no stage deletes or renames a file.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_prepare.py --stage sources
  ... --stage simulate --workers 8      K simulated stacks per training case, one per test case -> <work>/sim/{train,test}
  ... --stage dataset907                nnU-Net raw Dataset907_BrainAnatomyFLAIR (symlinks into <work>/sim)
  ... --stage dataset908                fastMRI RSS stacks + outline labels -> Dataset908_FastMRIBrainOutline
  ... --stage splits                    splits_final.json of both datasets (five folds by patient; fold 0 is trained)
"""
import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.anatomy import outline as OUT  # noqa: E402
from anatobind.anatomy.labels import LABELS_JSON, to_student  # noqa: E402
from anatobind.anatomy.simulate import simulate, stack_affine  # noqa: E402
from anatobind.anatomy.sources import all_cases, split_by_patient  # noqa: E402
from anatobind.data_engine.fastmri import rss_h5_to_nifti  # noqa: E402
from anatobind.data_engine.fastmri_knee import volume_geometry  # noqa: E402
from anatobind.infer.knee import NNUNET_ROOT  # noqa: E402
from anatobind.nnunet.brain_disease import FM, binary_label  # noqa: E402
from anatobind.nnunet.brain_lesion import assign_normal_folds, make_splits  # noqa: E402

WORK = FM / "derived/brain_anatomy"
K_TRAIN = 4
DATASET907 = "Dataset907_BrainAnatomyFLAIR"
DATASET908 = "Dataset908_FastMRIBrainOutline"
FASTMRI_SEG = FM / "derived/synthseg/fastmri_brain/seg_native"
KROOT = FM / "fastMRI_lh_brain_knee/kspace/brain"
TEST_SHARE = 0.2
GRID_TOL = 1e-3


def load_1mm_ras(path, order):
    """A NIfTI as a (x, y, z) RAS array at 1 mm isotropic spacing (resampled when the file is not), with its zooms."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if not np.allclose(zooms, 1.0, atol=1e-3):
        data = ndimage.zoom(data.astype(np.float32 if order else data.dtype), zooms, order=order, mode="constant", cval=0.0, grid_mode=False)
    return (data.astype(np.float32) if order else data), zooms


def check_one_grid(path, ref_path, tol=GRID_TOL):
    """Refuse a file that is not on the reference file's grid: same shape, affines within tol (read from the headers)."""
    a, b = nib.load(str(path)), nib.load(str(ref_path))
    if a.shape != b.shape or float(np.abs(a.affine - b.affine).max()) > tol:
        raise ValueError(f"{path} is not on the grid of {ref_path} (shape {a.shape} against {b.shape}, or the affines differ)")


def case_arrays(rec):
    """(flair, student labels, lesion mask or None) at 1 mm RAS for one case record. The teacher map and the lesion
    mask must lie on the FLAIR's grid; labels are never carried between grids."""
    check_one_grid(rec["anatomy"], rec["flair"])
    flair, _ = load_1mm_ras(rec["flair"], 1)
    seg, _ = load_1mm_ras(rec["anatomy"], 0)
    lesion = None
    if rec.get("lesion"):
        check_one_grid(rec["lesion"], rec["flair"])
        raw, _ = load_1mm_ras(rec["lesion"], 0)
        lesion = binary_label(raw, tuple(rec["lesion_values"]), str(rec["lesion"]))
    return flair, to_student(seg), lesion


def sample_seed(case, k):
    return int(hashlib.sha256(f"{case}:{k}".encode()).hexdigest()[:8], 16)


def simulate_case(job):
    """job = (case, record with str paths, out_dir, ks). Writes <case>_s<k>_0000.nii.gz, <case>_s<k>.nii.gz and
    <case>_s<k>.json; returns one manifest row per sample."""
    case, rec, out_dir, ks = job
    out_dir = Path(out_dir)
    flair, student, lesion = case_arrays(rec)
    rows = []
    for k in ks:
        stack, labels, params = simulate(flair, student, lesion, np.random.default_rng(sample_seed(case, k)))
        aff = stack_affine(params["inplane_mm"], stack.shape)
        name = f"{case}_s{k}"
        nib.save(nib.Nifti1Image(stack, aff), str(out_dir / f"{name}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(labels, aff), str(out_dir / f"{name}.nii.gz"))
        row = {"sample": name, "case": case, "source": rec["source"], "patient": rec["patient"], "shape": list(stack.shape),
               "ignore_voxels": int((labels == LABELS_JSON["ignore"]).sum()),
               **{key: (list(v) if isinstance(v, tuple) else v) for key, v in params.items()}}
        (out_dir / f"{name}.json").write_text(json.dumps(row, indent=1))
        rows.append(row)
    return rows


def _refuse(path):
    if Path(path).exists():
        raise FileExistsError(f"{path} exists; outputs are never rebuilt in place")


def stage_sources(work):
    work = Path(work)
    _refuse(work / "cases.json")
    cases = all_cases()
    split = split_by_patient(cases, TEST_SHARE, seed=0)
    work.mkdir(parents=True, exist_ok=True)
    out = {c: {**{k: (str(v) if isinstance(v, Path) else (list(v) if isinstance(v, tuple) else v)) for k, v in r.items()}, "split": split[c]}
           for c, r in cases.items()}
    (work / "cases.json").write_text(json.dumps(out, indent=1))
    for source in sorted({r["source"] for r in cases.values()}):
        n = {s: sum(1 for c, r in cases.items() if r["source"] == source and split[c] == s) for s in ("train", "test")}
        p = {s: len({r["patient"] for c, r in cases.items() if r["source"] == source and split[c] == s}) for s in ("train", "test")}
        print(f"{source}: train {n['train']} cases / {p['train']} patients, test {n['test']} cases / {p['test']} patients")
    print(f"wrote {work / 'cases.json'}: {len(cases)} cases")


def stage_simulate(work, workers, k_train=K_TRAIN):
    work = Path(work)
    cases = json.loads((work / "cases.json").read_text())
    _refuse(work / "sim")
    (work / "sim" / "train").mkdir(parents=True)
    (work / "sim" / "test").mkdir()
    jobs = [(c, r, work / "sim" / r["split"], list(range(k_train)) if r["split"] == "train" else [0]) for c, r in sorted(cases.items())]
    rows = []
    with Pool(workers, initializer=os.nice, initargs=(19,)) as pool:
        for i, part in enumerate(pool.imap_unordered(simulate_case, jobs), start=1):
            rows.extend(part)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} cases simulated", flush=True)
    rows.sort(key=lambda r: r["sample"])
    (work / "sim" / "manifest.json").write_text(json.dumps(rows, indent=1))
    n_train = sum(1 for r in rows if cases[r["case"]]["split"] == "train")
    print(f"wrote {work / 'sim'}: {n_train} training samples, {len(rows) - n_train} test samples")


def stage_dataset907(raw_root, work):
    work, base = Path(work), Path(raw_root) / DATASET907
    _refuse(base)
    cases = json.loads((work / "cases.json").read_text())
    rows = json.loads((work / "sim" / "manifest.json").read_text())
    (base / "imagesTr").mkdir(parents=True)
    (base / "labelsTr").mkdir()
    (base / "imagesTs").mkdir()
    info, n_tr = {}, 0
    for r in rows:
        split = cases[r["case"]]["split"]
        src = work / "sim" / split
        if split == "train":
            os.symlink((src / f"{r['sample']}_0000.nii.gz").resolve(), base / "imagesTr" / f"{r['sample']}_0000.nii.gz")
            os.symlink((src / f"{r['sample']}.nii.gz").resolve(), base / "labelsTr" / f"{r['sample']}.nii.gz")
            n_tr += 1
        else:
            os.symlink((src / f"{r['sample']}_0000.nii.gz").resolve(), base / "imagesTs" / f"{r['sample']}_0000.nii.gz")
        info[r["sample"]] = {"case": r["case"], "patient": r["patient"], "source": r["source"], "split": split}
    meta = {"channel_names": {"0": "FLAIR"}, "labels": LABELS_JSON, "numTraining": n_tr, "file_ending": ".nii.gz"}
    (base / "dataset.json").write_text(json.dumps(meta, indent=1))
    (base / "cases.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {base}: {n_tr} training samples, {len(info) - n_tr} test samples")
    return base


def h5_of(stem):
    for split in ("multicoil_train", "multicoil_val"):
        p = KROOT / split / f"{stem}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def fastmri_stems(seg_dir=FASTMRI_SEG):
    return sorted(p.name[:-len("_seg.nii.gz")] for p in Path(seg_dir).glob("file_brain_AXFLAIR_*_seg.nii.gz"))


def outline_from_seg(seg_path):
    """(label, info, SynthSeg image) of one stack: the outline rule decides usability before anything is written."""
    seg = nib.load(str(seg_path))
    z = tuple(float(v) for v in seg.header.get_zooms()[:3])
    label, info = OUT.outline_label(np.asarray(seg.dataobj), z[0] * z[1], z[0] * z[1] * z[2])
    return label, info, seg


def write_outline_case(stem, image_path, label, seg, base, split):
    """Place one usable stack's label (training stacks only). image_path is the RSS NIfTI already written into the
    dataset folder; it must lie on the grid of the SynthSeg map the label was made from (shape and affine), else the
    stage stops: a label is never carried between grids."""
    img = nib.load(str(image_path))
    if img.shape != seg.shape or float(np.abs(img.affine - seg.affine).max()) > GRID_TOL:
        raise ValueError(f"{stem}: the RSS image is not on the grid of its SynthSeg map (shape {img.shape} against {seg.shape}, or the "
                         f"affines differ); the half-built dataset stays in {base} and is not reused")
    if split == "train":
        nib.save(nib.Nifti1Image(label, img.affine), str(Path(base) / "labelsTr" / f"{stem}.nii.gz"))


def stage_dataset908(raw_root, work, seg_dir=FASTMRI_SEG, convert=None, patient=None):
    """convert(stem, out_path) writes the RSS NIfTI (default: rss_h5_to_nifti of the fastMRI h5, pad_to_slices=0);
    patient: {stem: patient id} (default: the h5 header's patient_id)."""
    base = Path(raw_root) / DATASET908
    _refuse(base)
    stems = fastmri_stems(seg_dir)
    convert = convert or (lambda stem, out: rss_h5_to_nifti(h5_of(stem), out, pad_to_slices=0))
    patient = patient or {s: volume_geometry(h5_of(s))["patient_id"] for s in stems}
    return build_dataset908(base, stems, patient, seg_dir, convert)


def build_dataset908(base, stems, patient, seg_dir, convert):
    base = Path(base)
    (base / "imagesTr").mkdir(parents=True)
    (base / "labelsTr").mkdir()
    (base / "imagesTs").mkdir()
    split = split_by_patient({s: {"source": "fastmri", "patient": patient[s]} for s in stems}, TEST_SHARE, seed=0)
    info, excluded = {}, {}
    for i, s in enumerate(stems, start=1):
        label, res, seg = outline_from_seg(Path(seg_dir) / f"{s}_seg.nii.gz")
        if res["usable"]:
            folder = "imagesTr" if split[s] == "train" else "imagesTs"
            img_path = base / folder / f"{s}_0000.nii.gz"
            convert(s, img_path)
            write_outline_case(s, img_path, label, seg, base, split[s])
            info[s] = {"patient": patient[s], "split": split[s], **res}
        else:
            excluded[s] = {**res, "patient": patient[s]}
        if i % 50 == 0 or i == len(stems):
            print(f"{i}/{len(stems)} stacks", flush=True)
    n_tr = sum(1 for v in info.values() if v["split"] == "train")
    meta = {"channel_names": {"0": "FLAIR"}, "labels": OUT.LABELS_JSON, "numTraining": n_tr, "file_ending": ".nii.gz"}
    (base / "dataset.json").write_text(json.dumps(meta, indent=1))
    (base / "cases.json").write_text(json.dumps(info, indent=1))
    (base / "excluded.json").write_text(json.dumps(excluded, indent=1))
    print(f"wrote {base}: {n_tr} training stacks, {len(info) - n_tr} test stacks, {len(excluded)} excluded")
    return base


def patient_splits(cases_info, k=5, seed=0):
    """{case: fold} with every case of a patient in one fold; cases_info: {case: {'patient': ...}}."""
    fold_of = assign_normal_folds([v["patient"] for v in cases_info.values()], k, seed)
    return {c: fold_of[v["patient"]] for c, v in cases_info.items()}


def write_splits(preprocessed_root, dataset, case_fold, k=5):
    d = Path(preprocessed_root) / dataset
    if not d.is_dir():
        raise FileNotFoundError(f"{d} missing: run nnUNetv2_plan_and_preprocess first")
    p = d / "splits_final.json"
    _refuse(p)
    splits = make_splits(case_fold, k)
    p.write_text(json.dumps(splits, indent=1))
    return p, [len(s["val"]) for s in splits]


def stage_splits(raw_root, preprocessed_root):
    todo = []
    for dataset in (DATASET907, DATASET908):                 # every check first: no split file is written unless both can be
        d = Path(preprocessed_root) / dataset
        if not d.is_dir():
            raise FileNotFoundError(f"{d} missing: run nnUNetv2_plan_and_preprocess first")
        _refuse(d / "splits_final.json")
        info = json.loads((Path(raw_root) / dataset / "cases.json").read_text())
        todo.append((dataset, patient_splits({c: v for c, v in info.items() if v["split"] == "train"})))
    for dataset, case_fold in todo:
        p, sizes = write_splits(preprocessed_root, dataset, case_fold)
        print(f"wrote {p}: validation cases per fold {sizes}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the S4 anatomy datasets")
    ap.add_argument("--stage", choices=("sources", "simulate", "dataset907", "dataset908", "splits"), required=True)
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.stage == "sources":
        stage_sources(a.work)
    elif a.stage == "simulate":
        stage_simulate(a.work, a.workers)
    elif a.stage == "dataset907":
        stage_dataset907(NNUNET_ROOT / "raw", a.work)
    elif a.stage == "dataset908":
        stage_dataset908(NNUNET_ROOT / "raw", a.work)
    else:
        stage_splits(NNUNET_ROOT / "raw", NNUNET_ROOT / "preprocessed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `docs/verification/2026-10-02/brain_anatomy_flair/checks/sim_montage.py`**

```python
# Montages of simulated stacks for a human look (spec 2026-10-02 §5; USER_REPORTED): two samples per source, every
# second slice, image on top and student labels below (15 = ignore). Reads <work>/sim, writes PNGs into the folder
# given as the second argument (must not exist). Nothing else is written.
#
#   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python \
#       docs/verification/2026-10-02/brain_anatomy_flair/checks/sim_montage.py /data2/congcong/data/FM_data/derived/brain_anatomy \
#       docs/verification/2026-10-02/brain_anatomy_flair/simulation
import json
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

work, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    raise FileExistsError(f"{out} exists")
rows = json.loads((work / "sim" / "manifest.json").read_text())
cases = json.loads((work / "cases.json").read_text())
out.mkdir(parents=True)
for source in sorted({r["source"] for r in rows}):
    picked = [r for r in rows if r["source"] == source and r["sample"].endswith("_s0")][:2]
    for r in picked:
        d = work / "sim" / cases[r["case"]]["split"]
        stack = np.asarray(nib.load(str(d / f"{r['sample']}_0000.nii.gz")).dataobj)
        labels = np.asarray(nib.load(str(d / f"{r['sample']}.nii.gz")).dataobj)
        ks = list(range(0, stack.shape[2], 2))
        fig, axes = plt.subplots(2, len(ks), figsize=(3 * len(ks), 6.5))
        for i, k in enumerate(ks):
            for row in (0, 1):
                axes[row, i].imshow(stack[:, :, k].T, cmap="gray", vmin=0, vmax=1000, origin="lower")
                axes[row, i].axis("off")
            axes[0, i].set_title(f"slice {k}")
            axes[1, i].imshow(np.ma.masked_where(labels[:, :, k].T == 0, labels[:, :, k].T), cmap="tab20", vmin=0, vmax=15, alpha=0.5, origin="lower")
        fig.suptitle(f"{source} {r['sample']}: {r['n_slices']} slices, {r['inplane_mm']} mm, empty top {r['empty_top']}, tilt {r['theta_lr']:.1f}/{r['theta_ap']:.1f}; "
                     f"bottom: image, labels (15 = ignore)")
        fig.tight_layout()
        fig.savefig(out / f"sim_{source}_{r['sample']}.png", dpi=72)
        plt.close(fig)
        print(f"wrote {out}/sim_{source}_{r['sample']}.png")
```

- [ ] **Step 5: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_prepare.py -q -p no:cacheprovider`
Expected: `5 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/brain_anatomy_prepare.py docs/verification/2026-10-02/brain_anatomy_flair/checks/sim_montage.py tests/test_brain_anatomy_prepare.py
git commit -m "S4 anatomy: preparation script (sources, simulation, Dataset907/908, splits) and the simulation montage"
```

- [ ] **Step 8 (controller): sources and the split**

```bash
mkdir -p docs/verification/2026-10-02/brain_anatomy_flair/build
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_prepare.py --stage sources 2>&1 | tee docs/verification/2026-10-02/brain_anatomy_flair/build/sources.txt
```

Expected: sibbms 358 cases (MS 261 + Norm 97, 185 patients; four MS sessions with a teacher map have no FLAIR), pdgm 501 (495 patients), bmsr 461 (314 patients), 1320 in total; about 20 % of each source's patients in `test`. A `FileNotFoundError` names a session without a teacher map: stop and report (the exclusion list is not edited without the user).

- [ ] **Step 9 (controller): simulate** (CPU, 8 workers under nice; about 5 500 samples at 2–4 s each; run in the background and poll the log)

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_prepare.py --stage simulate --workers 8 > docs/verification/2026-10-02/brain_anatomy_flair/build/simulate.txt 2>&1
tail -3 docs/verification/2026-10-02/brain_anatomy_flair/build/simulate.txt
```

Expected last line: `wrote …/sim: <4 x training cases> training samples, <test cases> test samples`. Then the montages (two samples per source) for a human look:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python docs/verification/2026-10-02/brain_anatomy_flair/checks/sim_montage.py /data2/congcong/data/FM_data/derived/brain_anatomy docs/verification/2026-10-02/brain_anatomy_flair/simulation
```

Look at the six PNGs (image on top, labels below; 15 = ignore): the stack must start around the basal ganglia / upper cerebellum, end 2–3 slices above the vertex, keep the patient's left at high column indices, and show the tumour / metastases as ignore. Note what was seen in `docs/verification/2026-10-02/brain_anatomy_flair/simulation/README.md` (USER_REPORTED).

- [ ] **Step 10 (controller): Dataset907 and Dataset908**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_prepare.py --stage dataset907 2>&1 | tee docs/verification/2026-10-02/brain_anatomy_flair/build/dataset907.txt
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_prepare.py --stage dataset908 2>&1 | tee docs/verification/2026-10-02/brain_anatomy_flair/build/dataset908.txt
```

Expected: 907 `numTraining` = 4 x training cases; 908 about 433 usable stacks of 447 (the others in `excluded.json`, each with its volume), about 20 % of patients in `imagesTs`.

- [ ] **Step 11 (controller): nnU-Net planning and preprocessing** (CPU, background; 907 is the big one: about 5 500 stacks of 16 x 320 x 320)

```bash
bash -c 'source scripts/nnunet_env.sh && nice -n 19 nnUNetv2_plan_and_preprocess -d 907 -c 3d_fullres -np 8 --verify_dataset_integrity' > docs/verification/2026-10-02/brain_anatomy_flair/build/plan907.txt 2>&1
bash -c 'source scripts/nnunet_env.sh && nice -n 19 nnUNetv2_plan_and_preprocess -d 908 -c 2d -np 8 --verify_dataset_integrity' > docs/verification/2026-10-02/brain_anatomy_flair/build/plan908.txt 2>&1
```

Record the resolved plans (spacing, patch size, batch size) of both configurations:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python - <<'EOF' | tee docs/verification/2026-10-02/brain_anatomy_flair/build/plans.txt
import json
for d in ("Dataset907_BrainAnatomyFLAIR", "Dataset908_FastMRIBrainOutline"):
    p = json.load(open(f"/data2/congcong/data/FM_data/derived/nnunet/preprocessed/{d}/nnUNetPlans.json"))
    for c in ("3d_fullres", "2d"):
        if c in p["configurations"]:
            cfg = p["configurations"][c]
            print(d, c, "spacing", cfg["spacing"], "patch", cfg["patch_size"], "batch", cfg["batch_size"])
EOF
```

- [ ] **Step 12 (controller): splits**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_prepare.py --stage splits 2>&1 | tee docs/verification/2026-10-02/brain_anatomy_flair/build/splits.txt
```

Expected: two `splits_final.json` lines with five validation sizes each; the 907 folds hold whole patients (all K samples of a case in one fold).

- [ ] **Step 13 (controller): commit the build records**

```bash
git add docs/verification/2026-10-02/brain_anatomy_flair/build docs/verification/2026-10-02/brain_anatomy_flair/simulation
git commit -m "S4 anatomy: data build records (sources, simulation, Dataset907/908, plans, splits)"
```

---

### Task 8: Training launcher, inference entry and the launch

**Files:**
- Create: `scripts/brain_anatomy_train.py`
- Create: `scripts/infer_brain_anatomy.py`
- Test: `tests/test_brain_anatomy_scripts.py`

**Interfaces:**
- Consumes: `scripts/brain_detector_train.py` (`idle_gpus`, `query_busy_pids`, `query_nvidia_smi`), Task 6 (`run`), `anatobind.infer.knee.NNUNET_ROOT`.
- Produces: `scripts/brain_anatomy_train.py --jobs student outline --gpus … [--dry-run]` with `JOBS`, `result_dir`, `log_path`, `refusal`, `train_command`, `launch_command`, `plan(jobs, gpus, results_root, log_dir)`; `scripts/infer_brain_anatomy.py --h5 | --nifti, --out, --gpu [--box x0 y0 z0 x1 y1 z1]` with `parse`, `main(argv, run_fn)`.

The controller launches the trainings after the task review (Steps 7–9); the implementer stops after Step 6.

- [ ] **Step 1: Write the failing test**

`tests/test_brain_anatomy_scripts.py`:

```python
# tests/test_brain_anatomy_scripts.py
import importlib.util
from pathlib import Path

import pytest


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_launcher_refuses_existing_outputs_and_pins_one_job_per_card(tmp_path):
    t = _load("brain_anatomy_train")
    results, logs = tmp_path / "results", tmp_path / "logs"
    starts, reasons = t.plan(["student", "outline"], [3, 5], results, logs)
    assert starts == [("student", 3), ("outline", 5)] and reasons == {}
    t.result_dir(results, "student").mkdir(parents=True)
    starts, reasons = t.plan(["student", "outline"], [3], results, logs)
    assert starts == [("outline", 3)] and "result folder" in reasons["student"]
    logs.mkdir()
    t.log_path(logs, "outline").write_text("")
    starts, reasons = t.plan(["student", "outline"], [3, 5], results, logs)
    assert starts == [] and "log" in reasons["outline"]
    assert t.plan(["outline"], [], tmp_path / "r2", tmp_path / "l2") == ([], {"outline": "no idle GPU left"})
    # the student's classes have a side: no mirroring (spec A17); the outline keeps the default trainer
    assert t.train_command("student") == ["nnUNetv2_train", "907", "3d_fullres", "0", "-tr", "nnUNetTrainer_250epochs_NoMirroring"]
    assert t.train_command("outline") == ["nnUNetv2_train", "908", "2d", "0", "-tr", "nnUNetTrainer_250epochs"]
    cmd = t.launch_command("outline", 5, tmp_path, logs)
    assert cmd[:3] == ["setsid", "bash", "-c"] and "CUDA_VISIBLE_DEVICES=5 nice -n 19 nnUNetv2_train 908 2d 0" in cmd[3]
    assert "scripts/nnunet_env.sh" in cmd[3] and str(t.log_path(logs, "outline")) in cmd[3]
    assert t.result_dir(results, "outline").name == "fold_0" and "nnUNetTrainer_250epochs__nnUNetPlans__2d" in str(t.result_dir(results, "outline"))
    assert "nnUNetTrainer_250epochs_NoMirroring__nnUNetPlans__3d_fullres" in str(t.result_dir(results, "student"))
    assert "nnUNetTrainer_250epochs_NoMirroring" in t.log_path(logs, "student").name


def test_infer_entry_passes_the_arguments_through(tmp_path, capsys):
    s = _load("infer_brain_anatomy")
    seen = {}

    def fake_run(out, gpu, h5=None, nifti=None, box=None):
        seen.update(out=out, gpu=gpu, h5=h5, nifti=nifti, box=box)
        return {"brain_ml": 1234.5, "reliable_slices": [2, 12], "anatomy": str(out / "anatomy.nii.gz"), "box": list(box),
                "binding": {"host": "thalamus", "host_rule": "overlap", "host_side": "left", "host_fractions": {"thalamus": 1.0}}}

    s.main(["--h5", "/x/file.h5", "--out", str(tmp_path / "o"), "--gpu", "4", "--box", "1", "2", "3", "4", "5", "6"], run_fn=fake_run)
    assert seen == {"out": tmp_path / "o", "gpu": 4, "h5": Path("/x/file.h5"), "nifti": None, "box": (1, 2, 3, 4, 5, 6)}
    out = capsys.readouterr().out
    assert "brain 1234.5 mL" in out and "host thalamus (overlap), side left" in out
    with pytest.raises(SystemExit):
        s.parse(["--h5", "a", "--nifti", "b", "--out", "o", "--gpu", "0"])
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_scripts.py -q -p no:cacheprovider`
Expected: errors at import or collection (the module does not exist yet).

- [ ] **Step 3: Write `scripts/brain_anatomy_train.py`**

```python
#!/usr/bin/env python
# scripts/brain_anatomy_train.py
"""Launch the two S4 trainings on idle GPUs, one job per card (spec 2026-10-02 A9, A10, A15):
Dataset907 3d_fullres fold 0 (student, nnUNetTrainer_250epochs_NoMirroring: its classes have a side, spec A17) and
Dataset908 2d fold 0 (brain outline, nnUNetTrainer_250epochs).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_train.py --jobs student outline --gpus 0 1 2 3 4 5 6 7

A job whose result folder or log exists is refused (nnU-Net would start over inside the folder); a card with a
compute process or more than IDLE_MEM_THRESHOLD_MIB in use is never touched."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain_detector_train import idle_gpus, query_busy_pids, query_nvidia_smi  # noqa: E402
from anatobind.infer.brain_anatomy import OUTLINE, STUDENT  # noqa: E402
from anatobind.infer.knee import NNUNET_ROOT  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "logs" / "brain_anatomy"
JOBS = {"student": {**STUDENT, "name": "Dataset907_BrainAnatomyFLAIR"},       # id, config and trainer: the inference chain's
        "outline": {**OUTLINE, "name": "Dataset908_FastMRIBrainOutline"}}
FOLD = 0


def result_dir(results_root, job, fold=FOLD):
    j = JOBS[job]
    return Path(results_root) / j["name"] / f"{j['trainer']}__nnUNetPlans__{j['config']}" / f"fold_{fold}"


def log_path(log_dir, job, fold=FOLD):
    j = JOBS[job]
    return Path(log_dir) / f"{j['name']}_{j['config']}_{j['trainer']}_fold{fold}.log"


def refusal(results_root, log_dir, job):
    r, l = result_dir(results_root, job), log_path(log_dir, job)
    if r.exists():
        return f"result folder {r} exists"
    if l.exists():
        return f"log {l} exists"
    return None


def train_command(job, fold=FOLD):
    j = JOBS[job]
    return ["nnUNetv2_train", str(j["id"]), j["config"], str(fold), "-tr", j["trainer"]]


def launch_command(job, gpu, repo_root, log_dir):
    cmd = " ".join(shlex.quote(p) for p in train_command(job))
    chain = (f"source {shlex.quote(str(Path(repo_root) / 'scripts/nnunet_env.sh'))} && "
             f"CUDA_VISIBLE_DEVICES={int(gpu)} nice -n 19 {cmd} > {shlex.quote(str(log_path(log_dir, job)))} 2>&1")
    return ["setsid", "bash", "-c", chain]


def plan(jobs, gpus, results_root, log_dir):
    """[(job, gpu)] for the jobs that may start now, and the reasons the others do not."""
    starts, reasons = [], {}
    free = list(gpus)
    for job in jobs:
        why = refusal(results_root, log_dir, job)
        if why:
            reasons[job] = why
        elif not free:
            reasons[job] = "no idle GPU left"
        else:
            starts.append((job, free.pop(0)))
    return starts, reasons


def main(argv=None):
    ap = argparse.ArgumentParser(description="Launch the S4 trainings on idle GPUs")
    ap.add_argument("--jobs", nargs="+", choices=sorted(JOBS), default=["student", "outline"])
    ap.add_argument("--gpus", nargs="+", type=int, default=list(range(8)), help="candidate cards; only idle ones are used")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    idle = idle_gpus(query_nvidia_smi(), a.gpus, query_busy_pids())
    starts, reasons = plan(a.jobs, idle, NNUNET_ROOT / "results", LOG_DIR)
    for job, why in reasons.items():
        print(f"not started: {job}: {why}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for job, gpu in starts:
        cmd = launch_command(job, gpu, REPO, LOG_DIR)
        if a.dry_run:
            print(f"would launch {job} on GPU {gpu}: {' '.join(shlex.quote(c) for c in cmd)}")
            continue
        p = subprocess.Popen(cmd, cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        print(f"launched {job} on GPU {gpu} (pid {p.pid}); log {log_path(LOG_DIR, job)}; results {result_dir(NNUNET_ROOT / 'results', job)}")
    return 0 if starts or not a.jobs else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `scripts/infer_brain_anatomy.py`**

```python
#!/usr/bin/env python
# scripts/infer_brain_anatomy.py
"""S4 inference entry (spec 2026-10-02 §8): a fastMRI FLAIR stack -> skull-stripped stack -> anatomy map in SynthSeg
label values (+ an optional binding demonstration of one box with BrainBinder).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_anatomy.py \
      --h5 <fastMRI h5> --out <new directory> --gpu <idle gpu> [--box x0 y0 z0 x1 y1 z1]
  ... --nifti <RSS stack> ...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.infer.brain_anatomy import run  # noqa: E402


def parse(argv=None):
    ap = argparse.ArgumentParser(description="S4 brain anatomy on a fastMRI FLAIR stack")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5", type=Path, help="fastMRI h5 (RSS reconstruction is used)")
    src.add_argument("--nifti", type=Path, help="a stack already in the RSS NIfTI frame")
    ap.add_argument("--out", type=Path, required=True, help="output directory (must not exist)")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--box", type=int, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"), help="half-open box on the stack grid to bind")
    return ap.parse_args(argv)


def main(argv=None, run_fn=run):
    a = parse(argv)
    rec = run_fn(a.out, a.gpu, h5=a.h5, nifti=a.nifti, box=tuple(a.box) if a.box else None)
    print(f"brain {rec['brain_ml']} mL; reliable slices {rec['reliable_slices']}; anatomy -> {rec['anatomy']}")
    if "binding" in rec:
        b = rec["binding"]
        print(f"box {rec['box']}: host {b['host']} ({b['host_rule']}), side {b['host_side']}, fractions {b['host_fractions']}")
    return rec


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_scripts.py -q -p no:cacheprovider`
Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/brain_anatomy_train.py scripts/infer_brain_anatomy.py tests/test_brain_anatomy_scripts.py
git commit -m "S4 anatomy: training launcher (idle GPUs, fold 0 of 907 and 908) and inference entry"
```

- [ ] **Step 7 (controller): dry run, then launch on idle cards**

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_train.py --jobs student outline --dry-run
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_train.py --jobs student outline
```

Write `docs/verification/2026-10-02/brain_anatomy_flair/launch.md`: the GPU snapshot, the two launch lines (pids, logs, result folders), and after 20 minutes the rate of each training (seconds per epoch from `training_log_*.txt` in the result folders) with the projection (250 epochs x seconds per epoch). If a projection exceeds 24 h, tell the user before letting it run on. Commit: `git add docs/verification/2026-10-02/brain_anatomy_flair/launch.md && git commit -m "S4 anatomy: trainings launched (launch record)"`.

- [ ] **Step 8 (controller): wait** with a background `until` loop on the result folders (`checkpoint_final.pth` in both `fold_0` folders); never wait on a `pgrep -f` pattern that matches the loop itself; the launcher logs are block-buffered, so read `training_log_*.txt`.

- [ ] **Step 9 (controller): confirm** both runs ended (`Training done` in the training log, `checkpoint_final.pth` present) and write `docs/verification/2026-10-02/brain_anatomy_flair/training.txt` with the tail of both logs and the epoch counts. Commit it.

---

### Task 9: Evaluation script, the evaluation run and the inference smoke

**Files:**
- Create: `scripts/eval_brain_anatomy.py`
- Create: `docs/verification/2026-10-02/brain_anatomy_flair/checks/eval_montage.py`
- Test: `tests/test_brain_anatomy_eval_script.py`

**Interfaces:**
- Consumes: Tasks 1, 4, 5, 6; `anatobind.eval.brain_disease.code_version`; `anatobind.level_r.registry.load_registry`.
- Produces: `scripts/eval_brain_anatomy.py --out <record dir> --work <prediction dir> --gpu G [--sim-work] [--seg-dir]` with `link_stacks`, `mask_stacks`, `stem_metrics`, `student_of`, `evaluate_fastmri`, `evaluate_simulated`, `summarize`, `report`, `main(argv, predict)`; records `REPORT.md`, `verdict.json`, `per_stem.csv`, `per_stem.json`.

The controller runs the evaluation and the smoke after the task review (Steps 8–11); the implementer stops after Step 7.

- [ ] **Step 1: Write the failing test**

`tests/test_brain_anatomy_eval_script.py`:

```python
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
    assert "host agreement 0.5, mean host Dice" in text and "outline Dice 1.0 (gates 0.9 / 0.8 / 0.97)" in text     # the numbers, not the pass flags
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_eval_script.py -q -p no:cacheprovider`
Expected: errors at import or collection (the module does not exist yet).

- [ ] **Step 3: Write `scripts/eval_brain_anatomy.py`**

```python
#!/usr/bin/env python
# scripts/eval_brain_anatomy.py
"""S4 evaluation (spec 2026-10-02 §7, A11, A13): agreement of the student with the SynthSeg pseudo-labels on the
fastMRI stacks (reliable slices), host agreement over the fastMRI+ lesion boxes, the outline model's Dice on its test
stacks, and Dice on the simulated test set. All numbers are NOT_EVIDENCE (pseudo-labels).

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_anatomy.py \
      --out docs/verification/2026-10-02/brain_anatomy_flair/eval --work /data2/congcong/data/FM_data/derived/brain_anatomy/eval_<date> --gpu <idle>

Both directories must not exist. The two nnU-Net models run once each over all stacks (batch prediction)."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.anatomy.labels import HOST_IDS, to_student  # noqa: E402
from anatobind.anatomy.outline import postprocess  # noqa: E402
from anatobind.eval.brain_anatomy import (  # noqa: E402
    GATES, class_dice, host_agreement, outline_dice, pool_agreement, reliable_slices, summarize_dice, verdict,
)
from anatobind.eval.brain_disease import code_version  # noqa: E402
from anatobind.eval.lesion_boxes import load_label_map  # noqa: E402
from anatobind.infer.brain_anatomy import OUTLINE, STUDENT, run_nnunet  # noqa: E402
from anatobind.infer.knee import NNUNET_ROOT  # noqa: E402
from anatobind.level_r.registry import load_registry  # noqa: E402
from anatobind.nnunet.brain_disease import FM  # noqa: E402

WORK = FM / "derived/brain_anatomy"
FASTMRI_SEG = FM / "derived/synthseg/fastmri_brain/seg_native"
DATASET907 = "Dataset907_BrainAnatomyFLAIR"
DATASET908 = "Dataset908_FastMRIBrainOutline"
LOW_SLICES = (0, 1)


def link_stacks(dataset_dir, staging):
    """Every fastMRI stack of Dataset908 (training and test images) -> <staging>/<stem>_0000.nii.gz symlinks."""
    staging.mkdir(parents=True)
    stems = {}
    for folder in ("imagesTr", "imagesTs"):
        for p in sorted((dataset_dir / folder).glob("*_0000.nii.gz")):
            stem = p.name[:-len("_0000.nii.gz")]
            os.symlink(p.resolve(), staging / p.name)
            stems[stem] = folder
    return stems


def mask_stacks(staging, masks_dir, out_dir):
    """Skull-stripped copies of the stacks: image x postprocessed outline mask."""
    out_dir.mkdir(parents=True)
    for p in sorted(staging.glob("*_0000.nii.gz")):
        stem = p.name[:-len("_0000.nii.gz")]
        img = nib.load(str(p))
        mask = postprocess(load_label_map(masks_dir / f"{stem}.nii.gz"))
        if mask.shape != img.shape:
            raise ValueError(f"{stem}: outline {mask.shape} and stack {img.shape} differ")
        nib.save(nib.Nifti1Image(np.asarray(img.dataobj).astype(np.float32) * mask, img.affine), str(out_dir / p.name))
        nib.save(nib.Nifti1Image(mask, img.affine), str(out_dir / f"{stem}_mask.nii.gz"))


def stem_metrics(student, synthseg, mask, spacing, rows, is_test):
    """One fastMRI stack: Dice of the host classes on the reliable slices, host agreement of its lesion boxes, outline
    Dice when the stack is an outline-model test stack, and the student's brain share in the lowest two slices."""
    area = spacing[0] * spacing[1]
    reliable = reliable_slices(synthseg, area)
    d = class_dice(student, student_of(synthseg), slices=reliable) if len(reliable) else {c: None for c in HOST_IDS}
    agree = host_agreement(student, synthseg, spacing, rows, reliable)
    low = {int(k): round(float((student[:, :, k] > 0).sum() * area / 100.0), 1) for k in LOW_SLICES if k < student.shape[2]}   # cm2 of labelled tissue
    return {"reliable": [reliable.start, reliable.stop - 1] if len(reliable) else None, "dice": d, "agreement": agree,
            "outline_dice": outline_dice(mask, synthseg, area) if is_test else None, "low_slice_area_cm2": low}


def student_of(synthseg):
    return to_student(synthseg)


def evaluate_fastmri(stems, pred_student_dir, masked_dir, seg_dir, registry_rows):
    per_stem = {}
    for stem, folder in sorted(stems.items()):
        seg_img = nib.load(str(Path(seg_dir) / f"{stem}_seg.nii.gz"))
        synthseg = np.asarray(seg_img.dataobj).astype(np.int16)
        spacing = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
        student = load_label_map(pred_student_dir / f"{stem}.nii.gz")
        mask = load_label_map(masked_dir / f"{stem}_mask.nii.gz")
        if student.shape != synthseg.shape:
            raise ValueError(f"{stem}: student {student.shape} and SynthSeg {synthseg.shape} differ")
        per_stem[stem] = {"split": "test" if folder == "imagesTs" else "train",
                          **stem_metrics(student, synthseg, mask, spacing, registry_rows.get(stem, []), folder == "imagesTs")}
    return per_stem


def evaluate_simulated(pred_dir, label_dir, samples):
    per_sample = {}
    for s in sorted(samples):
        pred, ref = load_label_map(pred_dir / f"{s}.nii.gz"), load_label_map(label_dir / f"{s}.nii.gz")
        per_sample[s] = class_dice(pred, ref, classes=tuple(HOST_IDS) + (14,))
    return per_sample


def summarize(per_stem, per_sample):
    dice = summarize_dice([v["dice"] for v in per_stem.values()])
    agreement = pool_agreement([v["agreement"] for v in per_stem.values()])
    outl = [v["outline_dice"] for v in per_stem.values() if v["outline_dice"] is not None]
    outline = {"n_test_stacks": len(outl), "mean": float(np.mean(outl)) if outl else None, "min": float(np.min(outl)) if outl else None}
    sim = summarize_dice(list(per_sample.values())) if per_sample else None
    vent = [d.get(14) for d in per_sample.values() if d.get(14) is not None]
    if sim is not None:
        sim["ventricles"] = float(np.mean(vent)) if vent else None
    low = {k: float(np.median([v["low_slice_area_cm2"].get(k, v["low_slice_area_cm2"].get(str(k), 0.0)) for v in per_stem.values()])) for k in LOW_SLICES}
    v = verdict(agreement["rate"], dice["mean_host_dice"], outline["mean"])
    return {"verdict": v, "fastmri_dice": dice, "host_agreement": {k: val for k, val in agreement.items()}, "outline": outline,
            "simulated_test": sim, "low_slice_median_area_cm2": low, "n_stacks": len(per_stem)}


def report(summary, per_stem, out, command, code):
    v, d, a, o, s = summary["verdict"], summary["fastmri_dice"], summary["host_agreement"], summary["outline"], summary["simulated_test"]
    L = ["# S4 brain anatomy on fastMRI FLAIR: evaluation (NOT_EVIDENCE: agreement with SynthSeg pseudo-labels)\n\n",
         f"Verdict (spec A11): **{'pass' if v['pass'] else 'fail'}** — host agreement {v['values']['host_agreement']}, mean host Dice {v['values']['mean_host_dice']}, "
         f"outline Dice {v['values']['outline_dice']} (gates {GATES['host_agreement']} / {GATES['mean_host_dice']} / {GATES['outline_dice']}).\n\n",
         "The final judgement of S4 waits for the Level R reader labels (A12); the lowest two slices are reported, not judged.\n\n",
         "## fastMRI stacks (reliable slices only)\n\n", "```json\n", json.dumps({"n_stacks": summary["n_stacks"], "host_agreement": a, "mean_host_dice": d["mean_host_dice"]}, indent=1), "\n```\n\n",
         "| class | mean Dice over stacks |\n|---|---|\n"]
    L += [f"| {name} | {'n/a' if val is None else f'{val:.4f}'} |\n" for name, val in d["per_class"].items()]
    L += ["\n## Outline model (its test stacks)\n\n", "```json\n", json.dumps(o, indent=1), "\n```\n\n",
          "## Lowest two slices (student output, median labelled area in cm2; no reliable reference there)\n\n", "```json\n",
          json.dumps(summary["low_slice_median_area_cm2"], indent=1), "\n```\n\n"]
    if s is not None:
        L += ["## Simulated test set (A13, report only)\n\n", "```json\n", json.dumps(s, indent=1), "\n```\n\n"]
    L += ["## Command\n\n```\n", command, "\n```\n\n", f"Code: commit {code}\n"]
    (out / "REPORT.md").write_text("".join(L))
    (out / "verdict.json").write_text(json.dumps(summary, indent=1))
    with open(out / "per_stem.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "split", "reliable_first", "reliable_last", "mean_host_dice", "n_lesions", "n_evaluated", "n_agree", "outline_dice"])
        for stem, v in sorted(per_stem.items()):
            vals = [x for x in v["dice"].values() if x is not None]
            w.writerow([stem, v["split"], *(v["reliable"] or ["", ""]), f"{np.mean(vals):.4f}" if vals else "", v["agreement"]["n_lesions"],
                        v["agreement"]["n_evaluated"], v["agreement"]["n_agree"], "" if v["outline_dice"] is None else f"{v['outline_dice']:.4f}"])
    (out / "per_stem.json").write_text(json.dumps(per_stem, indent=1))


def main(argv=None, predict=run_nnunet):
    ap = argparse.ArgumentParser(description="Evaluate the S4 anatomy model")
    ap.add_argument("--out", type=Path, required=True, help="record directory (must not exist)")
    ap.add_argument("--work", type=Path, required=True, help="prediction directory under /data2 (must not exist)")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--sim-work", type=Path, default=WORK, help="where sim/test and the manifest live")
    ap.add_argument("--seg-dir", type=Path, default=FASTMRI_SEG)
    a = ap.parse_args(argv)
    for p in (a.out, a.work):
        if p.exists():
            raise FileExistsError(f"{p} exists")
    code = code_version(Path(__file__).resolve().parents[1])
    raw = NNUNET_ROOT / "raw"
    stems = link_stacks(raw / DATASET908, a.work / "stacks")
    predict(OUTLINE["id"], OUTLINE["config"], a.work / "stacks", a.work / "pred_outline", [0], a.gpu)
    mask_stacks(a.work / "stacks", a.work / "pred_outline", a.work / "stripped")
    predict(STUDENT["id"], STUDENT["config"], a.work / "stripped", a.work / "pred_student", [0], a.gpu)
    rows = {}
    for r in load_registry():
        rows.setdefault(r["file"], []).append(r)
    per_stem = evaluate_fastmri(stems, a.work / "pred_student", a.work / "stripped", a.seg_dir, rows)
    samples = [p.name[:-len("_0000.nii.gz")] for p in sorted((raw / DATASET907 / "imagesTs").glob("*_0000.nii.gz"))]
    per_sample = {}
    if samples:
        predict(STUDENT["id"], STUDENT["config"], raw / DATASET907 / "imagesTs", a.work / "pred_sim_test", [0], a.gpu)
        per_sample = evaluate_simulated(a.work / "pred_sim_test", a.sim_work / "sim" / "test", samples)
    summary = summarize(per_stem, per_sample)
    a.out.mkdir(parents=True)
    report(summary, per_stem, a.out, " ".join(sys.argv), code)
    v = summary["verdict"]
    print(f"stacks {summary['n_stacks']}; host agreement {v['values']['host_agreement']}; mean host Dice {v['values']['mean_host_dice']}; "
          f"outline Dice {v['values']['outline_dice']}; pass {v['pass']}")
    return summary


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `docs/verification/2026-10-02/brain_anatomy_flair/checks/eval_montage.py`**

```python
# Montages of the student's output on the lowest two slices and the vertex of fastMRI stacks (spec 2026-10-02 §7:
# those slices have no reliable reference, so a human looks; USER_REPORTED). Reads the evaluation work directory
# (<work>/stripped/<stem>_0000.nii.gz, <work>/pred_student/<stem>.nii.gz) and the SynthSeg maps; writes PNGs into the
# folder given as the last argument (must not exist).
#
#   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python \
#       docs/verification/2026-10-02/brain_anatomy_flair/checks/eval_montage.py <work> <seg_dir> <out_dir> [n_stacks]
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from anatobind.anatomy.labels import to_student  # noqa: E402
from anatobind.eval.brain_anatomy import reliable_slices  # noqa: E402

work, seg_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 6
if out.exists():
    raise FileExistsError(f"{out} exists")
stems = sorted(p.name[:-len("_0000.nii.gz")] for p in (work / "stripped").glob("*_0000.nii.gz"))
step = max(1, len(stems) // n)
out.mkdir(parents=True)
for stem in stems[::step][:n]:
    img = nib.load(str(work / "stripped" / f"{stem}_0000.nii.gz"))
    data, student = np.asarray(img.dataobj), np.asarray(nib.load(str(work / "pred_student" / f"{stem}.nii.gz")).dataobj)
    seg_img = nib.load(str(seg_dir / f"{stem}_seg.nii.gz"))
    teacher = to_student(np.asarray(seg_img.dataobj))
    z = seg_img.header.get_zooms()
    rel = reliable_slices(np.asarray(seg_img.dataobj), float(z[0]) * float(z[1]))
    top = (rel.stop if len(rel) else data.shape[2] - 2)
    ks = [0, 1] + [k for k in (top, top + 1) if k < data.shape[2]]
    fig, axes = plt.subplots(3, len(ks), figsize=(3.2 * len(ks), 9.5))
    for i, k in enumerate(ks):
        for row in range(3):
            axes[row, i].imshow(data[:, :, k].T, cmap="gray", vmin=0, vmax=np.percentile(data, 99.5), origin="lower")
            axes[row, i].axis("off")
        axes[0, i].set_title(f"slice {k}{' (reliable)' if k in rel else ''}")
        axes[1, i].imshow(np.ma.masked_where(student[:, :, k].T == 0, student[:, :, k].T), cmap="tab20", vmin=0, vmax=15, alpha=0.5, origin="lower")
        axes[2, i].imshow(np.ma.masked_where(teacher[:, :, k].T == 0, teacher[:, :, k].T), cmap="tab20", vmin=0, vmax=15, alpha=0.5, origin="lower")
    fig.suptitle(f"{stem}: stripped stack / student / SynthSeg (reliable slices {rel.start}-{rel.stop - 1 if len(rel) else 'none'})")
    fig.tight_layout()
    fig.savefig(out / f"eval_{stem}.png", dpi=72)
    plt.close(fig)
    print(f"wrote {out}/eval_{stem}.png")
```

- [ ] **Step 5: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_anatomy_eval_script.py -q -p no:cacheprovider`
Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_brain_anatomy.py docs/verification/2026-10-02/brain_anatomy_flair/checks/eval_montage.py tests/test_brain_anatomy_eval_script.py
git commit -m "S4 anatomy: evaluation script (fastMRI agreement, outline Dice, simulated test set, A11 verdict) and its montage"
```

- [ ] **Step 8 (controller): evaluate** (one idle GPU; both models predict all 447 stacks once; about an hour)

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_brain_anatomy.py --out docs/verification/2026-10-02/brain_anatomy_flair/eval --work /data2/congcong/data/FM_data/derived/brain_anatomy/eval_$(date +%Y%m%d_%H%M) --gpu <idle> 2>&1 | tee docs/verification/2026-10-02/brain_anatomy_flair/eval_output.txt
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python docs/verification/2026-10-02/brain_anatomy_flair/checks/eval_montage.py /data2/congcong/data/FM_data/derived/brain_anatomy/eval_<stamp> /data2/congcong/data/FM_data/derived/synthseg/fastmri_brain/seg_native docs/verification/2026-10-02/brain_anatomy_flair/eval/montage 6
```

Read `REPORT.md`: the verdict line is the result (A11); the lowest two slices are reported only. Look at the six montages (stripped stack / student / SynthSeg at slices 0, 1 and the two above the reliable range) and note what was seen (USER_REPORTED) in `docs/verification/2026-10-02/brain_anatomy_flair/eval/README.md`.

- [ ] **Step 9 (controller): inference smoke** on S2's smoke volume and on one lesion volume with a registry box:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python - <<'EOF'
from anatobind.level_r.registry import load_registry
r = sorted(load_registry(), key=lambda r: (r["file"], r["lesion_id"]))[0]
print(r["file"], r["lesion_id"], "--box", r["x0"], r["y0"], r["z0"], r["x1"], r["y1"], r["z1"] + 1)
EOF
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_anatomy.py --h5 /data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/brain/multicoil_train/file_brain_AXFLAIR_201_6002917.h5 --out /data2/congcong/data/FM_data/derived/brain_anatomy/infer_smoke/svd --gpu <idle>
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_anatomy.py --h5 <h5 of the printed file> --out /data2/congcong/data/FM_data/derived/brain_anatomy/infer_smoke/lesion --gpu <idle> --box <the printed six numbers>
```

Write `docs/verification/2026-10-02/brain_anatomy_flair/infer_smoke.md`: both commands, the printed lines, the record's class volumes and reliable slices, the binding of the box (host, rule, side, fractions) beside the host that `BrainLookup` on the SynthSeg map gives for the same box (one line of Python), with the NOT_EVIDENCE label.

- [ ] **Step 10 (controller): commit the records**

```bash
git add docs/verification/2026-10-02/brain_anatomy_flair/eval docs/verification/2026-10-02/brain_anatomy_flair/eval_output.txt docs/verification/2026-10-02/brain_anatomy_flair/infer_smoke.md
git commit -m "S4 anatomy: evaluation records, montages and inference smoke"
```

- [ ] **Step 11 (controller): report** the three A11 numbers and the verdict to the user; a fail stops here (no tuning) and the user decides.

---

### Task 10: Records index, documentation, whole-branch review, tag

**Files:**
- Create: `docs/verification/2026-10-02/brain_anatomy_flair/README.md`
- Modify: `CLAUDE.md` (code-map line for `anatobind/anatomy/`, status sentence, test count), `STATUS.md` (full rewrite, five sections)

- [ ] **Step 1: `docs/verification/2026-10-02/brain_anatomy_flair/README.md`** — one line per record in the folder (what it is, the command that made it); the verdict table (the A11 numbers beside the gates, the simulated test Dice, the outline Dice); "Known deviations": the SibBMS exclusions (9), the fastMRI stacks excluded from the outline model (`excluded.json`), BMSR resampled from 1.5 mm, SibBMS lesions not ignored (no same-grid masks), 3D FLAIR training versus 2D FLAIR target (A7), the lowest two slices unjudged (A12), left/right by the headers; "Timing" from `launch.md`, `training.txt` and the build logs. Every number copied from a file.
- [ ] **Step 2: Tests** — `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`; record the count (expected 872 passed, 1 skipped).
- [ ] **Step 3: `CLAUDE.md`** — add to the code map: `anatobind/anatomy/{labels,sources,simulate,outline}.py`, `anatobind/eval/brain_anatomy.py`, `anatobind/infer/brain_anatomy.py`, `scripts/brain_anatomy_{prepare,train}.py`, `eval_brain_anatomy.py`, `infer_brain_anatomy.py`, the spec, the plan, the records folder; update the status sentence (S4 verdict and where it is) and the test count. Change nothing else.
- [ ] **Step 4: `STATUS.md`** — rewritten with the five fixed sections (verified, with commands and raw outputs; decisions for the user: the A11 outcome and whether S4 waits for Level R or iterates on A7, push, the deletable list; next steps: the Level R reader, S5 integration, stage B; pitfalls: array frame, ignore label ids, scripts without deletion, template-space brains; the why of A1–A16).
- [ ] **Step 5: Commit** — `git add docs/verification/2026-10-02/brain_anatomy_flair/README.md CLAUDE.md STATUS.md && git commit -m "Docs: S4 brain anatomy on fastMRI FLAIR, records index, status and code map"`.
- [ ] **Step 6: Whole-branch review** on the most capable model over the range from the plan commit to HEAD (code, records, numbers against files, safety: no deletion calls, every output refused when it exists); fix; re-review; then tag `handoff/<date>-brain-anatomy-flair` on main. No push.
