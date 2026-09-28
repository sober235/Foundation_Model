# Brain Small-Lesion Detector (S2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate an nnU-Net v2 (2D) small-lesion detector on fastMRI+ brain FLAIR (165 lesion volumes with 1297 registry lesions + 88 normal volumes), judge the D1 gate on patient-level out-of-fold predictions, and ship an inference entry point for S5.

**Architecture:** Registry lesions' per-slice member boxes are painted into binary label maps on the raw RSS grid (no slice padding); nnU-Net trains on custom patient folds so its cross-validation outputs are out-of-fold predictions; connected components of the predicted maps become scored 3D boxes that the existing knee detection metrics score. A small launcher assigns folds to idle GPUs.

**Tech Stack:** Python 3.11, numpy, scipy.ndimage, nibabel, h5py, nnU-Net v2 (`nnUNetv2_*` in env nvgen), pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-brain-detector-design.md` (decisions D1–D10).

## Global Constraints

- Tests: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest <paths> -q -p no:cacheprovider`; tests never read `/data2`.
- nnU-Net env: `source scripts/nnunet_env.sh` (raw / preprocessed / results under `/data2/congcong/data/FM_data/derived/nnunet/`, `nnUNet_n_proc_DA=8`).
- Dataset id 903, name `Dataset903_FastMRIBrainSmallLesion`, labels `{"background": 0, "small_lesion": 1}`, trainer `nnUNetTrainer_250epochs`; primary configuration `2d` (decides the gate), `3d_fullres` reported only.
- Cases = 165 registry lesion volumes + 88 FLAIR volumes whose every fastMRI+ row is "Normal for age"; case name = h5 stem. Lesion patients keep `data/level_r/folds.json`; normal patients are sorted, shuffled with seed 0 and dealt into 5 folds; the two patient sets must not overlap.
- Images: `fastmri.rss_h5_to_nifti(h5, out, pad_to_slices=0)` — grid (col, row, slice), no padding; labels share the image's affine and shape.
- Gate (D1): 3D IoU ≥ 0.1 one-to-one, operating point at ≤ 2 FP per volume, lesion sensitivity ≥ 0.5; denominator = 1297 registry lesions; FP counted over all 253 volumes; normal-volume FP per volume reported separately.
- Decoding (D6): 26-connectivity, `BRAIN_MIN_VOXELS = 9`; the knee default `MIN_VOXELS = 27` must not change.
- GPUs (D7): four 80 GB cards, default 0–3, each fold pinned with `CUDA_VISIBLE_DEVICES`, idle-checked with nvidia-smi before launch, never a card another process is using. Timing probe first (D8): projected wall clock > 24 h → stop and ask the user.
- If the gate fails, do not tune to rescue it (D9): report and stop.
- Nothing is overwritten or deleted: every output directory or file that exists is refused. Commits: repository author, English messages, no Co-Authored-By or any AI trace (the harness reminder is overridden by the user's rule). No push, no merge.

## Review Focus

1. A lesion whose member box leaves the RSS grid — expected: the build refuses with the lesion id, never silently clips (Task 2 test).
2. A normal patient who is also a lesion patient — expected: fold assignment refuses (Task 2 test).
3. nnU-Net probability arrays in (C, Z, Y, X) order — expected: transposed and checked against the label map, as the knee loader does (Task 4 reuses `load_nnunet_probabilities`, test with a synthetic npz).
4. A validation case missing from nnU-Net's outputs — expected: the evaluation refuses and names the case instead of scoring fewer volumes (Task 4 test).
5. Inference on a volume with no detections — expected: an empty lesion table, not a crash (Task 5 test).

---

### Task 1: Brain decoding options in `anatobind/eval/lesion_boxes.py`

**Files:**
- Modify: `anatobind/eval/lesion_boxes.py` (constants and `decode_boxes` signature)
- Test: `tests/test_brain_detector_decode.py`

**Interfaces:**
- Produces: `BRAIN_MIN_VOXELS = 9`; `decode_boxes(label_map, probs=None, min_voxels=MIN_VOXELS, families=None)` — `families` maps label → family name, default the knee `FAMILY_OF_LABEL`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain_detector_decode.py
import numpy as np

from anatobind.eval.lesion_boxes import BRAIN_MIN_VOXELS, MIN_VOXELS, decode_boxes


def _single_slice_lesion():
    lab = np.zeros((20, 20, 4), np.uint8)
    lab[5:9, 6:10, 2] = 1                      # 4 x 4 x 1 = 16 voxels
    probs = np.zeros((2,) + lab.shape, np.float32)
    probs[1][lab == 1] = 0.8
    return lab, probs


def test_brain_minimum_keeps_a_single_slice_4x4_lesion_and_the_knee_default_is_unchanged():
    lab, probs = _single_slice_lesion()
    assert MIN_VOXELS == 27 and BRAIN_MIN_VOXELS == 9
    assert decode_boxes(lab, probs, families={1: "small_lesion"}) == []           # 16 < 27
    out = decode_boxes(lab, probs, min_voxels=BRAIN_MIN_VOXELS, families={1: "small_lesion"})
    assert len(out) == 1 and out[0]["family"] == "small_lesion" and out[0]["n_voxels"] == 16
    assert out[0]["box"] == (5, 6, 2, 9, 10, 3) and abs(out[0]["score"] - 0.8) < 1e-6


def test_default_families_are_the_knee_ones():
    lab = np.zeros((10, 10, 10), np.uint8)
    lab[1:4, 1:4, 1:4] = 2
    assert decode_boxes(lab)[0]["family"] == "Meniscal Tear"
```

- [ ] **Step 2: Run to verify it fails** — `... -m pytest tests/test_brain_detector_decode.py -q -p no:cacheprovider` → ImportError `BRAIN_MIN_VOXELS`.

- [ ] **Step 3: Implement** — in `anatobind/eval/lesion_boxes.py` add after `MIN_VOXELS`:

```python
BRAIN_MIN_VOXELS = 9                     # 3x3x1: a single-slice 4x4 brain lesion has 16 voxels (spec 2026-09-28 D6)
```

and change the head of `decode_boxes` to:

```python
def decode_boxes(label_map, probs=None, min_voxels=MIN_VOXELS, families=None):
    """families: {label: family name}; default the knee families."""
    out = []
    for label, family in (FAMILY_OF_LABEL if families is None else families).items():
```

(the rest of the function is unchanged).

- [ ] **Step 4: Run** `tests/test_brain_detector_decode.py tests/test_lesion_boxes.py` → all pass.

- [ ] **Step 5: Commit** — `git add anatobind/eval/lesion_boxes.py tests/test_brain_detector_decode.py && git commit -m "Lesion boxes: brain minimum component size and a families option for decode_boxes, knee default unchanged"`

---

### Task 2: Dataset logic `anatobind/nnunet/brain_lesion.py`

**Files:**
- Create: `anatobind/nnunet/brain_lesion.py`
- Test: `tests/test_brain_detector_dataset.py`

**Interfaces:**
- Produces:
  - constants `DATASET_ID = 903`, `DATASET_NAME`, `TRAINER`, `LABELS`, `FAMILY = "small_lesion"`, `FAMILIES = {1: FAMILY}`, `NORMAL_LABEL = "Normal for age"`
  - `normal_files(csv_rows) -> list[str]` (csv_rows = `csv.DictReader` rows of brain.csv, study-level rows included)
  - `check_members_inside(members_of, shape)` raises `ValueError` naming the lesion id
  - `paint_members(members_of, shape) -> uint8 (col, row, slice)`; `members_of = {lesion_id: [{"x", "y", "width", "height", "slice"}, ...]}` in the RSS frame
  - `gt_boxes(registry_rows) -> [{"lesion_id", "family", "box": (x0, y0, z0, x1, y1, z1 + 1)}]`
  - `assign_normal_folds(patients, k=5, seed=0) -> {patient: fold}`
  - `case_folds(lesion_cases, normal_cases, lesion_patient_fold, normal_patient_fold) -> {case: fold}` (`*_cases = {case: patient_id}`), raises `ValueError` on patient overlap
  - `make_splits(case_fold, k=5) -> [{"train": [...], "val": [...]}]`
  - `write_label(label, image_path, out_path)`; `build_raw(raw_root, cases, write_case) -> Path`; `write_splits(preprocessed_root, case_fold, k=5) -> Path`
  - `validation_path(results_root, config, fold, case)`, `validation_npz_path(...)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_brain_detector_dataset.py
import json

import nibabel as nib
import numpy as np
import pytest

from anatobind.nnunet.brain_lesion import (
    DATASET_NAME, assign_normal_folds, build_raw, case_folds, check_members_inside, gt_boxes, make_splits,
    normal_files, paint_members, validation_npz_path, validation_path, write_label, write_splits,
)


def _m(x, y, w, h, s):
    return {"x": x, "y": y, "width": w, "height": h, "slice": s}


def test_normal_files_need_every_row_to_be_normal_and_flair():
    rows = [{"file": "file_brain_AXFLAIR_1", "label": "Normal for age"},
            {"file": "file_brain_AXFLAIR_2", "label": "Normal for age"}, {"file": "file_brain_AXFLAIR_2", "label": "Mass "},
            {"file": "file_brain_AXT1_3", "label": "Normal for age"}, {"file": "file_brain_AXFLAIR_4", "label": " Normal for age "}]
    assert normal_files(rows) == ["file_brain_AXFLAIR_1", "file_brain_AXFLAIR_4"]


def test_paint_and_inside_check():
    members = {0: [_m(2, 3, 4, 2, 1), _m(2, 3, 4, 3, 2)], 1: [_m(10, 10, 2, 2, 0)]}
    lab = paint_members(members, (16, 16, 4))
    assert lab.dtype == np.uint8 and lab.sum() == 8 + 12 + 4 and lab[2:6, 3:5, 1].all() and lab[10:12, 10:12, 0].all()
    check_members_inside(members, (16, 16, 4))
    with pytest.raises(ValueError, match="lesion 7"):
        check_members_inside({7: [_m(14, 0, 4, 2, 0)]}, (16, 16, 4))
    with pytest.raises(ValueError, match="lesion 8"):
        check_members_inside({8: [_m(0, 0, 2, 2, 4)]}, (16, 16, 4))


def test_gt_boxes_close_the_inclusive_slice_range():
    rows = [{"lesion_id": 5, "x0": 2, "y0": 3, "x1": 6, "y1": 6, "z0": 1, "z1": 2}]
    assert gt_boxes(rows) == [{"lesion_id": 5, "family": "small_lesion", "box": (2, 3, 1, 6, 6, 3)}]


def test_folds_by_patient_and_overlap_refused():
    nf = assign_normal_folds(["n3", "n1", "n2", "n4", "n5", "n6"], 5, 0)
    assert set(nf) == {"n1", "n2", "n3", "n4", "n5", "n6"} and set(nf.values()) <= set(range(5)) and len(set(nf.values())) == 5
    assert assign_normal_folds(["n1", "n2", "n3", "n4", "n5", "n6"], 5, 0) == nf
    cf = case_folds({"a": "p1", "b": "p2"}, {"c": "n1"}, {"p1": 0, "p2": 3}, nf)
    assert cf == {"a": 0, "b": 3, "c": nf["n1"]}
    with pytest.raises(ValueError, match="p1"):
        case_folds({"a": "p1"}, {"c": "p1"}, {"p1": 0}, {"p1": 1})
    sp = make_splits({"a": 0, "b": 1, "c": 0}, 2)
    assert sp == [{"train": ["b"], "val": ["a", "c"]}, {"train": ["a", "c"], "val": ["b"]}]


def test_build_raw_writes_layout_and_refuses_existing(tmp_path):
    def write_case(case, img, lab):
        img.write_text("i")
        lab.write_text("l")
    base = build_raw(tmp_path, ["c1", "c2"], write_case)
    assert base.name == DATASET_NAME and (base / "imagesTr" / "c1_0000.nii.gz").exists() and (base / "labelsTr" / "c2.nii.gz").exists()
    meta = json.loads((base / "dataset.json").read_text())
    assert meta == {"channel_names": {"0": "FLAIR"}, "labels": {"background": 0, "small_lesion": 1}, "numTraining": 2, "file_ending": ".nii.gz"}
    with pytest.raises(FileExistsError):
        build_raw(tmp_path, ["c1"], write_case)


def test_write_label_shares_affine_and_shape_and_splits_refuse_overwrite(tmp_path):
    aff = np.diag([0.5, 0.5, 5.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 3), np.float32), aff), str(tmp_path / "img.nii.gz"))
    lab = np.zeros((8, 8, 3), np.uint8)
    lab[1:3, 1:3, 1] = 1
    write_label(lab, tmp_path / "img.nii.gz", tmp_path / "lab.nii.gz")
    back = nib.load(str(tmp_path / "lab.nii.gz"))
    assert np.allclose(back.affine, aff) and back.get_data_dtype() == np.uint8 and np.array_equal(np.asanyarray(back.dataobj), lab)
    with pytest.raises(ValueError):
        write_label(np.zeros((8, 8, 4), np.uint8), tmp_path / "img.nii.gz", tmp_path / "lab2.nii.gz")
    p = write_splits(tmp_path, {"a": 0, "b": 1}, 2)
    assert json.loads(p.read_text())[0]["val"] == ["a"]
    with pytest.raises(FileExistsError):
        write_splits(tmp_path, {"a": 0, "b": 1}, 2)


def test_validation_paths():
    p = validation_path("/r", "2d", 3, "case")
    assert str(p) == f"/r/{DATASET_NAME}/nnUNetTrainer_250epochs__nnUNetPlans__2d/fold_3/validation/case.nii.gz"
    assert str(validation_npz_path("/r", "2d", 3, "case")).endswith("validation/case.npz")
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement `anatobind/nnunet/brain_lesion.py`**

```python
"""nnU-Net v2 dataset for the brain small-lesion detector (spec 2026-09-28 §2).

Cases are the 165 Gate 0.5 registry volumes (1297 small lesions) and the 88 FLAIR volumes whose every fastMRI+ row
is "Normal for age". Each registry lesion's per-slice member boxes (RSS frame, the Level R export path) are painted
into a binary map on the raw RSS grid (col, row, slice); normal volumes are all zero. Lesion patients keep the Level R
fold table; normal patients are dealt into five folds with seed 0; the two sets must not overlap. nnU-Net's custom
splits make its cross-validation outputs out-of-fold predictions.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ID = 903
DATASET_NAME = f"Dataset{DATASET_ID}_FastMRIBrainSmallLesion"
TRAINER = "nnUNetTrainer_250epochs"
LABELS = {"background": 0, "small_lesion": 1}
FAMILY = "small_lesion"
FAMILIES = {1: FAMILY}
NORMAL_LABEL = "Normal for age"


def normal_files(csv_rows):
    labels = {}
    for r in csv_rows:
        if "AXFLAIR" in r["file"]:
            labels.setdefault(r["file"], set()).add(r["label"].strip())
    return sorted(f for f, s in labels.items() if s == {NORMAL_LABEL})


def check_members_inside(members_of, shape):
    nc, nr, ns = shape
    for lid, members in members_of.items():
        for m in members:
            if not (0 <= m["x"] and m["x"] + m["width"] <= nc and 0 <= m["y"] and m["y"] + m["height"] <= nr
                    and 0 <= m["slice"] < ns):
                raise ValueError(f"lesion {lid}: member box {m} leaves the grid {shape}")


def paint_members(members_of, shape):
    lab = np.zeros(shape, np.uint8)
    for members in members_of.values():
        for m in members:
            lab[m["x"]:m["x"] + m["width"], m["y"]:m["y"] + m["height"], m["slice"]] = 1
    return lab


def gt_boxes(registry_rows):
    """Registry boxes as (x0, y0, z0, x1, y1, z1 + 1) on the (col, row, slice) grid; x1 / y1 are already exclusive."""
    return [{"lesion_id": r["lesion_id"], "family": FAMILY, "box": (r["x0"], r["y0"], r["z0"], r["x1"], r["y1"], r["z1"] + 1)}
            for r in registry_rows]


def assign_normal_folds(patients, k=5, seed=0):
    ids = sorted(set(patients))
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {ids[i]: int(j % k) for j, i in enumerate(perm)}


def case_folds(lesion_cases, normal_cases, lesion_patient_fold, normal_patient_fold):
    overlap = sorted(set(lesion_cases.values()) & set(normal_cases.values()))
    if overlap:
        raise ValueError(f"patients {overlap} are both lesion and normal patients")
    out = {c: int(lesion_patient_fold[p]) for c, p in lesion_cases.items()}
    out.update({c: int(normal_patient_fold[p]) for c, p in normal_cases.items()})
    return out


def make_splits(case_fold, k=5):
    cases = sorted(case_fold)
    return [{"train": [c for c in cases if case_fold[c] != f], "val": [c for c in cases if case_fold[c] == f]} for f in range(k)]


def write_label(label, image_path, out_path):
    img = nib.load(str(image_path))
    if tuple(img.shape) != tuple(label.shape):
        raise ValueError(f"label shape {label.shape} != image shape {img.shape} ({image_path})")
    out = nib.Nifti1Image(np.asarray(label, np.uint8), img.affine)
    out.set_data_dtype("uint8")
    out.header.set_xyzt_units("mm")
    nib.save(out, str(out_path))


def build_raw(raw_root, cases, write_case):
    base = Path(raw_root) / DATASET_NAME
    if base.exists():
        raise FileExistsError(f"{base} exists; the dataset is never rebuilt in place")
    (base / "imagesTr").mkdir(parents=True)
    (base / "labelsTr").mkdir()
    for c in cases:
        write_case(c, base / "imagesTr" / f"{c}_0000.nii.gz", base / "labelsTr" / f"{c}.nii.gz")
    meta = {"channel_names": {"0": "FLAIR"}, "labels": LABELS, "numTraining": len(cases), "file_ending": ".nii.gz"}
    (base / "dataset.json").write_text(json.dumps(meta, indent=1))
    return base


def write_splits(preprocessed_root, case_fold, k=5):
    d = Path(preprocessed_root) / DATASET_NAME
    d.mkdir(parents=True, exist_ok=True)
    p = d / "splits_final.json"
    if p.exists():
        raise FileExistsError(f"{p} exists")
    p.write_text(json.dumps(make_splits(case_fold, k), indent=1))
    return p


def validation_path(results_root, config, fold, case):
    return Path(results_root) / DATASET_NAME / f"{TRAINER}__nnUNetPlans__{config}" / f"fold_{fold}" / "validation" / f"{case}.nii.gz"


def validation_npz_path(results_root, config, fold, case):
    return validation_path(results_root, config, fold, case).with_suffix("").with_suffix(".npz")
```

- [ ] **Step 4: Run** `tests/test_brain_detector_dataset.py` → pass.
- [ ] **Step 5: Commit** — `git commit -m "Brain detector dataset: normal-volume selection, member-box painting with a grid check, patient folds with an overlap refusal, nnU-Net raw layout and splits"`

---

### Task 3: Build Dataset903 on real data (`scripts/brain_detector_prepare.py`)

**Files:**
- Create: `scripts/brain_detector_prepare.py`
- Create: `docs/verification/<date>/brain_detector/dataset.md`

**Interfaces:**
- Consumes: Task 2; `anatobind.level_r.registry.load_registry`; `anatobind.level_r.export.small_lesion_rows / merged_lesions / match_registry`; `anatobind.data_engine.fastmri.read_fastmri_plus_rows, rss_h5_to_nifti`; `anatobind.data_engine.fastmri_knee.volume_geometry`.
- Produces: `raw/Dataset903_…/{imagesTr, labelsTr, dataset.json, cases.json}` where `cases.json = {case: {"patient_id": str, "kind": "lesion" | "normal"}}`; `preprocessed/Dataset903_…/splits_final.json`.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
# scripts/brain_detector_prepare.py
"""Build Dataset903 (spec 2026-09-28 §2) or write its splits.

  source scripts/nnunet_env.sh
  PYTHONPATH=. nice -n 19 python scripts/brain_detector_prepare.py --stage raw
  nice -n 19 nnUNetv2_plan_and_preprocess -d 903 -c 2d 3d_fullres --verify_dataset_integrity -np 4
  PYTHONPATH=. python scripts/brain_detector_prepare.py --stage splits
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import read_fastmri_plus_rows, rss_h5_to_nifti  # noqa: E402
from anatobind.data_engine.fastmri_knee import volume_geometry  # noqa: E402
from anatobind.level_r.export import match_registry, merged_lesions, small_lesion_rows  # noqa: E402
from anatobind.level_r.registry import load_registry  # noqa: E402
from anatobind.nnunet.brain_lesion import (  # noqa: E402
    DATASET_NAME, assign_normal_folds, build_raw, case_folds, check_members_inside, normal_files, paint_members,
    write_label, write_splits,
)

FM = Path("/data2/congcong/data/FM_data")
CSV = FM / "fastMRI_lh_brain_knee/Annotations/brain.csv"
KROOT = FM / "fastMRI_lh_brain_knee/kspace/brain"
FOLDS = Path("data/level_r/folds.json")


def h5_of(stem):
    for split in ("multicoil_train", "multicoil_val"):
        p = KROOT / split / f"{stem}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def stage_raw(raw_root):
    registry = load_registry()
    reg_by_file, csv_by_file = {}, {}
    for r in registry:
        reg_by_file.setdefault(r["file"], []).append(r)
    for r in small_lesion_rows(read_fastmri_plus_rows(CSV)):
        csv_by_file.setdefault(r["file"], []).append(r)
    with open(CSV, newline="") as fh:
        normals = normal_files(list(csv.DictReader(fh)))
    lesion_files = sorted(reg_by_file)
    if set(normals) & set(lesion_files):
        raise ValueError("a normal volume is also a registry volume")
    info = {}

    def write_case(case, img_path, lab_path):
        path = h5_of(case)
        g = volume_geometry(path)
        rss_h5_to_nifti(path, img_path, pad_to_slices=0)
        shape = nib.load(str(img_path)).shape
        if shape != (g["n_cols"], g["n_rows"], g["slices"]):
            raise ValueError(f"{case}: image {shape} != RSS {(g['n_cols'], g['n_rows'], g['slices'])}")
        if case in reg_by_file:
            matched = match_registry(reg_by_file[case], merged_lesions(csv_by_file.get(case, []), g["n_rows"]))
            members_of = {lid: L["members"] for lid, L in matched.items()}
            check_members_inside(members_of, shape)
            lab = paint_members(members_of, shape)
            kind = "lesion"
        else:
            lab = np.zeros(shape, np.uint8)
            kind = "normal"
        write_label(lab, img_path, lab_path)
        info[case] = {"patient_id": g["patient_id"], "kind": kind, "n_label_voxels": int(lab.sum())}
        print(f"{len(info)}/{len(lesion_files) + len(normals)} {case} {kind} {int(lab.sum())} label voxels", flush=True)

    base = build_raw(raw_root, lesion_files + normals, write_case)
    (base / "cases.json").write_text(json.dumps(info, indent=1))
    n_lesion = sum(1 for v in info.values() if v["kind"] == "lesion")
    print(f"wrote {base}: {len(info)} cases ({n_lesion} lesion, {len(info) - n_lesion} normal), "
          f"{sum(1 for v in info.values() if v['kind'] == 'normal' and v['n_label_voxels'])} normal cases with label voxels")


def stage_splits(raw_root, preprocessed_root):
    info = json.loads((Path(raw_root) / DATASET_NAME / "cases.json").read_text())
    lesion = {c: v["patient_id"] for c, v in info.items() if v["kind"] == "lesion"}
    normal = {c: v["patient_id"] for c, v in info.items() if v["kind"] == "normal"}
    fold_of = case_folds(lesion, normal, json.loads(FOLDS.read_text())["patient_fold"], assign_normal_folds(normal.values()))
    p = write_splits(preprocessed_root, fold_of)
    per = {f: sum(1 for v in fold_of.values() if v == f) for f in range(5)}
    print(f"wrote {p}: cases per fold {per}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("raw", "splits"), required=True)
    a = ap.parse_args()
    if a.stage == "raw":
        stage_raw(os.environ["nnUNet_raw"])
    else:
        stage_splits(os.environ["nnUNet_raw"], os.environ["nnUNet_preprocessed"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the raw stage** (`source scripts/nnunet_env.sh; PYTHONPATH=. nice -n 19 python scripts/brain_detector_prepare.py --stage raw | tee $D/prepare_raw.txt`, `D=docs/verification/<date>/brain_detector`). Expected last line: `253 cases (165 lesion, 88 normal), 0 normal cases with label voxels`. Anything else: stop and report.
- [ ] **Step 3: Label check** — a one-off check printed to `$D/label_check.txt`: for each lesion case, the label map equals `paint_members` of its registry members (re-derived) and the number of registry lesions per case matches `load_registry`; total lesions 1297.
- [ ] **Step 4: Plan and preprocess** — `nice -n 19 nnUNetv2_plan_and_preprocess -d 903 -c 2d 3d_fullres --verify_dataset_integrity -np 4 2>&1 | tee $D/plan_preprocess.txt` (CPU; record wall time and the 2d / 3d_fullres patch sizes from the plans file).
- [ ] **Step 5: Splits** — `PYTHONPATH=. python scripts/brain_detector_prepare.py --stage splits | tee $D/splits.txt`; every case appears in exactly one val list.
- [ ] **Step 6: Record** `$D/dataset.md` (commands, counts, cases per fold, plans summary) and commit script + records: `git commit -m "Brain detector: build Dataset903 from the registry and the normal FLAIR volumes, preprocess it and write the patient splits"`

---

### Task 4: Evaluation (`anatobind/eval/brain_detector.py`, `scripts/eval_brain_detector.py`)

**Files:**
- Create: `anatobind/eval/brain_detector.py`, `scripts/eval_brain_detector.py`
- Test: `tests/test_brain_detector_eval.py`

**Interfaces:**
- Consumes: `decode_boxes`, `BRAIN_MIN_VOXELS`, `load_label_map`, `load_nnunet_probabilities` (Task 1 / existing); `sweep`, `gate`, `operating_point`, `match_scan` (`detection_metrics`); `gt_boxes`, `FAMILIES`, `validation_path`, `validation_npz_path` (Task 2).
- Produces: `scan_record(case, gt, label_map, probs) -> dict`; `normal_fp_per_scan(scans, normal_cases, thr) -> float`; `strata_sensitivity(scans, thr, stratum_of) -> {stratum: {"n_gt", "n_hit", "sensitivity"}}`; `collect(results_root, config, splits, gt_of_case) -> list` (raises `FileNotFoundError` naming a missing case); `evaluate(scans, normal_cases) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_brain_detector_eval.py
import numpy as np
import pytest

from anatobind.eval.brain_detector import collect, evaluate, normal_fp_per_scan, scan_record, strata_sensitivity


def _det(box, score):
    return {"family": "small_lesion", "box": box, "score": score}


def _scans():
    gt_a = [{"lesion_id": 0, "family": "small_lesion", "box": (0, 0, 0, 4, 4, 1)},
            {"lesion_id": 1, "family": "small_lesion", "box": (10, 10, 0, 14, 14, 1)}]
    return [{"case": "a", "gt": gt_a, "dets": [_det((0, 0, 0, 4, 4, 1), 0.9), _det((20, 20, 0, 24, 24, 1), 0.3)]},
            {"case": "n", "gt": [], "dets": [_det((5, 5, 0, 8, 8, 1), 0.6)]}]


def test_scan_record_decodes_with_the_brain_minimum():
    lab = np.zeros((20, 20, 3), np.uint8)
    lab[2:6, 2:6, 1] = 1
    probs = np.zeros((2,) + lab.shape, np.float32)
    probs[1][lab == 1] = 0.7
    rec = scan_record("c", [], lab, probs)
    assert rec["case"] == "c" and len(rec["dets"]) == 1 and rec["dets"][0]["box"] == (2, 2, 1, 6, 6, 2)


def test_evaluate_gate_normal_fp_and_strata():
    scans = _scans()
    out = evaluate(scans, {"n"})
    op = out["gate"]
    # sensitivity is 0.5 at every threshold up to 0.9 (only lesion 0 is found); the operating point is the highest such threshold
    assert op["sensitivity_family"] == 0.5 and op["pass"] is True
    assert normal_fp_per_scan(scans, {"n"}, 0.5) == 1.0 and normal_fp_per_scan(scans, {"n"}, 0.7) == 0.0
    st = strata_sensitivity(scans, 0.5, {0: "small", 1: "large"})
    assert st == {"small": {"n_gt": 1, "n_hit": 1, "sensitivity": 1.0}, "large": {"n_gt": 1, "n_hit": 0, "sensitivity": 0.0}}


def test_collect_refuses_a_missing_validation_case(tmp_path):
    with pytest.raises(FileNotFoundError, match="case_x"):
        collect(tmp_path, "2d", [{"train": [], "val": ["case_x"]}], {"case_x": []})
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `anatobind/eval/brain_detector.py`**

```python
"""Out-of-fold evaluation of the brain small-lesion detector (spec 2026-09-28 §4): decode nnU-Net validation outputs
into scored 3D boxes, sweep thresholds, and read the D1 gate; normal-volume false positives and strata are reported,
not gated."""
from anatobind.eval.detection_metrics import gate, match_scan, sweep
from anatobind.eval.lesion_boxes import BRAIN_MIN_VOXELS, decode_boxes, load_label_map, load_nnunet_probabilities
from anatobind.nnunet.brain_lesion import FAMILIES, validation_npz_path, validation_path


def scan_record(case, gt, label_map, probs):
    return {"case": case, "gt": gt, "dets": decode_boxes(label_map, probs, BRAIN_MIN_VOXELS, FAMILIES)}


def collect(results_root, config, splits, gt_of_case):
    scans = []
    for fold, split in enumerate(splits):
        for case in split["val"]:
            nii, npz = validation_path(results_root, config, fold, case), validation_npz_path(results_root, config, fold, case)
            if not nii.exists() or not npz.exists():
                raise FileNotFoundError(f"fold {fold} case {case}: missing {nii if not nii.exists() else npz}")
            lab = load_label_map(nii)
            scans.append(scan_record(case, gt_of_case[case], lab, load_nnunet_probabilities(npz, lab)))
    return scans


def normal_fp_per_scan(scans, normal_cases, thr):
    normal = [s for s in scans if s["case"] in normal_cases]
    return sum(sum(1 for d in s["dets"] if d["score"] >= thr) for s in normal) / len(normal) if normal else 0.0


def strata_sensitivity(scans, thr, stratum_of):
    out = {}
    for s in scans:
        dets = [d for d in s["dets"] if d["score"] >= thr]
        pairs = match_scan(s["gt"], dets)
        for g, r in enumerate(s["gt"]):
            k = stratum_of[r["lesion_id"]]
            e = out.setdefault(k, {"n_gt": 0, "n_hit": 0})
            e["n_gt"] += 1
            e["n_hit"] += int(g in pairs)
    for e in out.values():
        e["sensitivity"] = e["n_hit"] / e["n_gt"]
    return out


def evaluate(scans, normal_cases):
    rows = sweep(scans)
    g = gate(rows)
    thr = g["thr"]
    return {"rows": rows, "gate": g, "normal_fp_per_scan": normal_fp_per_scan(scans, normal_cases, thr) if thr is not None else None,
            "n_scans": len(scans), "n_gt": sum(len(s["gt"]) for s in scans)}
```

- [ ] **Step 4: Write `scripts/eval_brain_detector.py`** — args `--config {2d,3d_fullres} --out DIR` (refuse an existing `--out`); reads `cases.json`, `splits_final.json`, the registry; builds `gt_of_case` with `gt_boxes` (normal cases `[]`); asserts `sum(n_gt) == 1297` and 253 scans; runs `collect` and `evaluate`; strata: `band`, `n_slices == 1` vs `> 1`, in-plane tertile of `inplane_mm` over the 1297, `stratum_geometry`, each via `strata_sensitivity` at the operating threshold; writes `REPORT.md` (gate line as JSON, FROC table from `rows`, normal FP per volume, strata tables, command), `froc.csv`, `output.txt`.

- [ ] **Step 5: Run tests; commit** — `git commit -m "Brain detector evaluation: out-of-fold decoding, the D1 gate, normal-volume false positives and strata, with the report script"`

---

### Task 5: Inference entry point (`anatobind/infer/brain.py`, `scripts/infer_brain_lesions.py`)

**Files:**
- Create: `anatobind/infer/brain.py`, `scripts/infer_brain_lesions.py`
- Test: `tests/test_brain_detector_infer.py`

**Interfaces:**
- Consumes: `nnunet_env` from `anatobind.infer.knee`; Task 1/2 names.
- Produces: `run_nnunet(dataset_id, in_dir, out_dir, folds, gpu, config)`; `lesion_rows(dets, label_map) -> [{"z0", "z1", "boxes": {"<slice>": [[row0, row1, col0, col1], ...]}, "score"}]`; `run(h5_path, out_dir, folds, gpu, config="2d", predict=run_nnunet) -> list`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_brain_detector_infer.py
import json

import numpy as np
import pytest

from anatobind.infer.brain import lesion_rows, run


def test_lesion_rows_give_per_slice_row_col_boxes():
    lab = np.zeros((20, 20, 4), np.uint8)
    lab[3:7, 5:8, 1] = 1
    lab[4:6, 5:9, 2] = 1
    dets = [{"box": (3, 5, 1, 7, 9, 3), "score": 0.8, "family": "small_lesion"}]
    assert lesion_rows(dets, lab) == [{"z0": 1, "z1": 2, "score": 0.8, "boxes": {"1": [[5, 8, 3, 7]], "2": [[5, 9, 4, 6]]}}]


def test_run_writes_a_table_and_handles_no_detections(tmp_path, monkeypatch):
    import anatobind.infer.brain as B

    def fake_rss_to_nifti(h5, out, pad_to_slices):
        import nibabel as nib
        nib.save(nib.Nifti1Image(np.zeros((16, 16, 3), np.float32), np.eye(4)), str(out))

    def fake_predict(dataset_id, in_dir, out_dir, folds, gpu, config):
        import nibabel as nib
        out_dir.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((16, 16, 3), np.uint8), np.eye(4)), str(out_dir / "case.nii.gz"))
        p = np.zeros((2, 3, 16, 16), np.float32)
        p[0] = 1.0
        np.savez(out_dir / "case.npz", probabilities=p)

    monkeypatch.setattr(B, "rss_h5_to_nifti", fake_rss_to_nifti)
    rows = run(tmp_path / "x.h5", tmp_path / "out", [0], 0, predict=fake_predict)
    assert rows == [] and json.loads((tmp_path / "out" / "lesions.json").read_text()) == []
    with pytest.raises(FileExistsError):
        run(tmp_path / "x.h5", tmp_path / "out", [0], 0, predict=fake_predict)
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `anatobind/infer/brain.py`**

```python
"""Brain small-lesion inference for S5 (spec 2026-09-28 §5): a fastMRI FLAIR h5 → nnU-Net (Dataset903) → scored
lesions with per-slice [row0, row1, col0, col1] boxes in the RSS frame, the Level R export's box format."""
import json
import subprocess
from pathlib import Path

import numpy as np

from anatobind.data_engine.fastmri import rss_h5_to_nifti
from anatobind.eval.lesion_boxes import BRAIN_MIN_VOXELS, decode_boxes, load_label_map, load_nnunet_probabilities
from anatobind.infer.knee import nnunet_env
from anatobind.nnunet.brain_lesion import DATASET_ID, FAMILIES, TRAINER


def run_nnunet(dataset_id, in_dir, out_dir, folds, gpu, config):
    cmd = ["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir), "-d", str(dataset_id), "-c", config, "-tr", TRAINER,
           "-f", *[str(f) for f in folds], "-npp", "2", "-nps", "2", "--disable_progress_bar", "--save_probabilities"]
    subprocess.run(cmd, check=True, env=nnunet_env(gpu))


def lesion_rows(dets, label_map):
    out = []
    for d in dets:
        c0, r0, s0, c1, r1, s1 = d["box"]
        boxes = {}
        for s in range(s0, s1):
            cols, rows = np.nonzero(label_map[c0:c1, r0:r1, s])
            if cols.size:
                boxes[str(s)] = [[int(r0 + rows.min()), int(r0 + rows.max() + 1), int(c0 + cols.min()), int(c0 + cols.max() + 1)]]
        out.append({"z0": int(s0), "z1": int(s1 - 1), "score": float(d["score"]), "boxes": boxes})
    return out


def run(h5_path, out_dir, folds, gpu, config="2d", predict=run_nnunet):
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    (out / "input").mkdir(parents=True)
    rss_h5_to_nifti(h5_path, out / "input" / "case_0000.nii.gz", pad_to_slices=0)
    predict(DATASET_ID, out / "input", out / "pred", folds, gpu, config)
    lab = load_label_map(out / "pred" / "case.nii.gz")
    dets = decode_boxes(lab, load_nnunet_probabilities(out / "pred" / "case.npz", lab), BRAIN_MIN_VOXELS, FAMILIES)
    rows = lesion_rows(dets, lab)
    (out / "lesions.json").write_text(json.dumps(rows, indent=1))
    return rows
```

- [ ] **Step 4: Write `scripts/infer_brain_lesions.py`** — args `--h5 --out --folds 0 1 2 3 4 --gpu 0 --config 2d`; calls `run`; prints the number of lesions and the output path.
- [ ] **Step 5: Run tests; commit** — `git commit -m "Brain detector inference: h5 to scored lesions with per-slice RSS boxes for S5"`

---

### Task 6: GPU launcher and timing probe (`scripts/brain_detector_train.py`)

**Files:**
- Create: `scripts/brain_detector_train.py`
- Test: `tests/test_brain_detector_train.py`

**Interfaces:**
- Produces: `idle_gpus(nvidia_smi_csv, candidates, busy_pids) -> list[int]` (a card is idle when used memory < 1000 MiB and no compute process); `assign(folds, gpus) -> [(fold, gpu)]` (round-robin; more folds than GPUs queue behind the first ones); `train_command(config, fold, trainer) -> list[str]`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_brain_detector_train.py
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("bdt", Path(__file__).resolve().parents[1] / "scripts/brain_detector_train.py")
bdt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdt)


def test_idle_gpus_parse_nvidia_smi_and_skip_busy_cards():
    csv = "0, 14\n1, 30000\n2, 14\n3, 14\n4, 14\n"
    assert bdt.idle_gpus(csv, [0, 1, 2, 3], busy_pids={}) == [0, 2, 3]
    assert bdt.idle_gpus(csv, [0, 2], busy_pids={2: [123]}) == [0]


def test_assign_folds_round_robin_and_command():
    assert bdt.assign([0, 1, 2, 3, 4], [0, 1, 2, 3]) == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0)]
    assert bdt.train_command("2d", 3, "nnUNetTrainer_250epochs") == ["nnUNetv2_train", "903", "2d", "3", "-tr", "nnUNetTrainer_250epochs", "--npz"]
```

- [ ] **Step 2: Implement** — `idle_gpus(csv_text, candidates, busy_pids)` parses `index, memory.used` lines (`nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits`), keeps candidates with memory < 1000 and no entry in `busy_pids` (gpu → pids, from `--query-compute-apps=gpu_uuid,pid` mapped by the caller); `assign` = `[(f, gpus[i % len(gpus)]) for i, f in enumerate(folds)]`; `train_command` as tested. `main()`: args `--config --folds --gpus 0 1 2 3 --trainer --dry-run`; queries nvidia-smi, refuses to start if fewer idle cards than requested and prints which are busy; for each GPU launches its folds sequentially in one `setsid bash -c` chain with `CUDA_VISIBLE_DEVICES=<gpu>`, `nice -n 19`, the nnU-Net env (`source scripts/nnunet_env.sh`), logging to `logs/brain_detector/<config>_fold<k>.log` (gitignored `logs/` already exists); `--dry-run` prints the chains without launching.
- [ ] **Step 3: Run tests; commit** — `git commit -m "Brain detector launcher: idle-GPU check, fold-to-GPU assignment and setsid training chains"`
- [ ] **Step 4: Timing probe (D8)** — `python scripts/brain_detector_train.py --config 2d --folds 0 --gpus 0 --trainer nnUNetTrainer_5epochs`; read the per-epoch time from the nnU-Net log (`epoch_time` lines); project `250 × epoch_time × 2` (four folds in parallel, then fold 4) for 2d and, after a 3d_fullres probe the same way, for 3d. Record in `docs/verification/<date>/brain_detector/timing.md` with the raw log lines. **If the projected 2d + 3d wall clock exceeds 24 h, stop here and report to the user.** The probe's results live under `nnUNetTrainer_5epochs__…` and are left in place.

---

### Task 7: Train, evaluate, infer smoke, documents

**Files:**
- Create: `docs/verification/<date>/brain_detector/{REPORT.md (2d), REPORT_3d.md, froc*.csv, output*.txt, infer_smoke.md}`
- Modify: `CLAUDE.md` (code-map paragraph, test count, status bullet), `STATUS.md` is the controller's.

- [ ] **Step 1: Launch 2d** — `python scripts/brain_detector_train.py --config 2d --folds 0 1 2 3 4 --gpus 0 1 2 3 --trainer nnUNetTrainer_250epochs`. Monitor the logs; each fold ends with a `validation` directory holding one `.nii.gz` + `.npz` per val case.
- [ ] **Step 2: Evaluate 2d** — `PYTHONPATH=. python scripts/eval_brain_detector.py --config 2d --out docs/verification/<date>/brain_detector/2d`. The REPORT's gate line is the S2 verdict (pass or fail are both valid; no tuning, D9).
- [ ] **Step 3: Launch and evaluate 3d_fullres** the same way into `…/3d_fullres` (report only).
- [ ] **Step 4: Inference smoke** — on one of the 130 SVD-impression FLAIR volumes without boxes: `python scripts/infer_brain_lesions.py --h5 <path> --out runs/brain_infer_smoke --folds 0 1 2 3 4 --gpu <idle>`; record the lesion count and one table row in `infer_smoke.md`.
- [ ] **Step 5: CLAUDE.md** — add after the `anatobind/relation/` paragraph: "`anatobind/nnunet/brain_lesion.py`（S2 脑侧小病灶检测器 Dataset903：框填掩膜、正常卷、患者折）、`anatobind/eval/brain_detector.py`（折外解码、D1 门、正常卷假阳、分层）、`anatobind/infer/brain.py`（h5 → 病灶表，给 S5）；脚本 `scripts/brain_detector_{prepare,train}.py`、`eval_brain_detector.py`、`infer_brain_lesions.py`；规格 `docs/superpowers/specs/2026-09-28-brain-detector-design.md`。" Update the test count; replace the status sentence with S2's verdict and a pointer to the report. Change nothing else.
- [ ] **Step 6: Full suite; commit** records and docs — `git commit -m "Brain detector S2: five-fold 2d and 3d_fullres results, the D1 gate verdict, an inference smoke run and the CLAUDE code map"`

---

## Self-review notes

- Spec coverage: §2 → Tasks 2–3; §3 → Tasks 3, 6, 7; §4 → Tasks 1, 4, 7; §5 → Task 5; §6 layout → all; §7 tests → Tasks 1, 2, 4, 5, 6; §8–9 → Task 7 + controller handoff; D7/D8 → Task 6.
- Types: boxes are `(col0, row0, slice0, col1, row1, slice1)` everywhere (decode output, `gt_boxes`, `iou3d` is axis-agnostic); `validation_path(results_root, config, fold, case)` in Tasks 2, 4; `FAMILIES = {1: "small_lesion"}` in Tasks 1, 2, 4, 5.
- Review Focus lines each have a test: grid check (Task 2), patient overlap (Task 2), npz axis order (reused loader, exercised in Task 5's fake npz), missing validation case (Task 4), no detections (Task 5).
