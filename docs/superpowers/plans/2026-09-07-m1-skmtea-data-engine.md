# M1 Plan 1: SKM-TEA Data Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the raw SKM-TEA release into the M1 training/evaluation dataset: screened lesion boxes with host labels, segmentation and images on one verified grid at 0.625 mm, clean plus noise- and undersampling-degraded reconstructions from the raw k-space, and a subject-level five-fold split.

**Architecture:** Pure-function modules under `anatobind/data_engine/` (annotations, frame, reconstruction, resampling, splits), each unit-tested on synthetic arrays, plus two thin scripts: a frame-verification report over all 155 scans (a gate before export) and a resumable exporter that writes one folder per scan under `/data2/congcong/data/FM_data/derived/skmtea/m1/`. Motion simulation is deliberately out of scope (separate plan once authorship is decided); this plan delivers noise and undersampling only.

**Tech Stack:** Python 3.11 in `~/anaconda3/envs/nvgen` (numpy 2.4, h5py 3.16, nibabel 5.4, scipy 1.17, pytest 9.1). Always run with `PYTHONNOUSERSITE=1`.

**Spec:** `RESEARCH_PLAN.md` v2.1 §4.1 (0.625 mm), §5.1 (SKM-TEA truth and inclusion rule D5), §6.1 (undersampling and noise operators), §7.2 (assets), §9.1 (M1 gate), §9.7 (cohort numbers); `docs/verification/2026-09-06/REPORT.md` (annotation findings). Read both before starting.

## Global Constraints

- Data is read only from `/data2/congcong/data/FM_data`; never write inside the raw dataset folders. Outputs go to `/data2/congcong/data/FM_data/derived/skmtea/m1/`.
- Raw h5: `SKM-TEA/files_recon_calib-24/MTR_xxx.h5` with `kspace (512,512,160,2,16)` complex64 in hybrid space (axis 0 = readout already in image domain; axes 1,2 = ky,kz), `maps (512,512,160,16,1)`, `target (512,512,160,2,1)`, `masks/poisson_{4,6,8,10,12,16}.0x (416,80)` bool. Eight scans have 16 coils, the rest 8; four scans have depth 168/152/156/144 instead of 160.
- Annotations: `SKM-TEA_ltr/annotations/v1.0.0/{train,val,test}.json`, COCO-like; `bbox = [x, y, z, w, h, d]` floats indexing the h5 array axes (verified 2026-09-07: boxes land on tissue only in the h5 frame); `tissue_id` 1 Meniscus, 2 ACL, 3 PCL, 4 Femoral Cartilage, 5 Patellar Cartilage, 6 Tibial Cartilage, −1 none; supercategories Meniscal Tear, Ligament Tear, Cartilage Lesion, Effusion.
- Segmentation: `SKM-TEA_ltr/segmentation_masks/dicom-track/MTR_xxx.nii.gz`, uint8 labels 1 patellar cartilage, 2 femoral cartilage, 3 tibial cartilage medial, 4 tibial cartilage lateral, 5 meniscus medial, 6 meniscus lateral. Its array must be transposed `(1, 0, 2)` to sit on the h5 grid (tissue/background contrast 3.7 vs 1.0 on MTR_201, verified 2026-09-07).
- Inclusion rule D5 (spec §5.1): drop boxes with any non-positive extent, drop boxes whose sorted endpoints leave the volume, drop records whose supercategory is incompatible with `tissue_id`; expected usable 465 = 311 in-segmentation / 116 effusion / 38 ligament. Host side for tissues 1 and 6 comes from overlap with the medial/lateral masks; overlap ratio in [0.4, 0.6] is "ambiguous".
- Resolution: in-plane 0.625 mm (2× block reduction from 0.3125 mm), slice 0.8 mm unchanged. Split: five folds by scan, 155 scans = 155 subjects.
- CPU only, `nice -n 19`, at most 4 worker processes × default thread pools. No GPU.
- Tests: `cd /data0/congcong/code/Project_Doing/foundation_model && PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q`. Every function is written test-first.
- Git: branch `plan-v3`, author identity from the repo config, no AI trailers in commit messages.

---

## File Structure

| File | Responsibility |
|---|---|
| `anatobind/data_engine/skmtea.py` | Constants (label tables), annotation loading, inclusion screening (D5), layer assignment, box normalisation, frame transform of the segmentation, host-side resolution |
| `anatobind/data_engine/skmtea_recon.py` | Hybrid k-space handling: adjoint SENSE reconstruction, Poisson mask embedding, undersampling, complex noise |
| `anatobind/data_engine/resample.py` | 2× in-plane block reduction for images (mean) and labels (majority), box scaling |
| `anatobind/data_engine/splits.py` | Deterministic stratified five-fold split by scan |
| `scripts/verify_skmtea_frames.py` | Gate: per-scan contrast test proving the transposed segmentation frame, written to `docs/verification/skmtea_frames.csv` |
| `scripts/build_skmtea_m1.py` | Resumable exporter: per scan, clean + noise ×3 + undersampled ×3 images, seg, screened boxes with host labels, global manifest |
| `tests/test_skmtea_annotations.py`, `tests/test_skmtea_frame.py`, `tests/test_skmtea_recon.py`, `tests/test_resample.py`, `tests/test_splits.py`, `tests/test_skmtea_export.py` | Unit tests on synthetic data (fast) |

---

### Task 1: Annotation loading and inclusion screening (rule D5)

**Files:**
- Create: `anatobind/data_engine/skmtea.py`
- Test: `tests/test_skmtea_annotations.py`

**Interfaces:**
- Produces: `SEG_LABELS: dict[int,str]`, `TISSUE_TO_SEG: dict[int, tuple[int,...]]`, `SUPER_TO_TISSUES: dict[str, set[int]]`, `layer_of(tissue_id) -> str`, `normalise_box(bbox) -> tuple[int,int,int,int,int,int]` (x0,y0,z0,x1,y1,z1 with x1 > x0 etc.), `screen_annotation(ann: dict, image: dict, supercategory: str) -> dict` with keys `keep: bool`, `reason: str`, `box: tuple|None`, `screen_split(json_path) -> list[dict]` (one row per annotation with `split, ann_id, scan_id, image_id, category_id, supercategory, tissue_id, layer, keep, reason, x0, y0, z0, x1, y1, z1, depth`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_skmtea_annotations.py
import json

import pytest

from anatobind.data_engine.skmtea import (
    SUPER_TO_TISSUES, TISSUE_TO_SEG, layer_of, normalise_box, screen_annotation, screen_split,
)

IMAGE = {"id": 7, "scan_id": "MTR_999", "matrix_shape": [512, 512, 160]}


def test_label_tables():
    assert TISSUE_TO_SEG[1] == (5, 6) and TISSUE_TO_SEG[6] == (3, 4)
    assert TISSUE_TO_SEG[4] == (2,) and TISSUE_TO_SEG[5] == (1,)
    assert SUPER_TO_TISSUES["Cartilage Lesion"] == {4, 5, 6}
    assert SUPER_TO_TISSUES["Effusion"] == {-1}


def test_layer_of():
    assert [layer_of(t) for t in (1, 4, 5, 6)] == ["in_seg"] * 4
    assert layer_of(-1) == "effusion" and layer_of(2) == "ligament" and layer_of(3) == "ligament"


def test_normalise_box_rounds_and_orders_endpoints():
    assert normalise_box([330.0, 232.0, 54.0, 5.0, 19.0, 10.0]) == (330, 232, 54, 335, 251, 64)
    assert normalise_box([10.0, 10.0, 10.0, -4.0, 3.0, 2.0]) == (6, 10, 10, 10, 13, 12)


def test_screen_keeps_a_valid_box():
    r = screen_annotation({"id": 1, "tissue_id": 4, "bbox": [10, 20, 30, 5, 6, 7]}, IMAGE, "Cartilage Lesion")
    assert r["keep"] and r["reason"] == "" and r["box"] == (10, 20, 30, 15, 26, 37)


def test_screen_drops_non_positive_extent():
    r = screen_annotation({"id": 1, "tissue_id": 4, "bbox": [10, 20, 30, -5, 6, 7]}, IMAGE, "Cartilage Lesion")
    assert not r["keep"] and r["reason"] == "non-positive extent"


def test_screen_drops_out_of_bounds_after_sorting():
    r = screen_annotation({"id": 1, "tissue_id": 4, "bbox": [500, 20, 30, 20, 6, 7]}, IMAGE, "Cartilage Lesion")
    assert not r["keep"] and r["reason"] == "out of bounds"


def test_screen_drops_category_tissue_mismatch():
    r = screen_annotation({"id": 1, "tissue_id": 2, "bbox": [10, 20, 30, 5, 6, 7]}, IMAGE, "Cartilage Lesion")
    assert not r["keep"] and r["reason"] == "category-tissue mismatch"


def test_screen_split_reads_coco_like_json(tmp_path):
    doc = {
        "categories": [{"id": 12, "name": "Cartilage Lesion (1)", "supercategory": "Cartilage Lesion"},
                       {"id": 16, "name": "Effusion", "supercategory": "Effusion"}],
        "images": [{"id": 7, "scan_id": "MTR_999", "matrix_shape": [512, 512, 160]}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 12, "tissue_id": 4, "bbox": [10, 20, 30, 5, 6, 7]},
            {"id": 2, "image_id": 7, "category_id": 16, "tissue_id": -1, "bbox": [10, 20, 30, 5, 6, -7]},
        ],
    }
    p = tmp_path / "train.json"
    p.write_text(json.dumps(doc))
    rows = screen_split(p)
    assert [r["keep"] for r in rows] == [True, False]
    assert rows[0]["layer"] == "in_seg" and rows[1]["layer"] == "effusion"
    assert rows[0]["split"] == "train" and rows[0]["scan_id"] == "MTR_999" and rows[0]["depth"] == 160
    assert (rows[0]["x0"], rows[0]["x1"]) == (10, 15)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_skmtea_annotations.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'anatobind.data_engine.skmtea'`

- [ ] **Step 3: Write the implementation**

```python
# anatobind/data_engine/skmtea.py
"""SKM-TEA annotations, inclusion rule D5, segmentation frame and host-side resolution."""
import json
from pathlib import Path

import numpy as np

SEG_LABELS = {
    1: "patellar_cartilage", 2: "femoral_cartilage",
    3: "tibial_cartilage_medial", 4: "tibial_cartilage_lateral",
    5: "meniscus_medial", 6: "meniscus_lateral",
}
TISSUE_NAMES = {1: "Meniscus", 2: "ACL", 3: "PCL", 4: "Femoral Cartilage", 5: "Patellar Cartilage",
                6: "Tibial Cartilage", -1: "none"}
TISSUE_TO_SEG = {1: (5, 6), 4: (2,), 5: (1,), 6: (3, 4), 2: (), 3: (), -1: ()}
SUPER_TO_TISSUES = {"Meniscal Tear": {1}, "Ligament Tear": {2, 3}, "Cartilage Lesion": {4, 5, 6}, "Effusion": {-1}}


def layer_of(tissue_id):
    if tissue_id in (1, 4, 5, 6):
        return "in_seg"
    if tissue_id == -1:
        return "effusion"
    return "ligament"


def normalise_box(bbox):
    """[x, y, z, w, h, d] -> integer (x0, y0, z0, x1, y1, z1) with sorted endpoints."""
    x, y, z, w, h, d = (int(round(float(v))) for v in bbox)
    lo = (min(x, x + w), min(y, y + h), min(z, z + d))
    hi = (max(x, x + w), max(y, y + h), max(z, z + d))
    return lo + hi


def screen_annotation(ann, image, supercategory):
    """Apply rule D5 to one record; the box is returned in h5 array coordinates."""
    w, h, d = (float(v) for v in ann["bbox"][3:6])
    shape = tuple(int(v) for v in image["matrix_shape"])
    if w <= 0 or h <= 0 or d <= 0:
        return {"keep": False, "reason": "non-positive extent", "box": None}
    box = normalise_box(ann["bbox"])
    if min(box[:3]) < 0 or any(box[3 + i] > shape[i] for i in range(3)):
        return {"keep": False, "reason": "out of bounds", "box": None}
    if ann["tissue_id"] not in SUPER_TO_TISSUES.get(supercategory, set()):
        return {"keep": False, "reason": "category-tissue mismatch", "box": None}
    return {"keep": True, "reason": "", "box": box}


def screen_split(json_path):
    json_path = Path(json_path)
    doc = json.loads(json_path.read_text())
    supercat = {c["id"]: c["supercategory"] for c in doc["categories"]}
    images = {im["id"]: im for im in doc["images"]}
    rows = []
    for ann in doc["annotations"]:
        im = images[ann["image_id"]]
        res = screen_annotation(ann, im, supercat[ann["category_id"]])
        box = res["box"] if res["box"] is not None else normalise_box(ann["bbox"])
        rows.append({
            "split": json_path.stem, "ann_id": ann["id"], "scan_id": im["scan_id"], "image_id": ann["image_id"],
            "category_id": ann["category_id"], "supercategory": supercat[ann["category_id"]],
            "tissue_id": ann["tissue_id"], "layer": layer_of(ann["tissue_id"]),
            "keep": res["keep"], "reason": res["reason"],
            "x0": box[0], "y0": box[1], "z0": box[2], "x1": box[3], "y1": box[4], "z1": box[5],
            "depth": int(im["matrix_shape"][2]),
        })
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_skmtea_annotations.py -q`
Expected: `8 passed`

- [ ] **Step 5: Check the rule against the real files (no code change)**

Run:
```bash
cd /data0/congcong/code/Project_Doing/foundation_model && PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -c "
from collections import Counter
from anatobind.data_engine.skmtea import screen_split
rows=[r for s in ('train','val','test') for r in screen_split(f'/data2/congcong/data/FM_data/SKM-TEA_ltr/annotations/v1.0.0/{s}.json')]
keep=[r for r in rows if r['keep']]
print(len(rows), 'kept', len(keep), Counter(r['layer'] for r in keep), Counter(r['reason'] for r in rows if not r['keep']))"
```
Expected: `476 kept 465 Counter({'in_seg': 311, 'effusion': 116, 'ligament': 38}) Counter({'non-positive extent': 9, 'category-tissue mismatch': 2})`. If the counts differ, stop and report; do not adjust the rule to fit.

- [ ] **Step 6: Commit**

```bash
git add anatobind/data_engine/skmtea.py tests/test_skmtea_annotations.py
git commit -m "SKM-TEA data engine: annotation screening (rule D5)"
```

---

### Task 2: Segmentation frame transform and the frame-verification gate

**Files:**
- Modify: `anatobind/data_engine/skmtea.py` (append)
- Create: `scripts/verify_skmtea_frames.py`
- Test: `tests/test_skmtea_frame.py`

**Interfaces:**
- Produces: `seg_nifti_to_h5_frame(arr: np.ndarray) -> np.ndarray` (transpose (1,0,2)), `tissue_contrast(magnitude: np.ndarray, seg_h5: np.ndarray) -> float`, `load_seg_h5_frame(nii_path) -> np.ndarray[uint8]`, `load_target_magnitude(h5_path, echo: int) -> np.ndarray[float32]` (512,512,depth).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_skmtea_frame.py
import h5py
import nibabel as nib
import numpy as np

from anatobind.data_engine.skmtea import (
    load_seg_h5_frame, load_target_magnitude, seg_nifti_to_h5_frame, tissue_contrast,
)


def test_seg_frame_swaps_the_first_two_axes():
    a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    b = seg_nifti_to_h5_frame(a)
    assert b.shape == (3, 2, 4) and b[1, 0, 2] == a[0, 1, 2]


def test_tissue_contrast_is_mean_inside_over_mean_outside():
    mag = np.ones((4, 4, 2), dtype=np.float32)
    seg = np.zeros((4, 4, 2), dtype=np.uint8)
    seg[:2] = 1
    mag[:2] = 3.0
    assert tissue_contrast(mag, seg) == 3.0


def test_loaders_return_h5_frame_and_echo_magnitude(tmp_path):
    seg = np.zeros((3, 2, 4), dtype=np.uint8)
    seg[0, 1, 2] = 5
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(tmp_path / "MTR_x.nii.gz"))
    tgt = np.zeros((2, 3, 4, 2, 1), dtype=np.complex64)
    tgt[1, 0, 2, 1, 0] = 3 + 4j
    with h5py.File(tmp_path / "MTR_x.h5", "w") as f:
        f.create_dataset("target", data=tgt)
    s = load_seg_h5_frame(tmp_path / "MTR_x.nii.gz")
    assert s.shape == (2, 3, 4) and s[1, 0, 2] == 5
    m = load_target_magnitude(tmp_path / "MTR_x.h5", echo=1)
    assert m.shape == (2, 3, 4) and m.dtype == np.float32 and m[1, 0, 2] == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_skmtea_frame.py -q`
Expected: `ImportError: cannot import name 'load_seg_h5_frame'`

- [ ] **Step 3: Append the implementation**

```python
# append to anatobind/data_engine/skmtea.py
import h5py
import nibabel as nib


def seg_nifti_to_h5_frame(arr):
    """The dicom-track NIfTI stores (y, x, z) relative to the h5 (x, y, z) grid."""
    return np.transpose(arr, (1, 0, 2))


def load_seg_h5_frame(nii_path):
    return seg_nifti_to_h5_frame(np.asarray(nib.load(str(nii_path)).dataobj)).astype(np.uint8)


def load_target_magnitude(h5_path, echo):
    with h5py.File(h5_path, "r") as f:
        return np.abs(f["target"][:, :, :, echo, 0]).astype(np.float32)


def tissue_contrast(magnitude, seg_h5):
    inside = seg_h5 > 0
    return float(magnitude[inside].mean() / magnitude[~inside].mean())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_skmtea_frame.py -q`
Expected: `3 passed`

- [ ] **Step 5: Write the verification script**

```python
# scripts/verify_skmtea_frames.py
"""Gate for the SKM-TEA export: for every scan the transposed segmentation must beat the identity frame."""
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.skmtea import load_target_magnitude, seg_nifti_to_h5_frame, tissue_contrast  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
RAW, SEG = FM / "SKM-TEA/files_recon_calib-24", FM / "SKM-TEA_ltr/segmentation_masks/dicom-track"
OUT = Path(__file__).resolve().parents[1] / "docs/verification/skmtea_frames.csv"

rows, bad = [], []
for nii in sorted(SEG.glob("MTR_*.nii.gz")):
    scan = nii.name[:-7]
    seg_nii = np.asarray(nib.load(str(nii)).dataobj)
    mag = load_target_magnitude(RAW / f"{scan}.h5", echo=0)
    seg_h5 = seg_nifti_to_h5_frame(seg_nii)
    if seg_h5.shape != mag.shape:
        rows.append({"scan": scan, "shape_ok": False, "contrast_identity": "", "contrast_transposed": ""})
        bad.append(scan)
        continue
    c_t = tissue_contrast(mag, seg_h5)
    c_i = tissue_contrast(mag, seg_nii) if seg_nii.shape == mag.shape else float("nan")
    rows.append({"scan": scan, "shape_ok": True, "contrast_identity": f"{c_i:.3f}", "contrast_transposed": f"{c_t:.3f}"})
    if not (c_t > 1.5 and (np.isnan(c_i) or c_t > c_i)):
        bad.append(scan)
    print(scan, rows[-1], flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} scans, {len(bad)} failing: {bad}")
sys.exit(1 if bad else 0)
```

- [ ] **Step 6: Run the gate (about 155 × 5 s)**

Run: `cd /data0/congcong/code/Project_Doing/foundation_model && PYTHONNOUSERSITE=1 nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/verify_skmtea_frames.py 2>&1 | tail -3`
Expected: `155 scans, 0 failing: []` and exit code 0. Any failing scan is reported to the user before continuing; do not special-case it silently.

- [ ] **Step 7: Commit**

```bash
git add anatobind/data_engine/skmtea.py tests/test_skmtea_frame.py scripts/verify_skmtea_frames.py docs/verification/skmtea_frames.csv
git commit -m "SKM-TEA data engine: segmentation frame transform and per-scan verification gate"
```

---

### Task 3: Host-side resolution for boxes

**Files:**
- Modify: `anatobind/data_engine/skmtea.py` (append)
- Test: `tests/test_skmtea_host.py`

**Interfaces:**
- Produces: `host_seg_label(seg_h5, box, tissue_id, pad=4, ambiguous=(0.4, 0.6)) -> dict` with keys `label: int|None`, `side: str` ("medial", "lateral", "single", "ambiguous", "none"), `ratio: float|None`, `n_voxels: int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_skmtea_host.py
import numpy as np

from anatobind.data_engine.skmtea import host_seg_label


def _seg():
    s = np.zeros((40, 40, 10), dtype=np.uint8)
    s[5:15, :, :] = 5   # medial meniscus block
    s[25:35, :, :] = 6  # lateral meniscus block
    s[:, :, 8:] = 2     # femoral cartilage slab at the top slices
    return s


def test_meniscus_box_on_medial_side():
    r = host_seg_label(_seg(), (6, 10, 2, 12, 20, 5), tissue_id=1)
    assert r["label"] == 5 and r["side"] == "medial" and r["ratio"] > 0.6


def test_meniscus_box_straddling_both_sides_is_ambiguous():
    s = _seg()
    s[15:25, :, :] = 0
    r = host_seg_label(s, (12, 10, 2, 28, 20, 5), tissue_id=1, pad=0)
    assert r["label"] is None and r["side"] == "ambiguous"


def test_single_label_tissue_reports_single():
    r = host_seg_label(_seg(), (0, 0, 8, 5, 5, 10), tissue_id=4)
    assert r["label"] == 2 and r["side"] == "single"


def test_no_tissue_voxels_near_box_is_none():
    r = host_seg_label(_seg(), (18, 18, 0, 22, 22, 2), tissue_id=4, pad=0)
    assert r["label"] is None and r["side"] == "none" and r["n_voxels"] == 0


def test_effusion_and_ligament_have_no_host():
    assert host_seg_label(_seg(), (0, 0, 0, 5, 5, 5), tissue_id=-1)["side"] == "none"
    assert host_seg_label(_seg(), (0, 0, 0, 5, 5, 5), tissue_id=2)["side"] == "none"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_skmtea_host.py -q`
Expected: `ImportError: cannot import name 'host_seg_label'`

- [ ] **Step 3: Append the implementation**

```python
# append to anatobind/data_engine/skmtea.py
def host_seg_label(seg_h5, box, tissue_id, pad=4, ambiguous=(0.4, 0.6)):
    """Which segmentation label hosts a lesion box (medial/lateral resolved by overlap)."""
    labels = TISSUE_TO_SEG.get(tissue_id, ())
    if not labels:
        return {"label": None, "side": "none", "ratio": None, "n_voxels": 0}
    x0, y0, z0, x1, y1, z1 = box
    sub = seg_h5[max(x0 - pad, 0):x1 + pad, max(y0 - pad, 0):y1 + pad, max(z0 - pad, 0):z1 + pad]
    counts = {l: int((sub == l).sum()) for l in labels}
    n = sum(counts.values())
    if n == 0:
        return {"label": None, "side": "none", "ratio": None, "n_voxels": 0}
    if len(labels) == 1:
        return {"label": labels[0], "side": "single", "ratio": 1.0, "n_voxels": n}
    medial, lateral = labels  # (5, 6) or (3, 4): medial label listed first
    ratio = counts[medial] / n
    if ratio >= ambiguous[1]:
        return {"label": medial, "side": "medial", "ratio": ratio, "n_voxels": n}
    if ratio <= ambiguous[0]:
        return {"label": lateral, "side": "lateral", "ratio": ratio, "n_voxels": n}
    return {"label": None, "side": "ambiguous", "ratio": ratio, "n_voxels": n}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_skmtea_host.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add anatobind/data_engine/skmtea.py tests/test_skmtea_host.py
git commit -m "SKM-TEA data engine: host-side resolution from medial/lateral masks"
```

---

### Task 4: Reconstruction, Poisson mask embedding, undersampling, noise

**Files:**
- Create: `anatobind/data_engine/skmtea_recon.py`
- Test: `tests/test_skmtea_recon.py`

**Interfaces:**
- Produces: `adjoint_sense(kspace_hybrid: complex (X,KY,KZ,C), maps: complex (X,Y,Z,C)) -> complex (X,Y,Z)`, `embed_poisson(mask: bool (416,80), ky: int, kz: int) -> bool (ky,kz)`, `undersample(kspace_hybrid, mask_kykz) -> complex`, `add_complex_noise(kspace_hybrid, sigma, rng) -> complex`, `noise_sigma(kspace_hybrid, fraction) -> float` (fraction of the RMS of the non-zero k-space entries).
- Convention (verified on MTR_001 echo 0: correlation 0.996 with `target`): axis 0 is already image domain; apply `ifftshift -> ifftn over axes (1,2) -> fftshift`, `norm="ortho"`, then `sum_c conj(maps_c) * x_c`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_skmtea_recon.py
import numpy as np

from anatobind.data_engine.skmtea_recon import (
    add_complex_noise, adjoint_sense, embed_poisson, noise_sigma, undersample,
)


def _forward(image, maps):
    """Test-only forward model matching the module's convention."""
    coil_imgs = image[..., None] * maps
    k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(coil_imgs, axes=(1, 2)), axes=(1, 2), norm="ortho"), axes=(1, 2))
    return k


def _case(seed=0, shape=(6, 8, 4), coils=3):
    rng = np.random.default_rng(seed)
    image = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    maps = rng.standard_normal(shape + (coils,)) + 1j * rng.standard_normal(shape + (coils,))
    maps /= np.sqrt((np.abs(maps) ** 2).sum(-1, keepdims=True))  # sum_c |S_c|^2 == 1
    return image, maps


def test_adjoint_sense_inverts_fully_sampled_forward_model():
    image, maps = _case()
    rec = adjoint_sense(_forward(image, maps), maps)
    assert rec.shape == image.shape
    assert np.allclose(rec, image, atol=1e-6)


def test_embed_poisson_centres_the_mask():
    m = np.ones((416, 80), dtype=bool)
    e = embed_poisson(m, ky=512, kz=160)
    assert e.shape == (512, 160) and e.dtype == bool
    assert e[48:464, 40:120].all() and not e[:48].any() and not e[464:].any() and not e[:, :40].any() and not e[:, 120:].any()


def test_undersample_zeroes_unsampled_lines_for_all_x_and_coils():
    k = np.ones((6, 8, 4, 3), dtype=np.complex64)
    mask = np.zeros((8, 4), dtype=bool)
    mask[2, 1] = True
    u = undersample(k, mask)
    assert u[:, 2, 1, :].all() and u.sum() == 6 * 3


def test_noise_sigma_is_fraction_of_rms_over_nonzero_entries():
    k = np.zeros((2, 4, 4, 1), dtype=np.complex64)
    k[0, 0, 0, 0] = 3 + 4j  # magnitude 5
    assert noise_sigma(k, fraction=0.5) == 2.5


def test_add_complex_noise_has_requested_std_and_zero_is_identity():
    k = np.zeros((32, 32, 32, 2), dtype=np.complex64)
    n = add_complex_noise(k, sigma=1.0, rng=np.random.default_rng(0))
    assert abs(np.sqrt((np.abs(n) ** 2).mean()) - 1.0) < 0.05
    assert np.array_equal(add_complex_noise(k, sigma=0.0, rng=np.random.default_rng(0)), k)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_skmtea_recon.py -q`
Expected: `ModuleNotFoundError: No module named 'anatobind.data_engine.skmtea_recon'`

- [ ] **Step 3: Write the implementation**

```python
# anatobind/data_engine/skmtea_recon.py
"""SKM-TEA hybrid k-space: adjoint SENSE, Poisson-disc mask embedding, undersampling, noise.

Layout (verified on MTR_001): kspace (X, KY, KZ, C) with the readout axis X already in the
image domain; maps (X, Y, Z, C). Reconstruction = ifftshift -> ifftn over (KY, KZ) -> fftshift,
orthonormal, then coil combination sum_c conj(S_c) * x_c.
"""
import numpy as np


def adjoint_sense(kspace_hybrid, maps):
    x = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace_hybrid, axes=(1, 2)), axes=(1, 2), norm="ortho"), axes=(1, 2))
    return (np.conj(maps) * x).sum(axis=-1)


def embed_poisson(mask, ky, kz):
    """Centre the (416, 80) acquisition-grid mask inside the (ky, kz) reconstruction grid."""
    out = np.zeros((ky, kz), dtype=bool)
    oy, oz = (ky - mask.shape[0]) // 2, (kz - mask.shape[1]) // 2
    out[oy:oy + mask.shape[0], oz:oz + mask.shape[1]] = mask.astype(bool)
    return out


def undersample(kspace_hybrid, mask_kykz):
    return kspace_hybrid * mask_kykz[None, :, :, None]


def noise_sigma(kspace_hybrid, fraction):
    mag = np.abs(kspace_hybrid)
    nz = mag[mag > 0]
    return float(fraction * np.sqrt((nz ** 2).mean()))


def add_complex_noise(kspace_hybrid, sigma, rng):
    if sigma == 0.0:
        return kspace_hybrid
    n = rng.standard_normal(kspace_hybrid.shape) + 1j * rng.standard_normal(kspace_hybrid.shape)
    return (kspace_hybrid + (sigma / np.sqrt(2.0)) * n).astype(kspace_hybrid.dtype)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_skmtea_recon.py -q`
Expected: `5 passed`

- [ ] **Step 5: Real-data check (no code change)**

Run:
```bash
cd /data0/congcong/code/Project_Doing/foundation_model && PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -c "
import h5py, numpy as np
from anatobind.data_engine.skmtea_recon import adjoint_sense, embed_poisson, undersample
f=h5py.File('/data2/congcong/data/FM_data/SKM-TEA/files_recon_calib-24/MTR_001.h5','r')
k=f['kspace'][:,:,:,0,:]; m=f['maps'][:,:,:,:,0]; t=f['target'][:,:,:,0,0]
def corr(a,b): a=a.ravel(); b=b.ravel(); return abs(np.vdot(a,b))/np.linalg.norm(a)/np.linalg.norm(b)
full=adjoint_sense(k,m); print('corr full vs target %.4f'%corr(full,t))
us=adjoint_sense(undersample(k, embed_poisson(f['masks/poisson_8.0x'][()],512,160)),m); print('corr 8x zero-filled vs target %.4f'%corr(us,t))"
```
Expected: full ≥ 0.99; 8× lower than full but above 0.8 (zero-filled aliasing). Record both numbers in the commit message.

- [ ] **Step 6: Commit**

```bash
git add anatobind/data_engine/skmtea_recon.py tests/test_skmtea_recon.py
git commit -m "SKM-TEA data engine: adjoint SENSE, Poisson mask embedding, undersampling and noise"
```

---

### Task 5: 2× in-plane resampling for images, labels and boxes

**Files:**
- Create: `anatobind/data_engine/resample.py`
- Test: `tests/test_resample.py`

**Interfaces:**
- Produces: `downsample2_inplane_image(vol: float (X,Y,Z)) -> float32 (X/2,Y/2,Z)` (2×2 mean), `downsample2_inplane_labels(lab: uint8 (X,Y,Z)) -> uint8 (X/2,Y/2,Z)` (majority of the 2×2 block, ties to the smallest label id), `scale_box_inplane(box, factor=0.5) -> tuple[int,...]` (floor on mins, ceil on maxes, z untouched).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_resample.py
import numpy as np

from anatobind.data_engine.resample import downsample2_inplane_image, downsample2_inplane_labels, scale_box_inplane


def test_image_downsample_averages_2x2_blocks_and_keeps_z():
    v = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    d = downsample2_inplane_image(v)
    assert d.shape == (2, 2, 2) and d.dtype == np.float32
    assert d[0, 0, 0] == v[0:2, 0:2, 0].mean()


def test_label_downsample_takes_the_majority_with_ties_to_smallest():
    lab = np.zeros((2, 2, 1), dtype=np.uint8)
    lab[0, 0, 0], lab[0, 1, 0], lab[1, 0, 0], lab[1, 1, 0] = 5, 5, 6, 0
    assert downsample2_inplane_labels(lab)[0, 0, 0] == 5
    lab[0, 1, 0] = 6  # now 5,6,6,0 -> 6
    assert downsample2_inplane_labels(lab)[0, 0, 0] == 6
    lab[:] = 0
    lab[0, 0, 0], lab[1, 1, 0] = 3, 4  # two zeros beat one 3 and one 4 -> 0
    assert downsample2_inplane_labels(lab)[0, 0, 0] == 0
    lab[:, :, 0] = [[3, 4], [4, 3]]  # exact 2/2 tie between 3 and 4 -> smallest label id 3
    assert downsample2_inplane_labels(lab)[0, 0, 0] == 3


def test_scale_box_inplane_floors_mins_and_ceils_maxes():
    assert scale_box_inplane((330, 232, 54, 335, 251, 64)) == (165, 116, 54, 168, 126, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_resample.py -q`
Expected: `ModuleNotFoundError: No module named 'anatobind.data_engine.resample'`

- [ ] **Step 3: Write the implementation**

```python
# anatobind/data_engine/resample.py
"""Exact 2x in-plane reductions used to bring SKM-TEA from 0.3125 mm to 0.625 mm."""
import math

import numpy as np


def _blocks(vol):
    x, y, z = vol.shape
    return vol[: x - x % 2, : y - y % 2].reshape(x // 2, 2, y // 2, 2, z)


def downsample2_inplane_image(vol):
    return _blocks(np.asarray(vol, dtype=np.float32)).mean(axis=(1, 3)).astype(np.float32)


def downsample2_inplane_labels(lab):
    b = _blocks(np.asarray(lab)).transpose(0, 2, 4, 1, 3).reshape(lab.shape[0] // 2, lab.shape[1] // 2, lab.shape[2], 4)
    n_labels = int(lab.max()) + 1
    counts = np.stack([(b == l).sum(-1) for l in range(n_labels)], axis=-1)  # (X/2, Y/2, Z, L)
    return np.argmax(counts, axis=-1).astype(np.uint8)  # argmax returns the smallest index on ties


def scale_box_inplane(box, factor=0.5):
    x0, y0, z0, x1, y1, z1 = box
    return (math.floor(x0 * factor), math.floor(y0 * factor), z0, math.ceil(x1 * factor), math.ceil(y1 * factor), z1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_resample.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add anatobind/data_engine/resample.py tests/test_resample.py
git commit -m "Data engine: exact 2x in-plane resampling for images, labels and boxes"
```

---

### Task 6: Five-fold split by scan

**Files:**
- Create: `anatobind/data_engine/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Produces: `five_fold_by_scan(scans: list[str], strata: dict[str,int], seed: int = 0, n_folds: int = 5) -> dict[str, int]` mapping scan -> fold index; `stratum_of(n_in_seg: int) -> int` (0 for 0, 1 for 1–2, 2 for ≥3).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_splits.py
from collections import Counter

from anatobind.data_engine.splits import five_fold_by_scan, stratum_of


def test_stratum_buckets():
    assert [stratum_of(n) for n in (0, 1, 2, 3, 9)] == [0, 1, 1, 2, 2]


def test_every_scan_gets_exactly_one_fold_and_sizes_are_balanced():
    scans = [f"MTR_{i:03d}" for i in range(155)]
    strata = {s: stratum_of(i % 4) for i, s in enumerate(scans)}
    folds = five_fold_by_scan(scans, strata, seed=0)
    assert set(folds) == set(scans)
    sizes = Counter(folds.values())
    assert set(sizes) == {0, 1, 2, 3, 4} and max(sizes.values()) - min(sizes.values()) <= 1


def test_strata_are_spread_across_folds_and_result_is_deterministic():
    scans = [f"MTR_{i:03d}" for i in range(155)]
    strata = {s: stratum_of(i % 4) for i, s in enumerate(scans)}
    a = five_fold_by_scan(scans, strata, seed=0)
    b = five_fold_by_scan(scans, strata, seed=0)
    assert a == b
    per_fold = Counter((a[s], strata[s]) for s in scans)
    for st in (0, 1, 2):
        counts = [per_fold[(f, st)] for f in range(5)]
        assert max(counts) - min(counts) <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_splits.py -q`
Expected: `ModuleNotFoundError: No module named 'anatobind.data_engine.splits'`

- [ ] **Step 3: Write the implementation**

```python
# anatobind/data_engine/splits.py
"""Deterministic stratified k-fold assignment at scan (= subject) level."""
import random


def stratum_of(n_in_seg):
    return 0 if n_in_seg == 0 else (1 if n_in_seg <= 2 else 2)


def five_fold_by_scan(scans, strata, seed=0, n_folds=5):
    rng = random.Random(seed)
    folds = {}
    next_fold = 0
    for st in sorted(set(strata.values())):
        members = sorted(s for s in scans if strata[s] == st)
        rng.shuffle(members)
        for s in members:  # deal round-robin, continuing the counter across strata keeps totals balanced
            folds[s] = next_fold % n_folds
            next_fold += 1
    return folds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_splits.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add anatobind/data_engine/splits.py tests/test_splits.py
git commit -m "Data engine: stratified five-fold split by scan"
```

---

### Task 7: Per-scan export and the M1 manifest

**Files:**
- Modify: `anatobind/data_engine/skmtea.py` (append `export_scan`)
- Create: `scripts/build_skmtea_m1.py`
- Test: `tests/test_skmtea_export.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `export_scan(h5_path, seg_nii_path, box_rows: list[dict], out_dir, conditions: dict, rng_seed: int) -> dict` (manifest row: `scan_id, out_dir, n_boxes_kept, n_ambiguous, files: list`), writing under `out_dir`:
  - `image_clean_e1.nii.gz`, `image_clean_e2.nii.gz` — magnitude of `target` echoes at 0.625 mm, float32, spacing (0.625, 0.625, 0.8)
  - `image_noise_q{1,2,3}_e1.nii.gz` — adjoint SENSE of noisy full k-space, noise fractions 0.25 / 0.5 / 1.0 of k-space RMS
  - `image_us{4,8,16}_e1.nii.gz` — adjoint SENSE of Poisson-undersampled k-space (masks from the file)
  - `seg.nii.gz` — labels at 0.625 mm (uint8), same affine
  - `boxes.csv` — one row per kept annotation: `ann_id, split, layer, supercategory, category_id, tissue_id, host_label, host_side, host_ratio, x0,y0,z0,x1,y1,z1` in 0.625 mm voxel coordinates, plus `x0_full…z1_full` in the 0.3125 mm grid
- Conditions are fixed here: `{"noise": [0.25, 0.5, 1.0], "us": [4, 8, 16]}`; degraded images are echo 1 only (echo 2 clean is kept for the arms that want two channels).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_skmtea_export.py
import csv

import h5py
import nibabel as nib
import numpy as np

from anatobind.data_engine.skmtea import export_scan


def _fake_scan(tmp_path, X=8, Y=8, Z=4, C=2):
    rng = np.random.default_rng(0)
    img = rng.standard_normal((X, Y, Z)) + 1j * rng.standard_normal((X, Y, Z))
    maps = rng.standard_normal((X, Y, Z, C)) + 1j * rng.standard_normal((X, Y, Z, C))
    maps /= np.sqrt((np.abs(maps) ** 2).sum(-1, keepdims=True))
    coil = img[..., None] * maps
    k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(coil, axes=(1, 2)), axes=(1, 2), norm="ortho"), axes=(1, 2))
    with h5py.File(tmp_path / "MTR_t.h5", "w") as f:
        f.create_dataset("kspace", data=np.stack([k, k], axis=3).astype(np.complex64))        # (X,KY,KZ,2,C)
        f.create_dataset("maps", data=maps[..., None].astype(np.complex64))                     # (X,Y,Z,C,1)
        f.create_dataset("target", data=np.stack([img, img], axis=3)[..., None].astype(np.complex64))  # (X,Y,Z,2,1)
        g = f.create_group("masks")
        for r in (4, 8, 16):
            m = np.zeros((Y - 2, Z - 2), dtype=bool); m[::2, :] = True
            g.create_dataset(f"poisson_{r}.0x", data=m)
    seg = np.zeros((Y, X, Z), dtype=np.uint8)  # NIfTI frame is (y, x, z)
    seg[:, :6, :] = 5  # h5-frame x < 6: medial meniscus; the padded box (x 0..7) stays mostly medial
    seg[:, 6:, :] = 6
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(tmp_path / "MTR_t.nii.gz"))
    return tmp_path / "MTR_t.h5", tmp_path / "MTR_t.nii.gz"


def test_export_scan_writes_images_seg_boxes_and_manifest_row(tmp_path):
    h5, nii = _fake_scan(tmp_path)
    boxes = [{"ann_id": 1, "split": "train", "layer": "in_seg", "supercategory": "Meniscal Tear", "category_id": 3,
              "tissue_id": 1, "keep": True, "x0": 0, "y0": 0, "z0": 0, "x1": 3, "y1": 4, "z1": 2},
             {"ann_id": 2, "split": "train", "layer": "in_seg", "supercategory": "Meniscal Tear", "category_id": 3,
              "tissue_id": 1, "keep": False, "x0": 0, "y0": 0, "z0": 0, "x1": 3, "y1": 4, "z1": 2}]
    out = tmp_path / "out"
    row = export_scan(h5, nii, boxes, out, conditions={"noise": [0.25, 0.5, 1.0], "us": [4, 8, 16]}, rng_seed=0)

    names = {p.name for p in out.iterdir()}
    for n in ["image_clean_e1.nii.gz", "image_clean_e2.nii.gz", "image_noise_q1_e1.nii.gz", "image_noise_q3_e1.nii.gz",
              "image_us4_e1.nii.gz", "image_us16_e1.nii.gz", "seg.nii.gz", "boxes.csv"]:
        assert n in names
    img = nib.load(str(out / "image_clean_e1.nii.gz"))
    assert img.shape == (4, 4, 4) and np.allclose(img.header.get_zooms(), (0.625, 0.625, 0.8))
    seg = nib.load(str(out / "seg.nii.gz"))
    assert seg.shape == (4, 4, 4) and set(np.unique(np.asarray(seg.dataobj)).tolist()) <= {5, 6}
    with open(out / "boxes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["ann_id"] == "1" and rows[0]["host_label"] == "5"
    assert (rows[0]["x0"], rows[0]["x1"]) == ("0", "2") and rows[0]["x1_full"] == "3"
    assert row["scan_id"] == "MTR_t" and row["n_boxes_kept"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_skmtea_export.py -q`
Expected: `ImportError: cannot import name 'export_scan'`

- [ ] **Step 3: Append the implementation**

```python
# append to anatobind/data_engine/skmtea.py
import csv

from anatobind.data_engine.resample import downsample2_inplane_image, downsample2_inplane_labels, scale_box_inplane
from anatobind.data_engine.skmtea_recon import add_complex_noise, adjoint_sense, embed_poisson, noise_sigma, undersample

SPACING_FULL = (0.3125, 0.3125, 0.8)
BOX_FIELDS = ["ann_id", "split", "layer", "supercategory", "category_id", "tissue_id", "host_label", "host_side",
              "host_ratio", "x0", "y0", "z0", "x1", "y1", "z1", "x0_full", "y0_full", "z0_full", "x1_full", "y1_full", "z1_full"]


def _affine(spacing):
    return np.diag([spacing[0], spacing[1], spacing[2], 1.0])


def _save(vol, spacing, path):
    img = nib.Nifti1Image(np.ascontiguousarray(vol), _affine(spacing))
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))


def export_scan(h5_path, seg_nii_path, box_rows, out_dir, conditions, rng_seed):
    """Write the M1 files for one scan; returns the manifest row."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spacing = (SPACING_FULL[0] * 2, SPACING_FULL[1] * 2, SPACING_FULL[2])
    files = []
    with h5py.File(h5_path, "r") as f:
        for echo in (0, 1):
            mag = np.abs(f["target"][:, :, :, echo, 0]).astype(np.float32)
            p = out_dir / f"image_clean_e{echo + 1}.nii.gz"
            _save(downsample2_inplane_image(mag), spacing, p)
            files.append(p.name)
        k = f["kspace"][:, :, :, 0, :]
        maps = f["maps"][:, :, :, :, 0]
        rng = np.random.default_rng(rng_seed)
        for q, frac in enumerate(conditions["noise"], start=1):
            rec = np.abs(adjoint_sense(add_complex_noise(k, noise_sigma(k, frac), rng), maps)).astype(np.float32)
            p = out_dir / f"image_noise_q{q}_e1.nii.gz"
            _save(downsample2_inplane_image(rec), spacing, p)
            files.append(p.name)
        for r in conditions["us"]:
            mask = embed_poisson(f[f"masks/poisson_{r}.0x"][()], ky=k.shape[1], kz=k.shape[2])
            rec = np.abs(adjoint_sense(undersample(k, mask), maps)).astype(np.float32)
            p = out_dir / f"image_us{r}_e1.nii.gz"
            _save(downsample2_inplane_image(rec), spacing, p)
            files.append(p.name)
    seg_h5 = load_seg_h5_frame(seg_nii_path)
    _save(downsample2_inplane_labels(seg_h5), spacing, out_dir / "seg.nii.gz")
    files.append("seg.nii.gz")
    rows, n_amb = [], 0
    for b in box_rows:
        if not b["keep"]:
            continue
        full = (b["x0"], b["y0"], b["z0"], b["x1"], b["y1"], b["z1"])
        host = host_seg_label(seg_h5, full, b["tissue_id"])
        n_amb += host["side"] == "ambiguous"
        half = scale_box_inplane(full)
        rows.append({
            "ann_id": b["ann_id"], "split": b["split"], "layer": b["layer"], "supercategory": b["supercategory"],
            "category_id": b["category_id"], "tissue_id": b["tissue_id"],
            "host_label": "" if host["label"] is None else host["label"], "host_side": host["side"],
            "host_ratio": "" if host["ratio"] is None else f"{host['ratio']:.3f}",
            "x0": half[0], "y0": half[1], "z0": half[2], "x1": half[3], "y1": half[4], "z1": half[5],
            "x0_full": full[0], "y0_full": full[1], "z0_full": full[2], "x1_full": full[3], "y1_full": full[4], "z1_full": full[5],
        })
    with open(out_dir / "boxes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BOX_FIELDS)
        w.writeheader()
        w.writerows(rows)
    files.append("boxes.csv")
    return {"scan_id": Path(h5_path).name[:-3], "out_dir": str(out_dir), "n_boxes_kept": len(rows),
            "n_ambiguous": n_amb, "files": files}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_skmtea_export.py -q`
Expected: `1 passed`. Then run the whole suite: `... -m pytest tests/ -q` → all green.

- [ ] **Step 5: Write the exporter CLI**

```python
#!/usr/bin/env python
# scripts/build_skmtea_m1.py
"""Export the M1 SKM-TEA dataset: one folder per scan, resumable, CPU process pool (nice 19).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/build_skmtea_m1.py --workers 4
Outputs: /data2/congcong/data/FM_data/derived/skmtea/m1/<scan>/..., annotations_screened.csv, splits.json, manifest.csv
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.skmtea import export_scan, screen_split  # noqa: E402
from anatobind.data_engine.splits import five_fold_by_scan, stratum_of  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
RAW, SEG, ANN = FM / "SKM-TEA/files_recon_calib-24", FM / "SKM-TEA_ltr/segmentation_masks/dicom-track", FM / "SKM-TEA_ltr/annotations/v1.0.0"
CONDITIONS = {"noise": [0.25, 0.5, 1.0], "us": [4, 8, 16]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, default=FM / "derived/skmtea/m1")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def job(args):
    scan, rows, out_dir, seed = args
    os.nice(19)
    t0 = time.time()
    try:
        row = export_scan(RAW / f"{scan}.h5", SEG / f"{scan}.nii.gz", rows, out_dir, CONDITIONS, rng_seed=seed)
        row["status"] = "ok"
    except Exception as exc:
        row = {"scan_id": scan, "out_dir": str(out_dir), "n_boxes_kept": "", "n_ambiguous": "", "files": [],
               "status": f"error: {type(exc).__name__}: {exc}"[:200]}
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    a = parse_args()
    a.out_root.mkdir(parents=True, exist_ok=True)
    rows = [r for s in ("train", "val", "test") for r in screen_split(ANN / f"{s}.json")]
    with open(a.out_root / "annotations_screened.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    scans = sorted(p.name[:-3] for p in RAW.glob("MTR_*.h5"))
    n_in_seg = Counter(r["scan_id"] for r in rows if r["keep"] and r["layer"] == "in_seg")
    folds = five_fold_by_scan(scans, {s: stratum_of(n_in_seg[s]) for s in scans}, seed=a.seed)
    (a.out_root / "splits.json").write_text(json.dumps({"seed": a.seed, "folds": folds}, indent=1))
    by_scan = {s: [r for r in rows if r["scan_id"] == s] for s in scans}
    pending = [s for s in scans if not (a.out_root / s / "boxes.csv").exists()]
    if a.limit:
        pending = pending[: a.limit]
    print(f"scans {len(scans)} | annotations {len(rows)} kept {sum(r['keep'] for r in rows)} | pending {len(pending)}", flush=True)
    if a.dry_run or not pending:
        return
    jobs = [(s, by_scan[s], a.out_root / s, a.seed + i) for i, s in enumerate(scans) if s in pending]
    manifest = a.out_root / "manifest.csv"
    existing = {}
    if manifest.exists():
        with open(manifest, newline="") as f:
            existing = {r["scan_id"]: r for r in csv.DictReader(f)}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for row in ex.map(job, jobs):
            row["files"] = ";".join(row["files"])
            existing[row["scan_id"]] = row
            with open(manifest, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["scan_id", "out_dir", "n_boxes_kept", "n_ambiguous", "files", "status", "seconds"])
                w.writeheader()
                w.writerows(existing[k] for k in sorted(existing))
            print(f"{row['scan_id']}: {row['status']} in {row['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Pilot on two scans, then the full run**

Run: `cd /data0/congcong/code/Project_Doing/foundation_model && PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/build_skmtea_m1.py --limit 2 --workers 2 2>&1 | tail -4`
Expected: two `ok` rows, about 60–120 s each; inspect `manifest.csv` and one `boxes.csv`. Then launch the full export in the background:
`nohup setsid env PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/build_skmtea_m1.py --workers 4 > /data2/congcong/data/FM_data/derived/skmtea/m1_build_$(date +%Y%m%d_%H%M).log 2>&1 < /dev/null &`
Expected total: 155 scans, ≈ 1 h with 4 workers, ≈ 45 GB on /data2.

- [ ] **Step 7: Acceptance checks (no code change)**

Run after the full export:
```bash
cd /data0/congcong/code/Project_Doing/foundation_model && PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -c "
import csv, glob, collections, nibabel as nib, numpy as np
M='/data2/congcong/data/FM_data/derived/skmtea/m1'
man=list(csv.DictReader(open(M+'/manifest.csv'))); print('manifest', collections.Counter(r['status'] for r in man))
boxes=[r for p in sorted(glob.glob(M+'/MTR_*/boxes.csv')) for r in csv.DictReader(open(p))]
print('boxes', len(boxes), collections.Counter(r['layer'] for r in boxes), 'host sides', collections.Counter(r['host_side'] for r in boxes if r['layer']=='in_seg'))
im=nib.load(M+'/MTR_001/image_clean_e1.nii.gz'); sg=nib.load(M+'/MTR_001/seg.nii.gz'); print(im.shape, sg.shape, im.header.get_zooms())"
```
Expected: 155 ok; 465 boxes = 311 in_seg / 116 effusion / 38 ligament; in_seg host sides mostly medial/lateral/single with a small `ambiguous` count (report it; the number goes into the plan §5.1); shapes (256,256,depth) with zooms (0.625, 0.625, 0.8).

- [ ] **Step 8: Commit and document**

```bash
git add anatobind/data_engine/skmtea.py scripts/build_skmtea_m1.py tests/test_skmtea_export.py
git commit -m "SKM-TEA data engine: per-scan M1 export with degraded reconstructions and host-labelled boxes"
```
Then add a short section to `docs/data_engine_synthseg.md`'s sibling `docs/data_engine_skmtea.md` (create it): where the outputs are, the frame transform, the conditions, counts from Step 7, the ambiguous-side count, and the runtime; commit it.

---

## Self-review notes

- Spec coverage: D5 screening (Task 1), frame/coordinate verification (Task 2, gate), side resolution (Task 3), operators from §6.1 except motion (Task 4; motion is a separate plan by design), 0.625 mm (Task 5), five-fold by scan (Task 6), assets and manifests (Task 7). §9.1's two arms and evaluation are Plans 2–4.
- Type consistency: `screen_split` rows use `keep/layer/x0..z1`; `export_scan` consumes exactly those keys; `host_seg_label` returns `label/side/ratio/n_voxels`; `five_fold_by_scan` returns scan→fold.
- Placeholders: none; every step has runnable code or an exact command with an expected result.
