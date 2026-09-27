# Brain Relation Baselines (PR-C, S1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the five relation baselines of v2.6 §10 (B0, Bprior, Bgeo+, B1, B2) on the 1297 fastMRI+ FLAIR small lesions, all reading one versioned feature table, with the outer-five-fold / inner-selection evaluation skeleton that later scores them against the sealed radiologist labels.

**Architecture:** One build script turns registry + SynthSeg + RSS into `derived/relation/v1/{table.csv, patches.npz, manifest.json}`. A `Table` object hands every arm the same columns (seven fixed host-class slots, §11 geometry, candidate flags, pseudo label C1) and the same patient folds. B0/Bprior/Bgeo+ are sklearn/numpy; B1/B2 are small torch models sharing one lesion encoder. `eval/relation_metrics.py` scores out-of-fold predictions against a pluggable label source (C1 now, sealed R later).

**Tech Stack:** Python 3.11, numpy 2.4, scipy 1.17 (EDT), scikit-learn 1.9 (Bgeo+), torch 2.5.1 (B1/B2), h5py, nibabel, pytest. Env: `~/anaconda3/envs/nvgen/bin/python` with `PYTHONNOUSERSITE=1 PYTHONPATH=.`.

**Spec:** `docs/superpowers/specs/2026-09-27-relation-baselines-design.md` (decisions P1–P16; §4 table columns; §5 arms; §6 evaluation; §9 acceptance).

## Global Constraints

- Every test command: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest <path> -q -p no:cacheprovider`. Tests never read `/data2` (spec §7); synthetic volumes only.
- Data is read only from `/data2/congcong/data/FM_data`; the real feature table is written only under `/data2/congcong/data/FM_data/derived/relation/`; smoke and run outputs go under the worktree's `runs/` (gitignored); an existing output directory is refused, never overwritten (spec §4.6; user rule: no data file is overwritten). Nothing is ever deleted.
- Grid convention: label maps and RSS volumes are `(col, row, slice)` arrays (like `derived/synthseg/*/seg_native`); lesion rects are `(col0, col1, row0, row1, slice)` from `geometry.member_rects` (spec §4.1).
- Slot order is `anatobind.eval.geometry.CLASS_NAMES` = `("white_matter", "cortex", "thalamus", "basal_ganglia", "brainstem", "cerebellum", "other_deep_grey")`; class id = index + 1; output column 7 is `none` (spec §4.3, §5).
- Constants copied from the spec: candidate radius 15 mm, distance cap 30 mm, soft-overlap sigma 1 mm, patch window 36 mm at 0.75 mm = 48 px, 3 slices, crops 22/32/42 px, midline share 0.4, tie tolerance 1e-3, bootstrap 10000 reps seed 0, AdamW lr 1e-3 wd 1e-4 batch 64 ≤ 40 epochs patience 8.
- Commits: author = repository local config (Congcong Liu); message in English, capitalised, describing what was done; **no Co-Authored-By / Generated-with / any AI trace**. One commit per task unless a step says otherwise.
- CPU jobs `nice -n 19` and at most 48 threads (CLAUDE.md); the run script caps numpy / scikit-learn / torch pools with `--threads` (default 16). Torch arms run on CPU by default: at plan validation 16 CPU threads did a real-data B1 epoch in 0.63 s and a B2 epoch in 0.24 s, while GPU 0 (approved for this project, but shared with another job at 99 % load) took 3.30 s and 1.59 s. Use `--device cuda` with `CUDA_VISIBLE_DEVICES=0` only when `nvidia-smi` shows GPU 0 idle; never move to another GPU without asking. Nothing runs longer than two hours without telling the user.
- `.github/workflows/tests.yml` is not modified; `requirements-ci.txt` gains one line (`scikit-learn==1.9.0`, Task 9) so CI can import sklearn; sklearn tests `pytest.importorskip("sklearn")`.

## File Map

| File | Responsibility |
|---|---|
| `anatobind/eval/geometry.py` (modify) | + per-class distance/index/inside maps, lesion mask/centroid, 10 slot features, side by label family, landmark distance, thirds |
| `anatobind/eval/lookup.py` (modify) | + `class_level_host` (C1 rule on slot features) |
| `anatobind/relation/__init__.py` (create) | empty |
| `anatobind/relation/table.py` (create) | column constants, `Table`, feature blocks, candidate masking, `write_table` / `load_table`, manifest |
| `anatobind/relation/build.py` (create) | `volume_rows`, `lesion_patch`, `build_table` with the §4.7 checks (loader injected for tests) |
| `anatobind/relation/labels.py` (create) | label sources C1 / R → `{lesion_id: acceptable set}`, `acceptable_matrix` |
| `anatobind/relation/baselines.py` (create) | B0, Bprior (4 variants), Bgeo+ (LR / HGB / MLP + grids) |
| `anatobind/relation/encoder.py` (create) | patch crop, flip/intensity augmentation, `LesionEncoder` |
| `anatobind/relation/models.py` (create) | `B2Model`, `B1Model`, `set_nll` |
| `anatobind/relation/cv.py` (create) | outer/inner folds, config orders, `select_config`, preds I/O, sklearn-arm drivers |
| `anatobind/relation/train.py` (create) | torch training loop, early stopping, refit, `run_torch_arm`, stage-2 hook |
| `anatobind/eval/relation_metrics.py` (create) | correctness, summary metrics, rescue/harm, patient bootstrap, McNemar, strata, Δd curve, `gate_r1` |
| `scripts/build_relation_table.py` (create) | real loaders → `derived/relation/v1` + verification numbers |
| `scripts/run_relation_baselines.py` (create) | all arms → `runs/relation/<run_id>/preds/*.csv`, `run.json` |
| `scripts/eval_relation_baselines.py` (create) | preds + labels → `REPORT.md`, `tables.csv`, `output.txt` |
| `tests/synth_relation.py` (create) | synthetic brain seg/RSS, synthetic registry/folds, synthetic `Table` + patches |
| `tests/test_relation_*.py` (create) | one file per task |

---

### Task 1: Slot geometry, side and landmark distance (`anatobind/eval/geometry.py`)

**Files:**
- Modify: `anatobind/eval/geometry.py` (append after `interface_margin`)
- Test: `tests/test_relation_geometry.py`

**Interfaces:**
- Consumes: existing `HOST_CLASSES`, `CLASS_NAMES`, `host_class_map`, `member_rects`.
- Produces (used by Tasks 2–5):
  - constants `CANDIDATE_MM = 15.0`, `DIST_CAP_MM = 30.0`, `SOFT_SIGMA_MM = 1.0`, `MIDLINE_SHARE = 0.4`, `LATERAL_VENTRICLES`, `LEFT_LABELS`, `RIGHT_LABELS`, `SLOT_FIELDS`
  - `lesion_mask(rects) -> (bbox, mask)`; `lesion_centroid(rects) -> np.ndarray(3,)`
  - `class_maps(class_map, spacing) -> {class id: (dist f32, idx i32 (3,...), inside f32)}`
  - `slot_features(class_map, maps, rects, spacing) -> {class id 1..7: {field: value}}`
  - `family_sides(seg) -> {"left": side, "right": side}`; `side_of(seg, rects, families) -> str`
  - `landmark_map(seg, spacing, labels=LATERAL_VENTRICLES) -> f32 map | None`; `min_in_lesion(dist_map, rects) -> float`
  - `third(index, lo, hi, names) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relation_geometry.py
import numpy as np
import pytest

from anatobind.eval.geometry import (
    CANDIDATE_MM, DIST_CAP_MM, CLASS_NAMES, LEFT_LABELS, RIGHT_LABELS, SLOT_FIELDS, class_maps, family_sides,
    host_class_map, landmark_map, lesion_centroid, lesion_mask, min_in_lesion, side_of, slot_features, third,
)

SP = (0.5, 0.5, 5.0)          # (col, row, slice) mm
WM, CTX = CLASS_NAMES.index("white_matter") + 1, CLASS_NAMES.index("cortex") + 1


def _seg():
    """(col, row, slice) = (40, 40, 4): left WM (2) cols 0..20 rows 0..20, right WM (41) cols 0..20 rows 20..40,
    cortex (3) cols 20..40; a lateral ventricle (4) at cols 2..6 rows 2..6."""
    seg = np.zeros((40, 40, 4), np.int16)
    seg[:20, :20, :] = 2
    seg[:20, 20:, :] = 41
    seg[20:, :, :] = 3
    seg[2:6, 2:6, :] = 4
    return seg


def test_lesion_mask_and_centroid_cover_the_union_of_rects():
    bbox, m = lesion_mask([(10, 14, 10, 12, 1), (12, 16, 10, 12, 2)])
    assert bbox == (10, 16, 10, 12, 1, 3) and m.shape == (6, 2, 2) and m.sum() == 16
    c = lesion_centroid([(10, 14, 10, 12, 1)])
    assert np.allclose(c, [11.5, 10.5, 1.0])


def test_slot_features_inside_white_matter():
    seg = _seg()
    cm = host_class_map(seg)
    maps = class_maps(cm, SP)
    f = slot_features(cm, maps, [(10, 14, 10, 14, 1)], SP)           # voxels cols 10..13, rows 10..13, slice 1
    assert set(f) == set(range(1, 8)) and set(f[WM]) == set(SLOT_FIELDS)
    assert f[WM]["ioa"] == 1.0 and f[WM]["min_surface_mm"] == 0.0 and f[WM]["signed_surface_mm"] < 0
    assert f[WM]["candidate"] and f[WM]["in_volume"]
    assert f[CTX]["ioa"] == 0.0 and f[CTX]["min_surface_mm"] == pytest.approx(3.5)     # col 13 -> col 20 = 7 px * 0.5 mm
    assert f[CTX]["dx_mm"] == pytest.approx(4.0) and f[CTX]["dy_mm"] == 0.0 and f[CTX]["dz_mm"] == 0.0
    # centroid voxel = rint(11.5, 11.5, 1) = (12, 12, 1); nearest cortex voxel is (20, 12, 1): 8 px * 0.5 mm along cols
    assert f[CTX]["centroid_distance_mm"] == pytest.approx(4.0)
    assert 0 < f[CTX]["soft_overlap"] < f[WM]["soft_overlap"] == 1.0
    absent = f[CLASS_NAMES.index("cerebellum") + 1]
    assert not absent["in_volume"] and not absent["candidate"] and absent["min_surface_mm"] == DIST_CAP_MM


def test_candidate_fallback_marks_the_nearest_class_when_none_is_within_15mm():
    seg = np.zeros((80, 20, 2), np.int16)
    seg[:4, :, :] = 2                                    # WM only at cols 0..4
    cm = host_class_map(seg)
    f = slot_features(cm, class_maps(cm, (1.0, 1.0, 5.0)), [(60, 64, 5, 9, 0)], (1.0, 1.0, 5.0))
    assert f[WM]["min_surface_mm"] == DIST_CAP_MM and f[WM]["candidate"]           # 57 mm away, capped, still the fallback candidate
    assert sum(v["candidate"] for v in f.values()) == 1


def _side_seg():
    """left family (2) at cols 0..20, right family (41) at cols 20..40, cortex (3) on rows 36..40, ventricle (4) at cols 2..6 rows 2..6."""
    seg = np.zeros((40, 40, 2), np.int16)
    seg[:20, :, :] = 2
    seg[20:, :, :] = 41
    seg[:, 36:, :] = 3
    seg[2:6, 2:6, :] = 4
    return seg


def test_side_by_label_family_follows_the_column_position_not_the_label_name():
    seg = _side_seg()
    fam = family_sides(seg)
    assert fam == {"left": "image_left", "right": "image_right"}
    assert side_of(seg, [(2, 8, 10, 14, 0)], fam) == "image_left"
    assert side_of(seg, [(30, 36, 10, 14, 0)], fam) == "image_right"
    assert side_of(seg, [(16, 24, 10, 14, 0)], fam) == "midline"                     # 4 cols of each family
    flipped = seg[::-1, :, :].copy()                                                   # label 2 now at large cols
    assert family_sides(flipped) == {"left": "image_right", "right": "image_left"}
    assert side_of(flipped, [(2, 8, 10, 14, 0)], family_sides(flipped)) == "image_left"


def test_side_falls_back_to_the_nearest_host_voxel_when_the_lesion_has_none():
    seg = _side_seg()
    assert side_of(seg, [(3, 5, 3, 5, 0)], family_sides(seg)) == "image_left"        # inside the ventricle, label 2 around it


def test_image_left_is_the_screen_left_of_the_reading_tool():
    """P8: image_left = the smaller RSS column. The Level R page paints a (slices, rows, cols) u16 slice into an
    ImageData of width C (columns) with no mirror, so column 0 is the left edge of the reader's screen."""
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "anatobind/level_r/app/app.js").read_text(encoding="utf-8")
    assert "const [S, R, C] = v.meta.shape, img = new ImageData(C, R)" in js
    assert "scale(-1" not in js and "rotate(" not in js


def test_landmark_distance_to_lateral_ventricles():
    seg = _seg()
    d = landmark_map(seg, SP)
    assert min_in_lesion(d, [(2, 6, 2, 6, 0)]) == 0.0
    assert min_in_lesion(d, [(10, 12, 2, 6, 0)]) == pytest.approx(2.5)          # col 10 -> col 5 = 5 px * 0.5 mm
    assert landmark_map(np.zeros((4, 4, 2), np.int16), SP) is None
    assert min_in_lesion(None, [(0, 1, 0, 1, 0)]) == DIST_CAP_MM


def test_third_splits_a_range_into_named_bins():
    names = ("top", "middle", "bottom")
    # t = (i - lo) / (hi - lo); first third is t < 1/3, second is t < 2/3: 3/9 == 1/3 exactly, so index 3 is "middle"
    assert [third(i, 0, 9, names) for i in (0, 2, 3, 4, 6, 9)] == ["top", "top", "middle", "middle", "bottom", "bottom"]
    assert third(5, 5, 5, names) == "middle"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_geometry.py -q -p no:cacheprovider`
Expected: ImportError on `CANDIDATE_MM` (names not defined).

- [ ] **Step 3: Implement (append to `anatobind/eval/geometry.py`)**

```python
# --- relation baselines (spec 2026-09-27 §4.2–4.3): per-slot geometry, side, landmarks ---------------------------
CANDIDATE_MM = 15.0        # §10.1: host classes within 15 mm of the lesion surface are candidates
DIST_CAP_MM = 30.0         # every distance is capped here; absent classes carry the cap
SOFT_SIGMA_MM = 1.0
MIDLINE_SHARE = 0.4        # both families >= 40 % of the host voxels -> midline
LATERAL_VENTRICLES = (4, 43, 5, 44)
LEFT_LABELS = (2, 3, 10, 11, 12, 13, 26, 7, 8, 17, 18, 28)
RIGHT_LABELS = (41, 42, 49, 50, 51, 52, 58, 46, 47, 53, 54, 60)
SLOT_FIELDS = ("in_volume", "candidate", "dx_mm", "dy_mm", "dz_mm", "centroid_distance_mm", "min_surface_mm",
               "signed_surface_mm", "ioa", "soft_overlap")


def lesion_mask(rects):
    """Union of clipped rects -> (bbox (c0, c1, r0, r1, s0, s1), bool mask over that bbox). Overlapping members count once."""
    c0, c1 = min(r[0] for r in rects), max(r[1] for r in rects)
    r0, r1 = min(r[2] for r in rects), max(r[3] for r in rects)
    s0, s1 = min(r[4] for r in rects), max(r[4] for r in rects) + 1
    m = np.zeros((c1 - c0, r1 - r0, s1 - s0), bool)
    for a, b, c, d, s in rects:
        m[a - c0:b - c0, c - r0:d - r0, s - s0] = True
    return (c0, c1, r0, r1, s0, s1), m


def lesion_centroid(rects):
    """Mean (col, row, slice) voxel coordinate of the lesion's voxels (voxel centres, float)."""
    (c0, _, r0, _, s0, _), m = lesion_mask(rects)
    idx = np.argwhere(m).astype(float)
    return idx.mean(0) + np.array([c0, r0, s0], float)


def class_maps(class_map, spacing):
    """{class id: (distance to the class in mm, nearest-voxel index (3, ...), distance to the class boundary from inside)}
    for every host class present. Distances are EDTs on the volume spacing. The outside distance stays float64: it is
    the same EDT as class_distance_maps, so d1 / d_interface / Δd recomputed from it equal the Gate 0.5 registry
    exactly (the build checks this to 1e-6)."""
    out = {}
    for c in np.unique(class_map):
        if c <= 0:
            continue
        member = class_map == c
        dist, idx = ndimage.distance_transform_edt(~member, sampling=spacing, return_indices=True)
        inside = ndimage.distance_transform_edt(member, sampling=spacing)
        out[int(c)] = (dist, idx.astype(np.int32), inside.astype(np.float32))
    return out


def _absent():
    return {"in_volume": False, "candidate": False, "dx_mm": 0.0, "dy_mm": 0.0, "dz_mm": 0.0,
            "centroid_distance_mm": DIST_CAP_MM, "min_surface_mm": DIST_CAP_MM, "signed_surface_mm": DIST_CAP_MM,
            "ioa": 0.0, "soft_overlap": 0.0}


def slot_features(class_map, maps, rects, spacing):
    """The ten spec §4.3 quantities for every host class id 1..len(CLASS_NAMES). dx/dy/dz point from the centroid voxel
    to the class voxel nearest to it (mm); signed_surface is -(max inside depth) when the lesion overlaps
    the class, else the minimum surface distance. If no class is within CANDIDATE_MM the nearest one is the candidate."""
    (c0, c1, r0, r1, s0, s1), m = lesion_mask(rects)
    cls = class_map[c0:c1, r0:r1, s0:s1][m]
    centroid = lesion_centroid(rects)
    cv = np.minimum(np.rint(centroid).astype(int), np.array(class_map.shape) - 1)   # the centroid voxel
    sp = np.asarray(spacing, float)
    out = {}
    for c in range(1, len(CLASS_NAMES) + 1):
        if c not in maps:
            out[c] = _absent()
            continue
        dist, idx, inside = maps[c]
        d = dist[c0:c1, r0:r1, s0:s1][m]
        min_surface = min(float(d.min()), DIST_CAP_MM)
        ioa = float((cls == c).mean())
        soft = float(np.exp(-d / SOFT_SIGMA_MM).mean())
        signed = -float(inside[c0:c1, r0:r1, s0:s1][m][cls == c].max()) if ioa > 0 else min_surface
        near = idx[:, cv[0], cv[1], cv[2]].astype(float)
        delta = (near - cv) * sp                          # from the centroid voxel to the class voxel nearest to it
        out[c] = {"in_volume": True, "candidate": bool(min_surface <= CANDIDATE_MM),
                  "dx_mm": float(delta[0]), "dy_mm": float(delta[1]), "dz_mm": float(delta[2]),
                  "centroid_distance_mm": min(float(np.linalg.norm(delta)), DIST_CAP_MM),
                  "min_surface_mm": min_surface, "signed_surface_mm": signed, "ioa": ioa, "soft_overlap": soft}
    if not any(v["candidate"] for v in out.values()):
        nearest = min((c for c in out if out[c]["in_volume"]), key=lambda c: out[c]["min_surface_mm"])
        out[nearest]["candidate"] = True
    return out


def family_sides(seg):
    """Which image side (smaller / larger column index) each SynthSeg label family occupies in this volume (P8)."""
    left_cols = np.nonzero(np.isin(seg, LEFT_LABELS))[0]
    right_cols = np.nonzero(np.isin(seg, RIGHT_LABELS))[0]
    if not left_cols.size or not right_cols.size:
        raise ValueError("both label families are needed to orient the volume")
    left_first = left_cols.mean() < right_cols.mean()
    return {"left": "image_left" if left_first else "image_right", "right": "image_right" if left_first else "image_left"}


def side_of(seg, rects, families):
    """image_left / image_right / midline by the family majority of the lesion's host voxels; no host voxel -> the family
    of the nearest host voxel to the lesion centroid."""
    (c0, c1, r0, r1, s0, s1), m = lesion_mask(rects)
    vals = seg[c0:c1, r0:r1, s0:s1][m]
    left, right = int(np.isin(vals, LEFT_LABELS).sum()), int(np.isin(vals, RIGHT_LABELS).sum())
    if left + right == 0:
        host = np.isin(seg, LEFT_LABELS + RIGHT_LABELS)
        _, idx = ndimage.distance_transform_edt(~host, return_indices=True)
        cv = np.rint(lesion_centroid(rects)).astype(int)
        lab = int(seg[tuple(idx[:, cv[0], cv[1], cv[2]])])
        return families["left"] if lab in LEFT_LABELS else families["right"]
    if min(left, right) / (left + right) >= MIDLINE_SHARE:
        return "midline"
    return families["left"] if left > right else families["right"]


def landmark_map(seg, spacing, labels=LATERAL_VENTRICLES):
    """Distance in mm to the nearest voxel of ``labels`` (None when the volume has none of them)."""
    member = np.isin(seg, labels)
    if not member.any():
        return None
    return ndimage.distance_transform_edt(~member, sampling=spacing).astype(np.float32)


def min_in_lesion(dist_map, rects):
    if dist_map is None:
        return DIST_CAP_MM
    (c0, c1, r0, r1, s0, s1), m = lesion_mask(rects)
    return min(float(dist_map[c0:c1, r0:r1, s0:s1][m].min()), DIST_CAP_MM)


def third(index, lo, hi, names):
    """Which third of [lo, hi] (inclusive) ``index`` falls in; a degenerate range is the middle third."""
    if hi <= lo:
        return names[1]
    t = (index - lo) / (hi - lo)
    return names[0] if t < 1 / 3 else names[1] if t < 2 / 3 else names[2]
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_geometry.py tests/test_geometry.py -q -p no:cacheprovider`
Expected: all pass, and the pre-existing `tests/test_geometry.py` still passes.

- [ ] **Step 5: Commit**

```bash
git add anatobind/eval/geometry.py tests/test_relation_geometry.py
git commit -m "Relation geometry: per-slot distances, overlaps and centroid vectors, image side by label family, landmark distance and thirds"
```

---

### Task 2: Class-level lookup C1 (`anatobind/eval/lookup.py`)

**Files:**
- Modify: `anatobind/eval/lookup.py` (append)
- Test: `tests/test_relation_lookup.py`

**Interfaces:**
- Consumes: `slot_features` output (Task 1).
- Produces: `class_level_host(slots) -> (class id, source, fraction)` with `source in ("overlap", "nearest")`; ties in overlap → the lower class id.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_relation_lookup.py
import numpy as np

from anatobind.eval.geometry import CLASS_NAMES, class_maps, host_class_map, slot_features
from anatobind.eval.lookup import class_level_host

WM, CTX, BG = (CLASS_NAMES.index(n) + 1 for n in ("white_matter", "cortex", "basal_ganglia"))
SP = (0.5, 0.5, 5.0)


def _seg():
    seg = np.zeros((40, 40, 2), np.int16)
    seg[:20, :, :] = 2                 # WM
    seg[20:30, :, :] = 3               # cortex
    seg[30:, :, :] = 24                # CSF: not a host
    seg[10:14, 30:34, :] = 11          # a caudate block inside the WM
    seg[14:18, 30:34, :] = 12          # a putamen block: caudate + putamen = basal ganglia
    return seg


def _slots(rects):
    cm = host_class_map(_seg())
    return slot_features(cm, class_maps(cm, SP), rects, SP)


def test_class_level_argmax_sums_the_members_of_a_class():
    # cols 8..18 rows 30..34: 2 WM cols + 4 caudate + 4 putamen -> class level: basal ganglia 8/10 > WM 2/10
    c, source, frac = class_level_host(_slots([(8, 18, 30, 34, 0)]))
    assert (c, source) == (BG, "overlap") and frac == 0.8


def test_zero_overlap_takes_the_nearest_in_volume_class():
    c, source, frac = class_level_host(_slots([(34, 38, 10, 14, 0)]))          # inside CSF
    assert (c, source, frac) == (CTX, "nearest", 0.0)


def test_overlap_tie_goes_to_the_lower_class_id():
    c, source, frac = class_level_host(_slots([(16, 24, 10, 14, 0)]))          # 4 WM cols + 4 cortex cols
    assert (c, source, frac) == (WM, "overlap", 0.5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_lookup.py -q -p no:cacheprovider`
Expected: ImportError `class_level_host`.

- [ ] **Step 3: Implement (append to `anatobind/eval/lookup.py`)**

```python
# --- brain: class-level lookup C1 (spec 2026-09-27 P6) -------------------------------------------------------------
def class_level_host(slots):
    """C1 on slot_features: the host class with the largest share of the lesion's voxels (sides merged, ventricles/CSF
    never candidates; ties -> lower class id); no overlap at all -> the in-volume class with the smallest surface
    distance. Returns (class id, "overlap" | "nearest", fraction)."""
    best = max(sorted(slots), key=lambda c: slots[c]["ioa"])
    if slots[best]["ioa"] > 0:
        return best, "overlap", slots[best]["ioa"]
    near = min((c for c in sorted(slots) if slots[c]["in_volume"]), key=lambda c: slots[c]["min_surface_mm"])
    return near, "nearest", 0.0
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_lookup.py tests/test_brain_lookup.py -q -p no:cacheprovider`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add anatobind/eval/lookup.py tests/test_relation_lookup.py
git commit -m "Brain lookup: class-level C1 host with overlap argmax, lower-id ties and nearest-class fallback"
```

---

### Task 3: Table schema, `Table`, feature blocks, candidate masking, write/load (`anatobind/relation/table.py`)

**Files:**
- Create: `anatobind/relation/__init__.py` (empty), `anatobind/relation/table.py`
- Test: `tests/test_relation_table.py`

**Interfaces (used by every later task):**
```python
SLOTS = CLASS_NAMES                                   # 7 names, fixed order
SLOT_PREFIX = {"white_matter": "wm", "cortex": "cortex", "thalamus": "thalamus", "basal_ganglia": "bg",
               "brainstem": "brainstem", "cerebellum": "cerebellum", "other_deep_grey": "odg"}
N_SLOTS, NONE_SLOT, N_OUT = 7, 7, 8
PAIR_FIELDS = ("dx_mm", "dy_mm", "dz_mm", "centroid_distance_mm", "signed_surface_mm", "min_surface_mm", "ioa", "soft_overlap")
LESION_GEOMETRY = ("extent_x_mm", "extent_y_mm", "extent_z_mm", "volume_mm3", "spacing_col_mm", "spacing_row_mm", "spacing_slice_mm", "slice_thickness_mm")
BRAIN_EXTRA = ("d_interface_mm", "delta_d_mm", "dist_cortex_mm", "dist_ventricle_mm")
SIDES = ("image_left", "image_right", "midline"); LESION_TYPES = ("nonspecific_wm_lesion", "lacunar_infarct")
LESION_TYPE_OF_LABEL = {"Nonspecific white matter lesion": "nonspecific_wm_lesion", "Lacunar infarct": "lacunar_infarct"}
ROW_THIRDS = ("top", "middle", "bottom"); SLICE_THIRDS = ("inferior", "middle", "superior")
PATCH_PX, PATCH_MM, PIXEL_MM, PATCH_SLICES, TABLE_VERSION = 48, 36.0, 0.75, 3, "v1"
LESION_COLUMNS: tuple                                  # ordered lesion-level columns (spec §4.2)
SLOT_COLUMNS: list                                     # 70 = 7 slots x SLOT_FIELDS, "<prefix>_<field>"
COLUMN_TYPES: dict                                     # column -> int | float | bool | str
def slot_col(slot_name, field) -> str
class Table:
    rows: list[dict]; lesion_id: np.ndarray(int); index: {lesion_id: row position}
    def column(name) -> np.ndarray
    def slot_matrix(field) -> np.ndarray (N, 7)
    def candidates() -> np.ndarray bool (N, 7)
    def c1_slot() -> np.ndarray int (N,)          # slot index of c1_class
    def folds() -> np.ndarray int (N,); def patients() -> np.ndarray str (N,)
    def slot_block() -> np.ndarray f32 (N, 7, 9)   # [candidate] + PAIR_FIELDS per slot
    def lesion_block() -> np.ndarray f32 (N, 17)   # LESION_GEOMETRY(8) + BRAIN_EXTRA(4) + side one-hot(3) + type one-hot(2)
    def subset(idx) -> Table
def features_flat(table) -> np.ndarray f32 (N, 80)          # slot_block.reshape(N, 63) ++ lesion_block
def features_per_slot(table) -> np.ndarray f32 (N, 7, 26)   # slot_block ++ lesion_block broadcast to every slot
def mask_to_candidates(probs, candidates) -> np.ndarray (N, 8)   # non-candidate slots -> 0, none kept, renormalised
def write_table(out_dir, rows, patches, manifest) -> Path      # refuses an existing out_dir
def load_table(dir, with_patches=True) -> (Table, dict | None)
def read_rows(csv_path) -> list[dict]                            # typed by COLUMN_TYPES
```
`patches` is `{"lesion_id": (N,) int64, "image": (N, 3, 48, 48) float16, "mask": (N, 3, 48, 48) bool, "classmap": (N, 3, 48, 48) int8}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relation_table.py
import json

import numpy as np
import pytest

from anatobind.eval.geometry import SLOT_FIELDS
from anatobind.relation.table import (
    BRAIN_EXTRA, COLUMN_TYPES, LESION_COLUMNS, LESION_GEOMETRY, N_OUT, PAIR_FIELDS, SLOTS, SLOT_COLUMNS, Table,
    features_flat, features_per_slot, load_table, mask_to_candidates, slot_col, write_table,
)


def _row(lid, patient, fold, c1="white_matter", side="image_left", ltype="nonspecific_wm_lesion", cand=("white_matter", "cortex")):
    r = {"lesion_id": lid, "file": f"file_{patient}", "patient_id": patient, "fold": fold, "stratum_geometry": "inplane_0.69_slice_5",
         "stratum_series": "200_201", "band": "0-2", "is_3mm": False, "lesion_type": ltype, "n_slices": 1, "inplane_mm": 4.8,
         "x0": 10, "y0": 10, "x1": 14, "y1": 14, "z0": 1, "z1": 1, "spacing_col_mm": 0.6875, "spacing_row_mm": 0.6875,
         "spacing_slice_mm": 5.0, "slice_thickness_mm": 5.0, "extent_x_mm": 2.75, "extent_y_mm": 2.75, "extent_z_mm": 5.0,
         "volume_mm3": 37.8, "centroid_col": 11.5, "centroid_row": 11.5, "centroid_slice": 1.0, "side": side,
         "row_third": "middle", "slice_third": "middle", "coarse_location": f"{side}|middle|middle", "d1_mm": 0.0,
         "d_interface_mm": 1.5, "delta_d_mm": 1.5, "dist_cortex_mm": 1.5, "dist_ventricle_mm": 12.0, "c1_class": c1,
         "c1_slot": SLOTS.index(c1), "c1_source": "overlap", "c1_overlap": 1.0, "registry_lookup_class": c1}
    for s in SLOTS:
        for f in SLOT_FIELDS:
            r[slot_col(s, f)] = {"in_volume": s in cand, "candidate": s in cand, "ioa": 1.0 if s == c1 else 0.0,
                                 "soft_overlap": 1.0 if s == c1 else 0.2}.get(f, 3.0 if s in cand else 30.0)
    return r


def test_column_lists_cover_the_spec_and_are_typed():
    assert len(SLOT_COLUMNS) == 70 and slot_col("basal_ganglia", "ioa") == "bg_ioa"
    assert set(LESION_COLUMNS) | set(SLOT_COLUMNS) == set(COLUMN_TYPES)
    assert "c1_slot" in LESION_COLUMNS and COLUMN_TYPES["is_3mm"] is bool and COLUMN_TYPES["d_interface_mm"] is float


def test_blocks_and_parity_between_flat_and_per_slot():
    t = Table([_row(0, "p0", 0), _row(1, "p1", 1, c1="cortex", side="midline", ltype="lacunar_infarct")])
    sb, lb = t.slot_block(), t.lesion_block()
    assert sb.shape == (2, 7, 9) and lb.shape == (2, 17) and sb.dtype == np.float32
    assert np.allclose(sb[0, 0], [1.0] + [t.rows[0][slot_col("white_matter", f)] for f in PAIR_FIELDS])
    assert np.allclose(lb[1, :8], [t.rows[1][c] for c in LESION_GEOMETRY])          # float32 block: 37.8 is not exact
    assert np.allclose(lb[1, 8:12], [t.rows[1][c] for c in BRAIN_EXTRA])
    assert list(lb[1, 12:15]) == [0.0, 0.0, 1.0] and list(lb[1, 15:]) == [0.0, 1.0]      # midline, lacunar
    flat, per = features_flat(t), features_per_slot(t)
    assert flat.shape == (2, 80) and per.shape == (2, 7, 26)
    assert np.array_equal(per[:, :, :9].reshape(2, 63), flat[:, :63]) and np.array_equal(per[:, 3, 9:], flat[:, 63:])
    assert t.c1_slot().tolist() == [0, 1] and t.candidates().sum(1).tolist() == [2, 2] and t.folds().tolist() == [0, 1]


def test_mask_to_candidates_zeroes_non_candidates_and_renormalises():
    probs = np.full((1, N_OUT), 1 / N_OUT)
    cand = np.zeros((1, 7), bool)
    cand[0, [0, 1]] = True
    out = mask_to_candidates(probs, cand)
    assert np.allclose(out[0, [0, 1, 7]], 1 / 3) and out[0, 2:7].sum() == 0 and np.isclose(out.sum(), 1)


def test_write_refuses_an_existing_directory_and_load_round_trips(tmp_path):
    rows = [_row(0, "p0", 0), _row(1, "p1", 1)]
    patches = {"lesion_id": np.array([0, 1]), "image": np.zeros((2, 3, 48, 48), np.float16),
               "mask": np.zeros((2, 3, 48, 48), bool), "classmap": np.zeros((2, 3, 48, 48), np.int8)}
    out = write_table(tmp_path / "v1", rows, patches, {"version": "v1"})
    assert (out / "table.csv").exists() and (out / "patches.npz").exists()
    assert json.loads((out / "manifest.json").read_text())["version"] == "v1"
    with pytest.raises(FileExistsError):
        write_table(tmp_path / "v1", rows, patches, {"version": "v1"})
    t, p = load_table(out)
    assert t.rows[1]["lesion_type"] == "nonspecific_wm_lesion" and t.rows[1]["is_3mm"] is False and t.rows[0]["fold"] == 0
    assert isinstance(t.rows[0]["d_interface_mm"], float) and p["image"].shape == (2, 3, 48, 48)
    assert t.subset(np.array([1])).lesion_id.tolist() == [1]
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_table.py -q -p no:cacheprovider`
Expected: ModuleNotFoundError `anatobind.relation`.

- [ ] **Step 3: Implement `anatobind/relation/table.py`** (and an empty `anatobind/relation/__init__.py`)

```python
"""The relation feature table (spec 2026-09-27 §4): one row per lesion, seven fixed host-class slots, the §11 geometry,
candidate flags and the pseudo label C1. Every arm reads this table and nothing else, which is what makes the
comparison fair by construction (P4). Patches live next to it in patches.npz."""
import csv
import json
from pathlib import Path

import numpy as np

from anatobind.eval.geometry import CLASS_NAMES, SLOT_FIELDS

SLOTS = CLASS_NAMES
SLOT_PREFIX = {"white_matter": "wm", "cortex": "cortex", "thalamus": "thalamus", "basal_ganglia": "bg",
               "brainstem": "brainstem", "cerebellum": "cerebellum", "other_deep_grey": "odg"}
N_SLOTS = len(SLOTS)
NONE_SLOT = N_SLOTS
N_OUT = N_SLOTS + 1
PAIR_FIELDS = ("dx_mm", "dy_mm", "dz_mm", "centroid_distance_mm", "signed_surface_mm", "min_surface_mm", "ioa", "soft_overlap")
LESION_GEOMETRY = ("extent_x_mm", "extent_y_mm", "extent_z_mm", "volume_mm3", "spacing_col_mm", "spacing_row_mm",
                   "spacing_slice_mm", "slice_thickness_mm")
BRAIN_EXTRA = ("d_interface_mm", "delta_d_mm", "dist_cortex_mm", "dist_ventricle_mm")
SIDES = ("image_left", "image_right", "midline")
LESION_TYPES = ("nonspecific_wm_lesion", "lacunar_infarct")
LESION_TYPE_OF_LABEL = {"Nonspecific white matter lesion": "nonspecific_wm_lesion", "Lacunar infarct": "lacunar_infarct"}
ROW_THIRDS = ("top", "middle", "bottom")
SLICE_THIRDS = ("inferior", "middle", "superior")
PATCH_PX, PATCH_MM, PIXEL_MM, PATCH_SLICES = 48, 36.0, 0.75, 3
TABLE_VERSION = "v1"

_INT = ("lesion_id", "fold", "n_slices", "x0", "y0", "x1", "y1", "z0", "z1", "c1_slot")
_FLOAT = ("inplane_mm", "spacing_col_mm", "spacing_row_mm", "spacing_slice_mm", "slice_thickness_mm", "extent_x_mm",
          "extent_y_mm", "extent_z_mm", "volume_mm3", "centroid_col", "centroid_row", "centroid_slice", "d1_mm",
          "d_interface_mm", "delta_d_mm", "dist_cortex_mm", "dist_ventricle_mm", "c1_overlap")
_BOOL = ("is_3mm",)
_STR = ("file", "patient_id", "stratum_geometry", "stratum_series", "band", "lesion_type", "side", "row_third",
        "slice_third", "coarse_location", "c1_class", "c1_source", "registry_lookup_class")
LESION_COLUMNS = ("lesion_id", "file", "patient_id", "fold", "stratum_geometry", "stratum_series", "band", "is_3mm",
                  "lesion_type", "n_slices", "inplane_mm", "x0", "y0", "x1", "y1", "z0", "z1", "spacing_col_mm",
                  "spacing_row_mm", "spacing_slice_mm", "slice_thickness_mm", "extent_x_mm", "extent_y_mm", "extent_z_mm",
                  "volume_mm3", "centroid_col", "centroid_row", "centroid_slice", "side", "row_third", "slice_third",
                  "coarse_location", "d1_mm", "d_interface_mm", "delta_d_mm", "dist_cortex_mm", "dist_ventricle_mm",
                  "c1_class", "c1_slot", "c1_source", "c1_overlap", "registry_lookup_class")


def slot_col(slot_name, field):
    return f"{SLOT_PREFIX[slot_name]}_{field}"


SLOT_COLUMNS = [slot_col(s, f) for s in SLOTS for f in SLOT_FIELDS]
COLUMN_TYPES = {**{c: int for c in _INT}, **{c: float for c in _FLOAT}, **{c: bool for c in _BOOL}, **{c: str for c in _STR}}
for _s in SLOTS:
    for _f in SLOT_FIELDS:
        COLUMN_TYPES[slot_col(_s, _f)] = bool if _f in ("in_volume", "candidate") else float
ALL_COLUMNS = list(LESION_COLUMNS) + SLOT_COLUMNS
assert set(ALL_COLUMNS) == set(COLUMN_TYPES)


def _parse(value, typ):
    if typ is bool:
        return value in ("True", "true", "1")
    return typ(value)


def read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return [{c: _parse(r[c], COLUMN_TYPES[c]) for c in ALL_COLUMNS} for r in csv.DictReader(fh)]


class Table:
    def __init__(self, rows):
        self.rows = list(rows)
        self.lesion_id = np.array([r["lesion_id"] for r in self.rows], dtype=np.int64)
        if len(set(self.lesion_id.tolist())) != len(self.rows):
            raise ValueError("duplicate lesion_id in table")
        self.index = {int(l): i for i, l in enumerate(self.lesion_id)}

    def __len__(self):
        return len(self.rows)

    def column(self, name):
        return np.array([r[name] for r in self.rows])

    def slot_matrix(self, field):
        return np.array([[r[slot_col(s, field)] for s in SLOTS] for r in self.rows])

    def candidates(self):
        return self.slot_matrix("candidate").astype(bool)

    def c1_slot(self):
        return self.column("c1_slot").astype(int)

    def folds(self):
        return self.column("fold").astype(int)

    def patients(self):
        return self.column("patient_id")

    def slot_block(self):
        cand = self.slot_matrix("candidate").astype(np.float32)[:, :, None]
        pair = np.stack([self.slot_matrix(f) for f in PAIR_FIELDS], -1).astype(np.float32)
        return np.concatenate([cand, pair], -1)

    def lesion_block(self):
        geo = np.array([[r[c] for c in LESION_GEOMETRY + BRAIN_EXTRA] for r in self.rows], np.float32).reshape(len(self), 12)
        side = np.array([[r["side"] == s for s in SIDES] for r in self.rows], np.float32)
        typ = np.array([[r["lesion_type"] == t for t in LESION_TYPES] for r in self.rows], np.float32)
        return np.concatenate([geo, side, typ], 1)

    def subset(self, idx):
        return Table([self.rows[i] for i in np.asarray(idx)])


def features_flat(table):
    n = len(table)
    return np.concatenate([table.slot_block().reshape(n, N_SLOTS * 9), table.lesion_block()], 1)


def features_per_slot(table):
    lb = table.lesion_block()
    return np.concatenate([table.slot_block(), np.repeat(lb[:, None, :], N_SLOTS, 1)], -1)


def mask_to_candidates(probs, candidates):
    """Zero every non-candidate slot (spec P7); the none column is kept; rows are renormalised."""
    p = np.array(probs, dtype=np.float64)
    keep = np.concatenate([np.asarray(candidates, bool), np.ones((len(p), 1), bool)], 1)
    p[~keep] = 0.0
    s = p.sum(1, keepdims=True)
    return p / np.where(s > 0, s, 1.0)


def write_table(out_dir, rows, patches, manifest):
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; a relation table is never overwritten (spec §4.6)")
    out.mkdir(parents=True)
    with open(out / "table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / "patches.npz", **patches)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


def load_table(dir, with_patches=True):
    d = Path(dir)
    table = Table(read_rows(d / "table.csv"))
    patches = None
    if with_patches:
        with np.load(d / "patches.npz") as z:
            patches = {k: z[k] for k in z.files}
        if patches["lesion_id"].tolist() != table.lesion_id.tolist():
            raise ValueError("patches.npz and table.csv disagree on lesion order")
    return table, patches
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_table.py -q -p no:cacheprovider`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add anatobind/relation/__init__.py anatobind/relation/table.py tests/test_relation_table.py
git commit -m "Relation table: column schema, Table with slot and lesion feature blocks, candidate masking, write and load"
```

---

### Task 4: Image patches (`anatobind/relation/build.py`, part 1)

**Files:**
- Create: `anatobind/relation/build.py`
- Test: `tests/test_relation_patches.py`

**Interfaces:**
- Consumes: `PATCH_PX`, `PATCH_MM`, `PIXEL_MM`, `PATCH_SLICES` (Task 3); `lesion_centroid` (Task 1).
- Produces: `zscore_volume(rss, seg) -> float32 (col, row, slice)`; `lesion_patch(zvol, classmap, rects, spacing, centroid) -> (image float16, mask bool, classmap int8)`, each `(PATCH_SLICES, PATCH_PX, PATCH_PX)` with axes **(slice, col, row)**: axis 1 is the column axis, which is what the flip augmentation of Task 10 flips.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relation_patches.py
import numpy as np
import pytest

from anatobind.eval.geometry import host_class_map, lesion_centroid
from anatobind.relation.build import lesion_patch, zscore_volume
from anatobind.relation.table import PATCH_PX, PATCH_SLICES


def _vol():
    seg = np.zeros((64, 64, 4), np.int16)
    seg[8:56, 8:56, :] = 2                                  # WM
    seg[40:56, 8:56, :] = 3                                 # cortex at cols 40..56
    rss = np.zeros((64, 64, 4), np.float32)
    rss[8:56, 8:56, :] = np.arange(8, 56, dtype=np.float32)[:, None, None]      # intensity = column index inside the brain
    return seg, rss


def test_zscore_uses_brain_voxels_only():
    seg, rss = _vol()
    z = zscore_volume(rss, seg)
    inside = z[seg > 0]
    assert abs(inside.mean()) < 1e-5 and abs(inside.std() - 1) < 1e-5 and z.dtype == np.float32
    with pytest.raises(ValueError):
        zscore_volume(np.ones((4, 4, 2), np.float32), np.ones((4, 4, 2), np.int16))     # constant brain


def test_patch_is_centred_resampled_and_edge_slices_are_replicated():
    seg, rss = _vol()
    z, cm = zscore_volume(rss, seg), host_class_map(seg)
    rects = [(30, 34, 30, 34, 0)]                           # voxels cols 30..33 rows 30..33 on slice 0; centroid (31.5, 31.5, 0)
    img, msk, cmp = lesion_patch(z, cm, rects, (0.5, 0.5, 5.0), lesion_centroid(rects))
    assert img.shape == msk.shape == cmp.shape == (PATCH_SLICES, PATCH_PX, PATCH_PX)
    assert img.dtype == np.float16 and msk.dtype == bool and cmp.dtype == np.int8
    assert np.array_equal(img[0], img[1]) and np.array_equal(cmp[0], cmp[1])           # slice -1 is clipped to slice 0
    c = PATCH_PX // 2                                       # pixel 24 sits 0.375 mm right of the centroid: col 32.25
    assert msk[1, c, c] and cmp[1, c, c] == 1               # inside the lesion, in WM (class 1)
    assert not msk[1, c + 3, c]                             # 2.625 mm = 5.25 cols away: col 36.75, outside the 30..34 box
    assert float(img[1, c + 8, c]) > float(img[1, c, c]) > float(img[1, c - 8, c])    # intensity ramps with the column
    assert cmp[1, c + 10, c] == 2 and cmp[1, c - 10, c] == 1                            # 7.875 mm right: col 47.25 = cortex; left: col 17.25 = WM
    # col 62.25 is outside the brain but inside the array: the class map is 0 and the image keeps the z-scored background
    assert cmp[1, c + 20, c] == 0 and float(img[1, c + 20, c]) == pytest.approx(float(z[62, 32, 0]), abs=1e-2)
    far = lesion_patch(z, cm, [(60, 64, 30, 34, 0)], (0.5, 0.5, 5.0), lesion_centroid([(60, 64, 30, 34, 0)]))[0]
    assert float(far[1, PATCH_PX - 1, c]) == 0.0                                         # beyond the array edge: zero-filled
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_patches.py -q -p no:cacheprovider`
Expected: ModuleNotFoundError `anatobind.relation.build`.

- [ ] **Step 3: Implement `anatobind/relation/build.py` (part 1)**

```python
"""Build the relation feature table (spec 2026-09-27 §4): per-lesion rows from SynthSeg geometry, image patches from
the RSS, the class-level pseudo label C1, the §4.7 checks and the manifest. Volumes come through an injected loader so
tests run on synthetic arrays."""
import numpy as np
from scipy import ndimage

from anatobind.relation.table import PATCH_PX, PATCH_SLICES, PIXEL_MM


def zscore_volume(rss, seg):
    """FLAIR z-scored over the brain voxels (seg > 0) of its own volume (spec §4.4)."""
    v = np.asarray(rss, np.float32)
    brain = np.asarray(seg) > 0
    mu, sd = float(v[brain].mean()), float(v[brain].std())
    if not sd > 0:
        raise ValueError("brain intensities are constant; refusing to z-score")
    return ((v - mu) / sd).astype(np.float32)


def lesion_patch(zvol, classmap, rects, spacing, centroid):
    """(image, mask, classmap) patches of PATCH_PX x PATCH_PX at PIXEL_MM centred on the lesion centroid, PATCH_SLICES
    slices around the centre slice with edge slices replicated. Axes: (slice, col, row). Image is bilinear,
    mask and classmap nearest-neighbour; outside the volume is zero."""
    nc, nr, ns = zvol.shape
    zc = int(np.clip(np.rint(centroid[2]), 0, ns - 1))
    u = (np.arange(PATCH_PX) - (PATCH_PX - 1) / 2) * PIXEL_MM
    cc, rr = np.meshgrid(centroid[0] + u / spacing[0], centroid[1] + u / spacing[1], indexing="ij")
    coords = np.stack([cc, rr])
    img = np.zeros((PATCH_SLICES, PATCH_PX, PATCH_PX), np.float32)
    msk = np.zeros((PATCH_SLICES, PATCH_PX, PATCH_PX), bool)
    cmp = np.zeros((PATCH_SLICES, PATCH_PX, PATCH_PX), np.int8)
    half = PATCH_SLICES // 2
    for k, dz in enumerate(range(-half, half + 1)):
        z = int(np.clip(zc + dz, 0, ns - 1))
        img[k] = ndimage.map_coordinates(zvol[:, :, z], coords, order=1, mode="constant", cval=0.0)
        plane = np.zeros((nc, nr), np.uint8)
        for c0, c1, r0, r1, s in rects:
            if s == z:
                plane[c0:c1, r0:r1] = 1
        msk[k] = ndimage.map_coordinates(plane, coords, order=0, mode="constant", cval=0) > 0
        cmp[k] = ndimage.map_coordinates(classmap[:, :, z].astype(np.int8), coords, order=0, mode="constant", cval=0)
    return img.astype(np.float16), msk, cmp
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_patches.py -q -p no:cacheprovider`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add anatobind/relation/build.py tests/test_relation_patches.py
git commit -m "Relation build: per-volume z-scoring and lesion-centred image, mask and class-map patches with edge-slice replication"
```

---

### Task 5: Rows, checks, manifest, build script and the synthetic helper (`build.py` part 2)

**Files:**
- Modify: `anatobind/relation/build.py` (append)
- Create: `scripts/build_relation_table.py`, `tests/synth_relation.py`
- Test: `tests/test_relation_build.py`

**Interfaces:**
- Consumes: Tasks 1–4; `anatobind.level_r.registry.load_registry / is_3mm / distance_band`; `anatobind.level_r.export.small_lesion_rows / merged_lesions / match_registry`; `anatobind.level_r.store.now_iso`; `anatobind.level_r.admin.sha256_file`.
- Produces:
  - `LABEL_TO_CLASS: {aseg label: class name}`; `class BuildError(ValueError)`
  - `volume_context(seg, spacing) -> dict(seg, cm, maps, spacing, families, vent, brain_slices)`
  - `lesion_row(reg, members, ctx, fold) -> dict` (all `LESION_COLUMNS` + `SLOT_COLUMNS`)
  - `check_rows(rows, registry_rows, patient_fold, patches, min_agreement=MIN_C1_AGREEMENT) -> report dict` (raises `BuildError`)
  - `build_table(registry_rows, patient_fold, load_volume, out_dir, sources, log=print, min_agreement=MIN_C1_AGREEMENT) -> (rows, patches, manifest)` where `load_volume(file) -> (seg int16 (col,row,slice), spacing (col,row,slice) mm, rss float32 (col,row,slice), members_of {lesion_id: [member dict]})`; `patches` also carries a 0-d string `meta` (window, pixel, slices, axes)
  - `tests/synth_relation.py`: `synthetic_volume(seed) -> (seg, spacing, rss)`, `synthetic_registry(files, per_file, seed) -> (registry_rows, members_of_by_file)`, `synthetic_table(n_patients, per_patient, seed) -> (Table, patches)`

- [ ] **Step 1: Write `tests/synth_relation.py`**

```python
"""Synthetic brains, registries and tables for the relation tests (nothing here reads /data2)."""
import numpy as np

from anatobind.eval.geometry import (
    CLASS_NAMES, SLOT_FIELDS, class_maps, host_class_map, interface_margin, lesion_class_distances, member_rects,
)
from anatobind.eval.lookup import BRAIN_PARENCHYMA, BrainLookup
from anatobind.level_r.registry import distance_band
from anatobind.relation.table import (
    LESION_TYPES, PATCH_PX, PATCH_SLICES, SIDES, SLOTS, Table, slot_col,
)

SPACING = (0.5, 0.5, 5.0)


def synthetic_volume(seed=0, shape=(64, 64, 4)):
    """Brain block cols 8..56 rows 8..56: left WM (2) cols 8..32, right WM (41) cols 32..56, cortex (3 / 42) rows 48..56,
    a lateral ventricle (4) cols 20..24 rows 20..24, a thalamus (10) cols 28..32 rows 28..32. RSS = noise + brain offset."""
    rng = np.random.default_rng(seed)
    seg = np.zeros(shape, np.int16)
    seg[8:32, 8:56, :] = 2
    seg[32:56, 8:56, :] = 41
    seg[8:32, 48:56, :] = 3
    seg[32:56, 48:56, :] = 42
    seg[20:24, 20:24, :] = 4
    seg[28:32, 28:32, :] = 10
    rss = rng.normal(0, 1, shape).astype(np.float32)
    rss[seg > 0] += 10.0
    rss[np.isin(seg, (3, 42))] += 3.0
    return seg, SPACING, rss


def synthetic_registry(files, per_file, seed=0):
    """Single-slice 4 x 4 boxes inside the brain, registry rows shaped like Gate 0.5's lesions.csv (after load_registry),
    and the member boxes per lesion. Registry geometry is computed with the same functions the builder uses."""
    rng = np.random.default_rng(seed)
    rows, members_of = [], {}
    lid = 0
    for fi, f in enumerate(files):
        seg, sp, _ = synthetic_volume(seed)
        cm = host_class_map(seg)
        maps = class_maps(cm, sp)
        look = BrainLookup(seg, sp, BRAIN_PARENCHYMA)
        members_of[f] = {}
        for _ in range(per_file):
            x, y, s = int(rng.integers(10, 50)), int(rng.integers(10, 50)), int(rng.integers(0, seg.shape[2]))
            member = {"x": x, "width": 4, "y": y, "height": 4, "slice": s}
            rects = member_rects([member], seg.shape)
            _, d1, d2 = interface_margin(lesion_class_distances({c: maps[c][0] for c in maps}, rects))
            rows.append({"lesion_id": lid, "file": f, "patient_id": f"patient_{fi}", "series": "200", "stratum_series": "200_201",
                         "stratum_geometry": "inplane_0.69_slice_5", "label": "Nonspecific white matter lesion" if lid % 5 else "Lacunar infarct",
                         "z0": s, "z1": s, "n_slices": 1, "x0": x, "y0": y, "x1": x + 4, "y1": y + 4, "inplane_mm": 2.0,
                         "spacing_row_mm": sp[1], "spacing_col_mm": sp[0], "spacing_slice_mm": sp[2],
                         "host_lookup_all": str(look.host(rects)[0]), "host_lookup_parenchyma": str(look.host(rects)[0]),
                         "host_class_nearest": "white_matter", "d1_mm": float(d1), "d_interface_mm": float(d2),
                         "delta_d_mm": float(d2 - d1), "status": "ok", "band": distance_band(float(d2))})
            members_of[f][lid] = [member]
            lid += 1
    return rows, members_of


def synthetic_table(n_patients=6, per_patient=4, seed=0):
    """A Table with random but self-consistent geometry: c1 is the argmax-ioa slot, candidates are the slots within 15 mm,
    patches are random. Enough structure for Bgeo+ to fit C1 and for the CV / metrics / training tests."""
    rng = np.random.default_rng(seed)
    rows = []
    lid = 0
    for p in range(n_patients):
        for _ in range(per_patient):
            ioa = np.zeros(7)
            k = int(rng.integers(0, 2))                                    # WM or cortex carries the overlap
            ioa[k] = float(rng.uniform(0.5, 1.0))
            ioa[1 - k] = 1.0 - ioa[k] if rng.uniform() < 0.5 else 0.0
            dist = np.where(ioa > 0, 0.0, rng.uniform(1.0, 40.0, 7))
            dist[2] = min(dist[2], 10.0)                                   # thalamus is always a candidate
            side = SIDES[int(rng.integers(0, 3))]
            ltype = LESION_TYPES[int(rng.integers(0, 2))]
            d_int = float(rng.uniform(0.0, 8.0))
            row = {"lesion_id": lid, "file": f"file_{p}", "patient_id": f"patient_{p}", "fold": p % 5,
                   "stratum_geometry": "inplane_0.69_slice_5", "stratum_series": "200_201", "band": distance_band(d_int),
                   "is_3mm": False, "lesion_type": ltype, "n_slices": 1, "inplane_mm": 3.0, "x0": 10, "y0": 10, "x1": 14, "y1": 14,
                   "z0": 1, "z1": 1, "spacing_col_mm": 0.6875, "spacing_row_mm": 0.6875, "spacing_slice_mm": 5.0,
                   "slice_thickness_mm": 5.0, "extent_x_mm": 2.75, "extent_y_mm": 2.75, "extent_z_mm": 5.0, "volume_mm3": 37.8,
                   "centroid_col": 11.5, "centroid_row": 11.5, "centroid_slice": 1.0, "side": side, "row_third": "middle",
                   "slice_third": "middle", "coarse_location": f"{side}|middle|middle", "d1_mm": 0.0, "d_interface_mm": d_int,
                   "delta_d_mm": d_int, "dist_cortex_mm": float(dist[1]), "dist_ventricle_mm": float(rng.uniform(0, 30)),
                   "c1_class": SLOTS[int(np.argmax(ioa))], "c1_slot": int(np.argmax(ioa)), "c1_source": "overlap",
                   "c1_overlap": float(ioa.max()), "registry_lookup_class": SLOTS[int(np.argmax(ioa))]}
            for i, s in enumerate(SLOTS):
                vals = {"in_volume": True, "candidate": bool(dist[i] <= 15.0), "dx_mm": float(rng.normal(0, 5)),
                        "dy_mm": float(rng.normal(0, 5)), "dz_mm": 0.0, "centroid_distance_mm": float(dist[i] + 1.0),
                        "min_surface_mm": float(dist[i]), "signed_surface_mm": float(-2.0 if ioa[i] > 0 else dist[i]),
                        "ioa": float(ioa[i]), "soft_overlap": float(np.exp(-dist[i]))}
                for f in SLOT_FIELDS:
                    row[slot_col(s, f)] = vals[f]
            rows.append(row)
            lid += 1
    n = len(rows)
    patches = {"lesion_id": np.arange(n), "image": rng.normal(0, 1, (n, PATCH_SLICES, PATCH_PX, PATCH_PX)).astype(np.float16),
               "mask": np.zeros((n, PATCH_SLICES, PATCH_PX, PATCH_PX), bool), "classmap": np.zeros((n, PATCH_SLICES, PATCH_PX, PATCH_PX), np.int8)}
    patches["mask"][:, :, 20:28, 20:28] = True
    return Table(rows), patches
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_relation_build.py
import json

import numpy as np
import pytest

from anatobind.eval.geometry import CLASS_NAMES
from anatobind.relation.build import BuildError, build_table, check_rows, lesion_row, volume_context
from anatobind.relation.table import ALL_COLUMNS, load_table
from synth_relation import synthetic_registry, synthetic_volume

FILES = ["file_brain_AXFLAIR_200_1", "file_brain_AXFLAIR_200_2", "file_brain_AXFLAIR_200_3"]


def _loader(members_of):
    def load_volume(f):
        seg, sp, rss = synthetic_volume(0)
        return seg, sp, rss, members_of[f]
    return load_volume


def _build(tmp_path, per_file=3, mutate=None):
    reg, members = synthetic_registry(FILES, per_file)
    if mutate:
        mutate(reg)
    folds = {f"patient_{i}": i for i in range(len(FILES))}
    return build_table(reg, folds, _loader(members), tmp_path / "v1", {"registry_sha256": "x", "folds_sha256": "y"}, log=lambda *_: None)


def test_lesion_row_has_every_column_and_matches_the_registry_geometry():
    reg, members = synthetic_registry(FILES[:1], 2)
    seg, sp, _ = synthetic_volume(0)
    ctx = volume_context(seg, sp)
    row = lesion_row(reg[0], members[FILES[0]][0], ctx, 0)
    assert set(row) == set(ALL_COLUMNS)
    assert row["d_interface_mm"] == pytest.approx(reg[0]["d_interface_mm"]) and row["d1_mm"] == pytest.approx(reg[0]["d1_mm"])
    assert row["c1_class"] in CLASS_NAMES and row["side"] in ("image_left", "image_right", "midline")
    assert row["coarse_location"] == f"{row['side']}|{row['row_third']}|{row['slice_third']}"
    assert row["volume_mm3"] == pytest.approx(16 * 0.5 * 0.5 * 5.0) and row["extent_z_mm"] == 5.0
    assert row["lesion_type"] == "lacunar_infarct"                                  # lesion 0: lid % 5 == 0


def test_build_writes_table_patches_and_manifest_with_the_checks(tmp_path):
    rows, patches, manifest = _build(tmp_path)
    t, p = load_table(tmp_path / "v1")
    assert len(t) == 9 and p["image"].shape == (9, 3, 48, 48) and t.lesion_id.tolist() == list(range(9))
    m = json.loads((tmp_path / "v1" / "manifest.json").read_text())
    assert m["n_lesions"] == 9 and m["n_patients"] == 3 and m["c1_agreement"]["fraction"] == 1.0
    assert m["per_fold"]["0"]["lesions"] == 3 and m["registry_sha256"] == "x" and m["parameters"]["candidate_mm"] == 15.0
    assert sum(m["c1_distribution"].values()) == 9


def test_build_refuses_a_patient_missing_from_the_fold_table(tmp_path):
    reg, members = synthetic_registry(FILES, 2)
    with pytest.raises(BuildError, match="fold"):
        build_table(reg, {"patient_0": 0}, _loader(members), tmp_path / "v1", {}, log=lambda *_: None)
    assert not (tmp_path / "v1").exists()


def test_build_refuses_when_c1_disagrees_with_the_registry_too_often(tmp_path):
    def corrupt(reg):
        for r in reg:
            r["host_lookup_parenchyma"] = "16"                                     # brainstem for everyone
    with pytest.raises(BuildError, match="C1"):
        _build(tmp_path, mutate=corrupt)


def test_build_refuses_when_a_registry_distance_differs(tmp_path):
    def corrupt(reg):
        reg[0]["d_interface_mm"] += 0.5
    with pytest.raises(BuildError, match="d_interface"):
        _build(tmp_path, mutate=corrupt)


def test_check_rows_lists_mismatches_below_the_threshold(tmp_path):
    rows, patches, manifest = _build(tmp_path)
    rows[0]["registry_lookup_class"] = "brainstem"
    report = check_rows(rows, [], None, patches, min_agreement=0.5)
    assert report["c1_agreement"]["mismatches"] == [0] and report["c1_agreement"]["fraction"] == pytest.approx(8 / 9)


def _break(case, rows, patches, reg):
    if case == "duplicate":
        rows[1]["lesion_id"] = rows[0]["lesion_id"]
    elif case == "registry_set":
        reg.append({**reg[0], "lesion_id": 999})
    elif case == "two_folds":
        rows[1]["patient_id"], rows[1]["fold"] = rows[0]["patient_id"], rows[0]["fold"] + 1
    elif case == "no_candidate":
        for k in rows[0]:
            if k.endswith("_candidate"):
                rows[0][k] = False
    elif case == "constant_patch":
        patches["image"][0] = 0
    return rows, patches, reg


@pytest.mark.parametrize("case,match", [("duplicate", "duplicate"), ("registry_set", "registry"), ("two_folds", "fold"),
                                        ("no_candidate", "candidate"), ("constant_patch", "constant")])
def test_check_rows_refuses_each_broken_invariant(tmp_path, case, match):
    rows, patches, _ = _build(tmp_path)
    reg, _ = synthetic_registry(FILES, 3)
    rows, patches, reg = _break(case, rows, dict(patches, image=patches["image"].copy()), reg)
    with pytest.raises(BuildError, match=match):
        check_rows(rows, reg if case == "registry_set" else [], None, patches, min_agreement=0.0)
```

- [ ] **Step 3: Run to verify failure**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_build.py -q -p no:cacheprovider`
Expected: ImportError `BuildError`.

- [ ] **Step 4: Implement (append to `anatobind/relation/build.py`; move the new imports up into the module's import block next to part 1's)**

```python
from anatobind.eval.geometry import (
    CANDIDATE_MM, CLASS_NAMES, DIST_CAP_MM, HOST_CLASSES, SLOT_FIELDS, SOFT_SIGMA_MM, class_maps, family_sides,
    host_class_map, interface_margin, landmark_map, lesion_centroid, lesion_class_distances, member_rects,
    min_in_lesion, side_of, slot_features, third,
)
import json

from anatobind.eval.lookup import class_level_host
from anatobind.level_r.registry import is_3mm
from anatobind.level_r.store import now_iso
from anatobind.relation.table import (
    LESION_TYPE_OF_LABEL, PATCH_MM, PATCH_SLICES, ROW_THIRDS, SLICE_THIRDS, SLOTS, TABLE_VERSION, slot_col, write_table,
)

LABEL_TO_CLASS = {label: name for name, labels in HOST_CLASSES.items() for label in labels}
MIN_C1_AGREEMENT = 0.99
TOL_MM = 1e-6


class BuildError(ValueError):
    pass


def volume_context(seg, spacing):
    cm = host_class_map(seg)
    slices = np.nonzero((seg > 0).any((0, 1)))[0]
    return {"seg": seg, "cm": cm, "maps": class_maps(cm, spacing), "spacing": tuple(float(s) for s in spacing),
            "families": family_sides(seg), "vent": landmark_map(seg, spacing),
            "brain_slices": (int(slices.min()), int(slices.max())) if slices.size else (0, seg.shape[2] - 1)}


def lesion_row(reg, members, ctx, fold):
    seg, cm, maps, sp = ctx["seg"], ctx["cm"], ctx["maps"], ctx["spacing"]
    rects = member_rects(members, seg.shape)
    if not rects:
        raise BuildError(f"lesion {reg['lesion_id']}: no voxel on the grid")
    slots = slot_features(cm, maps, rects, sp)
    c1, source, frac = class_level_host(slots)
    _, d1, d2 = interface_margin(lesion_class_distances({c: maps[c][0] for c in maps}, rects))
    centroid = lesion_centroid(rects)
    zc = int(np.clip(np.rint(centroid[2]), 0, seg.shape[2] - 1))
    brain_rows = np.nonzero((seg[:, :, zc] > 0).any(0))[0]
    row_third = third(int(np.rint(centroid[1])), int(brain_rows.min()), int(brain_rows.max()), ROW_THIRDS) if brain_rows.size else ROW_THIRDS[1]
    slice_third = third(zc, ctx["brain_slices"][0], ctx["brain_slices"][1], SLICE_THIRDS)
    side = side_of(seg, rects, ctx["families"])
    volume = sum((b - a) * (d - c) for a, b, c, d, _ in rects) * sp[0] * sp[1] * sp[2]
    row = {"lesion_id": reg["lesion_id"], "file": reg["file"], "patient_id": reg["patient_id"], "fold": fold,
           "stratum_geometry": reg["stratum_geometry"], "stratum_series": reg["stratum_series"], "band": reg["band"],
           "is_3mm": is_3mm(reg["stratum_geometry"]), "lesion_type": LESION_TYPE_OF_LABEL[reg["label"]],
           "n_slices": reg["n_slices"], "inplane_mm": reg["inplane_mm"], "x0": reg["x0"], "y0": reg["y0"], "x1": reg["x1"],
           "y1": reg["y1"], "z0": reg["z0"], "z1": reg["z1"], "spacing_col_mm": sp[0], "spacing_row_mm": sp[1],
           "spacing_slice_mm": sp[2], "slice_thickness_mm": sp[2], "extent_x_mm": (reg["x1"] - reg["x0"]) * sp[0],
           "extent_y_mm": (reg["y1"] - reg["y0"]) * sp[1], "extent_z_mm": (reg["z1"] - reg["z0"] + 1) * sp[2],
           "volume_mm3": float(volume), "centroid_col": float(centroid[0]), "centroid_row": float(centroid[1]),
           "centroid_slice": float(centroid[2]), "side": side, "row_third": row_third, "slice_third": slice_third,
           "coarse_location": f"{side}|{row_third}|{slice_third}", "d1_mm": float(d1), "d_interface_mm": float(d2),
           "delta_d_mm": float(d2 - d1), "dist_cortex_mm": slots[CLASS_NAMES.index("cortex") + 1]["min_surface_mm"],
           "dist_ventricle_mm": min_in_lesion(ctx["vent"], rects), "c1_class": CLASS_NAMES[c1 - 1], "c1_slot": c1 - 1,
           "c1_source": source, "c1_overlap": float(frac),
           "registry_lookup_class": LABEL_TO_CLASS.get(int(reg["host_lookup_parenchyma"]), "")}
    for i, s in enumerate(SLOTS, start=1):
        for f in SLOT_FIELDS:
            row[slot_col(s, f)] = slots[i][f]
    return row


def check_rows(rows, registry_rows, patient_fold, patches, min_agreement=MIN_C1_AGREEMENT):
    """The spec §4.7 checks; returns the manifest counts or raises BuildError."""
    ids = [r["lesion_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise BuildError("duplicate lesion_id")
    if registry_rows and set(ids) != {r["lesion_id"] for r in registry_rows}:
        raise BuildError(f"{len(ids)} rows for {len(registry_rows)} registry lesions: the lesion_id sets differ")
    reg_by_id = {r["lesion_id"]: r for r in registry_rows}
    folds_of = {}
    for r in rows:
        folds_of.setdefault(r["patient_id"], set()).add(r["fold"])
        if patient_fold is not None and patient_fold.get(r["patient_id"]) != r["fold"]:
            raise BuildError(f"patient {r['patient_id']} is not in the fold table with fold {r['fold']}")
        reg = reg_by_id.get(r["lesion_id"])
        if reg is not None:
            for k in ("d1_mm", "d_interface_mm", "delta_d_mm"):
                if abs(float(reg[k]) - r[k]) > TOL_MM:
                    raise BuildError(f"lesion {r['lesion_id']}: {k} {r[k]} differs from the registry {reg[k]}")
        if not any(r[slot_col(s, "candidate")] for s in SLOTS):
            raise BuildError(f"lesion {r['lesion_id']}: no candidate")
    if any(len(v) != 1 for v in folds_of.values()):
        raise BuildError("a patient appears in more than one fold (folds are by patient)")
    mismatches = [r["lesion_id"] for r in rows if r["c1_class"] != r["registry_lookup_class"]]
    fraction = 1.0 - len(mismatches) / len(rows)
    if fraction < min_agreement:
        raise BuildError(f"C1 agrees with the registry lookup on {fraction:.4f} < {min_agreement}; mismatches {mismatches[:20]}")
    img = np.asarray(patches["image"], np.float32).reshape(len(rows), -1)
    if (img.std(1) == 0).any():
        raise BuildError("a constant image patch")
    per_fold = {}
    for r in rows:
        d = per_fold.setdefault(str(r["fold"]), {"patients": set(), "lesions": 0})
        d["patients"].add(r["patient_id"])
        d["lesions"] += 1
    counts = {}
    for r in rows:
        counts[r["c1_class"]] = counts.get(r["c1_class"], 0) + 1
    sources = {}
    for r in rows:
        sources[r["c1_source"]] = sources.get(r["c1_source"], 0) + 1
    forced = sum(1 for r in rows if all(r[slot_col(s, "min_surface_mm")] > CANDIDATE_MM for s in SLOTS if r[slot_col(s, "candidate")]))
    return {"n_lesions": len(rows), "n_patients": len(folds_of),
            "per_fold": {k: {"patients": len(v["patients"]), "lesions": v["lesions"]} for k, v in sorted(per_fold.items())},
            "c1_distribution": counts, "c1_source_counts": sources, "n_candidate_fallback": forced,
            "c1_agreement": {"fraction": fraction, "mismatches": mismatches}}


def build_table(registry_rows, patient_fold, load_volume, out_dir, sources, log=print, min_agreement=MIN_C1_AGREEMENT):
    by_file = {}
    for r in registry_rows:
        by_file.setdefault(r["file"], []).append(r)
    rows, images, masks, cmaps, ids = [], [], [], [], []
    for i, f in enumerate(sorted(by_file)):
        seg, spacing, rss, members_of = load_volume(f)
        ctx = volume_context(seg, spacing)
        z = zscore_volume(rss, seg)
        for reg in sorted(by_file[f], key=lambda r: r["lesion_id"]):
            if reg["patient_id"] not in patient_fold:
                raise BuildError(f"patient {reg['patient_id']} is not in the fold table")
            rects = member_rects(members_of[reg["lesion_id"]], seg.shape)
            rows.append(lesion_row(reg, members_of[reg["lesion_id"]], ctx, patient_fold[reg["patient_id"]]))
            img, msk, cmp = lesion_patch(z, ctx["cm"], rects, spacing, lesion_centroid(rects))
            images.append(img)
            masks.append(msk)
            cmaps.append(cmp)
            ids.append(reg["lesion_id"])
        log(f"{i + 1}/{len(by_file)} {f}: {len(by_file[f])} lesions")
    order = np.argsort(ids)
    rows = [rows[k] for k in order]
    patches = {"lesion_id": np.array(ids)[order], "image": np.stack(images)[order], "mask": np.stack(masks)[order],
               "classmap": np.stack(cmaps)[order],
               "meta": np.array(json.dumps({"window_mm": PATCH_MM, "pixel_mm": PIXEL_MM, "slices": "centre-1, centre, centre+1",
                                            "axes": "lesion, slice, col, row", "image": "per-volume z-score over seg > 0"}))}
    report = check_rows(rows, registry_rows, patient_fold, patches, min_agreement)
    manifest = {"version": TABLE_VERSION, "built_at": now_iso(), **sources,
                "parameters": {"candidate_mm": CANDIDATE_MM, "distance_cap_mm": DIST_CAP_MM, "soft_sigma_mm": SOFT_SIGMA_MM,
                               "patch_mm": PATCH_MM, "pixel_mm": PIXEL_MM, "patch_slices": PATCH_SLICES,
                               "min_c1_agreement": min_agreement}, **report}
    write_table(out_dir, rows, patches, manifest)
    return rows, patches, manifest
```

- [ ] **Step 5: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_build.py -q -p no:cacheprovider`
Expected: pass. (`test_build_refuses_when_c1_disagrees…` relies on the synthetic registry putting most lesions in WM/cortex, so "brainstem for everyone" breaks the 0.99 threshold; `check_rows` with `patient_fold=None` skips the fold-table comparison.)

- [ ] **Step 6: Write `scripts/build_relation_table.py`**

```python
#!/usr/bin/env python
# scripts/build_relation_table.py
"""Build the relation feature table (spec 2026-09-27 §4) from the Gate 0.5 registry, the Level R fold table, the
SynthSeg parcellations and the h5 RSS volumes. Refuses an existing output directory.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/build_relation_table.py \
      --out /data2/congcong/data/FM_data/derived/relation/v1 | tee docs/verification/<date>/relation_table_output.txt
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import read_fastmri_plus_rows  # noqa: E402
from anatobind.data_engine.fastmri_knee import volume_geometry  # noqa: E402
from anatobind.level_r.admin import sha256_file  # noqa: E402
from anatobind.level_r.export import match_registry, merged_lesions, small_lesion_rows  # noqa: E402
from anatobind.level_r.registry import REGISTRY, load_registry  # noqa: E402
from anatobind.relation.build import build_table  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
CSV = FM / "fastMRI_lh_brain_knee/Annotations/brain.csv"
KROOT = FM / "fastMRI_lh_brain_knee/kspace/brain"
SEG_ROOT = FM / "derived/synthseg/fastmri_brain/seg_native"
FOLDS = Path("data/level_r/folds.json")


def h5_of(stem):
    for split in ("multicoil_train", "multicoil_val"):
        p = KROOT / split / f"{stem}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def make_loader(registry_by_file, csv_by_file):
    def load_volume(f):
        path = h5_of(f)
        g = volume_geometry(path)
        with h5py.File(path) as h:
            rss = h["reconstruction_rss"][()]
        rss = np.ascontiguousarray(rss.transpose(2, 1, 0)).astype(np.float32)          # (slice, row, col) -> (col, row, slice)
        img = nib.load(str(SEG_ROOT / f"{f}_seg.nii.gz"))
        seg = np.asarray(img.dataobj).astype(np.int16)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        if seg.shape != rss.shape or seg.shape != (g["n_cols"], g["n_rows"], g["slices"]):
            raise ValueError(f"{f}: seg {seg.shape}, rss {rss.shape}, geometry {(g['n_cols'], g['n_rows'], g['slices'])}")
        matched = match_registry(registry_by_file[f], merged_lesions(csv_by_file.get(f, []), g["n_rows"]))
        return seg, zooms, rss, {lid: L["members"] for lid, L in matched.items()}
    return load_volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--folds", type=Path, default=FOLDS)
    ap.add_argument("--limit", type=int, default=0, help="first N files only (smoke runs)")
    ap.add_argument("--min-c1-agreement", type=float, default=0.99,
                    help="spec §4.7 threshold; the real build keeps 0.99, a smoke run on a few files may pass 0")
    a = ap.parse_args()
    registry = load_registry(a.registry)
    folds = json.loads(a.folds.read_text())["patient_fold"]
    registry_by_file, csv_by_file = {}, {}
    for r in registry:
        registry_by_file.setdefault(r["file"], []).append(r)
    for r in small_lesion_rows(read_fastmri_plus_rows(CSV)):
        csv_by_file.setdefault(r["file"], []).append(r)
    if a.limit:
        keep = sorted(registry_by_file)[:a.limit]
        registry = [r for r in registry if r["file"] in keep]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sources = {"git_commit": commit, "registry": str(a.registry), "registry_sha256": sha256_file(a.registry),
               "folds": str(a.folds), "folds_sha256": sha256_file(a.folds), "seg_root": str(SEG_ROOT), "kspace_root": str(KROOT)}
    rows, patches, manifest = build_table(registry, folds, make_loader(registry_by_file, csv_by_file), a.out, sources,
                                          log=lambda m: print(m, flush=True), min_agreement=a.min_c1_agreement)
    print(json.dumps({k: v for k, v in manifest.items() if k != "c1_agreement"}, indent=1))
    print("c1_agreement", manifest["c1_agreement"]["fraction"], "mismatches", manifest["c1_agreement"]["mismatches"])
    print(f"wrote {a.out}: {len(rows)} lesions, patches {patches['image'].shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Smoke the script on two real volumes into the scratch directory, then run the whole suite**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/build_relation_table.py --out runs/relation_smoke_v1 --limit 2 --min-c1-agreement 0`
Expected: prints the manifest for 2 files (a handful of lesions) with its `c1_agreement` fraction and mismatch list, no exception. (`--limit 2` restricts the registry to two files; the fold check passes because every registry patient is in `folds.json`; `--min-c1-agreement 0` because one mismatch among a handful of lesions says nothing about the 0.99 rule, which Task 15 applies to all 1297.) Report the smoke manifest's counts in the task report. Delete nothing afterwards; `runs/` is gitignored.

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add anatobind/relation/build.py scripts/build_relation_table.py tests/synth_relation.py tests/test_relation_build.py
git commit -m "Relation build: per-lesion rows with class-level C1, the spec 4.7 checks, the manifest, the build script and synthetic fixtures"
```

---

### Task 6: Label sources C1 and R (`anatobind/relation/labels.py`)

**Files:**
- Create: `anatobind/relation/labels.py`
- Test: `tests/test_relation_labels.py`

**Interfaces:**
- Consumes: `Table` (Task 3); `anatobind.eval.level_r_labels.load_fold / load_train_labels / load_test_labels / SealedAccessError`.
- Produces:
```python
HOST_OF_READER = {"other": "other_deep_grey"}                  # reader ontology "other" is the seventh slot
def c1_labels(table) -> dict {lesion_id: frozenset({c1_class})}
def from_level_r_rows(rows) -> (labels dict, excluded list)     # not_a_lesion -> excluded; hosts mapped through HOST_OF_READER
def r_train_labels(k, sealed_dir, manifest_path) -> (labels, excluded)
def r_test_labels(sealed_dir, manifest_path, unblind=False) -> (labels, excluded)   # every fold; raises SealedAccessError unless unblind is True
def acceptable_matrix(labels, lesion_ids) -> (has_label bool (N,), acceptable bool (N, 7))
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relation_labels.py
import csv

import numpy as np
import pytest

from anatobind.eval.level_r_labels import SealedAccessError
from anatobind.level_r.admin import FINAL_COLUMNS, seal
from anatobind.relation.labels import acceptable_matrix, c1_labels, from_level_r_rows, r_test_labels, r_train_labels
from synth_relation import synthetic_table


@pytest.fixture
def sealed(tmp_path):
    rows = [{"lesion_id": i, "status": "agreed", "primary_host": "white_matter" if i < 8 else None,
             "acceptable_hosts": '["white_matter", "cortex"]' if i == 1 else '["other"]' if i == 2 else '["white_matter"]' if i < 8 else "[]",
             "not_a_lesion": i >= 8, "lesion_type": None, "side": None, "lobe": None} for i in range(10)]
    final = tmp_path / "final.csv"
    with open(final, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    seal(final, {i: i % 5 for i in range(10)}, tmp_path / "sealed", tmp_path / "manifest.json", k=5, now="2026-10-01T00:00:00+00:00")
    return tmp_path / "sealed", tmp_path / "manifest.json"


def test_c1_labels_are_singletons_from_the_table():
    t, _ = synthetic_table(2, 3)
    lab = c1_labels(t)
    assert len(lab) == 6 and all(len(v) == 1 for v in lab.values()) and lab[0] == frozenset({t.rows[0]["c1_class"]})


def test_level_r_rows_map_other_and_exclude_not_a_lesion(sealed):
    d, m = sealed
    labels, excluded = r_test_labels(d, m, unblind=True)
    assert sorted(excluded) == [8, 9] and labels[1] == frozenset({"white_matter", "cortex"}) and labels[2] == frozenset({"other_deep_grey"})
    train, ex = r_train_labels(0, d, m)
    assert 0 not in train and 5 not in train and 1 in train and ex == [8, 9]      # fold 0 holds lesions 0 and 5


def test_test_labels_need_the_literal_unblind_flag(sealed):
    d, m = sealed
    with pytest.raises(SealedAccessError):
        r_test_labels(d, m)
    with pytest.raises(SealedAccessError):
        r_test_labels(d, m, unblind=1)


def test_acceptable_matrix_marks_slots_and_missing_labels():
    labels = {0: frozenset({"white_matter"}), 2: frozenset({"cortex", "thalamus"})}
    has, acc = acceptable_matrix(labels, np.array([0, 1, 2]))
    assert has.tolist() == [True, False, True] and acc.shape == (3, 7)
    assert acc[0].tolist() == [True] + [False] * 6 and acc[2].tolist() == [False, True, True, False, False, False, False]
    assert not acc[1].any()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_labels.py -q -p no:cacheprovider`
Expected: ModuleNotFoundError `anatobind.relation.labels`.

- [ ] **Step 3: Implement `anatobind/relation/labels.py`**

```python
"""Label sources for the relation baselines (spec 2026-09-27 §6.2): C1 from the table, R from the sealed Level R folds.
Both hand the evaluation the same structure, {lesion_id: frozenset of acceptable slot names}. Readers answer "other";
in the table that is the seventh slot, other_deep_grey."""
import json
from pathlib import Path

import numpy as np

from anatobind.eval.level_r_labels import SealedAccessError, load_test_labels, load_train_labels
from anatobind.relation.table import SLOTS

HOST_OF_READER = {"other": "other_deep_grey"}


def c1_labels(table):
    return {int(r["lesion_id"]): frozenset({r["c1_class"]}) for r in table.rows}


def from_level_r_rows(rows):
    labels, excluded = {}, []
    for r in rows:
        if r["not_a_lesion"]:
            excluded.append(int(r["lesion_id"]))
            continue
        hosts = frozenset(HOST_OF_READER.get(h, h) for h in r["acceptable_hosts"])
        unknown = hosts - set(SLOTS)
        if not hosts or unknown:
            raise ValueError(f"lesion {r['lesion_id']}: acceptable hosts {sorted(hosts)} are not table slots")
        labels[int(r["lesion_id"])] = hosts
    return labels, sorted(excluded)


def r_train_labels(k, sealed_dir, manifest_path):
    return from_level_r_rows(load_train_labels(k, sealed_dir, manifest_path))


def r_test_labels(sealed_dir, manifest_path, unblind=False):
    """Every sealed fold's final labels: the one-shot final evaluation (spec §6.8). unblind must be the literal True."""
    if unblind is not True:
        raise SealedAccessError("test labels open only with unblind=True, from the final evaluation script")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    folds = sorted(int(name[4:]) for name in manifest)
    rows = [r for k in folds for r in load_test_labels(k, unblind=True, sealed_dir=sealed_dir, manifest_path=manifest_path)]
    return from_level_r_rows(rows)


def acceptable_matrix(labels, lesion_ids):
    ids = np.asarray(lesion_ids)
    has = np.array([int(l) in labels for l in ids], bool)
    acc = np.zeros((len(ids), len(SLOTS)), bool)
    for i, l in enumerate(ids):
        for h in labels.get(int(l), ()):
            acc[i, SLOTS.index(h)] = True
    return has, acc
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_labels.py tests/test_level_r_labels.py -q -p no:cacheprovider`
Expected: pass (the sealed loader's access log is written under the tmp sealed dir).

- [ ] **Step 5: Commit**

```bash
git add anatobind/relation/labels.py tests/test_relation_labels.py
git commit -m "Relation labels: C1 from the table and sealed Level R folds as one acceptable-set structure, with the unblind guard"
```

---

### Task 7: Metrics, bootstrap, McNemar, strata, Gate R1 (`anatobind/eval/relation_metrics.py`)

**Files:**
- Create: `anatobind/eval/relation_metrics.py`
- Test: `tests/test_relation_metrics.py`

**Interfaces:**
```python
N_BOOT, SEED, DELTA_D_EDGES = 10_000, 0, (0.0, 1.0, 2.0, 4.0, 8.0, float("inf"))
def predicted_slot(probs) -> (N,) int                      # argmax over the 8 columns; 7 = none
def is_correct(probs, acceptable) -> (N,) bool             # predicted slot < 7 and acceptable[i, slot]
def topk_correct(probs, acceptable, k=2) -> (N,) bool
def summary(probs, acceptable) -> dict                     # n, accuracy, singleton_rate, top2_accuracy, singleton: {n, accuracy, macro_f1, balanced_accuracy, confusion (7 x 8 list)}
def rescue_harm(correct_model, correct_comp) -> dict       # rescue, harm, net (counts) and net_rate
def patient_bootstrap(values, patients, n_boot=N_BOOT, seed=SEED, alpha=0.05) -> (mean, lo, hi)
def mcnemar(b, c) -> float                                 # exact two-sided binomial p-value on the discordant pairs
def by_stratum(values, keys) -> {key: {"n": int, "mean": float}}
def delta_d_curve(values, delta_d, patients, edges=DELTA_D_EDGES, n_boot=N_BOOT, seed=SEED) -> list of {"bin", "n", "mean", "lo", "hi"}
def gate_r1(model_probs, comparators, acceptable, patients, n_boot=N_BOOT, seed=SEED) -> dict
    # comparators = {"bprior": probs, "bgeo": probs, "b2": probs}; returns per-comparator accuracy, net rescue mean/CI vs bgeo,
    # and the four booleans net_rescue_ci_low_gt_0, gt_bprior, gt_bgeo, gt_b2 plus "go" (all four)
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relation_metrics.py
import numpy as np
import pytest

from anatobind.eval.relation_metrics import (
    by_stratum, delta_d_curve, gate_r1, is_correct, mcnemar, patient_bootstrap, predicted_slot, rescue_harm, summary,
    topk_correct,
)


def _probs(slots):
    p = np.zeros((len(slots), 8))
    for i, s in enumerate(slots):
        p[i, s] = 1.0
    return p


def _acc(sets):
    a = np.zeros((len(sets), 7), bool)
    for i, s in enumerate(sets):
        a[i, list(s)] = True
    return a


def test_set_valued_correctness_and_none_is_never_correct():
    probs = _probs([0, 1, 7, 1])
    acc = _acc([{0}, {0, 1}, {0}, {0}])
    assert predicted_slot(probs).tolist() == [0, 1, 7, 1]
    assert is_correct(probs, acc).tolist() == [True, True, False, False]
    p = np.array([[0.5, 0.4, 0.1, 0, 0, 0, 0, 0]])
    assert topk_correct(p, _acc([{1}]), k=2).tolist() == [True] and topk_correct(p, _acc([{2}]), k=2).tolist() == [False]


def test_summary_reports_singleton_subset_metrics():
    probs = _probs([0, 1, 0, 1])
    acc = _acc([{0}, {1}, {0, 1}, {0}])
    s = summary(probs, acc)
    assert s["n"] == 4 and s["accuracy"] == 0.75 and s["singleton_rate"] == 0.75
    assert s["singleton"]["n"] == 3 and s["singleton"]["accuracy"] == pytest.approx(2 / 3)
    # singleton truth [0, 1, 0], predictions [0, 1, 1]: WM recall 1/2 precision 1, cortex recall 1 precision 1/2
    assert s["singleton"]["balanced_accuracy"] == pytest.approx(0.75)
    assert s["singleton"]["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert np.array(s["singleton"]["confusion"]).shape == (7, 8) and s["singleton"]["confusion"][0][1] == 1


def test_rescue_harm_and_mcnemar():
    model = np.array([True, True, False, True, False])
    comp = np.array([False, True, True, True, False])
    r = rescue_harm(model, comp)
    assert (r["rescue"], r["harm"], r["net"], r["net_rate"]) == (1, 1, 0, 0.0)
    assert mcnemar(1, 1) == 1.0 and mcnemar(0, 0) == 1.0 and mcnemar(10, 0) == pytest.approx(2 * 0.5 ** 10)


def test_patient_bootstrap_is_deterministic_and_brackets_the_mean():
    values = np.array([1, 1, 0, 0, 1, 0, 1, 1], float)
    patients = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    m1, lo1, hi1 = patient_bootstrap(values, patients, n_boot=500, seed=0)
    m2, lo2, hi2 = patient_bootstrap(values, patients, n_boot=500, seed=0)
    assert (m1, lo1, hi1) == (m2, lo2, hi2) and m1 == 0.625 and lo1 <= m1 <= hi1
    assert patient_bootstrap(np.ones(4), np.array(["a", "a", "b", "b"]), n_boot=50) == (1.0, 1.0, 1.0)


def test_strata_and_delta_d_curve():
    vals = np.array([1.0, 0.0, 1.0, 1.0])
    assert by_stratum(vals, np.array(["x", "x", "y", "y"])) == {"x": {"n": 2, "mean": 0.5}, "y": {"n": 2, "mean": 1.0}}
    curve = delta_d_curve(vals, np.array([0.0, 0.5, 3.0, 20.0]), np.array(["a", "b", "c", "d"]), n_boot=50)
    assert [c["bin"] for c in curve] == ["[0.0, 1.0)", "[1.0, 2.0)", "[2.0, 4.0)", "[4.0, 8.0)", "[8.0, inf)"]
    assert [c["n"] for c in curve] == [2, 0, 1, 0, 1] and curve[0]["mean"] == 0.5 and curve[1]["mean"] is None


def test_gate_r1_needs_all_four_conditions():
    acc = _acc([{0}] * 6 + [{1}] * 6)
    patients = np.array([f"p{i}" for i in range(12)])
    model = _probs([0] * 6 + [1] * 6)                                        # perfect
    bgeo = _probs([0] * 6 + [0] * 6)                                         # wrong on the cortex half
    g = gate_r1(model, {"bprior": bgeo, "bgeo": bgeo, "b2": bgeo}, acc, patients, n_boot=200)
    assert g["accuracy"]["model"] == 1.0 and g["accuracy"]["bgeo"] == 0.5 and g["net_rescue_vs_bgeo"]["mean"] == 0.5
    assert g["net_rescue_ci_low_gt_0"] and g["gt_bprior"] and g["gt_bgeo"] and g["gt_b2"] and g["go"]
    tie = gate_r1(bgeo, {"bprior": bgeo, "bgeo": bgeo, "b2": bgeo}, acc, patients, n_boot=200)
    assert not tie["gt_bgeo"] and not tie["go"] and tie["net_rescue_vs_bgeo"]["mean"] == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_metrics.py -q -p no:cacheprovider`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `anatobind/eval/relation_metrics.py`**

```python
"""Gate R1 statistics (v2.6 §12; spec 2026-09-27 §6): set-valued correctness, the singleton-subset metrics, rescue /
harm / net rescue against a comparator, patient-level cluster bootstrap, exact McNemar, strata and the Δd curve, and the
§12.4 GO rule as a function. Nothing here knows where predictions or labels came from."""
import numpy as np
from scipy.stats import binomtest

from anatobind.relation.table import N_OUT, N_SLOTS, NONE_SLOT

N_BOOT = 10_000
SEED = 0
DELTA_D_EDGES = (0.0, 1.0, 2.0, 4.0, 8.0, float("inf"))


def predicted_slot(probs):
    return np.asarray(probs).argmax(1)


def is_correct(probs, acceptable):
    pred = predicted_slot(probs)
    acc = np.asarray(acceptable, bool)
    ok = pred < NONE_SLOT
    out = np.zeros(len(pred), bool)
    out[ok] = acc[np.nonzero(ok)[0], pred[ok]]
    return out


def topk_correct(probs, acceptable, k=2):
    p = np.asarray(probs)[:, :N_SLOTS]
    top = np.argsort(-p, 1)[:, :k]
    acc = np.asarray(acceptable, bool)
    return np.array([acc[i, top[i]].any() for i in range(len(p))])


def _per_class(truth, pred):
    """Recall, precision and F1 per slot over the classes present in the truth; balanced accuracy = mean recall."""
    recalls, f1s = [], []
    for c in range(N_SLOTS):
        t, p = truth == c, pred == c
        if not t.any():
            continue
        tp = float((t & p).sum())
        recall = tp / t.sum()
        precision = tp / p.sum() if p.any() else 0.0
        recalls.append(recall)
        f1s.append(0.0 if tp == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(recalls)), float(np.mean(f1s))


def summary(probs, acceptable):
    acc = np.asarray(acceptable, bool)
    correct = is_correct(probs, acc)
    single = acc.sum(1) == 1
    out = {"n": int(len(correct)), "accuracy": float(correct.mean()) if len(correct) else float("nan"),
           "singleton_rate": float(single.mean()) if len(correct) else float("nan"),
           "top2_accuracy": float(topk_correct(probs, acc, 2).mean()) if len(correct) else float("nan")}
    truth, pred = acc[single].argmax(1), predicted_slot(probs)[single]
    conf = np.zeros((N_SLOTS, N_OUT), int)
    for t, p in zip(truth, pred):
        conf[t, p] += 1
    bal, f1 = _per_class(truth, pred) if single.any() else (float("nan"), float("nan"))
    out["singleton"] = {"n": int(single.sum()), "accuracy": float(correct[single].mean()) if single.any() else float("nan"),
                        "macro_f1": f1, "balanced_accuracy": bal, "confusion": conf.tolist()}
    return out


def rescue_harm(correct_model, correct_comp):
    m, c = np.asarray(correct_model, bool), np.asarray(correct_comp, bool)
    rescue, harm = int((m & ~c).sum()), int((~m & c).sum())
    return {"rescue": rescue, "harm": harm, "net": rescue - harm, "net_rate": (rescue - harm) / len(m) if len(m) else 0.0}


def patient_bootstrap(values, patients, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Mean of ``values`` with a percentile CI from resampling patients with replacement (clusters of lesions)."""
    v, p = np.asarray(values, float), np.asarray(patients)
    ids = np.unique(p)
    groups = [v[p == i] for i in ids]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups], float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    stat = sums[draws].sum(1) / counts[draws].sum(1)
    return float(v.mean()), float(np.quantile(stat, alpha / 2)), float(np.quantile(stat, 1 - alpha / 2))


def mcnemar(b, c):
    """Exact two-sided p-value for the discordant counts (rescue b, harm c)."""
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)


def by_stratum(values, keys):
    v, k = np.asarray(values, float), np.asarray(keys)
    return {str(s): {"n": int((k == s).sum()), "mean": float(v[k == s].mean())} for s in np.unique(k)}


def delta_d_curve(values, delta_d, patients, edges=DELTA_D_EDGES, n_boot=N_BOOT, seed=SEED):
    v, d, p = np.asarray(values, float), np.asarray(delta_d, float), np.asarray(patients)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d >= lo) & (d < hi)
        if sel.any():
            mean, ci_lo, ci_hi = patient_bootstrap(v[sel], p[sel], n_boot, seed)
        else:
            mean = ci_lo = ci_hi = None
        out.append({"bin": f"[{lo}, {hi})", "n": int(sel.sum()), "mean": mean, "lo": ci_lo, "hi": ci_hi})
    return out


def gate_r1(model_probs, comparators, acceptable, patients, n_boot=N_BOOT, seed=SEED):
    """v2.6 §12.4: net rescue vs Bgeo+ has a CI lower bound > 0, and the model beats Bprior, Bgeo+ and B2 in accuracy."""
    acc = np.asarray(acceptable, bool)
    cm = is_correct(model_probs, acc)
    correct = {name: is_correct(p, acc) for name, p in comparators.items()}
    diff = cm.astype(float) - correct["bgeo"].astype(float)
    mean, lo, hi = patient_bootstrap(diff, patients, n_boot, seed)
    accuracy = {"model": float(cm.mean()), **{k: float(v.mean()) for k, v in correct.items()}}
    out = {"accuracy": accuracy, "net_rescue_vs_bgeo": {"mean": mean, "lo": lo, "hi": hi, **rescue_harm(cm, correct["bgeo"])},
           "net_rescue_ci_low_gt_0": bool(lo > 0), "gt_bprior": bool(accuracy["model"] > accuracy["bprior"]),
           "gt_bgeo": bool(accuracy["model"] > accuracy["bgeo"]), "gt_b2": bool(accuracy["model"] > accuracy["b2"])}
    out["go"] = bool(out["net_rescue_ci_low_gt_0"] and out["gt_bprior"] and out["gt_bgeo"] and out["gt_b2"])
    return out
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_metrics.py -q -p no:cacheprovider`
Expected: pass. (`mcnemar(10, 0)`: binomtest(0, 10, 0.5) two-sided = 2 · 0.5¹⁰.)

- [ ] **Step 5: Commit**

```bash
git add anatobind/eval/relation_metrics.py tests/test_relation_metrics.py
git commit -m "Relation metrics: set-valued correctness, singleton-subset metrics, rescue and harm, patient bootstrap, McNemar, strata, delta-d curve and the Gate R1 rule"
```

---

### Task 8: B0 and Bprior (`anatobind/relation/baselines.py`, part 1)

**Files:**
- Create: `anatobind/relation/baselines.py`
- Test: `tests/test_relation_baselines.py`

**Interfaces:**
```python
PRIOR_VARIANTS = ("majority", "type", "type_side", "type_side_location")
def target_weights(acceptable) -> (N, 7) float            # 1/|Y| on each acceptable slot; rows without a label are zero
def b0_probs(table) -> (N, 8)                              # one-hot at c1_slot, candidate-masked
class PriorModel:
    def __init__(self, variant, alpha=1.0)
    def key(self, row) -> tuple
    def fit(self, table, acceptable) -> self                # counts weighted by target_weights
    def predict(self, table) -> (N, 8)                      # (count + alpha) / (total + 7 alpha); unseen key -> global counts; masked
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relation_baselines.py
import numpy as np
import pytest

from anatobind.relation.baselines import PRIOR_VARIANTS, PriorModel, b0_probs, target_weights
from anatobind.relation.labels import acceptable_matrix, c1_labels
from synth_relation import synthetic_table


def test_b0_is_the_c1_rule_and_respects_candidates():
    t, _ = synthetic_table(3, 4)
    p = b0_probs(t)
    assert p.shape == (12, 8) and p.argmax(1).tolist() == t.c1_slot().tolist() and np.allclose(p.sum(1), 1)


def test_target_weights_split_sets_evenly():
    acc = np.zeros((3, 7), bool)
    acc[0, 0] = True
    acc[1, [0, 1]] = True
    w = target_weights(acc)
    assert w[0].tolist() == [1] + [0] * 6 and w[1, 0] == w[1, 1] == 0.5 and w[2].sum() == 0


def test_prior_variants_count_with_smoothing_and_fall_back_on_unseen_keys():
    t, _ = synthetic_table(6, 4)
    _, acc = acceptable_matrix(c1_labels(t), t.lesion_id)
    for v in PRIOR_VARIANTS:
        p = PriorModel(v).fit(t, acc).predict(t)
        assert p.shape == (24, 8) and np.allclose(p.sum(1), 1) and (p[:, 7] == 0).all()
    m = PriorModel("majority").fit(t, acc)
    counts = np.bincount(t.c1_slot(), minlength=7).astype(float)
    expected = (counts + 1) / (counts.sum() + 7)
    cand = t.candidates()[0]
    raw = np.where(cand, expected, 0)
    assert np.allclose(m.predict(t)[0, :7], raw / raw.sum())
    other, _ = synthetic_table(1, 2, seed=99)
    other.rows[0]["coarse_location"] = "unseen|key|here"
    q = PriorModel("type_side_location").fit(t, acc).predict(other)
    assert np.allclose(q.sum(1), 1)                                          # unseen key -> global counts, still a distribution
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_baselines.py -q -p no:cacheprovider`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `anatobind/relation/baselines.py` (part 1)**

```python
"""The non-torch arms of spec 2026-09-27 §5: B0 (the C1 rule on the table), Bprior (smoothed counts) and Bgeo+
(sklearn learners on the flat feature vector). Every arm returns (N, 8) probabilities restricted to the candidate set."""
import numpy as np

from anatobind.relation.table import N_OUT, N_SLOTS, features_flat, mask_to_candidates

PRIOR_VARIANTS = ("majority", "type", "type_side", "type_side_location")


def target_weights(acceptable):
    acc = np.asarray(acceptable, float)
    n = acc.sum(1, keepdims=True)
    return np.where(n > 0, acc / np.where(n > 0, n, 1.0), 0.0)


def b0_probs(table):
    p = np.zeros((len(table), N_OUT))
    p[np.arange(len(table)), table.c1_slot()] = 1.0
    return mask_to_candidates(p, table.candidates())


class PriorModel:
    def __init__(self, variant, alpha=1.0):
        if variant not in PRIOR_VARIANTS:
            raise ValueError(variant)
        self.variant, self.alpha = variant, float(alpha)
        self.counts, self.total = {}, np.zeros(N_SLOTS)

    def key(self, row):
        if self.variant == "majority":
            return ()
        if self.variant == "type":
            return (row["lesion_type"],)
        if self.variant == "type_side":
            return (row["lesion_type"], row["side"])
        return (row["lesion_type"], row["coarse_location"])          # coarse_location already carries the side (P9)

    def fit(self, table, acceptable):
        w = target_weights(acceptable)
        self.counts, self.total = {}, np.zeros(N_SLOTS)
        for row, wi in zip(table.rows, w):
            self.counts[self.key(row)] = self.counts.get(self.key(row), np.zeros(N_SLOTS)) + wi
            self.total += wi
        return self

    def predict(self, table):
        p = np.zeros((len(table), N_OUT))
        for i, row in enumerate(table.rows):
            c = self.counts.get(self.key(row), self.total)
            p[i, :N_SLOTS] = (c + self.alpha) / (c.sum() + N_SLOTS * self.alpha)
        return mask_to_candidates(p, table.candidates())
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_baselines.py -q -p no:cacheprovider`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add anatobind/relation/baselines.py tests/test_relation_baselines.py
git commit -m "Relation baselines: B0 as the C1 rule on the table and the four smoothed Bprior variants"
```

---

### Task 9: Bgeo+ learners and grids (`baselines.py`, part 2)

**Files:**
- Modify: `anatobind/relation/baselines.py` (append), `requirements-ci.txt` (add `scikit-learn==1.9.0`)
- Test: `tests/test_relation_bgeo.py`

**Interfaces:**
```python
GEO_KINDS = ("lr", "hgb", "mlp")                                         # tie order across learners (spec §5.3)
# within a learner the grid order is the tie order: simpler / more regularised first (C ascending, lr and depth ascending, alpha descending)
GEO_GRIDS = {"lr": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
             "hgb": [{"learning_rate": lr, "max_depth": d} for lr in (0.03, 0.1) for d in (3, 6)],
             "mlp": [{"alpha": a} for a in (1e-3, 1e-4)]}
def expand_sets(X, acceptable) -> (X_rep, y_rep, w_rep)                    # one row per (lesion, acceptable slot), weight 1/|Y|
class GeoLearner:
    def __init__(self, kind, params, seed=0)
    def fit(self, X, acceptable) -> self                                  # standardises on X, then fits
    def predict_probs(self, X, candidates) -> (N, 8)                     # classes not seen in training get 0; masked
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relation_bgeo.py
import numpy as np
import pytest

pytest.importorskip("sklearn")

from anatobind.relation.baselines import GEO_GRIDS, GEO_KINDS, GeoLearner, expand_sets
from anatobind.relation.labels import acceptable_matrix, c1_labels
from anatobind.relation.table import features_flat
from synth_relation import synthetic_table


def test_expand_sets_duplicates_ambiguous_lesions_with_split_weights():
    X = np.arange(6, dtype=float).reshape(3, 2)
    acc = np.zeros((3, 7), bool)
    acc[0, 0] = True
    acc[1, [0, 1]] = True
    acc[2, 3] = True
    Xr, yr, wr = expand_sets(X, acc)
    assert Xr.shape == (4, 2) and yr.tolist() == [0, 0, 1, 3] and wr.tolist() == [1.0, 0.5, 0.5, 1.0]


@pytest.mark.parametrize("kind", GEO_KINDS)
def test_every_learner_fits_c1_from_the_geometry_and_masks_candidates(kind):
    t, _ = synthetic_table(8, 6)
    X = features_flat(t)
    _, acc = acceptable_matrix(c1_labels(t), t.lesion_id)
    # the least regularised grid point: this asks whether the learner can represent C1, not which setting the inner CV picks
    # (C = 0.01 reaches only 0.896 on these 48 rows)
    m = GeoLearner(kind, GEO_GRIDS[kind][-1]).fit(X, acc)
    p = m.predict_probs(X, t.candidates())
    assert p.shape == (48, 8) and np.allclose(p.sum(1), 1) and (p[:, 7] == 0).all()
    assert (p.argmax(1) == t.c1_slot()).mean() >= 0.9                         # C1 is a function of the geometry columns
    assert (p[~np.concatenate([t.candidates(), np.ones((48, 1), bool)], 1)] == 0).all()


def test_unseen_classes_get_zero_probability():
    t, _ = synthetic_table(4, 4)
    X = features_flat(t)
    _, acc = acceptable_matrix(c1_labels(t), t.lesion_id)
    p = GeoLearner("lr", {"C": 1.0}).fit(X, acc).predict_probs(X, np.ones((16, 7), bool))
    assert (p[:, 2:7] == 0).all()                                               # only WM / cortex occur in the synthetic labels
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_bgeo.py -q -p no:cacheprovider`
Expected: ImportError `GEO_GRIDS`.

- [ ] **Step 3: Implement (append to `anatobind/relation/baselines.py`)**

```python
# --- Bgeo+ (spec §5.3) ---------------------------------------------------------------------------------------------
GEO_KINDS = ("lr", "hgb", "mlp")
GEO_GRIDS = {"lr": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
             "hgb": [{"learning_rate": lr, "max_depth": d} for lr in (0.03, 0.1) for d in (3, 6)],
             "mlp": [{"alpha": a} for a in (1e-3, 1e-4)]}


def expand_sets(X, acceptable):
    """One training row per (lesion, acceptable slot) with weight 1 / |Y|; unlabeled lesions are dropped."""
    w = target_weights(acceptable)
    rows, ys, ws = [], [], []
    for i in range(len(w)):
        for s in np.nonzero(w[i] > 0)[0]:
            rows.append(i)
            ys.append(int(s))
            ws.append(float(w[i, s]))
    return np.asarray(X)[rows], np.array(ys, int), np.array(ws, float)


class GeoLearner:
    def __init__(self, kind, params, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        if kind not in GEO_KINDS:
            raise ValueError(kind)
        self.kind, self.params, self.scaler = kind, dict(params), StandardScaler()
        if kind == "lr":
            self.model = LogisticRegression(C=params["C"], max_iter=2000)
        elif kind == "hgb":
            self.model = HistGradientBoostingClassifier(learning_rate=params["learning_rate"], max_depth=params["max_depth"], random_state=seed)
        else:
            self.model = MLPClassifier(hidden_layer_sizes=(64, 64), alpha=params["alpha"], early_stopping=True, max_iter=500, random_state=seed)

    def fit(self, X, acceptable):
        Xr, y, w = expand_sets(X, acceptable)
        Xs = self.scaler.fit_transform(Xr)
        self.model.fit(Xs, y, sample_weight=w)       # all three accept sample_weight in scikit-learn 1.9
        return self

    def predict_probs(self, X, candidates):
        p = np.zeros((len(X), N_OUT))
        probs = self.model.predict_proba(self.scaler.transform(np.asarray(X)))
        for j, c in enumerate(self.model.classes_):
            p[:, int(c)] = probs[:, j]
        return mask_to_candidates(p, candidates)
```

Add to `requirements-ci.txt` (after `scipy==1.17.1`): `scikit-learn==1.9.0`.

- [ ] **Step 4: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_bgeo.py tests/test_relation_baselines.py -q -p no:cacheprovider`
Expected: pass. If the MLP arm falls below 0.9 agreement on the synthetic table with `early_stopping=True` and 48 rows (its validation split is tiny), lower nothing in the assertion; instead set `n_iter_no_change=20` in the MLP constructor and re-run — the spec's learner is the 64-64 MLP with early stopping, and the fix must keep it.

- [ ] **Step 5: Commit**

```bash
git add anatobind/relation/baselines.py requirements-ci.txt tests/test_relation_bgeo.py
git commit -m "Relation baselines: Bgeo+ logistic, gradient-boosting and MLP learners on the flat geometry with set-label expansion; add scikit-learn to the CI requirements"
```

---

### Task 10: Patch crops, flip-consistent augmentation and the lesion encoder (`anatobind/relation/encoder.py`)

**Files:**
- Create: `anatobind/relation/encoder.py`
- Test: `tests/test_relation_encoder.py`

**Interfaces:**
```python
TORCH_CONFIGS = ({"px": 32, "slices": 3}, {"px": 32, "slices": 1}, {"px": 22, "slices": 3}, {"px": 42, "slices": 3})
# tie order (P11): the §10.1 initial config first, then by input size px² x slices ascending (1024 < 1452 < 5292)
DX_INDEX, SIDE_LEFT, SIDE_RIGHT = 1, 21, 22        # positions inside the 26-wide per-slot geometry (Task 3 layout)
def config_key(cfg) -> str                          # "px32_s3"
def crop(image, mask, px, slices) -> np.ndarray float32 (N, 2 * slices, px, px)   # channels per slice: image, mask; slices=1 keeps the centre slice
def flip_batch(x, geo) -> (x_flipped, geo_flipped)   # flips the column axis (axis 2), negates dx in every slot, swaps the two side one-hots
def augment(x, geo, rng, flip_p=0.5, intensity=0.1) -> (x, geo)   # per-sample flips; image channels scaled by U(1 - intensity, 1 + intensity)
class LesionEncoder(nn.Module): forward(x (N, C, px, px)) -> (N, 128)
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relation_encoder.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.relation.encoder import DX_INDEX, SIDE_LEFT, SIDE_RIGHT, TORCH_CONFIGS, LesionEncoder, augment, config_key, crop, flip_batch
from anatobind.relation.table import features_per_slot
from synth_relation import synthetic_table


def test_crop_sizes_and_channel_layout():
    t, p = synthetic_table(2, 2)
    for cfg in TORCH_CONFIGS:
        x = crop(p["image"], p["mask"], cfg["px"], cfg["slices"])
        assert x.shape == (4, 2 * cfg["slices"], cfg["px"], cfg["px"]) and x.dtype == np.float32
    x = crop(p["image"], p["mask"], 32, 1)
    assert np.array_equal(x[:, 0], p["image"][:, 1, 8:40, 8:40].astype(np.float32))       # the centre slice, centred crop
    assert x[:, 1].max() == 1.0 and set(np.unique(x[:, 1])) <= {0.0, 1.0}                  # mask channel
    assert config_key(TORCH_CONFIGS[0]) == "px32_s3"


def test_flip_is_an_involution_and_keeps_geometry_consistent():
    t, p = synthetic_table(2, 2)
    x, geo = crop(p["image"], p["mask"], 32, 3), features_per_slot(t)
    xf, gf = flip_batch(x, geo)
    assert np.array_equal(xf, x[:, :, ::-1, :]) and np.array_equal(gf[:, :, DX_INDEX], -geo[:, :, DX_INDEX])
    assert np.array_equal(gf[:, :, SIDE_LEFT], geo[:, :, SIDE_RIGHT]) and np.array_equal(gf[:, :, SIDE_RIGHT], geo[:, :, SIDE_LEFT])
    xff, gff = flip_batch(xf, gf)
    assert np.array_equal(xff, x) and np.array_equal(gff, geo)
    others = [i for i in range(geo.shape[2]) if i not in (DX_INDEX, SIDE_LEFT, SIDE_RIGHT)]
    assert np.array_equal(gf[:, :, others], geo[:, :, others])


def test_augment_scales_only_image_channels_and_flips_per_sample():
    t, p = synthetic_table(4, 4)
    x, geo = crop(p["image"], p["mask"], 32, 3), features_per_slot(t)
    xa, ga = augment(x, geo, np.random.default_rng(0), flip_p=1.0, intensity=0.1)
    assert np.array_equal(xa[:, 1::2], x[:, 1::2, ::-1, :])                                  # masks flipped, never scaled
    ratio = xa[:, 0::2] / np.where(x[:, 0::2, ::-1, :] == 0, 1, x[:, 0::2, ::-1, :])
    assert (np.abs(ratio - 1) <= 0.1 + 1e-6).all() and np.array_equal(ga[:, :, DX_INDEX], -geo[:, :, DX_INDEX])
    xb, gb = augment(x, geo, np.random.default_rng(0), flip_p=0.0, intensity=0.0)
    assert np.array_equal(xb, x) and np.array_equal(gb, geo)


def test_encoder_output_is_128_dimensional_for_every_config():
    for cfg in TORCH_CONFIGS:
        enc = LesionEncoder(2 * cfg["slices"])
        out = enc(torch.zeros(3, 2 * cfg["slices"], cfg["px"], cfg["px"]))
        assert out.shape == (3, 128)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_encoder.py -q -p no:cacheprovider`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `anatobind/relation/encoder.py`**

```python
"""The shared lesion encoder E_u of spec 2026-09-27 §5.4 and the patch handling around it: centred crops of the stored
48 px window, and the augmentation that keeps image and geometry consistent (a left-right flip negates dx and swaps
the side one-hots, spec §5.7)."""
import numpy as np
import torch
import torch.nn as nn

from anatobind.relation.table import PATCH_PX, PATCH_SLICES

# tie order (P11): the §10.1 initial config, then by input size px² x slices ascending
TORCH_CONFIGS = ({"px": 32, "slices": 3}, {"px": 32, "slices": 1}, {"px": 22, "slices": 3}, {"px": 42, "slices": 3})
DX_INDEX = 1                     # per-slot layout: [candidate, dx, dy, dz, centroid_distance, signed, min_surface, ioa, soft, ...]
SIDE_LEFT, SIDE_RIGHT = 9 + 12, 9 + 13     # lesion block starts at 9: 8 geometry + 4 extras, then side one-hot (left, right, midline)


def config_key(cfg):
    return f"px{cfg['px']}_s{cfg['slices']}"


def crop(image, mask, px, slices):
    """(N, 2 * slices, px, px) float32: for each kept slice the image then the mask; the crop is centred in the window."""
    if slices not in (1, PATCH_SLICES):
        raise ValueError(slices)
    lo = (PATCH_PX - px) // 2
    keep = [PATCH_SLICES // 2] if slices == 1 else list(range(PATCH_SLICES))
    img = np.asarray(image, np.float32)[:, keep, lo:lo + px, lo:lo + px]
    msk = np.asarray(mask, np.float32)[:, keep, lo:lo + px, lo:lo + px]
    n = img.shape[0]
    out = np.empty((n, 2 * slices, px, px), np.float32)
    out[:, 0::2] = img
    out[:, 1::2] = msk
    return out


def flip_batch(x, geo):
    xf = np.ascontiguousarray(x[:, :, ::-1, :])
    gf = np.array(geo, copy=True)
    gf[:, :, DX_INDEX] = -geo[:, :, DX_INDEX]
    gf[:, :, SIDE_LEFT], gf[:, :, SIDE_RIGHT] = geo[:, :, SIDE_RIGHT], geo[:, :, SIDE_LEFT]
    return xf, gf


def augment(x, geo, rng, flip_p=0.5, intensity=0.1):
    x, geo = np.array(x, np.float32, copy=True), np.array(geo, np.float32, copy=True)
    flip = rng.uniform(size=len(x)) < flip_p
    if flip.any():
        xf, gf = flip_batch(x[flip], geo[flip])
        x[flip], geo[flip] = xf, gf
    if intensity > 0:
        scale = rng.uniform(1 - intensity, 1 + intensity, size=(len(x), 1, 1, 1)).astype(np.float32)
        x[:, 0::2] *= scale
    return x, geo


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(), nn.MaxPool2d(2))


class LesionEncoder(nn.Module):
    """Four conv blocks 32-64-64-128 and global average pooling -> 128 (spec §5.4)."""

    def __init__(self, in_ch):
        super().__init__()
        self.blocks = nn.Sequential(_block(in_ch, 32), _block(32, 64), _block(64, 64), _block(64, 128))
        self.out_dim = 128

    def forward(self, x):
        return self.blocks(x).mean((2, 3))
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_encoder.py -q -p no:cacheprovider`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add anatobind/relation/encoder.py tests/test_relation_encoder.py
git commit -m "Relation encoder: centred patch crops for the four configurations, flip-consistent augmentation and the shared 128-d lesion encoder"
```

---

### Task 11: B1 and B2 models with the set-valued loss (`anatobind/relation/models.py`)

**Files:**
- Create: `anatobind/relation/models.py`
- Test: `tests/test_relation_models.py`

**Interfaces:**
```python
GEO_DIM, D_MODEL, CLASS_EMBED = 26, 64, 32
class B2Model(nn.Module): __init__(in_ch); forward(x) -> logits (N, 8)
class B1Model(nn.Module): __init__(in_ch); forward(x, geo (N, 7, 26), present (N, 7) bool) -> logits (N, 8)   # wraps IndependentCandidateHead
def set_nll(logits, acceptable8) -> scalar tensor     # -log sum_{c in Y} softmax(logits)_c, mean over rows; acceptable8 (N, 8) bool
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relation_models.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.relation.encoder import crop
from anatobind.relation.models import B1Model, B2Model, set_nll
from anatobind.relation.table import features_per_slot
from synth_relation import synthetic_table


def _batch():
    t, p = synthetic_table(2, 3)
    x = torch.from_numpy(crop(p["image"], p["mask"], 32, 3))
    geo = torch.from_numpy(features_per_slot(t))
    present = torch.from_numpy(t.candidates())
    return x, geo, present, torch.from_numpy(t.c1_slot())


def test_b2_and_b1_shapes_and_absent_candidates():
    x, geo, present, _ = _batch()
    torch.manual_seed(0)
    assert B2Model(6)(x).shape == (6, 8)
    b1 = B1Model(6).eval()
    with torch.no_grad():
        probs = b1(x, geo, present).softmax(-1)
    assert probs.shape == (6, 8) and torch.allclose(probs.sum(-1), torch.ones(6))
    assert (probs[:, :7][~present] == 0).all()                       # the head masks absent candidates to -inf


def test_set_nll_equals_cross_entropy_for_singletons_and_pools_sets():
    logits = torch.tensor([[2.0, 1.0, 0.0, 0, 0, 0, 0, 0], [2.0, 1.0, 0.0, 0, 0, 0, 0, 0]])
    single = torch.zeros(2, 8, dtype=torch.bool)
    single[:, 1] = True
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 1]))
    assert torch.isclose(set_nll(logits, single), ce)
    both = single.clone()
    both[:, 0] = True
    p = logits.softmax(-1)
    assert torch.isclose(set_nll(logits, both), -(p[:, 0] + p[:, 1]).log().mean())
    assert set_nll(logits, both) < set_nll(logits, single)


def test_models_train_one_step_without_nan():
    x, geo, present, c1 = _batch()
    y = torch.zeros(6, 8, dtype=torch.bool)
    y[torch.arange(6), c1] = True          # C1 is always a candidate; a label outside the candidates would give B1 an infinite loss (cv.trainable_rows)
    for model, args in ((B2Model(6), (x,)), (B1Model(6), (x, geo, present))):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss = set_nll(model(*args), y)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_models.py -q -p no:cacheprovider`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `anatobind/relation/models.py`**

```python
"""B2 (local appearance: encoder + linear head) and B1 (independent candidate scoring on top of the same encoder,
spec 2026-09-27 §5.5–5.6), plus the set-valued negative log-likelihood of §5.7. B1 reuses IndependentCandidateHead
with one lesion per sample (M = 1, K = 7)."""
import torch
import torch.nn as nn

from anatobind.model.relation import IndependentCandidateHead
from anatobind.relation.encoder import LesionEncoder
from anatobind.relation.table import N_OUT, N_SLOTS

GEO_DIM, D_MODEL, CLASS_EMBED = 26, 64, 32


class B2Model(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.encoder = LesionEncoder(in_ch)
        self.head = nn.Linear(self.encoder.out_dim, N_OUT)

    def forward(self, x):
        return self.head(self.encoder(x))


class B1Model(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.encoder = LesionEncoder(in_ch)
        self.u_proj = nn.Linear(self.encoder.out_dim, D_MODEL)
        self.class_embed = nn.Embedding(N_SLOTS, CLASS_EMBED)
        self.a_mlp = nn.Sequential(nn.Linear(CLASS_EMBED + GEO_DIM, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.head = IndependentCandidateHead(d_model=D_MODEL, geometry_channels=GEO_DIM, geo_dim=32, hidden_dim=D_MODEL)

    def forward(self, x, geo, present):
        n = x.shape[0]
        emb = self.class_embed.weight[None].expand(n, N_SLOTS, CLASS_EMBED)
        a = self.a_mlp(torch.cat([emb, geo], -1))                       # (N, 7, 64)
        u = self.u_proj(self.encoder(x))[:, None, :]                     # (N, 1, 64)
        out = self.head(a, u, geo[:, :, None, :], present.bool())
        return out["host_logits"][:, 0, :]                               # (N, 8)


def set_nll(logits, acceptable8):
    """-log of the probability mass on the acceptable set, averaged over rows (cross-entropy for singletons)."""
    masked = logits.masked_fill(~acceptable8.bool(), float("-inf"))
    return (torch.logsumexp(logits, -1) - torch.logsumexp(masked, -1)).mean()
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_models.py tests/test_independent_candidate.py -q -p no:cacheprovider`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add anatobind/relation/models.py tests/test_relation_models.py
git commit -m "Relation models: B2 appearance head and B1 independent candidate scoring on the shared encoder, with the set-valued loss"
```

---

### Task 12: Folds, configuration selection, prediction files and the sklearn arm drivers (`anatobind/relation/cv.py`)

**Files:**
- Create: `anatobind/relation/cv.py`
- Test: `tests/test_relation_cv.py`

**Interfaces:**
```python
TIE_TOL = 1e-3
PROB_COLUMNS = [f"p_{SLOT_PREFIX[s]}" for s in SLOTS] + ["p_none"]
def outer_folds(table) -> list of (k, train_idx, test_idx)
def inner_folds(patients, k=5, seed=0) -> list of (train_idx, val_idx)      # by patient; indices into the given array
def select_config(scores, tol=TIE_TOL) -> (key, record)                       # scores: [(key, score)] in tie order; record {"scores", "chosen", "tie"}
def labels_for_fold(labels, k) -> dict                                       # labels is a dict or a callable(k) -> dict
def set_accuracy(probs, acceptable) -> float
def trainable_rows(has, acceptable, candidates) -> (N,) bool   # labelled and at least one acceptable slot is a candidate (P7); the same rows train every learned arm
def empty_preds(table) -> dict(lesion_id, fold, config (object), probs (N, 8))
def write_preds(path, preds); def read_preds(path) -> preds dict
def run_b0(table) -> (preds, record)
def run_bprior(table, labels, variant) -> (preds, record)
def run_bgeo(table, labels, kind, seed=0, inner_k=5) -> (preds, record)     # record["folds"][k] = {"scores", "chosen", "tie"}; record["mean_inner_score"]
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relation_cv.py
import numpy as np
import pytest

pytest.importorskip("sklearn")

from anatobind.relation.cv import (
    inner_folds, labels_for_fold, outer_folds, read_preds, run_b0, run_bgeo, run_bprior, select_config, trainable_rows,
    write_preds,
)
from anatobind.relation.labels import acceptable_matrix, c1_labels
from anatobind.eval.relation_metrics import is_correct
from synth_relation import synthetic_table


def test_outer_folds_come_from_the_table_and_cover_every_lesion():
    t, _ = synthetic_table(10, 3)
    folds = outer_folds(t)
    assert [k for k, _, _ in folds] == [0, 1, 2, 3, 4]
    assert sorted(np.concatenate([te for _, _, te in folds]).tolist()) == list(range(30))
    for k, tr, te in folds:
        assert set(t.patients()[tr]).isdisjoint(t.patients()[te]) and (t.folds()[te] == k).all()


def test_inner_folds_split_by_patient_and_are_seeded():
    patients = np.array([f"p{i // 3}" for i in range(30)])
    a, b = inner_folds(patients, 5, seed=0), inner_folds(patients, 5, seed=0)
    assert len(a) == 5 and all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    for tr, va in a:
        assert set(patients[tr]).isdisjoint(patients[va]) and len(tr) + len(va) == 30
    assert sorted(np.concatenate([va for _, va in a]).tolist()) == list(range(30))
    assert not all(np.array_equal(x[1], y[1]) for x, y in zip(a, inner_folds(patients, 5, seed=1)))


def test_select_config_prefers_the_earliest_within_tolerance():
    key, rec = select_config([("a", 0.90), ("b", 0.9005), ("c", 0.95)])
    assert key == "c" and rec["tie"] is False
    key, rec = select_config([("a", 0.9995), ("b", 1.0), ("c", 0.99)])
    assert key == "a" and rec["tie"] is True and rec["chosen"] == "a"


def test_preds_round_trip(tmp_path):
    t, _ = synthetic_table(2, 2)
    preds, _ = run_b0(t)
    write_preds(tmp_path / "b0.csv", preds)
    back = read_preds(tmp_path / "b0.csv")
    assert back["lesion_id"].tolist() == preds["lesion_id"].tolist() and np.allclose(back["probs"], preds["probs"])
    assert back["config"].tolist() == ["c1"] * 4 and back["fold"].tolist() == preds["fold"].tolist()


def test_bprior_and_bgeo_cover_every_lesion_once_and_are_deterministic():
    t, _ = synthetic_table(10, 4)
    labels = c1_labels(t)
    _, acc = acceptable_matrix(labels, t.lesion_id)
    p1, rec1 = run_bprior(t, labels, "type_side")
    assert p1["probs"].shape == (40, 8) and np.allclose(p1["probs"].sum(1), 1) and sorted(p1["lesion_id"].tolist()) == list(range(40))
    g1, r1 = run_bgeo(t, labels, "lr", seed=0, inner_k=3)
    g2, r2 = run_bgeo(t, labels, "lr", seed=0, inner_k=3)
    assert np.allclose(g1["probs"], g2["probs"]) and set(r1["folds"]) == {0, 1, 2, 3, 4}
    assert is_correct(g1["probs"], acc).mean() >= 0.8 and 0 <= r1["mean_inner_score"] <= 1
    assert all(r1["folds"][k]["chosen"] in {str(c) for c, _ in r1["folds"][k]["scores"]} for k in r1["folds"])


def test_labels_for_fold_accepts_a_dict_or_a_callable():
    d = {1: frozenset({"cortex"})}
    assert labels_for_fold(d, 3) is d and labels_for_fold(lambda k: {k: frozenset({"white_matter"})}, 3) == {3: frozenset({"white_matter"})}


def test_rows_whose_acceptable_set_misses_every_candidate_do_not_train():
    has = np.array([True, True, False])
    acc = np.zeros((3, 7), bool)
    acc[0, 0] = acc[1, 4] = True                                      # lesion 1: brainstem
    cand = np.zeros((3, 7), bool)
    cand[:, [0, 1]] = True                                            # brainstem is not a candidate anywhere
    assert trainable_rows(has, acc, cand).tolist() == [True, False, False]
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_cv.py -q -p no:cacheprovider`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `anatobind/relation/cv.py`**

```python
"""Outer five folds by patient (the Level R fold table), inner patient folds for selection, the P11 tie rule, the
out-of-fold prediction files, and the drivers of the arms that need no torch (spec 2026-09-27 §5.8, §6.1)."""
import csv
import json
from pathlib import Path

import numpy as np

from anatobind.eval.relation_metrics import is_correct
from anatobind.relation.baselines import GEO_GRIDS, GeoLearner, PriorModel, b0_probs
from anatobind.relation.labels import acceptable_matrix
from anatobind.relation.table import N_OUT, SLOTS, SLOT_PREFIX, features_flat

TIE_TOL = 1e-3
PROB_COLUMNS = [f"p_{SLOT_PREFIX[s]}" for s in SLOTS] + ["p_none"]


def outer_folds(table):
    folds = table.folds()
    return [(int(k), np.nonzero(folds != k)[0], np.nonzero(folds == k)[0]) for k in sorted(set(folds.tolist()))]


def inner_folds(patients, k=5, seed=0):
    """Patients shuffled with ``seed`` and dealt round-robin into k groups; indices are positions in ``patients``."""
    p = np.asarray(patients)
    ids = np.unique(p)
    ids = ids[np.random.default_rng(seed).permutation(len(ids))]
    out = []
    for g in range(k):
        held = set(ids[g::k].tolist())
        val = np.array([i for i in range(len(p)) if p[i] in held], int)
        train = np.array([i for i in range(len(p)) if p[i] not in held], int)
        if len(val):
            out.append((train, val))
    return out


def select_config(scores, tol=TIE_TOL):
    """scores: [(key, score)] in tie order. The first key whose score is within ``tol`` of the best wins (P11)."""
    best = max(s for _, s in scores)
    chosen = next(k for k, s in scores if s >= best - tol)
    tie = sum(1 for _, s in scores if s >= best - tol) > 1
    return chosen, {"scores": [(str(k), float(s)) for k, s in scores], "chosen": str(chosen), "tie": bool(tie)}


def labels_for_fold(labels, k):
    return labels(k) if callable(labels) else labels


def set_accuracy(probs, acceptable):
    c = is_correct(probs, acceptable)
    return float(c.mean()) if len(c) else float("nan")


def trainable_rows(has, acceptable, candidates):
    """A lesion trains an arm only if it is labelled and one of its acceptable slots is a candidate: every arm's output
    is restricted to the candidates (P7), so a truth outside them is unanswerable, and for B1 its loss would be infinite.
    Such lesions still count, as wrong for every arm, in evaluation."""
    return np.asarray(has, bool) & (np.asarray(acceptable, bool) & np.asarray(candidates, bool)).any(1)


def empty_preds(table):
    return {"lesion_id": table.lesion_id.copy(), "fold": table.folds().copy(), "config": np.array([""] * len(table), object),
            "probs": np.zeros((len(table), N_OUT))}


def write_preds(path, preds):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lesion_id", "fold", "config"] + PROB_COLUMNS)
        for i in range(len(preds["lesion_id"])):
            w.writerow([int(preds["lesion_id"][i]), int(preds["fold"][i]), preds["config"][i]] + [f"{v:.8f}" for v in preds["probs"][i]])


def read_preds(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {"lesion_id": np.array([int(r["lesion_id"]) for r in rows]), "fold": np.array([int(r["fold"]) for r in rows]),
            "config": np.array([r["config"] for r in rows], object),
            "probs": np.array([[float(r[c]) for c in PROB_COLUMNS] for r in rows])}


def run_b0(table):
    preds = empty_preds(table)
    preds["probs"] = b0_probs(table)
    preds["config"][:] = "c1"
    return preds, {"arm": "b0"}


def run_bprior(table, labels, variant):
    preds = empty_preds(table)
    preds["config"][:] = variant
    record = {"arm": f"bprior_{variant}", "n_untrainable": {}}
    for k, tr, te in outer_folds(table):
        has, acc = acceptable_matrix(labels_for_fold(labels, k), table.lesion_id)
        ok = trainable_rows(has, acc, table.candidates())
        record["n_untrainable"][k] = int((has[tr] & ~ok[tr]).sum())
        tr = tr[ok[tr]]
        preds["probs"][te] = PriorModel(variant).fit(table.subset(tr), acc[tr]).predict(table.subset(te))
    return preds, record


def run_bgeo(table, labels, kind, seed=0, inner_k=5):
    X = features_flat(table)
    cand = table.candidates()
    patients = table.patients()
    preds = empty_preds(table)
    record = {"arm": f"bgeo_{kind}", "folds": {}}
    for k, tr, te in outer_folds(table):
        has, acc = acceptable_matrix(labels_for_fold(labels, k), table.lesion_id)
        ok = trainable_rows(has, acc, cand)
        n_untrainable = int((has[tr] & ~ok[tr]).sum())
        tr = tr[ok[tr]]
        scores = []
        for params in GEO_GRIDS[kind]:
            accs = []
            for itr, ival in inner_folds(patients[tr], inner_k, seed):
                m = GeoLearner(kind, params, seed).fit(X[tr[itr]], acc[tr[itr]])
                accs.append(set_accuracy(m.predict_probs(X[tr[ival]], cand[tr[ival]]), acc[tr[ival]]))
            scores.append((json.dumps(params, sort_keys=True), float(np.mean(accs))))
        chosen, rec = select_config(scores)
        params = json.loads(chosen)
        preds["probs"][te] = GeoLearner(kind, params, seed).fit(X[tr], acc[tr]).predict_probs(X[te], cand[te])
        preds["config"][te] = chosen
        record["folds"][k] = {**rec, "n_untrainable": n_untrainable}
    record["mean_inner_score"] = float(np.mean([max(s for _, s in record["folds"][k]["scores"]) for k in record["folds"]]))
    return preds, record
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_cv.py -q -p no:cacheprovider`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add anatobind/relation/cv.py tests/test_relation_cv.py
git commit -m "Relation CV: patient folds, inner selection with the tie rule, prediction files and the B0, Bprior and Bgeo+ drivers"
```

---

### Task 13: Torch training loop with early stopping, refit and the stage-2 hook (`anatobind/relation/train.py`)

**Files:**
- Create: `anatobind/relation/train.py`
- Test: `tests/test_relation_train.py`

**Interfaces:**
```python
def acceptable8(acc7) -> (N, 8) bool
def fit_torch_arm(arm, x, geo, present, acc8, train_idx, val_idx, seed, device, epochs=40, patience=8, init_state=None, lr=1e-3, wd=1e-4, batch=64) -> (state_dict, best_epoch, best_val_acc)
def predict_torch_arm(arm, state, x, geo, present, idx, device) -> probs (n, 8) candidate-masked
def run_torch_arm(arm, table, patches, labels, configs=TORCH_CONFIGS, seed=0, device="cpu", epochs=40, patience=8, inner_k=5, init=None, log=None) -> (preds, record, states)
    # log(str) is called once per outer fold with the chosen config, epochs, untrainable count and seconds (progress on a shared GPU)
    # record["folds"][k] = {"scores", "chosen", "tie", "epochs", "inner_epochs", "n_untrainable"}; states = {k: state_dict}
    # init = {k: (config_key, state_dict)} (stage 2, P13): fold k fine-tunes its stage-1 model, so only that config is tried
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relation_train.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.relation.encoder import TORCH_CONFIGS, crop
from anatobind.relation.labels import acceptable_matrix, c1_labels
from anatobind.relation.table import features_per_slot, slot_col
from anatobind.relation.train import acceptable8, fit_torch_arm, predict_torch_arm, run_torch_arm
from synth_relation import synthetic_table


def _setup():
    t, p = synthetic_table(5, 4)
    labels = c1_labels(t)
    _, acc = acceptable_matrix(labels, t.lesion_id)
    x = crop(p["image"], p["mask"], 22, 1)
    return t, p, labels, acceptable8(acc), x, features_per_slot(t), t.candidates()


def test_fit_predict_shapes_early_stopping_and_determinism():
    t, p, labels, acc8, x, geo, present = _setup()
    tr, va = np.arange(0, 12), np.arange(12, 20)
    st1, ep1, va1 = fit_torch_arm("b1", x, geo, present, acc8, tr, va, seed=0, device="cpu", epochs=3, patience=1)
    st2, ep2, va2 = fit_torch_arm("b1", x, geo, present, acc8, tr, va, seed=0, device="cpu", epochs=3, patience=1)
    assert 1 <= ep1 <= 3 and 0 <= va1 <= 1 and ep1 == ep2 and va1 == va2
    p1, p2 = (predict_torch_arm("b1", s, x, geo, present, va, "cpu") for s in (st1, st2))
    assert p1.shape == (8, 8) and np.allclose(p1, p2) and np.allclose(p1.sum(1), 1) and (p1[:, :7][~present[va]] == 0).all()
    st, ep, acc = fit_torch_arm("b2", x, geo, present, acc8, tr, None, seed=0, device="cpu", epochs=2, patience=1)
    assert ep == 2 and np.isnan(acc)                                                     # refit without a validation fold


def test_run_torch_arm_covers_every_lesion_once_and_records_the_selection():
    t, p, labels, _, _, _, _ = _setup()
    lines = []
    preds, rec, states = run_torch_arm("b2", t, p, labels, configs=TORCH_CONFIGS[:2], seed=0, device="cpu", epochs=3, patience=1, inner_k=2,
                                       log=lines.append)
    assert sorted(preds["lesion_id"].tolist()) == list(range(20)) and np.allclose(preds["probs"].sum(1), 1)
    assert len(lines) == 5 and lines[0].startswith("b2 fold 0: chose px32_s")
    assert set(rec["folds"]) == {0, 1, 2, 3, 4} and set(states) == {0, 1, 2, 3, 4}
    for k, f in rec["folds"].items():
        assert f["chosen"] in ("px32_s3", "px32_s1") and f["n_untrainable"] == 0
        assert f["epochs"] == max(1, int(round(float(np.median(f["inner_epochs"][f["chosen"]])))))   # refit = median best inner epoch
    assert set(preds["config"].tolist()) <= {"px32_s3", "px32_s1"}


def test_stage_two_fine_tunes_each_fold_with_its_stage_one_config():
    t, p, labels, _, _, _, _ = _setup()
    _, rec1, states = run_torch_arm("b1", t, p, labels, configs=TORCH_CONFIGS[:2], seed=0, device="cpu", epochs=1, patience=1, inner_k=2)
    init = {k: (rec1["folds"][k]["chosen"], states[k]) for k in states}
    preds, rec2, _ = run_torch_arm("b1", t, p, labels, configs=TORCH_CONFIGS, seed=1, device="cpu", epochs=1, patience=1, inner_k=2, init=init)
    assert np.allclose(preds["probs"].sum(1), 1) and rec2["init_from_states"] is True
    for k in rec2["folds"]:
        assert rec2["folds"][k]["chosen"] == rec1["folds"][k]["chosen"] and len(rec2["folds"][k]["scores"]) == 1


def test_a_label_outside_the_candidates_is_dropped_from_training_not_turned_into_an_infinite_loss():
    t, p, labels, _, _, _, _ = _setup()
    t.rows[0][slot_col("brainstem", "candidate")] = False          # brainstem is not a candidate of lesion 0
    bad = dict(labels)
    bad[0] = frozenset({"brainstem"})
    preds, rec, _ = run_torch_arm("b1", t, p, bad, configs=TORCH_CONFIGS[:1], seed=0, device="cpu", epochs=1, patience=1, inner_k=2)
    assert np.isfinite(preds["probs"]).all() and sum(f["n_untrainable"] for f in rec["folds"].values()) == 4   # lesion 0 trains 4 outer folds
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_train.py -q -p no:cacheprovider`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `anatobind/relation/train.py`**

```python
"""Training of the torch arms (spec 2026-09-27 §5.7–5.9): AdamW, the set-valued loss, flip-consistent augmentation,
early stopping on the inner validation fold, refit on the outer training fold for the median best epoch, and the
stage-2 hook that starts from stage-1 weights."""
import copy
import time

import numpy as np
import torch

from anatobind.relation.cv import (
    empty_preds, inner_folds, labels_for_fold, outer_folds, select_config, set_accuracy, trainable_rows,
)
from anatobind.relation.encoder import TORCH_CONFIGS, augment, config_key, crop
from anatobind.relation.labels import acceptable_matrix
from anatobind.relation.models import B1Model, B2Model, set_nll
from anatobind.relation.table import N_OUT, N_SLOTS, features_per_slot, mask_to_candidates


def acceptable8(acc7):
    a = np.asarray(acc7, bool)
    return np.concatenate([a, np.zeros((len(a), 1), bool)], 1)


def _model(arm, in_ch):
    if arm == "b1":
        return B1Model(in_ch)
    if arm == "b2":
        return B2Model(in_ch)
    raise ValueError(arm)


def _forward(model, arm, xb, gb, pb):
    return model(xb, gb, pb) if arm == "b1" else model(xb)


def _tensors(device, *arrays):
    return [torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays]


def predict_torch_arm(arm, state, x, geo, present, idx, device, batch=256):
    model = _model(arm, x.shape[1]).to(device)
    model.load_state_dict(state)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            xb, gb, pb = _tensors(device, x[b].astype(np.float32), geo[b].astype(np.float32), present[b])
            out.append(_forward(model, arm, xb, gb, pb).softmax(-1).cpu().numpy())
    probs = np.concatenate(out) if out else np.zeros((0, N_OUT))
    return mask_to_candidates(probs, present[idx])


def fit_torch_arm(arm, x, geo, present, acc8, train_idx, val_idx, seed, device, epochs=40, patience=8, init_state=None,
                  lr=1e-3, wd=1e-4, batch=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _model(arm, x.shape[1]).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_state, best_acc, best_epoch, bad = copy.deepcopy(model.state_dict()), -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            xa, ga = augment(x[b], geo[b], rng)
            xb, gb, pb, yb = _tensors(device, xa, ga, present[b], acc8[b])
            loss = set_nll(_forward(model, arm, xb, gb, pb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val_idx is None:
            continue
        val_acc = set_accuracy(predict_torch_arm(arm, model.state_dict(), x, geo, present, val_idx, device), acc8[val_idx][:, :N_SLOTS])
        if val_acc > best_acc:
            best_state, best_acc, best_epoch, bad = copy.deepcopy(model.state_dict()), val_acc, epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if val_idx is None:
        return copy.deepcopy(model.state_dict()), epochs, float("nan")
    return best_state, best_epoch, best_acc


def run_torch_arm(arm, table, patches, labels, configs=TORCH_CONFIGS, seed=0, device="cpu", epochs=40, patience=8, inner_k=5,
                  init=None, log=None):
    geo, present, patients = features_per_slot(table), table.candidates(), table.patients()
    by_key = {config_key(c): c for c in configs}
    xs = {key: crop(patches["image"], patches["mask"], c["px"], c["slices"]) for key, c in by_key.items()}
    preds = empty_preds(table)
    record = {"arm": arm, "init_from_states": init is not None, "folds": {}}
    states = {}
    for k, tr, te in outer_folds(table):
        t0 = time.time()
        has, acc = acceptable_matrix(labels_for_fold(labels, k), table.lesion_id)
        acc8 = acceptable8(acc)
        ok = trainable_rows(has, acc, present)
        n_untrainable = int((has[tr] & ~ok[tr]).sum())
        tr = tr[ok[tr]]
        init_state = None
        keys = list(by_key)
        if init is not None:                                     # stage 2: fine-tune the stage-1 model of this fold
            init_key, init_state = init[k]
            keys = [init_key]
        scores, best_epochs, inner_epochs = [], {}, {}
        for key in keys:
            accs, eps = [], []
            for itr, ival in inner_folds(patients[tr], inner_k, seed):
                _, ep, va = fit_torch_arm(arm, xs[key], geo, present, acc8, tr[itr], tr[ival], seed, device, epochs, patience, init_state)
                accs.append(va)
                eps.append(ep)
            scores.append((key, float(np.mean(accs))))
            inner_epochs[key] = eps
            best_epochs[key] = max(1, int(round(float(np.median(eps)))))
        chosen, rec = select_config(scores)
        state, _, _ = fit_torch_arm(arm, xs[chosen], geo, present, acc8, tr, None, seed, device, best_epochs[chosen], patience, init_state)
        preds["probs"][te] = predict_torch_arm(arm, state, xs[chosen], geo, present, te, device)
        preds["config"][te] = chosen
        record["folds"][k] = {**rec, "epochs": best_epochs[chosen], "inner_epochs": inner_epochs, "n_untrainable": n_untrainable}
        states[k] = {n: v.detach().cpu() for n, v in state.items()}
        if log:
            log(f"{arm} fold {k}: chose {chosen} (tie {rec['tie']}), refit {best_epochs[chosen]} epochs, "
                f"untrainable {n_untrainable}, {time.time() - t0:.0f} s")
    return preds, record, states
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_train.py -q -p no:cacheprovider`
Expected: pass (CPU, well under a minute).

- [ ] **Step 5: Commit**

```bash
git add anatobind/relation/train.py tests/test_relation_train.py
git commit -m "Relation training: AdamW loop with the set-valued loss, early stopping, median-epoch refit, out-of-fold prediction and the stage-two initialisation hook"
```

---

### Task 14: Run and evaluation scripts with the end-to-end test

**Files:**
- Create: `scripts/run_relation_baselines.py`, `scripts/eval_relation_baselines.py`
- Test: `tests/test_relation_scripts.py`

**Interfaces:**
- `scripts/run_relation_baselines.py`: `main(argv=None)`; args `--table DIR --out DIR --labels {C1,R} [--sealed-dir --manifest] [--arms b0,bprior,bgeo,b1,b2] [--device cpu|cuda] [--threads 16] [--epochs 40] [--patience 8] [--seed 0] [--inner-k 5] [--init-from RUN_DIR]`. Writes `preds/b0.csv`, `preds/bprior_<variant>.csv`, `preds/bgeo_<kind>.csv`, `preds/b1.csv`, `preds/b2.csv`, `models/<arm>_fold<k>.pt`, `run.json` (`labels`, `seed`, `git_commit`, `table_manifest_sha256`, `records` per arm, `bgeo_best`, `bgeo_selection`). Refuses an existing `--out`.
- `scripts/eval_relation_baselines.py`: `main(argv=None)`; args `--run DIR --table DIR --labels {C1,R} [--unblind] [--sealed-dir --manifest] --out DIR [--n-boot 10000]`. Writes `REPORT.md`, `tables.csv`, `output.txt`. `--labels R` without `--unblind` exits with status 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_relation_scripts.py
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

from anatobind.level_r.admin import FINAL_COLUMNS, seal
from anatobind.relation.cv import read_preds
from anatobind.relation.table import write_table
from synth_relation import synthetic_table


def _script(name):
    """scripts/ is not a package; load it the way tests/test_eval_knee_folds_script.py does."""
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_main = _script("run_relation_baselines").main
eval_main = _script("eval_relation_baselines").main


@pytest.fixture
def table_dir(tmp_path):
    t, p = synthetic_table(10, 3)
    write_table(tmp_path / "v1", t.rows, p, {"version": "v1"})
    return tmp_path / "v1", t


def test_run_then_eval_on_c1_and_on_synthetic_sealed_labels(tmp_path, table_dir):
    tdir, t = table_dir
    run = tmp_path / "run"
    run_main(["--table", str(tdir), "--out", str(run), "--labels", "C1", "--device", "cpu", "--epochs", "1", "--patience", "1", "--inner-k", "2"])
    names = {"b0", "bprior_majority", "bprior_type", "bprior_type_side", "bprior_type_side_location", "bgeo_lr", "bgeo_hgb", "bgeo_mlp", "b1", "b2"}
    assert {p.stem for p in (run / "preds").glob("*.csv")} == names
    assert sorted(read_preds(run / "preds" / "b1.csv")["lesion_id"].tolist()) == list(range(30))
    meta = json.loads((run / "run.json").read_text())
    assert meta["bgeo_best"] in ("bgeo_lr", "bgeo_hgb", "bgeo_mlp") and meta["labels"] == "C1" and set(meta["records"]) == names
    assert (run / "models" / "b1_fold0.pt").exists()
    with pytest.raises(FileExistsError):
        run_main(["--table", str(tdir), "--out", str(run), "--labels", "C1", "--device", "cpu", "--epochs", "1"])

    out = tmp_path / "eval_c1"
    eval_main(["--run", str(run), "--table", str(tdir), "--labels", "C1", "--out", str(out), "--n-boot", "50"])
    report = (out / "REPORT.md").read_text()
    for needle in ("NOT_EVIDENCE", "Gate R1", "agreement with C1", "tie", "bgeo_", "## Strata", "delta_d"):
        assert needle in report
    rows = list(csv.DictReader(open(out / "tables.csv")))
    assert {r["arm"] for r in rows} >= names and (out / "output.txt").exists()

    final = tmp_path / "final.csv"
    with open(final, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
        w.writeheader()
        for r in t.rows:
            w.writerow({"lesion_id": r["lesion_id"], "status": "agreed", "primary_host": r["c1_class"],
                        "acceptable_hosts": json.dumps([r["c1_class"]]), "not_a_lesion": r["lesion_id"] == 0,
                        "lesion_type": None, "side": None, "lobe": None})
    seal(final, {r["lesion_id"]: r["fold"] for r in t.rows}, tmp_path / "sealed", tmp_path / "manifest.json", k=5, now="2026-10-01T00:00:00+00:00")
    with pytest.raises(SystemExit):
        eval_main(["--run", str(run), "--table", str(tdir), "--labels", "R", "--sealed-dir", str(tmp_path / "sealed"),
                   "--manifest", str(tmp_path / "manifest.json"), "--out", str(tmp_path / "eval_r")])
    eval_main(["--run", str(run), "--table", str(tdir), "--labels", "R", "--unblind", "--sealed-dir", str(tmp_path / "sealed"),
               "--manifest", str(tmp_path / "manifest.json"), "--out", str(tmp_path / "eval_r"), "--n-boot", "50"])
    report_r = (tmp_path / "eval_r" / "REPORT.md").read_text()
    assert "NOT_EVIDENCE" not in report_r and "excluded (not_a_lesion): 1" in report_r
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_scripts.py -q -p no:cacheprovider`
Expected: FileNotFoundError while loading `scripts/run_relation_baselines.py` (the scripts do not exist yet; `scripts/` is not a package and gets no `__init__.py`).

- [ ] **Step 3: Write `scripts/run_relation_baselines.py`**

```python
#!/usr/bin/env python
# scripts/run_relation_baselines.py
"""Outer five-fold / inner-selection runs of the five relation arms on one feature table (spec 2026-09-27 §5–6.1).

  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/run_relation_baselines.py \
      --table /data2/congcong/data/FM_data/derived/relation/v1 --out runs/relation/c1_stage1 --labels C1 --device cuda

Stage 2 (P13): --labels R --sealed-dir ... --manifest ... --init-from runs/relation/c1_stage1
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.level_r.admin import sha256_file  # noqa: E402
from anatobind.relation.baselines import GEO_KINDS, PRIOR_VARIANTS  # noqa: E402
from anatobind.relation.cv import run_b0, run_bgeo, run_bprior, select_config, write_preds  # noqa: E402
from anatobind.relation.labels import c1_labels, r_train_labels  # noqa: E402
from anatobind.relation.table import load_table  # noqa: E402

ALL_ARMS = ("b0", "bprior", "bgeo", "b1", "b2")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", choices=("C1", "R"), required=True)
    ap.add_argument("--sealed-dir", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16, help="cap on BLAS / OpenMP / torch threads (CLAUDE.md: <= 48)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner-k", type=int, default=5)
    ap.add_argument("--init-from", type=Path, help="a previous run directory whose models/<arm>_fold<k>.pt initialise B1/B2 (stage 2)")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists; runs are never overwritten")
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=a.threads)                  # numpy BLAS and scikit-learn OpenMP pools for the rest of the run
    arms = a.arms.split(",")
    table, patches = load_table(a.table)
    if a.labels == "C1":
        labels = c1_labels(table)
    else:
        if not (a.sealed_dir and a.manifest):
            ap.error("--labels R needs --sealed-dir and --manifest")
        labels = lambda k: r_train_labels(k, a.sealed_dir, a.manifest)[0]      # noqa: E731  training folds only; never the test fold
    (a.out / "preds").mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run = {"table": str(a.table), "table_manifest_sha256": sha256_file(a.table / "manifest.json"), "labels": a.labels,
           "seed": a.seed, "epochs": a.epochs, "patience": a.patience, "inner_k": a.inner_k, "device": a.device, "threads": a.threads,
           "init_from": str(a.init_from) if a.init_from else None, "git_commit": commit, "records": {}}

    def save(name, preds, record):
        write_preds(a.out / "preds" / f"{name}.csv", preds)
        run["records"][name] = record
        print(f"{name}: written", flush=True)

    if "b0" in arms:
        save("b0", *run_b0(table))
    if "bprior" in arms:
        for v in PRIOR_VARIANTS:
            save(f"bprior_{v}", *run_bprior(table, labels, v))
    if "bgeo" in arms:
        scores = []
        for kind in GEO_KINDS:
            preds, rec = run_bgeo(table, labels, kind, a.seed, a.inner_k)
            save(f"bgeo_{kind}", preds, rec)
            scores.append((f"bgeo_{kind}", rec["mean_inner_score"]))
        best, sel = select_config(scores)
        run["bgeo_best"], run["bgeo_selection"] = best, sel
    torch_arms = [x for x in ("b1", "b2") if x in arms]
    if torch_arms:
        import torch
        from anatobind.relation.train import run_torch_arm
        torch.set_num_threads(a.threads)
        (a.out / "models").mkdir(exist_ok=True)
        for arm in torch_arms:
            init = None
            if a.init_from:                                  # stage 2: each fold fine-tunes its stage-1 model and config
                prev = json.loads((a.init_from / "run.json").read_text(encoding="utf-8"))["records"][arm]["folds"]
                init = {k: (prev[str(k)]["chosen"], torch.load(a.init_from / "models" / f"{arm}_fold{k}.pt", map_location="cpu", weights_only=True))
                        for k in sorted(set(table.folds().tolist()))}
            preds, rec, states = run_torch_arm(arm, table, patches, labels, seed=a.seed, device=a.device, epochs=a.epochs,
                                               patience=a.patience, inner_k=a.inner_k, init=init,
                                               log=lambda m: print(m, flush=True))
            for k, st in states.items():
                torch.save(st, a.out / "models" / f"{arm}_fold{k}.pt")
            save(arm, preds, rec)
    (a.out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(f"run written to {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `scripts/eval_relation_baselines.py`**

```python
#!/usr/bin/env python
# scripts/eval_relation_baselines.py
"""Score a run's out-of-fold predictions against a label source (spec 2026-09-27 §6.2–6.8).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_relation_baselines.py \
      --run runs/relation/c1_stage1 --table /data2/congcong/data/FM_data/derived/relation/v1 --labels C1 \
      --out docs/verification/<date>/relation_baselines

--labels R is the one-shot final evaluation: it needs --unblind, --sealed-dir and --manifest, and every access is logged
by the sealed loader.
"""
import argparse
import csv
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.relation_metrics import (  # noqa: E402
    by_stratum, delta_d_curve, gate_r1, is_correct, mcnemar, patient_bootstrap, rescue_harm, summary,
)
from anatobind.relation.baselines import PRIOR_VARIANTS  # noqa: E402
from anatobind.relation.cv import read_preds  # noqa: E402
from anatobind.relation.labels import acceptable_matrix, c1_labels, r_test_labels  # noqa: E402
from anatobind.relation.table import load_table  # noqa: E402

STRATA = ("band", "stratum_geometry", "stratum_series", "lesion_type", "is_3mm")


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def fmt(x):
    return "" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)


def evaluate(run_dir, table, labels, excluded, evidence, n_boot):
    run = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    has, acc = acceptable_matrix(labels, table.lesion_id)
    keep = np.nonzero(has)[0]
    arms = {}
    for p in sorted((Path(run_dir) / "preds").glob("*.csv")):
        pr = read_preds(p)
        order = [pr["lesion_id"].tolist().index(int(l)) for l in table.lesion_id]
        arms[p.stem] = pr["probs"][order][keep]
    acc, patients = acc[keep], table.patients()[keep]
    c1 = table.c1_slot()[keep]
    bgeo_best = run.get("bgeo_best") or next(a for a in arms if a.startswith("bgeo_"))
    # spec §6.3: the Bprior comparator is the best of the four variants on these labels (conservative for any model
    # compared with it); ties go to the simpler variant, in PRIOR_VARIANTS order
    prior_arms = [f"bprior_{v}" for v in PRIOR_VARIANTS if f"bprior_{v}" in arms]
    prior_best = max(prior_arms, key=lambda name: is_correct(arms[name], acc).mean())
    comparators = {"b0": arms["b0"], "bprior": arms[prior_best], "bgeo": arms[bgeo_best], "b2": arms.get("b2")}
    comparators = {k: v for k, v in comparators.items() if v is not None}
    correct = {a: is_correct(p, acc) for a, p in arms.items()}
    rows_long, report = [], []
    report.append(f"# Relation baselines report ({evidence})\n")
    if evidence == "NOT_EVIDENCE":
        report.append("**NOT_EVIDENCE**: every number below is scored against the pseudo label C1, which is a function of the "
                      "geometry columns the arms read; it proves the pipeline runs and is fair, not that any arm binds better.\n")
    report.append(f"run: {run_dir} (git {run.get('git_commit', '')}, table manifest {run.get('table_manifest_sha256', '')[:12]}, labels {run.get('labels')})\n")
    report.append(f"lesions scored: {len(keep)} of {len(table)}; excluded (not_a_lesion): {len(excluded)}\n")
    report.append("## Arms and metrics\n")
    rows = []
    for a, p in arms.items():
        s = summary(p, acc)
        agree = float((p.argmax(1) == c1).mean())
        rows.append([a, s["n"], fmt(s["accuracy"]), fmt(s["singleton_rate"]), fmt(s["top2_accuracy"]), fmt(s["singleton"]["accuracy"]),
                     fmt(s["singleton"]["macro_f1"]), fmt(s["singleton"]["balanced_accuracy"]), fmt(agree)])
        for m, v in (("accuracy", s["accuracy"]), ("singleton_rate", s["singleton_rate"]), ("top2_accuracy", s["top2_accuracy"]),
                     ("singleton_accuracy", s["singleton"]["accuracy"]), ("macro_f1", s["singleton"]["macro_f1"]),
                     ("balanced_accuracy", s["singleton"]["balanced_accuracy"]), ("agreement_with_c1", agree)):
            rows_long.append({"arm": a, "metric": m, "stratum": "all", "value": v, "lo": "", "hi": ""})
    report.append(md_table(["arm", "n", "accuracy", "singleton rate", "top-2", "singleton acc", "macro-F1", "balanced acc", "agreement with C1"], rows) + "\n")
    report.append(f"Bgeo+ comparator: {bgeo_best} (selection {json.dumps(run.get('bgeo_selection', {}))}); "
                  f"Bprior comparator: {prior_best} (best of the four variants on these labels)\n")
    report.append("## Rescue / harm against the comparators (patient bootstrap 95% CI, exact McNemar)\n")
    rows = []
    for a in arms:
        if a in ("b0",) or a.startswith("bprior"):
            continue
        for cname, cp in comparators.items():
            if cname == "b2" and a == "b2":
                continue
            rh = rescue_harm(correct[a], is_correct(cp, acc))
            diff = correct[a].astype(float) - is_correct(cp, acc).astype(float)
            mean, lo, hi = patient_bootstrap(diff, patients, n_boot)
            pv = mcnemar(rh["rescue"], rh["harm"])
            rows.append([a, cname, rh["rescue"], rh["harm"], rh["net"], fmt(mean), fmt(lo), fmt(hi), fmt(pv)])
            rows_long.append({"arm": a, "metric": f"net_rescue_vs_{cname}", "stratum": "all", "value": mean, "lo": lo, "hi": hi})
    report.append(md_table(["arm", "comparator", "rescue", "harm", "net", "net rate", "CI lo", "CI hi", "McNemar p"], rows) + "\n")
    report.append("## Strata (set-valued accuracy)\n")
    for key in STRATA:
        keys = table.column(key)[keep].astype(str)
        rows = []
        for a in arms:
            st = by_stratum(correct[a].astype(float), keys)
            rows.append([a] + [f"{st[s]['mean']:.4f} (n={st[s]['n']})" for s in sorted(st)])
            for s in st:
                rows_long.append({"arm": a, "metric": "accuracy", "stratum": f"{key}={s}", "value": st[s]["mean"], "lo": "", "hi": ""})
        report.append(f"### {key}\n" + md_table(["arm"] + sorted(set(keys.tolist())), rows) + "\n")
    report.append("## Net rescue vs Bgeo+ along delta_d (patient bootstrap)\n")
    dd = table.column("delta_d_mm")[keep].astype(float)
    for a in ("b1", "b2"):
        if a not in arms:
            continue
        diff = correct[a].astype(float) - is_correct(comparators["bgeo"], acc).astype(float)
        curve = delta_d_curve(diff, dd, patients, n_boot=n_boot)
        report.append(f"### {a}\n" + md_table(["bin", "n", "mean", "lo", "hi"], [[c["bin"], c["n"], fmt(c["mean"]), fmt(c["lo"]), fmt(c["hi"])] for c in curve]) + "\n")
        for c in curve:
            rows_long.append({"arm": a, "metric": "net_rescue_vs_bgeo", "stratum": f"delta_d={c['bin']}", "value": c["mean"], "lo": c["lo"], "hi": c["hi"]})
    report.append("## Gate R1 rehearsal (v2.6 §12.4)\n")
    for a in ("b1", "b2"):
        if a in arms and "b2" in comparators:
            g = gate_r1(arms[a], {"bprior": comparators["bprior"], "bgeo": comparators["bgeo"], "b2": comparators["b2"]}, acc, patients, n_boot)
            report.append(f"- {a}: go={g['go']} (net rescue vs Bgeo+ {g['net_rescue_vs_bgeo']['mean']:.4f} [{g['net_rescue_vs_bgeo']['lo']:.4f}, "
                          f"{g['net_rescue_vs_bgeo']['hi']:.4f}]; > Bprior {g['gt_bprior']}, > Bgeo+ {g['gt_bgeo']}, > B2 {g['gt_b2']})")
    report.append("\n## Sanity checks\n")
    report.append("- agreement with C1 per arm is in the metrics table: Bgeo+ close to 1 and B2 clearly lower is the written evidence that C1 is a function of the geometry columns.")
    ties = []
    for name, rec in run.get("records", {}).items():
        for k, f in rec.get("folds", {}).items():
            if f.get("tie"):
                ties.append(f"{name} fold {k}: chose {f['chosen']} among {f['scores']} (tie rule P11)")
    report.append("- inner-selection ties: " + (f"{len(ties)}\n  - " + "\n  - ".join(ties) if ties else "none"))
    report.append("\n## Commands\n")
    report.append("```\nscripts/run_relation_baselines.py (see run.json)\nscripts/eval_relation_baselines.py " + " ".join(sys.argv[1:]) + "\n```\n")
    return "\n".join(report), rows_long


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--labels", choices=("C1", "R"), required=True)
    ap.add_argument("--unblind", action="store_true")
    ap.add_argument("--sealed-dir", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    table, _ = load_table(a.table, with_patches=False)
    if a.labels == "C1":
        labels, excluded, evidence = c1_labels(table), [], "NOT_EVIDENCE"
    else:
        if not a.unblind or not (a.sealed_dir and a.manifest):
            sys.exit("--labels R is the one-shot final evaluation: it needs --unblind, --sealed-dir and --manifest (v2.6 §12.7)")
        labels, excluded = r_test_labels(a.sealed_dir, a.manifest, unblind=True)
        evidence = "R"
    buf = io.StringIO()
    with redirect_stdout(buf):
        report, rows_long = evaluate(a.run, table, labels, excluded, evidence, a.n_boot)
        print(report)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "REPORT.md").write_text(report, encoding="utf-8")
    with open(a.out / "tables.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "metric", "stratum", "value", "lo", "hi"])
        w.writeheader()
        w.writerows(rows_long)
    (a.out / "output.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the end-to-end test, then the whole suite**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relation_scripts.py -q -p no:cacheprovider`
Expected: pass (about a minute on CPU: ten arms on 30 lesions, one epoch).

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`
Expected: all pass (612 + the new relation tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_relation_baselines.py scripts/eval_relation_baselines.py tests/test_relation_scripts.py
git commit -m "Relation scripts: run the five arms to out-of-fold prediction files and score them against C1 or the sealed labels into a report"
```

---

### Task 15: The real table, the stage-1 run, the report and the documents

**Files:**
- Create: `docs/verification/<YYYY-MM-DD>/relation_table.md`, `docs/verification/<YYYY-MM-DD>/relation_baselines/{REPORT.md, tables.csv, output.txt}` (the date is the day the commands run)
- Modify: `CLAUDE.md` (code map + status bullet + test count), `docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md` (§25 item 11), `STATUS.md` (rewritten at the handoff by the controller, from the numbers below)

**Interfaces:** consumes everything above; produces the S1 acceptance evidence (spec §9).

Every step below uses the same verification directory. Set it once from the date of Step 1 and set it again, to the same date, in any new shell: `D=docs/verification/<YYYY-MM-DD>`.

- [ ] **Step 1: Build the real table (about 6 minutes, `nice 19`)**

```bash
D=docs/verification/$(date +%F); mkdir -p $D
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/build_relation_table.py \
    --out /data2/congcong/data/FM_data/derived/relation/v1 | tee $D/relation_table_output.txt
```
Expected, as measured when this plan was validated (2026-09-27, the plan's own code run in a throwaway copy of the branch; not a deliverable, re-run and record your own output): `n_lesions 1297`, `n_patients 165`, five folds of 33 patients with 280/250/276/334/157 lesions (the Level R fold table), `c1_distribution` white_matter 985 / cortex 310 / basal_ganglia 2, `c1_source_counts` overlap 1278 / nearest 19, `n_candidate_fallback` 1 (lesion 1021, nearest class 17.6 mm away), `c1_agreement` 0.99537 with mismatches [64, 147, 327, 836, 971, 1086] (all white matter / cortex boundary lesions whose class-level argmax differs from the label-level one), wall clock 6 min. Any difference from these numbers is a finding: stop and report it before going on. If `c1_agreement` < 0.99 the build refuses: stop, report the mismatch list to the user, do not lower the threshold.

- [ ] **Step 2: Write `docs/verification/<date>/relation_table.md`**

Contents, in this order: the command; the manifest JSON as printed (counts, per_fold, c1_distribution, c1_source_counts, n_candidate_fallback, parameters, git commit, registry sha256); the C1 agreement fraction and the full mismatch list with one line per mismatch giving `lesion_id`, registry lookup class, C1 class and `c1_source`; the elapsed time; the sentence "每个数字来自上面的命令输出；C1 与注册表标签级查表的差异逐条列出（P6）。" Nothing in this file is a claim about binding quality.

- [ ] **Step 3: Run the five arms on C1 (CPU, 16 threads)**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/run_relation_baselines.py \
    --table /data2/congcong/data/FM_data/derived/relation/v1 --out runs/relation/c1_stage1 --labels C1 --device cpu --threads 16 \
    2>&1 | tee $D/relation_run_output.txt
```
Expected: ten `<arm>: written` lines, five per-fold lines for each of B1 and B2, and `run.json`. At plan validation B0, the four Bprior variants and Bgeo+ LR took about a minute, Bgeo+ HGB and MLP about six; B1 and B2 then run 5 × (4 configs × 5 inner folds + 1 refit) fits at 0.63 s and 0.24 s per epoch, with early stopping, so the whole run should take 30–60 minutes. If the first B1 fold line has not appeared after 20 minutes, or the run passes two hours, stop it and tell the user; do not change epochs, configs or the device on your own.

- [ ] **Step 4: Score against C1**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/eval_relation_baselines.py \
    --run runs/relation/c1_stage1 --table /data2/congcong/data/FM_data/derived/relation/v1 --labels C1 --out $D/relation_baselines
```
Expected: `REPORT.md` headed `(NOT_EVIDENCE)`; B0 accuracy 1.0000 by construction; Bgeo+ agreement with C1 near 1; B2 lower; ties listed. Read the report once and check that every table has ten arms and that no cell is empty; append nothing to it by hand.

- [ ] **Step 5: `CLAUDE.md`**

Add to the code map (after the `anatobind/level_r/` paragraph):

```
- `anatobind/relation/`（PR-C，脑侧关系基线 S1）：`table.py`（特征表列定义、`Table`、槽级 / 病灶级特征块、候选掩蔽、读写）、`build.py`（建表：逐病灶几何、类级 C1、小块、§4.7 检查、manifest）、`labels.py`（标签源 C1 / R）、`baselines.py`（B0、Bprior 四变体、Bgeo+ 三学习器）、`encoder.py`（E_u、裁剪、翻转一致增广）、`models.py`（B1 / B2、集合损失）、`cv.py`（折、平局规则、折外预测文件、sklearn 臂驱动）、`train.py`（torch 训练循环、阶段 2 钩子）；`anatobind/eval/relation_metrics.py`（集合值正确性、rescue / harm、患者 bootstrap、McNemar、分层、Δd 曲线、Gate R1）；`geometry.py` 加 16 维对量与侧别，`lookup.py` 加类级 C1。脚本 `scripts/build_relation_table.py`、`run_relation_baselines.py`、`eval_relation_baselines.py`。特征表在 `derived/relation/v1/`，运行在 `runs/relation/`（不入库）。规格 `docs/superpowers/specs/2026-09-27-relation-baselines-design.md`，计划 `docs/superpowers/plans/2026-09-27-relation-baselines.md`。
```

Replace the test-count clause in the 测试 line with the number the full suite prints after Task 14, and replace the **当前状态** bullet with one sentence per the handoff: S1 implemented, stage-1 run scored against C1 (NOT_EVIDENCE), waiting for S2 and the doctors' labels; point to `STATUS.md` and the spec.

- [ ] **Step 6: v2.6 §25 item 11 (append after item 10, verbatim)**

```
11. **训练先于读片与训练期六个子项目（2026-09-27，用户拍板）**：全部学习方法在医生标签存在之前冻结（§12.7 封存与 §12.8 冻结自动满足）；医生阶段分两轮：pilot 150 标签判 §10.1 升级规则 → 全集读完封存 → 触发则用外层训练折的医生标签微调（Bgeo+ 同步重拟合）→ 封存折上终测一次。理由：伪标签 C1 是 §11 几何量的确定函数，只用伪标签训练的模型约等于 B0，不留第二轮 Gate R1 按构造 NO-GO。训练池 = Gate 0.5 注册表的 1297 个病灶（不含无小病灶标注的卷）。疾病层 D 定义为整个检查一个印象，真值为 fastMRI+ 研究级标签（FLAIR：正常 94 / 小血管病 150），依赖脑侧检测器。训练期子项目顺序：S1 PR-C（`docs/superpowers/specs/2026-09-27-relation-baselines-design.md`）→ S2 脑侧小病灶检测器 → S3 PR-D → S4 脑侧解剖层（自有 FLAIR 解剖分割器、皮层分区 → 脑叶、A_local_quality 作 SynthSeg 验证）→ S5 疾病印象与整句拼装 → S6 膝侧 nnDetection。偏离：§17 的 Bprior / Bgeo+ 不写进 `lookup.py` 而进新包 `anatobind/relation/`；§12.8 中期分析不实现，S3 前决定；阶段 1 内层平局按规格 P11 记录。
```

- [ ] **Step 7: Run the whole suite one last time, commit the documents and the verification files**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`
Expected: all pass; record the count for CLAUDE.md.

```bash
git add docs/verification/$(date +%F)/relation_table.md docs/verification/$(date +%F)/relation_table_output.txt \
        docs/verification/$(date +%F)/relation_run_output.txt docs/verification/$(date +%F)/relation_baselines \
        CLAUDE.md docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md
git commit -m "Relation baselines S1: real feature table, stage-1 run scored against C1 (NOT_EVIDENCE), v2.6 decision 11 and the CLAUDE code map"
```

`STATUS.md` is rewritten by the controller at the session boundary (five sections: verified numbers with commands from the two output files; decisions pending — the S2 spec, reader recruitment, push; next steps — S2 spec, S3 after S1's report is read; pitfalls — `derived/relation/v1` is never overwritten, `runs/` is untracked, C1 numbers are NOT_EVIDENCE; why — P1–P16), then `git merge --no-ff build/relation-baselines` into main and `git tag handoff/<date>-relation-baselines`. No push unless the user says so.

---

## Self-review notes (done while writing; kept for the executor)

- Spec coverage: §4.1–4.7 → Tasks 1–5; §5.1–5.3 → Tasks 8–9; §5.4–5.7 → Tasks 10–11, 13; §5.8 → Tasks 12–13; §5.9 → Task 13 (`init`, `trainable_rows`) and Task 14 (`--init-from`, `--labels R`); §6.1 → Task 12 (`write_preds`); §6.2 → Tasks 6, 14; §6.3–6.7 → Tasks 7, 14; §6.8 → Task 14; §7 → every task's tests; §8–§9 → Task 15 and the controller's handoff.
- Type consistency: `slot_features` returns class ids 1..7 keyed dicts; `Table.slot_block` orders slots by `SLOTS` (= `CLASS_NAMES`) so slot index = class id − 1 everywhere (`c1_slot = c1 − 1` in Task 5). `features_per_slot` is (N, 7, 26) with `[candidate, 8 pair fields, 8 geometry, 4 extras, 3 side, 2 type]`; `DX_INDEX = 1`, `SIDE_LEFT = 21`, `SIDE_RIGHT = 22` (Task 10) follow from that layout. Predictions are always (N, 8) with column 7 = none; `acceptable` is (N, 7) except `acceptable8` (N, 8) for the loss.
- The `mcnemar` expectation in Task 7 (`2 * 0.5 ** 10`) is scipy's two-sided exact binomial p for 0 successes in 10 trials at p = 0.5.
- Validation (2026-09-27, before execution): every code block of this plan was extracted into a throwaway copy of the branch (`git archive` of b93616d, scratchpad, not a deliverable). The full suite passed there, 677 = 612 existing + 65 new. On real data the build script made all 1297 rows in 6 min and passed every §4.7 check (agreement 0.99537; the counts are in Task 15 Step 1). A CPU rehearsal of Task 15 Steps 3–4 with `--epochs 2` ran all ten arms and the report in 8 min 10 s: B0, Bprior and Bgeo+ LR about 1 min, Bgeo+ HGB and MLP about 6 min, B1 about 13 s and B2 about 10 s per outer fold at 2 epochs. The rehearsal report showed the expected sanity-check direction (NOT_EVIDENCE, 2 epochs): agreement with C1 was Bgeo+ HGB 0.996 and B2 0.820. Nothing from the throwaway copy goes into the repository; Task 15 re-runs everything.
