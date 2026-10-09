# AnatoBind Brain A/U/R — Part 1 (data, targets, model, losses, probes) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build everything the spec needs before a training run can start: the sample table with per-sequence supervision flags, the lesion-instance and host targets, crops with validity and physical coordinates, the variable-size 3D Swin backbone with RoPE and M_valid, the entity / event / sequence / mask heads, the per-lesion host competition, the point-sampled losses, the crop dataset, the preparation and probe scripts, and the pre-flight records (P0). Training, evaluation and inference are Part 2 (a separate plan written after this one's probes).

**Architecture:** One package `anatobind/aur/` on top of the existing `anatobind.anatomy`, `anatobind.eval` and `anatobind.nnunet.brain_disease` helpers. Volumes are read as (x, y, z) and turned to (z, y, x); crops never resize, they pad and carry a validity mask, physical coordinates (mm) and local coordinates ([-1, 1]). The model is the plan's Swin (MONAI's window utilities, our own attention with RoPE and key masking, zeroed invalid tokens), two query decoders (32 identity entities, 64 anonymous events), a mask head evaluated at sampled points in training, an S head, and a 13 + 1 candidate competition per lesion. The relation truth is the mask overlap of the SynthSeg map (pseudo target).

**Tech Stack:** Python 3.11 in `~/anaconda3/envs/nvgen` (torch 2.5.1+cu121, MONAI 1.5.2, nibabel, scipy, numpy, einops, matplotlib), pytest. `torchrun` for the four-card probe.

**Spec:** `docs/superpowers/specs/2026-10-08-anatobind-brain-aur-design.md` (decisions N1–N18; the plan argues from it).

**Validated source:** every code block below was run and tested in the controller's dry-run copy before it entered this plan (36 tests passing; single-card and two-card probes run). Implementers transcribe the blocks verbatim; the controller byte-compares each file against the dry-run copy.

## Global Constraints

- **Nothing is ever deleted, renamed or overwritten** (spec N17): no `rm`, no file removal or renaming calls in Python or shell, no truncating writes to existing files. Every output directory or file is checked before it is created and refused when it exists; reruns take a new name. (A PreToolUse hook blocks shell commands whose text even mentions such calls.)
- Data is read only under `/data2/congcong/data/FM_data`; writes go under `/data2/congcong/data/FM_data/derived/aur/`, the repo `docs/` and `logs/` (gitignored). `/data0/congcong/data` is a cold backup: never read.
- Environment prefix for every Python command: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python`; CPU work at nice 19, at most 48 threads in total; GPUs: only idle cards (no compute process, under 1000 MiB), at most four, chosen at launch (N16); never signal a process this session did not start.
- Splits are by patient (N12): PDGM / BMSR / SibBMS from `/data2/congcong/data/FM_data/derived/brain_anatomy/cases.json`, ISLES by `anatobind.anatomy.sources.split_by_patient(seed=0)`; test patients never enter training.
- No mirroring anywhere (S4 lesson A17): the host classes have a side. Intensity augmentation and in-plane rotation of at most ±10° only.
- Label spaces (N2): entities = the 32 SynthSeg labels of `anatobind.eval.lookup.BRAIN_ALL` in that order; hosts = the 13 classes of `anatobind.anatomy.labels.HOST_IDS` in that order plus index 13 "no_host"; sequence types `("T1", "T1c", "T2", "FLAIR", "DWI", "ADC")`.
- Every number written into a record is copied from a file or a printed output; anything that compares two pseudo-label maps is labelled NOT_EVIDENCE (N18).
- Tests: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`; the full suite must stay green (872 passed, 1 skipped before this plan; + 36 after it).
- Commits: author is the repository's configured user (Congcong Liu), messages in English, no AI attribution lines of any kind; `git add` explicit paths only, never `-A` or `.` (untracked user files sit in the checkout).
- Work on `main` directly (the user's choice for S4; same here); the records folder is `docs/verification/2026-10-08/anatobind_brain_aur/`.

## Review Focus

1. A crop that overlaps the volume only partly (ISLES volumes are 112 × 112 × 73 voxels, smaller than the 128 × 160 × 160 crop in every axis): the padded voxels must be invalid, fill -1 in the image, 0 in the labels, never serve as attention keys and never leak into valid tokens — pinned in `tests/test_aur_crops.py` (extract), `tests/test_aur_swin.py` (no leak), `tests/test_aur_dataset.py` (crop larger than the volume).
2. A case whose SynthSeg map holds no host voxel at all, or a lesion without any host within 10 mm: the target must be `NO_HOST`, not an exception or a fabricated host — `tests/test_aur_targets.py`.
3. A crop with more lesion instances than event queries (BMSR cases reach 69 metastases): the Hungarian matching must assign the 64 queries and leave the rest unmatched without error — `tests/test_aur_losses.py`.
4. Points sampled for the mask losses that fall on lesion voxels (entity loss) or on components under the volume floor (event loss) must carry no loss — `tests/test_aur_losses.py`.
5. A sample whose U is not supervised (every SibBMS row, PDGM T1 / T2, BMSR T1pre / FLAIR, ISLES ADC) must add exactly nothing to L_U while still training A and S — `tests/test_aur_losses.py` and `tests/test_aur_dataset.py` (flags carried through collate).

## File Structure

| file | responsibility |
|---|---|
| `anatobind/aur/__init__.py` | empty |
| `anatobind/aur/labels.py` | entity / host / sequence label spaces and the maps between SynthSeg values and them |
| `anatobind/aur/samples.py` | the sample table (one row per case × sequence) with supervision flags and splits |
| `anatobind/aur/targets.py` | lesion instances, soft host distribution, primary host, hard negatives |
| `anatobind/aur/crops.py` | (z, y, x) order, crop windows, padding + validity, coordinates, normalisation, augmentation, rotation |
| `anatobind/aur/rope.py` | physical-coordinate 3D rotary embedding |
| `anatobind/aur/swin.py` | the variable-size Swin backbone (RoPE attention, key masking, merging, four levels) |
| `anatobind/aur/heads.py` | query decoders (entities, events), mask head, sequence head |
| `anatobind/aur/relation.py` | host grouping of entities, geometry features, candidate competition |
| `anatobind/aur/losses.py` | point sampling, entity / event / sequence / relation losses, the total |
| `anatobind/aur/model.py` | `AnatoBindBrain`: assembly, masks on demand, `bind` |
| `anatobind/aur/dataset.py` | volume loading, crops with targets, `AURDataset`, `collate`, event targets at points |
| `scripts/aur_prepare.py` | `--stage samples / grids / isles_check` |
| `scripts/aur_probe.py` | memory / speed probe, single card and `torchrun` DDP |
| `tests/test_aur_*.py` | one test file per module (`labels, samples, targets, crops, rope, swin, heads, relation, losses, model, dataset`), plus `test_aur_scripts.py` |
| `docs/verification/2026-10-08/anatobind_brain_aur/` | P0 records: sample table counts, grid check, ISLES check, SibBMS note, probes, README |

---
### Task 1: Label spaces

**Files:**
- Create: `anatobind/aur/__init__.py` (empty)
- Create: `anatobind/aur/labels.py`
- Test: `tests/test_aur_labels.py`

**Interfaces:**
- Consumes: `anatobind.anatomy.labels.{HOST_IDS, HOST_CLASS_OF, SIDE_OF, NAMES, to_student}`, `anatobind.eval.lookup.BRAIN_ALL`
- Produces: `ENTITY_LABELS` (tuple of 32 ints), `N_ENTITIES = 32`, `ENTITY_INDEX`, `N_HOSTS = 13`, `NO_HOST = 13`, `N_HOST_CLASSES = 14`, `HOST_NAMES` (14 names), `HOST_TISSUE`, `HOST_SIDE`, `TISSUES` (7), `SEQ_TYPES`, `SEQ_INDEX`, `NEAR_MM = 10.0`, `ENTITY_HOST` (32 ints, host index or -1), `entity_map(seg) -> uint8`, `entity_to_synthseg(emap) -> int16`, `host_map(seg) -> uint8`, `contralateral(host) -> int | None`

Create the empty `anatobind/aur/__init__.py` with `: > anatobind/aur/__init__.py` (the folder does not exist yet: `mkdir -p anatobind/aur` first) and add it to the commit.

- [ ] **Step 1: Write the failing test** `tests/test_aur_labels.py`

```python
# tests/test_aur_labels.py
import numpy as np
import pytest

import anatobind.aur.labels as L


def test_entity_map_round_trips_the_32_structures_and_drops_everything_else():
    assert L.N_ENTITIES == 32 and len(set(L.ENTITY_LABELS)) == 32
    seg = np.array(L.ENTITY_LABELS + (0, 77, 255), np.int16).reshape(5, 7)
    e = L.entity_map(seg)
    assert e.dtype == np.uint8 and list(e.ravel()[:32]) == list(range(1, 33)) and list(e.ravel()[32:]) == [0, 0, 0]
    back = L.entity_to_synthseg(e)
    assert back.dtype == np.int16 and list(back.ravel()[:32]) == list(L.ENTITY_LABELS) and list(back.ravel()[32:]) == [0, 0, 0]
    with pytest.raises(ValueError, match="outside"):
        L.entity_map(np.array([300]))
    with pytest.raises(ValueError, match="outside"):
        L.entity_to_synthseg(np.array([33]))


def test_host_map_follows_the_s4_host_classes_and_leaves_landmarks_out():
    assert L.N_HOSTS == 13 and L.NO_HOST == 13 and L.N_HOST_CLASSES == 14 and L.HOST_NAMES[-1] == "no_host"
    seg = np.array([2, 41, 4, 24, 16, 11, 50, 0, 7, 46], np.int16)
    h = L.host_map(seg)
    names = [L.HOST_NAMES[v - 1] if v else "none" for v in h]
    assert names == ["white_matter_left", "white_matter_right", "none", "none", "brainstem", "basal_ganglia_left",
                     "basal_ganglia_right", "none", "cerebellum_left", "cerebellum_right"]


def test_contralateral_is_an_involution_and_the_brainstem_has_none():
    for h in range(L.N_HOSTS):
        c = L.contralateral(h)
        if L.HOST_SIDE[h] is None:
            assert c is None and L.HOST_NAMES[h] == "brainstem"
        else:
            assert c != h and L.contralateral(c) == h and L.HOST_TISSUE[c] == L.HOST_TISSUE[h]
    assert len(L.TISSUES) == 7


def test_entity_host_mapping_and_sequence_types():
    assert L.ENTITY_HOST[L.ENTITY_INDEX[11]] == L.HOST_NAMES.index("basal_ganglia_left")
    assert L.ENTITY_HOST[L.ENTITY_INDEX[4]] == -1 and L.ENTITY_HOST[L.ENTITY_INDEX[24]] == -1
    assert L.ENTITY_HOST[L.ENTITY_INDEX[16]] == L.HOST_NAMES.index("brainstem")
    assert L.SEQ_TYPES == ("T1", "T1c", "T2", "FLAIR", "DWI", "ADC") and L.SEQ_INDEX["DWI"] == 4
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_labels.py -q -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError / ImportError / AttributeError for the new names).

- [ ] **Step 3: Write `anatobind/aur/labels.py`**

```python
"""Label spaces of the AnatoBind brain model (spec 2026-10-08 N2, §4).

A: one entity per SynthSeg structure of anatobind.eval.lookup.BRAIN_ALL (32, in that tuple's order); entity map value
i + 1 stands for ENTITY_LABELS[i], 0 for none. R: the 13 sided host classes of the S4 label space
(anatobind.anatomy.labels.HOST_IDS, in that order) as host indices 0..12, plus NO_HOST = 13 for a lesion without a
parenchymal host within NEAR_MM; ventricles and CSF are landmarks, never hosts (V7 §5). S: six sequence types.
Everything derived from SynthSeg is a pseudo-label: NOT_EVIDENCE."""
import numpy as np

from anatobind.anatomy.labels import HOST_CLASS_OF, HOST_IDS, NAMES, SIDE_OF, to_student
from anatobind.eval.lookup import BRAIN_ALL

ENTITY_LABELS = tuple(int(l) for l in BRAIN_ALL)      # SynthSeg values; entity index = position in this tuple
N_ENTITIES = len(ENTITY_LABELS)                       # 32
ENTITY_INDEX = {label: i for i, label in enumerate(ENTITY_LABELS)}
N_HOSTS = len(HOST_IDS)                               # 13
NO_HOST = N_HOSTS                                     # host index 13
N_HOST_CLASSES = N_HOSTS + 1
HOST_NAMES = tuple(NAMES[c] for c in HOST_IDS) + ("no_host",)
HOST_TISSUE = tuple(HOST_CLASS_OF[c] for c in HOST_IDS)        # tissue family of each host index
HOST_SIDE = tuple(SIDE_OF[c] for c in HOST_IDS)                # "left" / "right" / None (brainstem)
TISSUES = tuple(dict.fromkeys(HOST_TISSUE))                    # the 7 tissue families, first-seen order
SEQ_TYPES = ("T1", "T1c", "T2", "FLAIR", "DWI", "ADC")
SEQ_INDEX = {s: i for i, s in enumerate(SEQ_TYPES)}
NEAR_MM = 10.0                                        # a lesion that overlaps no host takes the nearest one within this

_ENTITY_LUT = np.zeros(256, np.uint8)
for _i, _l in enumerate(ENTITY_LABELS):
    _ENTITY_LUT[_l] = _i + 1
_HOST_LUT = np.zeros(16, np.uint8)                    # S4 student id -> host index + 1 (ventricles 14 -> 0)
for _i, _c in enumerate(HOST_IDS):
    _HOST_LUT[_c] = _i + 1
ENTITY_HOST = tuple(int(_HOST_LUT[to_student(np.array([l]))[0]]) - 1 for l in ENTITY_LABELS)   # host index or -1


def entity_map(seg):
    """SynthSeg map -> uint8 entity map (0 none, i + 1 for ENTITY_LABELS[i])."""
    s = np.asarray(seg)
    if s.min() < 0 or s.max() > 255:
        raise ValueError(f"label values outside 0..255: {int(s.min())}..{int(s.max())}")
    return _ENTITY_LUT[s.astype(np.int64)]


def entity_to_synthseg(emap):
    """Entity map -> SynthSeg values (int16); 0 stays 0."""
    e = np.asarray(emap)
    if e.min() < 0 or e.max() > N_ENTITIES:
        raise ValueError(f"entity values outside 0..{N_ENTITIES}: {int(e.min())}..{int(e.max())}")
    lut = np.zeros(N_ENTITIES + 1, np.int16)
    lut[1:] = ENTITY_LABELS
    return lut[e.astype(np.int64)]


def host_map(seg):
    """SynthSeg map -> uint8 host map (0 none: background, ventricles, CSF, unlisted; i + 1 for host index i)."""
    return _HOST_LUT[to_student(seg).astype(np.int64)]


def contralateral(host):
    """Host index of the same tissue on the other side; None for the brainstem."""
    side = HOST_SIDE[host]
    if side is None:
        return None
    other = "right" if side == "left" else "left"
    return next(i for i in range(N_HOSTS) if HOST_TISSUE[i] == HOST_TISSUE[host] and HOST_SIDE[i] == other)
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_labels.py -q -p no:cacheprovider`
Expected: PASS, no warnings about the new code.

- [ ] **Step 5: Commit**

```bash
git add tests/test_aur_labels.py anatobind/aur/labels.py
git commit -m "AUR: label spaces of the brain model (32 entities, 13 + 1 hosts, 6 sequence types)"
```

---

### Task 2: Sample table

**Files:**
- Create: `anatobind/aur/samples.py`
- Test: `tests/test_aur_samples.py`

**Interfaces:**
- Consumes: `anatobind.anatomy.sources.{FM, sibbms_cases, split_by_patient, check_split}`, `anatobind.nnunet.brain_disease.{list_cases, channel_path, label_path, anatomy_path, patient_of}`
- Produces: `SEQUENCES`, `ALL_LESION_VALUES`, `DISEASE_OF`, `SOURCES`, `disease_samples(source, root)`, `sibbms_samples(root)`, `all_samples(root) -> list[dict]`, `assign_splits(rows, s4_cases, seed=0) -> list[dict]` (adds `split`), `counts(rows)`, `write_samples(path, rows)` (refuses an existing path), `read_samples(path)`
- Row keys: `case, source, patient, sequence, source_sequence, image, anatomy, lesion (str | None), u_supervised (bool), u_values (list), a_ignore_values (list), split`

- [ ] **Step 1: Write the failing test** `tests/test_aur_samples.py`

```python
# tests/test_aur_samples.py
import pytest

import anatobind.aur.samples as S


def _root(tmp_path):
    """The directory skeleton the four sources are listed from (files need not hold data; SibBMS needs them to exist)."""
    root = tmp_path / "FM"
    (root / "UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5/UCSF-PDGM-0004_nifti").mkdir(parents=True)
    (root / "UCSF-BMSR_cbb/UCSF-BMSR/UCSF_BrainMetastases_TRAIN/100101A").mkdir(parents=True)
    (root / "ISLES_ltr/ISLES-2022/sub-strokecase0001").mkdir(parents=True)
    anat = root / "SibBMS_ms/sibbms/Output/MS/sub-001/ses-001/anat"
    anat.mkdir(parents=True)
    for name in ("sub-001_ses-001_FLAIR.nii.gz", "sub-001_ses-001_T1w.nii.gz"):      # no T2w for this session
        (anat / name).write_bytes(b"")
    seg = root / "derived/synthseg/sibbms/seg_native"
    seg.mkdir(parents=True)
    (seg / "MS_sub-001_ses-001_T1w_seg.nii.gz").write_bytes(b"")
    return root


def test_rows_carry_the_supervision_of_each_sequence(tmp_path):
    rows = S.all_samples(_root(tmp_path))
    by = {(r["source"], r["source_sequence"]): r for r in rows}
    assert len(rows) == 4 + 3 + 2 + 2
    assert by[("pdgm", "FLAIR")]["u_supervised"] and by[("pdgm", "FLAIR")]["u_values"] == [1, 2, 4]
    assert by[("pdgm", "T1c")]["u_values"] == [1, 4] and not by[("pdgm", "T1")]["u_supervised"]
    assert by[("bmsr", "T1post")]["sequence"] == "T1c" and by[("bmsr", "T1post")]["u_values"] == [1]
    assert not by[("bmsr", "FLAIR")]["u_supervised"] and by[("bmsr", "FLAIR")]["a_ignore_values"] == [1]
    assert by[("isles", "DWI")]["u_values"] == [1] and by[("isles", "DWI")]["image"].endswith("sub-strokecase0001_ses-0001_dwi.nii.gz")
    assert not by[("isles", "ADC")]["u_supervised"]
    sib = [r for r in rows if r["source"] == "sibbms"]
    assert sorted(r["sequence"] for r in sib) == ["FLAIR", "T1"] and all(r["lesion"] is None and not r["u_supervised"] for r in sib)
    assert all(r["anatomy"].endswith("_seg.nii.gz") for r in rows)
    assert by[("pdgm", "T1")]["patient"] == "UCSF-PDGM-0004" and by[("isles", "DWI")]["patient"] == "sub-strokecase0001"


def test_splits_come_from_the_s4_table_and_isles_is_split_by_patient(tmp_path):
    rows = S.all_samples(_root(tmp_path))
    s4 = {"UCSF-PDGM-0004": {"split": "test"}, "100101A": {"split": "train"}, "MS_sub-001_ses-001": {"split": "train"}}
    out = S.assign_splits(rows, s4)
    split = {(r["source"], r["source_sequence"]): r["split"] for r in out}
    assert split[("pdgm", "T1")] == "test" and split[("pdgm", "FLAIR")] == "test" and split[("bmsr", "T1post")] == "train"
    assert split[("sibbms", "FLAIR")] == "train" and split[("isles", "DWI")] in ("train", "test")
    c = S.counts(out)
    assert c["pdgm"]["test"] == (1, 4, 2) and c["bmsr"]["train"] == (1, 3, 1) and c["sibbms"]["train"] == (1, 2, 0)
    with pytest.raises(KeyError, match="no split"):
        S.assign_splits(rows, {k: v for k, v in s4.items() if k != "100101A"})


def test_the_sample_table_is_written_once(tmp_path):
    p = tmp_path / "samples.json"
    S.write_samples(p, [{"case": "a"}])
    assert S.read_samples(p) == [{"case": "a"}]
    with pytest.raises(FileExistsError):
        S.write_samples(p, [])
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_samples.py -q -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError / ImportError / AttributeError for the new names).

- [ ] **Step 3: Write `anatobind/aur/samples.py`**

```python
"""The sample table: one row per (case, sequence) with its supervision flags and split (spec 2026-10-08 §3).

A row carries the image, the SynthSeg map on the same grid (A is always supervised), the source lesion map and the
label values that mean "lesion on this sequence" (U is supervised only where the lesion is visible, N3). Sources:
PDGM, BMSR, ISLES through anatobind.nnunet.brain_disease; SibBMS through anatobind.anatomy.sources (A and S only:
its lesions are not annotated on the template grid). Splits follow the S4 case table for PDGM / BMSR / SibBMS and
split_by_patient for ISLES (N12)."""
import json
from pathlib import Path

from anatobind.anatomy.sources import FM, check_split, sibbms_cases, split_by_patient
from anatobind.nnunet.brain_disease import anatomy_path, channel_path, label_path, list_cases, patient_of

# source -> {name of the sequence in the source: (sequence type, lesion values that are visible on it; () = A/S only)}
SEQUENCES = {
    "pdgm": {"T1": ("T1", ()), "T1c": ("T1c", (1, 4)), "T2": ("T2", ()), "FLAIR": ("FLAIR", (1, 2, 4))},
    "bmsr": {"T1pre": ("T1", ()), "T1post": ("T1c", (1,)), "FLAIR": ("FLAIR", ())},
    "isles": {"DWI": ("DWI", (1,)), "ADC": ("ADC", ())},
    "sibbms": {"T1w": ("T1", ()), "T2w": ("T2", ()), "FLAIR": ("FLAIR", ())},
}
ALL_LESION_VALUES = {"pdgm": (1, 2, 4), "bmsr": (1,), "isles": (1,), "sibbms": ()}     # every lesion voxel, any sequence
DISEASE_OF = {"pdgm": "glioma", "bmsr": "metastasis", "isles": "infarct"}
SOURCES = ("pdgm", "bmsr", "isles", "sibbms")


def _row(case, source, patient, seq_name, image, anatomy, lesion):
    seq_type, values = SEQUENCES[source][seq_name]
    return {"case": case, "source": source, "patient": patient, "sequence": seq_type, "source_sequence": seq_name,
            "image": str(image), "anatomy": str(anatomy), "lesion": None if lesion is None else str(lesion),
            "u_supervised": bool(values), "u_values": list(values), "a_ignore_values": list(ALL_LESION_VALUES[source])}


def disease_samples(source, root=FM):
    """Rows of PDGM / BMSR / ISLES: every sequence of every case."""
    d = DISEASE_OF[source]
    rows = []
    for case in list_cases(d, root):
        for seq_name in SEQUENCES[source]:
            rows.append(_row(case, source, patient_of(d, case), seq_name, channel_path(d, case, seq_name, root),
                             anatomy_path(d, case, root), label_path(d, case, root)))
    return rows


def sibbms_samples(root=FM):
    """Rows of SibBMS: the sequences present beside each session's FLAIR (T1w, T2w, FLAIR); no lesion map."""
    rows = []
    for case, r in sibbms_cases(root).items():
        flair = Path(r["flair"])
        stem = flair.name[:-len("_FLAIR.nii.gz")]
        for seq_name in SEQUENCES["sibbms"]:
            p = flair.parent / f"{stem}_{seq_name}.nii.gz"
            if p.is_file():
                rows.append(_row(case, "sibbms", r["patient"], seq_name, p, r["anatomy"], None))
    return rows


def all_samples(root=FM):
    rows = []
    for source in SOURCES:
        rows += sibbms_samples(root) if source == "sibbms" else disease_samples(source, root)
    return rows


def assign_splits(rows, s4_cases, seed=0):
    """Add "split" to every row. PDGM / BMSR / SibBMS cases take the split of the S4 case table (the same test
    patients as S4); ISLES cases are split by patient here. Every case must get a split; no patient may be in both."""
    split = {case: r["split"] for case, r in s4_cases.items()}
    isles = {r["case"]: {"source": "isles", "patient": r["patient"]} for r in rows if r["source"] == "isles"}
    if isles:
        split.update(split_by_patient(isles, seed=seed))
    out = []
    for r in rows:
        if r["case"] not in split:
            raise KeyError(f"{r['case']} ({r['source']}): no split for this case")
        out.append({**r, "split": split[r["case"]]})
    by_case = {r["case"]: r for r in out}
    check_split(by_case, {c: r["split"] for c, r in by_case.items()})
    return out


def counts(rows):
    """{source: {"train": (cases, rows, rows with U), "test": ...}} for the records."""
    out = {}
    for split in ("train", "test"):
        for source in SOURCES:
            sel = [r for r in rows if r["source"] == source and r["split"] == split]
            out.setdefault(source, {})[split] = (len({r["case"] for r in sel}), len(sel), sum(r["u_supervised"] for r in sel))
    return out


def write_samples(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; nothing is overwritten here, use a new name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))


def read_samples(path):
    return json.loads(Path(path).read_text())
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_samples.py -q -p no:cacheprovider`
Expected: PASS, no warnings about the new code.

- [ ] **Step 5: Commit**

```bash
git add tests/test_aur_samples.py anatobind/aur/samples.py
git commit -m "AUR: the sample table with per-sequence supervision flags and patient splits"
```

---

### Task 3: Training targets

**Files:**
- Create: `anatobind/aur/targets.py`
- Test: `tests/test_aur_targets.py`

**Interfaces:**
- Consumes: `anatobind.aur.labels.{N_HOSTS, N_HOST_CLASSES, NO_HOST, NEAR_MM, contralateral, host_map}`, `anatobind.eval.lesion_components.{components, component_rows}`
- Produces: `lesion_instances(lesion, values, voxel_mm3) -> (int32 instance map, bool small-component mask)`, `host_targets(inst, seg, spacing) -> {"probs": (N, 14) float32, "host": (N,) int64, "negatives": (N, 2) int64}`

- [ ] **Step 1: Write the failing test** `tests/test_aur_targets.py`

```python
# tests/test_aur_targets.py
import numpy as np

import anatobind.aur.targets as T
from anatobind.aur.labels import HOST_NAMES, NO_HOST


def _seg():
    """(40, 20, 6) 1 mm map: left white matter (2) at x < 20, right (41) at x >= 20, a ventricle (4) block, and a
    background corridor at y >= 16 that is more than 10 mm from any host."""
    seg = np.zeros((40, 20, 6), np.int16)
    seg[:20, :5, :] = 2
    seg[20:, :5, :] = 41
    seg[8:12, 5:8, :] = 4             # ventricle beside the left white matter (host within 10 mm)
    return seg


def test_lesion_instances_drop_tiny_components_but_mark_them():
    les = np.zeros((40, 20, 6), np.uint8)
    les[2:5, 1:4, 1:4] = 1            # 27 voxels
    les[30, 1, 1] = 1                 # 1 voxel: under 10 mm3 at 1 mm3 voxels
    les[10, 10, 5] = 2                # a value that is not lesion on this sequence
    inst, small = T.lesion_instances(les, (1,), 1.0)
    assert inst.dtype == np.int32 and inst.max() == 1 and (inst > 0).sum() == 27
    assert small.sum() == 1 and small[30, 1, 1] and not inst[10, 10, 5]
    inst0, small0 = T.lesion_instances(les, (), 1.0)
    assert inst0.max() == 0 and not small0.any()


def test_host_targets_overlap_nearest_and_none():
    seg = _seg()
    inst = np.zeros(seg.shape, np.int32)
    inst[2:6, 0:3, :] = 1                      # inside the left white matter
    inst[17:23, 0:3, 0:2] = 2                  # 3 columns left (18 voxels), 3 right (18): 50 / 50
    inst[9:11, 5:7, :] = 3                     # inside the ventricle, 1 mm from the left white matter
    inst[30:34, 17:20, :] = 4                  # background, 12+ mm from any host
    out = T.host_targets(inst, seg, (1.0, 1.0, 1.0))
    wl, wr = HOST_NAMES.index("white_matter_left"), HOST_NAMES.index("white_matter_right")
    assert out["probs"].shape == (4, 14) and out["host"].tolist()[0] == wl and out["probs"][0, wl] == 1.0
    assert out["probs"][1, wl] == 0.5 and out["probs"][1, wr] == 0.5 and out["host"][1] == wl    # tie: the lower index
    assert out["host"][2] == wl and out["probs"][2, wl] == 1.0                                   # nearest within 10 mm
    assert out["host"][3] == NO_HOST and out["probs"][3, NO_HOST] == 1.0
    assert out["negatives"][0].tolist() == [wr, -1]              # contralateral, no second host
    assert out["negatives"][1].tolist() == [wr, wr]              # the second-largest host is the contralateral one here
    assert out["negatives"][3].tolist() == [-1, -1]


def test_host_targets_without_instances_or_without_hosts():
    seg = _seg()
    out = T.host_targets(np.zeros(seg.shape, np.int32), seg, (1.0, 1.0, 1.0))
    assert out["probs"].shape == (0, 14) and out["host"].shape == (0,) and out["negatives"].shape == (0, 2)
    inst = np.zeros(seg.shape, np.int32)
    inst[0:3, 0:3, 0:3] = 1
    out = T.host_targets(inst, np.zeros(seg.shape, np.int16), (1.0, 1.0, 1.0))
    assert out["host"].tolist() == [NO_HOST]
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_targets.py -q -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError / ImportError / AttributeError for the new names).

- [ ] **Step 3: Write `anatobind/aur/targets.py`**

```python
"""Training targets from the pseudo-labels (spec 2026-10-08 §4.2–4.3): lesion instances, each instance's soft host
distribution, its primary host and its hard negatives. The host truth is the overlap of the instance with the host
classes of the SynthSeg map (plan §13); it is a pseudo target, NOT_EVIDENCE."""
import numpy as np
from scipy import ndimage

from anatobind.aur.labels import N_HOSTS, N_HOST_CLASSES, NEAR_MM, NO_HOST, contralateral, host_map
from anatobind.eval.lesion_components import component_rows, components


def lesion_instances(lesion, values, voxel_mm3):
    """lesion: the source label map; values: the label values that are lesion on this sequence.
    Returns (instance map int32 with ids 1..N, 0 elsewhere; boolean mask of the components under the volume floor).
    Components under MIN_MM3 are neither instances nor background: callers leave them out of the loss."""
    mask = np.isin(np.asarray(lesion), list(values)) if len(values) else np.zeros(np.shape(lesion), bool)
    comp, n = components(mask)
    inst = np.zeros(comp.shape, np.int32)
    small = np.zeros(comp.shape, bool)
    k = 0
    for r in component_rows(comp, n, voxel_mm3, "lesion"):
        sel = comp == r["component"]
        if r["ignore"]:
            small |= sel
        else:
            k += 1
            inst[sel] = k
    return inst, small


def host_targets(inst, seg, spacing):
    """{"probs": (N, 14) float32, "host": (N,) int64, "negatives": (N, 2) int64} for the instances 1..N of `inst`.
    probs: the share of the instance's voxels in each host class (N9); without any overlap the nearest host within
    NEAR_MM gets 1, else NO_HOST gets 1. host = argmax. negatives (N10): the contralateral host and the host with the
    second-largest share; -1 when there is none (brainstem has no contralateral; a single-host lesion has no second)."""
    hm = host_map(seg)
    n = int(inst.max()) if inst.size else 0
    probs = np.zeros((n, N_HOST_CLASSES), np.float32)
    host = np.full(n, NO_HOST, np.int64)
    neg = np.full((n, 2), -1, np.int64)
    nearest = None
    for k in range(1, n + 1):
        sel = inst == k
        counts = np.bincount(hm[sel].astype(np.int64), minlength=N_HOSTS + 1)[1:]
        if counts.sum() > 0:
            probs[k - 1, :N_HOSTS] = counts / counts.sum()
        else:
            if nearest is None:
                if (hm > 0).any():
                    dist, idx = ndimage.distance_transform_edt(hm == 0, sampling=spacing, return_indices=True)
                    nearest = (dist, hm[tuple(idx)])
                else:
                    nearest = (None, None)
            dist, lab = nearest
            if dist is not None and float(dist[sel].min()) <= NEAR_MM:
                j = int(np.argmin(dist[sel]))
                probs[k - 1, int(lab[sel][j]) - 1] = 1.0
            else:
                probs[k - 1, NO_HOST] = 1.0
        host[k - 1] = int(np.argmax(probs[k - 1]))
        if host[k - 1] != NO_HOST:
            c = contralateral(int(host[k - 1]))
            neg[k - 1, 0] = -1 if c is None else c
            order = np.argsort(-probs[k - 1, :N_HOSTS], kind="stable")
            second = int(order[1])
            neg[k - 1, 1] = second if probs[k - 1, second] > 0 else -1
    return {"probs": probs, "host": host, "negatives": neg}
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_targets.py -q -p no:cacheprovider`
Expected: PASS, no warnings about the new code.

- [ ] **Step 5: Commit**

```bash
git add tests/test_aur_targets.py anatobind/aur/targets.py
git commit -m "AUR: lesion instances and the soft host targets with hard negatives"
```

---

### Task 4: Crops, coordinates, normalisation, augmentation

**Files:**
- Create: `anatobind/aur/crops.py`
- Test: `tests/test_aur_crops.py`

**Interfaces:**
- Consumes: `anatobind.anatomy.simulate.bias_field`
- Produces: `CROP = (128, 160, 160)`, `PATCH = (2, 4, 4)`, `to_zyx(arr)`, `spacing_zyx(spacing)`, `crop_window(shape, crop, rng, centre=None) -> [(start, stop)] * 3`, `extract(arr, window, fill) -> (crop, valid)`, `coordinates_mm(window, spacing) -> (3, D, H, W) float32`, `local_coordinates(window, shape)`, `normalise(image)`, `augment(image, rng, p=0.8)`, `rotate_inplane(arrays, angle_deg, orders)`

- [ ] **Step 1: Write the failing test** `tests/test_aur_crops.py`

```python
# tests/test_aur_crops.py
import numpy as np

import anatobind.aur.crops as C


def test_zyx_order_and_spacing():
    a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    z = C.to_zyx(a)
    assert z.shape == (4, 3, 2) and z[1, 2, 0] == a[0, 2, 1] and z.flags["C_CONTIGUOUS"]
    assert C.spacing_zyx((0.5, 0.75, 5.0)) == (5.0, 0.75, 0.5)


def test_crop_window_inside_centred_and_pushed_back():
    rng = np.random.default_rng(0)
    w = C.crop_window((50, 60, 70), (16, 32, 32), rng)
    assert all(0 <= a and b <= n for (a, b), n in zip(w, (50, 60, 70))) and all(b - a == c for (a, b), c in zip(w, (16, 32, 32)))
    w = C.crop_window((10, 60, 70), (16, 32, 32), rng)
    assert w[0] == (-3, 13)                                   # smaller volume: the crop is centred on it
    w = C.crop_window((50, 60, 70), (16, 32, 32), rng, centre=(2, 59, 35))
    assert w[0] == (0, 16) and w[1] == (28, 60) and w[2] == (19, 51)   # pushed back inside where it fits


def test_extract_fills_the_outside_and_marks_validity():
    a = np.arange(5 * 6 * 7, dtype=np.int16).reshape(5, 6, 7)
    crop, valid = C.extract(a, [(-2, 4), (3, 9), (0, 7)], fill=-7)
    assert crop.shape == (6, 6, 7) and valid.shape == (6, 6, 7)
    assert crop[0, 0, 0] == -7 and not valid[0, 0, 0] and valid[2, 0, 0] and crop[2, 0, 0] == a[0, 3, 0]
    assert valid.sum() == 4 * 3 * 7 and crop[valid].tolist() == a[0:4, 3:6, :].ravel().tolist()
    crop, valid = C.extract(a, [(100, 104), (0, 6), (0, 7)])
    assert not valid.any() and (crop == 0).all()


def test_coordinates_physical_and_local():
    mm = C.coordinates_mm([(-1, 2), (0, 2), (4, 6)], (5.0, 1.0, 0.5))
    assert mm.shape == (3, 3, 2, 2) and mm[0, :, 0, 0].tolist() == [-2.5, 2.5, 7.5] and mm[2, 0, 0, :].tolist() == [2.25, 2.75]
    loc = C.local_coordinates([(0, 4), (0, 2), (2, 4)], (4, 2, 4))
    assert np.allclose(loc[0, :, 0, 0], [-0.75, -0.25, 0.25, 0.75]) and np.allclose(loc[2, 0, 0, :], [0.25, 0.75])


def test_normalise_augment_and_rotation():
    rng = np.random.default_rng(1)
    img = np.zeros((8, 16, 16), np.float32)
    img[2:6, 4:12, 4:12] = rng.uniform(100, 200, (4, 8, 8))
    n = C.normalise(img)
    assert n.dtype == np.float32 and n[0, 0, 0] == -1.0 and -1.0 <= n.min() and n.max() <= 1.0 and n[2:6, 4:12, 4:12].mean() > -0.5
    assert (C.normalise(np.zeros((2, 2, 2))) == -1.0).all()
    a = C.augment(n, rng, p=1.0)
    assert a.shape == n.shape and a.dtype == np.float32 and (a[n == -1.0] == -1.0).all() and not np.allclose(a, n)
    labels = np.zeros((8, 16, 16), np.int32)
    labels[2:6, 4:12, 4:12] = 3
    img_r, lab_r = C.rotate_inplane([n, labels], 10.0, [1, 0])
    assert img_r.shape == n.shape and lab_r.dtype == np.int32 and set(np.unique(lab_r)) <= {0, 3} and lab_r.sum() > 0
    same = C.rotate_inplane([labels], 0.0, [0])[0]
    assert (same == labels).all()
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_crops.py -q -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError / ImportError / AttributeError for the new names).

- [ ] **Step 3: Write `anatobind/aur/crops.py`**

```python
"""Crops in (z, y, x) order with padding, validity, physical coordinates and intensity handling (spec N4, §6).

Volumes are read as (x, y, z) arrays (nibabel) and turned to (z, y, x) so that the anisotropic slice axis is the
model's D axis, where the (2, 4, 4) patch stem expects it. A crop may run outside the volume: the outside is filled and
marked invalid (M_valid); nothing is resized. Coordinates are physical (mm, the volume's own frame) for the RoPE and
normalised to [-1, 1] over the volume's extent for PE_local (plan §7)."""
import numpy as np
from scipy import ndimage

from anatobind.anatomy.simulate import bias_field

CROP = (128, 160, 160)          # (z, y, x) voxels, spec §6
PATCH = (2, 4, 4)


def to_zyx(arr):
    return np.ascontiguousarray(np.transpose(np.asarray(arr), (2, 1, 0)))


def spacing_zyx(spacing):
    return (float(spacing[2]), float(spacing[1]), float(spacing[0]))


def crop_window(shape, crop, rng, centre=None):
    """[(start, stop)] per axis in volume voxel coordinates (may leave the volume). Without a centre the window is
    uniformly random inside the volume where the volume is larger than the crop; where it is smaller the crop is
    centred on the volume. With a centre the window is placed on it and then pushed back inside the volume where it
    fits."""
    out = []
    for n, c, ctr in zip(shape, crop, centre if centre is not None else (None, None, None)):
        if n <= c:
            start = -((c - n) // 2)
        elif ctr is None:
            start = int(rng.integers(0, n - c + 1))
        else:
            start = min(max(int(round(ctr)) - c // 2, 0), n - c)
        out.append((start, start + c))
    return out


def extract(arr, window, fill=0):
    """(arr inside the window with the outside filled, boolean validity of the same shape)."""
    crop = np.full([b - a for a, b in window], fill, dtype=arr.dtype)
    valid = np.zeros(crop.shape, bool)
    src = tuple(slice(max(a, 0), min(b, n)) for (a, b), n in zip(window, arr.shape))
    dst = tuple(slice(s.start - a, s.stop - a) for s, (a, b) in zip(src, window))
    if all(s.stop > s.start for s in src):
        crop[dst] = arr[src]
        valid[dst] = True
    return crop, valid


def coordinates_mm(window, spacing):
    """(3, D, H, W) float32: physical coordinates (mm) of the crop's voxel centres in the volume's frame."""
    axes = [(np.arange(a, b, dtype=np.float32) + 0.5) * s for (a, b), s in zip(window, spacing)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), 0).astype(np.float32)


def local_coordinates(window, shape):
    """(3, D, H, W) float32 in [-1, 1]: position relative to the volume's extent (PE_local); outside voxels run beyond."""
    axes = [((np.arange(a, b, dtype=np.float32) + 0.5) / n) * 2 - 1 for (a, b), n in zip(window, shape)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), 0).astype(np.float32)


def normalise(image):
    """Whole-volume robust scaling: the 0.5th–99.5th percentiles of the non-zero voxels to [-1, 1]; zeros stay at -1."""
    img = np.asarray(image, np.float32)
    vals = img[img != 0]
    if vals.size == 0:
        return np.full(img.shape, -1.0, np.float32)
    lo, hi = np.percentile(vals, [0.5, 99.5])
    hi = hi if hi > lo else lo + 1.0
    out = (np.clip(img, lo, hi) - lo) / (hi - lo) * 2 - 1
    out[img == 0] = -1.0
    return out.astype(np.float32)


def augment(image, rng, p=0.8):
    """Intensity augmentation only, no mirroring (S4 A17): a second-order bias field, gamma, in-plane blur and Gaussian
    noise, each with probability p, on an image in [-1, 1]; invalid voxels (exactly -1) keep their value."""
    img = np.asarray(image, np.float32).copy()
    fg = img > -1.0
    if rng.random() < p:
        field = bias_field(img.shape, rng.uniform(-1, 1, 6), amp=0.2).astype(np.float32)
        img = np.where(fg, (img + 1.0) * field - 1.0, img)
    if rng.random() < p:
        g = float(rng.uniform(0.7, 1.4))
        img = np.where(fg, np.power(np.clip((img + 1.0) / 2.0, 0.0, 1.0), g) * 2.0 - 1.0, img)
    if rng.random() < p * 0.5:
        img = np.where(fg, ndimage.gaussian_filter(img, sigma=(0.0, float(rng.uniform(0.3, 0.8)), float(rng.uniform(0.3, 0.8)))), img)
    if rng.random() < p:
        img = np.where(fg, img + rng.normal(0.0, float(rng.uniform(0.0, 0.05)), img.shape).astype(np.float32), img)
    return img.astype(np.float32)


def rotate_inplane(arrays, angle_deg, orders):
    """Rotate every array about the z axis (the in-plane rotation, axes (1, 2) in (z, y, x)) by the same angle, each
    with its own interpolation order (1 for images, 0 for labels and masks); shapes are kept."""
    out = []
    for a, order in zip(arrays, orders):
        r = ndimage.rotate(a, angle_deg, axes=(1, 2), reshape=False, order=order, mode="constant", cval=0, prefilter=False)
        out.append(r.astype(a.dtype) if order == 0 else r)
    return out
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_crops.py -q -p no:cacheprovider`
Expected: PASS, no warnings about the new code.

- [ ] **Step 5: Commit**

```bash
git add tests/test_aur_crops.py anatobind/aur/crops.py
git commit -m "AUR: crops in zyx with validity, physical and local coordinates, normalisation and augmentation"
```

---

### Task 5: Physical-coordinate RoPE

**Files:**
- Create: `anatobind/aur/rope.py`
- Test: `tests/test_aur_rope.py`

**Interfaces:**
- Produces: `DIMS_PER_AXIS = 8`, `BASE = 1000.0`, `rope_frequencies(dims_per_axis, base, device)`, `rotate_half(x)`, `apply_rope(x (B, heads, N, d), coords (B, N, 3)) -> same shape and dtype`

- [ ] **Step 1: Write the failing test** `tests/test_aur_rope.py`

```python
# tests/test_aur_rope.py
import pytest
import torch

import anatobind.aur.rope as R


def test_rotation_touches_only_the_first_24_dims_and_keeps_shape_and_dtype():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 32)
    coords = torch.rand(2, 5, 3) * 100
    y = R.apply_rope(x, coords)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.allclose(y[..., 24:], x[..., 24:]) and not torch.allclose(y[..., :24], x[..., :24])
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4)          # a rotation keeps the norm
    assert torch.allclose(R.apply_rope(x, torch.zeros(2, 5, 3)), x, atol=1e-6)  # the origin rotates by nothing
    yh = R.apply_rope(x.half(), coords)
    assert yh.dtype == torch.float16 and torch.allclose(yh.float(), y, atol=2e-2)
    with pytest.raises(ValueError, match="too small"):
        R.apply_rope(torch.randn(1, 1, 2, 16), torch.zeros(1, 2, 3))


def test_attention_logits_depend_on_coordinate_differences_only():
    torch.manual_seed(1)
    q, k = torch.randn(1, 2, 6, 32), torch.randn(1, 2, 6, 32)
    coords = torch.rand(1, 6, 3) * 50
    shift = torch.tensor([[[12.5, -40.0, 7.0]]])
    a = R.apply_rope(q, coords) @ R.apply_rope(k, coords).transpose(-2, -1)
    b = R.apply_rope(q, coords + shift) @ R.apply_rope(k, coords + shift).transpose(-2, -1)
    assert torch.allclose(a, b, atol=1e-3)
    c = R.apply_rope(q, coords * 1.5) @ R.apply_rope(k, coords * 1.5).transpose(-2, -1)
    assert not torch.allclose(a, c, atol=1e-2)                                    # distances do matter
    f = R.rope_frequencies()
    assert f.shape == (4,) and f[0] == 1.0 and f[-1] == pytest.approx(1000 ** (-6 / 8))
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_rope.py -q -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError / ImportError / AttributeError for the new names).

- [ ] **Step 3: Write `anatobind/aur/rope.py`**

```python
"""Physical-coordinate 3D rotary position embedding (plan §7; spec N5).

Each attention head's dimension is split into three blocks of DIMS_PER_AXIS (z, y, x) that are rotated by the token's
physical coordinate along that axis, in mm; the remaining dimensions are left alone. Rotating q and k alike makes
q·k a function of the coordinate difference only, so the attention is translation-invariant and sees distances in
mm, whatever the voxel spacing. Computed in float32: at 160 mm and 1 rad / mm the angles are too large for half
precision."""
import torch

DIMS_PER_AXIS = 8            # 3 x 8 = 24 rotated dimensions of each head (heads have 32)
BASE = 1000.0                # frequencies 1 .. 1000^(-3/4) rad / mm: wavelengths 2π·(1 .. 178) mm


def rope_frequencies(dims_per_axis=DIMS_PER_AXIS, base=BASE, device=None):
    """(dims_per_axis / 2,) angular frequencies in rad / mm."""
    i = torch.arange(0, dims_per_axis, 2, device=device, dtype=torch.float32)
    return 1.0 / (base ** (i / dims_per_axis))


def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), -1).flatten(-2)


def apply_rope(x, coords, dims_per_axis=DIMS_PER_AXIS, base=BASE):
    """x: (B, heads, N, d) ; coords: (B, N, 3) mm (z, y, x). Returns x with the first 3·dims_per_axis dimensions of
    every head rotated by the axis coordinates; same shape and dtype as x."""
    if x.shape[-1] < 3 * dims_per_axis:
        raise ValueError(f"head dimension {x.shape[-1]} is too small for 3 x {dims_per_axis} rotated dimensions")
    freqs = rope_frequencies(dims_per_axis, base, x.device)
    xf = x.float()
    cf = coords.float()
    outs = []
    for a in range(3):
        ang = (cf[..., a, None] * freqs).repeat_interleave(2, -1)[:, None]        # (B, 1, N, dims_per_axis)
        part = xf[..., a * dims_per_axis:(a + 1) * dims_per_axis]
        outs.append(part * ang.cos() + rotate_half(part) * ang.sin())
    outs.append(xf[..., 3 * dims_per_axis:])
    return torch.cat(outs, -1).to(x.dtype)
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_rope.py -q -p no:cacheprovider`
Expected: PASS, no warnings about the new code.

- [ ] **Step 5: Commit**

```bash
git add tests/test_aur_rope.py anatobind/aur/rope.py
git commit -m "AUR: physical-coordinate 3D rotary position embedding"
```

---

### Task 6: Variable-size Swin backbone

**Files:**
- Create: `anatobind/aur/swin.py`
- Test: `tests/test_aur_swin.py`

**Interfaces:**
- Consumes: `monai.networks.nets.swin_unetr.{window_partition, window_reverse, compute_mask, get_window_size}`, `anatobind.aur.rope.apply_rope`
- Produces: `CHANNELS, DEPTHS, HEADS, WINDOW, PATCH`, `WindowAttention`, `SwinBlock`, `PatchMerging`, `pool_coords`, `pool_valid`, `Stage`, `SwinBackbone(embed, depths, heads, window, patch, in_channels=1, use_checkpoint=False)` with `.channels` and `forward(image, valid, coords, local) -> list of 4 levels {feat (B, C, D', H', W'), valid (B, D', H', W') float, coords (B, 3, D', H', W'), local (B, 3, D', H', W')}`

- [ ] **Step 1: Write the failing test** `tests/test_aur_swin.py`

```python
# tests/test_aur_swin.py
import torch

import anatobind.aur.swin as S

TINY = dict(embed=32, depths=(1, 2, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4))


def _inputs(shape=(8, 32, 32), valid_until=None, seed=0):
    torch.manual_seed(seed)
    img = torch.randn(2, 1, *shape)
    valid = torch.ones(2, *shape)
    if valid_until is not None:
        valid[:, :, valid_until:] = 0.0                     # the lower rows are padding
        img[:, :, valid_until:] = -1.0
    axes = [torch.arange(n, dtype=torch.float32) + 0.5 for n in shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0)[None].expand(2, -1, -1, -1, -1).clone()
    local = torch.stack(torch.meshgrid(*[(a / n) * 2 - 1 for a, n in zip(axes, shape)], indexing="ij"), 0)[None].expand(2, -1, -1, -1, -1).clone()
    return img, valid, coords, local


def test_four_levels_with_the_plans_strides_channels_and_carried_geometry():
    bb = S.SwinBackbone(**TINY)
    img, valid, coords, local = _inputs()
    levels = bb(img, valid, coords, local)
    assert [tuple(l["feat"].shape) for l in levels] == [(2, 32, 4, 8, 8), (2, 64, 2, 4, 4), (2, 128, 1, 2, 2), (2, 256, 1, 1, 1)]
    assert [tuple(l["valid"].shape[1:]) for l in levels] == [(4, 8, 8), (2, 4, 4), (1, 2, 2), (1, 1, 1)]
    assert tuple(levels[0]["coords"].shape) == (2, 3, 4, 8, 8) and tuple(levels[2]["local"].shape) == (2, 3, 1, 2, 2)
    assert torch.allclose(levels[0]["coords"][0, :, 0, 0, 0], torch.tensor([1.0, 2.0, 2.0]))     # patch centre in mm
    assert torch.allclose(levels[1]["coords"][0, :, 0, 0, 0], torch.tensor([2.0, 4.0, 4.0]))     # merged: mean of 8
    assert bb.channels == (32, 64, 128, 256) and bb.num_parameters() > 0
    full = S.SwinBackbone()
    assert full.channels == S.CHANNELS and 10e6 < full.num_parameters() < 15e6       # 12.9 M measured


def test_padding_never_leaks_into_valid_tokens():
    torch.manual_seed(0)
    bb = S.SwinBackbone(**TINY).eval()
    img, valid, coords, local = _inputs(valid_until=16)
    img2 = img.clone()
    img2[:, :, 16:] = torch.randn_like(img2[:, :, 16:]) * 5                  # change the padded voxels only
    with torch.no_grad():
        a, b = bb(img, valid, coords, local), bb(img2, valid, coords, local)
    for la, lb in zip(a, b):
        v = la["valid"] > 0.5
        assert torch.allclose(la["feat"].permute(0, 2, 3, 4, 1)[v], lb["feat"].permute(0, 2, 3, 4, 1)[v], atol=1e-5)
        assert torch.allclose(la["valid"], lb["valid"])
    assert a[0]["valid"][0, :, 4:].sum() == 0 and a[0]["valid"][0, :, :4].sum() == 4 * 4 * 8
    assert a[1]["valid"][0, :, 2:].sum() == 0 and a[1]["valid"][0, :, :2].sum() == 2 * 2 * 4


def test_shifted_windows_checkpointing_and_odd_shapes():
    bb = S.SwinBackbone(**dict(TINY, depths=(2, 2, 1, 1)), use_checkpoint=True)
    img, valid, coords, local = _inputs(shape=(6, 36, 28))                     # not window multiples: internal padding
    img.requires_grad_(True)
    levels = bb(img, valid, coords, local)
    assert tuple(levels[0]["feat"].shape[2:]) == (3, 9, 7) and tuple(levels[1]["feat"].shape[2:]) == (2, 5, 4)
    levels[3]["feat"].sum().backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()
    try:
        bb(torch.randn(1, 1, 7, 32, 32), torch.ones(1, 7, 32, 32), torch.zeros(1, 3, 7, 32, 32), torch.zeros(1, 3, 7, 32, 32))
    except ValueError as e:
        assert "multiple of the patch" in str(e)
    else:
        raise AssertionError("a shape that is not a patch multiple must be refused")
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_swin.py -q -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError / ImportError / AttributeError for the new names).

- [ ] **Step 3: Write `anatobind/aur/swin.py`**

```python
"""Variable-size 3D Swin backbone with M_valid key masking and physical-coordinate RoPE (spec N5; plan §4–§8).

The plan's geometry: Conv3D patch stem (2, 4, 4), channels [64, 128, 256, 512], depths [2, 2, 6, 2], heads
[2, 4, 8, 16], window (4, 8, 8). Four feature levels F1..F4 at strides (2, 4, 4), (4, 8, 8), (8, 16, 16), (16, 32, 32):
F_k is the output of stage k before its merging. The window partition, reverse and shift mask come from MONAI; the
attention is our own (RoPE instead of a relative position bias, invalid tokens never serve as keys), and the features
of invalid tokens are zeroed before every stage and before merging so that padding never leaks into valid tokens.
Every level is returned with its own validity, physical coordinates (mm) and local coordinates ([-1, 1])."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import compute_mask, get_window_size, window_partition, window_reverse
from torch.utils.checkpoint import checkpoint

from anatobind.aur.rope import apply_rope

CHANNELS = (64, 128, 256, 512)
DEPTHS = (2, 2, 6, 2)
HEADS = (2, 4, 8, 16)
WINDOW = (4, 8, 8)
PATCH = (2, 4, 4)
KEY_MASK = -1e4


class WindowAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, coords, valid, shift_mask):
        """x (B', N, C) tokens of B' windows; coords (B', N, 3) mm; valid (B', N) bool; shift_mask (nW, N, N) or None."""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, coords), apply_rope(k, coords)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if shift_mask is not None:
            nw = shift_mask.shape[0]
            attn = (attn.view(b // nw, nw, self.heads, n, n) + shift_mask[None, :, None].to(attn.dtype)).view(b, self.heads, n, n)
        attn = attn.masked_fill(~valid[:, None, None, :], KEY_MASK)
        attn = attn.softmax(-1).to(v.dtype)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, c))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window, shift, use_checkpoint=False):
        super().__init__()
        self.window, self.shift, self.use_checkpoint = tuple(window), tuple(shift), use_checkpoint
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def _attention(self, x, coords, valid, shift_mask):
        """x (B, D, H, W, C); coords (B, D, H, W, 3); valid (B, D, H, W) float."""
        b, d, h, w, c = x.shape
        window, shift = get_window_size((d, h, w), self.window, self.shift)
        pads = [(ws - s % ws) % ws for s, ws in zip((d, h, w), window)]
        pad = (0, pads[2], 0, pads[1], 0, pads[0])
        x = F.pad(self.norm1(x), (0, 0) + pad)
        coords = F.pad(coords, (0, 0) + pad)
        valid = F.pad(valid, pad)
        dims = [b, *x.shape[1:4]]
        shifted = any(s > 0 for s in shift)
        if shifted:
            roll = tuple(-s for s in shift)
            x, coords, valid = (torch.roll(t, shifts=roll, dims=(1, 2, 3)) for t in (x, coords, valid))
        xw = window_partition(x, window)
        cw = window_partition(coords, window)
        vw = window_partition(valid[..., None], window).squeeze(-1) > 0.5
        aw = self.attn(xw, cw, vw, shift_mask if shifted else None)
        x = window_reverse(aw.view(-1, *window, c), window, dims)
        if shifted:
            x = torch.roll(x, shifts=tuple(shift), dims=(1, 2, 3))
        return x[:, :d, :h, :w].contiguous()

    def forward(self, x, coords, valid, shift_mask):
        if self.use_checkpoint and x.requires_grad:
            x = x + checkpoint(self._attention, x, coords, valid, shift_mask, use_reentrant=False)
            return x + checkpoint(lambda t: self.mlp(self.norm2(t)), x, use_reentrant=False)
        x = x + self._attention(x, coords, valid, shift_mask)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """2 x 2 x 2 tokens -> one token of twice the channels (MONAI's V2 merging); coordinates are averaged and the
    validity is 'any', both over the available tokens of each 2-block."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x, coords, valid):
        b, d, h, w, c = x.shape
        x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x = torch.cat([x[:, i::2, j::2, k::2, :] for i in range(2) for j in range(2) for k in range(2)], -1)
        x = self.reduction(self.norm(x))
        return x, pool_coords(coords.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1), pool_valid(valid)


def pool_coords(c):
    """(B, 3, D, H, W) coordinates -> (B, 3, ceil(D/2), ceil(H/2), ceil(W/2)): the mean of each 2-block, odd edges
    replicated so that a lone edge token keeps its own coordinate."""
    d, h, w = c.shape[2:]
    c = F.pad(c, (0, w % 2, 0, h % 2, 0, d % 2), mode="replicate")
    return F.avg_pool3d(c, 2, 2)


def pool_valid(v):
    """(B, D, H, W) validity -> 'any' over each 2-block."""
    d, h, w = v.shape[1:]
    v = F.pad(v[:, None], (0, w % 2, 0, h % 2, 0, d % 2))
    return F.max_pool3d(v, 2, 2)[:, 0]


class Stage(nn.Module):
    def __init__(self, dim, depth, heads, window, merge, use_checkpoint=False):
        super().__init__()
        self.window = tuple(window)
        shift = tuple(s // 2 for s in window)
        self.blocks = nn.ModuleList(SwinBlock(dim, heads, window, (0, 0, 0) if i % 2 == 0 else shift, use_checkpoint)
                                    for i in range(depth))
        self.merge = PatchMerging(dim) if merge else None

    def forward(self, x, coords, valid):
        """Returns (features of this stage (B, D, H, W, C), (x, coords, valid) for the next stage)."""
        b, d, h, w, _ = x.shape
        x = x * valid[..., None]
        window, shift = get_window_size((d, h, w), self.window, tuple(s // 2 for s in self.window))
        padded = [int(math.ceil(s / ws)) * ws for s, ws in zip((d, h, w), window)]
        mask = compute_mask(padded, window, shift, x.device)
        for block in self.blocks:
            x = block(x, coords, valid, mask)
        x = x * valid[..., None]
        if self.merge is None:
            return x, (x, coords, valid)
        return x, self.merge(x, coords, valid)


class SwinBackbone(nn.Module):
    def __init__(self, embed=CHANNELS[0], depths=DEPTHS, heads=HEADS, window=WINDOW, patch=PATCH, in_channels=1,
                 use_checkpoint=False):
        super().__init__()
        self.patch = tuple(patch)
        self.channels = tuple(embed * 2 ** i for i in range(len(depths)))
        self.patch_embed = nn.Conv3d(in_channels, embed, kernel_size=self.patch, stride=self.patch)
        self.local_embed = nn.Sequential(nn.Linear(3, embed), nn.GELU(), nn.Linear(embed, embed))
        self.stages = nn.ModuleList(Stage(self.channels[i], depths[i], heads[i], window, merge=i < len(depths) - 1,
                                          use_checkpoint=use_checkpoint) for i in range(len(depths)))

    def forward(self, image, valid, coords, local):
        """image (B, 1, D, H, W) with D, H, W multiples of the patch; valid (B, D, H, W) float 0/1; coords and local
        (B, 3, D, H, W). Contract: the invalid voxels of `image` hold the constant -1 (the normalised background value,
        crops.normalise / dataset.make_crop), so a patch that is only partly valid is a valid token that sees that
        constant like any image border; no image content ever lies under `valid == 0`. Returns a list of levels, each {"feat": (B, C, D', H', W'), "valid": (B, D', H', W') float,
        "coords": (B, 3, D', H', W') mm, "local": (B, 3, D', H', W')}."""
        if any(s % p for s, p in zip(image.shape[2:], self.patch)):
            raise ValueError(f"spatial shape {tuple(image.shape[2:])} is not a multiple of the patch {self.patch}")
        x = self.patch_embed(image)
        c = F.avg_pool3d(coords, self.patch, self.patch)
        l = F.avg_pool3d(local, self.patch, self.patch)
        v = F.max_pool3d(valid[:, None], self.patch, self.patch)[:, 0]
        x = x + self.local_embed(l.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x, c = x.permute(0, 2, 3, 4, 1), c.permute(0, 2, 3, 4, 1)
        levels = []
        for stage in self.stages:
            feat, (x_next, c_next, v_next) = stage(x, c, v)
            levels.append({"feat": feat.permute(0, 4, 1, 2, 3).contiguous(), "valid": v,
                           "coords": c.permute(0, 4, 1, 2, 3).contiguous(), "local": l})
            l = pool_coords(l) if stage.merge is not None else l
            x, c, v = x_next, c_next, v_next
        return levels

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_swin.py -q -p no:cacheprovider`
Expected: PASS, no warnings about the new code.

- [ ] **Step 5: Commit**

```bash
git add tests/test_aur_swin.py anatobind/aur/swin.py
git commit -m "AUR: variable-size 3D Swin backbone with RoPE attention and M_valid key masking"
```

---

### Task 7: Entity, event, mask and sequence heads

**Files:**
- Create: `anatobind/aur/heads.py`
- Test: `tests/test_aur_heads.py`

**Interfaces:**
- Consumes: backbone levels (Task 6), `anatobind.aur.labels.{N_ENTITIES, SEQ_TYPES}`
- Produces: `DecoderLayer`, `QueryDecoder(mem_channels, d_model, n_queries, layers, heads).forward(levels)`, `EntityDecoder(channels, d_model=256, K=32, layers=3, heads=8) -> {embed (B, K, d), presence (B, K)}`, `EventDecoder(channels, d_model=256, M=64, ...) -> {embed (B, M, d), presence (B, M)}`, `MaskHead(channels, d_model, patch, dim=64, mask_dim=32)` with `.pixels(levels, image) -> {coarse, full}`, `.coarse_masks(embed, pix)`, `.full_masks(embed, pix, points=None)`, `SequenceHead(c4, d_model) -> {embed (B, d), logits (B, 6)}`

- [ ] **Step 1: Write the failing test** `tests/test_aur_heads.py`

```python
# tests/test_aur_heads.py
import torch

import anatobind.aur.heads as H
from anatobind.aur.labels import N_ENTITIES
from anatobind.aur.swin import SwinBackbone

TINY = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4))


def _levels(shape=(8, 32, 32)):
    torch.manual_seed(0)
    bb = SwinBackbone(**TINY)
    img = torch.randn(2, 1, *shape)
    valid = torch.ones(2, *shape)
    valid[1, :, 16:] = 0.0
    axes = [torch.arange(n, dtype=torch.float32) + 0.5 for n in shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0)[None].expand(2, -1, -1, -1, -1).clone()
    local = torch.stack(torch.meshgrid(*[(a / n) * 2 - 1 for a, n in zip(axes, shape)], indexing="ij"), 0)[None].expand(2, -1, -1, -1, -1).clone()
    return bb, img, bb(img, valid, coords, local)


def test_entity_event_sequence_and_mask_heads_shapes():
    bb, img, levels = _levels()
    d = 16
    a = H.EntityDecoder(bb.channels, d, layers=3, heads=2)(levels)
    u = H.EventDecoder(bb.channels, d, M=5, layers=3, heads=2)(levels)
    s = H.SequenceHead(bb.channels[3], d)(levels[3])
    assert a["embed"].shape == (2, N_ENTITIES, d) and a["presence"].shape == (2, N_ENTITIES)
    assert u["embed"].shape == (2, 5, d) and u["presence"].shape == (2, 5)
    assert s["embed"].shape == (2, d) and s["logits"].shape == (2, 6)
    mh = H.MaskHead(bb.channels, d, (2, 4, 4), dim=8, mask_dim=4)
    pix = mh.pixels(levels, img)
    assert pix["coarse"].shape == (2, 8, 4, 8, 8) and pix["full"].shape == (2, 4, 8, 32, 32)
    full = mh.full_masks(a["embed"], pix)
    assert full.shape == (2, N_ENTITIES, 8, 32, 32)
    points = torch.tensor([[0, 5, 8 * 32 * 32 - 1], [7, 7, 100]])
    pts = mh.full_masks(a["embed"], pix, points)
    assert pts.shape == (2, N_ENTITIES, 3)
    assert torch.allclose(pts[0, :, 1], full[0].flatten(1)[:, 5], atol=1e-5) and torch.allclose(pts[1, :, 0], full[1].flatten(1)[:, 7], atol=1e-5)
    assert mh.coarse_masks(u["embed"], pix).shape == (2, 5, 4, 8, 8)


def test_query_decoder_ignores_padded_memory():
    bb, img, levels = _levels()
    torch.manual_seed(1)
    dec = H.EntityDecoder(bb.channels, 16, layers=3, heads=2).eval()
    with torch.no_grad():
        a = dec(levels)
        for lv in levels:                                             # scribble on the padded tokens of sample 1
            pad = lv["valid"][1] < 0.5
            lv["feat"][1].permute(1, 2, 3, 0)[pad] = 7.0
        b = dec(levels)
    assert torch.allclose(a["embed"][1], b["embed"][1], atol=1e-5)
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_heads.py -q -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError / ImportError / AttributeError for the new names).

- [ ] **Step 3: Write `anatobind/aur/heads.py`**

```python
"""Entity, event, mask and sequence heads (spec N6, N7, §5; plan §9–§11).

QueryDecoder: learned queries attend the feature levels in turn, coarse to fine, with key padding masks from M_valid
and a coordinate embedding on the memory tokens (PE_local). EntityDecoder has K = 32 identity-anchored queries (query
k is ENTITY_LABELS[k], no matching); EventDecoder has M anonymous queries matched to lesion instances by the
Hungarian algorithm in losses.py. MaskHead turns the pyramid into pixel features once per forward; a mask is the dot
product of a query's mask embedding with those features, at full resolution (ConvTranspose up from F1 fused with the
image, as the knee FullResMaskHead) or at the coarse F1 grid for the relation geometry. During training masks are
evaluated at sampled points only."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from anatobind.aur.labels import N_ENTITIES, SEQ_TYPES


class DecoderLayer(nn.Module):
    """Pre-norm cross-attention (with key padding), query self-attention, feed-forward."""

    def __init__(self, d_model, heads):
        super().__init__()
        self.cross = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, q, mem, pad):
        h = self.n1(q)
        q = q + self.cross(h, mem, mem, key_padding_mask=pad, need_weights=False)[0]
        h = self.n2(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        return q + self.ff(self.n3(q))


class QueryDecoder(nn.Module):
    def __init__(self, mem_channels, d_model, n_queries, layers, heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.proj = nn.ModuleList(nn.Conv3d(c, d_model, 1) for c in mem_channels)
        self.coord = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.layers = nn.ModuleList(DecoderLayer(d_model, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, levels):
        """levels: the backbone levels this decoder reads, in the order the layers visit them (coarse to fine)."""
        toks, pads = [], []
        for p, lv in zip(self.proj, levels):
            t = rearrange(p(lv["feat"]), "b c d h w -> b (d h w) c")
            t = t + self.coord(rearrange(lv["local"], "b c d h w -> b (d h w) c")).to(t.dtype)
            toks.append(t)
            pads.append(rearrange(lv["valid"], "b d h w -> b (d h w)") < 0.5)
        q = self.query[None].expand(toks[0].shape[0], -1, -1)
        for i, layer in enumerate(self.layers):
            j = i % len(toks)
            q = layer(q, toks[j], pads[j])
        return self.norm(q)


class EntityDecoder(nn.Module):
    """K identity-anchored anatomy queries -> presence logit, embedding."""

    def __init__(self, channels, d_model=256, K=N_ENTITIES, layers=3, heads=8):
        super().__init__()
        self.K = K
        c1, c2, c3, c4 = channels
        self.stack = QueryDecoder((c4, c3, c2), d_model, K, layers, heads)
        self.presence = nn.Linear(d_model, 1)

    def forward(self, levels):
        q = self.stack([levels[3], levels[2], levels[1]])
        return {"embed": q, "presence": self.presence(q).squeeze(-1)}


class EventDecoder(nn.Module):
    """M anonymous abnormality queries -> presence logit, embedding."""

    def __init__(self, channels, d_model=256, M=64, layers=3, heads=8):
        super().__init__()
        self.M = M
        c1, c2, c3, c4 = channels
        self.stack = QueryDecoder((c3, c2, c1), d_model, M, layers, heads)
        self.presence = nn.Linear(d_model, 1)

    def forward(self, levels):
        q = self.stack([levels[2], levels[1], levels[0]])
        return {"embed": q, "presence": self.presence(q).squeeze(-1)}


class MaskHead(nn.Module):
    """Pixel features from the pyramid: coarse (F1 grid, `dim` channels) and full resolution (`mask_dim` channels)."""

    def __init__(self, channels, d_model, patch, dim=64, mask_dim=32):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv3d(c, dim, 1) for c in channels)
        self.smooth = nn.ModuleList(nn.Sequential(nn.Conv3d(dim, dim, 3, padding=1), nn.GELU()) for _ in channels)
        self.up = nn.ConvTranspose3d(dim, mask_dim, kernel_size=tuple(patch), stride=tuple(patch))
        self.img = nn.Sequential(nn.Conv3d(1, mask_dim, 3, padding=1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv3d(2 * mask_dim, mask_dim, 3, padding=1), nn.GELU(), nn.Conv3d(mask_dim, mask_dim, 1))
        self.embed_full = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, mask_dim))
        self.embed_coarse = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, dim))

    def pixels(self, levels, image):
        """{"coarse": (B, dim, D1, H1, W1), "full": (B, mask_dim, D, H, W)}."""
        feats = [lv["feat"] for lv in levels]
        p = self.smooth[3](self.lateral[3](feats[3]))
        for i in (2, 1, 0):
            lat = self.lateral[i](feats[i])
            p = self.smooth[i](lat + F.interpolate(p, size=lat.shape[2:], mode="trilinear", align_corners=False))
        full = self.fuse(torch.cat([self.up(p), self.img(image)], 1))
        return {"coarse": p, "full": full}

    def coarse_masks(self, embed, pix):
        """(B, Q, D1, H1, W1) mask logits on the F1 grid."""
        return torch.einsum("bqc,bcdhw->bqdhw", self.embed_coarse(embed), pix["coarse"])

    def full_masks(self, embed, pix, points=None):
        """Full-resolution mask logits: (B, Q, D, H, W), or (B, Q, P) at the flat voxel indices `points` (B, P)."""
        e = self.embed_full(embed)
        if points is None:
            return torch.einsum("bqc,bcdhw->bqdhw", e, pix["full"])
        flat = pix["full"].flatten(2)                                                       # (B, C, V)
        gathered = torch.gather(flat, 2, points[:, None, :].expand(-1, flat.shape[1], -1))  # (B, C, P)
        return torch.einsum("bqc,bcp->bqp", e, gathered)


class SequenceHead(nn.Module):
    """S token: masked mean of F4 -> sequence type logits and an embedding the relation tokens read."""

    def __init__(self, c4, d_model, n_types=len(SEQ_TYPES)):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(c4, d_model), nn.GELU())
        self.cls = nn.Linear(d_model, n_types)

    def forward(self, level):
        f = rearrange(level["feat"], "b c d h w -> b (d h w) c")
        w = rearrange(level["valid"], "b d h w -> b (d h w)").to(f.dtype)
        pooled = (f * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1.0)
        s = self.proj(pooled)
        return {"embed": s, "logits": self.cls(s)}
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_heads.py -q -p no:cacheprovider`
Expected: PASS, no warnings about the new code.

- [ ] **Step 5: Commit**

```bash
git add tests/test_aur_heads.py anatobind/aur/heads.py
git commit -m "AUR: entity, event, mask and sequence heads with key padding masks"
```

---

### Task 8: Relation tokens and host competition

**Files:**
- Create: `anatobind/aur/relation.py`
- Test: `tests/test_aur_relation.py`

**Interfaces:**
- Consumes: `anatobind.aur.labels.{ENTITY_HOST, HOST_TISSUE, N_HOSTS, TISSUES}`
- Produces: `GEO_DIM = 15`, `DIST_CAP_MM`, `SCALE_MM`, `host_from_entities(x (B, 32, ...)) -> (B, 13, ...)`, `host_masks_from_entities(entity_probs) -> (B, 13, ...)`, `geometry(event_probs (N, D, H, W), host_probs (13, D, H, W), coords (3, D, H, W)) -> (N, 13, 15)`, `CandidateCompetition(d_model, d_geo=15, layers=2, heads=4).forward(event_embed (N, d), host_embed (13, d), seq_embed (d,), geo (N, 13, 15)) -> (N, 14)`

- [ ] **Step 1: Write the failing test** `tests/test_aur_relation.py`

```python
# tests/test_aur_relation.py
import torch

import anatobind.aur.relation as R
from anatobind.aur.labels import ENTITY_HOST, ENTITY_INDEX, HOST_NAMES, HOST_SIDE, N_ENTITIES, N_HOSTS, TISSUES


def test_host_grouping_of_entities():
    emb = torch.zeros(1, N_ENTITIES, 4)
    for i, h in enumerate(ENTITY_HOST):
        emb[0, i] = h if h >= 0 else 99
    hosts = R.host_from_entities(emb)
    assert hosts.shape == (1, N_HOSTS, 4) and torch.allclose(hosts[0, :, 0], torch.arange(N_HOSTS).float())
    probs = torch.zeros(1, N_ENTITIES, 2, 4, 4)
    probs[0, ENTITY_INDEX[11], :, :2] = 0.9                           # left caudate -> basal ganglia left
    probs[0, ENTITY_INDEX[12], :, 2:] = 0.4                           # left putamen, same host
    hm = R.host_masks_from_entities(probs)
    bg = HOST_NAMES.index("basal_ganglia_left")
    assert hm.shape == (1, N_HOSTS, 2, 4, 4) and hm[0, bg, 0, 0, 0] == 0.9 and hm[0, bg, 0, 3, 0] == 0.4 and hm[0].sum() == hm[0, bg].sum()


def test_geometry_between_events_and_hosts():
    # an event fully inside host 0 (left white matter), host 1 (right white matter) on the other side
    D, Hh, W = 2, 4, 8
    axes = [torch.arange(n, dtype=torch.float32) + 0.5 for n in (D, Hh, W)]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0) * 2.0           # 2 mm voxels
    host = torch.zeros(N_HOSTS, D, Hh, W)
    host[0, :, :, :4] = 1.0
    host[1, :, :, 4:] = 1.0
    ev = torch.zeros(2, D, Hh, W)
    ev[0, :, 1:3, 1:3] = 1.0                                           # inside host 0
    ev[1, :, :, 3:5] = 1.0                                             # half in each
    g = R.geometry(ev, host, coords)
    assert g.shape == (2, N_HOSTS, R.GEO_DIM) and R.GEO_DIM == 15
    assert g[0, 0, 0] == 1.0 and g[0, 1, 0] == 0.0 and g[1, 0, 0] == 0.5 and g[1, 1, 0] == 0.5
    assert g[0, 0, 4] == 0.0 and abs(g[0, 1, 4] * 100 - 27 ** 0.5) < 1e-3       # centroid (2, 4, 4) mm to voxel (3, 3, 9) mm
    assert g[0, 0, 5] == 1.0 and g[0, 2, 5] == 0.0 and g[0, 2, 4] == R.DIST_CAP_MM / 100
    assert g[0, 0, 6] < 0 and g[0, 1, 7] == 1.0 and g[0, 0, 7] == -1.0           # event on the left; host sides by identity
    assert g[0, HOST_NAMES.index("brainstem"), 7] == 0.0 and all(g[0, i, 7] == (-1.0 if HOST_SIDE[i] == "left" else 1.0 if HOST_SIDE[i] == "right" else 0.0) for i in range(N_HOSTS))
    flipped = R.geometry(ev, host.flip(-1), coords)                                 # the right host at small x: the event is now on the right
    assert flipped[0, 0, 6] > 0 and flipped[0, 1, 7] == 1.0
    one_side = host.clone()
    one_side[1] = 0.0                                                               # a crop without any right host
    assert (R.geometry(ev, one_side, coords)[:, :, 6] == 0.0).all()
    assert g[0, 0, 8:].sum() == 1.0 and g[0, 0, 8 + TISSUES.index("white_matter")] == 1.0
    disp = g[0, 1, 1:4] * 100
    assert disp[2] > 0 and abs(disp[0]) < 1e-4                                 # host 1 lies at larger x, same z


def test_candidate_competition_shapes():
    torch.manual_seed(0)
    cc = R.CandidateCompetition(16, layers=2, heads=2)
    logits = cc(torch.randn(3, 16), torch.randn(N_HOSTS, 16), torch.randn(16), torch.randn(3, N_HOSTS, R.GEO_DIM))
    assert logits.shape == (3, N_HOSTS + 1) and torch.isfinite(logits).all()
    assert cc(torch.randn(0, 16), torch.randn(N_HOSTS, 16), torch.randn(16), torch.randn(0, N_HOSTS, R.GEO_DIM)).shape == (0, 14)
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_relation.py -q -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError / ImportError / AttributeError for the new names).

- [ ] **Step 3: Write `anatobind/aur/relation.py`**

```python
"""Relation tokens and the per-lesion host competition (spec N8; plan §12; V7 §8).

For an abnormality j and each of the 13 host classes i a token r_ij = fuse([A_i, U_j, S, G_ij]) is built, where A_i is
the mean embedding of the entities that form host i, U_j the event embedding, S the sequence embedding and G_ij the
geometry between the predicted event mask and the predicted host mask on the coarse (F1) grid: overlap share, centroid
displacement (mm / 100), distance from the event centroid to the nearest host voxel (mm / 100, capped), host presence,
the event's side offset from the midline between the left and the right host masses (signed towards the patient's
right, 0 when one side is absent from the crop), the host's own side (-1 left, +1 right, 0 brainstem) and the host's
tissue family. The 13 tokens plus a
"no host" token attend to each other (only within the lesion) and each yields one logit: a 14-way host distribution."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from anatobind.aur.labels import ENTITY_HOST, HOST_SIDE, HOST_TISSUE, N_HOSTS, TISSUES

GEO_DIM = 1 + 3 + 1 + 1 + 1 + 1 + len(TISSUES)      # 15
DIST_CAP_MM = 200.0
SCALE_MM = 100.0

_HOST_OF_ENTITY = torch.tensor(ENTITY_HOST)                                    # (32,) host index or -1
_TISSUE_ONEHOT = F.one_hot(torch.tensor([TISSUES.index(t) for t in HOST_TISSUE]), len(TISSUES)).float()   # (13, 7)
_HOST_SIDE_SIGN = torch.tensor([-1.0 if s == "left" else 1.0 if s == "right" else 0.0 for s in HOST_SIDE])  # (13,)


def host_from_entities(x):
    """(B, 32, ...) per-entity tensor -> (B, 13, ...) per-host: mean (for embeddings) over the entities of each host."""
    out = []
    for h in range(N_HOSTS):
        idx = torch.nonzero(_HOST_OF_ENTITY == h).flatten().to(x.device)
        out.append(x.index_select(1, idx).mean(1))
    return torch.stack(out, 1)


def host_masks_from_entities(entity_probs):
    """(B, 32, ...) entity mask probabilities -> (B, 13, ...) host probabilities: max over the host's entities."""
    out = []
    for h in range(N_HOSTS):
        idx = torch.nonzero(_HOST_OF_ENTITY == h).flatten().to(entity_probs.device)
        out.append(entity_probs.index_select(1, idx).amax(1))
    return torch.stack(out, 1)


def geometry(event_probs, host_probs, coords):
    """event_probs (N, D, H, W) in [0, 1]; host_probs (13, D, H, W); coords (3, D, H, W) mm (z, y, x) on the same grid
    -> (N, 13, GEO_DIM) float32."""
    e = event_probs.flatten(1).float()                                  # (N, V)
    h = host_probs.flatten(1).float()                                   # (13, V)
    c = coords.flatten(1).float()                                       # (3, V)
    e_mass = e.sum(1).clamp(min=1e-6)
    h_mass = h.sum(1)
    overlap = (e @ h.t()) / e_mass[:, None]                             # (N, 13)
    e_cent = (e @ c.t()) / e_mass[:, None]                              # (N, 3)
    h_cent = (h @ c.t()) / h_mass.clamp(min=1e-6)[:, None]              # (13, 3)
    present = (h_mass > 0.5).float()
    disp = (h_cent[None] - e_cent[:, None]) / SCALE_MM                  # (N, 13, 3)
    dist = torch.full((e.shape[0], N_HOSTS), DIST_CAP_MM, device=e.device)
    for j in range(N_HOSTS):
        vox = c[:, h[j] > 0.5]                                          # (3, Mj)
        if vox.shape[1]:
            dist[:, j] = torch.cdist(e_cent, vox.t()).amin(1).clamp(max=DIST_CAP_MM)
    dist = torch.where(overlap > 0.05, torch.zeros_like(dist), dist)    # touching the host: no distance
    side = _HOST_SIDE_SIGN.to(e.device)
    left_mass, right_mass = h[side < 0].sum(0), h[side > 0].sum(0)                     # (V,) each
    if left_mass.sum() > 0.5 and right_mass.sum() > 0.5:
        lx, rx = (left_mass @ c[2]) / left_mass.sum(), (right_mass @ c[2]) / right_mass.sum()
        e_side = (e_cent[:, 2] - (lx + rx) / 2) * torch.sign(rx - lx) / SCALE_MM
    else:
        e_side = torch.zeros(e.shape[0], device=e.device)
    e_side = e_side[:, None, None].expand(-1, N_HOSTS, 1)
    h_side = side[None, :, None].expand(e.shape[0], -1, 1)
    tissue = _TISSUE_ONEHOT.to(e.device)[None].expand(e.shape[0], -1, -1)
    return torch.cat([overlap[..., None], disp, (dist / SCALE_MM)[..., None], present[None, :, None].expand(e.shape[0], -1, 1),
                      e_side, h_side, tissue], -1)


class CandidateCompetition(nn.Module):
    """13 candidate tokens + 1 no-host token per lesion -> 14 logits (spec N8)."""

    def __init__(self, d_model, d_geo=GEO_DIM, layers=2, heads=4):
        super().__init__()
        self.fuse = nn.Sequential(nn.Linear(3 * d_model + d_geo, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.none = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.host_embed = nn.Parameter(torch.randn(N_HOSTS + 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0, batch_first=True,
                                                               norm_first=True) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, event_embed, host_embed, seq_embed, geo):
        """event_embed (N, d); host_embed (13, d); seq_embed (d,); geo (N, 13, d_geo) -> (N, 14) logits."""
        n = event_embed.shape[0]
        s = seq_embed[None, None].expand(n, N_HOSTS, -1)
        r = self.fuse(torch.cat([host_embed[None].expand(n, -1, -1), event_embed[:, None].expand(-1, N_HOSTS, -1), s, geo.to(event_embed.dtype)], -1))
        none = self.none(torch.cat([event_embed, seq_embed[None].expand(n, -1)], -1))[:, None]
        tokens = torch.cat([r, none], 1) + self.host_embed[None]
        for block in self.blocks:
            tokens = block(tokens)
        return self.out(self.norm(tokens)).squeeze(-1)
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_relation.py -q -p no:cacheprovider`
Expected: PASS, no warnings about the new code.

- [ ] **Step 5: Commit**

```bash
git add tests/test_aur_relation.py anatobind/aur/relation.py
git commit -m "AUR: relation geometry and the per-lesion host competition"
```

---

### Task 9: Point-sampled losses

**Files:**
- Create: `anatobind/aur/losses.py`
- Test: `tests/test_aur_losses.py`

**Interfaces:**
- Consumes: `anatobind.aur.labels.N_HOST_CLASSES`, `scipy.optimize.linear_sum_assignment`
- Produces: `NO_OBJECT_WEIGHT, LAMBDA_R, LAMBDA_H, MARGIN, N_POINTS = 16384`, `sample_points(valid, n, generator=None) -> (B, n) long`, `gather(x (B, D, H, W), points) -> (B, P)`, `dice_loss(logits, target, weight)`, `entity_loss(mask_logits (B, K, P), presence (B, K), entity_pts (B, P), a_ignore_pts (B, P) bool, present (B, K) bool) -> {a_mask, a_presence}`, `match_events(presence (M,), mask_logits (M, P), target (N, P)) -> (qi, ti)`, `event_loss(presence (B, M), mask_logits (B, M, P), targets list[(N_b, P)], point_weight (B, P), u_supervised (B,)) -> ({u_presence, u_mask}, matches)`, `seq_loss(logits, seq) -> {s}`, `relation_loss(logits (N, 14), host (N,), negatives (N, 2), margin) -> {r_host, r_hard}`, `total(parts, lambda_r, lambda_h) -> (tensor, dict of floats)`

- [ ] **Step 1: Write the failing test** `tests/test_aur_losses.py`

```python
# tests/test_aur_losses.py
import pytest
import torch

import anatobind.aur.losses as L
from anatobind.aur.labels import N_HOST_CLASSES, NO_HOST


def test_points_are_drawn_from_valid_voxels_only():
    valid = torch.zeros(2, 2, 3, 4)
    valid[0, 0] = 1.0
    valid[1, 1, 2, 3] = 1.0
    pts = L.sample_points(valid, 50, torch.Generator().manual_seed(0))
    assert pts.shape == (2, 50) and (pts[0] < 12).all() and (pts[1] == 23).all()
    x = torch.arange(24.0).view(1, 2, 3, 4)
    assert L.gather(x, torch.tensor([[0, 23, 5]])).tolist() == [[0.0, 23.0, 5.0]]
    inst = torch.zeros(1, 2, 3, 4, dtype=torch.long)
    inst[0, 0, 0, :3] = 1                                       # a 3-voxel lesion
    inst[0, 1, 2, 3] = 2                                        # a 1-voxel lesion
    pts = L.sample_points(torch.ones(1, 2, 3, 4), 40, torch.Generator().manual_seed(0), instance=inst)
    got = L.gather(inst, pts)[0]
    assert pts.shape == (1, 40) and (got[:10] == 1).all() and (got[10:20] == 2).all()     # 20 focus points, 10 per instance
    assert L.sample_points(torch.ones(1, 2, 3, 4), 8, torch.Generator().manual_seed(0), instance=torch.zeros(1, 2, 3, 4, dtype=torch.long)).shape == (1, 8)


def test_entity_loss_prefers_the_right_masks_and_skips_lesion_points():
    pts = torch.tensor([[1, 1, 2, 0, 0, 2]])
    ignore = torch.tensor([[False, False, False, False, False, True]])
    present = torch.zeros(1, 32, dtype=torch.bool)
    present[0, :2] = True
    good = torch.full((1, 32, 6), -8.0)
    good[0, 0, :2] = 8.0
    good[0, 1, 2] = 8.0
    bad = -good
    lg = L.entity_loss(good, torch.full((1, 32), -5.0), pts, ignore, present)
    lb = L.entity_loss(bad, torch.full((1, 32), -5.0), pts, ignore, present)
    assert lg["a_mask"] < 0.05 < lb["a_mask"] and lg["a_presence"] > 0
    leaky = good.clone()
    leaky[0, 5, :] = 8.0                                           # an absent entity claiming every point: penalised
    assert L.entity_loss(leaky, torch.full((1, 32), -5.0), pts, ignore, present)["a_mask"] > lg["a_mask"] + 0.1
    worse = good.clone()
    worse[0, 1, 5] = 8.0                                           # wrong at an ignored (lesion) point: no penalty
    assert torch.allclose(L.entity_loss(worse, torch.full((1, 32), -5.0), pts, ignore, present)["a_mask"], lg["a_mask"])


def test_event_matching_and_loss():
    torch.manual_seed(0)
    target = torch.zeros(2, 6)
    target[0, :3] = 1.0
    target[1, 3:] = 1.0
    logits = torch.full((4, 6), -6.0)
    logits[2, 3:] = 6.0                                             # query 2 fits target 1
    logits[3, :3] = 6.0                                             # query 3 fits target 0
    presence = torch.tensor([-3.0, -3.0, 3.0, 3.0])
    qi, ti = L.match_events(presence, logits, target)
    assert sorted(zip(qi.tolist(), ti.tolist())) == [(2, 1), (3, 0)]
    e, t = L.match_events(presence, logits, torch.zeros(0, 6))
    assert e.numel() == 0 and t.numel() == 0
    out, matches = L.event_loss(presence[None], logits[None], [target], torch.ones(1, 6), torch.tensor([True]))
    assert out["u_mask"] < 0.05 and out["u_presence"] < 0.1 and matches[0][0].tolist() == qi.tolist()
    out2, matches2 = L.event_loss(presence[None], logits[None], [target], torch.ones(1, 6), torch.tensor([False]))
    assert out2["u_mask"] == 0 and out2["u_presence"] == 0 and matches2 == [None]
    out3, _ = L.event_loss(presence[None], torch.full_like(logits, -6.0)[None], [target], torch.ones(1, 6), torch.tensor([True]))
    assert out3["u_mask"] > out["u_mask"]
    weighted, _ = L.event_loss(presence[None], logits[None], [target], torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]), torch.tensor([True]))
    assert weighted["u_mask"] < 0.05                                 # a point under the volume floor carries no mask loss
    many = torch.eye(6)                                              # 6 instances, 4 queries: only 4 can be matched
    qi6, ti6 = L.match_events(presence, logits, many)
    assert qi6.numel() == 4 and ti6.numel() == 4 and len(set(ti6.tolist())) == 4
    crowded, m6 = L.event_loss(presence[None], logits[None], [many], torch.ones(1, 6), torch.tensor([True]))
    assert torch.isfinite(crowded["u_mask"]) and m6[0][0].numel() == 4


def test_relation_loss_and_total():
    logits = torch.full((2, N_HOST_CLASSES), -4.0)
    logits[0, 0], logits[1, NO_HOST] = 4.0, 4.0
    host = torch.tensor([0, NO_HOST])
    neg = torch.tensor([[1, 2], [-1, -1]])
    good = L.relation_loss(logits, host, neg)
    assert good["r_host"] < 0.01 and good["r_hard"] == 0.0
    logits[0, 1] = 4.5
    bad = L.relation_loss(logits, host, neg)
    assert bad["r_host"] > good["r_host"] and bad["r_hard"] > 0
    z = L.relation_loss(torch.zeros(0, N_HOST_CLASSES), torch.zeros(0, dtype=torch.long), torch.zeros(0, 2, dtype=torch.long))
    assert z["r_host"] == 0 and z["r_hard"] == 0
    t, logged = L.total({"a_mask": torch.tensor(1.0), "r_hard": torch.tensor(2.0)})
    assert float(t) == pytest.approx(1.0 + L.LAMBDA_R * L.LAMBDA_H * 2.0) and logged == {"a_mask": 1.0, "r_hard": 2.0}
    assert L.seq_loss(torch.tensor([[5.0, 0, 0, 0, 0, 0]]), torch.tensor([0]))["s"] < 0.05
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_losses.py -q -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError / ImportError / AttributeError for the new names).

- [ ] **Step 3: Write `anatobind/aur/losses.py`**

```python
"""Losses (spec N11, §6): L = L_A + L_S + L_U + λ_R (L_bind + λ_h L_hard).

Masks are supervised at sampled points (the same points for every query of a sample), so no dense (Q, D, H, W) mask is
ever built in training: half of the points are drawn per lesion instance (an equal quota for every instance of the
sample, so that a small lesion always carries positive points), the rest uniformly over the valid voxels. Entity
masks: BCE for every entity (an absent entity learns an empty mask), Dice for the entities present in the crop,
presence BCE for all; lesion voxels (a_ignore) carry no entity loss. Events: Hungarian matching on
presence + mask BCE + mask Dice, then presence BCE over every query (no-object weight) and mask losses over the matched
ones; samples whose U is not supervised contribute nothing to L_U. Relation: cross-entropy over the 14 host classes
plus a hinge on the hard negatives (contralateral, second host)."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

NO_OBJECT_WEIGHT = 0.1
LAMBDA_R = 1.0
LAMBDA_H = 0.2
MARGIN = 1.0
N_POINTS = 16384


FOCUS_SHARE = 0.5


def sample_points(valid, n, generator=None, instance=None, focus_share=FOCUS_SHARE):
    """valid (B, D, H, W) bool / float -> (B, n) flat indices (with replacement). Without an instance map: uniform over
    the valid voxels. With one ((B, D, H, W) crop-local ids, 0 background): the first focus_share of the points of a
    sample are split equally among its instances and drawn from each instance's voxels, the rest are uniform."""
    flat = (valid.flatten(1) > 0.5).float()
    out = torch.multinomial(flat + 1e-12, n, replacement=True, generator=generator)
    if instance is None:
        return out
    inst = instance.flatten(1)
    for b in range(flat.shape[0]):
        ids = torch.unique(inst[b])
        ids = ids[ids > 0]
        quota = int(n * focus_share) // max(int(ids.numel()), 1)
        if ids.numel() == 0 or quota == 0:
            continue
        pos = 0
        for k in ids:
            out[b, pos:pos + quota] = torch.multinomial((inst[b] == k).float(), quota, replacement=True, generator=generator)
            pos += quota
    return out


def gather(x, points):
    """x (B, D, H, W) -> (B, P) values at the flat indices `points` (B, P)."""
    return torch.gather(x.flatten(1), 1, points)


def dice_loss(logits, target, weight):
    """logits, target (B, Q, P); weight (B, Q) -> weighted mean soft Dice loss."""
    p = logits.sigmoid()
    inter = (p * target).sum(-1)
    dice = 1 - (2 * inter + 1) / (p.sum(-1) + target.sum(-1) + 1)
    return (dice * weight).sum() / weight.sum().clamp(min=1.0)


def entity_loss(mask_logits, presence, entity_pts, a_ignore_pts, present):
    """mask_logits (B, K, P); presence (B, K); entity_pts (B, P) entity map values at the points (0 none, 1..K);
    a_ignore_pts (B, P) bool (lesion voxels); present (B, K) bool: the entity has voxels in the crop. BCE over every
    entity (absent ones learn an empty mask), Dice over the present ones."""
    K = mask_logits.shape[1]
    target = (entity_pts[:, None, :] == torch.arange(1, K + 1, device=entity_pts.device)[None, :, None]).float()
    keep = (~a_ignore_pts).float()[:, None, :]
    bce = (F.binary_cross_entropy_with_logits(mask_logits, target, reduction="none") * keep).sum(-1) / keep.sum(-1).clamp(min=1.0)
    w = present.float()
    dropped = mask_logits.masked_fill(a_ignore_pts[:, None, :], -1e4)        # ignored points count as empty in the Dice
    mask = bce.mean() + dice_loss(dropped, target * keep, w)
    return {"a_mask": mask, "a_presence": F.binary_cross_entropy_with_logits(presence, w)}


def match_events(presence, mask_logits, target):
    """One sample: presence (M,), mask_logits (M, P), target (N, P) -> (query indices, target indices). The cost
    reads every point (points under the volume floor count as target 0 here, they only carry no loss afterwards)."""
    if target.shape[0] == 0:
        empty = torch.zeros(0, dtype=torch.long, device=presence.device)
        return empty, empty
    with torch.no_grad():
        l, t = mask_logits.float(), target.float()
        P = l.shape[1]
        bce = (F.softplus(l).sum(1)[:, None] - l @ t.t()) / P                                   # (M, N) mean BCE
        p = l.sigmoid()
        dice = 1 - (2 * (p @ t.t()) + 1) / (p.sum(1)[:, None] + t.sum(1)[None, :] + 1)
        cost = -presence.float().sigmoid()[:, None] + bce + dice
        qi, ti = linear_sum_assignment(cost.cpu().numpy())
    return (torch.as_tensor(qi, dtype=torch.long, device=presence.device),
            torch.as_tensor(ti, dtype=torch.long, device=presence.device))


def event_loss(presence, mask_logits, targets, point_weight, u_supervised):
    """presence (B, M); mask_logits (B, M, P); targets: list of (N_b, P) float; point_weight (B, P) in {0, 1} (0 at the
    voxels of components under the volume floor); u_supervised (B,) bool. Returns ({"u_presence", "u_mask"}, matches)."""
    B, M, P = mask_logits.shape
    matches, pres_terms, mask_terms, n_matched = [], [], [], 0
    for b in range(B):
        if not bool(u_supervised[b]):
            matches.append(None)
            continue
        qi, ti = match_events(presence[b], mask_logits[b], targets[b])
        matches.append((qi, ti))
        tgt = torch.zeros(M, device=presence.device)
        tgt[qi] = 1.0
        w = torch.full((M,), NO_OBJECT_WEIGHT, device=presence.device)
        w[qi] = 1.0
        pres_terms.append((F.binary_cross_entropy_with_logits(presence[b], tgt, reduction="none") * w).sum() / w.sum())
        if qi.numel():
            l, t, k = mask_logits[b, qi], targets[b][ti], point_weight[b][None]
            bce = (F.binary_cross_entropy_with_logits(l, t, reduction="none") * k).sum(-1) / k.sum(-1).clamp(min=1.0)
            dropped = l.masked_fill(k < 0.5, -1e4)
            mask_terms.append(bce.sum() + dice_loss(dropped[None], (t * k)[None], torch.ones(1, qi.numel(), device=l.device)) * qi.numel())
            n_matched += qi.numel()
    zero = presence.new_zeros(())
    out = {"u_presence": torch.stack(pres_terms).mean() if pres_terms else zero,
           "u_mask": torch.stack(mask_terms).sum() / max(n_matched, 1) if mask_terms else zero}
    return out, matches


def seq_loss(logits, seq):
    return {"s": F.cross_entropy(logits, seq)}


def relation_loss(logits, host, negatives, margin=MARGIN):
    """logits (N, 14); host (N,) in 0..13; negatives (N, 2) host indices or -1 -> {"r_host", "r_hard"}; zeros when N = 0."""
    if logits.shape[0] == 0:
        z = logits.new_zeros(())
        return {"r_host": z, "r_hard": z}
    ce = F.cross_entropy(logits, host)
    pos = logits.gather(1, host[:, None])                                                      # (N, 1)
    neg = negatives.clamp(min=0)
    hinge = F.relu(margin + logits.gather(1, neg) - pos) * (negatives >= 0).float()
    return {"r_host": ce, "r_hard": hinge.sum() / (negatives >= 0).float().sum().clamp(min=1.0)}


def total(parts, lambda_r=LAMBDA_R, lambda_h=LAMBDA_H):
    """The spec's sum; missing parts count as zero. Returns (total, {name: float})."""
    get = lambda k: parts.get(k, 0.0)
    t = get("a_mask") + get("a_presence") + get("s") + get("u_presence") + get("u_mask") + lambda_r * (get("r_host") + lambda_h * get("r_hard"))
    return t, {k: float(v) for k, v in parts.items()}
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_losses.py -q -p no:cacheprovider`
Expected: PASS, no warnings about the new code.

- [ ] **Step 5: Commit**

```bash
git add tests/test_aur_losses.py anatobind/aur/losses.py
git commit -m "AUR: point-sampled entity, event, sequence and relation losses"
```

---

### Task 10: The model

**Files:**
- Create: `anatobind/aur/model.py`
- Test: `tests/test_aur_model.py`

**Interfaces:**
- Consumes: Tasks 6–9
- Produces: `DEFAULTS`, `AnatoBindBrain(**kwargs)` (unknown options raise KeyError) with `.cfg`, `.backbone`, `.entities`, `.events`, `.sequence`, `.masks`, `.relation`; `forward(image, valid, coords, local) -> {levels, entity_embed, entity_presence, event_embed, event_presence, seq_embed, seq_logits, pix}`; `entity_masks(out, points=None)`, `event_masks(out, points=None)`, `bind(out, b, event_idx) -> (N, 14)`, `num_parameters()`

- [ ] **Step 1: Write the failing test** `tests/test_aur_model.py`

```python
# tests/test_aur_model.py
import pytest
import torch

import anatobind.aur.losses as L
from anatobind.aur.labels import N_ENTITIES, N_HOST_CLASSES
from anatobind.aur.model import AnatoBindBrain

TINY = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4), d_model=16, n_events=4,
            mask_dim=4, pixel_dim=8, dec_layers=1, dec_heads=2, rel_layers=1, rel_heads=2, use_checkpoint=False)


def _batch(shape=(8, 32, 32)):
    torch.manual_seed(0)
    img = torch.randn(2, 1, *shape)
    valid = torch.ones(2, *shape)
    valid[1, :, 20:] = 0.0
    axes = [torch.arange(n, dtype=torch.float32) + 0.5 for n in shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0)[None].expand(2, -1, -1, -1, -1).clone()
    local = torch.stack(torch.meshgrid(*[(a / n) * 2 - 1 for a, n in zip(axes, shape)], indexing="ij"), 0)[None].expand(2, -1, -1, -1, -1).clone()
    return img, valid, coords, local


def test_forward_masks_bind_and_a_full_training_step_on_cpu():
    model = AnatoBindBrain(**TINY)
    img, valid, coords, local = _batch()
    out = model(img, valid, coords, local)
    assert out["entity_embed"].shape == (2, N_ENTITIES, 16) and out["event_presence"].shape == (2, 4) and out["seq_logits"].shape == (2, 6)
    pts = L.sample_points(valid, 64, torch.Generator().manual_seed(0))
    em, um = model.entity_masks(out, pts), model.event_masks(out, pts)
    assert em.shape == (2, N_ENTITIES, 64) and um.shape == (2, 4, 64)
    assert model.entity_masks(out).shape == (2, N_ENTITIES, 8, 32, 32)
    logits = model.bind(out, 1, torch.tensor([0, 2]))
    assert logits.shape == (2, N_HOST_CLASSES) and torch.isfinite(logits).all()
    # a complete loss with synthetic targets
    entity_pts = torch.randint(0, N_ENTITIES + 1, (2, 64))
    present = torch.ones(2, N_ENTITIES, dtype=torch.bool)
    targets = [torch.zeros(2, 64), torch.zeros(0, 64)]
    targets[0][0, :10] = 1.0
    targets[0][1, 20:30] = 1.0
    parts = {}
    parts.update(L.entity_loss(em, out["entity_presence"], entity_pts, torch.zeros(2, 64, dtype=torch.bool), present))
    u, matches = L.event_loss(out["event_presence"], um, targets, torch.ones(2, 64), torch.tensor([True, False]))
    parts.update(u)
    parts.update(L.seq_loss(out["seq_logits"], torch.tensor([3, 0])))
    qi, ti = matches[0]
    parts.update(L.relation_loss(model.bind(out, 0, qi), torch.tensor([0, 5])[ti], torch.tensor([[1, -1], [6, 2]])[ti]))
    total, logged = L.total(parts)
    total.backward()
    assert torch.isfinite(total) and all(torch.isfinite(torch.tensor(v)) for v in logged.values())
    assert model.backbone.patch_embed.weight.grad is not None and model.relation.out.weight.grad is not None
    with pytest.raises(KeyError, match="unknown model options"):
        AnatoBindBrain(nonsense=1)


def test_default_configuration_is_the_plans():
    model = AnatoBindBrain(use_checkpoint=False)
    assert model.backbone.channels == (64, 128, 256, 512) and model.entities.K == 32 and model.events.M == 64
    assert model.cfg["d_model"] == 256 and 20e6 < model.num_parameters() < 30e6 and 10e6 < model.backbone.num_parameters() < 15e6
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_model.py -q -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError / ImportError / AttributeError for the new names).

- [ ] **Step 3: Write `anatobind/aur/model.py`**

```python
"""The AnatoBind brain model: backbone + entity / event / sequence heads + mask head + host competition (spec §5)."""
import torch.nn as nn

from anatobind.aur.heads import EntityDecoder, EventDecoder, MaskHead, SequenceHead
from anatobind.aur.labels import N_ENTITIES
from anatobind.aur.relation import CandidateCompetition, geometry, host_from_entities, host_masks_from_entities
from anatobind.aur.swin import CHANNELS, DEPTHS, HEADS, PATCH, WINDOW, SwinBackbone

DEFAULTS = {"embed": CHANNELS[0], "depths": DEPTHS, "heads": HEADS, "window": WINDOW, "patch": PATCH, "d_model": 256,
            "n_events": 64, "mask_dim": 32, "pixel_dim": 64, "dec_layers": 3, "dec_heads": 8, "rel_layers": 2, "rel_heads": 4,
            "use_checkpoint": True}


class AnatoBindBrain(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = {**DEFAULTS, **kwargs}
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"unknown model options {sorted(unknown)}")
        self.cfg = cfg
        self.backbone = SwinBackbone(cfg["embed"], cfg["depths"], cfg["heads"], cfg["window"], cfg["patch"],
                                     use_checkpoint=cfg["use_checkpoint"])
        ch = self.backbone.channels
        d = cfg["d_model"]
        self.entities = EntityDecoder(ch, d, N_ENTITIES, cfg["dec_layers"], cfg["dec_heads"])
        self.events = EventDecoder(ch, d, cfg["n_events"], cfg["dec_layers"], cfg["dec_heads"])
        self.sequence = SequenceHead(ch[3], d)
        self.masks = MaskHead(ch, d, cfg["patch"], cfg["pixel_dim"], cfg["mask_dim"])
        self.relation = CandidateCompetition(d, layers=cfg["rel_layers"], heads=cfg["rel_heads"])

    def forward(self, image, valid, coords, local):
        """See SwinBackbone.forward for the inputs. Returns the heads' outputs and the pixel features; masks are made
        on demand with entity_masks / event_masks (points or full) so that training never materialises them densely."""
        levels = self.backbone(image, valid, coords, local)
        a, u, s = self.entities(levels), self.events(levels), self.sequence(levels[3])
        pix = self.masks.pixels(levels, image)
        return {"levels": levels, "entity_embed": a["embed"], "entity_presence": a["presence"], "event_embed": u["embed"],
                "event_presence": u["presence"], "seq_embed": s["embed"], "seq_logits": s["logits"], "pix": pix}

    def entity_masks(self, out, points=None):
        return self.masks.full_masks(out["entity_embed"], out["pix"], points)

    def event_masks(self, out, points=None):
        return self.masks.full_masks(out["event_embed"], out["pix"], points)

    def bind(self, out, b, event_idx):
        """Host logits (N, 14) of the events `event_idx` (N,) of sample b, from this sample's own predicted masks on
        the coarse grid (N8), the entity masks gated by the entity presence so that an absent entity holds no host
        mass. event_idx may be the matched queries (training) or the present ones (inference)."""
        pix = {"coarse": out["pix"]["coarse"][b:b + 1]}
        gate = out["entity_presence"][b].sigmoid()[:, None, None, None]
        ent = self.masks.coarse_masks(out["entity_embed"][b:b + 1], pix)[0].sigmoid() * gate             # (32, D1, H1, W1)
        ev = self.masks.coarse_masks(out["event_embed"][b:b + 1, event_idx], pix)[0].sigmoid()            # (N, D1, H1, W1)
        valid = out["levels"][0]["valid"][b]                                                              # (D1, H1, W1)
        hosts = host_masks_from_entities(ent[None])[0] * valid
        geo = geometry(ev * valid, hosts, out["levels"][0]["coords"][b])
        host_embed = host_from_entities(out["entity_embed"][b:b + 1])[0]
        return self.relation(out["event_embed"][b, event_idx], host_embed, out["seq_embed"][b], geo)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_model.py -q -p no:cacheprovider`
Expected: PASS, no warnings about the new code.

- [ ] **Step 5: Commit**

```bash
git add tests/test_aur_model.py anatobind/aur/model.py
git commit -m "AUR: the AnatoBind brain model (backbone, heads, masks on demand, host competition)"
```

---

### Task 11: Crop dataset

**Files:**
- Create: `anatobind/aur/dataset.py`
- Test: `tests/test_aur_dataset.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 4
- Produces: `ROTATION_DEG = 10.0`, `load_volume(row) -> dict`, `make_crop(vol, rng, crop=CROP, do_augment=True, lesion_centred=False) -> dict of tensors`, `AURDataset(rows, crop, crops_per_volume=2, do_augment=True, lesion_share=0.5, seed=0)` with `set_epoch`, `__getitem__ -> list of crops`; `collate(items) -> batch dict`; `event_targets_at_points(instance_pts (B, P), n_instances) -> list[(N_b, P)]`
- Crop keys: `image (1, D, H, W) f32, valid (D, H, W) f32, coords (3, D, H, W), local (3, D, H, W), entity (D, H, W) i64, instance (D, H, W) i64 (crop-local ids), point_weight (D, H, W) f32, a_ignore (D, H, W) bool, host (n,), host_probs (n, 14), negatives (n, 2), seq, u_supervised, n_instances, case`

- [ ] **Step 1: Write the failing test** `tests/test_aur_dataset.py`

```python
# tests/test_aur_dataset.py
import nibabel as nib
import numpy as np
import pytest
import torch

import anatobind.aur.dataset as D
from anatobind.aur.labels import HOST_NAMES, N_HOST_CLASSES, SEQ_INDEX


def _case(tmp_path, with_lesion=True):
    """A (24, 20, 10) volume at (1, 1, 2) mm: left white matter at x < 12, right at x >= 12, a ventricle, one lesion of
    3 x 3 x 2 voxels (36 mm3) in the left white matter and one single voxel (2 mm3, under the floor)."""
    shape, spacing = (24, 20, 10), (1.0, 1.0, 2.0)
    affine = np.diag(list(spacing) + [1.0])
    seg = np.zeros(shape, np.int16)
    seg[:12, :, :] = 2
    seg[12:, :, :] = 41
    seg[10:14, 8:12, :] = 4
    img = np.where(seg > 0, 300.0, 0.0).astype(np.float32) + np.random.default_rng(0).normal(0, 5, shape).astype(np.float32)
    les = np.zeros(shape, np.uint8)
    les[2:5, 2:5, 3:5] = 1
    les[20, 2, 2] = 1
    les[6:8, 14:16, 6:8] = 2                                  # a value that is lesion on other sequences only
    paths = {}
    for name, arr in (("image", img), ("anatomy", seg), ("lesion", les)):
        p = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(arr, affine), str(p))
        paths[name] = str(p)
    return {"case": "c1", "source": "pdgm", "patient": "p1", "sequence": "FLAIR", "source_sequence": "FLAIR", "image": paths["image"],
            "anatomy": paths["anatomy"], "lesion": paths["lesion"] if with_lesion else None, "u_supervised": with_lesion,
            "u_values": [1] if with_lesion else [], "a_ignore_values": [1, 2] if with_lesion else [], "split": "train"}


def test_load_volume_builds_the_targets_in_zyx(tmp_path):
    vol = D.load_volume(_case(tmp_path))
    assert vol["image"].shape == (10, 20, 24) and vol["spacing"] == (2.0, 1.0, 1.0) and vol["seq"] == SEQ_INDEX["FLAIR"]
    assert vol["entity"].shape == (10, 20, 24) and vol["instance"].max() == 1 and vol["small"].sum() == 1
    assert vol["a_ignore"].sum() == 18 + 1 + 8 and vol["hosts"]["host"].tolist() == [HOST_NAMES.index("white_matter_left")]
    assert vol["image"].min() == -1.0 and vol["image"].max() <= 1.0
    bare = D.load_volume(_case(tmp_path, with_lesion=False))
    assert bare["instance"].max() == 0 and not bare["u_supervised"] and bare["hosts"]["host"].shape == (0,)


def test_load_volume_reorients_every_source_to_ras(tmp_path):
    row = _case(tmp_path)
    ras = D.load_volume(row)
    lps = {}
    for name in ("image", "anatomy", "lesion"):                     # the same volume stored LPS: axes x and y flipped
        img = nib.load(row[name])
        arr = np.asarray(img.dataobj)[::-1, ::-1, :]
        aff = img.affine.copy()
        aff[0, 0], aff[1, 1] = -aff[0, 0], -aff[1, 1]
        aff[0, 3], aff[1, 3] = (arr.shape[0] - 1) * img.affine[0, 0], (arr.shape[1] - 1) * img.affine[1, 1]
        p = tmp_path / f"lps_{name}.nii.gz"
        nib.save(nib.Nifti1Image(np.ascontiguousarray(arr), aff), str(p))
        lps[name] = str(p)
    assert "".join(nib.aff2axcodes(nib.load(lps["image"]).affine)) == "LPS"
    back = D.load_volume({**row, **lps})
    for k in ("image", "entity", "instance", "small", "a_ignore"):
        assert np.array_equal(ras[k], back[k]), k
    assert back["spacing"] == ras["spacing"] == (2.0, 1.0, 1.0) and back["hosts"]["host"].tolist() == ras["hosts"]["host"].tolist()


def test_crops_carry_geometry_validity_and_renumbered_instances(tmp_path):
    vol = D.load_volume(_case(tmp_path))
    rng = np.random.default_rng(3)
    c = D.make_crop(vol, rng, crop=(8, 16, 16), do_augment=False, lesion_centred=True)
    assert c["image"].shape == (1, 8, 16, 16) and c["valid"].shape == (8, 16, 16) and c["coords"].shape == (3, 8, 16, 16)
    assert c["instance"].max() == 1 and c["n_instances"] == 1 and c["host"].tolist() == [HOST_NAMES.index("white_matter_left")]
    assert c["host_probs"].shape == (1, N_HOST_CLASSES) and c["negatives"].shape == (1, 2) and bool(c["u_supervised"])
    assert c["entity_present"].shape == (32,) and c["entity_present"].dtype == torch.bool and c["entity_present"][0] and not c["entity_present"][2]      # left white matter present, left cortex absent
    assert (c["image"][0][c["valid"] < 0.5] == -1.0).all() and c["point_weight"].min() >= 0 and c["entity"].dtype == torch.int64
    dz = c["coords"][0, 1, 0, 0] - c["coords"][0, 0, 0, 0]
    assert float(dz) == 2.0 and float(c["coords"][1, 0, 1, 0] - c["coords"][1, 0, 0, 0]) == 1.0
    a = D.make_crop(vol, np.random.default_rng(4), crop=(8, 16, 16), do_augment=True, lesion_centred=False)
    assert a["image"].shape == (1, 8, 16, 16) and (a["image"][0][a["valid"] < 0.5] == -1.0).all() and set(a["entity"].unique().tolist()) <= {0, 1, 2, 3, 4, 5}
    with pytest.raises(ValueError, match="multiple of the patch"):
        D.make_crop(vol, rng, crop=(7, 16, 16))
    big = D.make_crop(vol, rng, crop=(16, 32, 32), do_augment=False)          # a crop larger than the volume in every axis
    assert big["valid"].sum() == 10 * 20 * 24 and (big["image"][0][big["valid"] < 0.5] == -1.0).all() and big["entity"][0, 0, 0] == 0


def test_slivers_under_the_floor_leave_the_crop_without_an_instance():
    inst = np.zeros((4, 6, 6), np.int32)
    inst[0, 0, :3] = 1                                             # 3 voxels: at 2 mm3 each 6 mm3, under the 10 mm3 floor
    inst[1:4, 1:4, 1:4] = 2                                        # 27 voxels
    small = np.zeros(inst.shape, bool)
    small[3, 5, 5] = True
    renumbered, small2, kept = D.crop_instances(inst, small, 2.0)
    assert kept == [2] and renumbered.max() == 1 and (renumbered[1:4, 1:4, 1:4] == 1).all()
    assert small2[0, 0, :3].all() and small2[3, 5, 5] and small2.sum() == 4
    r1, s1, k1 = D.crop_instances(inst, small, 8.0)                 # 8 mm3 voxels: both instances reach the floor
    assert k1 == [1, 2] and r1.max() == 2 and s1.sum() == 1


def test_dataset_items_and_collate(tmp_path):
    rows = [_case(tmp_path), _case(tmp_path, with_lesion=False)]
    ds = D.AURDataset(rows, crop=(8, 16, 16), crops_per_volume=2, seed=1)
    item = ds[0]
    assert len(item) == 2 and item[0]["case"] == "c1"
    batch = D.collate([ds[0], ds[1]])
    assert batch["image"].shape == (4, 1, 8, 16, 16) and batch["seq"].tolist() == [3, 3, 3, 3] and batch["u_supervised"].tolist() == [True, True, False, False]
    assert len(batch["host"]) == 4 and batch["n_instances"][2] == 0 and batch["instance"].shape == (4, 8, 16, 16)
    assert batch["entity_present"].shape == (4, 32) and batch["entity_present"].dtype == torch.bool
    inst_pts = torch.tensor([[0, 2, 1]])
    t = D.event_targets_at_points(inst_pts, [2])
    assert t[0].tolist() == [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]] and D.event_targets_at_points(inst_pts, [0])[0].shape == (0, 3)
    ds.set_epoch(1)
    assert ds.epoch == 1 and len(ds) == 2
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_dataset.py -q -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError / ImportError / AttributeError for the new names).

- [ ] **Step 3: Write `anatobind/aur/dataset.py`**

```python
"""Volumes -> training crops with targets (spec §3.2, §6).

Every volume is first brought to RAS by axis flips / permutations (nibabel's closest canonical): the sources store LPS
(PDGM), LAS (ISLES, 9 BMSR cases) and RAS (SibBMS, BMSR) volumes, and the sided labels must lie on one side of the
array, or the sources would mirror each other (an implicit mirroring; S4 lesson A17). One dataset item is one sample row (a volume of one sequence) and yields `crops_per_volume` crops of it: the volume is
read once, normalised, its entity map, lesion instances and host targets built once, then each crop is cut, rotated
in-plane, intensity-augmented and labelled. Half of the crops of a volume with instances are centred on a random
instance voxel. Targets per crop: the entity map, the instance map (ids renumbered 1..n within the crop), the point
weight (0 on components under the volume floor), the entity-ignore mask (every lesion voxel of the case), the host /
host probabilities / hard negatives of the crop's instances, the sequence type and the U supervision flag. An
instance whose part inside the crop is under the volume floor is not an instance of the crop: its voxels join the
small mask (no loss). The host targets of an instance cut by the crop are those of the whole instance."""
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from anatobind.aur.crops import (CROP, PATCH, augment, coordinates_mm, crop_window, extract, local_coordinates, normalise,
                                 rotate_inplane, spacing_zyx, to_zyx)
from anatobind.aur.labels import N_ENTITIES, SEQ_INDEX, entity_map
from anatobind.aur.targets import host_targets, lesion_instances
from anatobind.eval.lesion_components import min_voxels_for

ROTATION_DEG = 10.0


def canonical(img):
    """The image reoriented to RAS by axis flips / permutations only (no resampling): the same anatomy lies on the same
    side of the array whatever the source stored."""
    return nib.as_closest_canonical(img)


def spacing_of(img):
    """Voxel spacing (x, y, z) in mm from the affine's column norms (right after a reorientation too)."""
    a = np.asarray(img.affine, dtype=np.float64)[:3, :3]
    return tuple(float(v) for v in np.sqrt((a ** 2).sum(0)))


def load_volume(row):
    """Read one sample row into (z, y, x) arrays with its targets; every file is reoriented to RAS first."""
    img_i, seg_i = canonical(nib.load(row["image"])), canonical(nib.load(row["anatomy"]))
    if img_i.shape[:3] != seg_i.shape[:3]:
        raise ValueError(f"{row['case']} {row['sequence']}: image {img_i.shape} and anatomy {seg_i.shape} differ")
    spacing = spacing_zyx(spacing_of(img_i))
    image = normalise(to_zyx(np.asarray(img_i.dataobj).astype(np.float32)))
    seg = to_zyx(np.asarray(seg_i.dataobj).astype(np.int16))
    entity = entity_map(seg)
    if row["lesion"]:
        les = to_zyx(np.asarray(canonical(nib.load(row["lesion"])).dataobj).astype(np.int16))
        if les.shape != seg.shape:
            raise ValueError(f"{row['case']}: lesion map {les.shape} and anatomy {seg.shape} differ")
        inst, small = lesion_instances(les, row["u_values"], float(np.prod(spacing)))
        a_ignore = np.isin(les, list(row["a_ignore_values"]))
    else:
        inst, small, a_ignore = np.zeros(seg.shape, np.int32), np.zeros(seg.shape, bool), np.zeros(seg.shape, bool)
    return {"image": image, "entity": entity, "instance": inst, "small": small, "a_ignore": a_ignore,
            "hosts": host_targets(inst, seg, spacing), "spacing": spacing, "seq": SEQ_INDEX[row["sequence"]],
            "u_supervised": bool(row["u_supervised"]), "case": row["case"]}


def make_crop(vol, rng, crop=CROP, do_augment=True, lesion_centred=False):
    """One crop of a loaded volume as tensors."""
    if any(c % p for c, p in zip(crop, PATCH)):
        raise ValueError(f"crop {crop} is not a multiple of the patch {PATCH}")
    inst = vol["instance"]
    centre = None
    if lesion_centred and inst.max() > 0:
        k = int(rng.integers(1, inst.max() + 1))
        vox = np.argwhere(inst == k)
        centre = vox[int(rng.integers(0, len(vox)))]
    window = crop_window(inst.shape, crop, rng, centre)
    image, valid = extract(vol["image"], window, fill=-1.0)
    if not valid.any():
        raise ValueError(f"{vol['case']}: the crop window {window} holds no voxel of the volume {inst.shape}")
    entity, _ = extract(vol["entity"], window, fill=0)
    instance, _ = extract(inst, window, fill=0)
    small, _ = extract(vol["small"], window, fill=False)
    a_ignore, _ = extract(vol["a_ignore"], window, fill=False)
    coords, local = coordinates_mm(window, vol["spacing"]), local_coordinates(window, inst.shape)
    if do_augment:
        angle = float(rng.uniform(-ROTATION_DEG, ROTATION_DEG))
        image, valid, entity, instance, small, a_ignore = rotate_inplane(
            [image, valid.astype(np.uint8), entity, instance, small.astype(np.uint8), a_ignore.astype(np.uint8)], angle, [1, 0, 0, 0, 0, 0])
        valid, small, a_ignore = valid.astype(bool), small.astype(bool), a_ignore.astype(bool)
        image = np.where(valid, image, -1.0).astype(np.float32)
        image = augment(image, rng)
    renumbered, small, ids = crop_instances(instance, small, float(np.prod(vol["spacing"])))
    sel = np.array(ids, np.int64) - 1
    hosts = vol["hosts"]
    entity_present = np.isin(np.arange(1, N_ENTITIES + 1), np.unique(entity))          # the entity has voxels in the crop
    return {"image": torch.from_numpy(np.ascontiguousarray(image))[None], "valid": torch.from_numpy(valid.astype(np.float32)),
            "coords": torch.from_numpy(coords), "local": torch.from_numpy(local),
            "entity": torch.from_numpy(entity.astype(np.int64)), "entity_present": torch.from_numpy(entity_present), "instance": torch.from_numpy(renumbered),
            "point_weight": torch.from_numpy((~small).astype(np.float32)), "a_ignore": torch.from_numpy(a_ignore),
            "host": torch.from_numpy(hosts["host"][sel]), "host_probs": torch.from_numpy(hosts["probs"][sel]),
            "negatives": torch.from_numpy(hosts["negatives"][sel]), "seq": torch.tensor(vol["seq"]),
            "u_supervised": torch.tensor(vol["u_supervised"]), "n_instances": len(ids), "case": vol["case"]}


def crop_instances(instance, small, voxel_mm3):
    """(instance map renumbered 1..n over the instances whose in-crop part reaches the volume floor, the small mask with
    the slivers added, the kept original ids in order)."""
    floor = min_voxels_for(voxel_mm3)
    ids = [int(k) for k in np.unique(instance) if k > 0]
    kept = [k for k in ids if int((instance == k).sum()) >= floor]
    renumbered = np.zeros(instance.shape, np.int64)
    for new, old in enumerate(kept, start=1):
        renumbered[instance == old] = new
    small = small | np.isin(instance, [k for k in ids if k not in kept])
    return renumbered, small, kept


class AURDataset(Dataset):
    def __init__(self, rows, crop=CROP, crops_per_volume=2, do_augment=True, lesion_share=0.5, seed=0):
        self.rows, self.crop, self.k, self.do_augment, self.lesion_share, self.seed = list(rows), tuple(crop), crops_per_volume, do_augment, lesion_share, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        vol = load_volume(self.rows[i])
        return [make_crop(vol, rng, self.crop, self.do_augment, lesion_centred=rng.random() < self.lesion_share) for _ in range(self.k)]


def collate(items):
    """A list of crop lists -> one batch dict: fixed-size tensors stacked, per-crop targets kept as lists."""
    crops = [c for item in items for c in item]
    out = {k: torch.stack([c[k] for c in crops]) for k in ("image", "valid", "coords", "local", "entity", "entity_present", "instance", "point_weight", "a_ignore", "seq", "u_supervised")}
    for k in ("host", "host_probs", "negatives", "n_instances", "case"):
        out[k] = [c[k] for c in crops]
    return out


def event_targets_at_points(instance_pts, n_instances):
    """instance_pts (B, P) crop-local instance ids at the sampled points -> list of (N_b, P) float targets."""
    out = []
    for b, n in enumerate(n_instances):
        ids = torch.arange(1, n + 1, device=instance_pts.device)
        out.append((instance_pts[b][None, :] == ids[:, None]).float())
    return out
```

- [ ] **Step 4: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_dataset.py -q -p no:cacheprovider`
Expected: PASS, no warnings about the new code.

- [ ] **Step 5: Commit**

```bash
git add tests/test_aur_dataset.py anatobind/aur/dataset.py
git commit -m "AUR: volumes to training crops with targets; dataset and collate"
```

---

### Task 12: Preparation and probe scripts

**Files:**
- Create: `scripts/aur_prepare.py`
- Create: `scripts/aur_probe.py`
- Test: `tests/test_aur_scripts.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 9, 10; `anatobind.infer.brain_disease.check_grid`
- Produces: `aur_prepare.py` with `stage_samples(out, root, s4_cases)`, `stage_grids(rows) -> bad rows`, `host_volumes_ml(seg, spacing)`, `stage_isles_check(rows, out, n_cases, n_montage)`, `main(argv)`; `aur_probe.py` with `synthetic_batch(batch, crop, device, seed)`, `step(...)`, `main(argv)` printing one JSON line

- [ ] **Step 1: Write the failing test** `tests/test_aur_scripts.py`

```python
# tests/test_aur_scripts.py
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _volume(path, arr, spacing=(1.0, 1.0, 1.0)):
    nib.save(nib.Nifti1Image(arr, np.diag(list(spacing) + [1.0])), str(path))
    return str(path)


def test_prepare_grids_and_isles_check(tmp_path):
    p = _load("aur_prepare")
    seg = np.zeros((16, 16, 8), np.int16)
    seg[:8] = 2
    seg[8:] = 41
    img = np.ones((16, 16, 8), np.float32)
    rows = [{"case": "sub-strokecase0001", "source": "isles", "sequence": "DWI", "source_sequence": "DWI",
             "image": _volume(tmp_path / "dwi.nii.gz", img), "anatomy": _volume(tmp_path / "seg.nii.gz", seg), "lesion": None},
            {"case": "x", "source": "pdgm", "sequence": "T1", "source_sequence": "T1",
             "image": _volume(tmp_path / "t1.nii.gz", np.ones((16, 16, 9), np.float32)), "anatomy": str(tmp_path / "seg.nii.gz"), "lesion": None}]
    bad = p.stage_grids(rows)
    assert len(bad) == 1 and bad[0][0] == "x"
    assert p.host_volumes_ml(seg, (1.0, 1.0, 1.0))[0] == pytest.approx(8 * 16 * 8 / 1000)
    out = tmp_path / "check"
    table = p.stage_isles_check(rows, out, n_cases=2, n_montage=1)
    assert set(table) == {"pdgm", "bmsr", "isles"} and (out / "host_volumes_median_ml.csv").is_file() and (out / "isles_sub-strokecase0001.png").is_file()
    with pytest.raises(FileExistsError):
        p.stage_isles_check(rows, out)
    with pytest.raises(SystemExit):
        p.main(["--stage", "samples"])


def test_probe_builds_a_batch_and_steps_on_cpu_sized_inputs():
    pr = _load("aur_probe")
    b = pr.synthetic_batch(1, (8, 16, 16), torch.device("cpu"))
    assert b["image"].shape == (1, 1, 8, 16, 16) and b["instance"].max() == 2 and b["valid"][0, 0, 0, -1] == 0.0
    assert b["coords"].shape == (1, 3, 8, 16, 16) and float(b["coords"][0, 0, 1, 0, 0] - b["coords"][0, 0, 0, 0, 0]) == 1.0
    assert callable(pr.step) and pr.main.__name__ == "main"       # the CUDA path is exercised by the controller's probe run
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_scripts.py -q -p no:cacheprovider`
Expected: FAIL (ModuleNotFoundError / ImportError / AttributeError for the new names).

- [ ] **Step 3: Write `scripts/aur_prepare.py`**

```python
#!/usr/bin/env python
# scripts/aur_prepare.py
"""Data preparation and the pre-flight checks of the AnatoBind brain model (spec §3, §12 P1–P2).

  --stage samples --out <json>              the sample table with splits (refuses an existing file)
  --stage grids --samples <json>            every row: image, anatomy and lesion on one grid (prints the failures)
  --stage isles_check --samples <json> --out <dir>   host volumes of the SynthSeg maps per source and six ISLES
                                            montages (DWI with the SynthSeg map), for a human to look at

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_prepare.py --stage samples --out /data2/congcong/data/FM_data/derived/aur/samples.json
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.anatomy.sources import FM  # noqa: E402
from anatobind.aur.labels import HOST_NAMES, N_HOSTS, host_map  # noqa: E402
from anatobind.aur.samples import all_samples, assign_splits, counts, write_samples  # noqa: E402
from anatobind.infer.brain_disease import check_grid  # noqa: E402

S4_CASES = FM / "derived/brain_anatomy/cases.json"
N_VOLUME_CASES = 30
N_MONTAGE = 6


def stage_samples(out, root=FM, s4_cases=S4_CASES):
    rows = assign_splits(all_samples(root), json.loads(Path(s4_cases).read_text()))
    write_samples(out, rows)
    c = counts(rows)
    for source, d in c.items():
        print(f"{source}: train {d['train'][0]} cases / {d['train'][1]} rows / {d['train'][2]} with U; "
              f"test {d['test'][0]} cases / {d['test'][1]} rows / {d['test'][2]} with U")
    print(f"wrote {out}: {len(rows)} rows")
    return rows


def stage_grids(rows):
    """Returns the rows whose files are not on one grid."""
    bad = []
    for r in rows:
        try:
            seg = nib.load(r["anatomy"])
            check_grid([r["image"]] + ([r["lesion"]] if r["lesion"] else []), seg)
        except (ValueError, FileNotFoundError) as e:
            bad.append((r["case"], r["sequence"], str(e)))
            print(f"BAD {r['case']} {r['sequence']}: {e}")
    print(f"{len(rows)} rows checked, {len(bad)} off their grid")
    return bad


def host_volumes_ml(seg, spacing):
    hm = host_map(seg)
    ml = float(np.prod(spacing)) / 1000.0
    return [float((hm == i + 1).sum()) * ml for i in range(N_HOSTS)]


def stage_isles_check(rows, out, n_cases=N_VOLUME_CASES, n_montage=N_MONTAGE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    out.mkdir(parents=True)
    table = {}
    for source in ("pdgm", "bmsr", "isles"):
        seen, vols = set(), []
        for r in rows:
            if r["source"] != source or r["case"] in seen:
                continue
            seen.add(r["case"])
            img = nib.load(r["anatomy"])
            vols.append(host_volumes_ml(np.asarray(img.dataobj), img.header.get_zooms()[:3]))
            if len(seen) >= n_cases:
                break
        table[source] = np.median(np.array(vols), 0).tolist() if vols else [float("nan")] * N_HOSTS
    lines = ["host," + ",".join(table)] + [f"{HOST_NAMES[i]}," + ",".join(f"{table[s][i]:.1f}" for s in table) for i in range(N_HOSTS)]
    (out / "host_volumes_median_ml.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    done = 0
    for r in rows:
        if r["source"] != "isles" or r["source_sequence"] != "DWI":
            continue
        img, seg = nib.load(r["image"]), nib.load(r["anatomy"])
        data, lab = np.asarray(img.dataobj).astype(np.float32), np.asarray(seg.dataobj)
        ks = [int(data.shape[2] * f) for f in (0.3, 0.45, 0.6, 0.75)]
        fig, axes = plt.subplots(2, len(ks), figsize=(3.2 * len(ks), 6.4))
        for i, k in enumerate(ks):
            for row in range(2):
                axes[row, i].imshow(data[:, :, k].T, cmap="gray", vmin=0, vmax=np.percentile(data, 99.5), origin="lower")
                axes[row, i].axis("off")
            axes[0, i].set_title(f"slice {k}")
            axes[1, i].imshow(np.ma.masked_where(lab[:, :, k].T == 0, host_map(lab[:, :, k]).T), cmap="tab20", vmin=0, vmax=15, alpha=0.5, origin="lower")
        fig.suptitle(f"{r['case']}: DWI / SynthSeg host classes")
        fig.tight_layout()
        png = out / f"isles_{r['case']}.png"
        fig.savefig(png, dpi=72)
        plt.close(fig)
        print(f"wrote {png}")
        done += 1
        if done >= n_montage:
            break
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description="AnatoBind brain data preparation")
    ap.add_argument("--stage", choices=("samples", "grids", "isles_check"), required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--samples", type=Path)
    a = ap.parse_args(argv)
    if a.stage == "samples":
        if a.out is None:
            ap.error("--out is needed")
        stage_samples(a.out)
        return 0
    if a.samples is None:
        ap.error("--samples is needed")
    rows = json.loads(a.samples.read_text())
    if a.stage == "grids":
        return 1 if stage_grids(rows) else 0
    if a.out is None:
        ap.error("--out is needed")
    stage_isles_check(rows, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `scripts/aur_probe.py`**

```python
#!/usr/bin/env python
# scripts/aur_probe.py
"""Memory and speed probe of the AnatoBind brain model (spec §12 P3): synthetic crops of the training size, the full
loss (A + S + U + R), AMP bfloat16, gradient checkpointing; reports peak memory and seconds per step as one JSON line.

Single card:  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_probe.py --gpu 7 --batch 2 --steps 20
Four cards:   PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=4,5,6,7 nice -n 19 ~/anaconda3/envs/nvgen/bin/torchrun --nproc_per_node 4 scripts/aur_probe.py --ddp --batch 2 --steps 200

Writes nothing; the caller redirects stdout into the records folder."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.aur import losses as L  # noqa: E402
from anatobind.aur.labels import N_ENTITIES, N_HOST_CLASSES  # noqa: E402
from anatobind.aur.model import AnatoBindBrain  # noqa: E402


def synthetic_batch(batch, crop, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    D, H, W = crop
    image = torch.randn(batch, 1, D, H, W, generator=g)
    valid = torch.ones(batch, D, H, W)
    valid[:, :, :, W - W // 8:] = 0.0
    axes = [torch.arange(n, dtype=torch.float32) + 0.5 for n in crop]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0)[None].expand(batch, -1, -1, -1, -1).clone()
    local = torch.stack(torch.meshgrid(*[(a / n) * 2 - 1 for a, n in zip(axes, crop)], indexing="ij"), 0)[None].expand(batch, -1, -1, -1, -1).clone()
    entity = torch.randint(0, N_ENTITIES + 1, (batch, D, H, W), generator=g)
    instance = torch.zeros(batch, D, H, W, dtype=torch.long)
    instance[:, D // 4:D // 2, H // 4:H // 2, W // 4:W // 2] = 1
    instance[:, D // 2:3 * D // 4, H // 2:3 * H // 4, W // 2:3 * W // 4] = 2
    return {k: v.to(device) for k, v in dict(image=image, valid=valid, coords=coords, local=local, entity=entity, instance=instance).items()}


def step(model, batch, points, opt, scaler_dtype):
    out_model = model.module if hasattr(model, "module") else model
    with torch.autocast("cuda", dtype=scaler_dtype):
        out = model(batch["image"], batch["valid"], batch["coords"], batch["local"])
        em, um = out_model.entity_masks(out, points), out_model.event_masks(out, points)
        entity_pts, inst_pts = L.gather(batch["entity"], points), L.gather(batch["instance"], points)
        B = em.shape[0]
        parts = L.entity_loss(em.float(), out["entity_presence"].float(), entity_pts, torch.zeros_like(entity_pts, dtype=torch.bool),
                              torch.ones(B, N_ENTITIES, dtype=torch.bool, device=em.device))
        targets = [(inst_pts[b][None] == torch.tensor([[1], [2]], device=em.device)).float() for b in range(B)]
        u, matches = L.event_loss(out["event_presence"].float(), um.float(), targets, torch.ones_like(inst_pts, dtype=torch.float32),
                                  torch.ones(B, dtype=torch.bool, device=em.device))
        parts.update(u)
        parts.update(L.seq_loss(out["seq_logits"].float(), torch.zeros(B, dtype=torch.long, device=em.device)))
        r = {"r_host": 0.0, "r_hard": 0.0}
        for b in range(B):
            qi, ti = matches[b]
            logits = out_model.bind(out, b, qi).float()
            part = L.relation_loss(logits, torch.tensor([0, 1], device=em.device)[ti], torch.tensor([[1, -1], [0, -1]], device=em.device)[ti])
            r = {k: r[k] + part[k] / B for k in r}
        parts.update(r)
        total, _ = L.total(parts)
    opt.zero_grad(set_to_none=True)
    total.backward()
    opt.step()
    return float(total)


def main(argv=None):
    ap = argparse.ArgumentParser(description="AnatoBind brain memory / speed probe")
    ap.add_argument("--gpu", type=int, default=None, help="card for the single-card probe (ignored under --ddp)")
    ap.add_argument("--ddp", action="store_true", help="run under torchrun, one process per card")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop", type=int, nargs=3, default=(128, 160, 160))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--points", type=int, default=L.N_POINTS)
    ap.add_argument("--no-checkpoint", action="store_true")
    a = ap.parse_args(argv)
    if a.ddp:
        torch.distributed.init_process_group("nccl")
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
        device = torch.device("cuda", a.gpu if a.gpu is not None else 0)
    torch.cuda.set_device(device)
    model = AnatoBindBrain(use_checkpoint=not a.no_checkpoint).to(device)
    if a.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = synthetic_batch(a.batch, tuple(a.crop), device, seed=rank)
    points = L.sample_points(batch["valid"].cpu(), a.points, torch.Generator().manual_seed(rank)).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    t0 = time.time()
    for i in range(a.steps):
        if i == 3:                                      # the first steps carry the warm-up
            torch.cuda.synchronize(device)
            t0 = time.time()
        losses.append(step(model, batch, points, opt, torch.bfloat16))
    torch.cuda.synchronize(device)
    per_step = (time.time() - t0) / max(a.steps - 3, 1)
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
    result = {"rank": rank, "world": world, "gpu": device.index, "batch": a.batch, "crop": list(a.crop), "points": a.points,
              "checkpoint": not a.no_checkpoint, "steps": a.steps, "s_per_step": round(per_step, 3), "peak_gib": round(peak, 2),
              "first_loss": round(losses[0], 4), "last_loss": round(losses[-1], 4),
              "params": (model.module if a.ddp else model).num_parameters()}
    print(json.dumps(result))
    if a.ddp:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the test to see it pass**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_aur_scripts.py -q -p no:cacheprovider`
Expected: PASS, no warnings about the new code.

- [ ] **Step 6: Commit**

```bash
git add tests/test_aur_scripts.py scripts/aur_prepare.py scripts/aur_probe.py
git commit -m "AUR: data preparation stages and the memory / speed probe"
```

---
### Task 13 (controller): pre-flight records P0 and the probes

**Files:**
- Create: `docs/verification/2026-10-08/anatobind_brain_aur/p0/{samples.txt, grids.txt, isles_check.txt, sibbms_note.md, probe_single.txt, probe_ddp.txt}`, `docs/verification/2026-10-08/anatobind_brain_aur/p0/isles_check/` (CSV + 6 PNGs)
- Create: `/data2/congcong/data/FM_data/derived/aur/samples.json`
- Modify: `docs/superpowers/specs/2026-10-08-anatobind-brain-aur-design.md` (§6 time line and N16 with the measured step times; dated amendment)

Every command below is run by the controller, its output captured with `2>&1 | tee <record>` into a new file (the records folder is created with `mkdir -p`; a record that exists already is a reason to pick a new name, never to overwrite).

- [ ] **Step 1: the sample table** (`samples.txt`)

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_prepare.py --stage samples --out /data2/congcong/data/FM_data/derived/aur/samples.json 2>&1 | tee docs/verification/2026-10-08/anatobind_brain_aur/p0/samples.txt
```

Expected: pdgm 400 / 101 cases, bmsr 359 / 102, isles about 200 / 50 (ceil(0.2 x patients) test), sibbms 276 / 82; rows ≈ 3900 train; "with U" ≈ 1360 train rows. Compare with spec §3.2 and write the real numbers into the README of Task 14.

- [ ] **Step 2: grids** (`grids.txt`): `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_prepare.py --stage grids --samples /data2/congcong/data/FM_data/derived/aur/samples.json 2>&1 | tee docs/verification/2026-10-08/anatobind_brain_aur/p0/grids.txt`. Expected: 0 rows off their grid (S7 checked PDGM / BMSR / ISLES per case; SibBMS FLAIR and its T1w map share the template grid). A row off its grid is listed in the README and left out of training by the Part 2 loader (ruling to record).

- [ ] **Step 3: ISLES SynthSeg check** (P1; `isles_check.txt`, `isles_check/`): `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_prepare.py --stage isles_check --samples /data2/congcong/data/FM_data/derived/aur/samples.json --out docs/verification/2026-10-08/anatobind_brain_aur/p0/isles_check 2>&1 | tee docs/verification/2026-10-08/anatobind_brain_aur/p0/isles_check.txt`. Look at the six montages and the median host volumes per source. Ruling to make: ISLES keeps A supervision only if its white matter / cortex / deep grey medians lie within about 30 % of PDGM's and the montages show the hosts where the DWI shows them (USER_REPORTED). Otherwise write the ruling "ISLES: U only" into `sibbms_note.md`'s sibling `isles_ruling.md` for Part 2.

- [ ] **Step 4: SibBMS note** (`sibbms_note.md`): write down the 2026-10-08 finding — the lesion annotations exist for 10 subjects only (`SibBMS_ms/sibbms/Output/Annotation/sub-*/ses-001/sub-*_Segmentation-label.nii.gz`, native grid 201 x 261 x 261 at 1 mm, not the 197 x 233 x 189 template grid of the FLAIR), so every SibBMS row has `u_supervised = False`; the 10 subjects stay as an external U check for Part 2.

- [ ] **Step 5: single-card probe** (P3; `probe_single.txt`): on one idle card `<g>`:

```bash
for B in 1 2 4; do PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_probe.py --gpu <g> --batch $B --steps 12; done 2>&1 | tee docs/verification/2026-10-08/anatobind_brain_aur/p0/probe_single.txt
for B in 2 4; do PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_probe.py --gpu <g> --batch $B --steps 12 --no-checkpoint; done 2>&1 | tee -a docs/verification/2026-10-08/anatobind_brain_aur/p0/probe_single.txt
```

Dry-run values on an A800 (2026-10-08, GPU 7): batch 1 / 2 with checkpointing 0.52 / 0.68 s per step, 3.3 / 6.0 GiB; batch 2 / 4 without checkpointing 0.54 / 0.84 s, 9.4 / 18.4 GiB; 22 719 561 parameters. Values within ±30 % of these are expected.

- [ ] **Step 6: four-card probe** (`probe_ddp.txt`): needs four idle cards at once; wait for them, do not take a busy one. Always with a hard timeout (the dry run's first four-card attempt hung in NCCL initialisation while another session took two of the cards; the two-card run gave 0.66 s per step, the same as one card):

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=<a>,<b>,<c>,<d> NCCL_DEBUG=WARN timeout 600 nice -n 19 ~/anaconda3/envs/nvgen/bin/torchrun --nproc_per_node 4 --master_port 29571 scripts/aur_probe.py --ddp --batch 4 --steps 100 --no-checkpoint 2>&1 | tee docs/verification/2026-10-08/anatobind_brain_aur/p0/probe_ddp.txt
```

If it times out: rerun once with `NCCL_P2P_DISABLE=1` added; if that times out too, record it and fall back to two cards (`--nproc_per_node 2`) and tell the user. Expected: about 0.9 s per step at batch 4 per card.

- [ ] **Step 7: recompute the training time and amend the spec** (§6 and N16, dated 2026-10-08, "实测后重算"): with the measured seconds per step `t` at the chosen per-card batch `b` on `n` cards, Stage II = 240 000 / (n·b) steps x t, Stage III = 80 000 / (n·b) x t. With the dry-run numbers (4 x batch 4, no checkpointing, ≈ 0.9 s): Stage II 15 000 steps ≈ 3.8 h, Stage III 5 000 steps ≈ 1.3 h. Write the learning rate for the chosen effective batch into §6 (square-root scaling from 2e-4 at batch 2: 4e-4 at 8, 5.7e-4 ≈ 5e-4 at 16, with 1 000 warm-up steps). Commit the spec amendment on its own: `git add docs/superpowers/specs/2026-10-08-anatobind-brain-aur-design.md && git commit -m "Spec: measured step times and the training schedule (P3)"`.

- [ ] **Step 8: Commit the records**

```bash
git add docs/verification/2026-10-08/anatobind_brain_aur/p0
git commit -m "AUR: pre-flight records (sample table, grids, ISLES check, SibBMS note, probes)"
```

---

### Task 14 (controller): records index, documentation, whole-branch review, tag

- [ ] **Step 1: `docs/verification/2026-10-08/anatobind_brain_aur/README.md`** — one line per record in `p0/`; the sample counts per source and split (copied from `samples.txt`); the grid result; the ISLES ruling; the SibBMS note; the probe numbers and the recomputed schedule; "Known deviations" (SibBMS U unsupervised; whatever the ISLES ruling changed; any parameter of DEFAULTS that differs from the spec); pointer to Part 2.
- [ ] **Step 2: tests** — `PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`; record the count (expected 908 passed, 1 skipped).
- [ ] **Step 3: `CLAUDE.md`** — code-map line for `anatobind/aur/` (one line listing the twelve files and what each does), the two scripts, the spec, this plan and the records folder; status sentence: Part 1 done, Part 2 (training / evaluation / inference) next; test count. Change nothing else.
- [ ] **Step 4: `STATUS.md`** — rewrite with the five fixed sections (verified with commands and outputs: sample counts, grid check, ISLES ruling, probe numbers, test count; decisions for the user: the ISLES ruling if it changed the spec, the per-card batch, push, the deletion list incl. the dry-run scratch and the probe outputs; next: Part 2 plan; pitfalls: zyx order, the four-card NCCL hang and the timeout rule, the hook and the words it blocks; the why of N1–N18 in one line each).
- [ ] **Step 5: Commit** — `git add docs/verification/2026-10-08/anatobind_brain_aur/README.md CLAUDE.md STATUS.md && git commit -m "Docs: AnatoBind brain A/U/R part 1, records index, status and code map"`.
- [ ] **Step 6: whole-branch review** on the most capable model over the range from this plan's commit to HEAD (code against the spec; label consistency; the no-leak and no-mirroring properties; every record number traceable; safety: no deletion calls, outputs refused when they exist); one fix wave; scoped re-review; then tag `handoff/<date>-anatobind-brain-aur-part1` on main. No push (the user pushes).
