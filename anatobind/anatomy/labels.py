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
