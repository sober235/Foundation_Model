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
