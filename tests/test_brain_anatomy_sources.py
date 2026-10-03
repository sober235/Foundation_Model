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
