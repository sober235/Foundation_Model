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
