# tests/test_aur_ssl_samples.py
import json

import nibabel as nib
import numpy as np
import pytest

import anatobind.aur.ssl.samples as S


def _nii(path, spacing=(0.7, 0.7, 0.7), shape=(4, 4, 4)):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), np.diag(list(spacing) + [1.0])), str(path))
    return str(path)


def _fm(tmp_path):
    """An FM root with three HCP subjects: two complete, one without a SynthSeg map."""
    root = tmp_path / "FM"
    for subj in ("100206", "100307", "100408"):
        for seq in ("T1w", "T2w"):
            _nii(root / "HCP_lh_T1T2" / subj / "T1w" / f"{seq}_acpc_dc_restore_brain.nii")
        if subj != "100408":
            _nii(root / "derived/synthseg/hcp/seg_native" / f"{subj}_T1w_T1w_acpc_dc_restore_brain_seg.nii.gz")
    return root


def _aur_rows():
    rows = []
    for source, patients in (("pdgm", [f"P{i}" for i in range(12)]), ("isles", [f"S{i}" for i in range(5)])):
        for p in patients:
            split = "test" if p.endswith("1") else "train"
            for seq in ("T1", "FLAIR") if source == "pdgm" else ("DWI",):
                rows.append({"case": p, "source": source, "patient": p, "sequence": seq, "source_sequence": seq, "image": f"/x/{p}_{seq}.nii.gz",
                             "anatomy": f"/x/{p}_seg.nii.gz", "lesion": f"/x/{p}_les.nii.gz", "u_supervised": seq != "T1", "u_values": [1],
                             "a_ignore_values": [1], "a_supervised": True, "r_supervised": True, "split": split,
                             "native_spacing": [1.0, 1.0, 1.0] if source == "pdgm" else [2.0, 2.0, 2.0], "resampled": source != "pdgm", "thick_slice": False})
    return rows


def test_hcp_rows_need_both_sequences_and_the_synthseg_map(tmp_path):
    root = _fm(tmp_path)
    rows, skipped = S.hcp_samples(root)
    assert len(rows) == 4 and skipped == ["100408"]
    r = {(x["patient"], x["sequence"]): x for x in rows}
    assert r[("100206", "T1")]["source"] == "hcp" and r[("100206", "T1")]["case"] == "HCP_100206" and r[("100206", "T2")]["source_sequence"] == "T2w"
    assert r[("100206", "T1")]["anatomy"].endswith("100206_T1w_T1w_acpc_dc_restore_brain_seg.nii.gz") and r[("100206", "T2")]["anatomy"] == r[("100206", "T1")]["anatomy"]
    assert r[("100307", "T2")]["lesion"] is None and not r[("100307", "T2")]["u_supervised"] and r[("100307", "T2")]["a_supervised"] and not r[("100307", "T2")]["r_supervised"]
    assert r[("100307", "T2")]["split"] == "train" and r[("100307", "T2")]["native_spacing"] == [pytest.approx(0.7)] * 3


def test_validation_patients_are_a_tenth_of_the_train_patients_per_source():
    rows = _aur_rows()
    val = S.validation_patients(rows, share=0.1, seed=0)
    train_pdgm = {r["patient"] for r in rows if r["source"] == "pdgm" and r["split"] == "train"}        # 10 patients -> ceil(1.0) = 1
    assert set(val) == {"pdgm", "isles"} and len(val["pdgm"]) == 1 and len(val["isles"]) == 1
    assert set(val["pdgm"]) <= train_pdgm and val == S.validation_patients(rows, share=0.1, seed=0)
    assert val != S.validation_patients(rows, share=0.1, seed=1) or len(train_pdgm) <= 2


def test_ssl_rows_hold_out_test_patients_and_mark_the_validation_ones(tmp_path):
    aur = _aur_rows()
    hcp, _ = S.hcp_samples(_fm(tmp_path))
    val = S.validation_patients(aur + hcp, share=0.5, seed=0)
    ssl = S.ssl_rows(aur, hcp, val)
    assert not any(r["split"] == "test" for r in ssl) and {r["ssl_split"] for r in ssl} == {"train", "val"}
    assert all(r["ssl_split"] == ("val" if r["patient"] in val[r["source"]] else "train") for r in ssl)
    assert sum(r["source"] == "hcp" for r in ssl) == 4 and all("ssl_split" in r for r in ssl)
    report = S.leakage_report(ssl, aur)
    assert report["ok"] and report["test_patient_rows_in_ssl"] == 0 and report["val_patient_rows_in_ssl_train"] == 0 and report["duplicate_images"] == 0
    assert report["test_patients"] == len({r["patient"] for r in aur if r["split"] == "test"})
    leaked = ssl + [{**aur[0], "patient": "P1", "ssl_split": "train", "image": "/x/P1_T1.nii.gz"}]
    bad = S.leakage_report(leaked, aur)
    assert not bad["ok"] and bad["test_patient_rows_in_ssl"] == 1
    inv = S.inventory(ssl)
    keys = {(r["source"], r["sequence"], r["ssl_split"]) for r in inv}
    assert ("hcp", "T1", "train") in keys or ("hcp", "T1", "val") in keys
    pdgm_train = [r for r in inv if r["source"] == "pdgm" and r["sequence"] == "T1" and r["ssl_split"] == "train"][0]
    assert pdgm_train["n_rows"] == pdgm_train["n_patients"] and pdgm_train["spacing"] == "1.0x1.0x1.0" and pdgm_train["thick_rows"] == 0


def test_the_manifest_directory_is_written_once(tmp_path):
    aur = _aur_rows()
    val = S.validation_patients(aur, share=0.1, seed=0)
    ssl = S.ssl_rows(aur, [], val)
    out = tmp_path / "ssl_manifest_v1"
    paths = S.write_manifest(out, ssl, val, S.leakage_report(ssl, aur), S.inventory(ssl))
    assert set(paths) == {"samples_ssl", "val_patients", "split_leakage_report", "data_inventory"}
    assert json.loads((out / "val_patients.json").read_text()) == val and (out / "data_inventory.csv").read_text().startswith("source,")
    assert json.loads((out / "samples_ssl.json").read_text()) == ssl
    with pytest.raises(FileExistsError):
        S.write_manifest(out, ssl, val, {}, [])
