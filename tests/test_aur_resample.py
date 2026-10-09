# tests/test_aur_resample.py
import json

import nibabel as nib
import numpy as np
import pytest

import anatobind.aur.resample as R


def _las_case(tmp_path, spacing=(2.0, 2.0, 2.0), shape=(20, 16, 12)):
    """A case stored LAS at `spacing`: an image, a label map with two labels and a lesion map, all on one grid."""
    affine = np.diag([-spacing[0], spacing[1], spacing[2], 1.0])
    affine[0, 3] = (shape[0] - 1) * spacing[0]
    rng = np.random.default_rng(0)
    img = rng.uniform(100, 200, shape).astype(np.float32)
    seg = np.zeros(shape, np.int16)
    seg[:10] = 2
    seg[10:] = 41
    les = np.zeros(shape, np.uint8)
    les[4:8, 4:8, 4:8] = 1
    paths = {}
    for name, arr in (("image", img), ("anatomy", seg), ("lesion", les)):
        p = tmp_path / "src" / f"case_{name}.nii.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(arr, affine), str(p))
        paths[name] = str(p)
    return [{"case": "c1", "source": "isles", "patient": "c1", "sequence": seq, "source_sequence": seq, "image": paths["image"],
             "anatomy": paths["anatomy"], "lesion": paths["lesion"], "u_supervised": seq == "DWI", "u_values": [1] if seq == "DWI" else [],
             "a_ignore_values": [1], "a_supervised": True, "r_supervised": True, "split": "train"} for seq in ("DWI", "ADC")]


def test_spacing_rules():
    assert not R.needs_resampling((1.0, 1.0, 1.0)) and not R.needs_resampling((1.0004, 0.9996, 1.0))
    assert R.needs_resampling((0.859, 0.859, 1.5)) and R.needs_resampling((2.0, 2.0, 2.0))
    assert R.thick_slice((1.797, 1.797, 4.8)) and R.thick_slice((0.75, 0.75, 5.0)) and not R.thick_slice((2.0, 2.0, 2.0))


def test_a_case_is_resampled_to_one_millimetre_ras(tmp_path):
    rows = _las_case(tmp_path)
    out_root = tmp_path / "out"
    new = R.resample_case(rows, out_root)
    assert len(new) == 2 and new[0]["image"] != rows[0]["image"] and new[0]["anatomy"] == new[1]["anatomy"]
    img, seg, les = (nib.load(new[0][k]) for k in ("image", "anatomy", "lesion"))
    assert "".join(nib.aff2axcodes(img.affine)) == "RAS" and img.header.get_xyzt_units()[0] == "mm"
    for v in (img, seg, les):
        assert np.allclose(v.header.get_zooms()[:3], 1.0) and v.shape == img.shape and np.allclose(v.affine, img.affine)
    assert img.shape == (39, 31, 23) or all(abs(s - 2 * n) <= 2 for s, n in zip(img.shape, (20, 16, 12)))
    assert img.get_fdata().dtype == np.float64 and np.asarray(img.dataobj).dtype == np.float32
    assert set(np.unique(np.asarray(seg.dataobj))) <= {0, 2, 41} and np.asarray(seg.dataobj).dtype == np.int16
    n_les = int((np.asarray(les.dataobj) > 0).sum())
    assert 0.8 * 64 * 8 <= n_les <= 1.2 * 64 * 8                      # 64 voxels of 8 mm3 each, nearest neighbour at 1 mm
    assert new[0]["native_spacing"] == [2.0, 2.0, 2.0] and new[0]["resampled"] and new[1]["u_supervised"] is False
    with pytest.raises(FileExistsError):
        R.resample_case(rows, out_root)
    again = R.resample_case(rows, out_root, resume=True)
    assert again[0]["image"] == new[0]["image"]


def test_plan_keeps_millimetre_rows_and_applies_the_thickness_rule(tmp_path):
    rows = _las_case(tmp_path)
    one_mm = _las_case(tmp_path / "mm", spacing=(1.0, 1.0, 1.0), shape=(8, 8, 8))
    for r in one_mm:
        r["case"] = "c2"
        r["source"] = "pdgm"
    thick = _las_case(tmp_path / "thick", spacing=(1.8, 1.8, 4.8), shape=(8, 8, 4))
    for r in thick:
        r["case"] = "c3"
    todo, kept = R.plan(rows + one_mm + thick)
    assert sorted(c for c, _ in todo) == ["c1", "c3"] and [r["case"] for r in kept] == ["c2", "c2"]
    assert all(r["native_spacing"] == [1.0, 1.0, 1.0] and r["resampled"] is False for r in kept)
    flagged = R.apply_thickness_rule(thick + rows)
    by = {(r["case"], r["source_sequence"]): r for r in flagged}
    assert by[("c3", "DWI")]["thick_slice"] and not by[("c3", "DWI")]["a_supervised"] and not by[("c3", "ADC")]["r_supervised"] and by[("c3", "DWI")]["u_supervised"] and not by[("c3", "ADC")]["u_supervised"]
    assert not by[("c1", "DWI")]["thick_slice"] and by[("c1", "DWI")]["a_supervised"] and by[("c1", "ADC")]["r_supervised"]


def test_the_table_is_written_once_with_counts(tmp_path):
    rows = _las_case(tmp_path)
    out = tmp_path / "samples_1mm.json"
    counts = R.write_table(out, rows)
    assert json.loads(out.read_text()) == rows and counts["isles"]["rows"] == 2
    with pytest.raises(FileExistsError):
        R.write_table(out, rows)
