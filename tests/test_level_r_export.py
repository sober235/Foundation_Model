import json

import h5py
import numpy as np
import pytest

from anatobind.level_r.blind import assert_blind
from anatobind.level_r.export import (
    boxes_by_slice, export_all, match_registry, merged_lesions, read_volume, small_lesion_rows, to_u16,
)
from anatobind.level_r.registry import lesion_code, volume_code

HEADER = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding>'
    "<encodedSpace><matrixSize><x>64</x><y>40</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>28</x><y>17.5</y><z>4.5</z></fieldOfView_mm></encodedSpace>"
    "<reconSpace><matrixSize><x>32</x><y>32</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>14</x><y>14</y><z>3</z></fieldOfView_mm></reconSpace>"
    "</encoding></ismrmrdHeader>"
)
STEM = "file_brain_AXFLAIR_200_1"


def _h5(path, patient="P1", slices=4, size=32, seed=0):
    rss = np.random.default_rng(seed).uniform(1e-5, 4e-4, (slices, size, size)).astype(np.float32)
    with h5py.File(path, "w") as h:
        h.create_dataset("reconstruction_rss", data=rss)
        h.create_dataset("ismrmrd_header", data=np.bytes_(HEADER.encode()))
        h.attrs["patient_id"] = patient
        h.attrs["acquisition"] = "AXFLAIR"
    return path, rss


def _csv_row(s, x, y, w, h, label="Lacunar infarct", file=STEM):
    return {"file": file, "slice": s, "x": x, "y": y, "width": w, "height": h, "label": label}


# CSV frame y counts from the bottom of a 32-row image: y=6, h=4 -> RSS rows [32-6-4, 32-6) = [22, 26)
CSV_ROWS = [_csv_row(1, 10, 6, 6, 4), _csv_row(2, 10, 6, 6, 4)]
REGISTRY = [{"lesion_id": 7, "file": STEM, "patient_id": "P1", "label": "Lacunar infarct",
             "z0": 1, "z1": 2, "x0": 10, "y0": 22, "x1": 16, "y1": 26}]


def test_small_lesion_filter_matches_gate05():
    rows = [_csv_row(1, 0, 0, 5, 5), _csv_row(1, 0, 0, 2, 5), _csv_row(1, 0, 0, 5, 5, label="Mass"),
            _csv_row(1, 0, 0, 5, 5, file="file_brain_AXT1_200_1"), _csv_row(1, 0, 0, 5, 5, label="Nonspecific white matter lesion")]
    assert small_lesion_rows(rows) == [rows[0], rows[4]]


def test_merge_and_boxes_by_slice_are_in_the_rss_frame():
    (L,) = merged_lesions(CSV_ROWS, n_rows=32)
    assert (L["z0"], L["z1"], L["x0"], L["y0"], L["x1"], L["y1"]) == (1, 2, 10, 22, 16, 26)
    assert boxes_by_slice(L["members"]) == {"1": [[22, 26, 10, 16]], "2": [[22, 26, 10, 16]]}


def test_match_registry_is_one_to_one_on_label_and_box():
    lesions = merged_lesions(CSV_ROWS, 32)
    assert match_registry(REGISTRY, lesions)[7] is lesions[0]
    with pytest.raises(ValueError):
        match_registry([{**REGISTRY[0], "x0": 11}], lesions)                       # box differs
    with pytest.raises(ValueError):
        match_registry([{**REGISTRY[0], "label": "Nonspecific white matter lesion"}], lesions)   # label differs


def test_to_u16_maps_p999_to_full_scale_and_reports_the_window():
    v = np.linspace(0.0, 1.0, 1001, dtype=np.float32).reshape(1, 7, 143)
    u, info = to_u16(v)
    assert u.dtype == np.dtype("<u2") and u.shape == v.shape
    assert info["scale_p999"] == pytest.approx(0.999) and u.max() == 65535 and u.min() == 0
    assert u.ravel()[500] == round(0.5 / 0.999 * 65535)
    lo, hi = info["window"]
    assert 0 <= lo < hi <= 65535 and lo == round(0.01 / 0.999 * 65535) and hi == round(0.995 / 0.999 * 65535)


def test_export_all_writes_blind_volumes_and_lesions_that_match_the_registry(tmp_path):
    h5, rss = _h5(tmp_path / f"{STEM}.h5")
    out = tmp_path / "level_r"
    recs = export_all(REGISTRY, CSV_ROWS + [_csv_row(1, 0, 0, 5, 5, label="Mass")], out, h5_of=lambda stem: h5, log=lambda *a: None)
    code = volume_code(STEM)
    assert recs == [{"lesion_id": 7, "code": lesion_code(7), "volume_code": code, "z0": 1, "z1": 2,
                     "boxes": {"1": [[22, 26, 10, 16]], "2": [[22, 26, 10, 16]]}}]
    assert json.loads((out / "lesions.json").read_text()) == recs
    u, meta = read_volume(out / "volumes", code)
    assert u.shape == (4, 32, 32) and u.dtype == np.dtype("<u2") and u.max() == 65535
    assert meta["shape"] == [4, 32, 32] and meta["spacing_slice_mm"] == 3.0 and meta["spacing_row_mm"] == pytest.approx(0.4375)
    assert len(meta["window"]) == 2 and (out / "volumes" / f"{code}.u16").stat().st_size == 4 * 32 * 32 * 2
    text = (out / "lesions.json").read_text() + (out / "volumes" / f"{code}.json").read_text()
    for leak in ("Lacunar", "label", "P1", "patient", STEM, "stratum", "series"):
        assert leak not in text
    assert_blind(recs)
    assert_blind(meta)


def test_export_all_refuses_a_registry_that_does_not_match_the_csv(tmp_path):
    h5, _ = _h5(tmp_path / f"{STEM}.h5")
    with pytest.raises(ValueError):
        export_all([{**REGISTRY[0], "z1": 3}], CSV_ROWS, tmp_path / "out", h5_of=lambda stem: h5, log=lambda *a: None)
