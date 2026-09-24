import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/brain_frame.py"
    spec = importlib.util.spec_from_file_location("brain_frame", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_freeze_t_takes_the_smallest_t_reaching_the_share_or_none():
    m = _load()
    assert m.freeze_t({2: 0.10, 3: 0.20, 4: 0.30, 5: 0.40}) == 3
    assert m.freeze_t({2: 0.16, 3: 0.20, 4: 0.30, 5: 0.40}) == 2
    assert m.freeze_t({2: 0.01, 3: 0.05, 4: 0.10, 5: 0.12}) is None


def test_series_and_geometry_strata():
    m = _load()
    assert m.series_of("file_brain_AXFLAIR_209_6001234") == "209"
    assert m.geometry_stratum(0.6875, 5.0) == "inplane_0.69_slice_5"
    assert m.geometry_stratum(0.8594, 3.0) == "inplane_0.86_slice_3"


def test_analyse_volume_flips_rows_merges_and_measures_one_lesion():
    m = _load()
    seg = np.zeros((40, 40, 4), np.int16)     # (col, row, slice)
    seg[:20, :, :] = 2                        # WM in cols 0..20
    seg[20:, :, :] = 3                        # cortex beyond
    n_rows = 40
    # CSV frame: y from the bottom. rows 10..14 from the top <=> y = 40 - 14 = 26, height 4
    rows = [{"file": "f", "slice": 1, "x": 10, "y": 26, "width": 6, "height": 4, "label": "Lacunar infarct"},
            {"file": "f", "slice": 2, "x": 10, "y": 26, "width": 6, "height": 4, "label": "Lacunar infarct"}]
    out = m.analyse_volume(seg, (0.5, 0.5, 5.0), rows, n_rows, {"file": "f", "patient_id": "p", "series": "200",
                                                                "spacing_row_mm": 0.5, "spacing_col_mm": 0.5, "spacing_slice_mm": 5.0})
    assert len(out) == 1
    L = out[0]
    assert (L["z0"], L["z1"], L["n_slices"], L["y0"], L["y1"]) == (1, 2, 2, 10, 14)
    assert L["host_lookup_all"] == 2 and L["host_class_nearest"] == "white_matter"
    assert L["d1_mm"] == 0.0 and L["d_interface_mm"] == pytest.approx((20 - 15) * 0.5) and L["delta_d_mm"] == L["d_interface_mm"]
    assert L["status"] == "ok" and L["inplane_mm"] == pytest.approx(6 * 0.5)


def test_summarise_reports_shares_per_t_and_the_majority_flag():
    m = _load()
    rows = [{"d_interface_mm": d, "delta_d_mm": d, "status": "ok", "patient_id": f"p{i % 3}", "n_slices": 1,
             "stratum_series": "200_201", "stratum_geometry": "inplane_0.69_slice_5", "host_class_nearest": "white_matter",
             "host_lookup_all": 2, "host_lookup_parenchyma": 2} for i, d in enumerate([0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 9.0, 10.0])]
    s = m.summarise(rows)
    assert s["n_lesions"] == 10 and s["n_patients"] == 3
    assert s["share_at"][2] == pytest.approx(0.3) and s["share_at"][5] == pytest.approx(0.6)
    assert s["t_frozen"] == 2 and s["majority_flag"] is False
    rows_easy = [{**r, "d_interface_mm": 9.0} for r in rows]
    assert m.summarise(rows_easy)["t_frozen"] is None
