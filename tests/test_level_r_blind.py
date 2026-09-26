import pytest

from anatobind.level_r.blind import FORBIDDEN, BlindingError, assert_blind, blind_lesion, blind_volume_meta

FULL_LESION = {"lesion_id": 3, "code": "ab12cd34", "volume_code": "0f0f0f0f", "z0": 2, "z1": 3, "boxes": {"2": [[10, 14, 20, 26]]},
               "label": "Lacunar infarct", "patient_id": "dcfc", "series": "200", "stratum_geometry": "inplane_0.69_slice_5",
               "d_interface_mm": 0.0, "delta_d_mm": 0.0, "file": "file_brain_AXFLAIR_200_6002425", "host_lookup_all": 41, "band": "0"}


def test_blind_lesion_keeps_only_the_whitelist():
    out = blind_lesion(FULL_LESION)
    assert out == {"lesion_id": 3, "code": "ab12cd34", "volume_code": "0f0f0f0f", "z0": 2, "z1": 3, "boxes": {"2": [[10, 14, 20, 26]]}}
    assert out is not FULL_LESION and "label" not in out


def test_blind_volume_meta_keeps_geometry_and_window_only():
    meta = {"shape": [16, 320, 320], "spacing_slice_mm": 5.0, "spacing_row_mm": 0.6875, "spacing_col_mm": 0.6875,
            "window": [1200, 48000], "scale_p999": 0.0002, "stem": "file_brain_AXFLAIR_200_6002425", "patient_id": "x"}
    assert blind_volume_meta(meta) == {"shape": [16, 320, 320], "spacing_slice_mm": 5.0, "spacing_row_mm": 0.6875,
                                       "spacing_col_mm": 0.6875, "window": [1200, 48000]}


@pytest.mark.parametrize("key", ["label", "d_interface_mm", "delta_d_mm", "d1_mm", "stratum_geometry", "series", "patient_id",
                                 "stem", "file", "host_lookup_all", "host_class_nearest", "band"])
def test_assert_blind_catches_a_forbidden_key_anywhere(key):
    with pytest.raises(BlindingError):
        assert_blind({"lesion": {"lesion_id": 1}, "extra": [{"ok": 1}, {key: 5}]})
    assert any(f in key for f in FORBIDDEN)


def test_assert_blind_passes_clean_structures_and_returns_them():
    obj = {"lesion": blind_lesion(FULL_LESION), "answer": {"primary_host": "white_matter", "acceptable_hosts": ["white_matter"]},
           "readers": [{"primary_host": "cortex"}], "done": 3, "total": 10, "next": None}
    assert assert_blind(obj) is obj
