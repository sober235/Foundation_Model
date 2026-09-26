import csv

import pytest

from anatobind.level_r.registry import BANDS, distance_band, is_3mm, lesion_code, load_registry, volume_code

HEADER = ["lesion_id", "file", "patient_id", "series", "stratum_series", "stratum_geometry", "label", "z0", "z1", "n_slices",
          "x0", "y0", "x1", "y1", "inplane_mm", "spacing_row_mm", "spacing_col_mm", "spacing_slice_mm", "host_lookup_all",
          "host_lookup_parenchyma", "host_class_nearest", "d1_mm", "d_interface_mm", "delta_d_mm", "status"]


def _row(lesion_id, file="file_brain_AXFLAIR_200_1", patient="P1", d_interface=0.0, stratum="inplane_0.69_slice_5", status="ok"):
    return dict(zip(HEADER, [lesion_id, file, patient, "200", "200_201", stratum, "Nonspecific white matter lesion", 2, 2, 1,
                             91, 182, 98, 188, 4.8125, 0.6875, 0.6875, 5.0, 41, 41, "white_matter", 0.0, d_interface,
                             d_interface, status]))


def write_registry(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    return path


def test_load_registry_parses_numbers_and_adds_the_band(tmp_path):
    p = write_registry(tmp_path / "lesions.csv", [_row(0, d_interface=0.0), _row(1, d_interface=5.09), _row(2, status="outside")])
    reg = load_registry(p)
    assert [r["lesion_id"] for r in reg] == [0, 1]                     # non-ok rows are dropped
    assert reg[0]["z0"] == 2 and isinstance(reg[0]["x1"], int) and reg[0]["spacing_row_mm"] == 0.6875
    assert reg[0]["band"] == "0" and reg[1]["band"] == ">4"
    assert reg[0]["patient_id"] == "P1" and reg[0]["file"] == "file_brain_AXFLAIR_200_1"


def test_load_registry_refuses_duplicate_ids(tmp_path):
    p = write_registry(tmp_path / "lesions.csv", [_row(3), _row(3)])
    with pytest.raises(ValueError):
        load_registry(p)


@pytest.mark.parametrize("d, band", [(0.0, "0"), (-0.0, "0"), (0.001, "0-2"), (2.0, "0-2"), (2.01, "2-4"), (4.0, "2-4"), (4.01, ">4"), (30.0, ">4")])
def test_distance_band_edges_follow_v26_item_9(d, band):
    assert distance_band(d) == band and band in BANDS


def test_is_3mm_reads_the_geometry_stratum():
    assert is_3mm("inplane_0.62_slice_3") and is_3mm("inplane_0.86_slice_3") and not is_3mm("inplane_0.69_slice_5")


def test_codes_are_eight_hex_chars_stable_and_distinct():
    a, b = volume_code("file_brain_AXFLAIR_200_6002425"), volume_code("file_brain_AXFLAIR_200_6002426")
    assert len(a) == 8 and int(a, 16) >= 0 and a != b and a == volume_code("file_brain_AXFLAIR_200_6002425")
    assert lesion_code(7) == lesion_code("7") and lesion_code(7) != lesion_code(8) and len(lesion_code(7)) == 8
    assert "6002425" not in a                                            # the stem is not recoverable by eye
