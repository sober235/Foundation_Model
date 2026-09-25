# tests/test_audit_fastmri_plus_boxes.py
import csv
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_fastmri_plus_boxes.py"
    spec = importlib.util.spec_from_file_location("audit_boxes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_box_contrast_is_high_on_the_bright_block_and_about_one_elsewhere():
    m = _load()
    img = np.ones((100, 100), np.float32)
    img[40:50, 60:70] = 10.0
    assert m.box_contrast(img, 40, 50, 60, 70) > 5.0
    assert m.box_contrast(img, 10, 20, 10, 20) == pytest.approx(1.0)


def test_volume_verdict_and_series():
    m = _load()
    assert m.volume_verdict([1.0, 1.1], [1.6, 1.4]) == "converted"
    assert m.volume_verdict([1.6, 1.4], [1.0, 1.1]) == "as_is"
    assert m.volume_verdict([1.0], [1.0]) == "tie"
    assert m.series_of("file_brain_AXFLAIR_203_6000123") == "203"


def test_decide_applies_the_organ_and_series_thresholds():
    m = _load()
    ok = {"knee": {"n": 100, "converted": 95, "per_series": {}},
          "brain": {"n": 100, "converted": 92, "per_series": {"200": {"n": 50, "converted": 45}, "203": {"n": 10, "converted": 8},
                                                             "205": {"n": 1, "converted": 0}}}}
    assert m.decide(ok) is True
    bad_series = {"knee": ok["knee"], "brain": {**ok["brain"], "per_series": {**ok["brain"]["per_series"], "203": {"n": 10, "converted": 5}}}}
    assert m.decide(bad_series) is False
    bad_knee = {**ok, "knee": {"n": 100, "converted": 80, "per_series": {}}}
    assert m.decide(bad_knee) is False


def test_audit_volume_converted_ratio_beats_as_is_on_a_flipped_box():
    m = _load()
    nr = 100
    rss = np.ones((5, nr, nr), np.float32)
    rss[2, 60:70, 60:70] = 10.0                    # bright block sits where the CONVERTED box lands
    row = {"slice": 2, "x": 60, "y": 30, "width": 10, "height": 10}   # as-is box (rows 30:40) is on background
    asis, conv = m.audit_volume(rss, [row])
    assert len(asis) == 1 and len(conv) == 1
    assert conv[0] > asis[0]
    assert asis[0] == pytest.approx(1.0, abs=0.05)
    assert conv[0] > 5.0


def test_audit_volume_skips_a_box_whose_slice_is_out_of_range():
    m = _load()
    rss = np.ones((3, 50, 50), np.float32)
    row = {"slice": 5, "x": 10, "y": 10, "width": 5, "height": 5}
    assert m.audit_volume(rss, [row]) == ([], [])


def test_audit_volume_skips_a_box_clipped_to_zero_area():
    m = _load()
    rss = np.ones((3, 50, 50), np.float32)
    row = {"slice": 0, "x": 50, "y": 10, "width": 5, "height": 5}     # x == n_cols: clipped column span is empty
    assert m.audit_volume(rss, [row]) == ([], [])


def test_summarise_counts_n_converted_and_per_series():
    m = _load()
    per_volume = {
        "f1": {"n_boxes": 2, "median_as_is": 1.0, "median_converted": 1.5, "verdict": "converted", "series": "200", "rss_rows": 320},
        "f2": {"n_boxes": 1, "median_as_is": 1.4, "median_converted": 1.1, "verdict": "as_is", "series": "200", "rss_rows": 320},
    }
    s = m.summarise(per_volume)
    assert s["n"] == 2
    assert s["converted"] == 1
    assert s["as_is"] == 1
    assert s["per_series"] == {"200": {"n": 2, "converted": 1, "rss_rows": [320]}}


def _write_brain_csv(path, stems, label):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "slice", "study_level", "x", "y", "width", "height", "label"])
        w.writeheader()
        for stem in stems:
            w.writerow({"file": stem, "slice": 1, "study_level": "No", "x": 10, "y": 10, "width": 5, "height": 5, "label": label})


def _write_brain_h5(root, stem, acquisition):
    d = root / "multicoil_train"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / f"{stem}.h5", "w") as h:
        h.create_dataset("reconstruction_rss", data=np.ones((3, 50, 50), np.float32))
        h.attrs["acquisition"] = acquisition


def test_audit_organ_file_filter_drops_rows_whose_file_fails_the_predicate(tmp_path):
    m = _load()
    flair, t1 = "file_brain_AXFLAIR_200_1000001", "file_brain_AXT1_200_1000002"
    csv_path = tmp_path / "brain.csv"
    _write_brain_csv(csv_path, [flair, t1], "Lacunar infarct")
    root = tmp_path / "brain_root"
    _write_brain_h5(root, flair, "AXFLAIR")
    _write_brain_h5(root, t1, "AXT1")

    per_volume = m.audit_organ(csv_path, root, {"Lacunar infarct"}, file_filter=lambda f: "AXFLAIR" in f)

    assert set(per_volume) == {flair}


def test_audit_organ_without_a_file_filter_keeps_every_matching_file(tmp_path):
    m = _load()
    flair, t1 = "file_brain_AXFLAIR_200_1000001", "file_brain_AXT1_200_1000002"
    csv_path = tmp_path / "brain.csv"
    _write_brain_csv(csv_path, [flair, t1], "Lacunar infarct")
    root = tmp_path / "brain_root"
    _write_brain_h5(root, flair, "AXFLAIR")
    _write_brain_h5(root, t1, "AXT1")

    per_volume = m.audit_organ(csv_path, root, {"Lacunar infarct"})

    assert set(per_volume) == {flair, t1}
