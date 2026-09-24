import csv

import numpy as np

from anatobind.data_engine.fastmri import merge_boxes_3d, read_fastmri_plus_rows, rows_to_rss_frame
from anatobind.data_engine.fastmri_knee import merge_to_3d, read_annotations

FIELDS = ["file", "slice", "study_level", "x", "y", "width", "height", "label"]


def _write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _csv_row(file, s, x, y, w, h, label, study="No"):
    return {"file": file, "slice": s, "study_level": study, "x": x, "y": y, "width": w, "height": h, "label": label}


def test_reader_strips_labels_and_skips_study_level_and_non_integer_rows(tmp_path):
    p = tmp_path / "knee.csv"
    _write_csv(p, [_csv_row("f1", 3, 10, 20, 5, 6, "Joint Effusion "),
                   _csv_row("f1", "", "", "", "", "", "artifact", study="Yes"),
                   _csv_row("f1", 4, "n/a", 1, 2, 2, "Meniscus Tear")])
    assert read_fastmri_plus_rows(p) == [
        {"file": "f1", "slice": 3, "x": 10, "y": 20, "width": 5, "height": 6, "label": "Joint Effusion"}]


def test_knee_reader_maps_labels_to_families_and_drops_the_rest(tmp_path):
    p = tmp_path / "knee.csv"
    _write_csv(p, [_csv_row("f1", 3, 1, 2, 5, 6, "Joint Effusion "), _csv_row("f1", 3, 1, 2, 5, 6, "Periarticular cysts")])
    rows = read_annotations(p)
    assert [(r["family"], r["label"]) for r in rows] == [("effusion", "Joint Effusion")]


def test_rows_to_rss_frame_flips_rows_per_file_and_keeps_width_and_height():
    rows = [{"file": "a", "slice": 0, "x": 10, "y": 20, "width": 5, "height": 6, "label": "L"},
            {"file": "b", "slice": 0, "x": 10, "y": 20, "width": 5, "height": 6, "label": "L"}]
    out = rows_to_rss_frame(rows, {"a": 320, "b": 276})
    assert (out[0]["x"], out[0]["y"], out[0]["width"], out[0]["height"]) == (10, 320 - 26, 5, 6)
    assert (out[1]["x"], out[1]["y"]) == (10, 276 - 26)
    assert rows_to_rss_frame(rows[:1], 320) == out[:1]
    assert rows[0]["y"] == 20                                    # input rows are not mutated


def _row(file, s, x, y, w, h, label="L"):
    return {"file": file, "slice": s, "x": x, "y": y, "width": w, "height": h, "label": label}


def test_merge_keeps_members_and_groups_by_the_given_key():
    rows = [_row("f", 3, 10, 10, 20, 20), _row("f", 4, 11, 10, 20, 20), _row("f", 4, 11, 10, 20, 20, "M")]
    lesions = merge_boxes_3d(rows, "label")
    assert sorted((L["label"], L["n_boxes"]) for L in lesions) == [("L", 2), ("M", 1)]
    L = next(L for L in lesions if L["label"] == "L")
    assert (L["z0"], L["z1"], L["x0"], L["y0"], L["x1"], L["y1"]) == (3, 4, 10, 10, 31, 30)
    assert [m["slice"] for m in L["members"]] == [3, 4]


def test_flipping_rows_commutes_with_merging():
    """A row flip is a reflection: in-plane IoU and slice adjacency are unchanged, so
    merge(flip(rows)) must equal flip(merge(rows)) box for box."""
    rng = np.random.default_rng(0)
    rows = [_row("f", int(rng.integers(0, 6)), int(rng.integers(0, 300)), int(rng.integers(0, 300)),
                 int(rng.integers(3, 20)), int(rng.integers(3, 20)), str(rng.choice(["L", "M"]))) for _ in range(120)]
    n = 320
    merged_after_flip = merge_boxes_3d(rows_to_rss_frame(rows, n), "label")
    merged_before_flip = merge_boxes_3d(rows, "label")

    def key(L):
        return (L["label"], L["z0"], L["z1"], L["x0"], L["x1"], L["n_boxes"])

    assert sorted((key(L), L["y0"], L["y1"]) for L in merged_after_flip) == \
        sorted((key(L), n - L["y1"], n - L["y0"]) for L in merged_before_flip)


def test_knee_merge_to_3d_is_the_generic_merge_keyed_by_family():
    rows = [{**_row("f", 3, 10, 10, 20, 20), "family": "meniscus"}, {**_row("f", 4, 10, 10, 20, 20), "family": "meniscus"}]
    lesions = merge_to_3d(rows)
    assert [L["family"] for L in lesions] == ["meniscus"] and lesions[0]["n_boxes"] == 2
