from collections import Counter

from anatobind.data_engine.fastmri_knee import FAMILIES, clean_boxes, merge_to_3d


def _box(file, slice_, x, y, w, h, family="meniscus"):
    return {"file": file, "slice": slice_, "x": x, "y": y, "width": w, "height": h, "family": family}


def test_degenerate_boxes_are_dropped_and_counted():
    rows = [_box("f1", 3, 10, 10, 20, 20), _box("f1", 4, 10, 10, 2, 20), _box("f1", 5, 10, 10, 20, 0)]
    kept, dropped = clean_boxes(rows)
    assert len(kept) == 1 and kept[0]["slice"] == 3
    assert dropped == Counter({"too_small": 2})


def test_overlapping_boxes_on_adjacent_slices_become_one_lesion():
    rows = [_box("f1", 3, 10, 10, 20, 20), _box("f1", 4, 11, 10, 20, 20), _box("f1", 5, 12, 11, 20, 20)]
    lesions = merge_to_3d(rows)
    assert len(lesions) == 1
    L = lesions[0]
    assert (L["z0"], L["z1"], L["n_boxes"]) == (3, 5, 3)
    assert (L["x0"], L["y0"], L["x1"], L["y1"]) == (10, 10, 32, 31)


def test_spatially_separate_runs_of_one_family_become_two_lesions():
    rows = [_box("f1", 3, 10, 10, 20, 20), _box("f1", 4, 10, 10, 20, 20),
            _box("f1", 3, 200, 200, 20, 20), _box("f1", 4, 200, 200, 20, 20)]
    lesions = merge_to_3d(rows)
    assert len(lesions) == 2
    assert sorted(L["x0"] for L in lesions) == [10, 200]


def test_a_slice_gap_breaks_a_lesion_in_two():
    rows = [_box("f1", 3, 10, 10, 20, 20), _box("f1", 5, 10, 10, 20, 20)]
    assert len(merge_to_3d(rows)) == 2


def test_a_single_slice_box_is_still_a_lesion():
    assert len(merge_to_3d([_box("f1", 7, 10, 10, 20, 20)])) == 1


def test_families_do_not_merge_into_each_other():
    rows = [_box("f1", 3, 10, 10, 20, 20, "meniscus"), _box("f1", 4, 10, 10, 20, 20, "cartilage")]
    lesions = merge_to_3d(rows)
    assert sorted(L["family"] for L in lesions) == ["cartilage", "meniscus"]
    assert set(FAMILIES) >= {"meniscus", "cartilage"}
