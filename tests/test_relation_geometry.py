import numpy as np
import pytest

from anatobind.eval.geometry import (
    CANDIDATE_MM, DIST_CAP_MM, CLASS_NAMES, LEFT_LABELS, RIGHT_LABELS, SLOT_FIELDS, class_maps, family_sides,
    host_class_map, landmark_map, lesion_centroid, lesion_mask, min_in_lesion, side_of, slot_features, third,
)

SP = (0.5, 0.5, 5.0)          # (col, row, slice) mm
WM, CTX = CLASS_NAMES.index("white_matter") + 1, CLASS_NAMES.index("cortex") + 1


def _seg():
    """(col, row, slice) = (40, 40, 4): left WM (2) cols 0..20 rows 0..20, right WM (41) cols 0..20 rows 20..40,
    cortex (3) cols 20..40; a lateral ventricle (4) at cols 2..6 rows 2..6."""
    seg = np.zeros((40, 40, 4), np.int16)
    seg[:20, :20, :] = 2
    seg[:20, 20:, :] = 41
    seg[20:, :, :] = 3
    seg[2:6, 2:6, :] = 4
    return seg


def test_lesion_mask_and_centroid_cover_the_union_of_rects():
    bbox, m = lesion_mask([(10, 14, 10, 12, 1), (12, 16, 10, 12, 2)])
    assert bbox == (10, 16, 10, 12, 1, 3) and m.shape == (6, 2, 2) and m.sum() == 16
    c = lesion_centroid([(10, 14, 10, 12, 1)])
    assert np.allclose(c, [11.5, 10.5, 1.0])


def test_slot_features_inside_white_matter():
    seg = _seg()
    cm = host_class_map(seg)
    maps = class_maps(cm, SP)
    f = slot_features(cm, maps, [(10, 14, 10, 14, 1)], SP)           # voxels cols 10..13, rows 10..13, slice 1
    assert set(f) == set(range(1, 8)) and set(f[WM]) == set(SLOT_FIELDS)
    assert f[WM]["ioa"] == 1.0 and f[WM]["min_surface_mm"] == 0.0 and f[WM]["signed_surface_mm"] < 0
    assert f[WM]["candidate"] and f[WM]["in_volume"]
    assert f[CTX]["ioa"] == 0.0 and f[CTX]["min_surface_mm"] == pytest.approx(3.5)     # col 13 -> col 20 = 7 px * 0.5 mm
    assert f[CTX]["dx_mm"] == pytest.approx(4.0) and f[CTX]["dy_mm"] == 0.0 and f[CTX]["dz_mm"] == 0.0
    # centroid voxel = rint(11.5, 11.5, 1) = (12, 12, 1); nearest cortex voxel is (20, 12, 1): 8 px * 0.5 mm along cols
    assert f[CTX]["centroid_distance_mm"] == pytest.approx(4.0)
    assert 0 < f[CTX]["soft_overlap"] < f[WM]["soft_overlap"] == 1.0
    absent = f[CLASS_NAMES.index("cerebellum") + 1]
    assert not absent["in_volume"] and not absent["candidate"] and absent["min_surface_mm"] == DIST_CAP_MM


def test_candidate_fallback_marks_the_nearest_class_when_none_is_within_15mm():
    seg = np.zeros((80, 20, 2), np.int16)
    seg[:4, :, :] = 2                                    # WM only at cols 0..4
    cm = host_class_map(seg)
    f = slot_features(cm, class_maps(cm, (1.0, 1.0, 5.0)), [(60, 64, 5, 9, 0)], (1.0, 1.0, 5.0))
    assert f[WM]["min_surface_mm"] == DIST_CAP_MM and f[WM]["candidate"]           # 57 mm away, capped, still the fallback candidate
    assert sum(v["candidate"] for v in f.values()) == 1


def _side_seg():
    """left family (2) at cols 0..20, right family (41) at cols 20..40, cortex (3) on rows 36..40, ventricle (4) at cols 2..6 rows 2..6."""
    seg = np.zeros((40, 40, 2), np.int16)
    seg[:20, :, :] = 2
    seg[20:, :, :] = 41
    seg[:, 36:, :] = 3
    seg[2:6, 2:6, :] = 4
    return seg


def test_side_by_label_family_follows_the_column_position_not_the_label_name():
    seg = _side_seg()
    fam = family_sides(seg)
    assert fam == {"left": "image_left", "right": "image_right"}
    assert side_of(seg, [(2, 8, 10, 14, 0)], fam) == "image_left"
    assert side_of(seg, [(30, 36, 10, 14, 0)], fam) == "image_right"
    assert side_of(seg, [(16, 24, 10, 14, 0)], fam) == "midline"                     # 4 cols of each family
    flipped = seg[::-1, :, :].copy()                                                   # label 2 now at large cols
    assert family_sides(flipped) == {"left": "image_right", "right": "image_left"}
    assert side_of(flipped, [(2, 8, 10, 14, 0)], family_sides(flipped)) == "image_left"


def test_side_falls_back_to_the_nearest_host_voxel_when_the_lesion_has_none():
    seg = _side_seg()
    assert side_of(seg, [(3, 5, 3, 5, 0)], family_sides(seg)) == "image_left"        # inside the ventricle, label 2 around it


def test_image_left_is_the_screen_left_of_the_reading_tool():
    """P8: image_left = the smaller RSS column. The Level R page paints a (slices, rows, cols) u16 slice into an
    ImageData of width C (columns) with no mirror, so column 0 is the left edge of the reader's screen."""
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "anatobind/level_r/app/app.js").read_text(encoding="utf-8")
    assert "const [S, R, C] = v.meta.shape, img = new ImageData(C, R)" in js
    assert "scale(-1" not in js and "rotate(" not in js


def test_landmark_distance_to_lateral_ventricles():
    seg = _seg()
    d = landmark_map(seg, SP)
    assert min_in_lesion(d, [(2, 6, 2, 6, 0)]) == 0.0
    assert min_in_lesion(d, [(10, 12, 2, 6, 0)]) == pytest.approx(2.5)          # col 10 -> col 5 = 5 px * 0.5 mm
    assert landmark_map(np.zeros((4, 4, 2), np.int16), SP) is None
    assert min_in_lesion(None, [(0, 1, 0, 1, 0)]) == DIST_CAP_MM


def test_third_splits_a_range_into_named_bins():
    names = ("top", "middle", "bottom")
    # t = (i - lo) / (hi - lo); first third is t < 1/3, second is t < 2/3: 3/9 == 1/3 exactly, so index 3 is "middle"
    assert [third(i, 0, 9, names) for i in (0, 2, 3, 4, 6, 9)] == ["top", "top", "middle", "middle", "bottom", "bottom"]
    assert third(5, 5, 5, names) == "middle"
