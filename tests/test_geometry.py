import numpy as np
import pytest

from anatobind.eval.geometry import (
    CLASS_NAMES, HOST_CLASSES, LANDMARKS, class_distance_maps, host_class_map, interface_margin,
    lesion_class_distances, member_rects,
)

SP = (0.5, 0.5, 5.0)          # (col, row, slice) mm, like a fastMRI FLAIR stack


def _seg():
    """(col, row, slice) = (40, 40, 4): left WM (2) on rows 0..20 and right WM (41) on rows 20..40 for cols 0..20,
    cortex (3) for cols 20..40, a lateral ventricle (4) inside the WM at cols 2..6, rows 2..6."""
    seg = np.zeros((40, 40, 4), np.int16)
    seg[:20, :20, :] = 2
    seg[:20, 20:, :] = 41
    seg[20:, :, :] = 3
    seg[2:6, 2:6, :] = 4
    return seg


def _member(x, w, y, h, s):
    return {"x": x, "width": w, "y": y, "height": h, "slice": s}


def test_host_class_map_merges_sides_and_drops_landmarks():
    m = host_class_map(_seg())
    wm, cortex = CLASS_NAMES.index("white_matter") + 1, CLASS_NAMES.index("cortex") + 1
    assert m[10, 10, 0] == wm and m[10, 30, 0] == wm          # left and right WM are one class
    assert m[30, 10, 0] == cortex
    assert m[3, 3, 0] == 0                                     # ventricle is a landmark, not a host
    assert set(LANDMARKS) == {4, 43, 5, 44, 14, 15, 24}
    assert sum(len(v) for v in HOST_CLASSES.values()) == 25


def test_a_lesion_inside_white_matter_has_d1_zero_and_d2_the_in_plane_distance_to_cortex():
    seg = _seg()
    dist = class_distance_maps(host_class_map(seg), SP)
    rects = member_rects([_member(10, 6, 10, 6, 0)], seg.shape)        # cols 10..15, rows 10..15, slice 0
    d = lesion_class_distances(dist, rects)
    c1, d1, d2 = interface_margin(d)
    assert c1 == CLASS_NAMES.index("white_matter") + 1 and d1 == 0.0
    assert d2 == pytest.approx((20 - 15) * 0.5)                         # 5 voxels to the first cortex column


def test_a_lesion_straddling_the_boundary_has_zero_interface_distance():
    seg = _seg()
    dist = class_distance_maps(host_class_map(seg), SP)
    _, d1, d2 = interface_margin(lesion_class_distances(dist, member_rects([_member(18, 4, 10, 4, 0)], seg.shape)))
    assert d1 == 0.0 and d2 == 0.0


def test_the_ventricle_does_not_create_an_interface():
    seg = _seg()
    dist = class_distance_maps(host_class_map(seg), SP)
    _, d1, d2 = interface_margin(lesion_class_distances(dist, member_rects([_member(7, 2, 7, 2, 0)], seg.shape)))
    assert d1 == 0.0 and d2 == pytest.approx((20 - 8) * 0.5)           # cortex, 12 voxels away; the ventricle two voxels away is ignored


def test_cross_slice_neighbours_are_a_slice_thickness_away():
    seg = np.zeros((40, 40, 4), np.int16)
    seg[:, :, :] = 2
    seg[:, :, 3] = 3                                                    # cortex only on the last slice
    dist = class_distance_maps(host_class_map(seg), SP)
    _, _, d2 = interface_margin(lesion_class_distances(dist, member_rects([_member(10, 4, 10, 4, 0)], seg.shape)))
    assert d2 == pytest.approx(3 * 5.0)                                 # never <= 4 mm: t <= 4 is an in-plane criterion


def test_a_single_class_gives_an_infinite_second_distance():
    seg = np.full((10, 10, 2), 2, np.int16)
    dist = class_distance_maps(host_class_map(seg), SP)
    c1, d1, d2 = interface_margin(lesion_class_distances(dist, member_rects([_member(1, 2, 1, 2, 0)], seg.shape)))
    assert d1 == 0.0 and d2 == float("inf")


def test_member_rects_clip_to_the_grid_and_drop_boxes_off_it():
    rects = member_rects([_member(-3, 6, 38, 6, 1), _member(5, 3, 5, 3, 9)], (40, 40, 4))
    assert rects == [(0, 3, 38, 40, 1)]
