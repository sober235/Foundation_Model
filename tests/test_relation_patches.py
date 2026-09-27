import numpy as np
import pytest

from anatobind.eval.geometry import host_class_map, lesion_centroid
from anatobind.relation.build import lesion_patch, zscore_volume
from anatobind.relation.table import PATCH_PX, PATCH_SLICES


def _vol():
    seg = np.zeros((64, 64, 4), np.int16)
    seg[8:56, 8:56, :] = 2                                  # WM
    seg[40:56, 8:56, :] = 3                                 # cortex at cols 40..56
    rss = np.zeros((64, 64, 4), np.float32)
    rss[8:56, 8:56, :] = np.arange(8, 56, dtype=np.float32)[:, None, None]      # intensity = column index inside the brain
    return seg, rss


def test_zscore_uses_brain_voxels_only():
    seg, rss = _vol()
    z = zscore_volume(rss, seg)
    inside = z[seg > 0]
    assert abs(inside.mean()) < 1e-5 and abs(inside.std() - 1) < 1e-5 and z.dtype == np.float32
    with pytest.raises(ValueError):
        zscore_volume(np.ones((4, 4, 2), np.float32), np.ones((4, 4, 2), np.int16))     # constant brain


def test_patch_is_centred_resampled_and_edge_slices_are_replicated():
    seg, rss = _vol()
    z, cm = zscore_volume(rss, seg), host_class_map(seg)
    rects = [(30, 34, 30, 34, 0)]                           # voxels cols 30..33 rows 30..33 on slice 0; centroid (31.5, 31.5, 0)
    img, msk, cmp = lesion_patch(z, cm, rects, (0.5, 0.5, 5.0), lesion_centroid(rects))
    assert img.shape == msk.shape == cmp.shape == (PATCH_SLICES, PATCH_PX, PATCH_PX)
    assert img.dtype == np.float16 and msk.dtype == bool and cmp.dtype == np.int8
    assert np.array_equal(img[0], img[1]) and np.array_equal(cmp[0], cmp[1])           # slice -1 is clipped to slice 0
    c = PATCH_PX // 2                                       # pixel 24 sits 0.375 mm right of the centroid: col 32.25
    assert msk[1, c, c] and cmp[1, c, c] == 1               # inside the lesion, in WM (class 1)
    assert not msk[1, c + 3, c]                             # 2.625 mm = 5.25 cols away: col 36.75, outside the 30..34 box
    assert float(img[1, c + 8, c]) > float(img[1, c, c]) > float(img[1, c - 8, c])    # intensity ramps with the column
    assert cmp[1, c + 10, c] == 2 and cmp[1, c - 10, c] == 1                            # 7.875 mm right: col 47.25 = cortex; left: col 17.25 = WM
    # col 62.25 is outside the brain but inside the array: the class map is 0 and the image keeps the z-scored background
    assert cmp[1, c + 20, c] == 0 and float(img[1, c + 20, c]) == pytest.approx(float(z[62, 32, 0]), abs=1e-2)
    far = lesion_patch(z, cm, [(60, 64, 30, 34, 0)], (0.5, 0.5, 5.0), lesion_centroid([(60, 64, 30, 34, 0)]))[0]
    assert float(far[1, PATCH_PX - 1, c]) == 0.0                                         # beyond the array edge: zero-filled
