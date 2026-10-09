# tests/test_aur_crops.py
import numpy as np
import pytest

import anatobind.aur.crops as C


def test_zyx_order_and_spacing():
    a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    z = C.to_zyx(a)
    assert z.shape == (4, 3, 2) and z[1, 2, 0] == a[0, 2, 1] and z.flags["C_CONTIGUOUS"]
    assert C.spacing_zyx((0.5, 0.75, 5.0)) == (5.0, 0.75, 0.5)


def test_crop_window_inside_centred_and_pushed_back():
    rng = np.random.default_rng(0)
    w = C.crop_window((50, 60, 70), (16, 32, 32), rng)
    assert all(0 <= a and b <= n for (a, b), n in zip(w, (50, 60, 70))) and all(b - a == c for (a, b), c in zip(w, (16, 32, 32)))
    w = C.crop_window((10, 60, 70), (16, 32, 32), rng)
    assert w[0] == (-3, 13)                                   # smaller volume: the crop is centred on it
    w = C.crop_window((50, 60, 70), (16, 32, 32), rng, centre=(2, 59, 35))
    assert w[0] == (0, 16) and w[1] == (28, 60) and w[2] == (19, 51)   # pushed back inside where it fits


def test_extract_fills_the_outside_and_marks_validity():
    a = np.arange(5 * 6 * 7, dtype=np.int16).reshape(5, 6, 7)
    crop, valid = C.extract(a, [(-2, 4), (3, 9), (0, 7)], fill=-7)
    assert crop.shape == (6, 6, 7) and valid.shape == (6, 6, 7)
    assert crop[0, 0, 0] == -7 and not valid[0, 0, 0] and valid[2, 0, 0] and crop[2, 0, 0] == a[0, 3, 0]
    assert valid.sum() == 4 * 3 * 7 and crop[valid].tolist() == a[0:4, 3:6, :].ravel().tolist()
    crop, valid = C.extract(a, [(100, 104), (0, 6), (0, 7)])
    assert not valid.any() and (crop == 0).all()


def test_coordinates_physical_and_local():
    mm = C.coordinates_mm([(-1, 2), (0, 2), (4, 6)], (5.0, 1.0, 0.5))
    assert mm.shape == (3, 3, 2, 2) and mm[0, :, 0, 0].tolist() == [-2.5, 2.5, 7.5] and mm[2, 0, 0, :].tolist() == [2.25, 2.75]
    loc = C.local_coordinates([(0, 4), (0, 2), (2, 4)], (4, 2, 4))
    assert np.allclose(loc[0, :, 0, 0], [-0.75, -0.25, 0.25, 0.75]) and np.allclose(loc[2, 0, 0, :], [0.25, 0.75])


def test_normalise_augment_and_rotation():
    rng = np.random.default_rng(1)
    img = np.zeros((8, 16, 16), np.float32)
    img[2:6, 4:12, 4:12] = rng.uniform(100, 200, (4, 8, 8))
    n = C.normalise(img)
    assert n.dtype == np.float32 and n[0, 0, 0] == -1.0 and -1.0 <= n.min() and n.max() <= 1.0 and n[2:6, 4:12, 4:12].mean() > -0.5
    assert (C.normalise(np.zeros((2, 2, 2))) == -1.0).all()
    a = C.augment(n, rng, p=1.0)
    assert a.shape == n.shape and a.dtype == np.float32 and (a[n == -1.0] == -1.0).all() and not np.allclose(a, n)
    labels = np.zeros((8, 16, 16), np.int32)
    labels[2:6, 4:12, 4:12] = 3
    img_r, lab_r = C.rotate_inplane([n, labels], 10.0, [1, 0])
    assert img_r.shape == n.shape and lab_r.dtype == np.int32 and set(np.unique(lab_r)) <= {0, 3} and lab_r.sum() > 0
    same = C.rotate_inplane([labels], 0.0, [0])[0]
    assert (same == labels).all()


def test_physical_coordinates_follow_the_affine():
    window = [(0, 2), (1, 3), (-1, 1)]                                  # (z, y, x) voxel ranges, x partly outside
    affine = np.array([[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 20.0], [0.0, 0.0, 2.0, 30.0], [0.0, 0.0, 0.0, 1.0]])
    c = C.coordinates_mm(window, affine=affine)
    assert c.shape == (3, 2, 2, 2) and c.dtype == np.float32
    assert c[0, 1, 0, 0] - c[0, 0, 0, 0] == 2.0 and c[1, 0, 1, 0] - c[1, 0, 0, 0] == 1.0 and c[2, 0, 0, 1] - c[2, 0, 0, 0] == 1.0
    assert c[2, 0, 0, 0] == 9.0 and c[1, 0, 0, 0] == 21.0 and c[0, 0, 0, 0] == 30.0      # x = -1 + 10, y = 1 + 20, z = 0 + 30
    oblique = affine.copy()
    oblique[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])      # a 90 degree turn of the axes
    o = C.coordinates_mm(window, affine=oblique)
    assert o[2, 0, 0, 0] == 9.0 and o[1, 0, 0, 0] == 19.0                                  # world x = -y_vox + 10 = 9, world y = x_vox + 20 = 19
    legacy = C.coordinates_mm(window, (2.0, 1.0, 1.0))
    assert legacy[0, 0, 0, 0] == 1.0 and legacy[2, 0, 0, 0] == -0.5
    with pytest.raises(ValueError):
        C.coordinates_mm(window)
    with pytest.raises(ValueError):
        C.coordinates_mm(window, affine=np.eye(3))
