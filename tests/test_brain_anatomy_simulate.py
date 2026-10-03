# tests/test_brain_anatomy_simulate.py
import numpy as np
import pytest

import anatobind.anatomy.simulate as SIM
from anatobind.anatomy.labels import IGNORE, to_student


def _brain(shape=(60, 70, 80)):
    """A 1 mm RAS 'brain': an ellipsoid of left (2 / 3) and right (41 / 42) white matter and cortex with ventricles (4) in the
    middle, sitting on the lower two thirds of the volume; the FLAIR image follows the labels."""
    x, y, z = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    cx, cy, cz = 29.5, 34.5, 32.0
    rad = ((x - cx) / 25) ** 2 + ((y - cy) / 30) ** 2 + ((z - cz) / 28) ** 2
    seg = np.zeros(shape, np.int16)
    inside, shell = rad <= 1.0, (rad <= 1.0) & (rad > 0.7)
    seg[inside & (x < cx)] = 2             # RAS: axis 0 grows towards the patient's right, so the left half has low indices
    seg[inside & (x >= cx)] = 41
    seg[shell & (x < cx)] = 3
    seg[shell & (x >= cx)] = 42
    seg[(rad <= 0.05)] = 4
    img = np.zeros(shape, np.float32)
    img[np.isin(seg, (2, 41))] = 300.0
    img[np.isin(seg, (3, 42))] = 450.0
    img[seg == 4] = 80.0
    return img, seg


def test_the_fastmri_frame_flips_left_and_right_only():
    img, seg = _brain()
    f = SIM.to_fastmri_frame(seg)
    assert f.shape == seg.shape and np.array_equal(f, seg[::-1])
    left = np.argwhere(f == 2)[:, 0].mean()
    right = np.argwhere(f == 41)[:, 0].mean()
    assert left > right                                         # patient left now at high axis-0 indices, as in fastMRI


def test_parameters_are_deterministic_and_inside_the_measured_ranges():
    a, b = SIM.sample_params(np.random.default_rng(3)), SIM.sample_params(np.random.default_rng(3))
    assert a == b
    tops = [SIM.sample_params(np.random.default_rng(i))["empty_top"] for i in range(400)]
    assert set(tops) <= {1, 2, 3, 4, 5} and 2 <= float(np.median(tops)) <= 3
    for i in range(50):
        p = SIM.sample_params(np.random.default_rng(i))
        assert p["n_slices"] in (14, 16) and p["inplane_mm"] in (0.6875, 0.86) and p["matrix"] in ((320, 320), (260, 320), (276, 276))
        assert abs(p["theta_lr"]) <= 10 and abs(p["theta_ap"]) <= 5 and 0.7 <= p["gamma"] <= 1.4


def test_bottom_slice_and_slab_groups():
    profile = [0] * 20 + [600] * 70 + [0] * 10              # brain in 1 mm slices 20..89: top index 89
    assert SIM.top_index(profile) == 89 and SIM.top_index([0, 100]) == -1
    assert SIM.bottom_index(profile, 16, 2) == 89 + 1 + 10 - 80          # 20: the stack ends 2 slices above the brain
    assert SIM.bottom_index(profile, 16, 3) == 25 and SIM.bottom_index(profile, 14, 3) == 35
    assert SIM.bottom_index([600] * 30, 16, 2) == 0                       # a short brain: the bottom is clamped
    groups = SIM.slab_groups(4, 16, 30)
    assert len(groups) == 16 and groups[0] == (4, 9) and groups[5] == (29, 30) and groups[6] == (30, 30)   # beyond the top: empty
    assert SIM.slab_groups(0, 14, 100)[-1] == (65, 70)


def test_votes_average_and_ignore_on_a_tiny_stack():
    lab = np.zeros((2, 2, 7), np.uint8)
    lab[0, 0, :] = [1, 1, 2, 2, 2, 0, 0]          # 2 wins the first group (3 of 5)
    lab[0, 1, :] = [1, 1, 2, 2, 9, 0, 0]          # tie 1 vs 2 in slices 0-4: the centre slice (2) holds 2 -> 2
    lab[1, 0, :] = [3, 0, 0, 0, 3, 14, 14]        # 0 wins (3 of 5); second group (5, 7) -> 14
    img = np.zeros((2, 2, 7), np.float32)
    img[1, 1, :] = [10, 20, 30, 40, 50, 100, 200]
    groups = SIM.slab_groups(0, 3, 7)
    assert groups == [(0, 5), (5, 7), (7, 7)]
    v = SIM.vote_labels(lab, groups)
    assert v[0, 0, 0] == 2 and v[0, 1, 0] == 2 and v[1, 0, 0] == 0 and v[1, 0, 1] == 14 and v[:, :, 2].max() == 0
    a = SIM.average_slices(img, groups)
    assert a[1, 1, 0] == pytest.approx(30.0) and a[1, 1, 1] == pytest.approx(150.0) and a[1, 1, 2] == 0.0
    mask = np.zeros((2, 2, 7), np.uint8)
    mask[0, 0, 4] = 1                             # one lesion slice is enough
    ig = SIM.ignore_any(mask, groups)
    assert ig[0, 0, 0] and not ig[0, 0, 1] and not ig[0, 1, 0]


def test_fit_to_matrix_crops_and_pads_around_the_centre():
    s = np.arange(5 * 6 * 2, dtype=np.float32).reshape(5, 6, 2)
    out = SIM.fit_to_matrix(s, (3, 10), (2.0, 3.0))
    assert out.shape == (3, 10, 2) and out[0, 0, 0] == 0.0 and out[1, 2, 0] == s[2, 0, 0]       # x: rows 1..3 kept; y padded by 2
    back = SIM.fit_to_matrix(out, (5, 6), (1.5, 5.0))
    assert back.shape == (5, 6, 2) and back[2, 0, 0] == s[2, 0, 0] and back[0, :, :].max() == 0.0


def test_rotation_by_zero_is_identity_and_small_tilts_keep_the_brain():
    img, seg = _brain()
    stu = to_student(seg)
    c = np.argwhere(stu > 0).mean(axis=0)
    assert np.array_equal(SIM.rotate(stu, 0.0, 0.0, c, order=0), stu)
    r = SIM.rotate(stu, 10.0, -5.0, c, order=0)
    assert r.dtype == stu.dtype and abs(int((r > 0).sum()) - int((stu > 0).sum())) < 0.03 * (stu > 0).sum()
    assert np.allclose(np.argwhere(r > 0).mean(axis=0), c, atol=1.5)


def test_simulate_end_to_end_gives_a_fastmri_shaped_stack():
    img, seg = _brain()
    stu = to_student(seg)
    lesion = np.zeros(seg.shape, np.uint8)
    lesion[30:36, 30:40, 20:40] = 1
    params = {"theta_lr": 4.0, "theta_ap": -2.0, "empty_top": 3, "n_slices": 16, "inplane_mm": 0.6875, "matrix": (260, 320),
              "gamma": 1.0, "blur": 0.3, "noise": 0.02, "bias": [0.2, -0.1, 0.3, 0.0, 0.1, -0.2]}
    stack, labels, out = SIM.simulate(img, stu, lesion, np.random.default_rng(0), params)
    assert stack.shape == (260, 320, 16) and labels.shape == (260, 320, 16) and stack.dtype == np.float32 and labels.dtype == np.uint8
    assert set(np.unique(labels)) <= set(range(16)) and (labels == IGNORE).any()
    assert out["bottom_slice_1mm"] >= 0 and labels[:, :, -1].max() == 0                      # the top slices are above the brain
    present = [k for k in range(16) if (labels[:, :, k] > 0).any()]
    assert 11 <= len(present) <= 13 and present[0] <= 1 and 16 - 1 - present[-1] in (3, 4)          # 3 empty slabs above the vertex (4 when the top slab is thin)
    assert out["bottom_area_share"] is None or 0.0 <= out["bottom_area_share"] <= 1.0
    wl, wr = np.argwhere(labels == 1)[:, 0].mean(), np.argwhere(labels == 2)[:, 0].mean()
    assert wl > wr                                                                            # fastMRI frame: left at high indices
    brain = (labels > 0) & (labels != IGNORE)
    assert np.isfinite(stack).all() and 900 < np.percentile(stack[brain], 99) <= 1000.5 and np.percentile(stack[brain], 1) >= -0.5
    aff = SIM.stack_affine(0.6875, stack.shape)
    assert np.allclose(np.diag(aff)[:3], [-0.6875, -0.6875, 5.0])
    stack2, labels2, _ = SIM.simulate(img, stu, lesion, np.random.default_rng(0), params)
    assert np.array_equal(labels, labels2) and np.allclose(stack, stack2)
    with pytest.raises(ValueError, match="differ"):
        SIM.simulate(img[:-1], stu, None, np.random.default_rng(0), params)
