"""Pure metric regressions for the frozen AUR acceptance protocol."""
import importlib

import numpy as np
import pytest


def api():
    try:
        return importlib.import_module('anatobind.aur.gate_metrics')
    except ModuleNotFoundError:
        pytest.fail('The gate metric implementation is missing')


def test_presence_and_visible_area_use_truth_denominators():
    s = dict(labels=[2, 3, 4], truth=[100, 0, 20], predicted=[5, 7, 0], intersection=[0, 0, 0])
    assert api().class_presence(s) == [True, None, False]
    assert api().visible_classes(s, .25) == [True, False, False]


def test_host_macro_excludes_absent_truth_and_empty_support():
    h = dict(labels=['a', 'b', 'c'], truth=[10, 0, 5], predicted=[10, 3, 5],
             intersection=[8, 0, 2], dice=[.8, 0., .4])
    assert api().observation_host_macro(h) == pytest.approx(.6)
    assert api().host_dice_by_host(h) == [.8, None, .4]
    assert api().observation_host_macro(dict(h, truth=[0, 0, 0], intersection=[0, 0, 0])) is None


def test_matched_dice_is_full_mask_overlap_for_only_matched_pairs():
    gt = np.zeros((1, 4, 4), np.int32); gt[0, :2, :2] = 1; gt[0, 3, 3] = 2
    pred = np.zeros_like(gt); pred[0, :2, :1] = 7
    assert api().matched_instance_dice(pred, gt, {'1': 7}) == {1: pytest.approx(2 / 3)}
    with pytest.raises(ValueError): api().matched_instance_dice(pred, gt, {2: 99})


def test_planar_host_reference_retains_nearest_and_no_host():
    seg = np.zeros((1, 40, 40), np.int16); seg[0, :, :20] = 2
    inst = np.zeros_like(seg); inst[0, 5, 25] = 1; inst[0, 5, 39] = 2
    shares = api().host_shares(inst, seg, (1., 1., 1.))
    assert shares.shape == (2, 14)
    assert shares.argmax(1).tolist() == [0, 13]
    assert api().top_share(shares).tolist() == [1., 1.]


def test_stride_and_half_up_centroid_planes():
    assert api().stride_planes(16, 5.) == list(range(0, 16, 2))
    assert api().stride_planes(160, 1.)[-1] == 152
    assert api().stride_planes(3, 20.) == [0, 1, 2]
    inst = np.zeros((6, 4, 4), np.int32); inst[2:6, 0, 0] = 1; inst[0, 3, 3] = 2
    assert api().lesion_centroid_planes(inst) == {1: 4, 2: 0}


@pytest.mark.parametrize('spacing', [0, -1, float('nan'), float('inf')])
def test_stride_rejects_unusable_physical_spacing(spacing):
    with pytest.raises(ValueError): api().stride_planes(16, spacing)


def test_nonfinite_or_misaligned_support_is_rejected():
    s = dict(labels=[2], truth=[100], predicted=[float('nan')], intersection=[0])
    with pytest.raises(ValueError): api().class_presence(s)
    with pytest.raises(ValueError): api().visible_classes(dict(s, predicted=[]), 1.)
