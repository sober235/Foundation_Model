# tests/test_aur_targets.py
import numpy as np

import anatobind.aur.targets as T
from anatobind.aur.labels import HOST_NAMES, NO_HOST


def _seg():
    """(40, 20, 6) 1 mm map: left white matter (2) at x < 20, right (41) at x >= 20, a ventricle (4) block, and a
    background corridor at y >= 16 that is more than 10 mm from any host."""
    seg = np.zeros((40, 20, 6), np.int16)
    seg[:20, :5, :] = 2
    seg[20:, :5, :] = 41
    seg[8:12, 5:8, :] = 4             # ventricle beside the left white matter (host within 10 mm)
    return seg


def test_lesion_instances_drop_tiny_components_but_mark_them():
    les = np.zeros((40, 20, 6), np.uint8)
    les[2:5, 1:4, 1:4] = 1            # 27 voxels
    les[30, 1, 1] = 1                 # 1 voxel: under 10 mm3 at 1 mm3 voxels
    les[10, 10, 5] = 2                # a value that is not lesion on this sequence
    inst, small = T.lesion_instances(les, (1,), 1.0)
    assert inst.dtype == np.int32 and inst.max() == 1 and (inst > 0).sum() == 27
    assert small.sum() == 1 and small[30, 1, 1] and not inst[10, 10, 5]
    inst0, small0 = T.lesion_instances(les, (), 1.0)
    assert inst0.max() == 0 and not small0.any()


def test_host_targets_overlap_nearest_and_none():
    seg = _seg()
    inst = np.zeros(seg.shape, np.int32)
    inst[2:6, 0:3, :] = 1                      # inside the left white matter
    inst[17:23, 0:3, 0:2] = 2                  # 3 columns left (18 voxels), 3 right (18): 50 / 50
    inst[9:11, 5:7, :] = 3                     # inside the ventricle, 1 mm from the left white matter
    inst[30:34, 17:20, :] = 4                  # background, 12+ mm from any host
    out = T.host_targets(inst, seg, (1.0, 1.0, 1.0))
    wl, wr = HOST_NAMES.index("white_matter_left"), HOST_NAMES.index("white_matter_right")
    assert out["probs"].shape == (4, 14) and out["host"].tolist()[0] == wl and out["probs"][0, wl] == 1.0
    assert out["probs"][1, wl] == 0.5 and out["probs"][1, wr] == 0.5 and out["host"][1] == wl    # tie: the lower index
    assert out["host"][2] == wl and out["probs"][2, wl] == 1.0                                   # nearest within 10 mm
    assert out["host"][3] == NO_HOST and out["probs"][3, NO_HOST] == 1.0
    assert out["negatives"][0].tolist() == [wr, -1]              # contralateral, no second host
    assert out["negatives"][1].tolist() == [wr, wr]              # the second-largest host is the contralateral one here
    assert out["negatives"][3].tolist() == [-1, -1]


def test_host_targets_without_instances_or_without_hosts():
    seg = _seg()
    out = T.host_targets(np.zeros(seg.shape, np.int32), seg, (1.0, 1.0, 1.0))
    assert out["probs"].shape == (0, 14) and out["host"].shape == (0,) and out["negatives"].shape == (0, 2)
    inst = np.zeros(seg.shape, np.int32)
    inst[0:3, 0:3, 0:3] = 1
    out = T.host_targets(inst, np.zeros(seg.shape, np.int16), (1.0, 1.0, 1.0))
    assert out["host"].tolist() == [NO_HOST]
