import numpy as np

from anatobind.eval.geometry import CLASS_NAMES, class_maps, host_class_map, slot_features
from anatobind.eval.lookup import class_level_host

WM, CTX, BG = (CLASS_NAMES.index(n) + 1 for n in ("white_matter", "cortex", "basal_ganglia"))
SP = (0.5, 0.5, 5.0)


def _seg():
    seg = np.zeros((40, 40, 2), np.int16)
    seg[:20, :, :] = 2                 # WM
    seg[20:30, :, :] = 3               # cortex
    seg[30:, :, :] = 24                # CSF: not a host
    seg[10:14, 30:34, :] = 11          # a caudate block inside the WM
    seg[14:18, 30:34, :] = 12          # a putamen block: caudate + putamen = basal ganglia
    return seg


def _slots(rects):
    cm = host_class_map(_seg())
    return slot_features(cm, class_maps(cm, SP), rects, SP)


def test_class_level_argmax_sums_the_members_of_a_class():
    # cols 8..18 rows 30..34: 2 WM cols + 4 caudate + 4 putamen -> class level: basal ganglia 8/10 > WM 2/10
    c, source, frac = class_level_host(_slots([(8, 18, 30, 34, 0)]))
    assert (c, source) == (BG, "overlap") and frac == 0.8


def test_zero_overlap_takes_the_nearest_in_volume_class():
    c, source, frac = class_level_host(_slots([(34, 38, 10, 14, 0)]))          # inside CSF
    assert (c, source, frac) == (CTX, "nearest", 0.0)


def test_overlap_tie_goes_to_the_lower_class_id():
    c, source, frac = class_level_host(_slots([(16, 24, 10, 14, 0)]))          # 4 WM cols + 4 cortex cols
    assert (c, source, frac) == (WM, "overlap", 0.5)
