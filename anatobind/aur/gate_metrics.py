"""Pure measurements for registered AUR pseudo-reference acceptance gates."""
import math

import numpy as np

from anatobind.aur.labels import discrete_labels
from anatobind.aur.targets import host_targets

PRESENCE_FRACTION = .05
VISIBLE_AREA_MM2 = 20.
STRIDE_MM = 8.
AMBIGUOUS_SHARE = .6


def _positive(value):
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError('Invalid physical measurement')
    return float(value)


def support_arrays(supports):
    """Check finite, aligned nonnegative counts without exposing record values."""
    labels = supports['labels']
    arrays = [np.asarray(supports[k], dtype=np.float64) for k in ('truth', 'predicted', 'intersection')]
    if any(a.shape != (len(labels),) or not np.isfinite(a).all() or (a < 0).any()
           or (a != np.floor(a)).any() for a in arrays):
        raise ValueError('Invalid anatomy support')
    truth, predicted, intersection = arrays
    if (intersection > np.minimum(truth, predicted)).any():
        raise ValueError('Invalid anatomy intersection')
    return truth, predicted, intersection


def class_presence(supports, *, fraction=PRESENCE_FRACTION):
    _positive(fraction)
    truth, predicted, _ = support_arrays(supports)
    return [bool(p >= fraction * t) if t > 0 else None for t, p in zip(truth, predicted)]


def visible_classes(supports, pixel_area_mm2, *, min_area_mm2=VISIBLE_AREA_MM2):
    truth, _, _ = support_arrays(supports)
    return (truth * _positive(pixel_area_mm2) >= _positive(min_area_mm2)).tolist()


def host_dice_by_host(hosts):
    truth, predicted, intersection = support_arrays(hosts)
    values = hosts.get('dice')
    if values is not None and len(values) != len(truth):
        raise ValueError('Invalid anatomy Dice')
    out = []
    for i, (t, p, inter) in enumerate(zip(truth, predicted, intersection)):
        if t == 0:
            out.append(None)
            continue
        value = 2 * inter / (t + p)
        if values is not None:
            recorded = values[i]
            if (recorded is None or isinstance(recorded, bool) or not math.isfinite(recorded)
                    or not math.isclose(value, recorded, rel_tol=0, abs_tol=1e-12)):
                raise ValueError('Invalid anatomy Dice')
        out.append(float(value))
    return out


def observation_host_macro(hosts):
    values = [v for v in host_dice_by_host(hosts) if v is not None]
    return float(np.mean(values)) if values else None


def _instances(values):
    array = discrete_labels(values, dtype=np.int64)
    if array.ndim != 3 or (array < 0).any():
        raise ValueError('Invalid instance grid')
    return array


def matched_instance_dice(pred_instances, gt_instances, matches):
    pred, gt = _instances(pred_instances), _instances(gt_instances)
    if pred.shape != gt.shape:
        raise ValueError('Invalid matched grids')
    result = {}; used = set()
    for truth_id, predicted_id in matches.items():
        k, j = int(truth_id), int(predicted_id)
        if k <= 0 or j <= 0 or k in result or j in used or not (gt == k).any() or not (pred == j).any():
            raise ValueError('Invalid instance match')
        a, b = gt == k, pred == j
        result[k] = float(2 * np.count_nonzero(a & b) / (np.count_nonzero(a) + np.count_nonzero(b)))
        used.add(j)
    return result


def host_shares(instances, seg_synthseg, spacing):
    inst = _instances(instances)
    seg = discrete_labels(seg_synthseg)
    if seg.shape != inst.shape or len(spacing) != 3:
        raise ValueError('Invalid relation grid')
    spacing = tuple(_positive(s) for s in spacing)
    ids = np.unique(inst[inst > 0])
    if ids.size and not np.array_equal(ids, np.arange(1, int(ids[-1]) + 1)):
        raise ValueError('Relation instance IDs must be contiguous')
    return host_targets(inst, seg, spacing)['probs']


def top_share(probs):
    values = np.asarray(probs, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 14 or not np.isfinite(values).all()
            or (values < 0).any() or not np.allclose(values.sum(1), 1, atol=1e-6)):
        raise ValueError('Invalid relation shares')
    return values.max(1)


def stride_planes(depth, slice_spacing_mm, *, stride_mm=STRIDE_MM):
    if isinstance(depth, bool) or int(depth) != depth or depth <= 0:
        raise ValueError('Invalid observation depth')
    step = max(1, int(round(_positive(stride_mm) / _positive(slice_spacing_mm))))
    return list(range(0, int(depth), step))


def lesion_centroid_planes(instances):
    inst = _instances(instances)
    result = {}
    for k in np.unique(inst[inst > 0]):
        centre = float(np.nonzero(inst == k)[0].mean())
        result[int(k)] = int(np.clip(np.floor(centre + .5), 0, inst.shape[0] - 1))
    return result
