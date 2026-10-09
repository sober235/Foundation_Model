"""Training targets from the pseudo-labels (spec 2026-10-08 §4.2–4.3): lesion instances, each instance's soft host
distribution, its primary host and its hard negatives. The host truth is the overlap of the instance with the host
classes of the SynthSeg map (plan §13); it is a pseudo target, NOT_EVIDENCE."""
import numpy as np
from scipy import ndimage

from anatobind.aur.labels import N_HOSTS, N_HOST_CLASSES, NEAR_MM, NO_HOST, contralateral, host_map
from anatobind.eval.lesion_components import component_rows, components


def lesion_instances(lesion, values, voxel_mm3):
    """lesion: the source label map; values: the label values that are lesion on this sequence.
    Returns (instance map int32 with ids 1..N, 0 elsewhere; boolean mask of the components under the volume floor).
    Components under MIN_MM3 are neither instances nor background: callers leave them out of the loss."""
    mask = np.isin(np.asarray(lesion), list(values)) if len(values) else np.zeros(np.shape(lesion), bool)
    comp, n = components(mask)
    inst = np.zeros(comp.shape, np.int32)
    small = np.zeros(comp.shape, bool)
    k = 0
    for r in component_rows(comp, n, voxel_mm3, "lesion"):
        sel = comp == r["component"]
        if r["ignore"]:
            small |= sel
        else:
            k += 1
            inst[sel] = k
    return inst, small


def host_targets(inst, seg, spacing):
    """{"probs": (N, 14) float32, "host": (N,) int64, "negatives": (N, 2) int64} for the instances 1..N of `inst`.
    probs: the share of the instance's voxels in each host class (N9); without any overlap the nearest host within
    NEAR_MM gets 1, else NO_HOST gets 1. host = argmax. negatives (N10): the contralateral host and the host with the
    second-largest share; -1 when there is none (brainstem has no contralateral; a single-host lesion has no second)."""
    hm = host_map(seg)
    n = int(inst.max()) if inst.size else 0
    probs = np.zeros((n, N_HOST_CLASSES), np.float32)
    host = np.full(n, NO_HOST, np.int64)
    neg = np.full((n, 2), -1, np.int64)
    nearest = None
    for k in range(1, n + 1):
        sel = inst == k
        counts = np.bincount(hm[sel].astype(np.int64), minlength=N_HOSTS + 1)[1:]
        if counts.sum() > 0:
            probs[k - 1, :N_HOSTS] = counts / counts.sum()
        else:
            if nearest is None:
                if (hm > 0).any():
                    dist, idx = ndimage.distance_transform_edt(hm == 0, sampling=spacing, return_indices=True)
                    nearest = (dist, hm[tuple(idx)])
                else:
                    nearest = (None, None)
            dist, lab = nearest
            if dist is not None and float(dist[sel].min()) <= NEAR_MM:
                j = int(np.argmin(dist[sel]))
                probs[k - 1, int(lab[sel][j]) - 1] = 1.0
            else:
                probs[k - 1, NO_HOST] = 1.0
        host[k - 1] = int(np.argmax(probs[k - 1]))
        if host[k - 1] != NO_HOST:
            c = contralateral(int(host[k - 1]))
            neg[k - 1, 0] = -1 if c is None else c
            order = np.argsort(-probs[k - 1, :N_HOSTS], kind="stable")
            second = int(order[1])
            neg[k - 1, 1] = second if probs[k - 1, second] > 0 else -1
    return {"probs": probs, "host": host, "negatives": neg}
