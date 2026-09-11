"""Detections, class-agnostic IoU matching and the four outcome buckets (RESEARCH_PLAN v2.2 13.6)."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from anatobind.eval.lookup import FAMILY_OF_LABEL, FAMILY_OF_TISSUE

MISS, WRONG_CLASS, WRONG_HOST, CORRECT = "miss", "wrong_class", "wrong_host", "correct"
BUCKETS = (CORRECT, WRONG_HOST, WRONG_CLASS, MISS)


def iou3d(a, b):
    """(N, 6) and (M, 6) corner boxes (z0, y0, x0, z1, y1, x1) -> (N, M) IoU."""
    a = np.asarray(a, dtype=float).reshape(-1, 6)
    b = np.asarray(b, dtype=float).reshape(-1, 6)
    lo = np.maximum(a[:, None, :3], b[None, :, :3])
    hi = np.minimum(a[:, None, 3:], b[None, :, 3:])
    inter = np.clip(hi - lo, 0, None).prod(-1)
    va = np.clip(a[:, 3:] - a[:, :3], 0, None).prod(-1)
    vb = np.clip(b[:, 3:] - b[:, :3], 0, None).prod(-1)
    union = va[:, None] + vb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def detected(cls_prob):
    """DETR default: a query is a detection when its most probable class is not no-object (the last)."""
    return np.flatnonzero(cls_prob.argmax(-1) != cls_prob.shape[-1] - 1)


def match(gt_boxes, pred_boxes, thr):
    """One-to-one Hungarian on IoU, class-agnostic; pairs below thr are dropped. {gt index: pred index}."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return {}
    iou = iou3d(gt_boxes, pred_boxes)
    rows, cols = linear_sum_assignment(-iou)
    return {int(g): int(p) for g, p in zip(rows, cols) if iou[g, p] >= thr}


def bucket(gt_cls, tissue_id, pred_cls, b0_label):
    """Family-level outcome of one in-segmentation lesion."""
    if pred_cls is None:
        return MISS
    if pred_cls != gt_cls:
        return WRONG_CLASS
    if not isinstance(b0_label, (int, np.integer)) or FAMILY_OF_LABEL[int(b0_label)] != FAMILY_OF_TISSUE[tissue_id]:
        return WRONG_HOST
    return CORRECT
