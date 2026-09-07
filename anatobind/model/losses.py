"""Losses for the arm-B minimal path (spec 5.3, restricted to L_A + L_UB + L_rel).

Boxes are ``(cz, cy, cx, dz, dy, dx)`` normalised to the crop extent.

The relation target is built from ``host_label`` alone -- the annotator's
``tissue_id``, side-resolved by the data engine (spec 5.1).  ``IoA`` is allowed
as an input feature to the relation module but must never define the truth; a
truth derived from overlap hands the M1 comparison to the seg-then-lookup
baseline by construction.  ``build_relation_targets`` therefore takes no
geometry, and a test asserts it never will.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

CLS_WEIGHT = 1.0
L1_WEIGHT = 5.0
GIOU_WEIGHT = 2.0
LAMBDA_REL = 1.0  # spec 5.3
NO_OBJECT_WEIGHT = 0.1


def box_corners(b):
    """(cz, cy, cx, dz, dy, dx) -> (z0, y0, x0, z1, y1, x1)."""
    centre, size = b[..., :3], b[..., 3:]
    return torch.cat([centre - size / 2, centre + size / 2], -1)


def giou3d(a, b):
    """Generalised IoU on corner boxes; 1 when identical, negative when far apart."""
    lo = torch.max(a[..., :3], b[..., :3])
    hi = torch.min(a[..., 3:], b[..., 3:])
    inter = (hi - lo).clamp(min=0).prod(-1)
    va = (a[..., 3:] - a[..., :3]).clamp(min=0).prod(-1)
    vb = (b[..., 3:] - b[..., :3]).clamp(min=0).prod(-1)
    union = (va + vb - inter).clamp(min=1e-7)
    hull = (torch.max(a[..., 3:], b[..., 3:]) - torch.min(a[..., :3], b[..., :3]))
    hull = hull.clamp(min=0).prod(-1).clamp(min=1e-7)
    return inter / union - (hull - union) / hull


def hungarian_match(logits, boxes, tgt_classes, tgt_boxes):
    """One sample: (M, C+1) logits and (M, 6) boxes against N targets."""
    if tgt_classes.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=logits.device)
        return empty, empty
    prob = logits.softmax(-1)
    cost_cls = -prob[:, tgt_classes]
    cost_l1 = torch.cdist(boxes, tgt_boxes, p=1)
    cost_giou = -giou3d(box_corners(boxes)[:, None], box_corners(tgt_boxes)[None])
    cost = CLS_WEIGHT * cost_cls + L1_WEIGHT * cost_l1 + GIOU_WEIGHT * cost_giou
    pi, ti = linear_sum_assignment(cost.detach().cpu().numpy())
    return (torch.as_tensor(pi, dtype=torch.long, device=logits.device),
            torch.as_tensor(ti, dtype=torch.long, device=logits.device))


def a_loss(masks, presence, seg, present):
    """Identity-anchored: query k is segmentation label k+1, no matching."""
    K = masks.shape[1]
    tgt = torch.stack([(seg == k + 1).float() for k in range(K)], 1)
    bce = F.binary_cross_entropy_with_logits(masks, tgt, reduction="none").flatten(2).mean(-1)
    p = masks.sigmoid()
    inter = (p * tgt).flatten(2).sum(-1)
    denom = p.flatten(2).sum(-1) + tgt.flatten(2).sum(-1)
    dice = 1 - (2 * inter + 1) / (denom + 1)
    w = present.float()
    return {
        "mask": ((bce + dice) * w).sum() / w.sum().clamp(min=1.0),
        "presence": F.binary_cross_entropy_with_logits(presence, w),
    }


def ub_loss(logits, boxes, tgt_classes, tgt_boxes, matches):
    """DETR-style: class CE over every query, box terms over matched queries."""
    B, M, C = logits.shape
    device = logits.device
    target = torch.full((B, M), C - 1, dtype=torch.long, device=device)
    l1, giou, n_box = logits.new_zeros(()), logits.new_zeros(()), 0
    for b, (pi, ti) in enumerate(matches):
        if pi.numel() == 0:
            continue
        target[b, pi] = tgt_classes[b][ti]
        pred, tgt = boxes[b][pi], tgt_boxes[b][ti]
        l1 = l1 + (pred - tgt).abs().sum()
        giou = giou + (1 - giou3d(box_corners(pred), box_corners(tgt))).sum()
        n_box += pi.numel()
    weight = torch.ones(C, device=device)
    weight[-1] = NO_OBJECT_WEIGHT
    n = max(n_box, 1)
    return {
        "cls": F.cross_entropy(logits.reshape(-1, C), target.reshape(-1), weight=weight),
        "l1": l1 / n,
        "giou": giou / n,
    }


def build_relation_targets(matches, host_labels, present, K, M, device):
    """Relation truth from host_label only -- deliberately blind to geometry.

    ``host_label == UNKNOWN_HOST`` (0) means the annotator's side could not be
    resolved.  Unknown is not a label: those queries are dropped from the host
    CE (ignore_index) and from the relation BCE mask rather than being bound to
    a fabricated structure.
    """
    B = len(matches)
    host_target = torch.full((B, M), -100, dtype=torch.long, device=device)
    rel_target = torch.zeros(B, K, M, device=device)
    rel_mask = torch.zeros(B, K, M, device=device)
    for b, (pi, ti) in enumerate(matches):
        if pi.numel() == 0:
            continue
        host = host_labels[b][ti].to(device)
        known = host > 0
        if not bool(known.any()):
            continue
        pi, host = pi[known], host[known] - 1
        host_target[b, pi] = host
        rel_mask[b][:, pi] = present[b].float()[:, None]
        rel_target[b, host, pi] = 1.0
    return host_target, rel_target, rel_mask


def rel_loss(host_logits, R, host_target, rel_target, rel_mask):
    C = host_logits.shape[-1]
    ce = F.cross_entropy(host_logits.reshape(-1, C), host_target.reshape(-1),
                         ignore_index=-100)
    if not torch.isfinite(ce):  # no matched query in this batch
        ce = host_logits.new_zeros(())
    bce = F.binary_cross_entropy_with_logits(R, rel_target, reduction="none")
    return {"host_ce": ce, "rel_bce": (bce * rel_mask).sum() / rel_mask.sum().clamp(min=1.0)}


def total_loss(parts, lambda_rel=LAMBDA_REL):
    total = (
        parts["mask"] + parts["presence"]
        + CLS_WEIGHT * parts["cls"] + 5.0 * parts["l1"] + 2.0 * parts["giou"]
        + lambda_rel * (parts["host_ce"] + parts["rel_bce"])
    )
    return total, {k: float(v) for k, v in parts.items()}
