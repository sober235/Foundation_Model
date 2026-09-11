"""Stage-II losses for the whole-volume upstream (RESEARCH_PLAN v2.2 13.6).

Anatomy: per-structure BCE + Dice over the valid (unpadded) slices, weighted by presence, plus a
presence BCE. Lesions: the DETR set loss on the final decoder layer and on every auxiliary layer,
each layer Hungarian-matched on its own. Losses run in float32 whatever the forward's autocast.
"""
import torch
import torch.nn.functional as F

from anatobind.model.armb import boxes_vox_to_norm
from anatobind.model.losses import CLS_WEIGHT, GIOU_WEIGHT, L1_WEIGHT, hungarian_match, ub_loss


def valid_slices(valid_depth, depth, device):
    """(B, Z) bool: True on slices that hold image, False on the depth padding."""
    return torch.arange(depth, device=device)[None, :] < valid_depth.to(device)[:, None]


def anatomy_loss(masks, presence, seg, present, valid_depth):
    B, K, Z, Y, X = masks.shape
    vz = valid_slices(valid_depth, Z, masks.device)[:, None, :, None, None].float()
    tgt = torch.stack([(seg == k + 1).float() for k in range(K)], 1)
    logits = masks.float()
    n_vox = (valid_depth.to(masks.device).float() * Y * X)[:, None]
    bce = (F.binary_cross_entropy_with_logits(logits, tgt, reduction="none") * vz).sum((2, 3, 4)) / n_vox
    p = logits.sigmoid() * vz
    inter = (p * tgt).sum((2, 3, 4))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3, 4)) + (tgt * vz).sum((2, 3, 4)) + 1)
    w = present.float()
    return {"mask": ((bce + dice) * w).sum() / w.sum().clamp(min=1.0),
            "presence": F.binary_cross_entropy_with_logits(presence.float(), w)}


def _weighted(parts):
    return CLS_WEIGHT * parts["cls"] + L1_WEIGHT * parts["l1"] + GIOU_WEIGHT * parts["giou"]


def detection_loss(logits, boxes, tgt_classes, tgt_norm):
    matches = [hungarian_match(logits[b], boxes[b], tgt_classes[b], tgt_norm[b]) for b in range(logits.shape[0])]
    return ub_loss(logits, boxes, tgt_classes, tgt_norm, matches)


def upstream_loss(out, batch):
    device = out["masks"].device
    shape = out["masks"].shape[2:]
    parts = anatomy_loss(out["masks"], out["presence"], batch["seg"], batch["present"], batch["valid_depth"])
    tgt_classes = [c.to(device) for c in batch["box_classes"]]
    tgt_norm = [boxes_vox_to_norm(b.to(device).float(), shape) for b in batch["boxes"]]
    final = detection_loss(out["logits"].float(), out["boxes"].float(), tgt_classes, tgt_norm)
    parts.update(final)
    aux = sum((_weighted(detection_loss(a["logits"].float(), a["boxes"].float(), tgt_classes, tgt_norm))
               for a in out.get("aux", [])), torch.zeros((), device=device))
    parts["aux"] = aux
    total = parts["mask"] + parts["presence"] + _weighted(final) + aux
    return total, {k: float(v) for k, v in parts.items()}
