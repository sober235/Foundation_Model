"""Stage-II losses for the whole-volume upstream (RESEARCH_PLAN v2.2 13.6).

Anatomy: per-structure BCE + Dice over the valid (unpadded) slices, weighted by presence, plus a
presence BCE. Lesions: the centre-heatmap losses of anatobind/model/dense_head.py (focal loss on the
heatmaps, L1 on the offset and log size). Losses run in float32 whatever the forward's autocast.
"""
import torch
import torch.nn.functional as F

from anatobind.model.dense_head import centre_loss


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


def upstream_loss(out, batch):
    parts = anatomy_loss(out["masks"], out["presence"], batch["seg"], batch["present"], batch["valid_depth"])
    parts.update(centre_loss(out, batch))
    total = parts["mask"] + parts["presence"] + parts["heat"] + parts["offset"] + parts["size"]
    return total, {k: float(v) for k, v in parts.items()}
