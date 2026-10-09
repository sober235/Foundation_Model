"""Losses (spec N11, §6): L = L_A + L_S + L_U + λ_R (L_bind + λ_h L_hard).

Masks are supervised at sampled points (uniform over the valid voxels of each crop, the same points for every query of
a sample), so no dense (Q, D, H, W) mask is ever built in training. Entity masks: BCE + Dice for the entities present
in the crop, presence BCE for all; lesion voxels (a_ignore) carry no entity loss. Events: Hungarian matching on
presence + mask BCE + mask Dice, then presence BCE over every query (no-object weight) and mask losses over the matched
ones; samples whose U is not supervised contribute nothing to L_U. Relation: cross-entropy over the 14 host classes
plus a hinge on the hard negatives (contralateral, second host)."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from anatobind.aur.labels import N_HOST_CLASSES

NO_OBJECT_WEIGHT = 0.1
LAMBDA_R = 1.0
LAMBDA_H = 0.2
MARGIN = 1.0
N_POINTS = 16384


def sample_points(valid, n, generator=None, instance=None, focus_share=0.30, boundary_share=0.20):
    """Mix valid uniform points, per-instance lesion points and peri-lesion background.

    Where lesions are absent, sample uniformly; each represented lesion receives
    a positive-point quota if n is sufficient.
    """
    mask = valid > 0.5
    flat = mask.flatten(1).float()
    if not bool((flat.sum(-1) > 0).all()):
        raise ValueError("crop contains no valid voxels")
    out = torch.multinomial(flat, n, replacement=True, generator=generator)
    if instance is None:
        return out
    if instance.shape != valid.shape:
        raise ValueError("instance and valid shapes differ")
    instance = instance.long() * mask.long()
    for b in range(mask.shape[0]):
        ids = torch.unique(instance[b])
        ids = ids[ids > 0]
        if not ids.numel():
            continue
        focus = min(n, max(int(n * focus_share), int(ids.numel())))
        quota, remainder = divmod(focus, int(ids.numel()))
        offset = 0
        for j, ident in enumerate(ids):
            q = quota + int(j < remainder)
            if q:
                out[b, offset:offset + q] = torch.multinomial(
                    (instance[b].flatten() == ident).float(), q, replacement=True, generator=generator)
                offset += q
        nearby = F.max_pool3d((instance[b] > 0).float()[None, None], 3, stride=1, padding=1)[0, 0] > 0
        near_negative = nearby & (instance[b] == 0) & mask[b]
        q = min(int(n * boundary_share), n - offset)
        if q and bool(near_negative.any()):
            out[b, offset:offset + q] = torch.multinomial(
                near_negative.flatten().float(), q, replacement=True, generator=generator)
    return out


def gather(x, points):
    """x (B, D, H, W) -> (B, P) values at the flat indices `points` (B, P)."""
    return torch.gather(x.flatten(1), 1, points)


def dice_loss(logits, target, weight):
    """logits, target (B, Q, P); weight (B, Q) -> weighted mean soft Dice loss."""
    p = logits.sigmoid()
    inter = (p * target).sum(-1)
    dice = 1 - (2 * inter + 1) / (p.sum(-1) + target.sum(-1) + 1)
    return (dice * weight).sum() / weight.sum().clamp(min=1.0)


def entity_loss(mask_logits, presence, entity_pts, a_ignore_pts, present, a_supervised=None):
    """Optionally omit A supervision for samples failing anatomy pseudo-label QC."""
    K = mask_logits.shape[1]
    target = (entity_pts[:, None, :] == torch.arange(1, K + 1, device=entity_pts.device)[None, :, None]).float()
    keep = (~a_ignore_pts).float()[:, None, :]
    bce = (F.binary_cross_entropy_with_logits(mask_logits, target, reduction="none") * keep).sum(-1) / keep.sum(-1).clamp(min=1)
    present = present.float()
    pred = mask_logits.sigmoid() * keep
    dice = 1 - (2 * (pred * target).sum(-1) + 1) / (pred.sum(-1) + (target * keep).sum(-1) + 1)
    per_mask = bce.mean(1) + (dice * present).sum(1) / present.sum(1).clamp(min=1)
    per_pres = F.binary_cross_entropy_with_logits(presence, present, reduction="none").mean(1)
    enabled = torch.ones_like(per_mask) if a_supervised is None else a_supervised.to(
        device=per_mask.device, dtype=per_mask.dtype)
    if enabled.shape != per_mask.shape:
        raise ValueError("a_supervised must have shape (B,)")
    denom = enabled.sum().clamp(min=1)
    return {"a_mask": (per_mask * enabled).sum() / denom,
            "a_presence": (per_pres * enabled).sum() / denom}


def match_events(presence, mask_logits, target, point_weight=None):
    """One sample: presence (M,), mask_logits (M, P), target (N, P) -> (query indices, target indices)."""
    if target.shape[0] == 0:
        empty = torch.zeros(0, dtype=torch.long, device=presence.device)
        return empty, empty
    with torch.no_grad():
        l, t = mask_logits.float(), target.float()
        P = l.shape[1]
        weight = torch.ones(P, device=l.device) if point_weight is None else point_weight.float()
        normalizer = weight.sum().clamp(min=1)
        bce = ((F.softplus(l) * weight).sum(1)[:, None] - l @ (t * weight).t()) / normalizer
        p = l.sigmoid() * weight
        dice = 1 - (2 * (p @ t.t()) + 1) / (p.sum(1)[:, None] + (t * weight).sum(1)[None, :] + 1)
        cost = -presence.float().sigmoid()[:, None] + bce + dice
        qi, ti = linear_sum_assignment(cost.cpu().numpy())
    return (torch.as_tensor(qi, dtype=torch.long, device=presence.device),
            torch.as_tensor(ti, dtype=torch.long, device=presence.device))


def event_loss(presence, mask_logits, targets, point_weight, u_supervised):
    """presence (B, M); mask_logits (B, M, P); targets: list of (N_b, P) float; point_weight (B, P) in {0, 1} (0 at the
    voxels of components under the volume floor); u_supervised (B,) bool. Returns ({"u_presence", "u_mask"}, matches)."""
    B, M, P = mask_logits.shape
    matches, pres_terms, mask_terms, n_matched = [], [], [], 0
    for b in range(B):
        if not bool(u_supervised[b]):
            matches.append(None)
            continue
        qi, ti = match_events(presence[b], mask_logits[b], targets[b], point_weight[b])
        matches.append((qi, ti))
        tgt = torch.zeros(M, device=presence.device)
        tgt[qi] = 1.0
        w = torch.full((M,), NO_OBJECT_WEIGHT, device=presence.device)
        w[qi] = 1.0
        pres_terms.append((F.binary_cross_entropy_with_logits(presence[b], tgt, reduction="none") * w).sum() / w.sum())
        if qi.numel():
            l, t, k = mask_logits[b, qi], targets[b][ti], point_weight[b][None]
            bce = (F.binary_cross_entropy_with_logits(l, t, reduction="none") * k).sum(-1) / k.sum(-1).clamp(min=1.0)
            dropped = l.masked_fill(k < 0.5, -1e4)
            mask_terms.append(bce.sum() + dice_loss(dropped[None], (t * k)[None], torch.ones(1, qi.numel(), device=l.device)) * qi.numel())
            n_matched += qi.numel()
    zero = presence.new_zeros(())
    out = {"u_presence": torch.stack(pres_terms).mean() if pres_terms else zero,
           "u_mask": torch.stack(mask_terms).sum() / max(n_matched, 1) if mask_terms else zero}
    return out, matches


def seq_loss(logits, seq):
    return {"s": F.cross_entropy(logits, seq)}


def relation_loss(logits, host, negatives, margin=MARGIN):
    """logits (N, 14); host (N,) in 0..13; negatives (N, 2) host indices or -1 -> {"r_host", "r_hard"}; zeros when N = 0."""
    if logits.shape[0] == 0:
        z = logits.new_zeros(())
        return {"r_host": z, "r_hard": z}
    ce = F.cross_entropy(logits, host)
    pos = logits.gather(1, host[:, None])                                                      # (N, 1)
    neg = negatives.clamp(min=0)
    hinge = F.relu(margin + logits.gather(1, neg) - pos) * (negatives >= 0).float()
    return {"r_host": ce, "r_hard": hinge.sum() / (negatives >= 0).float().sum().clamp(min=1.0)}


def total(parts, lambda_r=LAMBDA_R, lambda_h=LAMBDA_H):
    """The spec's sum; missing parts count as zero. Returns (total, {name: float})."""
    get = lambda k: parts.get(k, 0.0)
    t = get("a_mask") + get("a_presence") + get("s") + get("u_presence") + get("u_mask") + lambda_r * (get("r_host") + lambda_h * get("r_hard"))
    return t, {k: float(v) for k, v in parts.items()}
