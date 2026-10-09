"""Stage I contrastive consistency (SSL-first plan §3.2, task I-B; decision Q11: τ = 0.2, weight 0.1).

Two intensity views of one crop (bias field, gamma, noise; no blur, no mirroring; the same geometry) go through the
backbone; the valid tokens of F4 are mean-pooled, projected to 128-D and L2-normalised. The loss is a patient-safe
InfoNCE: for an anchor, the positive is the other view of its crop; the candidates are every view of every crop in the
global batch (gathered across the ranks with gradient) except the anchor itself and every crop of the anchor's
patient, which is neither a positive nor a negative. A batch of one patient gives a zero loss, not a wrong one."""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

TEMPERATURE = 0.2
CONTRAST_WEIGHT = 0.1


class Projector(nn.Module):
    def __init__(self, in_channels=512, hidden=512, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_channels, hidden), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, feat, valid):
        """feat (B, C, d, h, w), valid (B, d, h, w) float -> (B, out) unit vectors (mean over the valid tokens)."""
        v = (valid > 0.5).to(feat.dtype)[:, None]
        pooled = (feat * v).flatten(2).sum(-1) / v.flatten(2).sum(-1).clamp(min=1.0)
        return F.normalize(self.net(pooled), dim=-1)


def gather_all(t):
    """All ranks' copies of `t` (with gradient to the local copy); just [t] without a process group."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return [t]
    out = list(torch.distributed.nn.functional.all_gather(t))
    return out


def info_nce(z1, z2, patient, temperature=TEMPERATURE):
    """z1, z2 (N, d) unit vectors of the two views (global batch), patient (N,) int ids -> (loss, top-1 accuracy).
    Anchors: all 2N views. Candidates: all 2N views. Excluded per anchor: itself and the views of other crops of the
    same patient. Positive: the other view of the same crop."""
    n = z1.shape[0]
    z = torch.cat([z1, z2], 0)                                                      # (2N, d)
    logits = (z @ z.t()) / temperature                                              # (2N, 2N)
    idx = torch.arange(2 * n, device=z.device)
    crop = idx % n
    pid = torch.cat([patient, patient], 0)
    same_patient = pid[:, None] == pid[None, :]
    same_crop = crop[:, None] == crop[None, :]
    positive = same_crop & (idx[:, None] != idx[None, :])
    excluded = (same_patient & ~same_crop) | (idx[:, None] == idx[None, :])
    logits = logits.masked_fill(excluded, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_idx = positive.float().argmax(1, keepdim=True)                              # exactly one positive per anchor
    loss = -log_prob.gather(1, pos_idx).mean()
    with torch.no_grad():
        acc = (logits.argmax(1, keepdim=True) == pos_idx).float().mean()
    return loss, acc


def contrast_loss(z1_local, z2_local, patient_local, temperature=TEMPERATURE):
    """The InfoNCE over the global batch: gathers the two views and the patient ids from every rank."""
    z1 = torch.cat(gather_all(z1_local), 0)
    z2 = torch.cat(gather_all(z2_local), 0)
    patient = torch.cat(gather_all(patient_local), 0)
    return info_nce(z1, z2, patient, temperature)
