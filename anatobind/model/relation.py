"""Lesion--anatomy binding modules.

Two formulations intentionally coexist:

1. IndependentCandidateHead is the B1 baseline/prototype. Each lesion-candidate
   pair is scored independently before a host softmax; there is no candidate
   interaction. It must not be described as B3 host competition.

2. RelationModule is the earlier global KxM pair-Transformer formulation from
   spec 4.6. It is kept as an ablation and for backward compatibility.

Both consume geometry as an input feature. Relation truth must still come from
the annotated host relation, never from overlap or distance heuristics.
"""

import torch
import torch.nn as nn

GEOMETRY_CHANNELS = 5  # dz, dy, dx (mm), ||d||, IoA


def geometry_features(masks, boxes_vox, spacing_mm):
    """(B,K,D,H,W) masks and (B,M,6) voxel boxes (z0,y0,x0,z1,y1,x1) -> (B,K,M,5).

    Distances are millimetres, so anisotropic spacing is honoured.  Detached:
    spec 4.6 does not require a gradient path through the geometry.
    """
    with torch.no_grad():
        m = masks.float()
        B, K, D, H, W = m.shape
        M = boxes_vox.shape[1]
        dev = m.device
        area = m.flatten(2).sum(-1).clamp(min=1.0)
        cz = (m.sum((3, 4)) * torch.arange(D, device=dev, dtype=m.dtype)).sum(-1) / area
        cy = (m.sum((2, 4)) * torch.arange(H, device=dev, dtype=m.dtype)).sum(-1) / area
        cx = (m.sum((2, 3)) * torch.arange(W, device=dev, dtype=m.dtype)).sum(-1) / area
        centroid = torch.stack([cz, cy, cx], -1)
        centre = (boxes_vox[..., :3] + boxes_vox[..., 3:]) / 2.0
        sp = spacing_mm.to(dev).reshape(-1, 1, 1, 3)
        delta = (centroid[:, :, None, :] - centre[:, None, :, :]) * sp
        dist = delta.norm(dim=-1, keepdim=True)

        ioa = torch.zeros(B, K, M, device=dev, dtype=m.dtype)
        idx = boxes_vox.round().long()
        for b in range(B):
            for j in range(M):
                z0, y0, x0, z1, y1, x1 = idx[b, j].tolist()
                if z1 <= z0 or y1 <= y0 or x1 <= x0:
                    continue
                vol = (z1 - z0) * (y1 - y0) * (x1 - x0)
                ioa[b, :, j] = m[b][:, z0:z1, y0:y1, x0:x1].flatten(1).sum(-1) / vol
        return torch.cat([delta, dist, ioa[..., None]], -1)


class IndependentCandidateHead(nn.Module):
    """Independent per-candidate scorer (B1), followed by host softmax.

    Inputs:
        a_embed: (B, K, d) anatomy entity embeddings.
        u_embed: (B, M, d) lesion entity embeddings.
        geo: (B, K, M, G) pairwise physical geometry.
        present: (B, K) bool mask for anatomy candidates present in the scan.
        local_evidence: optional (B, K, M, local_dim) image/boundary evidence.

    Returns:
        host_logits: (B, M, K + 1), final column is ``none``.
        pair_repr: (B, K, M, d), independently scored pair representations.

    There is intentionally no interaction across anatomy candidates or lesions.
    A true B3 candidate-competition head must add interaction across the K host
    candidates for each lesion and should be implemented as a separate module.
    """

    def __init__(
        self,
        d_model=128,
        geometry_channels=GEOMETRY_CHANNELS,
        geo_dim=64,
        local_dim=0,
        hidden_dim=None,
    ):
        super().__init__()
        self.local_dim = int(local_dim)
        hidden_dim = int(hidden_dim or d_model)
        self.g = nn.Sequential(
            nn.Linear(geometry_channels, geo_dim),
            nn.GELU(),
            nn.Linear(geo_dim, geo_dim),
        )
        pair_in = 2 * d_model + geo_dim + self.local_dim
        self.pair = nn.Sequential(
            nn.Linear(pair_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.h_host = nn.Linear(d_model, 1)
        self.h_none = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, a_embed, u_embed, geo, present, local_evidence=None):
        B, K, d = a_embed.shape
        Bu, M, du = u_embed.shape
        if Bu != B or du != d:
            raise ValueError("a_embed and u_embed must share batch size and embedding dimension")
        if geo.shape[:3] != (B, K, M):
            raise ValueError("geo must have shape (B, K, M, G)")
        if present.shape != (B, K):
            raise ValueError("present must have shape (B, K)")

        pieces = [
            a_embed[:, :, None, :].expand(B, K, M, d),
            u_embed[:, None, :, :].expand(B, K, M, d),
            self.g(geo),
        ]
        if self.local_dim:
            if local_evidence is None or local_evidence.shape != (B, K, M, self.local_dim):
                raise ValueError(
                    "local_evidence must have shape (B, K, M, local_dim) when local_dim > 0"
                )
            pieces.append(local_evidence)
        elif local_evidence is not None:
            raise ValueError("local_evidence was provided but local_dim == 0")

        pair_repr = self.pair(torch.cat(pieces, dim=-1))
        host = self.h_host(pair_repr).squeeze(-1).transpose(1, 2)
        host = host.masked_fill(~present[:, None, :], float("-inf"))
        none = self.h_none(u_embed)
        return {
            "pair_repr": pair_repr,
            "host_logits": torch.cat([host, none], dim=-1),
        }


class _PairBlock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x, pad):
        h = self.n1(x)
        x = x + self.attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class RelationModule(nn.Module):
    def __init__(self, d_model=128, K=6, geo_dim=64, layers=2, heads=8):
        super().__init__()
        self.K = K
        self.g = nn.Sequential(
            nn.Linear(GEOMETRY_CHANNELS, geo_dim), nn.GELU(), nn.Linear(geo_dim, geo_dim)
        )
        self.phi = nn.Sequential(
            nn.Linear(2 * d_model + geo_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(_PairBlock(d_model, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.h_R = nn.Linear(d_model, 1)
        self.h_host = nn.Linear(d_model, 1)
        self.h_none = nn.Linear(d_model, 1)

    def forward(self, a_embed, u_embed, geo, present):
        B, K, d = a_embed.shape
        M = u_embed.shape[1]
        pair = torch.cat([
            a_embed[:, :, None, :].expand(B, K, M, d),
            u_embed[:, None, :, :].expand(B, K, M, d),
            self.g(geo),
        ], -1)
        r = self.phi(pair).reshape(B, K * M, d)

        pad = ~present[:, :, None].expand(B, K, M).reshape(B, K * M)
        # a fully padded row would make the attention softmax undefined; the
        # host logits below are masked to -inf for those rows anyway
        pad = torch.where(pad.all(-1, keepdim=True), torch.zeros_like(pad), pad)
        for block in self.blocks:
            r = block(r, pad)
        r = self.norm(r).reshape(B, K, M, d)

        host = self.h_host(r).squeeze(-1).transpose(1, 2)
        host = host.masked_fill(~present[:, None, :], float("-inf"))
        return {
            "R": self.h_R(r).squeeze(-1),
            "host_logits": torch.cat([host, self.h_none(u_embed)], -1),
        }
