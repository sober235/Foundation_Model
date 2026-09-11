"""Entity and event decoders for the arm-B minimal path.

``ADecoder`` is identity-anchored (spec 4.3, decision 2A): query ``k`` is
permanently segmentation label ``k + 1``, so there is no matching step anywhere
in it.  ``UBDecoder`` is DETR-style (spec 4.5) and is matched to targets by the
Hungarian algorithm in ``losses.py``.

Minimal-path deviation from spec 4.3/4.5: the cross-attention is plain, not
Mask2Former masked attention.  Masked attention is a convergence optimisation
and adds a mask-feedback path that would obscure a plumbing bug; it belongs in
the gate run, not here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class _Layer(nn.Module):
    """Pre-norm cross-attention, query self-attention, feed-forward."""

    def __init__(self, d_model, heads):
        super().__init__()
        self.cross = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, q, mem):
        h = self.n1(q)
        q = q + self.cross(h, mem, mem, need_weights=False)[0]
        h = self.n2(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        return q + self.ff(self.n3(q))


class _QueryStack(nn.Module):
    """Learned queries attending the feature levels in turn, coarse to fine."""

    def __init__(self, mem_channels, d_model, n_queries, layers, heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.proj = nn.ModuleList(nn.Conv3d(c, d_model, 1) for c in mem_channels)
        self.layers = nn.ModuleList(_Layer(d_model, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mems, return_all=False):
        toks = [rearrange(p(m), "b c d h w -> b (d h w) c") for p, m in zip(self.proj, mems)]
        q = self.query[None].expand(mems[0].shape[0], -1, -1)
        every = []
        for i, layer in enumerate(self.layers):
            q = layer(q, toks[i % len(toks)])
            if return_all:
                every.append(self.norm(q))
        return every if return_all else self.norm(q)


class ADecoder(nn.Module):
    """K identity-anchored anatomy queries -> mask, presence, embedding."""

    def __init__(self, channels, d_model=128, K=6, layers=3, heads=4):
        super().__init__()
        self.K = K
        c1, c2, c3, c4 = channels
        self.stack = _QueryStack((c4, c3, c2), d_model, K, layers, heads)
        self.mask_feat = nn.Conv3d(c1, d_model, 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.presence = nn.Linear(d_model, 1)

    def forward(self, feats, out_shape):
        q = self.stack([feats[3], feats[2], feats[1]])
        masks = torch.einsum("bkc,bcdhw->bkdhw", self.mask_embed(q), self.mask_feat(feats[0]))
        masks = F.interpolate(masks, size=tuple(out_shape), mode="trilinear", align_corners=False)
        return {"masks": masks, "presence": self.presence(q).squeeze(-1), "embed": q}


class UBDecoder(nn.Module):
    """M event queries -> class logits, normalised box, embedding."""

    def __init__(self, channels, d_model=128, M=8, layers=3, num_classes=2, heads=4):
        super().__init__()
        self.M = M
        c1, c2, c3, _ = channels
        self.stack = _QueryStack((c3, c2, c1), d_model, M, layers, heads)
        self.cls = nn.Linear(d_model, num_classes + 1)
        self.box = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 6)
        )

    def _heads(self, q):
        return {"logits": self.cls(q), "boxes": self.box(q).sigmoid(), "embed": q}

    def forward(self, feats, aux=False):
        mems = [feats[2], feats[1], feats[0]]
        if not aux:
            # (cz, cy, cx, dz, dy, dx) normalised to the input extent
            return self._heads(self.stack(mems))
        every = self.stack(mems, return_all=True)
        out = self._heads(every[-1])
        out["aux"] = [{"logits": self.cls(q), "boxes": self.box(q).sigmoid()} for q in every[:-1]]
        return out


class FullResMaskHead(nn.Module):
    """Anatomy masks at the input resolution.

    F1 sits on a stride (2, 4, 4) grid -- 2.5 mm in-plane on the 0.625 mm export -- while femoral and
    tibial cartilage are 2-3 mm thick, so masks computed on F1 cannot place a cartilage boundary.
    F1 is lifted back to full resolution by a transposed convolution and fused with a shallow
    convolution of the image; each query's embedding then dots with the fused pixel features.
    """

    def __init__(self, c1, d_model, mask_dim=32, stride=(2, 4, 4)):
        super().__init__()
        self.up = nn.ConvTranspose3d(c1, mask_dim, kernel_size=stride, stride=stride)
        self.img = nn.Sequential(nn.Conv3d(1, mask_dim, 3, padding=1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv3d(2 * mask_dim, mask_dim, 3, padding=1), nn.GELU(),
                                  nn.Conv3d(mask_dim, mask_dim, 1))
        self.embed = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, mask_dim))

    def forward(self, q, f1, image):
        pix = self.fuse(torch.cat([self.up(f1), self.img(image)], 1))
        return torch.einsum("bkc,bczyx->bkzyx", self.embed(q), pix)
