"""Entity, event, mask and sequence heads (spec N6, N7, §5; plan §9–§11).

QueryDecoder: learned queries attend the feature levels in turn, coarse to fine, with key padding masks from M_valid
and a coordinate embedding on the memory tokens (PE_local). EntityDecoder has K = 32 identity-anchored queries (query
k is ENTITY_LABELS[k], no matching); EventDecoder has M anonymous queries matched to lesion instances by the
Hungarian algorithm in losses.py. MaskHead turns the pyramid into pixel features once per forward; a mask is the dot
product of a query's mask embedding with those features, at full resolution (ConvTranspose up from F1 fused with the
image, as the knee FullResMaskHead) or at the coarse F1 grid for the relation geometry. During training masks are
evaluated at sampled points only."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from anatobind.aur.labels import N_ENTITIES, SEQ_TYPES


class DecoderLayer(nn.Module):
    """Pre-norm cross-attention (with key padding), query self-attention, feed-forward."""

    def __init__(self, d_model, heads):
        super().__init__()
        self.cross = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, q, mem, pad):
        h = self.n1(q)
        q = q + self.cross(h, mem, mem, key_padding_mask=pad, need_weights=False)[0]
        h = self.n2(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        return q + self.ff(self.n3(q))


class QueryDecoder(nn.Module):
    def __init__(self, mem_channels, d_model, n_queries, layers, heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.proj = nn.ModuleList(nn.Conv3d(c, d_model, 1) for c in mem_channels)
        self.coord = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.layers = nn.ModuleList(DecoderLayer(d_model, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, levels):
        """levels: the backbone levels this decoder reads, in the order the layers visit them (coarse to fine)."""
        toks, pads = [], []
        for p, lv in zip(self.proj, levels):
            t = rearrange(p(lv["feat"]), "b c d h w -> b (d h w) c")
            t = t + self.coord(rearrange(lv["local"], "b c d h w -> b (d h w) c")).to(t.dtype)
            toks.append(t)
            pads.append(rearrange(lv["valid"], "b d h w -> b (d h w)") < 0.5)
        q = self.query[None].expand(toks[0].shape[0], -1, -1)
        for i, layer in enumerate(self.layers):
            j = i % len(toks)
            q = layer(q, toks[j], pads[j])
        return self.norm(q)


class EntityDecoder(nn.Module):
    """K identity-anchored anatomy queries -> presence logit, embedding."""

    def __init__(self, channels, d_model=256, K=N_ENTITIES, layers=3, heads=8):
        super().__init__()
        self.K = K
        c1, c2, c3, c4 = channels
        self.stack = QueryDecoder((c4, c3, c2), d_model, K, layers, heads)
        self.presence = nn.Linear(d_model, 1)

    def forward(self, levels):
        q = self.stack([levels[3], levels[2], levels[1]])
        return {"embed": q, "presence": self.presence(q).squeeze(-1)}


class EventDecoder(nn.Module):
    """M anonymous abnormality queries -> presence logit, embedding."""

    def __init__(self, channels, d_model=256, M=64, layers=3, heads=8):
        super().__init__()
        self.M = M
        c1, c2, c3, c4 = channels
        self.stack = QueryDecoder((c3, c2, c1), d_model, M, layers, heads)
        self.presence = nn.Linear(d_model, 1)

    def forward(self, levels):
        q = self.stack([levels[2], levels[1], levels[0]])
        return {"embed": q, "presence": self.presence(q).squeeze(-1)}


class MaskHead(nn.Module):
    """Pixel features from the pyramid: coarse (F1 grid, `dim` channels) and full resolution (`mask_dim` channels)."""

    def __init__(self, channels, d_model, patch, dim=64, mask_dim=32):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv3d(c, dim, 1) for c in channels)
        self.smooth = nn.ModuleList(nn.Sequential(nn.Conv3d(dim, dim, 3, padding=1), nn.GELU()) for _ in channels)
        self.up = nn.ConvTranspose3d(dim, mask_dim, kernel_size=tuple(patch), stride=tuple(patch))
        self.img = nn.Sequential(nn.Conv3d(1, mask_dim, 3, padding=1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv3d(2 * mask_dim, mask_dim, 3, padding=1), nn.GELU(), nn.Conv3d(mask_dim, mask_dim, 1))
        self.embed_full = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, mask_dim))
        self.embed_coarse = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, dim))

    def pixels(self, levels, image):
        """{"coarse": (B, dim, D1, H1, W1), "full": (B, mask_dim, D, H, W)}."""
        feats = [lv["feat"] for lv in levels]
        p = self.smooth[3](self.lateral[3](feats[3]))
        for i in (2, 1, 0):
            lat = self.lateral[i](feats[i])
            p = self.smooth[i](lat + F.interpolate(p, size=lat.shape[2:], mode="trilinear", align_corners=False))
        full = self.fuse(torch.cat([self.up(p), self.img(image)], 1))
        return {"coarse": p, "full": full}

    def coarse_masks(self, embed, pix):
        """(B, Q, D1, H1, W1) mask logits on the F1 grid."""
        return torch.einsum("bqc,bcdhw->bqdhw", self.embed_coarse(embed), pix["coarse"])

    def full_masks(self, embed, pix, points=None):
        """Full-resolution mask logits: (B, Q, D, H, W), or (B, Q, P) at the flat voxel indices `points` (B, P)."""
        e = self.embed_full(embed)
        if points is None:
            return torch.einsum("bqc,bcdhw->bqdhw", e, pix["full"])
        flat = pix["full"].flatten(2)                                                       # (B, C, V)
        gathered = torch.gather(flat, 2, points[:, None, :].expand(-1, flat.shape[1], -1))  # (B, C, P)
        return torch.einsum("bqc,bcp->bqp", e, gathered)


class SequenceHead(nn.Module):
    """S token: masked mean of F4 -> sequence type logits and an embedding the relation tokens read."""

    def __init__(self, c4, d_model, n_types=len(SEQ_TYPES)):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(c4, d_model), nn.GELU())
        self.cls = nn.Linear(d_model, n_types)

    def forward(self, level):
        f = rearrange(level["feat"], "b c d h w -> b (d h w) c")
        w = rearrange(level["valid"], "b d h w -> b (d h w)").to(f.dtype)
        pooled = (f * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1.0)
        s = self.proj(pooled)
        return {"embed": s, "logits": self.cls(s)}
