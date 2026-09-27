"""B2 (local appearance: encoder + linear head) and B1 (independent candidate scoring on top of the same encoder,
spec 2026-09-27 §5.5–5.6), plus the set-valued negative log-likelihood of §5.7. B1 reuses IndependentCandidateHead
with one lesion per sample (M = 1, K = 7)."""
import torch
import torch.nn as nn

from anatobind.model.relation import IndependentCandidateHead
from anatobind.relation.encoder import LesionEncoder
from anatobind.relation.table import N_OUT, N_SLOTS

GEO_DIM, D_MODEL, CLASS_EMBED = 26, 64, 32


class B2Model(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.encoder = LesionEncoder(in_ch)
        self.head = nn.Linear(self.encoder.out_dim, N_OUT)

    def forward(self, x):
        return self.head(self.encoder(x))


class B1Model(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.encoder = LesionEncoder(in_ch)
        self.u_proj = nn.Linear(self.encoder.out_dim, D_MODEL)
        self.class_embed = nn.Embedding(N_SLOTS, CLASS_EMBED)
        self.a_mlp = nn.Sequential(nn.Linear(CLASS_EMBED + GEO_DIM, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.head = IndependentCandidateHead(d_model=D_MODEL, geometry_channels=GEO_DIM, geo_dim=32, hidden_dim=D_MODEL)

    def forward(self, x, geo, present):
        n = x.shape[0]
        emb = self.class_embed.weight[None].expand(n, N_SLOTS, CLASS_EMBED)
        a = self.a_mlp(torch.cat([emb, geo], -1))                       # (N, 7, 64)
        u = self.u_proj(self.encoder(x))[:, None, :]                     # (N, 1, 64)
        out = self.head(a, u, geo[:, :, None, :], present.bool())
        return out["host_logits"][:, 0, :]                               # (N, 8)


def set_nll(logits, acceptable8):
    """-log of the probability mass on the acceptable set, averaged over rows (cross-entropy for singletons)."""
    masked = logits.masked_fill(~acceptable8.bool(), float("-inf"))
    return (torch.logsumexp(logits, -1) - torch.logsumexp(masked, -1)).mean()
