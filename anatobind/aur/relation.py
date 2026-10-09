"""Relation tokens and the per-lesion host competition (spec N8; plan §12; V7 §8).

For an abnormality j and each of the 13 host classes i a token r_ij = fuse([A_i, U_j, S, G_ij]) is built, where A_i is
the mean embedding of the entities that form host i, U_j the event embedding, S the sequence embedding and G_ij the
geometry between the predicted event mask and the predicted host mask on the coarse (F1) grid: overlap share, centroid
displacement (mm / 100), distance from the event centroid to the nearest host voxel (mm / 100, capped), host presence,
the event's side offset from the midline between the left and the right host masses (signed towards the patient's
right, 0 when one side is absent from the crop), the host's own side (-1 left, +1 right, 0 brainstem) and the host's
tissue family. The 13 tokens plus a
"no host" token attend to each other (only within the lesion) and each yields one logit: a 14-way host distribution."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from anatobind.aur.labels import ENTITY_HOST, HOST_SIDE, HOST_TISSUE, N_HOSTS, TISSUES

GEO_DIM = 1 + 3 + 1 + 1 + 1 + 1 + len(TISSUES)      # 15
DIST_CAP_MM = 200.0
SCALE_MM = 100.0

_HOST_OF_ENTITY = torch.tensor(ENTITY_HOST)                                    # (32,) host index or -1
_TISSUE_ONEHOT = F.one_hot(torch.tensor([TISSUES.index(t) for t in HOST_TISSUE]), len(TISSUES)).float()   # (13, 7)
_HOST_SIDE_SIGN = torch.tensor([-1.0 if s == "left" else 1.0 if s == "right" else 0.0 for s in HOST_SIDE])  # (13,)


def host_from_entities(x):
    """(B, 32, ...) per-entity tensor -> (B, 13, ...) per-host: mean (for embeddings) over the entities of each host."""
    out = []
    for h in range(N_HOSTS):
        idx = torch.nonzero(_HOST_OF_ENTITY == h).flatten().to(x.device)
        out.append(x.index_select(1, idx).mean(1))
    return torch.stack(out, 1)


def host_masks_from_entities(entity_probs):
    """(B, 32, ...) entity mask probabilities -> (B, 13, ...) host probabilities: max over the host's entities."""
    out = []
    for h in range(N_HOSTS):
        idx = torch.nonzero(_HOST_OF_ENTITY == h).flatten().to(entity_probs.device)
        out.append(entity_probs.index_select(1, idx).amax(1))
    return torch.stack(out, 1)


def geometry(event_probs, host_probs, coords):
    """event_probs (N, D, H, W) in [0, 1]; host_probs (13, D, H, W); coords (3, D, H, W) mm (z, y, x) on the same grid
    -> (N, 13, GEO_DIM) float32."""
    e = event_probs.flatten(1).float()                                  # (N, V)
    h = host_probs.flatten(1).float()                                   # (13, V)
    c = coords.flatten(1).float()                                       # (3, V)
    e_mass = e.sum(1).clamp(min=1e-6)
    h_mass = h.sum(1)
    overlap = (e @ h.t()) / e_mass[:, None]                             # (N, 13)
    e_cent = (e @ c.t()) / e_mass[:, None]                              # (N, 3)
    h_cent = (h @ c.t()) / h_mass.clamp(min=1e-6)[:, None]              # (13, 3)
    present = (h_mass > 0.5).float()
    disp = (h_cent[None] - e_cent[:, None]) / SCALE_MM                  # (N, 13, 3)
    dist = torch.full((e.shape[0], N_HOSTS), DIST_CAP_MM, device=e.device)
    for j in range(N_HOSTS):
        vox = c[:, h[j] > 0.5]                                          # (3, Mj)
        if vox.shape[1]:
            dist[:, j] = torch.cdist(e_cent, vox.t()).amin(1).clamp(max=DIST_CAP_MM)
    dist = torch.where(overlap > 0.05, torch.zeros_like(dist), dist)    # touching the host: no distance
    side = _HOST_SIDE_SIGN.to(e.device)
    left_mass, right_mass = h[side < 0].sum(0), h[side > 0].sum(0)                     # (V,) each
    if left_mass.sum() > 0.5 and right_mass.sum() > 0.5:
        lx, rx = (left_mass @ c[2]) / left_mass.sum(), (right_mass @ c[2]) / right_mass.sum()
        e_side = (e_cent[:, 2] - (lx + rx) / 2) * torch.sign(rx - lx) / SCALE_MM
    else:
        e_side = torch.zeros(e.shape[0], device=e.device)
    e_side = e_side[:, None, None].expand(-1, N_HOSTS, 1)
    h_side = side[None, :, None].expand(e.shape[0], -1, 1)
    tissue = _TISSUE_ONEHOT.to(e.device)[None].expand(e.shape[0], -1, -1)
    return torch.cat([overlap[..., None], disp, (dist / SCALE_MM)[..., None], present[None, :, None].expand(e.shape[0], -1, 1),
                      e_side, h_side, tissue], -1)


class CandidateCompetition(nn.Module):
    """13 candidate tokens + 1 no-host token per lesion -> 14 logits (spec N8)."""

    def __init__(self, d_model, d_geo=GEO_DIM, layers=2, heads=4):
        super().__init__()
        self.fuse = nn.Sequential(nn.Linear(3 * d_model + d_geo, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.none = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.host_embed = nn.Parameter(torch.randn(N_HOSTS + 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0, batch_first=True,
                                                               norm_first=True) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, event_embed, host_embed, seq_embed, geo):
        """event_embed (N, d); host_embed (13, d); seq_embed (d,); geo (N, 13, d_geo) -> (N, 14) logits."""
        n = event_embed.shape[0]
        s = seq_embed[None, None].expand(n, N_HOSTS, -1)
        r = self.fuse(torch.cat([host_embed[None].expand(n, -1, -1), event_embed[:, None].expand(-1, N_HOSTS, -1), s, geo.to(event_embed.dtype)], -1))
        none = self.none(torch.cat([event_embed, seq_embed[None].expand(n, -1)], -1))[:, None]
        tokens = torch.cat([r, none], 1) + self.host_embed[None]
        for block in self.blocks:
            tokens = block(tokens)
        return self.out(self.norm(tokens)).squeeze(-1)
