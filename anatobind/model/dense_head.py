"""Dense lesion detector on the F1 grid: per-class centre heatmaps, a sub-cell offset and a box size.

Replaces the DETR queries for U_B (RESEARCH_PLAN v2.2 13.6, user decision Q17, 2026-09-11). On a
synthetic test -- one bright 6x12x12 block at a random place in a 32x64x64 volume -- the DETR decoder
without reference points still predicted the average box after 24,000 samples (mean IoU 0.01, with
or without coordinates in its memory), while this head reached a mean IoU of 0.89 after 8,000
samples and 0.94 after 24,000.

Cell i of an axis with stride s covers voxels [i*s, (i+1)*s). A box centre c (voxels) sits at
c/s - 0.5 in cell units; the heatmap peak goes on the nearest cell, the offset is the remainder and
the size is the log of the box extent in cells.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = (2, 4, 4)
PRIOR_BIAS = -2.19  # sigmoid(-2.19) = 0.1, the usual focal-loss start


class CentreHead(nn.Module):
    def __init__(self, c1, num_classes=4, hidden=64, embed_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv3d(c1, hidden, 3, padding=1), nn.GELU(),
                                   nn.Conv3d(hidden, hidden, 3, padding=1), nn.GELU())
        self.heat = nn.Conv3d(hidden, num_classes, 1)
        self.offset = nn.Conv3d(hidden, 3, 1)
        self.size = nn.Conv3d(hidden, 3, 1)
        self.embed = nn.Conv3d(hidden, embed_dim, 1)
        nn.init.constant_(self.heat.bias, PRIOR_BIAS)

    def forward(self, f1):
        h = self.trunk(f1)
        return {"heat": self.heat(h), "offset": self.offset(h), "size": self.size(h), "feat": self.embed(h)}


def centre_targets(boxes, classes, grid, num_classes, device):
    """Heatmaps (C, Z, Y, X) and, per object, its peak cell, sub-cell offset and log size in cells."""
    heat = torch.zeros(num_classes, *grid, device=device)
    stride = torch.tensor(STRIDE, dtype=torch.float32, device=device)
    limit = torch.tensor(grid, dtype=torch.float32, device=device) - 1
    zz, yy, xx = torch.meshgrid(*(torch.arange(n, dtype=torch.float32, device=device) for n in grid), indexing="ij")
    cells, offsets, sizes = [], [], []
    for box, cls in zip(boxes.float(), classes):
        centre = (box[:3] + box[3:]) / 2 / stride - 0.5
        extent = (box[3:] - box[:3]).clamp(min=1.0) / stride
        cell = torch.minimum(centre.round().clamp(min=0), limit).long()
        sigma = (extent / 6).clamp(min=0.5)
        g = torch.exp(-0.5 * (((zz - centre[0]) / sigma[0]) ** 2 + ((yy - centre[1]) / sigma[1]) ** 2
                              + ((xx - centre[2]) / sigma[2]) ** 2))
        heat[cls] = torch.maximum(heat[cls], g)
        heat[cls, cell[0], cell[1], cell[2]] = 1.0
        cells.append(cell)
        offsets.append(centre - cell)
        sizes.append(extent.log())
    return heat, cells, offsets, sizes


def centre_loss(out, batch):
    """CornerNet focal loss on the heatmaps (cells in the depth padding excluded) and L1 on the offset and
    log size at each object's peak cell, all normalised by the number of objects."""
    heat = out["heat"].float()
    B, C, Z, Y, X = heat.shape
    dev = heat.device
    p = heat.sigmoid().clamp(1e-4, 1 - 1e-4)
    cell_z = torch.arange(Z, device=dev, dtype=torch.float32) * STRIDE[0] + STRIDE[0] / 2
    total = {k: heat.new_zeros(()) for k in ("heat", "offset", "size")}
    n_obj = 0
    for b in range(B):
        tgt, cells, offsets, sizes = centre_targets(batch["boxes"][b].to(dev), batch["box_classes"][b].tolist(),
                                                    (Z, Y, X), C, dev)
        valid = (cell_z < float(batch["valid_depth"][b]))[None, :, None, None].expand_as(tgt)
        pos = tgt.eq(1.0)
        pos_loss = -(torch.log(p[b]) * (1 - p[b]) ** 2)[pos].sum()
        neg_loss = -(torch.log(1 - p[b]) * p[b] ** 2 * (1 - tgt) ** 4)[~pos & valid].sum()
        total["heat"] = total["heat"] + pos_loss + neg_loss
        for cell, off, size in zip(cells, offsets, sizes):
            z, y, x = cell.tolist()
            total["offset"] = total["offset"] + (out["offset"][b, :, z, y, x].float() - off).abs().sum()
            total["size"] = total["size"] + (out["size"][b, :, z, y, x].float() - size).abs().sum()
        n_obj += len(cells)
    n = max(n_obj, 1)
    return {k: v / n for k, v in total.items()}


def decode_centres(out, M, valid_depth):
    """Top-M peaks of the first sample -> boxes (M, 6) in voxels, class probabilities (M, C + 1) and
    embeddings (M, E). A slot's no-object probability is 1 - its peak score, so "the most probable class
    is not no-object" means "the peak score is at least 0.5". Cells centred in the depth padding never peak.
    """
    heat = out["heat"][0].float().sigmoid()
    C, Z, Y, X = heat.shape
    dev = heat.device
    cell_z = torch.arange(Z, device=dev, dtype=torch.float32) * STRIDE[0] + STRIDE[0] / 2
    heat = heat * (cell_z < valid_depth)[None, :, None, None]
    keep = heat == F.max_pool3d(heat[None], 3, stride=1, padding=1)[0]
    scores, idx = (heat * keep).flatten().topk(M)
    cls = idx // (Z * Y * X)
    rest = idx % (Z * Y * X)
    z, y, x = rest // (Y * X), (rest // X) % Y, rest % X
    stride = torch.tensor(STRIDE, dtype=torch.float32, device=dev)
    offset = out["offset"][0][:, z, y, x].float().T
    size = out["size"][0][:, z, y, x].float().T.exp() * stride
    centre = (torch.stack([z, y, x], 1).float() + offset + 0.5) * stride
    boxes = torch.cat([centre - size / 2, centre + size / 2], 1)
    cls_prob = torch.zeros(M, C + 1, device=dev)
    cls_prob[torch.arange(M, device=dev), cls] = scores
    cls_prob[:, C] = 1 - scores
    return boxes, cls_prob, out["feat"][0][:, z, y, x].T
