"""Dense lesion detector on the 2D grid: per-class centre heatmaps, a sub-cell offset and a box size.

2D counterpart of dense_head.py, used for 2D slices. On the F1 grid (stride 2), the model produces
per-class centre heatmaps, sub-cell offsets, and log box sizes.

Cell i of an axis with stride s covers pixels [i*s, (i+1)*s). A box centre c (pixels) sits at
c/s - 0.5 in cell units; the heatmap peak goes on the nearest cell, the offset is the remainder and
the size is the log of the box extent in cells.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 2
PRIOR_BIAS = -2.19  # sigmoid(-2.19) = 0.1, the usual focal-loss start


class CentreHead2D(nn.Module):
    def __init__(self, c1, num_classes=5, hidden=64, embed_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(c1, hidden, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.heat = nn.Conv2d(hidden, num_classes, 1)
        self.offset = nn.Conv2d(hidden, 2, 1)
        self.size = nn.Conv2d(hidden, 2, 1)
        self.embed = nn.Conv2d(hidden, embed_dim, 1)
        nn.init.constant_(self.heat.bias, PRIOR_BIAS)

    def forward(self, f1):
        h = self.trunk(f1)
        return {"heat": self.heat(h), "offset": self.offset(h), "size": self.size(h), "feat": self.embed(h)}


def centre_targets_2d(boxes, classes, grid, num_classes, device):
    """Heatmaps (C, Y, X) and, per object, its peak cell, sub-cell offset and log size in cells."""
    heat = torch.zeros(num_classes, *grid, device=device)
    stride = float(STRIDE)
    limit = torch.tensor(grid, dtype=torch.float32, device=device) - 1
    yy, xx = torch.meshgrid(*(torch.arange(n, dtype=torch.float32, device=device) for n in grid), indexing="ij")
    cells, offsets, sizes = [], [], []
    for box, cls in zip(boxes.float(), classes):
        centre = (box[:2] + box[2:]) / 2 / stride - 0.5
        extent = (box[2:] - box[:2]).clamp(min=1.0) / stride
        cell = torch.minimum(centre.round().clamp(min=0), limit).long()
        sigma = (extent / 6).clamp(min=0.5)
        g = torch.exp(-0.5 * (((yy - centre[0]) / sigma[0]) ** 2 + ((xx - centre[1]) / sigma[1]) ** 2))
        heat[cls] = torch.maximum(heat[cls], g)
        heat[cls, cell[0], cell[1]] = 1.0
        cells.append(cell)
        offsets.append(centre - cell)
        sizes.append(extent.log())
    return heat, cells, offsets, sizes


def centre_loss_2d(out, batch):
    """CornerNet focal loss on the heatmaps and L1 on the offset and log size at each object's peak cell,
    all normalised by the number of objects."""
    heat = out["heat"].float()
    B, C, Y, X = heat.shape
    dev = heat.device
    p = heat.sigmoid().clamp(1e-4, 1 - 1e-4)
    total = {k: heat.new_zeros(()) for k in ("heat", "offset", "size")}
    n_obj = 0
    for b in range(B):
        tgt, cells, offsets, sizes = centre_targets_2d(batch["boxes"][b].to(dev), batch["box_classes"][b].tolist(),
                                                       (Y, X), C, dev)
        pos = tgt.eq(1.0)
        pos_loss = -(torch.log(p[b]) * (1 - p[b]) ** 2)[pos].sum()
        neg_loss = -(torch.log(1 - p[b]) * p[b] ** 2 * (1 - tgt) ** 4)[~pos].sum()
        total["heat"] = total["heat"] + pos_loss + neg_loss
        for cell, off, size in zip(cells, offsets, sizes):
            y, x = cell.tolist()
            total["offset"] = total["offset"] + (out["offset"][b, :, y, x].float() - off).abs().sum()
            total["size"] = total["size"] + (out["size"][b, :, y, x].float() - size).abs().sum()
        n_obj += len(cells)
    n = max(n_obj, 1)
    return {k: v / n for k, v in total.items()}


def decode_centres_2d(out, M):
    """Top-M peaks of the first sample -> boxes (M, 4) in pixels, class probabilities (M, C + 1) and
    embeddings (M, E). A slot's no-object probability is 1 - its peak score.
    """
    heat = out["heat"][0].float().sigmoid()
    C, Y, X = heat.shape
    dev = heat.device
    keep = heat == F.max_pool2d(heat[None], 3, stride=1, padding=1)[0]
    scores, idx = (heat * keep).flatten().topk(M)
    cls = idx // (Y * X)
    rest = idx % (Y * X)
    y, x = rest // X, rest % X
    stride = float(STRIDE)
    offset = out["offset"][0][:, y, x].float().T
    size = out["size"][0][:, y, x].float().T.exp() * stride
    centre = (torch.stack([y, x], 1).float() + offset + 0.5) * stride
    boxes = torch.cat([centre - size / 2, centre + size / 2], 1)
    cls_prob = torch.zeros(M, C + 1, device=dev)
    cls_prob[torch.arange(M, device=dev), cls] = scores
    cls_prob[:, C] = 1 - scores
    return boxes, cls_prob, out["feat"][0][:, y, x].T
