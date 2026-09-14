"""两个无参照的可靠性头（leg 2 spec 2.3）。

逐病灶头只能评价检测器找到的东西，漏检对它不可见；扫描级头专门覆盖漏检，因此必须是两个。
两者都只看单张图的特征与该图上检出的统计，不看任何干净图参照。
"""
import numpy as np
import torch
import torch.nn as nn


def _mlp(d_in, hidden):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, 1))


class LesionReliability(nn.Module):
    def __init__(self, embed_dim=256, n_scalar=3, hidden=128):
        super().__init__()
        self.net = _mlp(embed_dim + n_scalar, hidden)

    def forward(self, feat, scalars):
        return self.net(torch.cat([feat, scalars], -1)).squeeze(-1)


class ScanReliability(nn.Module):
    def __init__(self, global_dim, n_scalar=5, hidden=128):
        super().__init__()
        self.net = _mlp(global_dim + n_scalar, hidden)

    def forward(self, global_feat, scalars):
        return self.net(torch.cat([global_feat, scalars], -1)).squeeze(-1)


def lesion_scalars(row):
    """峰值得分、top-1 与 no-object 之差、框体积的对数。"""
    z0, y0, x0, z1, y1, x1 = row["box"]
    vol = max((z1 - z0) * (y1 - y0) * (x1 - x0), 1.0)
    return np.array([row["score"], 2 * row["score"] - 1.0, np.log(vol)], dtype=np.float32)


def scan_scalars(rows):
    """检出数，以及峰值得分的最大 / 中位 / 最小与均值；没有检出时用 0 填。"""
    if not rows:
        return np.zeros(5, dtype=np.float32)
    s = np.array([r["score"] for r in rows], dtype=np.float32)
    return np.array([len(rows), s.max(), float(np.median(s)), s.min(), s.mean()], dtype=np.float32)
