"""Stage I heads (SSL-first plan §3.2): the masked-patch decoder and the MIM loss.

The decoder reads F1 (stride = the patch, 64 channels) and predicts, for every patch, its 2 x 4 x 4 normalised voxel
intensities with two 1 x 1 x 1 convolutions; the prediction is rearranged to a voxel image. It is a light head that is
never carried into Stage II. The loss is a Huber loss on the hidden foreground voxels of each sample, normalised per
sample (a big head and a small head weigh the same), averaged over the samples that hide anything."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from anatobind.aur.swin import PATCH

HUBER_DELTA = 1.0


class MaskedPatchDecoder(nn.Module):
    def __init__(self, in_channels=64, patch=PATCH, hidden=None):
        super().__init__()
        self.patch = tuple(patch)
        hidden = hidden or in_channels
        self.net = nn.Sequential(nn.Conv3d(in_channels, hidden, 1), nn.GELU(), nn.Conv3d(hidden, math.prod(self.patch), 1))

    def forward(self, f1):
        """f1 (B, C, D', H', W') -> (B, 1, D' p0, H' p1, W' p2) predicted intensities."""
        return patches_to_voxels(self.net(f1), self.patch)


def patches_to_voxels(x, patch=PATCH):
    """(B, p0 p1 p2, D', H', W') -> (B, 1, D' p0, H' p1, W' p2): channel c = (i p1 + j) p2 + k lands at voxel offset
    (i, j, k) of its patch."""
    b, c, d, h, w = x.shape
    p0, p1, p2 = patch
    if c != p0 * p1 * p2:
        raise ValueError(f"{c} channels do not hold a {patch} patch")
    x = x.view(b, p0, p1, p2, d, h, w).permute(0, 4, 1, 5, 2, 6, 3)
    return x.reshape(b, 1, d * p0, h * p1, w * p2)


def voxels_to_patches(x, patch=PATCH):
    """The inverse of patches_to_voxels: (B, 1, D, H, W) -> (B, p0 p1 p2, D', H', W')."""
    b, _, D, H, W = x.shape
    p0, p1, p2 = patch
    d, h, w = D // p0, H // p1, W // p2
    x = x.view(b, d, p0, h, p1, w, p2).permute(0, 2, 4, 6, 1, 3, 5)
    return x.reshape(b, p0 * p1 * p2, d, h, w)


def mim_loss(pred, target, weight, delta=HUBER_DELTA):
    """pred, target (B, 1, D, H, W); weight (B, D, H, W) in {0, 1} -> (loss, voxels read): Huber on the weighted voxels,
    normalised per sample, averaged over the samples with any weight; zero (with a gradient path) when none has."""
    per_voxel = F.smooth_l1_loss(pred[:, 0], target[:, 0], reduction="none", beta=delta) * weight
    n = weight.flatten(1).sum(1)
    per_sample = per_voxel.flatten(1).sum(1) / n.clamp(min=1.0)
    has = (n > 0).to(per_sample.dtype)
    return (per_sample * has).sum() / has.sum().clamp(min=1.0), int(n.sum())


def interpolation_baseline(target, weight, valid, patch=PATCH):
    """A no-learning reference for the MIM metric: every voxel predicted by the mean of the visible voxels of its
    sample. Returns the same per-sample-normalised Huber as mim_loss. The hidden voxels' own values never enter."""
    visible = (valid > 0.5).float() * (1.0 - weight)
    mean = (target[:, 0] * visible).flatten(1).sum(1) / visible.flatten(1).sum(1).clamp(min=1.0)
    pred = mean.view(-1, 1, 1, 1, 1).expand_as(target)
    return mim_loss(pred, target, weight)[0]
