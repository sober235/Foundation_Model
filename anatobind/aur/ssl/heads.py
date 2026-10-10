"""Stage I heads (SSL-first plan §3.2): the masked-patch decoder and the MIM loss.

The decoder reads all four levels of the backbone: every level is projected to `width` channels by a 1 x 1 x 1
convolution, the coarser map is upsampled (trilinear) to the next finer grid and added (top-down, F4 -> F1), and two
1 x 1 x 1 convolutions predict, for every F1 patch, its 2 x 4 x 4 normalised voxel intensities, rearranged to a voxel
image. Every stage therefore receives the reconstruction gradient: the first Stage I run (2026-10-09) decoded from F1
alone, which left stages 2-4 (10 of the 12 Swin blocks) to the contrastive term and failed gate G1. The decoder never
reads the input image, only the levels, so the hidden voxels stay out of reach. It is a light head that is never
carried into Stage II. The loss is a Huber loss on the hidden foreground voxels of each sample, normalised per sample
(a big head and a small head weigh the same), averaged over the samples that hide anything."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from anatobind.aur.swin import CHANNELS, PATCH

HUBER_DELTA = 1.0


class MaskedPatchDecoder(nn.Module):
    def __init__(self, channels=CHANNELS, patch=PATCH, width=64, hidden=None):
        super().__init__()
        self.patch = tuple(patch)
        self.channels = tuple(channels)
        hidden = hidden or width
        self.lateral = nn.ModuleList(nn.Conv3d(c, width, 1) for c in self.channels)
        self.net = nn.Sequential(nn.GELU(), nn.Conv3d(width, hidden, 1), nn.GELU(), nn.Conv3d(hidden, math.prod(self.patch), 1))

    def forward(self, levels):
        """levels: the backbone's list of {"feat": (B, C_k, D_k, H_k, W_k)}, finest first -> (B, 1, D1 p0, H1 p1, W1 p2)
        predicted intensities on the F1 grid."""
        if len(levels) != len(self.lateral):
            raise ValueError(f"the decoder reads {len(self.lateral)} levels, got {len(levels)}")
        x = self.lateral[-1](levels[-1]["feat"])
        for k in range(len(levels) - 2, -1, -1):
            lat = self.lateral[k](levels[k]["feat"])
            x = lat + F.interpolate(x.float(), size=lat.shape[2:], mode="trilinear", align_corners=False).to(lat.dtype)
        return patches_to_voxels(self.net(x), self.patch)


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
