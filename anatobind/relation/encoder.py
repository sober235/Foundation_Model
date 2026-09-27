"""The shared lesion encoder E_u of spec 2026-09-27 §5.4 and the patch handling around it: centred crops of the stored
48 px window, and the augmentation that keeps image and geometry consistent (a left-right flip negates dx and swaps
the side one-hots, spec §5.7)."""
import numpy as np
import torch
import torch.nn as nn

from anatobind.relation.table import PATCH_PX, PATCH_SLICES

# tie order (P11): the §10.1 initial config, then by input size px² x slices ascending
TORCH_CONFIGS = ({"px": 32, "slices": 3}, {"px": 32, "slices": 1}, {"px": 22, "slices": 3}, {"px": 42, "slices": 3})
DX_INDEX = 1                     # per-slot layout: [candidate, dx, dy, dz, centroid_distance, signed, min_surface, ioa, soft, ...]
SIDE_LEFT, SIDE_RIGHT = 9 + 12, 9 + 13     # lesion block starts at 9: 8 geometry + 4 extras, then side one-hot (left, right, midline)


def config_key(cfg):
    return f"px{cfg['px']}_s{cfg['slices']}"


def crop(image, mask, px, slices):
    """(N, 2 * slices, px, px) float32: for each kept slice the image then the mask; the crop is centred in the window."""
    if slices not in (1, PATCH_SLICES):
        raise ValueError(slices)
    lo = (PATCH_PX - px) // 2
    keep = [PATCH_SLICES // 2] if slices == 1 else list(range(PATCH_SLICES))
    img = np.asarray(image, np.float32)[:, keep, lo:lo + px, lo:lo + px]
    msk = np.asarray(mask, np.float32)[:, keep, lo:lo + px, lo:lo + px]
    n = img.shape[0]
    out = np.empty((n, 2 * slices, px, px), np.float32)
    out[:, 0::2] = img
    out[:, 1::2] = msk
    return out


def flip_batch(x, geo):
    xf = np.ascontiguousarray(x[:, :, ::-1, :])
    gf = np.array(geo, copy=True)
    gf[:, :, DX_INDEX] = -geo[:, :, DX_INDEX]
    gf[:, :, SIDE_LEFT], gf[:, :, SIDE_RIGHT] = geo[:, :, SIDE_RIGHT], geo[:, :, SIDE_LEFT]
    return xf, gf


def augment(x, geo, rng, flip_p=0.5, intensity=0.1):
    x, geo = np.array(x, np.float32, copy=True), np.array(geo, np.float32, copy=True)
    flip = rng.uniform(size=len(x)) < flip_p
    if flip.any():
        xf, gf = flip_batch(x[flip], geo[flip])
        x[flip], geo[flip] = xf, gf
    if intensity > 0:
        scale = rng.uniform(1 - intensity, 1 + intensity, size=(len(x), 1, 1, 1)).astype(np.float32)
        x[:, 0::2] *= scale
    return x, geo


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(), nn.MaxPool2d(2))


class LesionEncoder(nn.Module):
    """Four conv blocks 32-64-64-128 and global average pooling -> 128 (spec §5.4)."""

    def __init__(self, in_ch):
        super().__init__()
        self.blocks = nn.Sequential(_block(in_ch, 32), _block(32, 64), _block(64, 64), _block(64, 128))
        self.out_dim = 128

    def forward(self, x):
        return self.blocks(x).mean((2, 3))
