"""Block masking on the patch grid for Stage I (SSL-first plan §3.1–3.2; decision Q11 of 2026-10-09).

A hidden patch is a patch the network may not see (its voxels are replaced before the stem, its token gets the mask
token; anatobind.aur.swin). Masks are drawn on the patch grid of a crop over its *foreground* patches (valid patches
with at least one voxel above the background constant): contiguous 3D blocks whose edges are drawn in millimetres
(16–32 mm by default, rounded to whole patches, at least one patch) are placed at random foreground patches until the
hidden share of the foreground reaches the ratio. Masking pure background would be free: every hidden patch lies in
the foreground, and the ratio is a share of the foreground, not of the crop. Randomness comes from a torch generator,
so a (seed, step, rank) triple reproduces a mask. The MIM target weight is `hidden voxels ∧ valid ∧ foreground`."""
import math

import torch
import torch.nn.functional as F

from anatobind.aur.swin import PATCH, upsample_patches

MASK_RATIO = 0.6
BLOCK_MM = (16.0, 32.0)
BACKGROUND = -1.0


def patch_grid(shape, patch=PATCH):
    if any(s % p for s, p in zip(shape, patch)):
        raise ValueError(f"spatial shape {tuple(shape)} is not a multiple of the patch {tuple(patch)}")
    return tuple(s // p for s, p in zip(shape, patch))


def foreground_patches(image, valid, patch=PATCH, background=BACKGROUND):
    """image (B, 1, D, H, W), valid (B, D, H, W) -> (B, D', H', W') bool: the patch holds a valid voxel above the
    background constant."""
    fg = ((image[:, 0] > background) & (valid > 0.5)).float()[:, None]
    return F.max_pool3d(fg, kernel_size=tuple(patch), stride=tuple(patch))[:, 0] > 0.5


def block_mask(foreground, generator, ratio=MASK_RATIO, block_mm=BLOCK_MM, spacing_mm=(1.0, 1.0, 1.0), patch=PATCH, max_blocks=4096):
    """foreground (D', H', W') bool -> hidden (D', H', W') bool, a subset of the foreground covering about `ratio` of it
    with random 3D blocks. Returns an empty mask when the foreground is empty."""
    fg = foreground.bool()
    n_fg = int(fg.sum())
    hidden = torch.zeros_like(fg)
    if n_fg == 0 or ratio <= 0:
        return hidden
    centres = torch.nonzero(fg.cpu())                                        # (n_fg, 3) patch indices
    sizes_mm = [(float(p) * float(s)) for p, s in zip(patch, spacing_mm)]   # patch edge in mm per axis
    target = int(round(ratio * n_fg))
    lo, hi = float(block_mm[0]), float(block_mm[1])
    for _ in range(max_blocks):
        if int(hidden.sum()) >= target:
            break
        c = centres[int(torch.randint(len(centres), (1,), generator=generator))]
        edges = lo + (hi - lo) * torch.rand(3, generator=generator)
        half = [max(1, int(round(float(e) / mm))) // 2 for e, mm in zip(edges, sizes_mm)]
        full = [max(1, int(round(float(e) / mm))) for e, mm in zip(edges, sizes_mm)]
        sl = tuple(slice(max(int(ci) - h, 0), min(int(ci) - h + f, n)) for ci, h, f, n in zip(c.tolist(), half, full, fg.shape))
        hidden[sl] |= fg[sl]
    return hidden


def batch_masks(foreground, generator, ratio=MASK_RATIO, block_mm=BLOCK_MM, spacing_mm=(1.0, 1.0, 1.0), patch=PATCH):
    """(B, D', H', W') foreground -> (B, D', H', W') hidden, one independent block mask per sample."""
    return torch.stack([block_mask(foreground[b], generator, ratio, block_mm, spacing_mm, patch) for b in range(foreground.shape[0])])


def hidden_voxels(hidden, patch=PATCH):
    """(B, D', H', W') hidden patches -> (B, D, H, W) bool voxels."""
    return upsample_patches(hidden.bool(), patch)


def target_weight(hidden, image, valid, patch=PATCH, background=BACKGROUND):
    """(B, D, H, W) float: 1 on the voxels the MIM loss reads (hidden ∧ valid ∧ foreground), 0 elsewhere."""
    return (hidden_voxels(hidden, patch) & (valid > 0.5) & (image[:, 0] > background)).float()


def hidden_share(hidden, foreground):
    """The share of the foreground patches that is hidden, per sample (B,)."""
    fg = foreground.bool().flatten(1).float().sum(1)
    return hidden.bool().flatten(1).float().sum(1) / fg.clamp(min=1.0)


def expected_blocks(ratio, n_fg, block_mm, spacing_mm, patch):
    """A rough count of blocks for a mask (for the records), not used by the sampler."""
    mid = [(block_mm[0] + block_mm[1]) / 2 / (p * s) for p, s in zip(patch, spacing_mm)]
    return int(math.ceil(ratio * n_fg / max(1.0, math.prod(mid))))
