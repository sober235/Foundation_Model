"""Variable-size 3D Swin backbone with M_valid key masking and physical-coordinate RoPE (spec N5; plan §4–§8).

The plan's geometry: Conv3D patch stem (2, 4, 4), channels [64, 128, 256, 512], depths [2, 2, 6, 2], heads
[2, 4, 8, 16], window (4, 8, 8). Four feature levels F1..F4 at strides (2, 4, 4), (4, 8, 8), (8, 16, 16), (16, 32, 32):
F_k is the output of stage k before its merging. The window partition, reverse and shift mask come from MONAI; the
attention is our own (RoPE instead of a relative position bias, invalid tokens never serve as keys), and the features
of invalid tokens are zeroed before every stage and before merging so that padding never leaks into valid tokens.
Every level is returned with its own validity, physical coordinates (mm) and local coordinates ([-1, 1]).

Stage I (masked image modelling, SSL-first plan §3.1): `forward(..., visible=...)` takes the patches whose image content
the network may see. The voxels of the other patches are set to MASK_FILL before the patch stem (the stem's kernel
equals its stride, so no hidden voxel reaches any token) and their tokens receive a learned mask token; they stay valid
tokens (keys and queries) with their coordinates. M_valid (the volume's extent) and the visibility are two different
things: a hidden patch is not an invalid one."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import compute_mask, get_window_size, window_partition, window_reverse
from torch.utils.checkpoint import checkpoint

from anatobind.aur.rope import apply_rope

CHANNELS = (64, 128, 256, 512)
DEPTHS = (2, 2, 6, 2)
HEADS = (2, 4, 8, 16)
WINDOW = (4, 8, 8)
PATCH = (2, 4, 4)
KEY_MASK = -1e4
MASK_FILL = 0.0                 # the voxel value of a hidden patch before the stem (Stage I)


def upsample_patches(mask, patch):
    """(B, D', H', W') patch-level mask -> (B, D' p0, H' p1, W' p2) voxel-level mask (each patch repeated)."""
    out = mask
    for axis, p in enumerate(patch, start=1):
        out = out.repeat_interleave(p, dim=axis)
    return out


class WindowAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, coords, valid, shift_mask):
        """x (B', N, C) tokens of B' windows; coords (B', N, 3) mm; valid (B', N) bool; shift_mask (nW, N, N) or None."""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, coords), apply_rope(k, coords)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if shift_mask is not None:
            nw = shift_mask.shape[0]
            attn = (attn.view(b // nw, nw, self.heads, n, n) + shift_mask[None, :, None].to(attn.dtype)).view(b, self.heads, n, n)
        attn = attn.masked_fill(~valid[:, None, None, :], KEY_MASK)
        attn = attn.softmax(-1).to(v.dtype)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, c))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window, shift, use_checkpoint=False):
        super().__init__()
        self.window, self.shift, self.use_checkpoint = tuple(window), tuple(shift), use_checkpoint
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def _attention(self, x, coords, valid, shift_mask):
        """x (B, D, H, W, C); coords (B, D, H, W, 3); valid (B, D, H, W) float."""
        b, d, h, w, c = x.shape
        window, shift = get_window_size((d, h, w), self.window, self.shift)
        pads = [(ws - s % ws) % ws for s, ws in zip((d, h, w), window)]
        pad = (0, pads[2], 0, pads[1], 0, pads[0])
        x = F.pad(self.norm1(x), (0, 0) + pad)
        coords = F.pad(coords, (0, 0) + pad)
        valid = F.pad(valid, pad)
        dims = [b, *x.shape[1:4]]
        shifted = any(s > 0 for s in shift)
        if shifted:
            roll = tuple(-s for s in shift)
            x, coords, valid = (torch.roll(t, shifts=roll, dims=(1, 2, 3)) for t in (x, coords, valid))
        xw = window_partition(x, window)
        cw = window_partition(coords, window)
        vw = window_partition(valid[..., None], window).squeeze(-1) > 0.5
        aw = self.attn(xw, cw, vw, shift_mask if shifted else None)
        x = window_reverse(aw.view(-1, *window, c), window, dims)
        if shifted:
            x = torch.roll(x, shifts=tuple(shift), dims=(1, 2, 3))
        return x[:, :d, :h, :w].contiguous()

    def forward(self, x, coords, valid, shift_mask):
        if self.use_checkpoint and x.requires_grad:
            x = x + checkpoint(self._attention, x, coords, valid, shift_mask, use_reentrant=False)
            return x + checkpoint(lambda t: self.mlp(self.norm2(t)), x, use_reentrant=False)
        x = x + self._attention(x, coords, valid, shift_mask)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """2 x 2 x 2 tokens -> one token of twice the channels (MONAI's V2 merging); coordinates are averaged and the
    validity is 'any', both over the available tokens of each 2-block."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x, coords, valid):
        b, d, h, w, c = x.shape
        x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x = torch.cat([x[:, i::2, j::2, k::2, :] for i in range(2) for j in range(2) for k in range(2)], -1)
        x = self.reduction(self.norm(x))
        return x, pool_coords(coords.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1), pool_valid(valid)


def pool_coords(c):
    """(B, 3, D, H, W) coordinates -> (B, 3, ceil(D/2), ceil(H/2), ceil(W/2)): the mean of each 2-block, odd edges
    replicated so that a lone edge token keeps its own coordinate."""
    d, h, w = c.shape[2:]
    c = F.pad(c, (0, w % 2, 0, h % 2, 0, d % 2), mode="replicate")
    return F.avg_pool3d(c, 2, 2)


def pool_valid(v):
    """(B, D, H, W) validity -> 'any' over each 2-block."""
    d, h, w = v.shape[1:]
    v = F.pad(v[:, None], (0, w % 2, 0, h % 2, 0, d % 2))
    return F.max_pool3d(v, 2, 2)[:, 0]


class Stage(nn.Module):
    def __init__(self, dim, depth, heads, window, merge, use_checkpoint=False):
        super().__init__()
        self.window = tuple(window)
        shift = tuple(s // 2 for s in window)
        self.blocks = nn.ModuleList(SwinBlock(dim, heads, window, (0, 0, 0) if i % 2 == 0 else shift, use_checkpoint)
                                    for i in range(depth))
        self.merge = PatchMerging(dim) if merge else None

    def forward(self, x, coords, valid):
        """Returns (features of this stage (B, D, H, W, C), (x, coords, valid) for the next stage)."""
        b, d, h, w, _ = x.shape
        x = x * valid[..., None]
        window, shift = get_window_size((d, h, w), self.window, tuple(s // 2 for s in self.window))
        padded = [int(math.ceil(s / ws)) * ws for s, ws in zip((d, h, w), window)]
        mask = compute_mask(padded, window, shift, x.device)
        for block in self.blocks:
            x = block(x, coords, valid, mask)
        x = x * valid[..., None]
        if self.merge is None:
            return x, (x, coords, valid)
        return x, self.merge(x, coords, valid)


class SwinBackbone(nn.Module):
    def __init__(self, embed=CHANNELS[0], depths=DEPTHS, heads=HEADS, window=WINDOW, patch=PATCH, in_channels=1,
                 use_checkpoint=False):
        super().__init__()
        self.patch = tuple(patch)
        self.channels = tuple(embed * 2 ** i for i in range(len(depths)))
        self.patch_embed = nn.Conv3d(in_channels, embed, kernel_size=self.patch, stride=self.patch)
        self.local_embed = nn.Sequential(nn.Linear(3, embed), nn.GELU(), nn.Linear(embed, embed))
        self.mask_token = nn.Parameter(torch.zeros(embed))
        self.stages = nn.ModuleList(Stage(self.channels[i], depths[i], heads[i], window, merge=i < len(depths) - 1,
                                          use_checkpoint=use_checkpoint) for i in range(len(depths)))

    def forward(self, image, valid, coords, local, visible=None):
        """image (B, 1, D, H, W) with D, H, W multiples of the patch; valid (B, D, H, W) float 0/1; coords and local
        (B, 3, D, H, W); visible (B, D/p0, H/p1, W/p2) bool or None (every patch visible). Contract: the invalid voxels
        of `image` hold the constant -1 (the normalised background value, crops.normalise / dataset.make_crop), so a
        patch that is only partly valid is a valid token that sees that constant like any image border; no image
        content ever lies under `valid == 0`. With `visible`, the voxels of the hidden patches are replaced by MASK_FILL
        before the stem and their tokens get the mask token added (see the module docstring). Returns a list of levels,
        each {"feat": (B, C, D', H', W'), "valid": (B, D', H', W') float, "coords": (B, 3, D', H', W') mm,
        "local": (B, 3, D', H', W')}."""
        if any(s % p for s, p in zip(image.shape[2:], self.patch)):
            raise ValueError(f"spatial shape {tuple(image.shape[2:])} is not a multiple of the patch {self.patch}")
        if visible is not None:
            grid = tuple(s // p for s, p in zip(image.shape[2:], self.patch))
            if tuple(visible.shape) != (image.shape[0],) + grid:
                raise ValueError(f"visible {tuple(visible.shape)} is not the patch grid {(image.shape[0],) + grid}")
            visible = visible.bool()
            image = torch.where(upsample_patches(visible, self.patch)[:, None], image, torch.full_like(image, MASK_FILL))
        x = self.patch_embed(image)
        if visible is not None:
            x = x + (~visible)[:, None].to(x.dtype) * self.mask_token.view(1, -1, 1, 1, 1).to(x.dtype)
        c = F.avg_pool3d(coords, self.patch, self.patch)
        l = F.avg_pool3d(local, self.patch, self.patch)
        v = F.max_pool3d(valid[:, None], self.patch, self.patch)[:, 0]
        x = x + self.local_embed(l.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x, c = x.permute(0, 2, 3, 4, 1), c.permute(0, 2, 3, 4, 1)
        levels = []
        for stage in self.stages:
            feat, (x_next, c_next, v_next) = stage(x, c, v)
            levels.append({"feat": feat.permute(0, 4, 1, 2, 3).contiguous(), "valid": v,
                           "coords": c.permute(0, 4, 1, 2, 3).contiguous(), "local": l})
            l = pool_coords(l) if stage.merge is not None else l
            x, c, v = x_next, c_next, v_next
        return levels

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
