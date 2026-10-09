"""Physical-coordinate 3D rotary position embedding (plan §7; spec N5).

Each attention head's dimension is split into three blocks of DIMS_PER_AXIS (z, y, x) that are rotated by the token's
physical coordinate along that axis, in mm; the remaining dimensions are left alone. Rotating q and k alike makes
q·k a function of the coordinate difference only, so the attention is translation-invariant and sees distances in
mm, whatever the voxel spacing. Computed in float32: at 160 mm and 1 rad / mm the angles are too large for half
precision."""
import torch

DIMS_PER_AXIS = 8            # 3 x 8 = 24 rotated dimensions of each head (heads have 32)
BASE = 1000.0                # frequencies 1 .. 1000^(-3/4) rad / mm: wavelengths 2π·(1 .. 178) mm


def rope_frequencies(dims_per_axis=DIMS_PER_AXIS, base=BASE, device=None):
    """(dims_per_axis / 2,) angular frequencies in rad / mm."""
    i = torch.arange(0, dims_per_axis, 2, device=device, dtype=torch.float32)
    return 1.0 / (base ** (i / dims_per_axis))


def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), -1).flatten(-2)


def apply_rope(x, coords, dims_per_axis=DIMS_PER_AXIS, base=BASE):
    """x: (B, heads, N, d) ; coords: (B, N, 3) mm (z, y, x). Returns x with the first 3·dims_per_axis dimensions of
    every head rotated by the axis coordinates; same shape and dtype as x."""
    if x.shape[-1] < 3 * dims_per_axis:
        raise ValueError(f"head dimension {x.shape[-1]} is too small for 3 x {dims_per_axis} rotated dimensions")
    freqs = rope_frequencies(dims_per_axis, base, x.device)
    xf = x.float()
    cf = coords.float()
    outs = []
    for a in range(3):
        ang = (cf[..., a, None] * freqs).repeat_interleave(2, -1)[:, None]        # (B, 1, N, dims_per_axis)
        part = xf[..., a * dims_per_axis:(a + 1) * dims_per_axis]
        outs.append(part * ang.cos() + rotate_half(part) * ang.sin())
    outs.append(xf[..., 3 * dims_per_axis:])
    return torch.cat(outs, -1).to(x.dtype)
