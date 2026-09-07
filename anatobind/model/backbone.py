"""Shrunk variable-size 3D Swin backbone for the arm-B minimal path.

Decision A10 (2026-09-07): MONAI's ``SwinTransformer`` is configured to the
geometry of RESEARCH_PLAN.md 4.2 rather than hand-writing the blocks.  The stem
(2, 4, 4) and window (4, 8, 8) are the plan's; ``embed_dim`` is shrunk from 64 to
32 per decision A5, so this is a throwaway scaffold, not the gate-run scale.

Deferred against spec, and required before the M1 gate run:
  * ``M_valid`` key masking (4.1, 4.2) -- crops here are always taken fully
    inside the volume, so nothing is padded.  Sliding-window inference and
    variable-size volumes need it.
  * physical-coordinate 3D RoPE (4.2) -- MONAI uses relative position bias.
    4.2 itself flags the physical-PE claim as an assertion needing ablation.

MONAI exposes the patch-embedding output and the post-merge output of each
stage, so F1 carries no attention; it is the high-resolution map the mask head
reads.  F2..F4 are the cross-attention memory.
"""

import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer


class Backbone(nn.Module):
    def __init__(
        self,
        embed_dim=32,
        depths=(2, 2, 6, 2),
        num_heads=(2, 4, 8, 16),
        patch_size=(2, 4, 4),
        window_size=(4, 8, 8),
        in_chans=1,
    ):
        super().__init__()
        self.swin = SwinTransformer(
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            spatial_dims=3,
        )
        # F1..F4 stop at stride (16,32,32); the last stage would be dead weight.
        del self.swin.layers4
        self.channels = tuple(embed_dim * 2**i for i in range(4))

    def forward(self, x):
        s = self.swin
        x0 = s.pos_drop(s.patch_embed(x))
        f1 = s.proj_out(x0, True)
        x1 = s.layers1[0](x0.contiguous())
        f2 = s.proj_out(x1, True)
        x2 = s.layers2[0](x1.contiguous())
        f3 = s.proj_out(x2, True)
        x3 = s.layers3[0](x2.contiguous())
        f4 = s.proj_out(x3, True)
        return f1, f2, f3, f4

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
