"""2.5D 病灶检测器（leg 2 spec 2.1）：5 层薄块经 3D stem 压成 2D，再走 MONAI Swin 2D。

不用全 3D 主干：只有 5 层，stem 步长 2 加三次 patch merging 会把深度降到 0。第一条腿已经证明
瓶颈在样本效率而非网络容量，2.5D 让每个标注层都是一个样本（约 8800 个），而非合并后的约 3000 个 3D 病灶。
"""
import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer

from anatobind.model.dense_head_2d import CentreHead2D


class Detector2D(nn.Module):
    def __init__(self, num_classes=5, embed_dim=48, stem_ch=32, d_model=256, slab=5, use_checkpoint=False):
        super().__init__()
        self.slab = slab
        self.stem = nn.Sequential(
            nn.Conv3d(1, stem_ch, (slab, 3, 3), padding=(0, 1, 1)), nn.GELU(),
            nn.Conv3d(stem_ch, stem_ch, (1, 3, 3), padding=(0, 1, 1)), nn.GELU(),
        )
        self.swin = SwinTransformer(
            in_chans=stem_ch, embed_dim=embed_dim, window_size=(8, 8), patch_size=(2, 2),
            depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), spatial_dims=2, use_checkpoint=use_checkpoint,
        )
        del self.swin.layers4          # F1..F4 止于 stride 16；最后一级是死重量
        self.channels = tuple(embed_dim * 2 ** i for i in range(4))
        self.head = CentreHead2D(c1=self.channels[0], num_classes=num_classes, embed_dim=d_model)

    def forward(self, slab):
        x = self.stem(slab).squeeze(2)                      # (B, stem_ch, H, W)
        s = self.swin
        x0 = s.pos_drop(s.patch_embed(x))
        f1 = s.proj_out(x0, True)
        x1 = s.layers1[0](x0.contiguous())
        x2 = s.layers2[0](x1.contiguous())
        x3 = s.layers3[0](x2.contiguous())
        f4 = s.proj_out(x3, True)
        return {**self.head(f1), "global_feat": f4.mean(dim=(-2, -1))}
