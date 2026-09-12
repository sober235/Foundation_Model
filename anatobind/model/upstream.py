"""Stage-II perception for the M1 batch (RESEARCH_PLAN v2.2 13.6).

image -> anatomy entities A (six identity-anchored queries: masks at the input resolution, presence,
embeddings) and lesion events U_B (a dense centre-heatmap head on the F1 grid: per-class heatmaps,
sub-cell offsets, box sizes and per-cell embeddings; user decision Q17 replaced the DETR queries, see
anatobind/model/dense_head.py). forward() takes the image and nothing else, so no ground truth can
reach a prediction; tests/test_upstream.py pins the signature (gate G1). There is no relation module
here: binders are trained later on cached, frozen predictions.
"""
import torch.nn as nn

from anatobind.model.backbone import Backbone
from anatobind.model.decoders import FullResMaskHead, PixelDecoder, _QueryStack
from anatobind.model.dense_head import CentreHead


class Upstream(nn.Module):
    def __init__(self, K=6, M=20, num_classes=4, d_model=256, embed_dim=64, layers=6, heads=8,
                 mask_dim=32, use_checkpoint=False, pixel_dim=64):
        super().__init__()
        self.K, self.M = K, M  # M: lesion slots kept when the heatmaps are decoded
        self.backbone = Backbone(embed_dim=embed_dim, use_checkpoint=use_checkpoint)
        c1, c2, c3, c4 = self.backbone.channels
        self.a_stack = _QueryStack((c4, c3, c2), d_model, K, layers, heads)
        # pixel_dim=0 keeps the 2026-09-12 pilot's configuration (raw F1 under the heads) for comparison
        self.pixel = PixelDecoder(self.backbone.channels, pixel_dim) if pixel_dim else None
        head_in = pixel_dim or c1
        self.mask_head = FullResMaskHead(head_in, d_model, mask_dim)
        self.presence = nn.Linear(d_model, 1)
        self.u_head = CentreHead(head_in, num_classes=num_classes, embed_dim=d_model)

    def forward(self, image):
        f1, f2, f3, f4 = self.backbone(image)
        q = self.a_stack([f4, f3, f2])
        pix = self.pixel((f1, f2, f3, f4)) if self.pixel is not None else f1
        return {"masks": self.mask_head(q, pix, image), "presence": self.presence(q).squeeze(-1), "a_embed": q,
                **self.u_head(pix)}
