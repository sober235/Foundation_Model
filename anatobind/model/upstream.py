"""Stage-II perception for the M1 batch (RESEARCH_PLAN v2.2 13.6).

image -> anatomy entities A (six identity-anchored queries: masks at the input resolution,
presence, embeddings) and lesion events U_B (DETR queries: four classes plus no-object, a box
normalised to the input extent, embeddings). forward() takes the image and nothing else, so no
ground truth can reach a prediction; tests/test_upstream.py pins the signature (gate G1).
There is no relation module here: binders are trained later on cached, frozen predictions.
"""
import torch.nn as nn

from anatobind.model.backbone import Backbone
from anatobind.model.decoders import FullResMaskHead, UBDecoder, _QueryStack


class Upstream(nn.Module):
    def __init__(self, K=6, M=20, num_classes=4, d_model=256, embed_dim=64, layers=6, heads=8,
                 mask_dim=32, use_checkpoint=False):
        super().__init__()
        self.K, self.M = K, M
        self.backbone = Backbone(embed_dim=embed_dim, use_checkpoint=use_checkpoint)
        c1, c2, c3, c4 = self.backbone.channels
        self.a_stack = _QueryStack((c4, c3, c2), d_model, K, layers, heads)
        self.mask_head = FullResMaskHead(c1, d_model, mask_dim)
        self.presence = nn.Linear(d_model, 1)
        self.u_dec = UBDecoder(self.backbone.channels, d_model=d_model, M=M, layers=layers,
                               num_classes=num_classes, heads=heads)

    def forward(self, image):
        f1, f2, f3, f4 = self.backbone(image)
        q = self.a_stack([f4, f3, f2])
        u = self.u_dec((f1, f2, f3, f4), aux=self.training)
        out = {"masks": self.mask_head(q, f1, image), "presence": self.presence(q).squeeze(-1), "a_embed": q,
               "logits": u["logits"], "boxes": u["boxes"], "u_embed": u["embed"]}
        if "aux" in u:
            out["aux"] = u["aux"]
        return out
