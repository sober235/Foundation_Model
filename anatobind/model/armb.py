"""Arm-B minimal path: backbone -> A masks + U_B boxes -> relation -> host CE.

MINIMAL PATH, and the reason no number from this model is M1 evidence: the
geometry features and the existence gating are fed **ground-truth** masks,
presence and matched boxes.  Since IoA is one of the geometry channels, the
relation head can reach a high host accuracy by learning argmax IoA -- which is
the seg-then-lookup baseline.  Before the gate run both call sites marked
``MINIMAL PATH`` must switch to predicted quantities.
"""

import torch
import torch.nn as nn

from anatobind.model.backbone import Backbone
from anatobind.model.decoders import ADecoder, UBDecoder
from anatobind.model.losses import (
    a_loss, build_relation_targets, hungarian_match, rel_loss, total_loss, ub_loss,
)
from anatobind.model.relation import RelationModule, geometry_features


def boxes_vox_to_norm(boxes, shape):
    """(N,6) corner voxels -> (N,6) normalised (cz,cy,cx,dz,dy,dx)."""
    scale = torch.as_tensor(shape, dtype=boxes.dtype, device=boxes.device).repeat(2)
    n = boxes / scale
    centre = (n[..., :3] + n[..., 3:]) / 2
    size = (n[..., 3:] - n[..., :3]).clamp(min=1e-6)
    return torch.cat([centre, size], -1)


class ArmBMinimal(nn.Module):
    def __init__(self, K=6, M=8, d_model=128, embed_dim=32, num_classes=2):
        super().__init__()
        self.K, self.M = K, M
        self.backbone = Backbone(embed_dim=embed_dim)
        c = self.backbone.channels
        self.a_dec = ADecoder(c, d_model=d_model, K=K)
        self.u_dec = UBDecoder(c, d_model=d_model, M=M, num_classes=num_classes)
        self.relation = RelationModule(d_model=d_model, K=K)

    def forward(self, batch):
        image = batch["image"]
        shape = image.shape[2:]
        feats = self.backbone(image)
        a = self.a_dec(feats, out_shape=shape)
        u = self.u_dec(feats)

        tgt_norm = [boxes_vox_to_norm(b.to(image.device), shape) for b in batch["boxes"]]
        matches = [
            hungarian_match(u["logits"][b], u["boxes"][b],
                            batch["box_classes"][b].to(image.device), tgt_norm[b])
            for b in range(image.shape[0])
        ]

        # MINIMAL PATH: ground-truth masks, presence and boxes feed the geometry.
        gt_masks = torch.stack(
            [(batch["seg"] == k + 1) for k in range(self.K)], 1
        ).to(image.dtype)
        query_boxes = image.new_zeros(image.shape[0], self.M, 6)
        for b, (pi, ti) in enumerate(matches):
            if pi.numel():
                query_boxes[b, pi] = batch["boxes"][b][ti].to(image.device, image.dtype)
        geo = geometry_features(gt_masks, query_boxes, batch["spacing_mm"])
        rel = self.relation(a["embed"], u["embed"], geo, batch["present"])

        return {**a, **u, **rel, "matches": matches, "tgt_norm": tgt_norm}

    def compute_loss(self, out, batch):
        device = out["masks"].device
        parts = {}
        parts.update(a_loss(out["masks"], out["presence"], batch["seg"], batch["present"]))
        parts.update(ub_loss(out["logits"], out["boxes"],
                             [c.to(device) for c in batch["box_classes"]],
                             out["tgt_norm"], out["matches"]))
        host_target, rel_target, rel_mask = build_relation_targets(
            out["matches"], batch["host_label"], batch["present"],
            self.K, self.M, device,
        )
        parts.update(rel_loss(out["host_logits"], out["R"], host_target, rel_target, rel_mask))
        return total_loss(parts)

    @torch.no_grad()
    def host_accuracy(self, out, batch):
        """NOT EVIDENCE -- see the module docstring."""
        hit = total = 0
        for b, (pi, ti) in enumerate(out["matches"]):
            if not pi.numel():
                continue
            host = batch["host_label"][b][ti].to(out["host_logits"].device)
            known = host > 0  # unresolved side: no truth to score against
            if not bool(known.any()):
                continue
            pred = out["host_logits"][b, pi[known]].argmax(-1)
            hit += int((pred == host[known] - 1).sum())
            total += int(known.sum())
        return hit, total
