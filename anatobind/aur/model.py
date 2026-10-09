"""The AnatoBind brain model: backbone + entity / event / sequence heads + mask head + host competition (spec §5)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from anatobind.aur.heads import EntityDecoder, EventDecoder, MaskHead, SequenceHead
from anatobind.aur.labels import N_ENTITIES
from anatobind.aur.relation import CandidateCompetition, geometry, host_from_entities, host_masks_from_entities
from anatobind.aur.swin import CHANNELS, DEPTHS, HEADS, PATCH, WINDOW, SwinBackbone

DEFAULTS = {"embed": CHANNELS[0], "depths": DEPTHS, "heads": HEADS, "window": WINDOW, "patch": PATCH, "d_model": 256,
            "n_events": 64, "mask_dim": 32, "pixel_dim": 64, "dec_layers": 3, "dec_heads": 8, "rel_layers": 2, "rel_heads": 4,
            "use_checkpoint": True}


class AnatoBindBrain(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = {**DEFAULTS, **kwargs}
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"unknown model options {sorted(unknown)}")
        self.cfg = cfg
        self.backbone = SwinBackbone(cfg["embed"], cfg["depths"], cfg["heads"], cfg["window"], cfg["patch"],
                                     use_checkpoint=cfg["use_checkpoint"])
        ch = self.backbone.channels
        d = cfg["d_model"]
        self.entities = EntityDecoder(ch, d, N_ENTITIES, cfg["dec_layers"], cfg["dec_heads"])
        self.events = EventDecoder(ch, d, cfg["n_events"], cfg["dec_layers"], cfg["dec_heads"])
        self.sequence = SequenceHead(ch[3], d)
        self.masks = MaskHead(ch, d, cfg["patch"], cfg["pixel_dim"], cfg["mask_dim"])
        self.relation = CandidateCompetition(d, layers=cfg["rel_layers"], heads=cfg["rel_heads"])

    def forward(self, image, valid, coords, local):
        """See SwinBackbone.forward for the inputs. Returns the heads' outputs and the pixel features; masks are made
        on demand with entity_masks / event_masks (points or full) so that training never materialises them densely."""
        levels = self.backbone(image, valid, coords, local)
        a, u, s = self.entities(levels), self.events(levels), self.sequence(levels[3])
        pix = self.masks.pixels(levels, image)
        return {"levels": levels, "entity_embed": a["embed"], "entity_presence": a["presence"], "event_embed": u["embed"],
                "event_presence": u["presence"], "seq_embed": s["embed"], "seq_logits": s["logits"], "pix": pix}

    def entity_masks(self, out, points=None):
        return self.masks.full_masks(out["entity_embed"], out["pix"], points)

    def event_masks(self, out, points=None):
        return self.masks.full_masks(out["event_embed"], out["pix"], points)

    def bind(self, out, b, event_idx):
        """Host logits (N, 14) of the events `event_idx` (N,) of sample b, from this sample's own predicted masks on
        the coarse grid (N8). event_idx may be the matched queries (training) or the present ones (inference)."""
        pix = {"coarse": out["pix"]["coarse"][b:b + 1]}
        ent = self.masks.coarse_masks(out["entity_embed"][b:b + 1], pix)[0].sigmoid()                    # (32, D1, H1, W1)
        ev = self.masks.coarse_masks(out["event_embed"][b:b + 1, event_idx], pix)[0].sigmoid()            # (N, D1, H1, W1)
        valid = out["levels"][0]["valid"][b]                                                              # (D1, H1, W1)
        hosts = host_masks_from_entities(ent[None])[0] * valid
        geo = geometry(ev * valid, hosts, out["levels"][0]["coords"][b])
        host_embed = host_from_entities(out["entity_embed"][b:b + 1])[0]
        return self.relation(out["event_embed"][b, event_idx], host_embed, out["seq_embed"][b], geo)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
