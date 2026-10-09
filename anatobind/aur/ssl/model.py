"""The Stage I model (SSL-first plan §3; decision Q11 of 2026-10-09): the shared Swin backbone with the two auxiliary
heads, and the Stage I objective L_I = L_MIM + λ_c L_contrast.

Each of the two views gets its own block mask (drawn here, on the device, from a generator seeded per step and rank),
goes through the backbone with the hidden patches replaced, and is reconstructed by the masked-patch decoder; the MIM
loss of a view reads only its own hidden foreground voxels against the view's own (unmasked) intensities, so neither
view ever supplies the other's targets. The projector pools F4 of each masked view; the InfoNCE pairs the two views of
a crop against the other patients' crops of the global batch. Only the backbone is carried into Stage II."""
import torch
import torch.nn as nn

from anatobind.aur.ssl.contrast import CONTRAST_WEIGHT, TEMPERATURE, Projector, contrast_loss
from anatobind.aur.ssl.heads import MaskedPatchDecoder, interpolation_baseline, mim_loss
from anatobind.aur.ssl.masking import BLOCK_MM, MASK_RATIO, batch_masks, foreground_patches, hidden_share, target_weight
from anatobind.aur.swin import CHANNELS, DEPTHS, HEADS, PATCH, WINDOW, SwinBackbone

DEFAULTS = {"embed": CHANNELS[0], "depths": DEPTHS, "heads": HEADS, "window": WINDOW, "patch": PATCH, "use_checkpoint": False,
            "mask_ratio": MASK_RATIO, "block_mm": BLOCK_MM, "contrast_dim": 128, "contrast_weight": CONTRAST_WEIGHT,
            "temperature": TEMPERATURE}


class StageOne(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = {**DEFAULTS, **kwargs}
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"unknown Stage I options {sorted(unknown)}")
        self.cfg = cfg
        self.backbone = SwinBackbone(cfg["embed"], cfg["depths"], cfg["heads"], cfg["window"], cfg["patch"], use_checkpoint=cfg["use_checkpoint"])
        self.decoder = MaskedPatchDecoder(self.backbone.channels[0], cfg["patch"])
        self.projector = Projector(self.backbone.channels[3], self.backbone.channels[3], cfg["contrast_dim"])

    def view(self, image, valid, coords, local, hidden):
        """One masked view -> (reconstruction (B, 1, D, H, W), projection (B, d), levels)."""
        levels = self.backbone(image, valid, coords, local, visible=~hidden)
        return self.decoder(levels[0]["feat"]), self.projector(levels[3]["feat"], levels[3]["valid"]), levels

    def forward(self, batch, generator):
        """batch: view1, view2 (B, 1, D, H, W); valid (B, D, H, W); coords, local (B, 3, D, H, W); spacing (B, 3) mm (z, y, x);
        patient (B,) int64. Returns {"loss", "mim", "contrast", ...stats}. The masks are drawn from `generator`."""
        v1, v2, valid, coords, local = batch["view1"], batch["view2"], batch["valid"], batch["coords"], batch["local"]
        spacing = tuple(float(s) for s in batch["spacing"][0].tolist())
        fg = foreground_patches(v1, valid, self.cfg["patch"])
        hidden1 = batch_masks(fg, generator, self.cfg["mask_ratio"], self.cfg["block_mm"], spacing, self.cfg["patch"])
        hidden2 = batch_masks(fg, generator, self.cfg["mask_ratio"], self.cfg["block_mm"], spacing, self.cfg["patch"])
        pred1, z1, _ = self.view(v1, valid, coords, local, hidden1)
        pred2, z2, _ = self.view(v2, valid, coords, local, hidden2)
        w1, w2 = target_weight(hidden1, v1, valid, self.cfg["patch"]), target_weight(hidden2, v2, valid, self.cfg["patch"])
        mim1, n1 = mim_loss(pred1, v1, w1)
        mim2, n2 = mim_loss(pred2, v2, w2)
        mim = 0.5 * (mim1 + mim2)
        contrast, acc = contrast_loss(z1, z2, batch["patient"], self.cfg["temperature"])
        loss = mim + self.cfg["contrast_weight"] * contrast
        with torch.no_grad():
            baseline = 0.5 * (interpolation_baseline(v1, w1, valid, self.cfg["patch"]) + interpolation_baseline(v2, w2, valid, self.cfg["patch"]))
            share = 0.5 * (hidden_share(hidden1, fg) + hidden_share(hidden2, fg)).mean()
        return {"loss": loss, "mim": mim, "contrast": contrast, "contrast_acc": acc, "mim_baseline": baseline,
                "hidden_share": share, "hidden_voxels": torch.tensor(float(n1 + n2)), "z1": z1, "z2": z2}

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def effective_rank(z, eps=1e-8):
    """The exponential of the entropy of the normalised singular values of z (N, d): a collapse monitor (plan §3.3)."""
    z = z.float() - z.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(z)
    p = s / s.sum().clamp(min=eps)
    return torch.exp(-(p * torch.log(p.clamp(min=eps))).sum())
