"""Attribution diagnostics of a Stage I model (review of 2026-10-10; G1.md, round 1): which term drives which part of
the backbone, and how much the reconstruction leans on F1 rather than the deeper levels. Recorded at the early probes
so that a failed round can be attributed before the next round's single change is chosen (pre-registered rule in
G1.md). Read-only on the weights; NOT a gate.

gradient_balance: on one batch, the per-parameter gradient RMS of each backbone part (stem = patch embedding and local
coordinate embedding; the blocks of each stage; each merge) from the reconstruction loss alone and from the weighted
contrastive loss alone (the masks drawn from the same seed for both). reconstruction_ablation: on one batch, the masked
reconstruction loss of view 1 with all levels, without F1 (levels 2-4 only) and with F1 only, and the no-learning
baseline."""
import torch

from anatobind.aur.ssl.heads import interpolation_baseline, mim_loss
from anatobind.aur.ssl.masking import batch_masks, foreground_patches, target_weight


def _parts(model):
    bb = model.backbone
    parts = {"stem": [bb.patch_embed, bb.local_embed]}
    for k, stage in enumerate(bb.stages, start=1):
        parts[f"stage{k}.blocks"] = [stage.blocks]
        if stage.merge is not None:
            parts[f"stage{k}.merge"] = [stage.merge]
    return parts


def _rms(modules):
    g = [p.grad.flatten().float() for m in modules for p in m.parameters() if p.grad is not None]
    return float(torch.cat(g).pow(2).mean().sqrt()) if g else 0.0


def gradient_balance(model, batch, seed=0):
    """{"parts": {name: {"mim", "contrast", "ratio_mim_over_contrast"}}, "mim", "contrast", "contrast_weight"}; the
    contrast column is the gradient of contrast_weight x InfoNCE, the term as it enters the loss."""
    w = float(model.cfg["contrast_weight"])
    parts = _parts(model)
    res, values = {}, {}
    was_training = model.training
    model.train()
    for term in ("mim", "contrast"):
        model.zero_grad(set_to_none=True)
        out = model(batch, torch.Generator().manual_seed(seed))
        values[term] = float(out[term])
        (out["mim"] if term == "mim" else w * out["contrast"]).backward()
        res[term] = {name: _rms(mods) for name, mods in parts.items()}
        del out
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    table = {name: {"mim": res["mim"][name], "contrast": res["contrast"][name],
                    "ratio_mim_over_contrast": res["mim"][name] / max(res["contrast"][name], 1e-30)} for name in parts}
    return {"parts": table, "mim": values["mim"], "contrast": values["contrast"], "contrast_weight": w}


@torch.no_grad()
def reconstruction_ablation(model, batch, seed=0):
    """The masked reconstruction loss of view 1 with the decoder reading all levels, levels 2-4 only, F1 only."""
    cfg = model.cfg
    v1, valid = batch["view1"], batch["valid"]
    spacing = tuple(float(s) for s in batch["spacing"][0].tolist())
    fg = foreground_patches(v1, valid, cfg["patch"])
    hidden = batch_masks(fg, torch.Generator().manual_seed(seed), cfg["mask_ratio"], cfg["block_mm"], spacing, cfg["patch"])
    levels = model.backbone(v1, valid, batch["coords"], batch["local"], visible=~hidden)
    weight = target_weight(hidden, v1, valid, cfg["patch"])
    out = {}
    for name, use in (("full", None), ("no_f1", (False, True, True, True)), ("f1_only", (True, False, False, False))):
        out[name] = float(mim_loss(model.decoder(levels, use=use).float(), v1, weight)[0])
    out["baseline"] = float(interpolation_baseline(v1, weight, valid, cfg["patch"]))
    out["hidden_voxels"] = int(weight.sum())
    return out
