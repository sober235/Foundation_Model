"""Upstream predictions for evaluation and for the stage-III binders (RESEARCH_PLAN v2.2 13.6).

The label map gives each voxel the anatomy query with the highest probability when that
probability exceeds 0.5, else background. Lesion slots are the top-M peaks of the centre heatmaps
(anatobind/model/dense_head.py), with boxes as voxel corners (z0, y0, x0, z1, y1, x1); padding is at
the end of Z, so these are also coordinates in the unpadded volume.
"""
import numpy as np
import torch

from anatobind.model.dense_head import decode_centres

MASK_THRESHOLD = 0.5


@torch.no_grad()
def predict_volume(model, image, valid_depth, device):
    model.eval()
    x = image.to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(x)
    prob = out["masks"].float().sigmoid()[0, :, :valid_depth]
    best = prob.max(0)
    label_map = torch.where(best.values > MASK_THRESHOLD, best.indices + 1, torch.zeros_like(best.indices))
    boxes, cls_prob, embed = decode_centres(out, getattr(model, "M", 20), valid_depth)
    return {
        "label_map": label_map.to(torch.uint8).cpu().numpy(),
        "presence": out["presence"].float().sigmoid()[0].cpu().numpy(),
        "cls_prob": cls_prob.cpu().numpy(),
        "boxes_vox": boxes.cpu().numpy(),
        "a_embed": out["a_embed"][0].to(torch.float16).cpu().numpy(),
        "u_embed": embed.to(torch.float16).cpu().numpy(),
    }


def save_prediction(path, pred):
    np.savez_compressed(path, **pred)


def load_prediction(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
