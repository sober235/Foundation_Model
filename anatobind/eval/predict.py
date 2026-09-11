"""Upstream predictions for evaluation and for the stage-III binders (RESEARCH_PLAN v2.2 13.6).

The label map gives each voxel the anatomy query with the highest probability when that
probability exceeds 0.5, else background. Boxes are converted from the normalised
(cz, cy, cx, dz, dy, dx) of the padded input to voxel corners (z0, y0, x0, z1, y1, x1); padding is
at the end of Z, so these are also coordinates in the unpadded volume.
"""
import numpy as np
import torch

from anatobind.model.losses import box_corners

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
    scale = torch.tensor(x.shape[2:], dtype=torch.float32, device=device).repeat(2)
    return {
        "label_map": label_map.to(torch.uint8).cpu().numpy(),
        "presence": out["presence"].float().sigmoid()[0].cpu().numpy(),
        "cls_prob": out["logits"].float().softmax(-1)[0].cpu().numpy(),
        "boxes_vox": (box_corners(out["boxes"].float()[0]) * scale).cpu().numpy(),
        "a_embed": out["a_embed"][0].to(torch.float16).cpu().numpy(),
        "u_embed": out["u_embed"][0].to(torch.float16).cpu().numpy(),
    }


def save_prediction(path, pred):
    np.savez_compressed(path, **pred)


def load_prediction(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
