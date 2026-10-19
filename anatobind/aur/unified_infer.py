"""Mode-explicit, original-grid inference for validated plane/volume inputs.

The model is loaded once by the caller. Source references stay private; no original DICOM is rewritten.
Area is the only physical size reported for a standalone plane. All outputs remain NOT_EVIDENCE.
"""
from dataclasses import dataclass, field

import numpy as np
import nibabel as nib
import torch
from scipy import ndimage

from anatobind.aur.crops import normalise, to_zyx
from anatobind.aur.infer import bind_instances, instances_from_probability, predict_volume, to_original


def model_from_unified_export(path, device="cpu"):
    """Strictly load one version-one technical-pilot export for either input mode.

    Only tensor/builtin checkpoint data are accepted. This entry does not mark a checkpoint as clinically validated
    or bypass the still-unimplemented formal joint G1 evaluation.
    """
    from anatobind.aur.model import AnatoBindBrain

    exported = torch.load(path, map_location="cpu", weights_only=True)
    if (exported.get("format_version") != 1 or exported.get("stage") not in ("II", "III")
            or exported.get("model_config", {}).get("dual_input") is not True
            or not isinstance(exported.get("data_contract"), dict)):
        raise ValueError("expected a complete unified AUR technical-pilot export")
    model = AnatoBindBrain(**exported["model_config"])
    model.load_state_dict(exported["model_state_dict"], strict=True)
    metadata = {"evidence": "NOT_EVIDENCE", "pilot_only": True, "stage": exported["stage"],
                "lineage": exported.get("lineage"), "data_contract": exported["data_contract"]}
    return model.to(device).eval(), metadata


@dataclass
class UnifiedPrediction:
    input_mode: str
    anatomy: np.ndarray = field(repr=False)
    instances: np.ndarray = field(repr=False)
    affine_ras: np.ndarray = field(repr=False)
    lesions: list = field(repr=False)
    source_refs: tuple = field(repr=False)
    frame_map: tuple = field(repr=False)
    evidence: str = "NOT_EVIDENCE"


class _ModeView:
    """Dispatch the existing sliding-window implementation without cloning any parameters."""
    def __init__(self, model, mode):
        self.model, self.mode = model, mode

    def __call__(self, *args):
        return self.model(*args, input_mode=self.mode)

    def __getattr__(self, name):
        return getattr(self.model, name)


def plane_instances(probability, affine, threshold=.3, score_threshold=None, min_area_mm2=1.):
    """8-connected planar instances; the affine's third axis never defines an observed volume."""
    probability, affine = np.asarray(probability), np.asarray(affine)
    if probability.ndim != 3 or probability.shape[0] != 1 or not np.isfinite(probability).all():
        raise ValueError("plane probabilities must be finite with shape (1, H, W)")
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("invalid plane affine")
    area = float(np.linalg.norm(np.cross(affine[:3, 0], affine[:3, 1])))
    if area <= 0 or not np.isfinite(min_area_mm2) or min_area_mm2 <= 0:
        raise ValueError("plane area and minimum area must be positive")
    if not 0 <= threshold <= 1 or (score_threshold is not None and not 0 <= score_threshold <= 1):
        raise ValueError("probability thresholds must be in [0, 1]")
    components, _ = ndimage.label(probability[0] >= threshold, structure=np.ones((3, 3), bool))
    result = np.zeros_like(probability, dtype=np.int32)
    rows = []
    for old, slices in enumerate(ndimage.find_objects(components), start=1):
        if slices is None:
            continue
        mask = components == old
        count, score = int(mask.sum()), float(probability[0][mask].mean())
        if count * area < min_area_mm2 * (1 - 1e-6) or (score_threshold is not None and score < score_threshold):
            continue
        index = len(rows) + 1
        result[0][mask] = index
        rows.append({"instance": index, "box": [0, slices[0].start, slices[1].start,
                                                   1, slices[0].stop, slices[1].stop],
                     "n_voxels": count, "area_mm2": count * area, "score": score})
    return result, rows


@torch.no_grad()
def infer_input(model, source, crop, device="cpu", *, entity_threshold=.5, presence_threshold=.5,
                lesion_threshold=.3, score_threshold=None, min_area_mm2=1., overlap=.5):
    """Infer on a validated DICOM input or an equivalent object carrying its original grid and provenance.

    This is a technical compatibility entry point, not a newly calibrated clinical operating point. Standalone
    planes are normalized from the observed plane only; volumes retain legacy whole-volume normalization.
    """
    mode, image = source.input_mode, np.asarray(source.image_dhw)
    affine, crop = np.asarray(source.affine_ras, dtype=np.float64), tuple(crop)
    if mode not in ("2d", "3d") or image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError("invalid observed input mode or image")
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) <= 0:
        raise ValueError("invalid input geometry")
    if mode == "2d" and (image.shape[0] != 1 or len(crop) != 3 or crop[0] != 1):
        raise ValueError("2d inference requires a true depth-one image and crop")
    patch = model.backbone.patch_for(mode)
    if len(crop) != 3 or any(c <= 0 or c % p for c, p in zip(crop, patch)):
        raise ValueError("inference crop must be positive and divisible by the selected patch")
    if not 0 <= entity_threshold <= 1 or not 0 <= presence_threshold <= 1 or not 0 <= overlap < 1:
        raise ValueError("invalid inference threshold or overlap")
    original_affine = affine.copy()
    if mode == "3d":
        # Match the existing NIfTI entry point: flips/permutations only, followed by inverse output mapping.
        original = nib.Nifti1Image(np.asarray(image, dtype=np.float32).transpose(2, 1, 0), affine)
        canonical = nib.as_closest_canonical(original)
        image, affine = to_zyx(np.asarray(canonical.dataobj)), canonical.affine
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if next(model.parameters()).device != device:
        raise ValueError("model and requested inference device differ")
    view, image = _ModeView(model, mode), normalise(image)
    prediction = predict_volume(view, image, affine, crop, device, overlap=overlap,
                                presence_threshold=presence_threshold, entity_threshold=entity_threshold)
    if mode == "2d":
        instances, rows = plane_instances(prediction["lesion_prob"], affine, lesion_threshold,
                                          score_threshold, min_area_mm2)
    else:
        if not 0 <= lesion_threshold <= 1 or (score_threshold is not None and not 0 <= score_threshold <= 1):
            raise ValueError("probability thresholds must be in [0, 1]")
        instances, rows = instances_from_probability(prediction["lesion_prob"],
                                                     abs(float(np.linalg.det(affine[:3, :3]))),
                                                     lesion_threshold, score_threshold)
    bindings = {row["instance"]: row for row in bind_instances(view, image, affine, instances, crop, device)}
    anatomy = prediction["entity"]
    if mode == "3d":
        anatomy = to_zyx(to_original(anatomy.transpose(2, 1, 0), original_affine))
        instances = to_zyx(to_original(instances.transpose(2, 1, 0), original_affine))
    lesions = []
    for row in rows:
        bound = bindings[row["instance"]]
        voxels = np.argwhere(instances == row["instance"])
        row = {**row, "box": [int(v) for v in np.concatenate((voxels.min(0), voxels.max(0) + 1))]}
        lesions.append({**row, "host": bound["host"], "host_probs": bound["host_probs"].tolist(),
                        "query": bound["query"], "query_iou": bound["query_iou"],
                        "in_crop_share": bound["in_crop_share"], "evidence": "NOT_EVIDENCE"})
    return UnifiedPrediction(mode, anatomy, instances, original_affine, lesions,
                             tuple(source.source_refs), tuple(source.frame_map))
