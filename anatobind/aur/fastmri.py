"""AnatoBind-Brain on the fastMRI FLAIR stacks (spec §7, report only): the S4 measure (spec 2026-10-02 A11), so that the
two anatomy models can be read side by side.

Input: the skull-stripped stacks of the S4 evaluation (`derived/brain_anatomy/eval_20261004_0233/stripped/`, the same
input the S4 student saw; 320 x 320 x 16 at 0.69 x 0.69 x 5 mm) and SynthSeg's map on the same grid. A stack is
resampled to the 1 mm RAS grid that covers it (trilinear), normalised as in training, predicted by the whole-volume
inference, and the predicted SynthSeg-valued map is brought back to the stack's own grid by nearest neighbour. The
metrics are S4's functions: the host Dice (S4's 13 compact host classes) on the reliable slices, and the agreement of
the box lookup on the predicted anatomy with the lookup on SynthSeg's for every fastMRI+ lesion box whose slices are
all reliable. The stacks are 5 mm thick: the 1 mm volume interpolates four slices out of five, which the model never
saw in training (its thick-slice rows had no A supervision); this is part of what the row measures. NOT_EVIDENCE."""
import nibabel as nib
import nibabel.processing
import numpy as np

from anatobind.anatomy.labels import HOST_IDS, to_student
from anatobind.aur.crops import normalise, to_zyx
from anatobind.aur.infer import predict_volume
from anatobind.aur.labels import entity_to_synthseg
from anatobind.eval.brain_anatomy import class_dice, host_agreement, pool_agreement, reliable_slices, summarize_dice

TARGET_MM = 1.0


def grid_1mm(img):
    """(shape, affine) of the 1 mm RAS-aligned grid covering the image (nibabel's output grid)."""
    out = nibabel.processing.resample_to_output(img, voxel_sizes=(TARGET_MM,) * 3, order=0, mode="constant", cval=0.0)
    return out.shape, out.affine


def evaluate_stack(model, image_path, seg_path, rows, crop, device, batch_size=1):
    """One stack: {"reliable": [first, last] | None, "dice": {S4 class id: Dice | None}, "agreement": S4's box
    agreement, "pred_native": the predicted SynthSeg-valued map on the stack's grid}."""
    img, seg_img = nib.load(str(image_path)), nib.load(str(seg_path))
    if img.shape != seg_img.shape or not np.allclose(img.affine, seg_img.affine, atol=1e-3):
        raise ValueError(f"{image_path}: the stack {img.shape} and SynthSeg's map {seg_img.shape} differ in grid")
    synthseg = np.asarray(seg_img.dataobj).astype(np.int16)
    spacing = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
    one_mm = nib.as_closest_canonical(nibabel.processing.resample_from_to(img, grid_1mm(img), order=1, mode="constant", cval=0.0))
    image = normalise(to_zyx(np.asarray(one_mm.dataobj).astype(np.float32)))
    pred = predict_volume(model, image, one_mm.affine, crop, device, batch_size=batch_size)
    on_grid = nib.Nifti1Image(entity_to_synthseg(pred["entity"]).transpose(2, 1, 0).astype(np.int16), one_mm.affine)
    pred_native = np.asarray(nibabel.processing.resample_from_to(on_grid, img, order=0, mode="constant", cval=0).dataobj).astype(np.int16)
    student = to_student(pred_native)
    reliable = reliable_slices(synthseg, spacing[0] * spacing[1])
    dice = class_dice(student, to_student(synthseg), slices=reliable) if len(reliable) else {c: None for c in HOST_IDS}
    return {"reliable": [reliable.start, reliable.stop - 1] if len(reliable) else None, "dice": dice,
            "agreement": host_agreement(student, synthseg, spacing, rows, reliable), "pred_native": pred_native}


def summarize(per_stem):
    return {"evidence": "NOT_EVIDENCE: agreement with SynthSeg pseudo-labels on fastMRI FLAIR stacks (the S4 measure; report only)",
            "n_stacks": len(per_stem), "fastmri_dice": summarize_dice([v["dice"] for v in per_stem.values()]),
            "host_agreement": pool_agreement([v["agreement"] for v in per_stem.values()])}
