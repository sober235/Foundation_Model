"""S4 inference chain (spec 2026-10-02 §8): a fastMRI FLAIR stack -> brain outline -> skull-stripped stack -> student
anatomy -> a label map in SynthSeg values that BrainBinder reads like a SynthSeg map; optional binding of one box.

Inputs: a fastMRI h5 (RSS reconstruction, the frame of S2's `rss_h5_to_nifti`) or a NIfTI stack already in that frame.
The output directory must not exist, and a run that fails half-way leaves its partial output there (nothing is ever
deleted): the error says so and the rerun takes a new directory. The student sees the stack with everything outside
the outline set to zero; its labels are not clipped to the outline afterwards. Everything downstream of the two nnU-Net
models is a pseudo-label: NOT_EVIDENCE."""
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.anatomy.labels import NAMES, to_synthseg
from anatobind.anatomy.outline import postprocess
from anatobind.bind.brain_lookup import BrainBinder
from anatobind.data_engine.fastmri import rss_h5_to_nifti
from anatobind.eval.brain_anatomy import reliable_slices
from anatobind.eval.lesion_boxes import load_label_map
from anatobind.infer.knee import nnunet_env

OUTLINE = {"id": 908, "config": "2d"}
STUDENT = {"id": 907, "config": "3d_fullres"}
TRAINER = "nnUNetTrainer_250epochs"
FOLDS = [0]


def run_nnunet(dataset_id, config, in_dir, out_dir, folds, gpu):
    cmd = ["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir), "-d", str(dataset_id), "-c", config, "-tr", TRAINER,
           "-f", *[str(f) for f in folds], "-npp", "2", "-nps", "2", "--disable_progress_bar"]
    subprocess.run(["nice", "-n", "19", *cmd], check=True, env=nnunet_env(gpu))


def stage_input(h5, nifti, out):
    """The RSS stack as float32 in <out>/input_outline/case_0000.nii.gz; returns the loaded image."""
    d = out / "input_outline"
    d.mkdir(parents=True)
    if h5 is not None:
        rss_h5_to_nifti(h5, d / "case_0000.nii.gz", pad_to_slices=0)
    else:
        src = nib.load(str(nifti))
        nib.save(nib.Nifti1Image(np.asarray(src.dataobj).astype(np.float32), src.affine), str(d / "case_0000.nii.gz"))
    return nib.load(str(d / "case_0000.nii.gz"))


def _stage(name, out, fn):
    """Run one stage; a failure is re-raised with the stage's name and the fact that the partial output stays."""
    try:
        return fn()
    except Exception as e:
        raise RuntimeError(f"{name} failed; the partial output stays in {out} (nothing is deleted here): "
                           f"rerun into a new output directory") from e


def check_box(box, shape):
    """(x0, y0, z0, x1, y1, z1) as ints; refuses a box that is empty or leaves the grid."""
    x0, y0, z0, x1, y1, z1 = [int(v) for v in box]
    if not (0 <= x0 < x1 <= shape[0] and 0 <= y0 < y1 <= shape[1] and 0 <= z0 < z1 <= shape[2]):
        raise ValueError(f"box {[x0, y0, z0, x1, y1, z1]} is empty or outside the grid {tuple(shape)}")
    return x0, y0, z0, x1, y1, z1


def run(out_dir, gpu, h5=None, nifti=None, box=None, predict=run_nnunet):
    """box: (x0, y0, z0, x1, y1, z1), half-open, on the stack's (col, row, slice) grid; bound with BrainBinder."""
    if (h5 is None) == (nifti is None):
        raise ValueError("give exactly one of h5 and nifti")
    source = Path(h5 if h5 is not None else nifti)
    if not source.is_file():
        raise FileNotFoundError(f"{source} is missing")
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; nothing is deleted or overwritten here, use a new output directory")
    img = _stage("staging the input", out, lambda: stage_input(h5, nifti, out))
    data = np.asarray(img.dataobj).astype(np.float32)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if box is not None:
        box = _stage("checking the box", out, lambda: check_box(box, data.shape))
    _stage("the outline prediction", out,
           lambda: predict(OUTLINE["id"], OUTLINE["config"], out / "input_outline", out / "pred_outline", FOLDS, gpu))
    mask = postprocess(load_label_map(out / "pred_outline" / "case.nii.gz"))
    if mask.shape != data.shape:
        raise ValueError(f"outline {mask.shape} and stack {data.shape} differ")
    nib.save(nib.Nifti1Image(mask, img.affine), str(out / "brain_mask.nii.gz"))
    (out / "input_student").mkdir()
    nib.save(nib.Nifti1Image(data * mask, img.affine), str(out / "input_student" / "case_0000.nii.gz"))
    _stage("the student prediction", out,
           lambda: predict(STUDENT["id"], STUDENT["config"], out / "input_student", out / "pred_student", FOLDS, gpu))
    student = load_label_map(out / "pred_student" / "case.nii.gz")
    if student.shape != data.shape:
        raise ValueError(f"student {student.shape} and stack {data.shape} differ")
    anatomy = to_synthseg(student)
    nib.save(nib.Nifti1Image(anatomy, img.affine), str(out / "anatomy.nii.gz"))
    voxel_ml = float(np.prod(zooms)) / 1000.0
    reliable = reliable_slices(anatomy, zooms[0] * zooms[1])
    record = {"input": str(h5 if h5 is not None else nifti), "shape": list(data.shape), "spacing_mm": list(zooms),
              "brain_ml": round(float(mask.sum()) * voxel_ml, 1),
              "class_volumes_ml": {NAMES[c]: round(float((student == c).sum()) * voxel_ml, 3) for c in range(1, 15)},
              "reliable_slices": [reliable.start, reliable.stop - 1] if len(reliable) else None,
              "anatomy": str(out / "anatomy.nii.gz"), "brain_mask": str(out / "brain_mask.nii.gz"),
              "anatomy_source": "S4 student on FLAIR, trained on SynthSeg pseudo-labels (NOT_EVIDENCE)"}
    if box is not None:
        x0, y0, z0, x1, y1, z1 = box
        sl = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
        record["box"] = [x0, y0, z0, x1, y1, z1]
        record["binding"] = BrainBinder(anatomy, zooms).bind(sl, np.ones((x1 - x0, y1 - y0, z1 - z0), bool))
    (out / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    return record
