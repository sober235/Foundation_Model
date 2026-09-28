"""nnDetection task for the brain small-lesion detector's second arm (spec 2026-09-28 brain-nndet §4).

Same 253 cases, images and folds as Dataset903. Each registry lesion becomes one instance of class 0, painted
largest-first (ties: smaller lesion id first) so a lesion sharing voxels with a bigger one keeps its own; the
instance map's non-zero voxels equal Dataset903's binary label. Nothing here imports nnDetection.
"""
import json
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np

TASK_ID = 903
TASK_NAME = f"Task{TASK_ID}_FastMRIBrainSmallLesion"
PLAN_ID = "D3V001_3d"
MODEL_ID = f"RetinaUNetV001_{PLAN_ID}"
CLASS_ID = 0
DATASET_META = {"task": TASK_NAME, "name": "FastMRIBrainSmallLesion", "dim": 3, "test_labels": False,
                "labels": {"0": "small_lesion"}, "modalities": {"0": "FLAIR"}}


def member_mask(members, shape):
    m = np.zeros(shape, bool)
    for b in members:
        m[b["x"]:b["x"] + b["width"], b["y"]:b["y"] + b["height"], b["slice"]] = True
    return m


def paint_instances(members_of, shape):
    """Instance map (uint16, (col, row, slice)) and {lesion_id: instance}; instances are numbered 1.. in painting
    order, largest painted-voxel count first, so smaller lesions overwrite the voxels they share."""
    masks = {lid: member_mask(ms, shape) for lid, ms in members_of.items()}
    order = sorted(masks, key=lambda lid: (-int(masks[lid].sum()), lid))
    inst = np.zeros(shape, np.uint16)
    instance_of = {}
    for k, lid in enumerate(order, start=1):
        inst[masks[lid]] = k
        instance_of[lid] = k
    for lid, k in instance_of.items():
        if not (inst == k).any():
            raise ValueError(f"lesion {lid} lost every voxel to later instances")
    return inst, instance_of


def tight_box(mask):
    idx = np.nonzero(mask)
    return tuple(int(v) for a in idx for v in (a.min(), a.max()))


def changed_boxes(members_of, inst, instance_of):
    """Lesion ids whose tight box in the instance map differs from the tight box of their own members."""
    return sorted(lid for lid, k in instance_of.items()
                  if tight_box(inst == k) != tight_box(member_mask(members_of[lid], inst.shape)))


def check_same_support(inst, binary, case):
    binary = np.asarray(binary)
    if inst.shape != binary.shape or not np.array_equal(inst > 0, binary > 0):
        raise ValueError(f"{case}: instance map support differs from the Dataset903 binary label")


def instances_json(instance_of):
    return {"instances": {str(k): CLASS_ID for k in sorted(instance_of.values())}}


def write_instance_label(inst, image_path, out_path):
    img = nib.load(str(image_path))
    if tuple(img.shape) != tuple(inst.shape):
        raise ValueError(f"label shape {inst.shape} != image shape {img.shape} ({image_path})")
    out = nib.Nifti1Image(np.asarray(inst, np.uint16), img.affine)
    out.set_data_dtype("uint16")
    out.header.set_xyzt_units("mm")
    nib.save(out, str(out_path))


def build_task(det_data, cases, write_case):
    """<det_data>/<TASK_NAME>/raw_splitted/{imagesTr,labelsTr} and dataset.json; never rebuilds an existing task."""
    base = Path(det_data) / TASK_NAME
    if base.exists():
        raise FileExistsError(f"{base} exists; the task is never rebuilt in place")
    (base / "raw_splitted" / "imagesTr").mkdir(parents=True)
    (base / "raw_splitted" / "labelsTr").mkdir()
    for c in cases:
        write_case(c, base / "raw_splitted" / "imagesTr" / f"{c}_0000.nii.gz", base / "raw_splitted" / "labelsTr")
    (base / "dataset.json").write_text(json.dumps(DATASET_META, indent=1))
    return base


def splits_payload(splits_json):
    """nnU-Net splits_final.json content -> nnDetection splits of plain lists (numpy 1 must read the pickle)."""
    return [{"train": [str(c) for c in s["train"]], "val": [str(c) for c in s["val"]]} for s in splits_json]


def write_splits_pkl(preprocessed_dir, splits_json):
    p = Path(preprocessed_dir) / "splits_final.pkl"
    if p.exists():
        raise FileExistsError(f"{p} exists")
    p.write_bytes(pickle.dumps(splits_payload(splits_json), protocol=4))
    return p


def read_splits_pkl(path):
    return pickle.loads(Path(path).read_bytes())
