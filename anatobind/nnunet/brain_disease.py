"""nnU-Net v2 datasets for the brain multi-disease detectors (spec 2026-09-29 §3).

One dataset per disease, each on the native sequences of its source: glioma (UCSF-PDGM), metastasis (UCSF-BMSR),
infarct (ISLES 2022). Images are symlinks to the read-only originals. Labels are rewritten as binary maps that keep
the original label's header, so SimpleITK reads the same geometry for them as for the images. Folds are by patient.
"""
import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from anatobind.nnunet.brain_lesion import assign_normal_folds, make_splits

FM = Path("/data2/congcong/data/FM_data")
TRAINER = "nnUNetTrainer_250epochs"
CONFIG = "3d_fullres"
DISEASES = {
    "glioma": {"id": 904, "name": "Dataset904_PDGMGlioma", "type": "tumor", "channels": ("T1", "T1c", "T2", "FLAIR"),
               "label_values": (1, 2, 4), "impression": "疑似胶质瘤"},
    "metastasis": {"id": 905, "name": "Dataset905_BMSRMetastasis", "type": "metastasis",
                   "channels": ("T1pre", "T1post", "FLAIR"), "label_values": (1,), "impression": "疑似脑转移瘤"},
    "infarct": {"id": 906, "name": "Dataset906_ISLESInfarct", "type": "infarct", "channels": ("DWI", "ADC"),
                "label_values": (1,), "impression": "疑似缺血性梗死"},
}
PATIENT_PATTERN = {"glioma": r"(UCSF-PDGM-\d+)(_FU\d+d)?", "metastasis": r"(\d+)[A-Z]", "infarct": r"(sub-strokecase\d+)"}


def patient_of(disease, case):
    m = re.fullmatch(PATIENT_PATTERN[disease], case)
    if not m:
        raise ValueError(f"unexpected {disease} case name {case!r}")
    return m.group(1)


def channel_path(disease, case, channel, root=FM):
    """Path of one sequence of one case in the source dataset (any sequence the source holds, not only the model's)."""
    root = Path(root)
    if disease == "glioma":
        return root / "UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5" / f"{case}_nifti" / f"{case}_{channel}.nii.gz"
    if disease == "metastasis":
        return root / "UCSF-BMSR_cbb/UCSF-BMSR/UCSF_BrainMetastases_TRAIN" / case / f"{case}_{channel}.nii.gz"
    if disease == "infarct":
        return root / "ISLES_ltr/ISLES-2022" / case / "ses-0001" / "dwi" / f"{case}_ses-0001_{channel.lower()}.nii.gz"
    raise KeyError(disease)


def label_path(disease, case, root=FM):
    root = Path(root)
    if disease == "glioma":
        return channel_path(disease, case, "tumor_segmentation", root)
    if disease == "metastasis":
        return channel_path(disease, case, "seg", root)
    if disease == "infarct":
        return root / "ISLES_ltr/ISLES-2022/derivatives" / case / "ses-0001" / f"{case}_ses-0001_msk.nii.gz"
    raise KeyError(disease)


def anatomy_path(disease, case, root=FM):
    """The SynthSeg pseudo-label of this case (derived/synthseg, native grid)."""
    stem = {"glioma": f"{case}_T1", "metastasis": f"{case}_T1pre", "infarct": f"{case}_ses-0001_dwi"}[disease]
    folder = {"glioma": "pdgm", "metastasis": "bmsr", "infarct": "isles"}[disease]
    return Path(root) / "derived/synthseg" / folder / "seg_native" / f"{stem}_seg.nii.gz"


def list_cases(disease, root=FM):
    root = Path(root)
    if disease == "glioma":
        base = root / "UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5"
        return sorted(d.name[:-len("_nifti")] for d in base.glob("UCSF-PDGM-*_nifti") if d.is_dir())
    if disease == "metastasis":
        base = root / "UCSF-BMSR_cbb/UCSF-BMSR/UCSF_BrainMetastases_TRAIN"
        return sorted(d.name for d in base.iterdir() if d.is_dir())
    if disease == "infarct":
        return sorted(d.name for d in (root / "ISLES_ltr/ISLES-2022").glob("sub-strokecase*") if d.is_dir())
    raise KeyError(disease)


def sources(disease, root=FM):
    return {c: {"channels": [channel_path(disease, c, k, root) for k in DISEASES[disease]["channels"]],
                "label": label_path(disease, c, root)} for c in list_cases(disease, root)}


def geometry(path):
    """Size, spacing, origin and direction as SimpleITK reads them: what nnU-Net's integrity check compares."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return tuple(r.GetSize()), np.array(r.GetSpacing()), np.array(r.GetOrigin()), np.array(r.GetDirection())


def check_same_grid(case, channel_paths, label):
    for p in (*channel_paths, label):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{case}: missing {p}")
    ref = geometry(label)
    for p in channel_paths:
        g = geometry(p)
        if g[0] != ref[0] or not all(np.allclose(a, b) for a, b in zip(g[1:], ref[1:])):
            raise ValueError(f"{case}: {Path(p).name} is not on the label's grid")


def binary_label(values, allowed, case):
    v = np.rint(np.asarray(values)).astype(np.int16)
    extra = sorted(set(np.unique(v).tolist()) - {0, *allowed})
    if extra:
        raise ValueError(f"{case}: label holds unexpected values {extra}")
    return (v > 0).astype(np.uint8)


def write_binary_label(src, dst, allowed, case):
    """Binary copy of a label that keeps its header; returns (foreground voxels, voxel volume in mm3)."""
    lab = nib.load(str(src))
    b = binary_label(lab.dataobj, allowed, case)
    hdr = lab.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image(b, lab.affine, hdr), str(dst))
    return int(b.sum()), float(np.prod(lab.header.get_zooms()[:3]))


def build_raw(raw_root, disease, srcs, extra=None):
    """<raw_root>/<dataset>/{imagesTr (symlinks), labelsTr, dataset.json, cases.json}; never rebuilds a dataset."""
    spec = DISEASES[disease]
    base = Path(raw_root) / spec["name"]
    if base.exists():
        raise FileExistsError(f"{base} exists; the dataset is never rebuilt in place")
    for case, s in srcs.items():
        patient_of(disease, case)
        check_same_grid(case, s["channels"], s["label"])
    (base / "imagesTr").mkdir(parents=True)
    (base / "labelsTr").mkdir()
    info = {}
    for i, (case, s) in enumerate(srcs.items(), start=1):
        for k, p in enumerate(s["channels"]):
            os.symlink(Path(p).resolve(), base / "imagesTr" / f"{case}_{k:04d}.nii.gz")
        n, vox = write_binary_label(s["label"], base / "labelsTr" / f"{case}.nii.gz", spec["label_values"], case)
        info[case] = {"patient": patient_of(disease, case), "voxel_mm3": vox, "n_label_voxels": n, **(extra or {}).get(case, {})}
        print(f"{i}/{len(srcs)} {case} {n} label voxels", flush=True)
    meta = {"channel_names": {str(k): n for k, n in enumerate(spec["channels"])}, "labels": {"background": 0, spec["type"]: 1},
            "numTraining": len(srcs), "file_ending": ".nii.gz"}
    (base / "dataset.json").write_text(json.dumps(meta, indent=1))
    (base / "cases.json").write_text(json.dumps(info, indent=1))
    return base


def case_folds(disease, cases, k=5, seed=0):
    patients = {c: patient_of(disease, c) for c in cases}
    fold_of = assign_normal_folds(patients.values(), k, seed)
    return {c: fold_of[p] for c, p in patients.items()}


def write_splits(preprocessed_root, disease, case_fold, k=5):
    d = Path(preprocessed_root) / DISEASES[disease]["name"]
    if not d.is_dir():
        raise FileNotFoundError(f"{d} missing: run nnUNetv2_plan_and_preprocess first")
    p = d / "splits_final.json"
    if p.exists():
        raise FileExistsError(f"{p} exists")
    splits = make_splits(case_fold, k)
    patients = [{patient_of(disease, c) for c in s["val"]} for s in splits]
    if sorted(c for s in splits for c in s["val"]) != sorted(case_fold) or sum(len(x) for x in patients) != len(set().union(*patients)):
        raise ValueError("folds do not partition the cases by patient")
    p.write_text(json.dumps(splits, indent=1))
    return p


def fold_dir(results_root, disease, fold, trainer=TRAINER):
    return Path(results_root) / DISEASES[disease]["name"] / f"{trainer}__nnUNetPlans__{CONFIG}" / f"fold_{int(fold)}"
