"""Plumbing for running SynthSeg over fastMRI brain volumes as brain pseudo-labels."""
import csv
from pathlib import Path

import nibabel as nib

from anatobind.data_engine.fastmri import rss_h5_to_nifti
from anatobind.data_engine.labels import resample_labels_to_reference

SPLIT_DIRS = ("multicoil_train", "multicoil_val")


def stage_niftis(h5_paths, stage_dir):
    """Convert fastMRI h5 files to ``<stage_dir>/<stem>.nii.gz``; existing outputs are kept."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for h5 in h5_paths:
        h5 = Path(h5)
        target = stage_dir / f"{h5.stem}.nii.gz"
        if not target.exists():
            rss_h5_to_nifti(h5, target)
        out.append(target)
    return out


def native_labels(seg_path, ref_path, out_path):
    """Resample a SynthSeg label map onto the reference volume's grid and save it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = resample_labels_to_reference(seg_path, ref_path)
    img.set_data_dtype("int16")
    nib.save(img, str(out_path))
    return out_path


def annotated_files(annotation_csv):
    """Unique, sorted file stems that have at least one fastMRI+ annotation row."""
    with open(annotation_csv, newline="") as f:
        stems = {row["file"] for row in csv.DictReader(f)}
    return sorted(stems)


def resolve_h5(stem, kspace_root):
    """Locate ``<stem>.h5`` under the fastMRI split folders of ``kspace_root``."""
    kspace_root = Path(kspace_root)
    for split in SPLIT_DIRS:
        p = kspace_root / split / f"{stem}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(f"{stem}.h5 not found under {kspace_root} in {SPLIT_DIRS}")


def synthseg_command(in_dir, out_dir, resample_dir, vol_csv, threads, synthseg_home, python,
                     robust=True, cpu=False):
    """argv for SynthSeg_predict.py in folder mode."""
    argv = [
        str(python),
        str(Path(synthseg_home) / "scripts" / "commands" / "SynthSeg_predict.py"),
        "--i", str(in_dir),
        "--o", str(out_dir),
        "--threads", str(threads),
    ]
    if robust:
        argv.append("--robust")
    if resample_dir is not None:
        argv += ["--resample", str(resample_dir)]
    if vol_csv is not None:
        argv += ["--vol", str(vol_csv)]
    if cpu:
        argv.append("--cpu")
    return argv
