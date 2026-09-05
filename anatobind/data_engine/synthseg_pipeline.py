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


def chunked(seq, n):
    """Split ``seq`` into consecutive lists of at most ``n`` items."""
    seq = list(seq)
    return [seq[i:i + n] for i in range(0, len(seq), n)]


MANIFEST_FIELDS = ("stem", "h5", "nii", "seg_1mm", "seg_native", "status")


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})
    return path


def pending_stems(stems, native_dir):
    """Stems whose native-grid label file does not exist yet under ``native_dir``."""
    native_dir = Path(native_dir)
    return [s for s in stems if not (native_dir / f"{s}_seg.nii.gz").exists()]


def run_batch(h5_paths, work_dir, synthseg_home, python, threads, runner, robust=True, cpu=True,
              native_dir=None):
    """Stage h5 files, run SynthSeg once over the staged folder, bring labels back to native grids.

    ``runner`` is called with the SynthSeg argv (``subprocess.run`` in production).
    Layout under ``work_dir``: stage/<stem>.nii.gz, seg_1mm/<stem>_synthseg.nii.gz,
    volumes.csv, manifest.csv; native labels go to ``native_dir`` (default
    work_dir/seg_native) as <stem>_seg.nii.gz.
    """
    work_dir = Path(work_dir)
    stage_dir, seg_dir = work_dir / "stage", work_dir / "seg_1mm"
    native_dir = Path(native_dir) if native_dir is not None else work_dir / "seg_native"
    niis = stage_niftis(h5_paths, stage_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    runner(synthseg_command(in_dir=stage_dir, out_dir=seg_dir, resample_dir=None,
                            vol_csv=work_dir / "volumes.csv", threads=threads,
                            synthseg_home=synthseg_home, python=python, robust=robust, cpu=cpu))
    rows = []
    for h5, nii in zip(h5_paths, niis):
        stem = nii.name[: -len(".nii.gz")]
        seg_1mm = seg_dir / f"{stem}_synthseg.nii.gz"
        row = {"stem": stem, "h5": str(h5), "nii": str(nii), "seg_1mm": "", "seg_native": "", "status": "missing"}
        if seg_1mm.exists():
            native = native_labels(seg_1mm, nii, native_dir / f"{stem}_seg.nii.gz")
            row.update(seg_1mm=str(seg_1mm), seg_native=str(native), status="ok")
        rows.append(row)
    write_manifest(rows, work_dir / "manifest.csv")
    return rows


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
