"""Gate for the SKM-TEA export: for every scan the transposed segmentation must beat the identity frame."""
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.skmtea import load_target_magnitude, seg_nifti_to_h5_frame, tissue_contrast  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
RAW, SEG = FM / "SKM-TEA/files_recon_calib-24", FM / "SKM-TEA_ltr/segmentation_masks/dicom-track"
OUT = Path(__file__).resolve().parents[1] / "docs/verification/skmtea_frames.csv"

rows, bad = [], []
for nii in sorted(SEG.glob("MTR_*.nii.gz")):
    scan = nii.name[:-7]
    seg_nii = np.asarray(nib.load(str(nii)).dataobj)
    mag = load_target_magnitude(RAW / f"{scan}.h5", echo=0)
    seg_h5 = seg_nifti_to_h5_frame(seg_nii)
    if seg_h5.shape != mag.shape:
        rows.append({"scan": scan, "shape_ok": False, "contrast_identity": "", "contrast_transposed": ""})
        bad.append(scan)
        continue
    c_t = tissue_contrast(mag, seg_h5)
    c_i = tissue_contrast(mag, seg_nii) if seg_nii.shape == mag.shape else float("nan")
    rows.append({"scan": scan, "shape_ok": True, "contrast_identity": f"{c_i:.3f}", "contrast_transposed": f"{c_t:.3f}"})
    if not (c_t > 1.5 and (np.isnan(c_i) or c_t > c_i)):
        bad.append(scan)
    print(scan, rows[-1], flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} scans, {len(bad)} failing: {bad}")
sys.exit(1 if bad else 0)
