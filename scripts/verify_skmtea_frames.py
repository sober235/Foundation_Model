"""Gate for the SKM-TEA export: the metadata-driven frame (transpose, plus a left-right flip for
"RL"-oriented scans) must place the segmentation on the raw target for every scan.

Metric: min over patellar (1) and femoral (2) cartilage of mean intensity inside the label divided by
mean intensity outside all labels, on echo 1 (cartilage is bright there). Pass = metric > 1.5 and
larger than the same metric for the untransformed NIfTI array.
"""
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.skmtea import load_orientations, load_target_magnitude, seg_nifti_to_h5_frame, tissue_contrast  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
RAW, SEG, ANN = FM / "SKM-TEA/files_recon_calib-24", FM / "SKM-TEA_ltr/segmentation_masks/dicom-track", FM / "SKM-TEA_ltr/annotations/v1.0.0"
OUT = Path(__file__).resolve().parents[1] / "docs/verification/skmtea_frames.csv"


def cartilage_metric(mag, seg):
    outside = mag[seg == 0].mean()
    vals = [mag[seg == l].mean() / outside for l in (1, 2) if (seg == l).any()]
    return float(min(vals)) if vals else float("nan")


orient = load_orientations(ANN)
rows, bad = [], []
for nii in sorted(SEG.glob("MTR_*.nii.gz")):
    scan = nii.name[:-7]
    seg_nii = np.asarray(nib.load(str(nii)).dataobj)
    mag = load_target_magnitude(RAW / f"{scan}.h5", echo=0)
    seg_h5 = seg_nifti_to_h5_frame(seg_nii, orient[scan])
    row = {"scan": scan, "orientation": "".join(orient[scan]), "shape_ok": seg_h5.shape == mag.shape}
    if not row["shape_ok"]:
        rows.append({**row, "cartilage_identity": "", "cartilage_frame": "", "overall_frame": ""})
        bad.append(scan)
        continue
    m_frame = cartilage_metric(mag, seg_h5)
    m_ident = cartilage_metric(mag, seg_nii) if seg_nii.shape == mag.shape else float("nan")
    rows.append({**row, "cartilage_identity": f"{m_ident:.3f}", "cartilage_frame": f"{m_frame:.3f}",
                 "overall_frame": f"{tissue_contrast(mag, seg_h5):.3f}"})
    if not (m_frame > 1.5 and (np.isnan(m_ident) or m_frame > m_ident)):
        bad.append(scan)
    print(scan, rows[-1], flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} scans, {len(bad)} failing: {bad}")
sys.exit(1 if bad else 0)
