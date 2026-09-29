# Read-only probe for S4: per slice, how much of the head's cross-section does SynthSeg's outline cover on fastMRI FLAIR,
# and how much of the outline is a hole? The brain fills a similar share of the head from the basal ganglia up to the
# centrum semiovale, so a low share in the lowest slices points at brain that the outline leaves out.
import csv
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage

D = Path("/data2/congcong/data/FM_data/derived")
MAN = {r["stem"]: r for r in csv.DictReader(open(D / "synthseg/fastmri_brain/manifest.csv"))}
segs = sorted((D / "synthseg/fastmri_brain/seg_native").glob("file_brain_AXFLAIR_*_seg.nii.gz"))
share = {k: [] for k in range(16)}
hole = {k: [] for k in range(16)}
low, failed = [], []
for sp in segs:
    stem = sp.name.replace("_seg.nii.gz", "")
    seg = np.asarray(nib.load(str(sp)).dataobj)
    a = np.asarray(nib.load(MAN[stem]["nii"]).dataobj, dtype=np.float32)
    mask = seg > 0
    if mask.sum() * float(np.prod(nib.load(str(sp)).header.get_zooms()[:3])) < 300e3:
        failed.append(stem)
        continue
    thr = float(np.percentile(a[a > 0], 20))
    r = []
    for k in range(mask.shape[2]):
        head = ndimage.binary_fill_holes(ndimage.binary_opening(a[:, :, k] > thr, iterations=2))
        filled = ndimage.binary_fill_holes(mask[:, :, k])
        h, m, f = int(head.sum()), int(mask[:, :, k].sum()), int(filled.sum())
        r.append(f / h if h > 2000 else np.nan)
        if k < 16 and h > 2000:
            share[k].append(f / h)
            hole[k].append((f - m) / f if f else np.nan)
    mid = np.nanmedian(r[2:7])
    if r[0] < 0.8 * mid or r[1] < 0.8 * mid:
        low.append((stem, round(float(r[0]), 2), round(float(r[1]), 2), round(float(mid), 2)))
print(f"{len(segs)} FLAIR volumes; outline under 300 mL (SynthSeg failed or nearly so): {len(failed)} {failed[:8]}")
print("slice | volumes | filled outline / head: 5th percentile, median, 95th | hole share of the filled outline: median, 95th")
for k in range(16):
    v, h = np.array(share[k]), np.array(hole[k])
    if len(v):
        print(f"{k:5d} | {len(v):7d} | {np.nanpercentile(v, 5):.2f}  {np.nanmedian(v):.2f}  {np.nanpercentile(v, 95):.2f} | {np.nanmedian(h):.3f}  {np.nanpercentile(h, 95):.3f}")
print(f"volumes whose slice 0 or 1 covers under 80 % of what slices 2-6 cover: {len(low)} of {len(segs) - len(failed)}")
print("examples (stem, slice 0, slice 1, slices 2-6):", low[:6])
