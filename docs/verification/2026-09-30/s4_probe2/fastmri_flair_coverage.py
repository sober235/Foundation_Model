# Read-only probe for the S4 simulation section: geometry of the fastMRI AXFLAIR stacks and where they sit on the
# brain, measured on the SynthSeg maps of the 447 annotated volumes (pseudo-labels; the lowest slices under-label the
# deep grey matter, so the per-slice class table is an indication, the brain-area profile is not affected by that).
#   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python docs/verification/2026-09-30/s4_probe2/fastmri_flair_coverage.py
import glob
from collections import Counter, defaultdict

import nibabel as nib
import numpy as np

from anatobind.eval.geometry import HOST_CLASSES, LANDMARKS

files = sorted(glob.glob("/data2/congcong/data/FM_data/derived/synthseg/fastmri_brain/seg_native/file_brain_AXFLAIR_*_seg.nii.gz"))
shapes, zooms, nslices = Counter(), Counter(), Counter()
present = defaultdict(lambda: defaultdict(int))
area = defaultdict(list)
bottom_share, top_empty, brain_height = [], [], []
for f in files:
    im = nib.load(f); a = np.asarray(im.dataobj)
    z = tuple(round(float(v), 3) for v in im.header.get_zooms()[:3])
    shapes[a.shape] += 1; zooms[z] += 1; ns = a.shape[2]; nslices[ns] += 1
    areas = [float((a[:, :, k] > 0).sum()) * z[0] * z[1] / 100 for k in range(ns)]          # cm2 per slice
    for k in range(ns):
        area[k].append(areas[k])
        for name, labels in HOST_CLASSES.items():
            if np.isin(a[:, :, k], labels).any():
                present[name][k] += 1
        if np.isin(a[:, :, k], LANDMARKS[:2]).any():
            present["lateral_ventricles"][k] += 1
    if max(areas) > 0:
        bottom_share.append(areas[0] / max(areas))
        with_brain = [k for k, s in enumerate(areas) if s > 5.0]
        top_empty.append(ns - 1 - max(with_brain))
        brain_height.append((max(with_brain) - min(with_brain) + 1) * z[2])
n = len(files)
q = lambda v: tuple(round(float(x), 2) for x in np.percentile(v, [5, 25, 50, 75, 95]))
print(f"AXFLAIR maps {n} | shapes {dict(shapes.most_common())} | zooms {dict(zooms.most_common())} | slices per stack {dict(sorted(nslices.items()))}")
print("share of volumes in which the class appears, per slice index (0 = lowest):")
print("slice           " + " ".join(f"{k:>4d}" for k in range(16)))
for name in list(HOST_CLASSES) + ["lateral_ventricles"]:
    print(f"{name:16s}" + " ".join(f"{present[name][k] / n:4.2f}" for k in range(16)))
print("median brain area (cm2)" + " ".join(f"{np.median(area[k]):4.0f}" for k in range(16)))
print(f"bottom slice area / largest slice area: 5/25/50/75/95th percentiles {q(bottom_share)}")
print(f"empty slices above the brain (area <= 5 cm2): {dict(sorted(Counter(top_empty).items()))}")
print(f"height of the brain within the stack (mm, slices with > 5 cm2): {q(brain_height)}")
