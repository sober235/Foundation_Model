# Read-only: headers and label content of the SynthSeg maps of the SibBMS T1w volumes (S4 teacher labels).
#   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python docs/verification/2026-09-30/s4_sibbms_synthseg/sibbms_seg_check.py
import csv
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.eval.geometry import HOST_CLASSES, LANDMARKS

R = Path("/data2/congcong/data/FM_data/derived/synthseg/sibbms")
rows = list(csv.DictReader(open(R / "manifest.csv")))
status = Counter(r["status"] for r in rows)
print(f"manifest rows {len(rows)}: {dict(status)}")
segs = sorted((R / "seg_native").glob("*_seg.nii.gz"))
print(f"seg_native files {len(segs)}: MS {sum(p.name.startswith('MS_') for p in segs)}, Norm {sum(p.name.startswith('Norm_') for p in segs)}")
shapes, zooms, codes, nlab = Counter(), Counter(), Counter(), Counter()
brain, host, vent, missing_host = [], [], [], []
host_labels = [l for labels in HOST_CLASSES.values() for l in labels]
for p in segs:
    im = nib.load(str(p)); a = np.asarray(im.dataobj)
    shapes[a.shape] += 1; zooms[tuple(round(float(z), 3) for z in im.header.get_zooms()[:3])] += 1
    codes[nib.aff2axcodes(im.affine)] += 1
    u = set(np.unique(a).tolist()); nlab[len(u)] += 1
    brain.append(float((a > 0).sum()) / 1000); host.append(float(np.isin(a, host_labels).sum()) / 1000)
    vent.append(float(np.isin(a, LANDMARKS).sum()) / 1000)
    lacking = [name for name, labels in HOST_CLASSES.items() if not u & set(labels)]
    if lacking:
        missing_host.append((p.name, lacking))
q = lambda v: tuple(round(float(x), 1) for x in np.percentile(v, [0, 5, 50, 95, 100]))
print(f"shapes {dict(shapes)}; zooms {dict(zooms)}; axcodes {dict(codes)}; distinct labels per map {dict(sorted(nlab.items()))}")
print(f"volume of all labels > 0 (mL) min/5th/median/95th/max {q(brain)}")
print(f"volume of the 7 host classes (mL) {q(host)}; ventricles + CSF landmarks (mL) {q(vent)}")
print(f"maps lacking a host class: {len(missing_host)} {missing_host[:5]}")
