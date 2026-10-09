# Orientation, spacing and shape of every case's anatomy map per source (final review I1, 2026-10-09). Reads the sample
# table and the NIfTI headers only (one map per case); prints the table.
#
#   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python \
#       docs/verification/2026-10-08/anatobind_brain_aur/checks/spacing_table.py /data2/congcong/data/FM_data/derived/aur/samples.json
import json
import sys
from collections import Counter

import nibabel as nib
import numpy as np

rows = json.loads(open(sys.argv[1]).read())
seen = set()
by_source = {}
for r in rows:
    if r["case"] in seen:
        continue
    seen.add(r["case"])
    img = nib.load(r["anatomy"])
    sp = tuple(round(float(z), 3) for z in img.header.get_zooms()[:3])
    by_source.setdefault(r["source"], []).append(("".join(nib.aff2axcodes(img.affine)), sp, tuple(int(s) for s in img.shape[:3])))
for source, items in by_source.items():
    print(f"== {source}: {len(items)} cases")
    print("   orientation:", dict(Counter(o for o, _, _ in items)))
    sp = np.array([s for _, s, _ in items])
    print("   spacing min / median / max per axis (x, y, z):", sp.min(0).tolist(), np.median(sp, 0).tolist(), sp.max(0).tolist())
    print("   most common (spacing, shape):", Counter((s, sh) for _, s, sh in items).most_common(6))
    thick = sum(1 for _, s, _ in items if max(s) >= 3.0)
    print(f"   cases with a voxel axis >= 3 mm: {thick}")
