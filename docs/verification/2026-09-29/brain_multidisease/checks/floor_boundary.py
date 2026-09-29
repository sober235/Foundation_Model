# Header-only scan: for every label file, the voxel volume, the 10 mm3 floor in voxels, and how close
# 10 / volume is to an integer (where a float32 header value could tip the floor by one voxel).
import collections
import numpy as np
from anatobind.eval.lesion_components import MIN_MM3, min_voxels_for
from anatobind.nnunet.brain_disease import DISEASES, geometry, label_path, list_cases

for d in DISEASES:
    cases = list_cases(d)
    floors = collections.Counter()
    near = []
    vols = []
    for c in cases:
        _, sp, _, _ = geometry(label_path(d, c))
        v = float(np.prod(sp))
        vols.append(v)
        f = min_voxels_for(v)
        floors[f] += 1
        q = MIN_MM3 / v
        gap = abs(q - round(q))
        if gap < 1e-4:
            near.append((c, tuple(float(x) for x in sp), v, q, f))
    print(f"{d}: {len(cases)} cases; voxel volume min {min(vols):.6f} median {np.median(vols):.6f} max {max(vols):.6f} mm3")
    print(f"  floor (voxels) -> cases: {dict(sorted(floors.items()))}")
    print(f"  cases with 10/volume within 1e-4 of an integer: {len(near)}")
    seen = collections.Counter((n[1], n[4]) for n in near)
    for (sp, f), k in sorted(seen.items(), key=lambda kv: -kv[1])[:12]:
        v = float(np.prod(sp))
        print(f"    spacing {sp} volume {v!r} 10/volume {MIN_MM3 / v!r} floor {f}: {k} cases")
