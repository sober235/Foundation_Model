# Header-only: for every case, does the SynthSeg map (anatomy_path) exist and share the grid of the label?
import collections
import numpy as np
import nibabel as nib
from anatobind.nnunet.brain_disease import DISEASES, anatomy_path, label_path, list_cases

for d in DISEASES:
    cases = list_cases(d)
    missing, shape_diff, affine_diff = [], [], []
    worst = 0.0
    for c in cases:
        a = anatomy_path(d, c)
        if not a.exists():
            missing.append(c)
            continue
        la, an = nib.load(str(label_path(d, c))), nib.load(str(a))
        if la.shape != an.shape:
            shape_diff.append((c, la.shape, an.shape))
            continue
        dev = float(np.abs(la.affine - an.affine).max())
        worst = max(worst, dev)
        if dev > 1e-3:
            affine_diff.append((c, dev))
    print(f"{d}: {len(cases)} cases; anatomy missing {len(missing)}; shape differs {len(shape_diff)}; "
          f"affine differs by > 1e-3 in {len(affine_diff)} (largest deviation {worst:.2e})")
    for x in (missing[:5] + shape_diff[:5] + affine_diff[:5]):
        print("   ", x)
