# Does any SynthSeg map of the three datasets have no host voxel at all (the only way a lesion gets host None)?
import numpy as np
import nibabel as nib
from anatobind.eval.geometry import host_class_map
from anatobind.nnunet.brain_disease import DISEASES, anatomy_path, list_cases

for d in DISEASES:
    cases = list_cases(d)
    none, small = [], []
    fr = []
    for c in cases:
        seg = np.asarray(nib.load(str(anatomy_path(d, c))).dataobj)
        n = int((host_class_map(seg) > 0).sum())
        fr.append(n)
        if n == 0:
            none.append(c)
        elif n < 1000:
            small.append((c, n))
    fr = np.array(fr)
    print(f"{d}: {len(cases)} maps; without any host voxel {len(none)}; with fewer than 1000 host voxels {len(small)}; "
          f"host voxels min {fr.min()} median {int(np.median(fr))}", flush=True)
    for x in none[:5] + small[:5]:
        print("   ", x, flush=True)
