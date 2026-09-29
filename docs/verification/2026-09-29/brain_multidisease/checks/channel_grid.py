# Header-only: largest affine deviation between every channel image and the case's SynthSeg map (what an inference
# call would be given), over all cases of the three datasets.
import numpy as np
import nibabel as nib
from anatobind.nnunet.brain_disease import DISEASES, anatomy_path, channel_path, list_cases

for d, spec in DISEASES.items():
    devs, shape_diff = [], 0
    for c in list_cases(d):
        an = nib.load(str(anatomy_path(d, c)))
        for k in spec["channels"]:
            im = nib.load(str(channel_path(d, c, k)))
            if im.shape != an.shape:
                shape_diff += 1
                continue
            devs.append(float(np.abs(im.affine - an.affine).max()))
    devs = np.array(devs)
    print(f"{d}: {len(devs)} channel files on the anatomy's shape, {shape_diff} with another shape; affine deviation "
          f"max {devs.max():.3e}, 99.9th percentile {np.percentile(devs, 99.9):.3e}, files with deviation > 0: {(devs > 0).sum()}, > 1e-6: {(devs > 1e-6).sum()}, > 1e-3: {(devs > 1e-3).sum()}")
