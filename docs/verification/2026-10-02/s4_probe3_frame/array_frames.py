# Read-only probe for the S4 simulation: in which array direction do left/right and anterior/posterior lie in the
# fastMRI RSS NIfTIs (as SynthSeg labelled them) and in the three training sources after nibabel's canonical (RAS)
# reorientation? The student is trained on arrays, not on world coordinates (nnU-Net reads arrays in index order), so
# the simulated stacks must follow the fastMRI array frame. Left/right of the pseudo-labels follow the headers (M10).
#   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python docs/verification/2026-10-02/s4_probe3_frame/array_frames.py
import glob

import nibabel as nib
import numpy as np

from anatobind.eval.geometry import HOST_CLASSES, LEFT_LABELS, RIGHT_LABELS

FM = "/data2/congcong/data/FM_data/derived/synthseg"


def offsets(arrays):
    """Mean index differences: LEFT minus RIGHT labels along axis 0; cerebellum minus thalamus along axes 1 and 2."""
    rows = []
    for a in arrays:
        L, R = np.argwhere(np.isin(a, LEFT_LABELS)), np.argwhere(np.isin(a, RIGHT_LABELS))
        cb, th = np.argwhere(np.isin(a, HOST_CLASSES["cerebellum"])), np.argwhere(np.isin(a, HOST_CLASSES["thalamus"]))
        if len(L) and len(R) and len(cb) and len(th):
            rows.append((L[:, 0].mean() - R[:, 0].mean(), cb[:, 1].mean() - th[:, 1].mean(), cb[:, 2].mean() - th[:, 2].mean()))
    return len(rows), np.mean(rows, 0)


fm = sorted(glob.glob(f"{FM}/fastmri_brain/seg_native/file_brain_AXFLAIR_*_seg.nii.gz"))[:60]
n, t = offsets([np.asarray(nib.load(f).dataobj) for f in fm])
im = nib.load(fm[0])
print(f"fastMRI RSS NIfTI ({n} maps used, native arrays): axcodes {nib.aff2axcodes(im.affine)}, affine diagonal {np.round(np.diag(im.affine)[:3], 3).tolist()}")
print(f"  LEFT - RIGHT along axis 0: {t[0]:+.1f} voxels | cerebellum - thalamus along axis 1: {t[1]:+.1f} | along axis 2: {t[2]:+.1f}")
for name, pat in (("sibbms", f"{FM}/sibbms/seg_native/MS_sub-00*_seg.nii.gz"), ("pdgm", f"{FM}/pdgm/seg_native/UCSF-PDGM-00*_seg.nii.gz"),
                  ("bmsr", f"{FM}/bmsr/seg_native/1001*_seg.nii.gz")):
    files = sorted(glob.glob(pat))[:12]
    im = nib.load(files[0])
    n, t = offsets([np.asarray(nib.as_closest_canonical(nib.load(f)).dataobj) for f in files])
    print(f"{name} ({n} maps): native axcodes {nib.aff2axcodes(im.affine)}, zooms {tuple(round(float(z), 2) for z in im.header.get_zooms()[:3])}; "
          f"after as_closest_canonical (RAS): LEFT - RIGHT axis 0 {t[0]:+.1f} | cerebellum - thalamus axis 1 {t[1]:+.1f} | axis 2 {t[2]:+.1f}")
print("reading: fastMRI arrays have LEFT at higher axis-0 indices and the cerebellum at lower axis-1 indices than the thalamus;\n"
      "RAS arrays have LEFT at lower axis-0 indices and the same axis-1/axis-2 signs -> flip axis 0 only to enter the fastMRI frame.")
