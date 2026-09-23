"""Bring an external NIfTI onto the SKM-TEA export frame the knee models were trained on (spec §3.4).

The export grid runs superior->inferior, anterior->posterior, left->right along (X, Y, Z) at
0.625 x 0.625 x 0.8 mm (docs/data_engine_skmtea.md), and its NIfTIs carry a diagonal affine with no
real orientation. `--frame h5` therefore takes such a file as is; `--frame world` trusts the affine
and reorders the axes to axis codes ('I', 'P', 'R'). nnU-Net resamples spacing itself from the
header, so only the axis order and the pixdims matter here.
"""
import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform

TARGET_AXCODES = ("I", "P", "R")


def to_export_frame(img):
    transform = ornt_transform(io_orientation(img.affine), axcodes2ornt(TARGET_AXCODES))
    arr = apply_orientation(np.asanyarray(img.dataobj), transform)
    affine = img.affine @ inv_ornt_aff(transform, img.shape)
    out = nib.Nifti1Image(np.ascontiguousarray(arr), affine)
    out.header.set_xyzt_units("mm")
    return out
