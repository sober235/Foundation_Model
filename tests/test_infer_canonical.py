import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from nibabel.orientations import apply_orientation, axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform

from anatobind.infer.canonical import TARGET_AXCODES, to_export_frame

SPACING = np.diag([0.5, 0.7, 0.9, 1.0])   # an RAS image with anisotropic voxels, reoriented to the requested codes


def _image(axcodes, shape=(6, 8, 10)):
    base = np.zeros(shape, np.float32)
    base[1, 2, 3] = 7.0
    t = ornt_transform(io_orientation(SPACING), axcodes2ornt(axcodes))
    img = nib.Nifti1Image(np.ascontiguousarray(apply_orientation(base, t)), SPACING @ inv_ornt_aff(t, shape))
    assert nib.aff2axcodes(img.affine) == tuple(axcodes)
    return img


def _marker(img):
    return np.argwhere(np.asanyarray(img.dataobj) == 7.0)[0]


def test_axes_come_out_as_superior_to_inferior_anterior_to_posterior_left_to_right():
    out = to_export_frame(_image(("R", "A", "S")))
    assert nib.aff2axcodes(out.affine) == TARGET_AXCODES == ("I", "P", "R")
    assert out.shape == (10, 8, 6)


def test_the_marker_voxel_keeps_its_world_coordinate():
    src = _image(("L", "P", "S"))
    out = to_export_frame(src)
    assert np.allclose(apply_affine(out.affine, _marker(out)), apply_affine(src.affine, _marker(src)))
    assert np.allclose(apply_affine(src.affine, _marker(src)), apply_affine(SPACING, (1, 2, 3)))


def test_an_image_already_in_the_export_frame_is_returned_unchanged():
    src = _image(("I", "P", "R"))
    out = to_export_frame(src)
    assert np.array_equal(np.asanyarray(out.dataobj), np.asanyarray(src.dataobj)) and np.allclose(out.affine, src.affine)
