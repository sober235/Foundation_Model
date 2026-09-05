"""Resample a label map (e.g. SynthSeg output at 1 mm) onto a reference grid.

Nearest-neighbour only: labels must stay integers and no new label values may
appear. Geometry is taken from the affines, not from array shapes.
"""
import nibabel as nib
import numpy as np

from anatobind.data_engine.labels import resample_labels_to_reference


def _label_image_1mm():
    # 20 x 20 x 20 mm cube at 1 mm; label 1 where world x > 0, else 2; 0 in one corner
    data = np.full((20, 20, 20), 2, dtype=np.int16)
    data[10:, :, :] = 1  # voxel x index >= 10  <->  world x >= 0.5 (centred affine below)
    data[:3, :3, :3] = 0
    aff = np.eye(4)
    aff[:3, 3] = [-9.5, -9.5, -9.5]  # voxel centres from -9.5 .. 9.5 mm
    return nib.Nifti1Image(data, aff)


def _reference_2mm_lps():
    # same physical cube at 2 mm, but stored with flipped x and y axes (L, P, S)
    aff = np.diag([-2.0, -2.0, 2.0, 1.0])
    aff[:3, 3] = [9.0, 9.0, -9.0]  # voxel 0 sits at +9 (x), +9 (y), -9 (z)
    return nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.float32), aff)


def test_resampled_labels_follow_reference_grid_and_affine():
    out = resample_labels_to_reference(_label_image_1mm(), _reference_2mm_lps())
    assert out.shape == (10, 10, 10)
    assert np.allclose(out.affine, _reference_2mm_lps().affine)


def test_resampled_labels_are_integers_from_the_original_label_set():
    out = resample_labels_to_reference(_label_image_1mm(), _reference_2mm_lps())
    data = np.asarray(out.dataobj)
    assert np.issubdtype(data.dtype, np.integer)
    assert set(np.unique(data).tolist()) <= {0, 1, 2}


def test_resampled_labels_land_on_correct_side_of_world_x():
    out = resample_labels_to_reference(_label_image_1mm(), _reference_2mm_lps())
    data = np.asarray(out.dataobj)
    ref = _reference_2mm_lps()
    # world x of reference voxel i along axis 0
    xs = np.array([ref.affine @ np.array([i, 5, 5, 1.0]) for i in range(10)])[:, 0]
    for i, x in enumerate(xs):
        expected = 1 if x > 0 else 2
        assert data[i, 5, 5] == expected, (i, x)
