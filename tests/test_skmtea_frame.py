import h5py
import nibabel as nib
import numpy as np

from anatobind.data_engine.skmtea import (
    load_seg_h5_frame, load_target_magnitude, seg_nifti_to_h5_frame, tissue_contrast,
)


def test_seg_frame_swaps_the_first_two_axes():
    a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    b = seg_nifti_to_h5_frame(a)
    assert b.shape == (3, 2, 4) and b[1, 0, 2] == a[0, 1, 2]


def test_tissue_contrast_is_mean_inside_over_mean_outside():
    mag = np.ones((4, 4, 2), dtype=np.float32)
    seg = np.zeros((4, 4, 2), dtype=np.uint8)
    seg[:2] = 1
    mag[:2] = 3.0
    assert tissue_contrast(mag, seg) == 3.0


def test_loaders_return_h5_frame_and_echo_magnitude(tmp_path):
    seg = np.zeros((3, 2, 4), dtype=np.uint8)
    seg[0, 1, 2] = 5
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(tmp_path / "MTR_x.nii.gz"))
    tgt = np.zeros((2, 3, 4, 2, 1), dtype=np.complex64)
    tgt[1, 0, 2, 1, 0] = 3 + 4j
    with h5py.File(tmp_path / "MTR_x.h5", "w") as f:
        f.create_dataset("target", data=tgt)
    s = load_seg_h5_frame(tmp_path / "MTR_x.nii.gz")
    assert s.shape == (2, 3, 4) and s[1, 0, 2] == 5
    m = load_target_magnitude(tmp_path / "MTR_x.h5", echo=1)
    assert m.shape == (2, 3, 4) and m.dtype == np.float32 and m[1, 0, 2] == 5.0
