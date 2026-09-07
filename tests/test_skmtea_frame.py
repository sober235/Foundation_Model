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


def test_rl_oriented_scans_are_also_flipped_along_the_last_axis():
    a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    lr = seg_nifti_to_h5_frame(a, orientation=("SI", "AP", "LR"))
    rl = seg_nifti_to_h5_frame(a, orientation=("SI", "AP", "RL"))
    assert np.array_equal(lr, np.transpose(a, (1, 0, 2)))
    assert np.array_equal(rl, np.flip(np.transpose(a, (1, 0, 2)), axis=2))


def test_load_seg_h5_frame_applies_the_orientation(tmp_path):
    seg = np.zeros((3, 2, 4), dtype=np.uint8)
    seg[0, 1, 0] = 5
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(tmp_path / "MTR_y.nii.gz"))
    s = load_seg_h5_frame(tmp_path / "MTR_y.nii.gz", orientation=("SI", "AP", "RL"))
    assert s.shape == (2, 3, 4) and s[1, 0, 3] == 5
