"""Staging fastMRI volumes for SynthSeg and bringing labels back to the native grid."""
import h5py
import nibabel as nib
import numpy as np

from anatobind.data_engine.synthseg_pipeline import native_labels, stage_niftis
from tests.test_fastmri_nifti import HEADER_XML


def _fake_h5(path, seed):
    rss = np.random.RandomState(seed).rand(3, 8, 6).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("reconstruction_rss", data=rss)
        f.create_dataset("ismrmrd_header", data=np.bytes_(HEADER_XML.encode()))
    return path


def test_stage_niftis_writes_one_nifti_per_h5_named_by_stem(tmp_path):
    h5s = [_fake_h5(tmp_path / f"file_brain_{i}.h5", i) for i in range(2)]
    stage = tmp_path / "stage"
    out = stage_niftis(h5s, stage)
    assert [p.name for p in out] == ["file_brain_0.nii.gz", "file_brain_1.nii.gz"]
    assert all(p.exists() for p in out)


def test_stage_niftis_skips_existing_outputs(tmp_path):
    h5s = [_fake_h5(tmp_path / "file_brain_0.h5", 0)]
    stage = tmp_path / "stage"
    first = stage_niftis(h5s, stage)[0]
    mtime = first.stat().st_mtime_ns
    again = stage_niftis(h5s, stage)[0]
    assert again == first and again.stat().st_mtime_ns == mtime


def test_native_labels_resamples_seg_onto_reference_grid(tmp_path):
    ref = nib.Nifti1Image(np.zeros((6, 8, 3), dtype=np.float32), np.diag([-0.5, -0.6875, 5.0, 1.0]))
    ref_p = tmp_path / "ref.nii.gz"
    nib.save(ref, str(ref_p))
    seg = nib.Nifti1Image(np.full((4, 6, 16), 3, dtype=np.int16), np.diag([-1.0, -1.0, 1.0, 1.0]))
    seg_p = tmp_path / "seg_1mm.nii.gz"
    nib.save(seg, str(seg_p))

    out_p = native_labels(seg_p, ref_p, tmp_path / "native" / "seg_native.nii.gz")

    out = nib.load(str(out_p))
    assert out.shape == (6, 8, 3)
    assert np.allclose(out.affine, ref.affine)
    assert out.get_data_dtype() == np.int16
    assert set(np.unique(np.asarray(out.dataobj)).tolist()) <= {0, 3}
