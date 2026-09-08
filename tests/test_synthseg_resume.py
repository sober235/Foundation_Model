"""Batch runner behaviours needed for a multi-hour run: shared output folder and resume."""
import nibabel as nib
import numpy as np

from anatobind.data_engine.synthseg_pipeline import pending_stems, run_batch
from tests.test_synthseg_batch import _fake_h5, _fake_synthseg_runner


def test_run_batch_can_write_native_labels_into_a_shared_folder(tmp_path):
    h5s = [_fake_h5(tmp_path / "file_brain_0.h5", 0)]
    shared = tmp_path / "all_native"
    rows = run_batch(h5s, tmp_path / "chunk0", synthseg_home="/s", python="/p", threads=2,
                     runner=_fake_synthseg_runner, native_dir=shared)
    assert rows[0]["seg_native"] == str(shared / "file_brain_0_seg.nii.gz")
    assert (shared / "file_brain_0_seg.nii.gz").exists()


def test_pending_stems_drops_those_with_existing_native_labels(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), np.eye(4)), str(native / "done_seg.nii.gz"))
    assert pending_stems(["done", "todo_a", "todo_b"], native) == ["todo_a", "todo_b"]
