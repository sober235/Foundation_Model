import json

import numpy as np
import pytest

from anatobind.infer.brain import lesion_rows, run


def test_lesion_rows_give_per_slice_row_col_boxes():
    lab = np.zeros((20, 20, 4), np.uint8)
    lab[3:7, 5:8, 1] = 1
    lab[4:6, 5:9, 2] = 1
    dets = [{"box": (3, 5, 1, 7, 9, 3), "score": 0.8, "family": "small_lesion"}]
    assert lesion_rows(dets, lab) == [{"z0": 1, "z1": 2, "score": 0.8, "boxes": {"1": [[5, 8, 3, 7]], "2": [[5, 9, 4, 6]]}}]


def test_lesion_rows_isolates_components_with_overlapping_boxes():
    """Two separate components whose bounding boxes overlap in-plane; each detection captures only its own component."""
    lab = np.zeros((20, 20, 2), np.uint8)
    # L-shape in slice 0
    lab[2:5, 2:4, 0] = 1  # vertical part
    lab[4:6, 2:6, 0] = 1  # horizontal part
    # Separate blob in slice 0, inside the L's bounding box but not connected
    lab[4:6, 4:6, 0] = 1  # This creates a separate component (no 26-connectivity to the L)

    # Bounding box for the L (enclosing both parts)
    dets = [
        {"box": (2, 2, 0, 6, 6, 1), "score": 0.9, "family": "small_lesion"},  # L's bbox
        {"box": (4, 4, 0, 6, 6, 1), "score": 0.8, "family": "small_lesion"},  # blob's bbox
    ]

    rows = lesion_rows(dets, lab)
    # Each row should cover only its own component's voxels
    assert len(rows) == 2
    # First row should have the L's voxels only
    assert rows[0]["score"] == 0.9
    assert "0" in rows[0]["boxes"]
    # Second row should have the blob's voxels only
    assert rows[1]["score"] == 0.8
    assert "0" in rows[1]["boxes"]


def test_run_writes_a_table_and_handles_no_detections(tmp_path, monkeypatch):
    import anatobind.infer.brain as B

    def fake_rss_to_nifti(h5, out, pad_to_slices):
        import nibabel as nib
        nib.save(nib.Nifti1Image(np.zeros((16, 16, 3), np.float32), np.eye(4)), str(out))

    def fake_predict(dataset_id, in_dir, out_dir, folds, gpu, config):
        import nibabel as nib
        out_dir.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((16, 16, 3), np.uint8), np.eye(4)), str(out_dir / "case.nii.gz"))
        p = np.zeros((2, 3, 16, 16), np.float32)
        p[0] = 1.0
        np.savez(out_dir / "case.npz", probabilities=p)

    monkeypatch.setattr(B, "rss_h5_to_nifti", fake_rss_to_nifti)
    rows = run(tmp_path / "x.h5", tmp_path / "out", [0], 0, predict=fake_predict)
    assert rows == [] and json.loads((tmp_path / "out" / "lesions.json").read_text()) == []
    with pytest.raises(FileExistsError):
        run(tmp_path / "x.h5", tmp_path / "out", [0], 0, predict=fake_predict)
