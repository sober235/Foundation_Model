import json

import numpy as np
import pytest
from scipy import ndimage

from anatobind.eval.lesion_boxes import BRAIN_MIN_VOXELS, STRUCTURE, decode_boxes
from anatobind.infer.brain import lesion_rows, run
from anatobind.nnunet.brain_lesion import FAMILIES


def test_lesion_rows_give_per_slice_row_col_boxes():
    lab = np.zeros((20, 20, 4), np.uint8)
    lab[3:7, 5:8, 1] = 1
    lab[4:6, 5:9, 2] = 1
    dets = [{"box": (3, 5, 1, 7, 9, 3), "score": 0.8, "family": "small_lesion"}]
    assert lesion_rows(dets, lab) == [{"z0": 1, "z1": 2, "score": 0.8, "boxes": {"1": [[5, 8, 3, 7]], "2": [[5, 9, 4, 6]]}}]


def test_lesion_rows_isolates_components_with_overlapping_boxes():
    """Two separate components whose bounding boxes overlap in-plane; each detection captures only its own component."""
    lab = np.zeros((12, 12, 3), np.uint8)            # (col, row, slice)
    lab[0, 0:10, 0] = 1; lab[0:10, 0, 0] = 1         # an L on slice 0
    lab[0:2, 0, 1] = 1                                # the L continues on slice 1 (cols 0..1, row 0)
    lab[6:9, 6:9, 1] = 1                              # a separate 3x3 blob on slice 1, inside the L's bounding box

    # Verify we have 2 components with the correct structure
    assert ndimage.label(lab == 1, structure=STRUCTURE)[1] == 2

    dets = decode_boxes(lab, None, BRAIN_MIN_VOXELS, FAMILIES)
    rows = lesion_rows(dets, lab)

    # Match rows by z0 since order may vary
    rows_by_z0 = {r["z0"]: r for r in rows}

    # The L: spans slices 0-1
    assert rows_by_z0[0]["z1"] == 1
    assert rows_by_z0[0]["boxes"] == {"0": [[0, 10, 0, 10]], "1": [[0, 1, 0, 2]]}

    # The blob: only on slice 1
    assert rows_by_z0[1]["z1"] == 1
    assert rows_by_z0[1]["boxes"] == {"1": [[6, 9, 6, 9]]}


def test_lesion_rows_keeps_empty_box_detection():
    """Detection box with no component voxels keeps a row with empty boxes (len(rows) == len(dets))."""
    lab = np.zeros((20, 20, 2), np.uint8)
    lab[5:8, 5:8, 0] = 1  # blob on slice 0
    dets = [{"box": (5, 5, 0, 8, 8, 1), "score": 0.9, "family": "small_lesion"},  # hits the blob
            {"box": (10, 10, 0, 15, 15, 1), "score": 0.7, "family": "small_lesion"}]  # empty space
    rows = lesion_rows(dets, lab)
    assert len(rows) == len(dets) == 2
    assert rows[1]["boxes"] == {}


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
