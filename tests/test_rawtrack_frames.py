import csv
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.data_engine.seg_frames import store_in_frame
from anatobind.data_engine.skmtea import (
    RAW_TRACK_FRAMES, load_rawtrack_seg_h5_frame, rawtrack_frame, rawtrack_spacing_h5_frame,
)

CSV_PATH = Path(__file__).resolve().parents[1] / "docs/verification/2026-09-11/rawtrack_frame.csv"


def test_the_frames_agree_with_the_measurement_on_all_scans():
    rows = list(csv.DictReader(open(CSV_PATH, newline="")))
    assert len(rows) == 155
    for o in ("LR", "RL"):
        best = Counter(r["best"] for r in rows if r["orientation"].endswith(o)).most_common(1)[0][0]
        assert RAW_TRACK_FRAMES[o] == best, o


def test_the_loader_applies_the_frame_and_permutes_the_spacing(tmp_path):
    h5 = np.zeros((4, 5, 3), np.uint8)
    h5[1, 2, 0] = 5
    for o in (("SI", "AP", "LR"), ("SI", "AP", "RL")):
        frame = rawtrack_frame(o)
        stored = np.ascontiguousarray(store_in_frame(h5, frame))
        path = tmp_path / f"{o[2]}.nii.gz"
        nib.save(nib.Nifti1Image(stored, np.diag([0.3, 0.4, 0.8, 1.0])), str(path))
        assert np.array_equal(load_rawtrack_seg_h5_frame(path, o), h5)
        expected = (0.4, 0.3, 0.8) if frame[0] else (0.3, 0.4, 0.8)
        assert np.allclose(rawtrack_spacing_h5_frame(path, o), expected)
