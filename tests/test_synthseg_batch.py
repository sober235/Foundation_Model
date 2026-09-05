"""One batch of the pseudo-label pipeline: stage -> SynthSeg (injected runner) -> native labels -> manifest."""
import csv

import h5py
import nibabel as nib
import numpy as np

from anatobind.data_engine.synthseg_pipeline import chunked, run_batch
from tests.test_fastmri_nifti import HEADER_XML


def _fake_h5(path, seed):
    rss = np.random.RandomState(seed).rand(3, 8, 6).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("reconstruction_rss", data=rss)
        f.create_dataset("ismrmrd_header", data=np.bytes_(HEADER_XML.encode()))
    return path


def _fake_synthseg_runner(argv):
    """Stand-in for SynthSeg_predict.py: writes <name>_synthseg.nii.gz at 1 mm for every staged image."""
    in_dir = argv[argv.index("--i") + 1]
    out_dir = argv[argv.index("--o") + 1]
    import pathlib

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    for p in sorted(pathlib.Path(in_dir).glob("*.nii.gz")):
        seg = nib.Nifti1Image(np.full((6, 8, 16), 2, dtype=np.int16), np.diag([-1.0, -1.0, 1.0, 1.0]))
        nib.save(seg, str(pathlib.Path(out_dir) / (p.name[: -len(".nii.gz")] + "_synthseg.nii.gz")))


def test_chunked_splits_into_fixed_size_pieces():
    assert chunked(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]


def test_run_batch_produces_native_labels_and_manifest(tmp_path):
    h5s = [_fake_h5(tmp_path / f"file_brain_{i}.h5", i) for i in range(2)]
    work = tmp_path / "work"
    calls = []

    def runner(argv):
        calls.append(argv)
        _fake_synthseg_runner(argv)

    rows = run_batch(h5s, work, synthseg_home="/s", python="/p", threads=4, runner=runner)

    assert len(calls) == 1  # one SynthSeg invocation per batch (folder mode)
    assert [r["status"] for r in rows] == ["ok", "ok"]
    for r in rows:
        seg = nib.load(r["seg_native"])
        assert seg.shape == (6, 8, 3)  # native fastMRI grid, not the 1 mm grid
        assert set(np.unique(np.asarray(seg.dataobj)).tolist()) <= {0, 2}
    with open(work / "manifest.csv", newline="") as f:
        rows_on_disk = list(csv.DictReader(f))
    assert [r["stem"] for r in rows_on_disk] == ["file_brain_0", "file_brain_1"]
    assert set(rows_on_disk[0].keys()) >= {"stem", "h5", "nii", "seg_1mm", "seg_native", "status"}


def test_run_batch_marks_missing_segmentations(tmp_path):
    h5s = [_fake_h5(tmp_path / "file_brain_0.h5", 0)]
    work = tmp_path / "work"
    rows = run_batch(h5s, work, synthseg_home="/s", python="/p", threads=4, runner=lambda argv: None)
    assert rows[0]["status"] == "missing"
    assert rows[0]["seg_native"] == ""
