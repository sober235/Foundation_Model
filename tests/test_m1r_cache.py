import nibabel as nib
import numpy as np

from synth import DEFAULT_BOXES, SHAPE_XYZ, write_synthetic_scan

from anatobind.train.cache import VIEWS, cache_scan, load_array, load_meta


def test_every_view_is_normalised_float16_in_the_model_frame(tmp_path):
    write_synthetic_scan(tmp_path / "exp", "MTR_001", DEFAULT_BOXES)
    assert cache_scan(tmp_path / "exp/MTR_001", tmp_path / "c/MTR_001") is True
    meta = load_meta(tmp_path / "c/MTR_001")
    X, Y, Z = SHAPE_XYZ
    assert meta["shape"] == [Z, Y, X] and meta["views"] == list(VIEWS)
    assert np.allclose(meta["spacing_mm"], (0.8, 0.625, 0.625))
    for v in VIEWS:
        a = load_array(tmp_path / "c/MTR_001", v).astype(np.float32)
        assert a.shape == (Z, Y, X)
        assert abs(a.mean()) < 0.05 and abs(a.std() - 1.0) < 0.05
    assert load_array(tmp_path / "c/MTR_001", "clean").dtype == np.float16


def test_the_segmentation_is_stored_in_the_model_frame(tmp_path):
    d = write_synthetic_scan(tmp_path / "exp", "MTR_001", DEFAULT_BOXES)
    cache_scan(d, tmp_path / "c/MTR_001")
    src = np.asarray(nib.load(str(d / "seg.nii.gz")).dataobj)
    seg = load_array(tmp_path / "c/MTR_001", "seg")
    assert seg.dtype == np.uint8 and np.array_equal(seg, np.transpose(src, (2, 1, 0)))


def test_a_finished_scan_is_skipped(tmp_path):
    d = write_synthetic_scan(tmp_path / "exp", "MTR_001", DEFAULT_BOXES)
    cache_scan(d, tmp_path / "c/MTR_001")
    assert cache_scan(d, tmp_path / "c/MTR_001") is False
