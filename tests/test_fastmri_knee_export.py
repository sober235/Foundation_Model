import json

import h5py
import numpy as np

from anatobind.data_engine.fastmri_knee import VIEWS, export_volume, make_folds


def _write_h5(path, slices=3, coils=2, ny=64, nx=40, patient="p1", seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(slices, coils, ny, nx)) + 1j * rng.normal(size=(slices, coils, ny, nx))
    k = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))
    with h5py.File(path, "w") as h:
        h.create_dataset("kspace", data=k.astype(np.complex64))
        h.attrs["patient_id"] = patient
        h.attrs["acquisition"] = "CORPD_FBK"
    return path


def test_every_volume_of_a_patient_lands_in_one_fold():
    patients = {f"file{i}": f"p{i // 2}" for i in range(20)}   # two volumes per patient
    folds = make_folds(patients, k=5, seed=0)
    by_patient = {}
    for vol, f in folds.items():
        by_patient.setdefault(patients[vol], set()).add(f)
    assert all(len(v) == 1 for v in by_patient.values())
    assert set(folds.values()) == {0, 1, 2, 3, 4}


def test_export_writes_every_view_and_a_readable_meta(tmp_path):
    h5 = _write_h5(tmp_path / "file1.h5")
    lesions = [{"file": "file1", "family": "meniscus", "z0": 1, "z1": 2,
                "x0": 4, "y0": 5, "x1": 12, "y1": 15, "n_boxes": 2}]
    row = export_volume(h5, lesions, tmp_path / "out" / "file1", seed=0, size=32)
    for v in VIEWS:
        a = np.load(tmp_path / "out" / "file1" / f"{v}.npy")
        assert a.shape == (3, 32, 32) and a.dtype == np.float16
    meta = json.loads((tmp_path / "out" / "file1" / "meta.json").read_text())
    assert meta["slices"] == 3 and meta["patient_id"] == "p1" and meta["n_lesions"] == 1
    assert row["status"] == "ok"


def test_the_degraded_views_differ_from_clean_but_share_its_shape(tmp_path):
    h5 = _write_h5(tmp_path / "file2.h5", patient="p2")
    export_volume(h5, [], tmp_path / "out" / "file2", seed=0, size=32)
    clean = np.load(tmp_path / "out" / "file2" / "clean.npy").astype(np.float32)
    for v in ("noise_q3", "us16"):
        other = np.load(tmp_path / "out" / "file2" / f"{v}.npy").astype(np.float32)
        assert other.shape == clean.shape and not np.allclose(other, clean)


def test_one_mask_serves_every_slice_but_noise_varies_per_slice():
    from anatobind.data_engine.fastmri_knee import _view_seed
    assert _view_seed("us16", 100, 0) == _view_seed("us16", 100, 5) == 100
    assert _view_seed("noise_q3", 100, 0) != _view_seed("noise_q3", 100, 5)


def test_the_saved_undersampled_volume_uses_one_independently_recomputed_mask(tmp_path):
    import h5py
    from anatobind.data_engine.fastmri_knee import (
        equispaced_mask, export_volume, reconstruct_rss,
    )
    from anatobind.train.dataset import normalise_volume

    h5 = _write_h5(tmp_path / "file9.h5", slices=3, patient="p9")
    export_volume(h5, [], tmp_path / "out" / "file9", seed=7, size=32)
    saved = np.load(tmp_path / "out" / "file9" / "us16.npy")
    with h5py.File(h5) as h:
        k = h["kspace"][()]
    mask = equispaced_mask(k.shape[-1], 16, 0.04, 7)          # one mask, recomputed from scratch
    expect = normalise_volume(
        np.stack([reconstruct_rss(k[s] * mask, 32) for s in range(k.shape[0])])
    ).astype(np.float16)
    np.testing.assert_array_equal(saved, expect)
