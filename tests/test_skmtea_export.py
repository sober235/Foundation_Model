import csv

import h5py
import nibabel as nib
import numpy as np

from anatobind.data_engine.skmtea import export_scan


def _fake_scan(tmp_path, X=8, Y=8, Z=4, C=2, seg_zooms=(0.3125, 0.3125, 0.8)):
    rng = np.random.default_rng(0)
    img = rng.standard_normal((X, Y, Z)) + 1j * rng.standard_normal((X, Y, Z))
    maps = rng.standard_normal((X, Y, Z, C)) + 1j * rng.standard_normal((X, Y, Z, C))
    maps /= np.sqrt((np.abs(maps) ** 2).sum(-1, keepdims=True))
    coil = img[..., None] * maps
    k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(coil, axes=(1, 2)), axes=(1, 2), norm="ortho"), axes=(1, 2))
    with h5py.File(tmp_path / "MTR_t.h5", "w") as f:
        f.create_dataset("kspace", data=np.stack([k, k], axis=3).astype(np.complex64))        # (X,KY,KZ,2,C)
        f.create_dataset("maps", data=maps[..., None].astype(np.complex64))                     # (X,Y,Z,C,1)
        f.create_dataset("target", data=np.stack([img, img], axis=3)[..., None].astype(np.complex64))  # (X,Y,Z,2,1)
        g = f.create_group("masks")
        for r in (4, 8, 16):
            m = np.zeros((Y - 2, Z - 2), dtype=bool)
            m[::2, :] = True
            g.create_dataset(f"poisson_{r}.0x", data=m)
    seg = np.zeros((Y, X, Z), dtype=np.uint8)  # NIfTI frame is (y, x, z)
    seg[:, :6, :] = 5  # h5-frame x < 6: medial meniscus; the padded box (x 0..7) stays mostly medial
    seg[:, 6:, :] = 6
    nib.save(nib.Nifti1Image(seg, np.diag(list(seg_zooms) + [1.0])), str(tmp_path / "MTR_t.nii.gz"))
    return tmp_path / "MTR_t.h5", tmp_path / "MTR_t.nii.gz"


def test_export_scan_writes_images_seg_boxes_and_manifest_row(tmp_path):
    h5, nii = _fake_scan(tmp_path)
    boxes = [{"ann_id": 1, "split": "train", "layer": "in_seg", "supercategory": "Meniscal Tear", "category_id": 3,
              "tissue_id": 1, "keep": True, "x0": 0, "y0": 0, "z0": 0, "x1": 3, "y1": 4, "z1": 2},
             {"ann_id": 2, "split": "train", "layer": "in_seg", "supercategory": "Meniscal Tear", "category_id": 3,
              "tissue_id": 1, "keep": False, "x0": 0, "y0": 0, "z0": 0, "x1": 3, "y1": 4, "z1": 2}]
    out = tmp_path / "out"
    row = export_scan(h5, nii, boxes, out, conditions={"noise": [0.25, 0.5, 1.0], "us": [4, 8, 16]}, rng_seed=0,
                      orientation=("SI", "AP", "LR"))

    names = {p.name for p in out.iterdir()}
    for n in ["image_clean_e1.nii.gz", "image_clean_e2.nii.gz", "image_noise_q1_e1.nii.gz", "image_noise_q3_e1.nii.gz",
              "image_us4_e1.nii.gz", "image_us16_e1.nii.gz", "seg.nii.gz", "boxes.csv"]:
        assert n in names
    img = nib.load(str(out / "image_clean_e1.nii.gz"))
    assert img.shape == (4, 4, 4) and np.allclose(img.header.get_zooms(), (0.625, 0.625, 0.8))
    seg = nib.load(str(out / "seg.nii.gz"))
    assert seg.shape == (4, 4, 4) and set(np.unique(np.asarray(seg.dataobj)).tolist()) <= {5, 6}
    with open(out / "boxes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["ann_id"] == "1" and rows[0]["host_label"] == "5"
    assert (rows[0]["x0"], rows[0]["x1"]) == ("0", "2") and rows[0]["x1_full"] == "3"
    assert row["scan_id"] == "MTR_t" and row["n_boxes_kept"] == 1


def test_export_uses_the_segmentation_spacing_permuted_to_the_h5_frame(tmp_path):
    """4 scans were reconstructed on a 180 mm FOV; a hardcoded 0.3125 mm mislabels them by 12.5%."""
    h5, nii = _fake_scan(tmp_path, seg_zooms=(0.4, 0.3, 0.8))  # the NIfTI is (y, x, z)
    out = tmp_path / "out"
    row = export_scan(h5, nii, [], out, conditions={"noise": [0.25], "us": [4]}, rng_seed=0,
                      orientation=("SI", "AP", "LR"))
    for name in ("image_clean_e1.nii.gz", "image_noise_q1_e1.nii.gz", "seg.nii.gz"):
        assert np.allclose(nib.load(str(out / name)).header.get_zooms(), (0.6, 0.8, 0.8)), name
    assert np.allclose([float(v) for v in row["spacing"].split(";")], (0.6, 0.8, 0.8))


def test_manifest_row_records_realised_degradation_and_host_gaps(tmp_path):
    """The coil confound was invisible because nothing recorded what the ladder actually delivered."""
    h5, nii = _fake_scan(tmp_path)
    boxes = [{"ann_id": 7, "split": "train", "layer": "in_seg", "supercategory": "Cartilage Lesion",
              "category_id": 1, "tissue_id": 4, "keep": True,
              "x0": 0, "y0": 0, "z0": 0, "x1": 2, "y1": 2, "z1": 2}]  # femoral cartilage, no label 2 in _seg
    row = export_scan(h5, nii, boxes, tmp_path / "out", conditions={"noise": [0.25, 0.5], "us": [4]},
                      rng_seed=0, orientation=("SI", "AP", "LR"))
    nrmse = {k: float(v) for k, v in (p.split("=") for p in row["nrmse"].split(";"))}
    assert set(nrmse) == {"noise_q1", "noise_q2", "us4"}
    assert 0.0 < nrmse["noise_q1"] < nrmse["noise_q2"]
    assert row["n_no_overlap"] == 1 and row["n_unresolved"] == 0
