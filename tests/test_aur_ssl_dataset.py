# tests/test_aur_ssl_dataset.py
import nibabel as nib
import numpy as np
import pytest
import torch

import anatobind.aur.ssl.dataset as D


def _row(tmp_path, source="pdgm", patient="p1", shape=(24, 20, 12), spacing=(1.0, 1.0, 1.0)):
    affine = np.diag([-spacing[0], spacing[1], spacing[2], 1.0])            # stored LAS
    affine[0, 3] = (shape[0] - 1) * spacing[0]
    rng = np.random.default_rng(0)
    img = np.zeros(shape, np.float32)
    img[4:20, 4:16, 2:10] = rng.uniform(100, 300, (16, 12, 8))
    p = tmp_path / f"{patient}.nii.gz"
    nib.save(nib.Nifti1Image(img, affine), str(p))
    return {"case": patient, "source": source, "patient": patient, "sequence": "T1", "source_sequence": "T1", "image": str(p),
            "anatomy": None, "lesion": None, "split": "train", "ssl_split": "train"}


def test_views_are_pointwise_and_keep_the_background():
    rng = np.random.default_rng(1)
    img = np.full((8, 16, 16), -1.0, np.float32)
    img[2:6, 4:12, 4:12] = rng.uniform(-0.8, 0.8, (4, 8, 8)).astype(np.float32)
    a = D.intensity_view(img, np.random.default_rng(5), p=1.0)
    assert a.shape == img.shape and a.dtype == np.float32 and (a[img == -1.0] == -1.0).all() and not np.allclose(a, img)
    changed = img.copy()
    changed[3, 6, 6] = 0.5                                                   # one voxel differs
    b = D.intensity_view(changed, np.random.default_rng(5), p=1.0)
    diff = np.argwhere(a != b)
    assert len(diff) == 1 and tuple(diff[0]) == (3, 6, 6)                  # a blur would spread the change to its neighbours
    assert np.array_equal(D.intensity_view(img, np.random.default_rng(0), p=0.0), img)


def test_crops_carry_two_views_on_one_geometry(tmp_path):
    vol = D.load_image(_row(tmp_path))
    assert vol["image"].shape == (12, 20, 24) and vol["spacing"] == (1.0, 1.0, 1.0) and vol["patient"] == D.patient_id("pdgm", "p1")
    c = D.make_ssl_crop(vol, np.random.default_rng(2), crop=(8, 16, 16), do_rotate=True)
    assert c["view1"].shape == (1, 8, 16, 16) and c["view2"].shape == (1, 8, 16, 16) and c["valid"].shape == (8, 16, 16)
    assert c["coords"].shape == (3, 8, 16, 16) and c["local"].shape == (3, 8, 16, 16) and c["coords"].dtype == torch.float32
    assert not torch.equal(c["view1"], c["view2"])
    inv = c["valid"] < 0.5
    assert (c["view1"][0][inv] == -1.0).all() and (c["view2"][0][inv] == -1.0).all()
    assert float(c["coords"][1, 0, 1, 0] - c["coords"][1, 0, 0, 0]) == 1.0           # the coordinate grid is not rotated
    assert c["spacing"].tolist() == [1.0, 1.0, 1.0] and c["seq"] == 0 and c["source"] == D.SOURCE_INDEX["pdgm"] and c["case"] == "p1"
    plain = D.make_ssl_crop(vol, np.random.default_rng(2), crop=(8, 16, 16), do_rotate=False, foreground_share=1.0)
    assert (plain["view1"][0] > -1.0).float().mean() > 0.3                             # centred on foreground
    with pytest.raises(ValueError, match="multiple of the patch"):
        D.make_ssl_crop(vol, np.random.default_rng(0), crop=(7, 16, 16))


def test_dataset_items_collate_and_sampling_weights(tmp_path):
    rows = [_row(tmp_path, "pdgm", "p1"), _row(tmp_path, "hcp", "h1"), _row(tmp_path, "hcp", "h2"), _row(tmp_path, "hcp", "h3")]
    ds = D.SSLDataset(rows, crop=(8, 16, 16), crops_per_volume=2, seed=1)
    item = ds[0]
    assert len(item) == 2 and item[0]["case"] == "p1"
    batch = D.collate_ssl([ds[0], ds[1]])
    assert batch["view1"].shape == (4, 1, 8, 16, 16) and batch["patient"].shape == (4,) and batch["patient"].dtype == torch.int64
    assert batch["patient"][0] == batch["patient"][1] and batch["patient"][0] != batch["patient"][2] and len(batch["case"]) == 4
    w, share = D.source_weights(rows, hcp_cap=0.25)
    assert pytest.approx(w.sum()) == 1.0 and share["hcp"] == pytest.approx(0.25) and share["pdgm"] == pytest.approx(0.75)
    assert w[0] == pytest.approx(0.75) and all(abs(x - 0.25 / 3) < 1e-9 for x in w[1:])
    two = rows[:1] + [dict(rows[0], sequence="FLAIR", source_sequence="FLAIR")] + rows[1:2]
    w2, share2 = D.source_weights(two, hcp_cap=0.25)
    assert share2["hcp"] == pytest.approx(0.25) and w2[0] == pytest.approx(w2[1]) and w2[0] + w2[1] == pytest.approx(0.75)   # a patient's rows share its weight
    no_hcp, share3 = D.source_weights(rows[:1])
    assert no_hcp.tolist() == [1.0] and share3 == {"pdgm": 1.0}
