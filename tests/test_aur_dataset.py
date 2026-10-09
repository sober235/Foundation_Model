# tests/test_aur_dataset.py
import nibabel as nib
import numpy as np
import pytest
import torch

import anatobind.aur.dataset as D
from anatobind.aur.labels import HOST_NAMES, N_HOST_CLASSES, SEQ_INDEX


def _case(tmp_path, with_lesion=True):
    """A (24, 20, 10) volume at (1, 1, 2) mm: left white matter at x < 12, right at x >= 12, a ventricle, one lesion of
    3 x 3 x 2 voxels (36 mm3) in the left white matter and one single voxel (2 mm3, under the floor)."""
    shape, spacing = (24, 20, 10), (1.0, 1.0, 2.0)
    affine = np.diag(list(spacing) + [1.0])
    seg = np.zeros(shape, np.int16)
    seg[:12, :, :] = 2
    seg[12:, :, :] = 41
    seg[10:14, 8:12, :] = 4
    img = np.where(seg > 0, 300.0, 0.0).astype(np.float32) + np.random.default_rng(0).normal(0, 5, shape).astype(np.float32)
    les = np.zeros(shape, np.uint8)
    les[2:5, 2:5, 3:5] = 1
    les[20, 2, 2] = 1
    les[6:8, 14:16, 6:8] = 2                                  # a value that is lesion on other sequences only
    paths = {}
    for name, arr in (("image", img), ("anatomy", seg), ("lesion", les)):
        p = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(arr, affine), str(p))
        paths[name] = str(p)
    return {"case": "c1", "source": "pdgm", "patient": "p1", "sequence": "FLAIR", "source_sequence": "FLAIR", "image": paths["image"],
            "anatomy": paths["anatomy"], "lesion": paths["lesion"] if with_lesion else None, "u_supervised": with_lesion,
            "u_values": [1] if with_lesion else [], "a_ignore_values": [1, 2] if with_lesion else [], "split": "train"}


def test_load_volume_builds_the_targets_in_zyx(tmp_path):
    vol = D.load_volume(_case(tmp_path))
    assert vol["image"].shape == (10, 20, 24) and vol["spacing"] == (2.0, 1.0, 1.0) and vol["seq"] == SEQ_INDEX["FLAIR"]
    assert vol["entity"].shape == (10, 20, 24) and vol["instance"].max() == 1 and vol["small"].sum() == 1
    assert vol["a_ignore"].sum() == 18 + 1 + 8 and vol["hosts"]["host"].tolist() == [HOST_NAMES.index("white_matter_left")]
    assert vol["image"].min() == -1.0 and vol["image"].max() <= 1.0
    bare = D.load_volume(_case(tmp_path, with_lesion=False))
    assert bare["instance"].max() == 0 and not bare["u_supervised"] and bare["hosts"]["host"].shape == (0,)


def test_crops_carry_geometry_validity_and_renumbered_instances(tmp_path):
    vol = D.load_volume(_case(tmp_path))
    rng = np.random.default_rng(3)
    c = D.make_crop(vol, rng, crop=(8, 16, 16), do_augment=False, lesion_centred=True)
    assert c["image"].shape == (1, 8, 16, 16) and c["valid"].shape == (8, 16, 16) and c["coords"].shape == (3, 8, 16, 16)
    assert c["instance"].max() == 1 and c["n_instances"] == 1 and c["host"].tolist() == [HOST_NAMES.index("white_matter_left")]
    assert c["host_probs"].shape == (1, N_HOST_CLASSES) and c["negatives"].shape == (1, 2) and bool(c["u_supervised"])
    assert (c["image"][0][c["valid"] < 0.5] == -1.0).all() and c["point_weight"].min() >= 0 and c["entity"].dtype == torch.int64
    dz = c["coords"][0, 1, 0, 0] - c["coords"][0, 0, 0, 0]
    assert float(dz) == 2.0 and float(c["coords"][1, 0, 1, 0] - c["coords"][1, 0, 0, 0]) == 1.0
    a = D.make_crop(vol, np.random.default_rng(4), crop=(8, 16, 16), do_augment=True, lesion_centred=False)
    assert a["image"].shape == (1, 8, 16, 16) and (a["image"][0][a["valid"] < 0.5] == -1.0).all() and set(a["entity"].unique().tolist()) <= {0, 1, 2, 3, 4, 5}
    with pytest.raises(ValueError, match="multiple of the patch"):
        D.make_crop(vol, rng, crop=(7, 16, 16))
    big = D.make_crop(vol, rng, crop=(16, 32, 32), do_augment=False)          # a crop larger than the volume in every axis
    assert big["valid"].sum() == 10 * 20 * 24 and (big["image"][0][big["valid"] < 0.5] == -1.0).all() and big["entity"][0, 0, 0] == 0


def test_slivers_under_the_floor_leave_the_crop_without_an_instance():
    inst = np.zeros((4, 6, 6), np.int32)
    inst[0, 0, :3] = 1                                             # 3 voxels: at 2 mm3 each 6 mm3, under the 10 mm3 floor
    inst[1:4, 1:4, 1:4] = 2                                        # 27 voxels
    small = np.zeros(inst.shape, bool)
    small[3, 5, 5] = True
    renumbered, small2, kept = D.crop_instances(inst, small, 2.0)
    assert kept == [2] and renumbered.max() == 1 and (renumbered[1:4, 1:4, 1:4] == 1).all()
    assert small2[0, 0, :3].all() and small2[3, 5, 5] and small2.sum() == 4
    r1, s1, k1 = D.crop_instances(inst, small, 8.0)                 # 8 mm3 voxels: both instances reach the floor
    assert k1 == [1, 2] and r1.max() == 2 and s1.sum() == 1


def test_dataset_items_and_collate(tmp_path):
    rows = [_case(tmp_path), _case(tmp_path, with_lesion=False)]
    ds = D.AURDataset(rows, crop=(8, 16, 16), crops_per_volume=2, seed=1)
    item = ds[0]
    assert len(item) == 2 and item[0]["case"] == "c1"
    batch = D.collate([ds[0], ds[1]])
    assert batch["image"].shape == (4, 1, 8, 16, 16) and batch["seq"].tolist() == [3, 3, 3, 3] and batch["u_supervised"].tolist() == [True, True, False, False]
    assert len(batch["host"]) == 4 and batch["n_instances"][2] == 0 and batch["instance"].shape == (4, 8, 16, 16)
    pts = torch.tensor([[0, 1, 2]])
    inst_pts = torch.tensor([[0, 2, 1]])
    t = D.event_targets_at_points(inst_pts, [2])
    assert t[0].tolist() == [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]] and D.event_targets_at_points(inst_pts, [0])[0].shape == (0, 3)
    ds.set_epoch(1)
    assert ds.epoch == 1 and len(ds) == 2
