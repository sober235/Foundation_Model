"""The unified adapter preserves old 3D crops and observes actual 2D planes."""
import importlib

import nibabel as nib
import numpy as np
import pytest
import torch

from anatobind.aur import dataset as legacy
from anatobind.aur.crops import normalise


def adapter():
    try:
        return importlib.import_module("anatobind.aur.unified_data")
    except ModuleNotFoundError:
        pytest.fail("The unified data adapter has not been implemented")


def volume():
    shape = (4, 8, 8)
    raw = np.full(shape, 10000.0, np.float32)
    raw[0] = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    inst = np.zeros(shape, np.int32)
    inst[0, 1:3, 1:3] = 1
    inst[0, 5:7, 5:7] = 1
    inst[1, 1:7, 1:7] = 1  # One source 3D instance, two visible planar components.
    inst[0, 3, 6] = 2
    small = np.zeros(shape, bool)
    small[0, 7, 0] = True
    probs = np.zeros((2, 14), np.float32)
    probs[0, 4], probs[1, 7] = 1.0, 1.0
    return {"image": normalise(raw), "raw_image": raw, "instance": inst,
            "entity": np.ones(shape, np.int16), "small": small, "a_ignore": inst > 0,
            "hosts": {"host": np.array([4, 7]), "probs": probs,
                      "negatives": np.full((2, 2), -1, np.int64)},
            "spacing": (5.0, 0.5, 2.0), "voxel_mm3": 5.0,
            "affine": np.diag([2.0, 0.5, 5.0, 1.0]), "seq": 3,
            "a_supervised": True, "u_supervised": True, "r_supervised": True,
            "case": "synthetic"}


def plane(vol=None, **kwargs):
    return adapter().make_unified_crop(volume() if vol is None else vol,
        np.random.default_rng(7), "2d", (1, 8, 8), do_augment=False,
        plane_index=0, **kwargs)


def test_true_plane_normalises_raw_plane_before_cropping():
    c = plane()
    assert c["image"].shape == (1, 1, 8, 8)
    assert float(c["image"][0, 0, 0, 0]) == -1.0
    assert float(c["image"][0, 0, -1, -1]) == 1.0
    assert (c["image"] > 0).sum() == 32
    assert c["input_mode"] == "2d" and c["measurement_unit"] == "mm2"
    assert c["voxel_volume_mm3"] is None and c["pixel_area_mm2"] == 1.0


def test_planar_components_keep_source_identity_ignore_and_area_units():
    c = plane(min_area_mm2=2.0)
    assert c["n_instances"] == 2
    assert c["source_instance_ids"].tolist() == [1, 1]
    assert torch.bincount(c["instance"].flatten()).tolist() == [56, 4, 4]
    assert float(c["point_weight"][0, 3, 6]) == 0.0  # Visible sliver below the area floor.
    assert float(c["point_weight"][0, 7, 0]) == 0.0  # Source 3D qualification remains intact.
    assert c["host"].tolist() == [4, 4]  # Preserve source host despite different planar anatomy.
    assert c["r_eligible"].tolist() == [False, False]
    assert bool(c["u_supervised"])


def test_planar_r_eligibility_tracks_component_renumbering():
    vol = volume()
    vol["source_r_eligible"] = {1: True, 2: False}
    c = plane(vol)
    assert c["source_instance_ids"].tolist() == [1, 1, 2]
    assert c["host"].tolist() == [4, 4, 7]
    assert c["r_eligible"].tolist() == [True, True, False]
    assert c["n_instances"] == 3 and bool(c["u_supervised"])
    vol["r_supervised"] = False
    assert not plane(vol)["r_eligible"].any()


def test_source_target_override_is_used_without_recomputing_plane_no_host():
    vol = volume()
    probs = np.eye(14, dtype=np.float32)[[2, 13]]
    vol["source_r_targets"] = {"host": np.array([2, 13]), "probs": probs,
                               "negatives": np.full((2, 2), -1, np.int64)}
    vol["source_r_eligible"] = [True, True]
    c = plane(vol)
    assert c["host"].tolist() == [2, 2, 13]
    assert c["r_eligible"].tolist() == [True, True, True]


def test_physical_coordinates_and_padding_stay_on_original_plane():
    c = adapter().make_unified_crop(volume(), np.random.default_rng(0), "2d", (1, 12, 12),
                                   do_augment=False, plane_index=0)
    assert c["valid"].sum() == 64
    assert (c["image"][0][c["valid"] == 0] == -1).all()
    assert (c["point_weight"][c["valid"] == 0] == 0).all()
    assert c["a_ignore"][c["valid"] == 0].all()
    assert c["coords"][:, 0, 2, 2].tolist() == [0.0, 0.0, 0.0]
    assert c["coords"][:, 0, 3, 3].tolist() == [0.0, 0.5, 2.0]


def test_local_plane_coordinates_do_not_reveal_source_depth_or_slice_index():
    a = adapter().make_unified_crop(volume(), np.random.default_rng(0), "2d", (1, 8, 8),
                                   do_augment=False, plane_index=2)
    deep = volume()
    for key in ("raw_image", "image", "instance", "entity", "small", "a_ignore"):
        deep[key] = np.concatenate([deep[key], deep[key]], axis=0)
    b = adapter().make_unified_crop(deep, np.random.default_rng(0), "2d", (1, 8, 8),
                                   do_augment=False, plane_index=6)
    assert (a["local"][0] == 0).all() and (b["local"][0] == 0).all()
    assert torch.equal(a["local"], b["local"])
    assert a["coords"][0, 0, 0, 0] == 10.0 and b["coords"][0, 0, 0, 0] == 30.0


def test_plane_intensity_scale_is_fixed_before_spatial_crop():
    c = adapter().make_unified_crop(volume(), np.random.default_rng(0), "2d", (1, 4, 4),
                                   do_augment=False, plane_index=0)
    assert c["image"].min() > 0 and c["image"].max() < 1


@pytest.mark.parametrize("change, kwargs, message", [
    ({}, {"n_events": 1}, "query capacity"),
    ({}, {"min_area_mm2": 0}, "minimum area"),
    ({}, {"plane_index": 4}, "plane index"),
    ({"raw_image": None}, {}, "raw image"),
])
def test_invalid_planes_and_capacity_are_explicit(change, kwargs, message):
    vol = {**volume(), **change}
    options = {"plane_index": 0, **kwargs}
    with pytest.raises(ValueError, match=message):
        adapter().make_unified_crop(vol, np.random.default_rng(0), "2d", (1, 8, 8),
                                   do_augment=False, **options)


@pytest.mark.parametrize("do_augment", [False, True])
def test_3d_legacy_outputs_and_rng_are_unchanged(do_augment):
    vol = volume()
    first, second = np.random.default_rng(6), np.random.default_rng(6)
    old = legacy.make_crop(vol, first, (4, 8, 8), do_augment=do_augment, lesion_centred=True)
    new = adapter().make_unified_crop(vol, second, "3d", (4, 8, 8),
                                     do_augment=do_augment, lesion_centred=True)
    for key, value in old.items():
        assert torch.equal(new[key], value) if isinstance(value, torch.Tensor) else new[key] == value
    assert first.random() == second.random()
    assert new["r_eligible"].tolist() == [True] * new["n_instances"]
    assert new["measurement_unit"] == "mm3" and new["voxel_volume_mm3"] == 5.0


def test_collation_keeps_instance_eligibility_and_refuses_mixed_modes():
    vol = volume()
    vol["source_r_eligible"] = {1: True}
    a, b = plane(vol), plane()
    batch = adapter().collate_unified([[a], [b]])
    assert batch["image"].shape == (2, 1, 1, 8, 8)
    assert batch["input_mode"] == "2d" and batch["spacing"].shape == (2, 3)
    assert [v.tolist() for v in batch["r_eligible"]] == [[True, True, False], [False, False, False]]
    assert batch["u_supervised"].tolist() == [True, True]
    three = adapter().make_unified_crop(vol, np.random.default_rng(0), "3d", (4, 8, 8), do_augment=False)
    with pytest.raises(ValueError, match="homogeneous"):
        adapter().collate_unified([[a], [three]])


def test_load_wrapper_keeps_unclipped_raw_and_split_provenance(tmp_path):
    img = np.arange(1, 257, dtype=np.float32).reshape(8, 8, 4)
    affine = np.diag([2.0, 0.5, 5.0, 1.0])
    paths = {}
    for name, array in (("image", img), ("anatomy", np.full(img.shape, 2, np.int16))):
        path = tmp_path / (name + ".nii.gz")
        nib.save(nib.Nifti1Image(array, affine), path)
        paths[name] = str(path)
    row = {**paths, "case": "synthetic", "patient": "fake", "source": "synthetic",
           "sequence": "FLAIR", "split": "train", "lesion": None,
           "u_supervised": False, "u_values": [], "a_ignore_values": []}
    vol = adapter().load_unified_volume(row)
    assert vol["raw_image"].shape == (4, 8, 8) and vol["raw_image"].max() == 256
    assert vol["image"].max() == 1.0 and vol["split"] == "train"
    assert vol["source"] == "synthetic" and vol["patient"] == "fake"
    assert np.array_equal(vol["raw_image"][0], img[:, :, 0].T)


def test_eligibility_and_source_host_survive_removal_of_an_earlier_instance():
    vol = volume()
    vol["instance"][0] = 0
    vol["instance"][0, 0, 0] = 1
    vol["instance"][0, 4:6, 4:6] = 2
    vol["source_r_eligible"] = [False, True]
    c = plane(vol, min_area_mm2=2.0)
    assert c["n_instances"] == 1 and c["source_instance_ids"].tolist() == [2]
    assert c["host"].tolist() == [7] and c["r_eligible"].tolist() == [True]
    assert c["point_weight"][0, 0, 0] == 0


def test_empty_plane_and_a_only_plane_do_not_invent_supervision():
    vol = volume()
    vol["u_supervised"] = False
    c = adapter().make_unified_crop(vol, np.random.default_rng(0), "2d", (1, 8, 8),
                                   do_augment=False, plane_index=3)
    assert c["n_instances"] == 0 and c["host"].shape == (0,)
    assert c["host_probs"].shape == (0, 14) and c["r_eligible"].shape == (0,)
    assert not c["u_supervised"] and c["a_supervised"] and c["point_weight"].sum() == 64


def test_plane_area_floor_is_independent_of_source_slice_thickness():
    thin, thick = volume(), volume()
    thick["affine"][2, 2] = 500.0
    thick["spacing"] = (500.0, 0.5, 2.0)
    thick["voxel_mm3"] = 500.0
    a, b = plane(thin, min_area_mm2=2.0), plane(thick, min_area_mm2=2.0)
    assert torch.equal(a["instance"], b["instance"]) and torch.equal(a["point_weight"], b["point_weight"])
    assert b["pixel_area_mm2"] == 1.0 and b["voxel_volume_mm3"] is None


@pytest.mark.parametrize("target_key, replacement", [
    ("host", np.array([-1, 7])),
    ("probs", np.full((2, 14), np.nan)),
    ("probs", np.zeros((2, 14))),
])
def test_eligible_relations_require_valid_source_targets(target_key, replacement):
    vol = volume()
    vol["source_r_eligible"] = [True, False]
    vol["hosts"][target_key] = replacement
    with pytest.raises(ValueError, match="source relation targets"):
        plane(vol)
