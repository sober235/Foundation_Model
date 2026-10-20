"""Image-only independent planes and legacy-compatible volumes for Stage I."""
import importlib

import nibabel as nib
import numpy as np
import pytest
import torch

from anatobind.aur.ssl import dataset as legacy
from anatobind.aur.ssl.model import StageOne


def api():
    return importlib.import_module("anatobind.aur.ssl.unified_data")


class ImageOnlyRow(dict):
    def __getitem__(self, key):
        if key in ("anatomy", "lesion", "instance", "u_values", "a_ignore_values"):
            raise AssertionError("SSL attempted to inspect supervision")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in ("anatomy", "lesion", "instance", "u_values", "a_ignore_values"):
            raise AssertionError("SSL attempted to inspect supervision")
        return super().get(key, default)


def image_row(tmp_path):
    raw = np.arange(1, 8 * 20 * 20 + 1, dtype=np.float32).reshape(20, 20, 8)
    raw[:, :, 3] *= 1000
    affine = np.array([[1.2, .1, .2, 7], [0, 1.5, .1, 11], [.2, .3, 2., 13], [0, 0, 0, 1.]])
    path = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(raw, affine), path)
    row = ImageOnlyRow(image=str(path), case="synthetic", source="pdgm", patient="synthetic", sequence="T1",
                       anatomy="forbidden", lesion="forbidden", split="train", ssl_split="train")
    return row, raw


@pytest.mark.parametrize("mode,crop", [("2d", (1, 16, 16)), ("3d", (8, 16, 16))])
def test_image_only_loader_and_crops_train_both_modes_with_no_label_access(tmp_path, monkeypatch, mode, crop):
    # Calling a supervised loader or reading a label must break this integration test.
    import anatobind.aur.dataset as supervised
    import anatobind.aur.unified_data as supervised_unified
    def forbidden(*args, **kwargs):
        raise AssertionError("SSL read a label or used the supervised loader")
    monkeypatch.setattr(supervised, "load_volume", forbidden)
    monkeypatch.setattr(supervised_unified, "load_unified_volume", forbidden)
    row, _ = image_row(tmp_path)
    load = nib.load
    monkeypatch.setattr(nib, "load", lambda path, *args, **kwargs: load(path, *args, **kwargs) if str(path) == row["image"] else forbidden())
    data = api()
    dataset = data.UnifiedSSLDataset([row], mode, crop, crops_per_volume=2, seed=8)
    batch = data.collate_unified_ssl([dataset[0]])
    assert batch["input_mode"] == mode and batch["view1"].shape == (2, 1, *crop)
    assert batch["coords"].shape == (2, 3, *crop)
    assert set(batch) == {*legacy.STACKED, "case", "input_mode"}
    model = StageOne(embed=32, depths=(1, 1, 1, 1), heads=(1, 2, 4, 8), window=(2, 4, 4),
                     dual_input=True, mask_ratio=.4, block_mm=(4., 8.))
    result = model(batch, torch.Generator().manual_seed(3), input_mode=mode)
    result["loss"].backward()
    assert torch.isfinite(result["loss"]) and result["hidden_voxels"] > 0
    for stage in model.backbone.stages:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stage.parameters())
    inactive = ((model.backbone.patch_embed, model.decoder.net[3]) if mode == "2d"
                else (model.backbone.patch_embed_2d, model.decoder.output_2d))
    assert all(p.grad is None for module in inactive for p in module.parameters())


def test_plane_normalisation_uses_complete_unclipped_plane_and_physical_coordinates(tmp_path, monkeypatch):
    data = api()
    row, raw = image_row(tmp_path)
    vol = data.load_unified_ssl_image(row)
    # Freeze intensity views only; crop selection and intensity normalization remain real.
    monkeypatch.setattr(legacy, "intensity_view", lambda image, rng: image.copy())
    crop = data.make_unified_ssl_crop(vol, np.random.default_rng(4), "2d", (1, 24, 24),
                                     do_rotate=False, foreground_share=0, plane_index=3)
    plane = raw[:, :, 3].T
    lo, hi = np.percentile(plane, [.5, 99.5])
    expected = (np.clip(plane, lo, hi) - lo) / (hi - lo) * 2 - 1
    assert np.allclose(crop["view1"][0, 0, 2:22, 2:22], expected, atol=1e-6)
    assert torch.equal(crop["local"][0], torch.zeros(1, 24, 24))
    # First observed voxel: xyz=(0,0,3), physical RAS=(7.6,11.3,19), returned zyx.
    assert torch.allclose(crop["coords"][:, 0, 2, 2], torch.tensor([19., 11.3, 7.6]))
    assert torch.all(crop["view1"][0][crop["valid"] == 0] == -1)


def test_three_dimensional_path_preserves_all_legacy_crop_tensors(tmp_path):
    data = api()
    row, _ = image_row(tmp_path)
    original = legacy.make_ssl_crop(legacy.load_image(row), np.random.default_rng(42), (8, 24, 24))
    unified = data.make_unified_ssl_crop(data.load_unified_ssl_image(row), np.random.default_rng(42), "3d", (8, 24, 24))
    assert all(torch.equal(original[key], unified[key]) for key in legacy.STACKED)
    assert original["case"] == unified["case"]


@pytest.mark.parametrize("kwargs", [dict(input_mode="2d", crop=(2, 16, 16)), dict(input_mode="invalid", crop=(1, 16, 16)),
                                    dict(input_mode="2d", crop=(1, 15, 16)), dict(input_mode="2d", crop=(1, 16, 16), plane_index=-1),
                                    dict(input_mode="2d", crop=(1, 16, 16), foreground_share=1.1)])
def test_invalid_crop_contracts_are_refused(tmp_path, kwargs):
    data = api()
    row, _ = image_row(tmp_path)
    with pytest.raises(ValueError):
        data.make_unified_ssl_crop(data.load_unified_ssl_image(row), np.random.default_rng(0), **kwargs)


def test_collation_refuses_mixed_modes_and_dataset_rejects_held_out_rows(tmp_path):
    data = api()
    row, _ = image_row(tmp_path)
    vol = data.load_unified_ssl_image(row)
    crops = [data.make_unified_ssl_crop(vol, np.random.default_rng(4), mode, shape)
             for mode, shape in [("2d", (1, 16, 16)), ("3d", (8, 16, 16))]]
    with pytest.raises(ValueError, match="mode"):
        data.collate_unified_ssl([crops])
    for key in ("split", "ssl_split"):
        bad = ImageOnlyRow(row)
        bad[key] = "test"
        with pytest.raises(ValueError, match="train"):
            data.UnifiedSSLDataset([bad], "2d", (1, 16, 16))


@pytest.mark.parametrize("role,split,ssl_split", [("train_ssl", "train", "train"), ("ssl_validation", "val", "val"),
                                               (None, "train", "val")])
def test_explicit_contract_roles_and_legacy_validation_are_supported(tmp_path, role, split, ssl_split):
    data = api()
    row, _ = image_row(tmp_path)
    row.update(split=split, ssl_split=ssl_split)
    if role is not None:
        row["role"] = role
    dataset = data.UnifiedSSLDataset([row], "2d", (1, 16, 16), ssl_split=ssl_split)
    assert len(dataset[0]) == 2


@pytest.mark.parametrize("role,split,ssl_split,requested", [("ssl_validation", "train", "val", "val"),
                                                         ("train_ssl", "val", "train", "train"),
                                                         ("ssl_validation", "val", "train", "val"),
                                                         ("train_supervised", "train", "train", "train")])
def test_conflicting_contract_roles_or_splits_are_refused(tmp_path, role, split, ssl_split, requested):
    data = api()
    row, _ = image_row(tmp_path)
    row.update(role=role, split=split, ssl_split=ssl_split)
    with pytest.raises(ValueError, match="partition"):
        data.UnifiedSSLDataset([row], "2d", (1, 16, 16), ssl_split=requested)
