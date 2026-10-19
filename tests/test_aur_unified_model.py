"""Behavior of the opt-in shared 2D/3D model and explicit legacy migration."""
import copy
import io

import pytest
import torch

from anatobind.aur.model import AnatoBindBrain
from anatobind.aur.ssl.model import StageOne
from anatobind.aur.swin import SwinBackbone


BACKBONE = dict(embed=32, depths=(1, 2, 1, 1), heads=(1, 2, 4, 8), window=(2, 4, 4))
MODEL = dict(**BACKBONE, d_model=16, n_events=3, mask_dim=4, pixel_dim=8,
             dec_layers=1, dec_heads=2, rel_layers=1, rel_heads=2, use_checkpoint=False)


def inputs(shape, batch_size=1):
    generator = torch.Generator().manual_seed(37)
    image = torch.rand(batch_size, 1, *shape, generator=generator) * 1.5 - 0.75
    valid = torch.ones(batch_size, *shape)
    valid[..., -4:] = 0
    image[..., -4:] = -1
    axes = [torch.arange(n, dtype=torch.float32) for n in shape]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))
    # An oblique physical plane: even D=1 has three changing patient-space components.
    affine = torch.tensor([[2.0, 0.3, 0.2], [0.0, 1.1, 0.4], [0.0, -0.2, 0.9]])
    coords = torch.einsum("ij,jdhw->idhw", affine, grid) + torch.tensor([7., 11., 13.])[:, None, None, None]
    local = torch.stack(torch.meshgrid(*[(a + 0.5) / n * 2 - 1 for a, n in zip(axes, shape)], indexing="ij"))
    return image, valid, coords[None].expand(batch_size, -1, -1, -1, -1).clone(), local[None].expand(batch_size, -1, -1, -1, -1).clone()


def ssl_batch(shape, batch_size=2):
    image, valid, coords, local = inputs(shape, batch_size)
    return dict(view1=image, view2=image.clone(), valid=valid, coords=coords, local=local,
                spacing=torch.tensor([[2., 1., 1.]] * batch_size), patient=torch.arange(batch_size))


def assert_same_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_tree(left[key], right[key])
    else:
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same_tree(a, b)


def nonzero_gradient(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_one_model_switches_modes_and_uses_only_the_selected_adapters():
    model = AnatoBindBrain(**MODEL, dual_input=True)
    shared_ids = [id(parameter) for parameter in model.backbone.stages.parameters()]
    for mode, shape in (("3d", (8, 32, 32)), ("2d", (1, 36, 28)), ("3d", (8, 32, 32))):
        model.zero_grad(set_to_none=True)
        out = model(*inputs(shape), input_mode=mode)
        anatomy, lesions = model.entity_masks(out), model.event_masks(out)
        relation = model.bind(out, 0, torch.tensor([0, 2]))
        assert anatomy.shape == (1, 32, *shape) and lesions.shape == (1, 3, *shape)
        assert relation.shape == (2, 14) and torch.isfinite(relation).all()
        (anatomy.square().mean() + lesions.square().mean() + relation.square().mean()).backward()
        selected_stem = model.backbone.patch_embed_2d if mode == "2d" else model.backbone.patch_embed
        selected_up = model.masks.up_2d if mode == "2d" else model.masks.up
        unused_stem = model.backbone.patch_embed if mode == "2d" else model.backbone.patch_embed_2d
        unused_up = model.masks.up if mode == "2d" else model.masks.up_2d
        for module in (selected_stem, selected_up, model.entities, model.events, model.relation, *model.backbone.stages):
            nonzero_gradient(module)
        assert all(parameter.grad is None for module in (unused_stem, unused_up) for parameter in module.parameters())
        if mode == "2d":
            assert all(level["feat"].shape[2] == 1 for level in out["levels"])
            assert torch.equal(out["levels"][0]["coords"], torch.nn.functional.avg_pool3d(inputs(shape)[2], (1, 4, 4)))
        assert [id(parameter) for parameter in model.backbone.stages.parameters()] == shared_ids


@pytest.mark.parametrize("mode,shape", [("2d", (2, 32, 32)), ("4d", (8, 32, 32)), ("2d", (1, 31, 32))])
def test_invalid_mode_or_shape_is_refused(mode, shape):
    model = AnatoBindBrain(**MODEL, dual_input=True)
    with pytest.raises(ValueError):
        model(*inputs(shape), input_mode=mode)


def test_legacy_interface_keeps_its_keys_and_refuses_implicit_two_dimensional_input():
    model = AnatoBindBrain(**MODEL)
    assert not any("_2d" in key for key in model.state_dict())
    with pytest.raises(ValueError):
        model(*inputs((1, 32, 32)), input_mode="2d")


@pytest.mark.parametrize("kind", ["model", "backbone", "ssl"])
def test_explicit_legacy_migration_preserves_every_three_dimensional_output(kind):
    from anatobind.aur.unified import migrate_legacy_state
    constructor, options = {"model": (AnatoBindBrain, MODEL), "backbone": (SwinBackbone, BACKBONE), "ssl": (StageOne, BACKBONE)}[kind]
    torch.manual_seed(9)
    legacy = constructor(**options).eval()
    unified = constructor(**options, dual_input=True).eval()
    audit = migrate_legacy_state(unified, legacy.state_dict())
    assert set(audit["copied_keys"]) == set(legacy.state_dict())
    assert set(audit["initialized_keys"]) == set(unified.state_dict()) - set(legacy.state_dict())
    if kind == "backbone":
        expected = {"patch_embed_2d.weight", "patch_embed_2d.bias"}
    else:
        output = "masks.up_2d" if kind == "model" else "decoder.output_2d"
        expected = {"backbone.patch_embed_2d.weight", "backbone.patch_embed_2d.bias", f"{output}.weight", f"{output}.bias"}
    assert set(audit["initialized_keys"]) == expected
    assert audit["initialized_keys"] and len(audit["state_sha256"]) == 64
    data = inputs((8, 32, 32))
    with torch.no_grad():
        if kind == "ssl":
            patch_shape = (1, 4, 8, 8)
            hidden = torch.zeros(patch_shape, dtype=torch.bool)
            hidden[..., ::2] = True
            old = legacy.view(*data, hidden)
            new = unified.view(*data, hidden, input_mode="3d")
        else:
            old, new = legacy(*data), unified(*data, input_mode="3d")
        assert_same_tree(old, new)
        if kind == "model":
            assert torch.equal(legacy.entity_masks(old), unified.entity_masks(new))
            assert torch.equal(legacy.event_masks(old), unified.event_masks(new))
            assert torch.equal(legacy.bind(old, 0, torch.tensor([0])), unified.bind(new, 0, torch.tensor([0])))


@pytest.mark.parametrize("damage", ["shared_missing", "extra", "wrong_shape", "wrong_dtype", "partial_adapters", "already_unified"])
def test_legacy_migration_rejects_every_unexpected_difference_without_mutation(damage):
    from anatobind.aur.unified import migrate_legacy_state
    legacy = AnatoBindBrain(**MODEL)
    target = AnatoBindBrain(**MODEL, dual_input=True)
    state = copy.deepcopy(legacy.state_dict())
    if damage == "shared_missing":
        state.pop("entities.presence.weight")
    elif damage == "extra":
        state["unexpected"] = torch.zeros(1)
    elif damage == "wrong_shape":
        state["backbone.patch_embed.weight"] = torch.zeros(1)
    elif damage == "wrong_dtype":
        state["backbone.patch_embed.weight"] = state["backbone.patch_embed.weight"].double()
    elif damage == "partial_adapters":
        state["backbone.patch_embed_2d.weight"] = target.backbone.patch_embed_2d.weight.detach().clone()
    else:
        state = target.state_dict()
    before = copy.deepcopy(target.state_dict())
    with pytest.raises((ValueError, RuntimeError)):
        migrate_legacy_state(target, state)
    assert_same_tree(before, target.state_dict())


def test_one_serialized_checkpoint_round_trips_both_modes_with_strict_loading():
    model = AnatoBindBrain(**MODEL, dual_input=True).eval()
    stream = io.BytesIO()
    torch.save(dict(model_state_dict=model.state_dict(), config=model.cfg, format_version=1), stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    loaded = AnatoBindBrain(**checkpoint["config"]).eval()
    loaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with torch.no_grad():
        for mode, shape in (("2d", (1, 32, 32)), ("3d", (8, 32, 32))):
            assert_same_tree(model(*inputs(shape), input_mode=mode), loaded(*inputs(shape), input_mode=mode))
    missing = dict(checkpoint["model_state_dict"])
    missing.pop("backbone.patch_embed_2d.weight")
    with pytest.raises(RuntimeError):
        loaded.load_state_dict(missing, strict=True)


@pytest.mark.parametrize("mode,shape", [("2d", (1, 32, 32)), ("3d", (8, 32, 32))])
def test_ssl_reconstruction_reaches_all_shared_stages_and_hides_image_content(mode, shape):
    model = StageOne(**BACKBONE, dual_input=True, mask_ratio=0.4, block_mm=(4., 8.), contrast_weight=0.)
    batch = ssl_batch(shape)
    out = model(batch, torch.Generator().manual_seed(5), input_mode=mode)
    assert torch.isfinite(out["loss"]) and out["hidden_voxels"] > 0
    out["mim"].backward()
    selected = model.backbone.patch_embed_2d if mode == "2d" else model.backbone.patch_embed
    nonzero_gradient(selected)
    selected_output = model.decoder.output_2d if mode == "2d" else model.decoder.net[3]
    unused_output = model.decoder.net[3] if mode == "2d" else model.decoder.output_2d
    nonzero_gradient(selected_output)
    assert all(parameter.grad is None for parameter in unused_output.parameters())
    from anatobind.aur.unified import inactive_adapter_parameters
    inactive_ids = {id(parameter) for parameter in inactive_adapter_parameters(model, mode)}
    assert {id(parameter) for parameter in unused_output.parameters()} <= inactive_ids
    for stage in model.backbone.stages:
        nonzero_gradient(stage.blocks)
        if stage.merge is not None:
            nonzero_gradient(stage.merge)
    from anatobind.aur.ssl.masking import foreground_patches, hidden_voxels, batch_masks
    patch = model.backbone.patch_for(mode)
    fg = foreground_patches(batch["view1"], batch["valid"], patch)
    hidden = batch_masks(fg, torch.Generator().manual_seed(2), ratio=.4, block_mm=(4., 8.), patch=patch)
    voxels = hidden_voxels(hidden, patch)
    image = batch["view1"].clone()
    image[:, 0][voxels] += 10
    model.eval()
    with torch.no_grad():
        original, _, _ = model.view(batch["view1"], batch["valid"], batch["coords"], batch["local"], hidden, input_mode=mode)
        changed, _, _ = model.view(image, batch["valid"], batch["coords"], batch["local"], hidden, input_mode=mode)
    assert original.shape == batch["view1"].shape and torch.equal(original, changed)


def test_dual_ssl_masks_use_each_samples_physical_spacing():
    from anatobind.aur.unified import sample_block_masks
    foreground = torch.ones(2, 1, 16, 16, dtype=torch.bool)
    masks = sample_block_masks(foreground, torch.Generator().manual_seed(11), torch.tensor([[1., 1., 1.], [1., 2., 2.]]),
                               patch=(1, 4, 4), ratio=.002, block_mm=(8., 8.))
    assert masks[0].sum() >= 2 and masks[1].sum() == 1
    with pytest.raises(ValueError):
        sample_block_masks(foreground, torch.Generator(), torch.tensor([[1., 0., 1.], [1., 1., 1.]]), patch=(1, 4, 4))


def test_inactive_adapters_do_not_move_after_previous_adam_moments():
    from anatobind.aur.unified import inactive_adapter_parameters
    model = AnatoBindBrain(**MODEL, dual_input=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.1)
    for mode, shape in (("2d", (1, 32, 32)), ("3d", (8, 32, 32))):
        optimizer.zero_grad(set_to_none=True)
        out = model(*inputs(shape), input_mode=mode)
        model.entity_masks(out).square().mean().backward()
        inactive = list(inactive_adapter_parameters(model, mode))
        snapshots = [parameter.detach().clone() for parameter in inactive]
        # Existing supervised training deliberately zero-touches otherwise unused parameters.
        for parameter in inactive:
            parameter.grad = torch.zeros_like(parameter)
        for parameter in inactive_adapter_parameters(model, mode):
            parameter.grad = None
        optimizer.step()
        assert all(torch.equal(before, after) for before, after in zip(snapshots, inactive))
