"""Original-grid A/U/R inference through one dual-input model."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anatobind.aur.model import AnatoBindBrain


def tiny_model():
    torch.manual_seed(19)
    return AnatoBindBrain(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1),
                         window=(2, 4, 4), d_model=16, n_events=2, mask_dim=4,
                         pixel_dim=8, dec_layers=1, dec_heads=2,
                         rel_layers=1, rel_heads=2, use_checkpoint=False, dual_input=True)


def test_one_model_preserves_input_grid_and_uses_area_only_for_a_real_plane():
    from anatobind.aur.unified_infer import infer_input

    model = tiny_model()
    affine = np.diag([2., 3., 5., 1.])
    for mode, shape in (("2d", (1, 12, 16)), ("3d", (4, 12, 16))):
        source = SimpleNamespace(image_dhw=np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
                                 affine_ras=affine, input_mode=mode, source_refs=("private-reference",),
                                 frame_map=tuple(range(shape[0])))
        result = infer_input(model, source, crop=(1 if mode == "2d" else 4, 16, 16),
                             device="cpu", lesion_threshold=0., presence_threshold=0., score_threshold=0.)
        assert result.anatomy.shape == shape and result.instances.shape == shape
        assert np.array_equal(result.affine_ras, affine)
        assert result.source_refs == source.source_refs and result.frame_map == source.frame_map
        assert result.evidence == "NOT_EVIDENCE" and len(result.lesions) == 1
        lesion = result.lesions[0]
        assert len(lesion["host_probs"]) == 14 and sum(lesion["host_probs"]) == pytest.approx(1.)
        if mode == "2d":
            assert lesion["area_mm2"] == 1152.
            assert "volume_mm3" not in lesion
        else:
            assert lesion["volume_mm3"] == pytest.approx(23040., rel=1e-12)
            assert "area_mm2" not in lesion


def test_plane_component_area_filter_is_independent_of_normal_axis_length():
    from anatobind.aur.unified_infer import plane_instances

    probability = np.zeros((1, 4, 5), np.float32)
    probability[0, 0, 0] = .9
    probability[0, 2, 2:4] = .8
    affine = np.diag([2., 3., 999., 1.])
    instances, records = plane_instances(probability, affine, threshold=.5, min_area_mm2=7.)
    assert instances[0, 0, 0] == 0
    assert len(records) == 1 and records[0]["area_mm2"] == 12.
    assert records[0]["n_voxels"] == 2 and records[0]["score"] == pytest.approx(.8)
    assert "volume_mm3" not in records[0]


def test_three_dimensional_input_uses_legacy_canonical_orientation_and_maps_back():
    from anatobind.aur.unified_infer import infer_input

    model, seen = tiny_model(), []
    hook = model.backbone.patch_embed.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    raw = np.arange(4 * 12 * 16, dtype=np.float32).reshape(4, 12, 16)
    original = np.diag([2., 3., 5., 1.])
    flipped = original.copy()
    flipped[0, 0], flipped[0, 3] = -2., 30.
    sources = [SimpleNamespace(image_dhw=raw, affine_ras=original, input_mode="3d", source_refs=(), frame_map=()),
               SimpleNamespace(image_dhw=raw[:, :, ::-1], affine_ras=flipped, input_mode="3d", source_refs=(), frame_map=())]
    results = [infer_input(model, source, (4, 16, 16), lesion_threshold=1.) for source in sources]
    hook.remove()
    assert len(seen) == 2 and torch.equal(seen[0], seen[1])
    assert np.array_equal(results[0].anatomy, results[1].anatomy[:, :, ::-1])
    assert np.array_equal(results[1].affine_ras, flipped)


@pytest.mark.parametrize("mode,shape,crop", [("2d", (2, 12, 16), (1, 16, 16)),
                                           ("2d", (1, 12, 16), (2, 16, 16)),
                                           ("other", (4, 12, 16), (4, 16, 16))])
def test_inference_refuses_ambiguous_dimensions(mode, shape, crop):
    from anatobind.aur.unified_infer import infer_input

    source = SimpleNamespace(image_dhw=np.ones(shape), affine_ras=np.eye(4), input_mode=mode,
                             source_refs=(), frame_map=())
    with pytest.raises(ValueError):
        infer_input(tiny_model(), source, crop, device="cpu")


def test_unified_export_loads_one_model_for_both_modes_and_refuses_partial_state(tmp_path):
    from anatobind.aur.unified_infer import model_from_unified_export

    model = tiny_model().eval()
    export = dict(format_version=1, stage="III", model_config=model.cfg,
                  model_state_dict=model.state_dict(), lineage={"kind": "synthetic_test"},
                  data_contract={"kind": "synthetic_test"})
    path = tmp_path / "unified.pt"
    torch.save(export, path)
    loaded, metadata = model_from_unified_export(path)
    assert metadata["evidence"] == "NOT_EVIDENCE" and metadata["pilot_only"]
    with torch.no_grad():
        for mode, depth in (("2d", 1), ("3d", 8)):
            shape = (depth, 32, 32)
            image, valid = torch.ones((1, 1, *shape)), torch.ones((1, *shape))
            coords = torch.stack(torch.meshgrid(*[torch.arange(n).float() for n in shape], indexing="ij"))[None]
            expected = model(image, valid, coords, coords, input_mode=mode)
            actual = loaded(image, valid, coords, coords, input_mode=mode)
            assert torch.equal(expected["entity_embed"], actual["entity_embed"])
            assert torch.equal(model.event_masks(expected), loaded.event_masks(actual))
    export["model_state_dict"].pop("backbone.patch_embed_2d.weight")
    broken = tmp_path / "broken.pt"
    torch.save(export, broken)
    with pytest.raises(RuntimeError):
        model_from_unified_export(broken)
    export["stage"] = "I"
    torch.save(export, tmp_path / "ssl.pt")
    with pytest.raises(ValueError):
        model_from_unified_export(tmp_path / "ssl.pt")
