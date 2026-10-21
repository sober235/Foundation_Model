"""Original-grid A/U/R inference through one dual-input model."""
from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path

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
    assert metadata["format_version"] == 1 and metadata["phase_completed"] is False
    assert metadata["quality_certified"] is False
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


@pytest.fixture(scope="module", params=["II", "III"])
def supervised_boundaries(request, tmp_path_factory):
    """Real synthetic runner boundaries, following the launcher certificate fixture."""
    from anatobind.aur.unified_run import UnifiedRun
    from test_aur_unified_run import CONTRACT, config, factory

    cfg = config(accumulation_steps=1)
    first = UnifiedRun("I", cfg, CONTRACT)
    runner = UnifiedRun("II", cfg, CONTRACT, initializer=first.export())
    if request.param == "III":
        runner = UnifiedRun("III", cfg, CONTRACT, initializer=runner.export())
    directory = tmp_path_factory.mktemp("supervised_boundaries")
    states = []
    for index in range(4):
        runner.step(factory)
        if index in (1, 3):
            path = runner.save(directory / f"step{index + 1}.pt")
            states.append(torch.load(path, weights_only=True))
    # Synthetic work cannot release a real gate. As in the existing launcher
    # certificate fixture, exercise the formal artifact schema on real states.
    for state in states:
        state["purpose"] = state["cfg"]["purpose"] = "formal"
        state["execution_id"] = "4" * 64
    return states


def certified_export(tmp_path, monkeypatch, boundaries, mutate=None):
    from anatobind.aur import unified_artifacts as artifacts
    from anatobind.aur.data_contract import file_sha256
    from anatobind.aur.unified_train import _state_hash

    # Only bypass the private-home location policy for pytest's synthetic files;
    # checkpoint deserialization, boundary checks and hashes stay real.
    monkeypatch.setattr(artifacts, "validate_private_output", lambda path: Path(path).resolve())
    selected, final = deepcopy(boundaries)
    if mutate is not None:
        mutate(selected, final)
    selected_path, final_path = tmp_path / "selected.pt", tmp_path / "final.pt"
    torch.save(selected, selected_path)
    torch.save(final, final_path)
    exported = deepcopy(selected)
    exported.pop("cfg")
    exported["completion_certificate"] = dict(
        version=1, stage=selected["stage"], purpose="formal", execution_id="4" * 64,
        executed_steps=4, budget_steps=4, selected_step=2,
        selected_model_state_sha256=_state_hash(selected["model_state_dict"]),
        selected_backbone_state_sha256=_state_hash(selected["backbone_state_dict"]),
        data_contract=selected["data_contract"],
        selected_checkpoint_path=str(selected_path.resolve()),
        selected_checkpoint_sha256=file_sha256(selected_path),
        final_checkpoint_path=str(final_path.resolve()), final_checkpoint_sha256=file_sha256(final_path),
        g1_report_sha256=None)
    assert artifacts.verify_completion_certificate(exported, data_contract=selected["data_contract"])
    return exported


def test_formal_export_loads_selected_weights_and_both_mode_aur_outputs(
        tmp_path, monkeypatch, supervised_boundaries):
    from anatobind.aur.unified_infer import model_from_unified_export

    exported = certified_export(tmp_path, monkeypatch, supervised_boundaries)
    path = tmp_path / "formal.pt"
    torch.save(exported, path)
    expected_model = AnatoBindBrain(**exported["model_config"]).eval()
    expected_model.load_state_dict(exported["model_state_dict"], strict=True)
    loaded, metadata = model_from_unified_export(path, data_contract=exported["data_contract"])
    assert not loaded.training
    assert metadata["format_version"] == 2 and metadata["stage"] == exported["stage"]
    assert metadata["pilot_only"] is False and metadata["phase_completed"] is True
    assert metadata["quality_certified"] is False and metadata["evidence"] == "NOT_EVIDENCE"
    assert exported["step"] == 2 and exported["complete"] is False
    assert metadata["data_contract"] == exported["data_contract"]
    # Omitting an external identity must still validate the embedded certificate.
    _, embedded_metadata = model_from_unified_export(path)
    assert embedded_metadata == metadata
    with torch.no_grad():
        for mode, depth in (("2d", 1), ("3d", 4)):
            shape = (depth, 16, 16)
            image, valid = torch.ones((1, 1, *shape)), torch.ones((1, *shape))
            coords = torch.stack(torch.meshgrid(*[torch.arange(n).float() for n in shape], indexing="ij"))[None]
            expected = expected_model(image, valid, coords, coords, input_mode=mode)
            actual = loaded(image, valid, coords, coords, input_mode=mode)
            for key in ("entity_embed", "entity_presence", "event_embed", "event_presence", "seq_logits"):
                assert torch.equal(expected[key], actual[key])
            assert torch.equal(expected_model.entity_masks(expected), loaded.entity_masks(actual))
            assert torch.equal(expected_model.event_masks(expected), loaded.event_masks(actual))
            indices = torch.arange(exported["model_config"]["n_events"])
            assert torch.equal(expected_model.bind(expected, 0, indices), loaded.bind(actual, 0, indices))


@pytest.mark.parametrize("damage", ["contract", "missing_certificate", "certificate_hash", "selected_state",
                                   "missing_boundary", "stage_i", "unsupported_format", "nonformal"])
def test_formal_export_rejects_unverified_identity_and_completion(
        tmp_path, monkeypatch, supervised_boundaries, damage):
    from anatobind.aur.unified_infer import model_from_unified_export

    exported = certified_export(tmp_path, monkeypatch, supervised_boundaries)
    kwargs = {}
    if damage == "contract":
        kwargs["data_contract"] = {**exported["data_contract"], "contract_sha256": "0" * 64}
    elif damage == "missing_certificate":
        exported.pop("completion_certificate")
    elif damage == "certificate_hash":
        exported["completion_certificate"]["final_checkpoint_sha256"] = "0" * 64
    elif damage == "selected_state":
        exported["model_state_dict"]["events.stack.query"] += 1
    elif damage == "missing_boundary":
        (tmp_path / "final.pt").unlink()
    elif damage == "stage_i":
        exported["stage"] = "I"
    elif damage == "unsupported_format":
        exported["format_version"] = 3
    elif damage == "nonformal":
        exported["purpose"] = "resource_probe"
    path = tmp_path / "PRIVATE_ARTIFACT.pt"
    torch.save(exported, path)
    with pytest.raises(ValueError) as error:
        model_from_unified_export(path, device="cuda:999", **kwargs)
    assert "PRIVATE_ARTIFACT" not in str(error.value) and str(tmp_path) not in str(error.value)


@pytest.mark.parametrize("damage", ["partial", "nan", "inf", "overflow", "architecture"])
def test_formal_export_rejects_certificate_bound_invalid_model_before_device_transfer(
        tmp_path, monkeypatch, supervised_boundaries, damage):
    from anatobind.aur.unified_infer import model_from_unified_export

    def mutate(selected, final):
        key = next(key for key in selected["model_state_dict"] if not key.startswith("backbone."))
        if damage == "partial":
            selected["model_state_dict"].pop(key)
        elif damage == "architecture":
            for state in (selected, final):
                state["model_config"]["n_events"] += 1
                state["cfg"]["model"]["n_events"] += 1
        else:
            value = selected["model_state_dict"][key]
            if damage == "overflow":
                value = value.double()
                selected["model_state_dict"][key] = value
            value.view(-1)[0] = {"nan": float("nan"), "inf": float("inf"), "overflow": 1e300}[damage]

    exported = certified_export(tmp_path, monkeypatch, supervised_boundaries, mutate)
    path = tmp_path / "PRIVATE_ARTIFACT.pt"
    torch.save(exported, path)
    def forbid_transfer(*args, **kwargs):
        pytest.fail("invalid model reached device transfer")
    monkeypatch.setattr(AnatoBindBrain, "to", forbid_transfer)
    with pytest.raises((ValueError, RuntimeError)) as error:
        model_from_unified_export(path, device="cuda:999")
    assert "PRIVATE_ARTIFACT" not in str(error.value) and str(tmp_path) not in str(error.value)


@pytest.mark.parametrize("payload", [[], None, {"format_version": True}])
def test_export_loader_rejects_malformed_payloads_without_echoing_paths(tmp_path, payload):
    from anatobind.aur.unified_infer import model_from_unified_export

    path = tmp_path / "PRIVATE_ARTIFACT.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError) as error:
        model_from_unified_export(path)
    assert "PRIVATE_ARTIFACT" not in str(error.value)


@pytest.mark.parametrize("damage", ["missing_file", "unreadable", "contract", "nonfinite", "bad_config"])
def test_pilot_export_preserves_strict_validation_and_private_errors(tmp_path, monkeypatch, damage):
    from anatobind.aur.unified_infer import model_from_unified_export

    model = tiny_model()
    exported = dict(format_version=1, stage="II", model_config=model.cfg,
                    model_state_dict=model.state_dict(), data_contract={"kind": "synthetic_test"})
    path = tmp_path / "PRIVATE_ARTIFACT.pt"
    kwargs = {}
    if damage == "contract":
        kwargs["data_contract"] = {"kind": "different"}
    elif damage == "nonfinite":
        exported["model_state_dict"]["events.stack.query"].view(-1)[0] = float("nan")
    elif damage == "bad_config":
        exported["model_config"]["PRIVATE_OPTION"] = True
    if damage == "unreadable":
        path.write_bytes(b"invalid checkpoint")
    elif damage != "missing_file":
        torch.save(exported, path)
    def forbid_transfer(*args, **kwargs):
        pytest.fail("invalid model reached device transfer")
    monkeypatch.setattr(AnatoBindBrain, "to", forbid_transfer)
    with pytest.raises((ValueError, RuntimeError)) as error:
        model_from_unified_export(path, device="cuda:999", **kwargs)
    assert "PRIVATE_ARTIFACT" not in str(error.value) and "PRIVATE_OPTION" not in str(error.value)
