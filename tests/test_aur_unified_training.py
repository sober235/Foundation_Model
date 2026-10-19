"""Recoverable single-process technical pilots, with per-instance relation eligibility."""
import copy
import importlib
import random

import numpy as np
import pytest
import torch


MODEL = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4),
             d_model=16, n_events=3, mask_dim=4, pixel_dim=8, dec_layers=1, dec_heads=2,
             rel_layers=1, rel_heads=2, use_checkpoint=False, dual_input=True)
CONTRACT = {"manifest_sha256": "1" * 64, "geometry_version": 1, "split": "TRAIN-only-synthetic"}


def pilot_class():
    try:
        return importlib.import_module("anatobind.aur.unified_train").UnifiedPilot
    except ModuleNotFoundError:
        pytest.fail("The recoverable unified pilot has not been implemented")


def config(**changes):
    cfg = dict(model=MODEL, seed=53, steps=8, lr=1e-3, weight_decay=.1, points=32,
               u_mask_profile="unmatched_empty_v1", mode_cycle=["2d", "3d"], pilot=True,
               ssl=dict(mask_ratio=.4, block_mm=(4., 8.), contrast_weight=0.))
    return {**copy.deepcopy(cfg), **changes}


def factory(mode, draw):
    shape = (1, 16, 16) if mode == "2d" else (4, 16, 16)
    # All global generators affect the draw; a resume must restore all three.
    image = torch.rand(1, 1, *shape) * .8 + random.random() * .05 + float(np.random.rand()) * .05
    axes = [torch.arange(n, dtype=torch.float32) for n in shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"))[None]
    instance = torch.zeros(1, *shape, dtype=torch.long)
    instance[:, :, 2:5, 2:5], instance[:, :, 9:12, 9:12] = 1, 2
    present = torch.zeros(1, 32, dtype=torch.bool)
    present[:, 0] = True
    return dict(image=image, valid=torch.ones(1, *shape), coords=coords, local=coords / 16 - .5,
                entity=torch.ones(1, *shape, dtype=torch.long), entity_present=present,
                instance=instance, point_weight=torch.ones(1, *shape), a_ignore=torch.zeros(1, *shape, dtype=torch.bool),
                seq=torch.tensor([3]), a_supervised=torch.tensor([True]), u_supervised=torch.tensor([True]),
                r_supervised=torch.tensor([True]), n_instances=[2], host=[torch.tensor([0, 3])],
                host_probs=[torch.eye(14)[[0, 3]]], negatives=[torch.full((2, 2), -1, dtype=torch.long)],
                r_eligible=[torch.tensor([True, True])], case=["synthetic"], input_mode=mode,
                spacing=torch.tensor([[2., 1., 1.]]), view1=image.clone(), view2=image + torch.rand_like(image) * .01,
                patient=torch.tensor([0]))


def initialized(stage, cfg=None):
    cls, cfg = pilot_class(), config() if cfg is None else cfg
    first = cls("I", cfg, CONTRACT)
    if stage == "I":
        return first
    second = cls("II", cfg, CONTRACT, initializer=first.export())
    if stage == "II":
        return second
    return cls("III", cfg, CONTRACT, initializer=second.export())


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("stage", ["I", "II", "III"])
def test_eight_steps_equal_three_then_resume_five_in_all_states(tmp_path, stage):
    full = initialized(stage)
    full_logs = [full.step(factory) for _ in range(8)]
    expected = full.save(tmp_path / "full.pt")
    broken = initialized(stage)
    broken_logs = [broken.step(factory) for _ in range(3)]
    saved = broken.save(tmp_path / "split.pt")
    # Loading a checkpoint must undo arbitrary unrelated process RNG consumption.
    torch.rand(7), random.random(), np.random.rand(9)
    resumed = pilot_class().resume(saved, config(), CONTRACT)
    broken_logs.extend(resumed.step(factory) for _ in range(5))
    final = resumed.save(tmp_path / "resumed.pt")
    assert full_logs == broken_logs
    assert full_logs[-1]["exposures"] == {"2d": 4, "3d": 4}
    assert_state_equal(torch.load(expected, weights_only=True), torch.load(final, weights_only=True))
    with pytest.raises(ValueError, match="budget"):
        resumed.step(factory)


@pytest.mark.parametrize("stage", ["I", "II", "III"])
def test_actual_mode_gradients_and_historical_inactive_adapters(stage):
    from anatobind.aur.unified import inactive_adapter_parameters
    runner = initialized(stage)
    shared = tuple(runner.model.backbone.stages.parameters())
    for mode in ("2d", "3d", "2d"):
        inactive = list(inactive_adapter_parameters(runner.model, mode))
        before = [parameter.detach().clone() for parameter in inactive]
        log = runner.step(factory)
        assert log["mode"] == mode and np.isfinite(log["loss"])
        stem = runner.model.backbone.patch_embed_2d if mode == "2d" else runner.model.backbone.patch_embed
        assert stem.weight.grad is not None and stem.weight.grad.abs().sum() > 0
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in shared)
        assert all(parameter.grad is None for parameter in inactive)
        assert all(torch.equal(a, b) for a, b in zip(before, inactive))


def test_stage_two_frozen_parts_never_get_gradients_or_decay():
    from anatobind.aur.train import stage_two_frozen
    runner = initialized("II")
    frozen = stage_two_frozen(runner.model)
    before = [parameter.detach().clone() for parameter in frozen]
    runner.step(factory)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in frozen)
    assert all(torch.equal(a, b) for a, b in zip(before, frozen))


@pytest.mark.parametrize("eligibility,expected_r", [([True, False], 1), ([False, False], 0)])
def test_relation_eligibility_filters_matched_targets_without_losing_u_or_host_zero(eligibility, expected_r):
    runner = initialized("III")

    def mixed(mode, draw):
        batch = factory(mode, draw)
        batch["r_eligible"] = [torch.tensor(eligibility)]
        batch["host"][0][1] = -999  # Unreliable, deliberately not silently recoded as NO_HOST.
        if not eligibility[0]:
            batch["host"][0][0] = -999
        return batch

    log = runner.step(mixed)
    assert log["u_matched_instances"] == 2 and log["r_instances"] == expected_r
    assert np.isfinite(log["loss"])
    assert log["r_host"] > 0 if expected_r else log["r_host"] == 0


def test_stage_inheritance_is_strict_and_exports_are_detached():
    cls = pilot_class()
    one = initialized("I")
    source = one.export()
    two = cls("II", config(), CONTRACT, initializer=source)
    assert_state_equal(one.model.backbone.state_dict(), two.model.backbone.state_dict())
    exported = two.export()
    three = cls("III", config(), CONTRACT, initializer=exported)
    assert_state_equal(two.model.state_dict(), three.model.state_dict())
    exported["model_state_dict"]["backbone.patch_embed.weight"].zero_()
    assert two.model.backbone.patch_embed.weight.abs().sum() > 0
    bad = copy.deepcopy(source)
    bad["backbone_state_dict"].pop("patch_embed_2d.weight")
    with pytest.raises((ValueError, RuntimeError)):
        cls("II", config(), CONTRACT, initializer=bad)
    with pytest.raises(ValueError):
        cls("III", config(), CONTRACT, initializer=source)
    wrong = config()
    wrong["model"]["n_events"] = 4
    with pytest.raises(ValueError, match="config|architecture"):
        cls("II", wrong, CONTRACT, initializer=source)
    with pytest.raises(ValueError):
        cls("II", config(), CONTRACT)


@pytest.mark.parametrize("stage,source_stage", [("II", "I"), ("III", "II")])
def test_stage_initializer_keeps_the_exact_source_partition_contract(stage, source_stage):
    source = initialized(source_stage).export()
    assert source.get("data_contract") == CONTRACT
    changed = {**CONTRACT, "manifest_sha256": "3" * 64}
    with pytest.raises(ValueError, match="contract"):
        pilot_class()(stage, config(), changed, initializer=source)


@pytest.mark.parametrize("damage", ["cursor_keys", "cursor_negative", "cursor_float", "cursor_count", "exposure_keys",
                                   "exposure_negative", "exposure_float", "exposure_below_cursor", "exposure_without_draw"])
def test_resume_rejects_corrupted_draw_cursors_and_exposures(tmp_path, damage):
    runner = initialized("I")
    runner.step(factory)  # Prefix: one 2D draw and no 3D draw.
    checkpoint = torch.load(runner.save(tmp_path / "original.pt"), weights_only=True)
    if damage == "cursor_keys":
        checkpoint["cursors"].pop("3d")
    elif damage == "cursor_negative":
        checkpoint["cursors"]["2d"] = -1
    elif damage == "cursor_float":
        checkpoint["cursors"]["2d"] = 1.
    elif damage == "cursor_count":
        checkpoint["cursors"]["2d"] = 2
    elif damage == "exposure_keys":
        checkpoint["exposures"]["4d"] = 0
    elif damage == "exposure_negative":
        checkpoint["exposures"]["2d"] = -1
    elif damage == "exposure_float":
        checkpoint["exposures"]["2d"] = 1.
    elif damage == "exposure_below_cursor":
        checkpoint["exposures"]["2d"] = 0
    else:
        checkpoint["exposures"]["3d"] = 1
    path = tmp_path / "corrupted.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="cursor|exposure|position"):
        pilot_class().resume(path, config(), CONTRACT)


def test_explicit_legacy_backbone_initialization_records_migration_lineage():
    from anatobind.aur.swin import SwinBackbone
    legacy = SwinBackbone(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4))
    runner = pilot_class()("I", config(), CONTRACT, initializer={"stage": "legacy_backbone",
        "backbone_state_dict": legacy.state_dict(), "lineage": {"source": "synthetic-registered"}})
    audit = runner.export()["lineage"]
    assert audit["kind"] == "legacy_backbone_migration" and audit["migration"]["initialized_keys"] == ["patch_embed_2d.bias", "patch_embed_2d.weight"]
    assert_state_equal(legacy.patch_embed.state_dict(), runner.model.backbone.patch_embed.state_dict())


@pytest.mark.parametrize("change", ["seed", "steps", "mode_cycle", "u_mask_profile", "manifest"])
def test_resume_refuses_changed_configuration_or_data(tmp_path, change):
    runner = initialized("I")
    path = runner.save(tmp_path / "checkpoint.pt")
    cfg, contract = config(), copy.deepcopy(CONTRACT)
    if change == "manifest":
        contract["manifest_sha256"] = "2" * 64
    elif change == "mode_cycle":
        cfg[change] = ["3d", "2d"]
    elif change == "u_mask_profile":
        cfg[change] = "legacy"
    else:
        cfg[change] += 1
    with pytest.raises(ValueError, match="contract|config"):
        pilot_class().resume(path, cfg, contract)


def test_fresh_atomic_save_and_safe_deserialization(tmp_path, monkeypatch):
    runner = initialized("I")
    path = tmp_path / "checkpoint.pt"
    runner.save(path)
    content = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.save(path)
    assert path.read_bytes() == content
    assert torch.load(path, weights_only=True)["stage"] == "I"

    def interrupted_save(payload, stream):
        stream.write(b"interrupted")
        raise OSError("synthetic interruption")

    monkeypatch.setattr(torch, "save", interrupted_save)
    incomplete = tmp_path / "incomplete.pt"
    with pytest.raises(OSError):
        runner.save(incomplete)
    assert not incomplete.exists() and sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint.pt"]


@pytest.mark.parametrize("field", ["torch_version", "default_dtype", "num_threads", "num_interop_threads",
                                   "deterministic_algorithms", "deterministic_warn_only", "cuda"])
def test_resume_refuses_tampered_numerical_environment(tmp_path, field):
    runner = initialized("I")
    original = runner.save(tmp_path / "original.pt")
    checkpoint = torch.load(original, weights_only=True)
    environment = checkpoint.get("numerical_environment")
    assert isinstance(environment, dict), "checkpoint must record its numerical backend contract"
    environment[field] = "synthetic-tampering"
    corrupted = tmp_path / "changed.pt"
    torch.save(checkpoint, corrupted)
    with pytest.raises(ValueError, match="numerical|environment"):
        pilot_class().resume(corrupted, config(), CONTRACT)


def test_resume_refuses_changed_live_dtype_and_thread_settings(tmp_path):
    runner = initialized("I")
    checkpoint = runner.save(tmp_path / "original.pt")
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match="numerical|environment"):
            pilot_class().resume(checkpoint, config(), CONTRACT)
        torch.set_default_dtype(dtype)
        torch.set_num_threads(1 if threads != 1 else 2)
        with pytest.raises(ValueError, match="numerical|environment"):
            pilot_class().resume(checkpoint, config(), CONTRACT)
    finally:
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)


def test_nonpilot_distributed_bad_mode_and_capacity_are_rejected(monkeypatch):
    cls = pilot_class()
    with pytest.raises(ValueError, match="pilot"):
        cls("I", config(pilot=False), CONTRACT)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single|distributed"):
        cls("I", config(), CONTRACT)
    monkeypatch.setenv("WORLD_SIZE", "1")
    runner = initialized("II")
    with pytest.raises(ValueError, match="mode"):
        runner.step(lambda mode, draw: factory("3d", draw))

    def overflow(mode, draw):
        batch = factory(mode, draw)
        batch["n_instances"] = [4]
        return batch

    with pytest.raises(ValueError, match="capacity"):
        runner.step(overflow)
