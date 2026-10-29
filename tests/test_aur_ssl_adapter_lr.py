"""The optional Stage I profile changes only the four 2D IO learning rates."""
from copy import deepcopy
from pathlib import Path
import tempfile

import pytest
import torch

from anatobind.aur.unified_run import UnifiedRun, validate_run_config
from test_aur_unified_run import CONTRACT, config, factory
from test_aur_unified_training import assert_state_equal


IO_2D = (
    "backbone.patch_embed_2d.weight", "backbone.patch_embed_2d.bias",
    "decoder.output_2d.weight", "decoder.output_2d.bias",
)


def test_omitted_profile_keeps_historical_canonical_configuration():
    original = config()
    normalized = validate_run_config(original)
    assert "ssl_2d_lr_scale" not in original
    assert "ssl_2d_lr_scale" not in normalized
    assert normalized == validate_run_config(normalized)
    missing = config()
    missing.pop("steps")
    with pytest.raises(ValueError, match="configuration"):
        validate_run_config(missing)
    with pytest.raises(ValueError, match="configuration"):
        validate_run_config(config(unknown_optional=1))


@pytest.mark.parametrize("scale", [1, 1., 5, .5])
def test_positive_numeric_profile_is_preserved(scale):
    assert validate_run_config(config(ssl_2d_lr_scale=scale))["ssl_2d_lr_scale"] == scale


@pytest.mark.parametrize("scale", [True, False, 0, -1, float("nan"), float("inf"),
                                  float("-inf"), "5", None, [], {}])
def test_invalid_profile_is_rejected(scale):
    with pytest.raises(ValueError, match="configuration"):
        validate_run_config(config(ssl_2d_lr_scale=scale))


@pytest.mark.parametrize("stage", ["II", "III"])
def test_nonunit_profile_is_stage_one_only(stage):
    # Missing initializer cannot mask a failure to enforce the recipe boundary.
    with pytest.raises(ValueError, match="Stage I"):
        UnifiedRun(stage, config(ssl_2d_lr_scale=5), CONTRACT)


def test_explicit_one_matches_legacy_model_rng_optimizer_and_actual_steps(tmp_path):
    legacy = UnifiedRun("I", config(), CONTRACT)
    before = deepcopy(legacy.model.state_dict())
    rng = legacy._rng()
    legacy_logs = [legacy.step(factory) for _ in range(4)]
    expected = torch.load(legacy.save(tmp_path / "legacy.pt"), weights_only=True)

    unit = UnifiedRun("I", config(ssl_2d_lr_scale=1), CONTRACT)
    assert_state_equal(before, unit.model.state_dict())
    assert_state_equal(rng, unit._rng())
    groups = unit.optimizer.param_groups
    assert [g["name"] for g in groups] == ["ssl"]
    assert [id(p) for p in groups[0]["params"]] == [id(p) for p in unit.model.parameters()]
    assert legacy_logs == [unit.step(factory) for _ in range(4)]
    actual = torch.load(unit.save(tmp_path / "unit.pt"), weights_only=True)
    assert actual["cfg"].pop("ssl_2d_lr_scale") == 1
    assert_state_equal(expected, actual)


@pytest.mark.parametrize("lr", [1e-3, {"ssl": 1e-3}])
def test_scale_five_has_exact_io_partition_and_same_schedule_factor(lr):
    run = UnifiedRun("I", config(lr=lr, ssl_2d_lr_scale=5), CONTRACT)
    groups = run.optimizer.param_groups
    assert [g["name"] for g in groups] == ["ssl", "ssl_2d"]
    named = dict(run.model.named_parameters())
    adapter_ids = {id(named[name]) for name in IO_2D}
    assert [id(p) for p in groups[1]["params"]] == [id(named[name]) for name in IO_2D]
    assert [id(p) for p in groups[0]["params"]] == [
        id(p) for p in run.model.parameters() if id(p) not in adapter_ids]
    all_ids = [id(p) for g in groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids)) == len(named)
    assert set(all_ids) == {id(p) for p in named.values()}
    assert run.scheduler.base_lrs == [1e-3, 5e-3]
    for _ in range(4):
        assert groups[1]["lr"] == pytest.approx(5 * groups[0]["lr"])
        assert run.scheduler.lr_lambdas[0](run.step_index) == run.scheduler.lr_lambdas[1](run.step_index)
        run.step(factory)
    assert groups[0]["lr"] == groups[1]["lr"] == 0


def test_scale_five_alternating_steps_preserve_inactive_io_and_update_active_io():
    run = UnifiedRun("I", config(ssl_2d_lr_scale=5), CONTRACT)
    named = dict(run.model.named_parameters())
    io_3d = ("backbone.patch_embed.weight", "backbone.patch_embed.bias",
             "decoder.net.3.weight", "decoder.net.3.bias")
    for mode in ("2d", "3d", "2d", "3d"):
        active, inactive = (IO_2D, io_3d) if mode == "2d" else (io_3d, IO_2D)
        before = {name: named[name].detach().clone() for name in (*active, *inactive)}
        assert run.step(factory)["mode"] == mode
        for name in inactive:
            assert named[name].grad is None
            assert torch.equal(before[name], named[name])
        for name in active:
            assert named[name].grad is not None and named[name].grad.abs().sum() > 0
        # The final scheduler rate becomes zero after the last update.
        assert any(not torch.equal(before[name], named[name]) for name in active)


def test_scale_five_resume_restores_actual_steps_all_state_and_group_order(tmp_path):
    cfg = config(ssl_2d_lr_scale=5)
    full = UnifiedRun("I", cfg, CONTRACT)
    expected_logs = [full.step(factory) for _ in range(4)]
    expected = torch.load(full.save(tmp_path / "full.pt"), weights_only=True)
    split = UnifiedRun("I", cfg, CONTRACT)
    logs = [split.step(factory)]
    path = split.save(tmp_path / "split.pt")
    torch.rand(11)
    restored = UnifiedRun.resume(path, cfg, CONTRACT)
    assert [g["name"] for g in restored.optimizer.param_groups] == ["ssl", "ssl_2d"]
    logs.extend(restored.step(factory) for _ in range(3))
    assert logs == expected_logs
    assert_state_equal(expected, torch.load(restored.save(tmp_path / "resumed.pt"), weights_only=True))
    for changed in (config(), config(ssl_2d_lr_scale=1), config(ssl_2d_lr_scale=4)):
        with pytest.raises(ValueError, match="configuration"):
            UnifiedRun.resume(path, changed, CONTRACT)


def test_historical_checkpoint_and_completion_certificate_stay_valid_without_profile(tmp_path, monkeypatch):
    from functools import partial
    from anatobind.aur.data_contract import file_sha256, private_output
    import anatobind.aur.unified_artifacts as artifacts
    from anatobind.aur.unified_artifacts import verify_completion_certificate
    from anatobind.aur.unified_train import _state_hash

    # Keep the real containment/Git checks, with a portable synthetic root.
    private_root = tmp_path / "private_home"
    private_root.mkdir(mode=0o700)
    validate = partial(private_output, private_root=private_root)
    monkeypatch.setattr(artifacts, "validate_private_output", validate)
    for rejected in (private_root, tmp_path / "outside"):
        with pytest.raises(ValueError, match="private home"):
            validate(rejected)

    # Synthetic boundaries use the actual private-path validator and writer.
    # Relabelling this fixture exercises the certificate schema, not formal qualification.
    run = UnifiedRun("I", config(), CONTRACT)
    with tempfile.TemporaryDirectory(prefix="aur-lr-certificate-test-", dir=private_root) as directory:
        directory = Path(directory)
        run.step(factory)
        old_path = run.save(directory / "old.pt")
        restored = UnifiedRun.resume(old_path, config(), CONTRACT)
        assert "ssl_2d_lr_scale" not in restored.cfg
        for _ in range(3):
            restored.step(factory)
        final_path = Path(restored.save(directory / "final.pt"))
        final = torch.load(final_path, weights_only=True)
        assert "ssl_2d_lr_scale" not in final["cfg"]
        final["purpose"] = final["cfg"]["purpose"] = "formal"
        final["execution_id"] = "4" * 64
        torch.save(final, final_path)
        initializer = deepcopy(final)
        initializer.pop("cfg")
        initializer["completion_certificate"] = dict(
            version=1, stage="I", purpose="formal", executed_steps=4, budget_steps=4,
            selected_step=4, execution_id="4" * 64, data_contract=deepcopy(CONTRACT),
            selected_model_state_sha256=_state_hash(final["model_state_dict"]),
            selected_backbone_state_sha256=_state_hash(final["backbone_state_dict"]),
            final_checkpoint_path=str(final_path), final_checkpoint_sha256=file_sha256(final_path),
            selected_checkpoint_path=str(final_path), selected_checkpoint_sha256=file_sha256(final_path),
        )
        assert verify_completion_certificate(initializer, data_contract=CONTRACT)
        with pytest.raises(ValueError, match="configuration"):
            UnifiedRun.resume(old_path, config(ssl_2d_lr_scale=5), CONTRACT)
