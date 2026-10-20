"""Explicit and atomic trusted legacy Stage I warm-start policies."""
import copy
import importlib
import pickle
from types import SimpleNamespace

import pytest
import torch

from anatobind.aur.ssl.model import StageOne

SMALL = dict(embed=16, depths=(1, 1, 1, 1), heads=(1, 2, 4, 8), window=(2, 4, 4))
NEW = {"backbone.patch_embed_2d.weight", "backbone.patch_embed_2d.bias", "decoder.output_2d.weight", "decoder.output_2d.bias"}


def api():
    return importlib.import_module("anatobind.aur.ssl.unified_init")


def legacy_checkpoint(tmp_path, policy="full_stage_one"):
    source = StageOne(**SMALL)
    meta = dict(config=source.cfg, stage="I", step=8500, seen_crops=306000)
    if policy == "full_stage_one":
        checkpoint = dict(model=source.state_dict(), meta=meta, optimizer={"historical": 1}, scheduler={"historical": 1}, rng={"torch": torch.get_rng_state()})
    else:
        checkpoint = dict(backbone_state_dict=source.backbone.state_dict(), meta=meta)
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)
    return source, checkpoint, path


def test_full_warm_start_copies_all_legacy_heads_and_seeds_exactly_four_new_tensors(tmp_path):
    source, checkpoint, path = legacy_checkpoint(tmp_path)
    target = StageOne(**SMALL, dual_input=True)
    before_rng = torch.get_rng_state().clone()
    lineage = api().warm_start_stage_one(target, path)
    assert torch.equal(before_rng, torch.get_rng_state())
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)
    assert set(lineage["copied_keys"]) == set(source.state_dict())
    assert set(lineage["initialized_keys"]) == NEW
    assert lineage["initialization_seed"] == 20261019
    assert lineage["historical_step"] == 8500 and lineage["historical_seen_crops"] == 306000
    assert lineage["step"] == 0 and lineage["seen_crops"] == {"2d": 0, "3d": 0}
    assert lineage["optimizer_reset"] and lineage["scheduler_reset"] and not lineage["exact_resume"]
    assert not lineage["legacy_complete_rng_restored"] and len(lineage["checkpoint_sha256"]) == 64
    other = StageOne(**SMALL, dual_input=True)
    api().warm_start_stage_one(other, path)
    assert all(torch.equal(target.state_dict()[key], other.state_dict()[key]) for key in NEW)
    third = StageOne(**SMALL, dual_input=True)
    api().warm_start_stage_one(third, path, seed=2)
    assert any(not torch.equal(target.state_dict()[key], third.state_dict()[key]) for key in NEW)


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "dtype", "nonfinite", "config", "unknown_config", "stage", "top_level"])
def test_incompatible_checkpoint_refuses_before_any_target_mutation(tmp_path, damage):
    _, checkpoint, path = legacy_checkpoint(tmp_path)
    key = "decoder.net.3.weight"
    if damage == "missing":
        checkpoint["model"].pop(key)
    elif damage == "extra":
        checkpoint["model"]["foreign"] = torch.zeros(1)
    elif damage == "shape":
        checkpoint["model"][key] = torch.zeros(1)
    elif damage == "dtype":
        checkpoint["model"][key] = checkpoint["model"][key].double()
    elif damage == "nonfinite":
        checkpoint["model"][key].flatten()[0] = float("nan")
    elif damage == "config":
        checkpoint["meta"]["config"]["window"] = (4, 8, 8)
    elif damage == "unknown_config":
        checkpoint["meta"]["config"]["foreign"] = True
    elif damage == "stage":
        checkpoint["meta"]["stage"] = "II"
    else:
        checkpoint["foreign"] = True
    torch.save(checkpoint, path)
    target = StageOne(**SMALL, dual_input=True)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises((ValueError, RuntimeError)):
        api().warm_start_stage_one(target, path)
    assert all(torch.equal(before[key], target.state_dict()[key]) for key in before)


def test_backbone_export_requires_explicit_policy_and_records_new_heads(tmp_path):
    source, checkpoint, path = legacy_checkpoint(tmp_path, "backbone_only")
    target = StageOne(**SMALL, dual_input=True)
    with pytest.raises(ValueError, match="policy"):
        api().warm_start_stage_one(target, path)
    lineage = api().warm_start_stage_one(target, path, policy="backbone_only")
    assert set(lineage["copied_keys"]) == {"backbone." + key for key in source.backbone.state_dict()}
    assert set(lineage["initialized_keys"]) == set(target.state_dict()) - set(lineage["copied_keys"])
    assert set(lineage["initialized_keys"]) > NEW
    assert all(torch.equal(target.backbone.state_dict()[key], value) for key, value in source.backbone.state_dict().items())


def test_full_checkpoint_cannot_silently_be_treated_as_backbone_export(tmp_path):
    _, _, path = legacy_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="policy"):
        api().warm_start_stage_one(StageOne(**SMALL, dual_input=True), path, policy="backbone_only")


def test_warm_start_never_falls_back_to_unsafe_pickle_even_for_ignored_optimizer(tmp_path):
    _, checkpoint, path = legacy_checkpoint(tmp_path)
    checkpoint["optimizer"] = SimpleNamespace(foreign_pickle=True)
    torch.save(checkpoint, path)
    target = StageOne(**SMALL, dual_input=True)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(pickle.UnpicklingError):
        api().warm_start_stage_one(target, path)
    assert all(torch.equal(before[key], target.state_dict()[key]) for key in before)
