"""Exact CUDA checkpoint restoration; subsequent nondeterministic kernels have a declared loss tolerance.

The original stronger future-parameter assertion failed even between identical uninterrupted CUDA runs. See the
recorded cuDNN/trilinear-backward controls; CPU uninterrupted/resumed parameter tests remain bitwise exact.
"""
import copy
from pathlib import Path
import runpy

import pytest
import torch

from anatobind.aur.unified_train import UnifiedPilot


def same_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same_tree(a, b)
    else:
        assert left == right


@pytest.mark.skipif(not torch.cuda.is_available(), reason="unified optimizer/RNG recovery requires CUDA")
@pytest.mark.parametrize("stage", ["I", "II", "III"])
def test_cuda_restores_complete_checkpoint_and_continues_mode_schedule(tmp_path, stage):
    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/aur_unified_pilot.py"))
    cfg = dict(model=copy.deepcopy(script["TINY"]), seed=17, steps=8, lr=5e-4,
               points=64, mode_cycle=["2d", "3d"], pilot=True,
               ssl={"contrast_weight": 0., "mask_ratio": .4, "block_mm": [4., 8.]})
    contract = {"kind": "synthetic_cuda_recovery", "float_rtol": 1e-5, "float_atol": 1e-6}
    initializer = None
    if stage != "I":
        initializer = UnifiedPilot("I", cfg, contract, device="cuda:0").export()
    if stage == "III":
        initializer = UnifiedPilot("II", cfg, contract, device="cuda:0", initializer=initializer).export()
    factory = script["SyntheticViews"](seed=17)
    continuous = UnifiedPilot(stage, cfg, contract, device="cuda:0", initializer=initializer)
    full_logs = [continuous.step(factory) for _ in range(8)]
    continuous.save(tmp_path / "continuous.pt")
    del continuous
    split = UnifiedPilot(stage, cfg, contract, device="cuda:0", initializer=initializer)
    for _ in range(3):
        split.step(factory)
    split.save(tmp_path / "split.pt")
    del split
    resumed = UnifiedPilot.resume(tmp_path / "split.pt", cfg, contract, device="cuda:0")
    resumed.save(tmp_path / "restored.pt")
    same_tree(torch.load(tmp_path / "split.pt", map_location="cpu", weights_only=True),
              torch.load(tmp_path / "restored.pt", map_location="cpu", weights_only=True))
    rest_logs = [resumed.step(factory) for _ in range(5)]
    resumed.save(tmp_path / "resumed.pt")
    for observed, reference in zip(rest_logs, full_logs[3:]):
        assert observed["mode"] == reference["mode"] and observed["exposures"] == reference["exposures"]
        assert observed["loss"] == pytest.approx(reference["loss"], rel=1e-5, abs=1e-6)
    continuous = torch.load(tmp_path / "continuous.pt", map_location="cpu", weights_only=True)
    recovered = torch.load(tmp_path / "resumed.pt", map_location="cpu", weights_only=True)
    for key in ("stage", "cfg", "data_contract", "step", "next_mode", "cursors", "exposures", "scheduler", "rng"):
        same_tree(continuous[key], recovered[key])
    assert continuous["optimizer"]["param_groups"] == recovered["optimizer"]["param_groups"]
    for parameter, state in continuous["optimizer"]["state"].items():
        assert torch.equal(state["step"], recovered["optimizer"]["state"][parameter]["step"])
