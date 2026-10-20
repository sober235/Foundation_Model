"""Explicit safe legacy Stage I initialization, never optimizer/RNG resume."""
import hashlib
from collections.abc import Mapping

import torch

from anatobind.aur.ssl.checkpoint import file_sha256
from anatobind.aur.ssl.model import DEFAULTS as MODEL_DEFAULTS

_NEW = {"backbone.patch_embed_2d.weight", "backbone.patch_embed_2d.bias",
        "decoder.output_2d.weight", "decoder.output_2d.bias"}


def _state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(f"{key}:{tuple(tensor.shape)}:{tensor.dtype}:".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_config(model, metadata):
    from anatobind.aur.ssl.train import DEFAULTS as TRAIN_DEFAULTS
    if not isinstance(metadata, Mapping) or metadata.get("stage", "I") != "I":
        raise ValueError("warm-start requires Stage I metadata")
    config = metadata.get("config")
    if not isinstance(config, Mapping) or set(config) - (set(MODEL_DEFAULTS) | set(TRAIN_DEFAULTS)):
        raise ValueError("warm-start checkpoint config is missing or contains unexpected options")
    if not {"embed", "depths", "heads"} <= set(config):
        raise ValueError("warm-start checkpoint config lacks architecture options")
    for key in MODEL_DEFAULTS:
        if key in ("dual_input", "use_checkpoint"):
            continue
        source = config.get(key, MODEL_DEFAULTS[key])
        target = model.cfg[key]
        if isinstance(target, (tuple, list)):
            source = tuple(source) if isinstance(source, (tuple, list)) else source
            target = tuple(target)
        if source != target:
            raise ValueError(f"warm-start checkpoint config mismatch for {key}")
    if config.get("dual_input", False):
        raise ValueError("warm-start accepts a legacy 3d checkpoint, not a unified resume")
    for key in ("step", "seen_crops"):
        value = metadata.get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("warm-start historical counters must be nonnegative integers")


def warm_start_stage_one(model, checkpoint, policy="full_stage_one", seed=20261019):
    """Safely copy a complete legacy SSL model, or an explicit backbone export.

    The trusted project TorchVersion scalar is the only added pickle allowlist
    entry. No checkpoint optimizer, scheduler or RNG is applied. Validation of
    the complete state/config finishes before any target parameter changes.
    Every missing 2D adapter is reset with the explicit seed independently of
    target construction and without advancing the caller's CPU RNG stream.
    """
    if policy not in ("full_stage_one", "backbone_only"):
        raise ValueError("unknown Stage I warm-start policy")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("warm-start seed must be a nonnegative integer")
    if not model.cfg.get("dual_input"):
        raise ValueError("warm-start target must be a unified Stage I model")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("warm-start initialization must occur on CPU before moving the model")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    key = "model" if policy == "full_stage_one" else "backbone_state_dict"
    allowed = {"model", "meta", "optimizer", "scheduler", "rng"} if policy == "full_stage_one" else {"backbone_state_dict", "meta"}
    if not isinstance(saved, Mapping) or key not in saved or set(saved) - allowed:
        raise ValueError("checkpoint format does not match the explicit warm-start policy")
    _validate_config(model, saved.get("meta"))
    if not isinstance(saved[key], Mapping):
        raise ValueError("warm-start source state must be a tensor mapping")
    source = dict(saved[key])
    if policy == "backbone_only":
        source = {"backbone." + key: value for key, value in source.items()}
    own = model.state_dict()
    expected = _NEW if policy == "full_stage_one" else {key for key in own if not key.startswith("backbone.")} | {key for key in _NEW if key.startswith("backbone.")}
    missing, extra = set(own) - set(source), set(source) - set(own)
    wrong = [key for key in set(own) & set(source) if not isinstance(source[key], torch.Tensor)
             or own[key].shape != source[key].shape or own[key].dtype != source[key].dtype
             or source[key].layout != torch.strided or not torch.isfinite(source[key]).all()]
    if missing != expected or extra or wrong:
        raise RuntimeError("warm-start state has unexpected missing/extra keys or invalid tensor shape/dtype/values")
    checkpoint_hash = file_sha256(checkpoint)
    source_hash = _state_hash(source)
    # Reset only complete leaf modules whose tensors were explicitly absent.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for name, module in sorted(model.named_modules()):
            keys = {f"{name}.{key}" for key in module.state_dict()}
            if keys and keys <= expected and hasattr(module, "reset_parameters"):
                module.reset_parameters()
    model.load_state_dict({**model.state_dict(), **source}, strict=True)
    copied = model.state_dict()
    if not all(torch.equal(copied[key], value) for key, value in source.items()):
        raise RuntimeError("warm-start copied tensors failed exact value verification")
    metadata = saved["meta"]
    return {"kind": "legacy_stage_one_warm_start", "policy": policy, "format_version": 1,
            "checkpoint_sha256": checkpoint_hash, "source_state_sha256": source_hash,
            "copied_keys": sorted(source), "initialized_keys": sorted(expected), "initialization_seed": seed,
            "initialization": "seeded_reset_parameters_for_absent_modules",
            "initialized_state_sha256": _state_hash({key: model.state_dict()[key] for key in expected}),
            "historical_step": metadata.get("step", 0), "historical_seen_crops": metadata.get("seen_crops", 0),
            "step": 0, "seen_crops": {"2d": 0, "3d": 0}, "optimizer_reset": True, "scheduler_reset": True,
            "exact_resume": False, "legacy_complete_rng_restored": False}
