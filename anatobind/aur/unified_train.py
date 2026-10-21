"""Recoverable alternating-mode I/II/III technical pilots, deliberately single process.

This runner is not a formal training recipe or G1 gate. It reuses the existing supervision, point sampling and stage
freezing; only per-instance R eligibility and the explicitly configured input mode adapt the supervised path.

CPU recovery is tested bitwise. CUDA restores the saved state exactly, while subsequent updates can vary because
PyTorch 2.5.1's trilinear CUDA backward has no deterministic implementation. Backend settings are frozen and checked;
recording them does not claim deterministic future CUDA kernels.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist

from anatobind.aur import losses as L
from anatobind.aur import train as T
from anatobind.aur.model import AnatoBindBrain, DEFAULTS as MODEL_DEFAULTS
from anatobind.aur.ssl.model import StageOne
from anatobind.aur.unified import inactive_adapter_parameters, migrate_legacy_state


FORMAT_VERSION = 1
_BACKBONE_KEYS = ("embed", "depths", "heads", "window", "patch", "use_checkpoint", "dual_input", "planar_merge")
_DEFAULTS = dict(model={}, seed=0, steps=8, lr=1e-3, weight_decay=.05, points=256,
                 u_mask_profile="unmatched_empty_v1", mode_cycle=["2d", "3d"], pilot=True,
                 ssl=dict(mask_ratio=.6, block_mm=[16., 32.], contrast_weight=0.),
                 lambda_r=L.LAMBDA_R, lambda_h=L.LAMBDA_H)


def _builtin(value):
    """Immutable-by-copy, finite, safe JSON configuration and dataset contracts."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _config(cfg):
    if set(cfg) - set(_DEFAULTS):
        raise ValueError("unknown unified pilot configuration keys")
    out = _builtin({**deepcopy(_DEFAULTS), **cfg})
    if set(out["model"]) - set(MODEL_DEFAULTS):
        raise ValueError("unknown model configuration keys")
    if "planar_merge" in out["model"] and type(out["model"]["planar_merge"]) is not bool:
        raise ValueError("planar_merge must be boolean")
    defaults = {k: v for k, v in MODEL_DEFAULTS.items() if k != "planar_merge"}
    out["model"] = _builtin({**defaults, **out["model"]})
    if out["pilot"] is not True or out["model"]["dual_input"] is not True:
        raise ValueError("only an explicit dual_input technical pilot is supported")
    for key in ("steps", "points"):
        if type(out[key]) is not int or out[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(out["seed"]) is not int or out["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(out["mode_cycle"], list) or set(out["mode_cycle"]) != {"2d", "3d"}:
        raise ValueError("mode_cycle must explicitly contain both 2d and 3d")
    if out["u_mask_profile"] not in L.U_MASK_PROFILES:
        raise ValueError("unsupported u_mask_profile")
    if set(out["ssl"]) - set(_DEFAULTS["ssl"]):
        raise ValueError("unknown SSL pilot configuration keys")
    out["ssl"] = {**_DEFAULTS["ssl"], **out["ssl"]}
    if not 0 < out["ssl"]["mask_ratio"] < 1 or out["ssl"]["contrast_weight"] != 0:
        raise ValueError("the technical pilot requires masking and contrast_weight=0")
    if (len(out["ssl"]["block_mm"]) != 2 or not 0 < out["ssl"]["block_mm"][0] <= out["ssl"]["block_mm"][1]
            or out["weight_decay"] < 0 or out["lambda_r"] < 0 or out["lambda_h"] < 0):
        raise ValueError("invalid SSL block or loss/optimizer configuration")
    rates = out["lr"].values() if isinstance(out["lr"], dict) else [out["lr"]]
    if not all(isinstance(rate, (int, float)) and rate > 0 for rate in rates):
        raise ValueError("learning rates must be positive")
    return out


def _state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(f"{key}:{tuple(value.shape)}:{value.dtype}:".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return deepcopy(value)


def _numerical_environment(device):
    environment = {"torch_version": str(torch.__version__), "torch_git_revision": torch.version.git_version,
        "default_dtype": str(torch.get_default_dtype()), "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled, "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
        "cuda": None}
    if device.type == "cuda":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        environment["cuda"] = {"runtime_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cublas_workspace_present": workspace is not None,
            "cublas_workspace_sha256": hashlib.sha256(workspace.encode()).hexdigest() if workspace is not None else None}
    return environment


class UnifiedPilot:
    def __init__(self, stage, cfg, data_contract, device="cpu", initializer=None):
        self._setup(stage, cfg, data_contract, device)
        self._build()
        if stage == "I":
            if initializer is None:
                self.lineage = {"kind": "random", "seed": self.cfg["seed"]}
            elif initializer.get("stage") == "legacy_backbone" and initializer.get("lineage"):
                audit = migrate_legacy_state(self.model.backbone, initializer["backbone_state_dict"])
                self.lineage = {"kind": "legacy_backbone_migration", "source": _builtin(initializer["lineage"]), "migration": audit}
            else:
                raise ValueError("Stage I initializer must be an explicitly registered legacy backbone; resume is separate")
        else:
            previous = "I" if stage == "II" else "II"
            if (initializer is None or initializer.get("format_version") != FORMAT_VERSION
                    or initializer.get("stage") != previous):
                raise ValueError(f"Stage {stage} requires a strict unified Stage {previous} initializer")
            if _builtin(initializer["model_config"]) != self.cfg["model"]:
                raise ValueError("initializer model architecture/configuration mismatch")
            if initializer.get("data_contract") != self.data_contract:
                raise ValueError("initializer source partition/data contract mismatch")
            state = initializer["backbone_state_dict"] if stage == "II" else initializer["model_state_dict"]
            (self.model.backbone if stage == "II" else self.model).load_state_dict(state, strict=True)
            self.lineage = {"kind": f"stage_{previous}_export", "source": deepcopy(initializer["lineage"]),
                            "source_state_sha256": _state_hash(state)}

    def _setup(self, stage, cfg, data_contract, device):
        if stage not in ("I", "II", "III"):
            raise ValueError("stage must be I, II or III")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (dist.is_initialized() and dist.get_world_size() != 1):
            raise ValueError("the unified technical pilot is single process; distributed execution is unsupported")
        self.stage, self.cfg, self.data_contract = stage, _config(cfg), _builtin(data_contract)
        self.device = torch.device(device)
        self.numerical_environment = _numerical_environment(self.device)
        self.step_index = 0
        self.cursors, self.exposures = {"2d": 0, "3d": 0}, {"2d": 0, "3d": 0}

    def _build(self):
        seed = self.cfg["seed"]
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if self.stage == "I":
            self.model = StageOne(**{key: self.cfg["model"][key] for key in _BACKBONE_KEYS if key != "planar_merge" or key in self.cfg["model"]}, **self.cfg["ssl"])
        else:
            self.model = AnatoBindBrain(**self.cfg["model"])
        self.model.to(self.device).train()
        names = ["ssl"] if self.stage == "I" else ["backbone", "heads"] + (["relation"] if self.stage == "III" else [])
        lr = self.cfg["lr"]
        if isinstance(lr, dict) and set(lr) != set(names):
            raise ValueError("learning-rate groups do not match the stage")
        rates = lr if isinstance(lr, dict) else dict.fromkeys(names, float(lr))
        groups = ([{"name": "ssl", "params": list(self.model.parameters()), "lr": rates["ssl"]}] if self.stage == "I"
                  else T.param_groups(self.model, self.stage, rates))
        self.optimizer = torch.optim.AdamW(groups, weight_decay=self.cfg["weight_decay"], betas=(.9, .95))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: T.lr_lambda(step, 0, self.cfg["steps"]))
        self.point_generator = torch.Generator(device=self.device).manual_seed(seed + 101)
        self.mask_generator = torch.Generator().manual_seed(seed + 202)

    @property
    def next_mode(self):
        return self.cfg["mode_cycle"][self.step_index % len(self.cfg["mode_cycle"])]

    def _check_numerical_environment(self):
        if self.numerical_environment != _numerical_environment(self.device):
            raise ValueError("numerical environment changed during the technical pilot")

    def _supervised(self, batch, mode):
        points = L.sample_points(batch["valid"], self.cfg["points"], self.point_generator, instance=batch["instance"])
        # The legacy sample-level R switch is disabled here. U matching remains unchanged and complete.
        base = {**batch, "r_supervised": torch.zeros_like(batch["r_supervised"])}
        forward = lambda *args: self.model(*args, input_mode=mode)
        total, logged, matches, extras = T._forward_losses(self.model, forward, base, points, "II",
            self.cfg["lambda_r"], self.cfg["lambda_h"], self.cfg["u_mask_profile"])
        logged["u_matched_instances"] = sum(int(match[0].numel()) for match in matches if match is not None)
        if self.stage != "III":
            return total, logged
        r_host, r_hard = total.new_zeros(()), total.new_zeros(())
        samples, instances, correct = 0, 0, 0
        for b, match in enumerate(matches):
            if match is None or not bool(batch["r_supervised"][b]):
                continue
            qi, ti = match
            eligible = batch["r_eligible"][b][ti].bool()
            qi, ti = qi[eligible], ti[eligible]
            if not qi.numel():
                continue
            host, negatives = batch["host"][b][ti], batch["negatives"][b][ti]
            if (host < 0).any() or (host >= 14).any() or (negatives < -1).any() or (negatives >= 14).any():
                raise ValueError("eligible relation targets are outside the AUR host contract")
            logits = self.model.bind(extras["out"], b, qi).float()
            parts = L.relation_loss(logits, host, negatives)
            r_host, r_hard = r_host + parts["r_host"], r_hard + parts["r_hard"]
            samples, instances = samples + 1, instances + int(qi.numel())
            correct += int((logits.argmax(1) == host).sum())
        r_host, r_hard = r_host / max(samples, 1), r_hard / max(samples, 1)
        logged.update(r_host=float(r_host), r_hard=float(r_hard), r_samples=samples, r_instances=instances, r_correct=correct)
        return total + self.cfg["lambda_r"] * (r_host + self.cfg["lambda_h"] * r_hard), logged

    def step(self, batch_factory):
        if self.step_index >= self.cfg["steps"]:
            raise ValueError("technical pilot step budget is exhausted")
        self._check_numerical_environment()
        mode, draw = self.next_mode, self.cursors[self.next_mode]
        batch = batch_factory(mode, draw)
        if batch.get("input_mode") != mode:
            raise ValueError("factory batch mode does not match the homogeneous mode schedule")
        if self.stage != "I":
            for b, count in enumerate(batch["n_instances"]):
                if count > self.cfg["model"]["n_events"]:
                    raise ValueError("instances exceed query capacity")
                if tuple(batch["r_eligible"][b].shape) != (count,):
                    raise ValueError("relation eligibility must align with instance targets")
        if not (batch["valid"].flatten(1).sum(1) > 0).all():
            raise ValueError("every observation must contain valid pixels")
        batch = T.to_device(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        if self.stage == "I":
            result = self.model(batch, self.mask_generator, input_mode=mode)
            loss = result["loss"]
            logged = {key: float(result[key]) for key in ("mim", "contrast", "hidden_share", "hidden_voxels")}
        else:
            loss, logged = self._supervised(batch, mode)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite technical pilot loss")
        loss.backward()
        for parameter in inactive_adapter_parameters(self.model, mode):
            parameter.grad = None
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.scheduler.step()
        self.step_index += 1
        self.cursors[mode] += 1
        self.exposures[mode] += int(batch["valid"].shape[0])
        return {**logged, "mode": mode, "step": self.step_index, "draw_index": draw, "loss": float(loss),
                "exposures": dict(self.exposures), "lr": {group["name"]: group["lr"] for group in self.optimizer.param_groups}}

    def export(self):
        """An in-memory stage initializer, never optimizer recovery or a claim of pretrained 2D quality."""
        self._check_numerical_environment()
        return {"format_version": FORMAT_VERSION, "stage": self.stage, "model_config": deepcopy(self.cfg["model"]),
                "model_state_dict": _cpu_copy(self.model.state_dict()), "backbone_state_dict": _cpu_copy(self.model.backbone.state_dict()),
                "lineage": deepcopy(self.lineage), "data_contract": deepcopy(self.data_contract),
                "numerical_environment": deepcopy(self.numerical_environment)}

    def _rng(self):
        algorithm, keys, pos, has_gauss, cached = np.random.get_state()
        return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                "python": random.getstate(), "numpy": {"algorithm": algorithm, "keys": keys.tolist(), "pos": pos,
                    "has_gauss": has_gauss, "cached": cached},
                "points": self.point_generator.get_state(), "masks": self.mask_generator.get_state()}

    def save(self, path):
        """Atomically create a fresh private checkpoint using an exclusive hard link in the same filesystem."""
        path = Path(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError("technical pilot checkpoints cannot be overwritten")
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        payload = {**self.export(), "cfg": deepcopy(self.cfg), "data_contract": deepcopy(self.data_contract),
                   "device_type": self.device.type, "step": self.step_index, "next_mode": self.next_mode,
                   "cursors": dict(self.cursors), "exposures": dict(self.exposures),
                   "optimizer": _cpu_copy(self.optimizer.state_dict()), "scheduler": self.scheduler.state_dict(), "rng": self._rng()}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".aur-unified-", delete=False) as stream:
                temporary = stream.name
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                os.unlink(temporary)
        return str(path)

    @classmethod
    def resume(cls, path, cfg, data_contract, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported unified checkpoint version")
        if checkpoint["cfg"] != _config(cfg) or checkpoint["data_contract"] != _builtin(data_contract):
            raise ValueError("resume configuration or data contract mismatch")
        if checkpoint["device_type"] != torch.device(device).type:
            raise ValueError("resume device type differs from the numerical recovery contract")
        step, cycle = checkpoint.get("step"), checkpoint["cfg"]["mode_cycle"]
        if type(step) is not int or not 0 <= step <= checkpoint["cfg"]["steps"]:
            raise ValueError("invalid saved step position")
        whole, remainder = divmod(step, len(cycle))
        expected_cursors = {mode: whole * cycle.count(mode) + cycle[:remainder].count(mode) for mode in ("2d", "3d")}
        for key in ("cursors", "exposures"):
            values = checkpoint.get(key)
            if (not isinstance(values, dict) or set(values) != {"2d", "3d"}
                    or any(type(value) is not int or value < 0 for value in values.values())):
                raise ValueError(f"invalid saved {key} position")
        if checkpoint["cursors"] != expected_cursors or checkpoint["next_mode"] != cycle[remainder]:
            raise ValueError("saved cursors do not match the mode-cycle position")
        if any(checkpoint["exposures"][mode] < count or (count == 0 and checkpoint["exposures"][mode] != 0)
               for mode, count in expected_cursors.items()):
            raise ValueError("saved exposures do not match draw cursors")
        obj = cls.__new__(cls)
        obj._setup(checkpoint["stage"], cfg, data_contract, device)
        if checkpoint.get("numerical_environment") != obj.numerical_environment:
            raise ValueError("resume numerical environment contract mismatch or missing")
        obj._build()
        obj.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        obj.optimizer.load_state_dict(checkpoint["optimizer"])
        obj.scheduler.load_state_dict(checkpoint["scheduler"])
        obj.lineage = checkpoint["lineage"]
        obj.step_index, obj.cursors, obj.exposures = checkpoint["step"], checkpoint["cursors"], checkpoint["exposures"]
        rng = checkpoint["rng"]
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        random.setstate(rng["python"])
        state = rng["numpy"]
        np.random.set_state((state["algorithm"], np.asarray(state["keys"], dtype=np.uint32), state["pos"], state["has_gauss"], state["cached"]))
        obj.point_generator.set_state(rng["points"])
        obj.mask_generator.set_state(rng["masks"])
        return obj
