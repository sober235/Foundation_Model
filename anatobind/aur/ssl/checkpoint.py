"""Stage I checkpoints (SSL-first plan §3.3): the full resume checkpoint and the backbone-only export.

Two files, two purposes. The resume checkpoint holds the whole Stage I model (backbone, decoder, projector), the
optimizer, the scheduler, the step, the crops seen, the RNG states and the run's metadata; it is what a stopped run
continues from. The export holds only the backbone's state dict and the metadata: it is what Stage II initialises
from, with a strict key-and-shape check (nothing is skipped silently, nothing falls back to a random backbone). Files
are never overwritten: every save goes to a new name."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch


def code_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=str(Path(__file__).resolve().parents[3])).stdout.strip()
    except Exception:
        return "unknown"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(config, manifest, step, seen_crops, seed, extra=None):
    meta = {"config": config, "manifest": str(manifest), "manifest_sha256": file_sha256(manifest) if manifest and Path(manifest).is_file() else None,
            "code_sha": code_sha(), "step": int(step), "seen_crops": int(seen_crops), "seed": int(seed), "torch": torch.__version__,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        meta.update(extra)
    return meta


def _fresh(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; checkpoints are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_resume(path, model, optimizer, scheduler, meta, rng_state=None):
    path = _fresh(path)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None, "meta": meta,
                "rng": rng_state if rng_state is not None else {"torch": torch.get_rng_state()}}, str(path))
    return str(path)


def load_resume(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ck = torch.load(str(path), map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler") is not None:
        scheduler.load_state_dict(ck["scheduler"])
    return ck["meta"], ck.get("rng")


def export_backbone(path, backbone, meta):
    """The backbone-only file Stage II initialises from."""
    path = _fresh(path)
    torch.save({"backbone_state_dict": {k: v.detach().cpu() for k, v in backbone.state_dict().items()}, "meta": meta}, str(path))
    (path.with_suffix(".json")).write_text(json.dumps(meta, indent=1, default=str))
    return str(path)


def load_backbone(path, backbone, map_location="cpu"):
    """Strict load into `backbone`: every key present, every shape equal; missing or unexpected keys raise with the
    lists in the message. Returns the metadata."""
    ck = torch.load(str(path), map_location=map_location, weights_only=False)
    state = ck["backbone_state_dict"]
    own = backbone.state_dict()
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    wrong = sorted(k for k in set(own) & set(state) if tuple(own[k].shape) != tuple(state[k].shape))
    if missing or unexpected or wrong:
        raise RuntimeError(f"backbone checkpoint {path} does not fit: missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)}), shape mismatch {wrong[:5]} ({len(wrong)})")
    backbone.load_state_dict(state, strict=True)
    return ck["meta"]
