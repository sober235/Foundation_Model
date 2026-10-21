"""Explicit mode contracts and audited legacy-to-unified parameter migration.

These helpers operate on model tensors only. They neither deserialize checkpoints nor relax ordinary strict loading.
The legacy 3D modules retain their names; only the listed 2D adapters may be absent during explicit migration.
"""
import hashlib

import torch


def input_patch(patch, dual_input, input_mode):
    """Select the observed patch without changing the model's configured legacy 3D patch."""
    if input_mode not in ("2d", "3d"):
        raise ValueError(f"unsupported input_mode {input_mode!r}; expected '2d' or '3d'")
    if input_mode == "2d" and not dual_input:
        raise ValueError("2d input requires an opt-in dual_input model")
    return (1, *patch[1:]) if input_mode == "2d" else tuple(patch)


def sample_block_masks(foreground, generator, spacing, patch, **kwargs):
    """Draw each sample's SSL mask in its own physical spacing (the opt-in unified path)."""
    from anatobind.aur.ssl.masking import block_mask

    spacing = torch.as_tensor(spacing)
    if tuple(spacing.shape) != (foreground.shape[0], 3) or not torch.isfinite(spacing).all() or not (spacing > 0).all():
        raise ValueError("SSL spacing must be a finite positive (B, 3) tensor")
    return torch.stack([block_mask(foreground[b], generator, spacing_mm=tuple(float(s) for s in spacing[b]),
                                   patch=patch, **kwargs) for b in range(foreground.shape[0])])


# All supported roots: a backbone, a mask/MIM head, or one complete Stage I / AUR model.
_ADAPTER_PAIRS = (("patch_embed", "patch_embed_2d"), ("backbone.patch_embed", "backbone.patch_embed_2d"),
                  ("up", "up_2d"), ("masks.up", "masks.up_2d"),
                  ("net.3", "output_2d"), ("decoder.net.3", "decoder.output_2d"))


def _adapter_pairs(module):
    modules = dict(module.named_modules())
    pairs = [(legacy, new) for legacy, new in _ADAPTER_PAIRS if legacy in modules and new in modules]
    for name in modules:
        if name == "merge_2d" or name.endswith(".merge_2d"):
            legacy = name[:-len("merge_2d")] + "merge"
            if legacy in modules:
                pairs.append((legacy, name))
    return pairs


def inactive_adapter_parameters(module, input_mode):
    """Parameters whose grads a homogeneous optimizer step must leave None, including after older Adam moments.

    Apply this after backward/reduction and before optimizer.step. It does not make alternating-mode DDP supported.
    """
    pairs = _adapter_pairs(module)
    input_patch((2, 4, 4), bool(pairs), input_mode)
    for legacy, new in pairs:
        yield from module.get_submodule(legacy if input_mode == "2d" else new).parameters()


def migrate_legacy_state(module, legacy_state):
    """Strictly copy every legacy tensor, retaining only new 2D adapters' constructor initialization.

    A unified state must use ordinary strict loading instead. Validation finishes before mutation. The audit contains
    architectural keys and a source-tensor digest, never private filenames, input arrays or metadata values.
    """
    pairs = _adapter_pairs(module)
    if not pairs:
        raise ValueError("legacy migration requires a dual_input target with explicit 2D adapters")
    own = module.state_dict()
    expected = {f"{new}.{suffix}" for _, new in pairs for suffix in module.get_submodule(new).state_dict()}
    missing = set(own) - set(legacy_state)
    extra = set(legacy_state) - set(own)
    wrong = [key for key in set(own) & set(legacy_state)
             if not isinstance(legacy_state[key], torch.Tensor) or own[key].shape != legacy_state[key].shape
             or own[key].dtype != legacy_state[key].dtype]
    if missing != expected or extra or wrong:
        raise RuntimeError(f"legacy state does not fit: expected only absent 2D adapters; missing {sorted(missing)}, "
                           f"unexpected {sorted(extra)}, shape/dtype mismatch {sorted(wrong)}")
    digest = hashlib.sha256()
    for key in sorted(legacy_state):
        tensor = legacy_state[key].detach().cpu().contiguous()
        digest.update(f"{key}:{tuple(tensor.shape)}:{tensor.dtype}:".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    module.load_state_dict({**own, **legacy_state}, strict=True)
    return {"migration_version": 1, "copied_keys": sorted(legacy_state), "initialized_keys": sorted(expected),
            "initialization": "target_constructor_values_retained", "state_sha256": digest.hexdigest()}
