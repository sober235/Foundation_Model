# tests/test_aur_model.py
import pytest
import torch

import anatobind.aur.losses as L
from anatobind.aur.labels import N_ENTITIES, N_HOST_CLASSES
from anatobind.aur.model import AnatoBindBrain

TINY = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4), d_model=16, n_events=4,
            mask_dim=4, pixel_dim=8, dec_layers=1, dec_heads=2, rel_layers=1, rel_heads=2, use_checkpoint=False)


def _batch(shape=(8, 32, 32)):
    torch.manual_seed(0)
    img = torch.randn(2, 1, *shape)
    valid = torch.ones(2, *shape)
    valid[1, :, 20:] = 0.0
    axes = [torch.arange(n, dtype=torch.float32) + 0.5 for n in shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0)[None].expand(2, -1, -1, -1, -1).clone()
    local = torch.stack(torch.meshgrid(*[(a / n) * 2 - 1 for a, n in zip(axes, shape)], indexing="ij"), 0)[None].expand(2, -1, -1, -1, -1).clone()
    return img, valid, coords, local


def test_forward_masks_bind_and_a_full_training_step_on_cpu():
    model = AnatoBindBrain(**TINY)
    img, valid, coords, local = _batch()
    out = model(img, valid, coords, local)
    assert out["entity_embed"].shape == (2, N_ENTITIES, 16) and out["event_presence"].shape == (2, 4) and out["seq_logits"].shape == (2, 6)
    pts = L.sample_points(valid, 64, torch.Generator().manual_seed(0))
    em, um = model.entity_masks(out, pts), model.event_masks(out, pts)
    assert em.shape == (2, N_ENTITIES, 64) and um.shape == (2, 4, 64)
    assert model.entity_masks(out).shape == (2, N_ENTITIES, 8, 32, 32)
    logits = model.bind(out, 1, torch.tensor([0, 2]))
    assert logits.shape == (2, N_HOST_CLASSES) and torch.isfinite(logits).all()
    # a complete loss with synthetic targets
    entity_pts = torch.randint(0, N_ENTITIES + 1, (2, 64))
    present = torch.ones(2, N_ENTITIES, dtype=torch.bool)
    targets = [torch.zeros(2, 64), torch.zeros(0, 64)]
    targets[0][0, :10] = 1.0
    targets[0][1, 20:30] = 1.0
    parts = {}
    parts.update(L.entity_loss(em, out["entity_presence"], entity_pts, torch.zeros(2, 64, dtype=torch.bool), present))
    u, matches = L.event_loss(out["event_presence"], um, targets, torch.ones(2, 64), torch.tensor([True, False]))
    parts.update(u)
    parts.update(L.seq_loss(out["seq_logits"], torch.tensor([3, 0])))
    qi, ti = matches[0]
    parts.update(L.relation_loss(model.bind(out, 0, qi), torch.tensor([0, 5])[ti], torch.tensor([[1, -1], [6, 2]])[ti]))
    total, logged = L.total(parts)
    total.backward()
    assert torch.isfinite(total) and all(torch.isfinite(torch.tensor(v)) for v in logged.values())
    assert model.backbone.patch_embed.weight.grad is not None and model.relation.out.weight.grad is not None
    with pytest.raises(KeyError, match="unknown model options"):
        AnatoBindBrain(nonsense=1)


def test_default_configuration_is_the_plans():
    model = AnatoBindBrain(use_checkpoint=False)
    assert model.backbone.channels == (64, 128, 256, 512) and model.entities.K == 32 and model.events.M == 64
    assert model.cfg["d_model"] == 256 and 20e6 < model.num_parameters() < 30e6 and 10e6 < model.backbone.num_parameters() < 15e6
