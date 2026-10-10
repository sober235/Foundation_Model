# tests/test_aur_ssl_heads.py
import pytest
import torch

import anatobind.aur.ssl.heads as H


def test_patches_and_voxels_are_inverse_rearrangements():
    x = torch.arange(2 * 1 * 4 * 8 * 8, dtype=torch.float32).view(2, 1, 4, 8, 8)
    p = H.voxels_to_patches(x)
    assert p.shape == (2, 32, 2, 2, 2) and torch.equal(H.patches_to_voxels(p), x)
    assert p[0, 0, 0, 0, 0] == x[0, 0, 0, 0, 0] and p[0, 1, 0, 0, 0] == x[0, 0, 0, 0, 1] and p[0, 4, 0, 0, 0] == x[0, 0, 0, 1, 0] and p[0, 16, 0, 0, 0] == x[0, 0, 1, 0, 0]


def _levels(b=2, channels=(8, 16, 32, 64), grids=((3, 5, 5), (2, 3, 3), (1, 2, 2), (1, 1, 1))):
    return [{"feat": torch.randn(b, c, *g, requires_grad=True)} for c, g in zip(channels, grids)]


def test_the_decoder_reads_every_level_and_predicts_a_voxel_image():
    """The 2026-10-09 G1 failure: a decoder on F1 alone left stages 2-4 without any reconstruction gradient."""
    dec = H.MaskedPatchDecoder(channels=(8, 16, 32, 64))
    levels = _levels()
    out = dec(levels)
    assert out.shape == (2, 1, 6, 20, 20)
    out.square().sum().backward()
    for i, lv in enumerate(levels):
        assert lv["feat"].grad is not None and lv["feat"].grad.abs().sum() > 0, f"level {i + 1} gets no gradient"
    with pytest.raises(ValueError, match="levels"):
        dec(levels[:3])


def test_mim_loss_reads_the_weighted_voxels_only_and_normalises_per_sample():
    target = torch.zeros(2, 1, 4, 8, 8)
    pred = target.clone()
    pred[0, 0, 0, 0, 0] = 10.0                                              # a wrong voxel
    weight = torch.zeros(2, 4, 8, 8)
    weight[1] = 1.0                                                          # the wrong voxel carries no weight
    loss, n = H.mim_loss(pred, target, weight)
    assert loss == 0.0 and n == 4 * 8 * 8
    weight[0, 0, 0, 0] = 1.0
    loss2, _ = H.mim_loss(pred, target, weight)
    assert torch.isclose(loss2, torch.tensor((10.0 - 0.5) / 2))               # Huber 9.5 on sample 0 (one voxel), 0 on sample 1
    pred.requires_grad_()
    zero, n0 = H.mim_loss(pred, target, torch.zeros(2, 4, 8, 8))
    assert zero == 0.0 and n0 == 0
    zero.backward()
    assert pred.grad is not None
    half = torch.zeros(2, 4, 8, 8)
    half[:, :2] = 1.0                                                        # the upper half hidden, the lower half visible
    base = H.interpolation_baseline(target + 1.0, half, torch.ones(2, 4, 8, 8))
    assert base == 0.0                                                       # a constant image is predicted by its visible mean
    ramp = torch.arange(4.0).view(1, 1, 4, 1, 1).expand(2, 1, 4, 8, 8)
    assert H.interpolation_baseline(ramp, half, torch.ones(2, 4, 8, 8)) > 0
