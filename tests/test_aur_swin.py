# tests/test_aur_swin.py
import torch

import anatobind.aur.swin as S

TINY = dict(embed=32, depths=(1, 2, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4))


def _inputs(shape=(8, 32, 32), valid_until=None, seed=0):
    torch.manual_seed(seed)
    img = torch.randn(2, 1, *shape)
    valid = torch.ones(2, *shape)
    if valid_until is not None:
        valid[:, :, valid_until:] = 0.0                     # the lower rows are padding
        img[:, :, valid_until:] = -1.0
    axes = [torch.arange(n, dtype=torch.float32) + 0.5 for n in shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0)[None].expand(2, -1, -1, -1, -1).clone()
    local = torch.stack(torch.meshgrid(*[(a / n) * 2 - 1 for a, n in zip(axes, shape)], indexing="ij"), 0)[None].expand(2, -1, -1, -1, -1).clone()
    return img, valid, coords, local


def test_four_levels_with_the_plans_strides_channels_and_carried_geometry():
    bb = S.SwinBackbone(**TINY)
    img, valid, coords, local = _inputs()
    levels = bb(img, valid, coords, local)
    assert [tuple(l["feat"].shape) for l in levels] == [(2, 32, 4, 8, 8), (2, 64, 2, 4, 4), (2, 128, 1, 2, 2), (2, 256, 1, 1, 1)]
    assert [tuple(l["valid"].shape[1:]) for l in levels] == [(4, 8, 8), (2, 4, 4), (1, 2, 2), (1, 1, 1)]
    assert tuple(levels[0]["coords"].shape) == (2, 3, 4, 8, 8) and tuple(levels[2]["local"].shape) == (2, 3, 1, 2, 2)
    assert torch.allclose(levels[0]["coords"][0, :, 0, 0, 0], torch.tensor([1.0, 2.0, 2.0]))     # patch centre in mm
    assert torch.allclose(levels[1]["coords"][0, :, 0, 0, 0], torch.tensor([2.0, 4.0, 4.0]))     # merged: mean of 8
    assert bb.channels == (32, 64, 128, 256) and bb.num_parameters() > 0
    full = S.SwinBackbone()
    assert full.channels == S.CHANNELS and 10e6 < full.num_parameters() < 15e6       # 12.9 M measured


def test_padding_never_leaks_into_valid_tokens():
    torch.manual_seed(0)
    bb = S.SwinBackbone(**TINY).eval()
    img, valid, coords, local = _inputs(valid_until=16)
    img2 = img.clone()
    img2[:, :, 16:] = torch.randn_like(img2[:, :, 16:]) * 5                  # change the padded voxels only
    with torch.no_grad():
        a, b = bb(img, valid, coords, local), bb(img2, valid, coords, local)
    for la, lb in zip(a, b):
        v = la["valid"] > 0.5
        assert torch.allclose(la["feat"].permute(0, 2, 3, 4, 1)[v], lb["feat"].permute(0, 2, 3, 4, 1)[v], atol=1e-5)
        assert torch.allclose(la["valid"], lb["valid"])
    assert a[0]["valid"][0, :, 4:].sum() == 0 and a[0]["valid"][0, :, :4].sum() == 4 * 4 * 8
    assert a[1]["valid"][0, :, 2:].sum() == 0 and a[1]["valid"][0, :, :2].sum() == 2 * 2 * 4


def test_shifted_windows_checkpointing_and_odd_shapes():
    bb = S.SwinBackbone(**dict(TINY, depths=(2, 2, 1, 1)), use_checkpoint=True)
    img, valid, coords, local = _inputs(shape=(6, 36, 28))                     # not window multiples: internal padding
    img.requires_grad_(True)
    levels = bb(img, valid, coords, local)
    assert tuple(levels[0]["feat"].shape[2:]) == (3, 9, 7) and tuple(levels[1]["feat"].shape[2:]) == (2, 5, 4)
    levels[3]["feat"].sum().backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()
    try:
        bb(torch.randn(1, 1, 7, 32, 32), torch.ones(1, 7, 32, 32), torch.zeros(1, 3, 7, 32, 32), torch.zeros(1, 3, 7, 32, 32))
    except ValueError as e:
        assert "multiple of the patch" in str(e)
    else:
        raise AssertionError("a shape that is not a patch multiple must be refused")
