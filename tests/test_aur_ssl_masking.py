# tests/test_aur_ssl_masking.py
import pytest
import torch

import anatobind.aur.ssl.masking as M
import anatobind.aur.swin as S


def _crop(b=1, shape=(16, 32, 32)):
    """An image whose foreground is a box over most of the volume; the last quarter of the columns is outside the volume."""
    d, h, w = shape
    image = torch.full((b, 1) + shape, -1.0)
    image[:, :, d // 8: d - d // 8, h // 8: h - h // 8, w // 8: w - w // 4 - w // 8] = torch.rand(b, 1, d - d // 4, h - h // 4, w - w // 4 - w // 4)
    valid = torch.ones((b,) + shape)
    valid[..., w - w // 4:] = 0.0
    return image, valid


def test_foreground_patches_follow_the_valid_foreground():
    image, valid = _crop()
    fg = M.foreground_patches(image, valid)
    assert fg.shape == (1, 8, 8, 8) and fg.dtype == torch.bool
    assert fg[0, 1:7, 1:7, 1:5].all() and not fg[0, 0].any() and not fg[0, :, :, 6:].any()     # x >= 24 is outside the volume
    with pytest.raises(ValueError):
        M.patch_grid((15, 32, 32))


def test_block_masks_hide_about_the_ratio_of_the_foreground_and_nothing_else():
    image, valid = _crop(b=2, shape=(32, 64, 64))
    fg = M.foreground_patches(image, valid)
    g = torch.Generator().manual_seed(0)
    hidden = M.batch_masks(fg, g, ratio=0.6, block_mm=(8.0, 16.0))
    assert hidden.shape == fg.shape and hidden.dtype == torch.bool and (hidden & ~fg).sum() == 0
    share = M.hidden_share(hidden, fg)
    assert ((share - 0.6).abs() < 0.08).all(), share
    again = M.batch_masks(fg, torch.Generator().manual_seed(0), ratio=0.6, block_mm=(8.0, 16.0))
    assert torch.equal(hidden, again)
    other = M.batch_masks(fg, torch.Generator().manual_seed(1), ratio=0.6, block_mm=(8.0, 16.0))
    assert not torch.equal(hidden, other)
    assert not M.block_mask(torch.zeros(4, 4, 4, dtype=torch.bool), g).any()
    vox = M.hidden_voxels(hidden)
    assert vox.shape == (2, 32, 64, 64) and vox[:, ::2, ::4, ::4].equal(hidden)
    w = M.target_weight(hidden, image, valid)
    assert w.shape == (2, 32, 64, 64) and (w[valid < 0.5] == 0).all() and (w[image[:, 0] <= -1.0] == 0).all()
    assert (w.flatten(1).sum(1) > 0).all()
    light = M.batch_masks(fg, torch.Generator().manual_seed(0), ratio=0.2, block_mm=(8.0, 16.0))
    assert (M.hidden_share(light, fg) < share).all()


def test_the_backbone_never_sees_a_hidden_voxel():
    torch.manual_seed(0)
    bb = S.SwinBackbone(embed=32, depths=(1, 1, 1, 1), heads=(1, 2, 4, 8)).eval()
    image, valid = _crop(shape=(16, 32, 32))
    coords = torch.stack(torch.meshgrid(torch.arange(16.0) * 2, torch.arange(32.0), torch.arange(32.0), indexing="ij"))[None]
    local = coords / 32 - 0.5
    fg = M.foreground_patches(image, valid)
    hidden = M.batch_masks(fg, torch.Generator().manual_seed(3), ratio=0.5, block_mm=(8.0, 16.0))
    visible = ~hidden
    assert hidden.any() and visible.any()
    with torch.no_grad():
        ref = bb(image, valid, coords, local, visible=visible)
        tampered = image.clone()
        hv = M.hidden_voxels(hidden)
        tampered[:, 0][hv] = torch.rand(int(hv.sum())) * 4 - 2                   # any content under the hidden patches
        same = bb(tampered, valid, coords, local, visible=visible)
        for a, b in zip(ref, same):
            assert torch.equal(a["feat"], b["feat"]) and torch.equal(a["valid"], b["valid"])
        shown = image.clone()
        vv = ~hv & (valid > 0.5)
        shown[:, 0][vv] = shown[:, 0][vv] + 0.5                                   # a change under a visible patch is seen
        differs = bb(shown, valid, coords, local, visible=visible)
        assert not torch.equal(ref[0]["feat"], differs[0]["feat"])
        plain = bb(image, valid, coords, local)
        assert not torch.equal(ref[0]["feat"], plain["feat"] if isinstance(plain, dict) else plain[0]["feat"])
    assert "mask_token" in bb.state_dict() and bb.mask_token.shape == (32,)
    with pytest.raises(ValueError, match="patch grid"):
        bb(image, valid, coords, local, visible=torch.ones(1, 8, 8, 9, dtype=torch.bool))
