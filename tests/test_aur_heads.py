# tests/test_aur_heads.py
import torch

import anatobind.aur.heads as H
from anatobind.aur.labels import N_ENTITIES
from anatobind.aur.swin import SwinBackbone

TINY = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4))


def _levels(shape=(8, 32, 32)):
    torch.manual_seed(0)
    bb = SwinBackbone(**TINY)
    img = torch.randn(2, 1, *shape)
    valid = torch.ones(2, *shape)
    valid[1, :, 16:] = 0.0
    axes = [torch.arange(n, dtype=torch.float32) + 0.5 for n in shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0)[None].expand(2, -1, -1, -1, -1).clone()
    local = torch.stack(torch.meshgrid(*[(a / n) * 2 - 1 for a, n in zip(axes, shape)], indexing="ij"), 0)[None].expand(2, -1, -1, -1, -1).clone()
    return bb, img, bb(img, valid, coords, local)


def test_entity_event_sequence_and_mask_heads_shapes():
    bb, img, levels = _levels()
    d = 16
    a = H.EntityDecoder(bb.channels, d, layers=3, heads=2)(levels)
    u = H.EventDecoder(bb.channels, d, M=5, layers=3, heads=2)(levels)
    s = H.SequenceHead(bb.channels[3], d)(levels[3])
    assert a["embed"].shape == (2, N_ENTITIES, d) and a["presence"].shape == (2, N_ENTITIES)
    assert u["embed"].shape == (2, 5, d) and u["presence"].shape == (2, 5)
    assert s["embed"].shape == (2, d) and s["logits"].shape == (2, 6)
    mh = H.MaskHead(bb.channels, d, (2, 4, 4), dim=8, mask_dim=4)
    pix = mh.pixels(levels, img)
    assert pix["coarse"].shape == (2, 8, 4, 8, 8) and pix["full"].shape == (2, 4, 8, 32, 32)
    full = mh.full_masks(a["embed"], pix)
    assert full.shape == (2, N_ENTITIES, 8, 32, 32)
    points = torch.tensor([[0, 5, 8 * 32 * 32 - 1], [7, 7, 100]])
    pts = mh.full_masks(a["embed"], pix, points)
    assert pts.shape == (2, N_ENTITIES, 3)
    assert torch.allclose(pts[0, :, 1], full[0].flatten(1)[:, 5], atol=1e-5) and torch.allclose(pts[1, :, 0], full[1].flatten(1)[:, 7], atol=1e-5)
    assert mh.coarse_masks(u["embed"], pix).shape == (2, 5, 4, 8, 8)


def test_query_decoder_ignores_padded_memory():
    bb, img, levels = _levels()
    torch.manual_seed(1)
    dec = H.EntityDecoder(bb.channels, 16, layers=3, heads=2).eval()
    with torch.no_grad():
        a = dec(levels)
        for lv in levels:                                             # scribble on the padded tokens of sample 1
            pad = lv["valid"][1] < 0.5
            lv["feat"][1].permute(1, 2, 3, 0)[pad] = 7.0
        b = dec(levels)
    assert torch.allclose(a["embed"][1], b["embed"][1], atol=1e-5)
