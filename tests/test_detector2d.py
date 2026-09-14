import inspect

import pytest

torch = pytest.importorskip("torch")

from anatobind.model.detector2d import Detector2D

SLAB = (1, 1, 5, 64, 64)


def test_forward_takes_the_slab_and_nothing_else():
    """没有真值能进入预测。"""
    assert list(inspect.signature(Detector2D.forward).parameters) == ["self", "slab"]


def test_output_is_on_the_stride_two_grid_plus_a_global_vector():
    torch.manual_seed(0)
    m = Detector2D(num_classes=5, embed_dim=12, stem_ch=8, d_model=16).eval()
    out = m(torch.randn(*SLAB))
    assert out["heat"].shape == (1, 5, 32, 32) and out["offset"].shape == (1, 2, 32, 32)
    assert out["size"].shape == (1, 2, 32, 32) and out["feat"].shape == (1, 16, 32, 32)
    assert out["global_feat"].ndim == 2 and out["global_feat"].shape[0] == 1


def test_the_stem_actually_reads_every_slice_of_the_slab():
    torch.manual_seed(0)
    m = Detector2D(num_classes=5, embed_dim=12, stem_ch=8, d_model=16).eval()
    x = torch.randn(*SLAB)
    with torch.no_grad():
        a = m(x)["heat"]
        x2 = x.clone()
        x2[:, :, 0] += 5.0                      # change only the first slice
        b = m(x2)["heat"]
    assert not torch.allclose(a, b)


def test_gradients_reach_the_stem_and_the_head():
    m = Detector2D(num_classes=5, embed_dim=12, stem_ch=8, d_model=16)
    m(torch.randn(*SLAB))["heat"].square().mean().backward()
    for p in (m.stem[0].weight, m.head.heat.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0
