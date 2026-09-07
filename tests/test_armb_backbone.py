import pytest

torch = pytest.importorskip("torch")

from anatobind.model.backbone import Backbone


@pytest.fixture(scope="module")
def net():
    return Backbone()


def test_feature_pyramid_shapes_follow_the_plan_strides(net):
    x = torch.randn(2, 1, 64, 128, 128)
    f1, f2, f3, f4 = net(x)
    # strides (2,4,4) (4,8,8) (8,16,16) (16,32,32), channels C 2C 4C 8C
    assert f1.shape == (2, 32, 32, 32, 32)
    assert f2.shape == (2, 64, 16, 16, 16)
    assert f3.shape == (2, 128, 8, 8, 8)
    assert f4.shape == (2, 256, 4, 4, 4)


def test_channel_dims_are_exposed_for_the_decoders(net):
    assert net.channels == (32, 64, 128, 256)


def test_scale_stays_in_the_throwaway_band(net):
    # A5: this scaffold must not silently become the plan's 12M default
    n = net.num_parameters()
    assert 3e5 < n < 6e6, f"backbone has {n} parameters"


def test_window_larger_than_the_feature_map_does_not_raise(net):
    # the coarsest stage is smaller than the (4,8,8) window
    f1, f2, f3, f4 = net(torch.randn(1, 1, 32, 64, 64))
    assert f4.shape == (1, 256, 2, 2, 2)


def test_gradients_reach_the_stem(net):
    net.zero_grad()
    out = net(torch.randn(1, 1, 32, 64, 64))
    sum(f.square().mean() for f in out).backward()
    stem = net.swin.patch_embed.proj.weight
    assert stem.grad is not None
    assert torch.isfinite(stem.grad).all()
    assert stem.grad.abs().sum() > 0


def test_unused_final_stage_is_not_carried(net):
    # F1..F4 stop at stride (16,32,32); layers4 would be dead weight
    assert not hasattr(net.swin, "layers4")
