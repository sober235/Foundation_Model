import inspect

import pytest

torch = pytest.importorskip("torch")

import anatobind.model.decoders as decoders
from anatobind.model.decoders import ADecoder, UBDecoder

CHANNELS = (32, 64, 128, 256)
SHAPES = [(2, 32, 8, 8, 8), (2, 64, 4, 4, 4), (2, 128, 2, 2, 2), (2, 256, 1, 1, 1)]


@pytest.fixture
def feats():
    return [torch.randn(*s) for s in SHAPES]


def test_a_decoder_shapes(feats):
    dec = ADecoder(CHANNELS, d_model=64, K=6, layers=2)
    out = dec(feats, out_shape=(16, 32, 32))
    assert out["masks"].shape == (2, 6, 16, 32, 32)
    assert out["presence"].shape == (2, 6)
    assert out["embed"].shape == (2, 6, 64)


def test_a_decoder_has_no_matching_step():
    """A queries are identity-anchored: query k is permanently structure k+1."""
    src = inspect.getsource(ADecoder)
    assert "linear_sum_assignment" not in src
    assert "hungarian" not in src.lower()


def test_a_decoder_query_count_is_fixed_and_ordered(feats):
    dec = ADecoder(CHANNELS, d_model=64, K=6, layers=2).eval()
    with torch.no_grad():
        a = dec(feats, out_shape=(16, 32, 32))["masks"]
        b = dec(feats, out_shape=(16, 32, 32))["masks"]
    # no sampling, no permutation: the same input yields the same query order
    torch.testing.assert_close(a, b)
    assert dec.K == 6


def test_ub_decoder_shapes_and_box_parametrisation(feats):
    dec = UBDecoder(CHANNELS, d_model=64, M=8, layers=2, num_classes=2)
    out = dec(feats)
    assert out["logits"].shape == (2, 8, 3)  # 2 classes + no-object
    assert out["boxes"].shape == (2, 8, 6)
    assert out["embed"].shape == (2, 8, 64)


def test_ub_boxes_are_normalised_with_positive_sizes(feats):
    dec = UBDecoder(CHANNELS, d_model=64, M=8, layers=2, num_classes=2)
    boxes = dec(feats)["boxes"]
    assert float(boxes.min()) >= 0.0 and float(boxes.max()) <= 1.0
    assert float(boxes[..., 3:].min()) > 0.0


def test_decoders_consume_every_feature_level(feats):
    """A level that never reaches the stack would be a silently dead branch."""
    dec = ADecoder(CHANNELS, d_model=64, K=6, layers=3)
    dec.zero_grad()
    for f in feats:
        f.requires_grad_(True)
    dec(feats, out_shape=(16, 32, 32))["masks"].square().mean().backward()
    for i, f in enumerate(feats):
        assert f.grad is not None and f.grad.abs().sum() > 0, f"level {i} unused"


def test_gradients_are_finite(feats):
    dec = UBDecoder(CHANNELS, d_model=64, M=8, layers=2, num_classes=2)
    out = dec(feats)
    (out["logits"].square().mean() + out["boxes"].square().mean()).backward()
    for p in dec.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
