import inspect

import pytest

torch = pytest.importorskip("torch")

from anatobind.model.upstream import Upstream
from anatobind.model.upstream_losses import upstream_loss, valid_slices

SMALL = dict(K=6, M=4, num_classes=4, d_model=32, embed_dim=16, layers=2, heads=4, mask_dim=8)
SHAPE = (32, 64, 64)
GRID = (16, 16, 16)       # F1 grid, stride (2, 4, 4)


def _batch(valid_depth=28):
    seg = torch.zeros(1, *SHAPE, dtype=torch.long)
    seg[0, 2:10, 10:30, 10:30] = 2
    seg[0, 12:20, 30:50, 20:40] = 5
    return {
        "image": torch.randn(1, 1, *SHAPE), "seg": seg, "valid_depth": torch.tensor([valid_depth]),
        "present": torch.tensor([[False, True, False, False, True, False]]),
        "boxes": [torch.tensor([[3.0, 12, 12, 8, 20, 20], [13.0, 32, 22, 18, 40, 30]])],
        "box_classes": [torch.tensor([1, 0])],
    }


@pytest.fixture
def model():
    torch.manual_seed(0)
    return Upstream(**SMALL)


def test_forward_takes_the_image_and_nothing_else():
    """Gate G1: no ground truth can reach a prediction."""
    assert list(inspect.signature(Upstream.forward).parameters) == ["self", "image"]


def test_output_shapes(model):
    model.train()
    out = model(_batch()["image"])
    assert out["masks"].shape == (1, 6, *SHAPE) and out["presence"].shape == (1, 6)
    assert out["a_embed"].shape == (1, 6, 32)
    assert out["heat"].shape == (1, 4, *GRID) and out["offset"].shape == (1, 3, *GRID)
    assert out["size"].shape == (1, 3, *GRID) and out["feat"].shape == (1, 32, *GRID)


def test_the_lesion_head_is_dense(model):
    assert hasattr(model, "u_head") and not hasattr(model, "u_dec")


def test_the_loss_is_finite_and_reaches_every_head(model):
    model.train()
    model.zero_grad()
    b = _batch()
    loss, parts = upstream_loss(model(b["image"]), b)
    assert torch.isfinite(loss) and set(parts) == {"mask", "presence", "heat", "offset", "size"}
    loss.backward()
    for p in (model.backbone.swin.patch_embed.proj.weight, model.mask_head.img[0].weight,
              model.presence.weight, model.u_head.heat.weight, model.u_head.size.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_padded_slices_do_not_contribute(model):
    model.eval()
    b1 = _batch(valid_depth=28)
    b2 = {**b1, "seg": b1["seg"].clone()}
    b2["seg"][0, 28:] = 2          # garbage in the depth padding
    with torch.no_grad():
        out = model(b1["image"])
        assert upstream_loss(out, b1)[1]["mask"] == pytest.approx(upstream_loss(out, b2)[1]["mask"])


def test_valid_slices_marks_the_padding():
    v = valid_slices(torch.tensor([2, 3]), 4, "cpu")
    assert v.tolist() == [[True, True, False, False], [True, True, True, False]]


def test_bfloat16_autocast_gives_a_finite_loss(model):
    model.train()
    b = _batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(b["image"])
    assert torch.isfinite(upstream_loss(out, b)[0])
