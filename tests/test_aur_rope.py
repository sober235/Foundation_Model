# tests/test_aur_rope.py
import pytest
import torch

import anatobind.aur.rope as R


def test_rotation_touches_only_the_first_24_dims_and_keeps_shape_and_dtype():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 32)
    coords = torch.rand(2, 5, 3) * 100
    y = R.apply_rope(x, coords)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.allclose(y[..., 24:], x[..., 24:]) and not torch.allclose(y[..., :24], x[..., :24])
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4)          # a rotation keeps the norm
    assert torch.allclose(R.apply_rope(x, torch.zeros(2, 5, 3)), x, atol=1e-6)  # the origin rotates by nothing
    yh = R.apply_rope(x.half(), coords)
    assert yh.dtype == torch.float16 and torch.allclose(yh.float(), y, atol=2e-2)
    with pytest.raises(ValueError, match="too small"):
        R.apply_rope(torch.randn(1, 1, 2, 16), torch.zeros(1, 2, 3))


def test_attention_logits_depend_on_coordinate_differences_only():
    torch.manual_seed(1)
    q, k = torch.randn(1, 2, 6, 32), torch.randn(1, 2, 6, 32)
    coords = torch.rand(1, 6, 3) * 50
    shift = torch.tensor([[[12.5, -40.0, 7.0]]])
    a = R.apply_rope(q, coords) @ R.apply_rope(k, coords).transpose(-2, -1)
    b = R.apply_rope(q, coords + shift) @ R.apply_rope(k, coords + shift).transpose(-2, -1)
    assert torch.allclose(a, b, atol=1e-3)
    c = R.apply_rope(q, coords * 1.5) @ R.apply_rope(k, coords * 1.5).transpose(-2, -1)
    assert not torch.allclose(a, c, atol=1e-2)                                    # distances do matter
    f = R.rope_frequencies()
    assert f.shape == (4,) and f[0] == 1.0 and f[-1] == pytest.approx(1000 ** (-6 / 8))
