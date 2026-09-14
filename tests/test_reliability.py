import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.model.reliability import (
    LesionReliability, ScanReliability, lesion_scalars, scan_scalars,
)


def _row(score, correct=1, box=(0, 0, 0, 4, 20, 20)):
    return {"score": score, "correct": correct, "box": list(box), "embed": np.zeros(8, np.float32)}


def test_lesion_scalars_are_score_margin_and_size():
    s = lesion_scalars(_row(0.8))
    assert s.shape == (3,) and s[0] == pytest.approx(0.8)
    assert s[2] > 0                                  # box volume in log units


def test_scan_scalars_survive_a_scan_with_no_detections():
    s = scan_scalars([])
    assert s.shape == (5,) and np.isfinite(s).all()


def test_both_heads_return_one_logit_per_item():
    lr = LesionReliability(embed_dim=8, n_scalar=3, hidden=16)
    sr = ScanReliability(global_dim=12, n_scalar=5, hidden=16)
    assert lr(torch.randn(4, 8), torch.randn(4, 3)).shape == (4,)
    assert sr(torch.randn(2, 12), torch.randn(2, 5)).shape == (2,)


def test_a_lesion_head_can_learn_a_separable_toy_problem():
    torch.manual_seed(0)
    n = 256
    feat = torch.randn(n, 8)
    scal = torch.randn(n, 3)
    y = (feat[:, 0] + scal[:, 0] > 0).float()
    head = LesionReliability(embed_dim=8, n_scalar=3, hidden=32)
    opt = torch.optim.Adam(head.parameters(), 1e-2)
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(feat, scal), y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(((head(feat, scal) > 0).float() == y).float().mean()) > 0.9
