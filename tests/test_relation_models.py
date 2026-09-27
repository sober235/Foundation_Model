import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.relation.encoder import crop
from anatobind.relation.models import B1Model, B2Model, set_nll
from anatobind.relation.table import features_per_slot
from synth_relation import synthetic_table


def _batch():
    t, p = synthetic_table(2, 3)
    x = torch.from_numpy(crop(p["image"], p["mask"], 32, 3))
    geo = torch.from_numpy(features_per_slot(t))
    present = torch.from_numpy(t.candidates())
    return x, geo, present, torch.from_numpy(t.c1_slot())


def test_b2_and_b1_shapes_and_absent_candidates():
    x, geo, present, _ = _batch()
    torch.manual_seed(0)
    assert B2Model(6)(x).shape == (6, 8)
    b1 = B1Model(6).eval()
    with torch.no_grad():
        probs = b1(x, geo, present).softmax(-1)
    assert probs.shape == (6, 8) and torch.allclose(probs.sum(-1), torch.ones(6))
    assert (probs[:, :7][~present] == 0).all()                       # the head masks absent candidates to -inf


def test_set_nll_equals_cross_entropy_for_singletons_and_pools_sets():
    logits = torch.tensor([[2.0, 1.0, 0.0, 0, 0, 0, 0, 0], [2.0, 1.0, 0.0, 0, 0, 0, 0, 0]])
    single = torch.zeros(2, 8, dtype=torch.bool)
    single[:, 1] = True
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 1]))
    assert torch.isclose(set_nll(logits, single), ce)
    both = single.clone()
    both[:, 0] = True
    p = logits.softmax(-1)
    assert torch.isclose(set_nll(logits, both), -(p[:, 0] + p[:, 1]).log().mean())
    assert set_nll(logits, both) < set_nll(logits, single)


def test_models_train_one_step_without_nan():
    x, geo, present, c1 = _batch()
    y = torch.zeros(6, 8, dtype=torch.bool)
    y[torch.arange(6), c1] = True          # C1 is always a candidate; a label outside the candidates would give B1 an infinite loss (cv.trainable_rows)
    for model, args in ((B2Model(6), (x,)), (B1Model(6), (x, geo, present))):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss = set_nll(model(*args), y)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)
