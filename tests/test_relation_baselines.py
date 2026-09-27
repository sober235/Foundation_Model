import numpy as np
import pytest

from anatobind.relation.baselines import PRIOR_VARIANTS, PriorModel, b0_probs, target_weights
from anatobind.relation.labels import acceptable_matrix, c1_labels
from synth_relation import synthetic_table


def test_b0_is_the_c1_rule_and_respects_candidates():
    t, _ = synthetic_table(3, 4)
    p = b0_probs(t)
    assert p.shape == (12, 8) and p.argmax(1).tolist() == t.c1_slot().tolist() and np.allclose(p.sum(1), 1)


def test_target_weights_split_sets_evenly():
    acc = np.zeros((3, 7), bool)
    acc[0, 0] = True
    acc[1, [0, 1]] = True
    w = target_weights(acc)
    assert w[0].tolist() == [1] + [0] * 6 and w[1, 0] == w[1, 1] == 0.5 and w[2].sum() == 0


def test_prior_variants_count_with_smoothing_and_fall_back_on_unseen_keys():
    t, _ = synthetic_table(6, 4)
    _, acc = acceptable_matrix(c1_labels(t), t.lesion_id)
    for v in PRIOR_VARIANTS:
        p = PriorModel(v).fit(t, acc).predict(t)
        assert p.shape == (24, 8) and np.allclose(p.sum(1), 1) and (p[:, 7] == 0).all()
    m = PriorModel("majority").fit(t, acc)
    counts = np.bincount(t.c1_slot(), minlength=7).astype(float)
    expected = (counts + 1) / (counts.sum() + 7)
    cand = t.candidates()[0]
    raw = np.where(cand, expected, 0)
    assert np.allclose(m.predict(t)[0, :7], raw / raw.sum())
    other, _ = synthetic_table(1, 2, seed=99)
    other.rows[0]["coarse_location"] = "unseen|key|here"
    q = PriorModel("type_side_location").fit(t, acc).predict(other)
    assert np.allclose(q.sum(1), 1)                                          # unseen key -> global counts, still a distribution
