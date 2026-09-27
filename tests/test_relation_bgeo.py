import numpy as np
import pytest

pytest.importorskip("sklearn")

from anatobind.relation.baselines import GEO_GRIDS, GEO_KINDS, GeoLearner, expand_sets
from anatobind.relation.labels import acceptable_matrix, c1_labels
from anatobind.relation.table import features_flat
from synth_relation import synthetic_table


def test_expand_sets_duplicates_ambiguous_lesions_with_split_weights():
    X = np.arange(6, dtype=float).reshape(3, 2)
    acc = np.zeros((3, 7), bool)
    acc[0, 0] = True
    acc[1, [0, 1]] = True
    acc[2, 3] = True
    Xr, yr, wr = expand_sets(X, acc)
    assert Xr.shape == (4, 2) and yr.tolist() == [0, 0, 1, 3] and wr.tolist() == [1.0, 0.5, 0.5, 1.0]


@pytest.mark.parametrize("kind", GEO_KINDS)
def test_every_learner_fits_c1_from_the_geometry_and_masks_candidates(kind):
    t, _ = synthetic_table(8, 6)
    X = features_flat(t)
    _, acc = acceptable_matrix(c1_labels(t), t.lesion_id)
    # the least regularised grid point: this asks whether the learner can represent C1, not which setting the inner CV picks
    # (C = 0.01 reaches only 0.896 on these 48 rows)
    m = GeoLearner(kind, GEO_GRIDS[kind][-1]).fit(X, acc)
    p = m.predict_probs(X, t.candidates())
    assert p.shape == (48, 8) and np.allclose(p.sum(1), 1) and (p[:, 7] == 0).all()
    assert (p.argmax(1) == t.c1_slot()).mean() >= 0.9                         # C1 is a function of the geometry columns
    assert (p[~np.concatenate([t.candidates(), np.ones((48, 1), bool)], 1)] == 0).all()


def test_unseen_classes_get_zero_probability():
    t, _ = synthetic_table(4, 4)
    X = features_flat(t)
    _, acc = acceptable_matrix(c1_labels(t), t.lesion_id)
    p = GeoLearner("lr", {"C": 1.0}).fit(X, acc).predict_probs(X, np.ones((16, 7), bool))
    assert (p[:, 2:7] == 0).all()                                               # only WM / cortex occur in the synthetic labels
