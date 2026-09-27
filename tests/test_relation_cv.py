import numpy as np
import pytest

pytest.importorskip("sklearn")

from anatobind.relation.cv import (
    inner_folds, labels_for_fold, outer_folds, read_preds, run_b0, run_bgeo, run_bprior, select_config, trainable_rows,
    write_preds,
)
from anatobind.relation.labels import acceptable_matrix, c1_labels
from anatobind.eval.relation_metrics import is_correct
from synth_relation import synthetic_table


def test_outer_folds_come_from_the_table_and_cover_every_lesion():
    t, _ = synthetic_table(10, 3)
    folds = outer_folds(t)
    assert [k for k, _, _ in folds] == [0, 1, 2, 3, 4]
    assert sorted(np.concatenate([te for _, _, te in folds]).tolist()) == list(range(30))
    for k, tr, te in folds:
        assert set(t.patients()[tr]).isdisjoint(t.patients()[te]) and (t.folds()[te] == k).all()


def test_inner_folds_split_by_patient_and_are_seeded():
    patients = np.array([f"p{i // 3}" for i in range(30)])
    a, b = inner_folds(patients, 5, seed=0), inner_folds(patients, 5, seed=0)
    assert len(a) == 5 and all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    for tr, va in a:
        assert set(patients[tr]).isdisjoint(patients[va]) and len(tr) + len(va) == 30
    assert sorted(np.concatenate([va for _, va in a]).tolist()) == list(range(30))
    assert not all(np.array_equal(x[1], y[1]) for x, y in zip(a, inner_folds(patients, 5, seed=1)))


def test_select_config_prefers_the_earliest_within_tolerance():
    key, rec = select_config([("a", 0.90), ("b", 0.9005), ("c", 0.95)])
    assert key == "c" and rec["tie"] is False
    key, rec = select_config([("a", 0.9995), ("b", 1.0), ("c", 0.99)])
    assert key == "a" and rec["tie"] is True and rec["chosen"] == "a"


def test_preds_round_trip(tmp_path):
    t, _ = synthetic_table(2, 2)
    preds, _ = run_b0(t)
    write_preds(tmp_path / "b0.csv", preds)
    back = read_preds(tmp_path / "b0.csv")
    assert back["lesion_id"].tolist() == preds["lesion_id"].tolist() and np.allclose(back["probs"], preds["probs"])
    assert back["config"].tolist() == ["c1"] * 4 and back["fold"].tolist() == preds["fold"].tolist()


def test_bprior_and_bgeo_cover_every_lesion_once_and_are_deterministic():
    t, _ = synthetic_table(10, 4)
    labels = c1_labels(t)
    _, acc = acceptable_matrix(labels, t.lesion_id)
    p1, rec1 = run_bprior(t, labels, "type_side")
    assert p1["probs"].shape == (40, 8) and np.allclose(p1["probs"].sum(1), 1) and sorted(p1["lesion_id"].tolist()) == list(range(40))
    g1, r1 = run_bgeo(t, labels, "lr", seed=0, inner_k=3)
    g2, r2 = run_bgeo(t, labels, "lr", seed=0, inner_k=3)
    assert np.allclose(g1["probs"], g2["probs"]) and set(r1["folds"]) == {0, 1, 2, 3, 4}
    assert is_correct(g1["probs"], acc).mean() >= 0.8 and 0 <= r1["mean_inner_score"] <= 1
    assert all(r1["folds"][k]["chosen"] in {str(c) for c, _ in r1["folds"][k]["scores"]} for k in r1["folds"])


def test_labels_for_fold_accepts_a_dict_or_a_callable():
    d = {1: frozenset({"cortex"})}
    assert labels_for_fold(d, 3) is d and labels_for_fold(lambda k: {k: frozenset({"white_matter"})}, 3) == {3: frozenset({"white_matter"})}


def test_rows_whose_acceptable_set_misses_every_candidate_do_not_train():
    has = np.array([True, True, False])
    acc = np.zeros((3, 7), bool)
    acc[0, 0] = acc[1, 4] = True                                      # lesion 1: brainstem
    cand = np.zeros((3, 7), bool)
    cand[:, [0, 1]] = True                                            # brainstem is not a candidate anywhere
    assert trainable_rows(has, acc, cand).tolist() == [True, False, False]
