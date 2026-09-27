import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.relation.encoder import TORCH_CONFIGS, crop
from anatobind.relation.labels import acceptable_matrix, c1_labels
from anatobind.relation.table import features_per_slot, slot_col
from anatobind.relation.train import acceptable8, fit_torch_arm, predict_torch_arm, run_torch_arm
from synth_relation import synthetic_table


def _setup():
    t, p = synthetic_table(5, 4)
    labels = c1_labels(t)
    _, acc = acceptable_matrix(labels, t.lesion_id)
    x = crop(p["image"], p["mask"], 22, 1)
    return t, p, labels, acceptable8(acc), x, features_per_slot(t), t.candidates()


def test_fit_predict_shapes_early_stopping_and_determinism():
    t, p, labels, acc8, x, geo, present = _setup()
    tr, va = np.arange(0, 12), np.arange(12, 20)
    st1, ep1, va1 = fit_torch_arm("b1", x, geo, present, acc8, tr, va, seed=0, device="cpu", epochs=3, patience=1)
    st2, ep2, va2 = fit_torch_arm("b1", x, geo, present, acc8, tr, va, seed=0, device="cpu", epochs=3, patience=1)
    assert 1 <= ep1 <= 3 and 0 <= va1 <= 1 and ep1 == ep2 and va1 == va2
    p1, p2 = (predict_torch_arm("b1", s, x, geo, present, va, "cpu") for s in (st1, st2))
    assert p1.shape == (8, 8) and np.allclose(p1, p2) and np.allclose(p1.sum(1), 1) and (p1[:, :7][~present[va]] == 0).all()
    st, ep, acc = fit_torch_arm("b2", x, geo, present, acc8, tr, None, seed=0, device="cpu", epochs=2, patience=1)
    assert ep == 2 and np.isnan(acc)                                                     # refit without a validation fold


def test_run_torch_arm_covers_every_lesion_once_and_records_the_selection():
    t, p, labels, _, _, _, _ = _setup()
    lines = []
    preds, rec, states = run_torch_arm("b2", t, p, labels, configs=TORCH_CONFIGS[:2], seed=0, device="cpu", epochs=3, patience=1, inner_k=2,
                                       log=lines.append)
    assert sorted(preds["lesion_id"].tolist()) == list(range(20)) and np.allclose(preds["probs"].sum(1), 1)
    assert len(lines) == 5 and lines[0].startswith("b2 fold 0: chose px32_s")
    assert set(rec["folds"]) == {0, 1, 2, 3, 4} and set(states) == {0, 1, 2, 3, 4}
    for k, f in rec["folds"].items():
        assert f["chosen"] in ("px32_s3", "px32_s1") and f["n_untrainable"] == 0
        assert f["epochs"] == max(1, int(round(float(np.median(f["inner_epochs"][f["chosen"]])))))   # refit = median best inner epoch
    assert set(preds["config"].tolist()) <= {"px32_s3", "px32_s1"}


def test_stage_two_fine_tunes_each_fold_with_its_stage_one_config():
    t, p, labels, _, _, _, _ = _setup()
    _, rec1, states = run_torch_arm("b1", t, p, labels, configs=TORCH_CONFIGS[:2], seed=0, device="cpu", epochs=1, patience=1, inner_k=2)
    init = {k: (rec1["folds"][k]["chosen"], states[k]) for k in states}
    preds, rec2, _ = run_torch_arm("b1", t, p, labels, configs=TORCH_CONFIGS, seed=1, device="cpu", epochs=1, patience=1, inner_k=2, init=init)
    assert np.allclose(preds["probs"].sum(1), 1) and rec2["init_from_states"] is True
    for k in rec2["folds"]:
        assert rec2["folds"][k]["chosen"] == rec1["folds"][k]["chosen"] and len(rec2["folds"][k]["scores"]) == 1


def test_a_label_outside_the_candidates_is_dropped_from_training_not_turned_into_an_infinite_loss():
    t, p, labels, _, _, _, _ = _setup()
    t.rows[0][slot_col("brainstem", "candidate")] = False          # brainstem is not a candidate of lesion 0
    bad = dict(labels)
    bad[0] = frozenset({"brainstem"})
    preds, rec, _ = run_torch_arm("b1", t, p, bad, configs=TORCH_CONFIGS[:1], seed=0, device="cpu", epochs=1, patience=1, inner_k=2)
    assert np.isfinite(preds["probs"]).all() and sum(f["n_untrainable"] for f in rec["folds"].values()) == 4   # lesion 0 trains 4 outer folds
