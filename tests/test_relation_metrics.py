import numpy as np
import pytest

from anatobind.eval.relation_metrics import (
    by_stratum, delta_d_curve, gate_r1, is_correct, mcnemar, patient_bootstrap, predicted_slot, rescue_harm, summary,
    topk_correct,
)


def _probs(slots):
    p = np.zeros((len(slots), 8))
    for i, s in enumerate(slots):
        p[i, s] = 1.0
    return p


def _acc(sets):
    a = np.zeros((len(sets), 7), bool)
    for i, s in enumerate(sets):
        a[i, list(s)] = True
    return a


def test_set_valued_correctness_and_none_is_never_correct():
    probs = _probs([0, 1, 7, 1])
    acc = _acc([{0}, {0, 1}, {0}, {0}])
    assert predicted_slot(probs).tolist() == [0, 1, 7, 1]
    assert is_correct(probs, acc).tolist() == [True, True, False, False]
    p = np.array([[0.5, 0.4, 0.1, 0, 0, 0, 0, 0]])
    assert topk_correct(p, _acc([{1}]), k=2).tolist() == [True] and topk_correct(p, _acc([{2}]), k=2).tolist() == [False]


def test_summary_reports_singleton_subset_metrics():
    probs = _probs([0, 1, 0, 1])
    acc = _acc([{0}, {1}, {0, 1}, {0}])
    s = summary(probs, acc)
    assert s["n"] == 4 and s["accuracy"] == 0.75 and s["singleton_rate"] == 0.75
    assert s["singleton"]["n"] == 3 and s["singleton"]["accuracy"] == pytest.approx(2 / 3)
    # singleton truth [0, 1, 0], predictions [0, 1, 1]: WM recall 1/2 precision 1, cortex recall 1 precision 1/2
    assert s["singleton"]["balanced_accuracy"] == pytest.approx(0.75)
    assert s["singleton"]["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert np.array(s["singleton"]["confusion"]).shape == (7, 8) and s["singleton"]["confusion"][0][1] == 1


def test_rescue_harm_and_mcnemar():
    model = np.array([True, True, False, True, False])
    comp = np.array([False, True, True, True, False])
    r = rescue_harm(model, comp)
    assert (r["rescue"], r["harm"], r["net"], r["net_rate"]) == (1, 1, 0, 0.0)
    assert mcnemar(1, 1) == 1.0 and mcnemar(0, 0) == 1.0 and mcnemar(10, 0) == pytest.approx(2 * 0.5 ** 10)


def test_patient_bootstrap_is_deterministic_and_brackets_the_mean():
    values = np.array([1, 1, 0, 0, 1, 0, 1, 1], float)
    patients = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    m1, lo1, hi1 = patient_bootstrap(values, patients, n_boot=500, seed=0)
    m2, lo2, hi2 = patient_bootstrap(values, patients, n_boot=500, seed=0)
    assert (m1, lo1, hi1) == (m2, lo2, hi2) and m1 == 0.625 and lo1 <= m1 <= hi1
    assert patient_bootstrap(np.ones(4), np.array(["a", "a", "b", "b"]), n_boot=50) == (1.0, 1.0, 1.0)


def test_strata_and_delta_d_curve():
    vals = np.array([1.0, 0.0, 1.0, 1.0])
    assert by_stratum(vals, np.array(["x", "x", "y", "y"])) == {"x": {"n": 2, "mean": 0.5}, "y": {"n": 2, "mean": 1.0}}
    curve = delta_d_curve(vals, np.array([0.0, 0.5, 3.0, 20.0]), np.array(["a", "b", "c", "d"]), n_boot=50)
    assert [c["bin"] for c in curve] == ["[0.0, 1.0)", "[1.0, 2.0)", "[2.0, 4.0)", "[4.0, 8.0)", "[8.0, inf)"]
    assert [c["n"] for c in curve] == [2, 0, 1, 0, 1] and curve[0]["mean"] == 0.5 and curve[1]["mean"] is None


def test_gate_r1_needs_all_four_conditions():
    acc = _acc([{0}] * 6 + [{1}] * 6)
    patients = np.array([f"p{i}" for i in range(12)])
    model = _probs([0] * 6 + [1] * 6)                                        # perfect
    bgeo = _probs([0] * 6 + [0] * 6)                                         # wrong on the cortex half
    g = gate_r1(model, {"bprior": bgeo, "bgeo": bgeo, "b2": bgeo}, acc, patients, n_boot=200)
    assert g["accuracy"]["model"] == 1.0 and g["accuracy"]["bgeo"] == 0.5 and g["net_rescue_vs_bgeo"]["mean"] == 0.5
    assert g["net_rescue_ci_low_gt_0"] and g["gt_bprior"] and g["gt_bgeo"] and g["gt_b2"] and g["go"]
    tie = gate_r1(bgeo, {"bprior": bgeo, "bgeo": bgeo, "b2": bgeo}, acc, patients, n_boot=200)
    assert not tie["gt_bgeo"] and not tie["go"] and tie["net_rescue_vs_bgeo"]["mean"] == 0.0
