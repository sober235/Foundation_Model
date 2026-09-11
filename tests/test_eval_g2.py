import pytest

from anatobind.eval.g2 import bootstrap_ci, delta, g2, paired_counts
from anatobind.eval.matching import CORRECT, MISS, WRONG_CLASS, WRONG_HOST


def _outcomes():
    table = {
        "A": [(CORRECT, WRONG_HOST, CORRECT), (CORRECT, CORRECT, CORRECT), (CORRECT, MISS, CORRECT)],
        "B": [(WRONG_HOST, WRONG_HOST, CORRECT), (CORRECT, WRONG_HOST, WRONG_CLASS)],
        "C": [(CORRECT, CORRECT, CORRECT)],
    }
    return {(scan, i): {"clean": c, "noise_q3": q3, "us16": u16}
            for scan, rows in table.items() for i, (c, q3, u16) in enumerate(rows)}


def test_the_paired_set_drops_lesions_missed_or_misclassified_in_either_view():
    scans, arr = paired_counts(_outcomes(), "noise_q3")
    assert scans == ["A", "B", "C"]
    assert arr.tolist() == [[2, 0, 1], [2, 1, 2], [1, 0, 0]]


def test_delta_is_the_paired_difference_in_wrong_host_rate():
    _, arr = paired_counts(_outcomes(), "noise_q3")
    d, r_clean, r_view = delta(arr)
    assert r_clean == pytest.approx(1 / 5) and r_view == pytest.approx(3 / 5) and d == pytest.approx(2 / 5)


def test_the_scan_bootstrap_brackets_the_estimate_and_is_reproducible():
    _, arr = paired_counts(_outcomes(), "noise_q3")
    lo, hi = bootstrap_ci(arr, reps=2000, seed=0)
    assert lo <= 0.4 <= hi and bootstrap_ci(arr, reps=2000, seed=0) == (lo, hi)


def test_a_large_consistent_increase_passes_and_no_change_fails():
    up = {}
    for i in range(40):
        up[(f"S{i}", 0)] = {"clean": CORRECT, "noise_q3": WRONG_HOST, "us16": CORRECT}
        up[(f"S{i}", 1)] = {"clean": CORRECT, "noise_q3": CORRECT, "us16": CORRECT}
    res = g2(up, reps=2000)
    assert res["noise_q3"]["pass"] and not res["us16"]["pass"] and res["pass"]
    flat = {k: {**v, "noise_q3": CORRECT} for k, v in up.items()}
    assert not g2(flat, reps=2000)["pass"]
