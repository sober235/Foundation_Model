import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/check_h1.py"
    spec = importlib.util.spec_from_file_location("check_h1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dets(n_correct, n_wrong):
    return [{"correct": 1} for _ in range(n_correct)] + [{"correct": 0} for _ in range(n_wrong)]


def test_h1_passes_when_there_are_enough_detections_and_a_balanced_mix():
    verdict = _load().h1_verdict(_dets(150, 150))
    assert verdict["n_det"] == 300
    assert verdict["correct_rate"] == pytest.approx(0.5)
    assert verdict["passed"] is True


def test_h1_fails_on_too_few_detections_even_with_a_balanced_mix():
    verdict = _load().h1_verdict(_dets(150, 149))          # 299 detections, correct rate ~0.502
    assert verdict["n_det"] == 299
    assert 0.2 <= verdict["correct_rate"] <= 0.8            # the mix is fine; count is the only problem
    assert verdict["passed"] is False


def test_h1_fails_when_the_detector_is_almost_always_right():
    """Near-perfect detections: nothing for a reliability head to learn from."""
    verdict = _load().h1_verdict(_dets(290, 10))            # 300 detections, correct rate ~0.967
    assert verdict["n_det"] == 300
    assert verdict["correct_rate"] > 0.8
    assert verdict["passed"] is False


def test_h1_fails_when_the_detector_is_almost_always_wrong():
    verdict = _load().h1_verdict(_dets(10, 290))            # 300 detections, correct rate ~0.033
    assert verdict["n_det"] == 300
    assert verdict["correct_rate"] < 0.2
    assert verdict["passed"] is False


def test_h1_fails_cleanly_with_no_detections_at_all():
    """The failure mode H1 exists to catch: an empty analysis subset must FAIL, not crash or pass."""
    verdict = _load().h1_verdict([])
    assert verdict == {"n_det": 0, "correct_rate": 0.0, "passed": False}


def test_h1_passes_at_the_exact_lower_bound_of_the_correct_rate():
    verdict = _load().h1_verdict(_dets(60, 240))            # 300 detections, correct rate exactly 0.2
    assert verdict["correct_rate"] == pytest.approx(0.2)
    assert verdict["passed"] is True


def test_h1_passes_at_the_exact_upper_bound_of_the_correct_rate():
    verdict = _load().h1_verdict(_dets(240, 60))            # 300 detections, correct rate exactly 0.8
    assert verdict["correct_rate"] == pytest.approx(0.8)
    assert verdict["passed"] is True


def test_h1_fails_just_below_the_lower_bound_of_the_correct_rate():
    verdict = _load().h1_verdict(_dets(59, 241))            # 300 detections, correct rate ~0.197
    assert verdict["correct_rate"] < 0.2
    assert verdict["passed"] is False


def test_h1_fails_just_above_the_upper_bound_of_the_correct_rate():
    verdict = _load().h1_verdict(_dets(241, 59))            # 300 detections, correct rate ~0.803
    assert verdict["correct_rate"] > 0.8
    assert verdict["passed"] is False


def test_threshold_search_picks_the_lowest_admissible_threshold():
    # 0.10 and 0.15 both land the rate in [0.2, 0.5]; the search must stop at the first (lowest), not
    # keep going and return some other admissible one.
    rate_of = {0.05: 0.9, 0.10: 0.35, 0.15: 0.3, 0.20: 0.1}.__getitem__
    chosen = _load().pick_threshold([0.05, 0.10, 0.15, 0.20], rate_of)
    assert chosen == pytest.approx(0.10)


def test_threshold_search_reports_failure_rather_than_guessing():
    # The rate never falls inside [0.2, 0.5] at any threshold tried; the search must say so, not
    # silently fall back to the first or last threshold it looked at.
    chosen = _load().pick_threshold([0.05, 0.10, 0.15, 0.20], lambda thr: 0.9)
    assert chosen is None


def test_the_threshold_is_chosen_without_the_held_out_fold_but_measured_with_it():
    asked = []

    def fake_gather(folds, thr):
        asked.append(tuple(folds))
        # scan-level rate lands in [0.2, 0.5] only from thr = 0.3 upward
        n_pos = 1 if thr >= 0.3 else 0
        scan = [{"label": 1}] * n_pos + [{"label": 0}] * (4 - n_pos)
        per = [{"correct": 1}] * 200 + [{"correct": 0}] * 200
        return per, scan

    chosen, per, scan = _load().choose_and_measure(fake_gather, [0.1, 0.2, 0.3, 0.4])
    assert chosen == 0.3
    assert all(0 not in f for f in asked[:-1]), "fold 0 must never be used to choose the threshold"
    assert 0 in asked[-1], "but it must be included in the final measurement"
