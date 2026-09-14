import numpy as np
import pytest

from anatobind.eval.gate import GATE_BINS, VIEW_ORDER, aurc, h2, h3, risk_coverage, threshold_at_risk


def test_risk_coverage_is_monotone_in_coverage_for_a_perfect_ranker():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    labels = np.array([1, 1, 1, 0, 0])          # 1 = correct
    cov, risk = risk_coverage(scores, labels)
    assert cov[0] < cov[-1] and risk[0] == 0.0
    assert risk[-1] == pytest.approx(0.4)


def test_a_perfect_ranker_beats_a_random_one_on_aurc():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    good = labels + rng.normal(0, 0.01, 200)
    bad = rng.normal(0, 1, 200)
    assert aurc(good, labels) < aurc(bad, labels)


def test_the_threshold_meets_the_requested_risk():
    scores = np.concatenate([np.linspace(0.6, 1.0, 90), np.linspace(0.0, 0.5, 10)])
    labels = np.concatenate([np.ones(90), np.zeros(10)])
    thr = threshold_at_risk(scores, labels, r_max=0.05)
    kept = scores >= thr
    assert kept.sum() > 0 and (1 - labels[kept].mean()) <= 0.05


def test_an_impossible_risk_target_returns_infinity():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 0, 1, 1])
    assert threshold_at_risk(scores, labels, r_max=0.01) == float("inf")


# ---------------------------------------------------------------------------
# h2 and h3 are the pre-registered criteria (spec 3.4) and had no unit tests of
# their own in the brief beyond the end-to-end script test. These lock down the
# decision logic directly, at the level of the dicts h2()/h3() return.
# ---------------------------------------------------------------------------

def _lesion_rows(n_patients=30, seed=0, bad_view=None, degenerate_view=None):
    """One synthetic per-lesion row per (patient, view). By default head_score tracks
    `correct` tightly while peak_score is uncorrelated noise, so the head clearly beats
    the peak in every bin. `bad_view` makes that one view's head_score equal peak_score
    (the head knows nothing extra there). `degenerate_view` forces that view's labels to
    be all-correct, i.e. no usable variation to rank against."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        patient = f"p{p}"
        for v in VIEW_ORDER:
            correct = 1.0 if v == degenerate_view else float(rng.integers(0, 2))
            peak = float(rng.uniform(0, 1))
            head = peak if v == bad_view else correct + rng.uniform(0, 0.05)
            rows.append({"view": v, "patient": patient, "correct": correct,
                         "peak_score": peak, "head_score": head})
    return rows


def test_h2_passes_when_the_head_beats_the_peak_in_every_degraded_bin():
    res = h2(_lesion_rows(), reps=200)
    assert all(res[v]["pass"] for v in GATE_BINS)
    assert res["pass"] is True


def test_h2_fails_when_a_degraded_bin_does_not_beat_the_peak():
    res = h2(_lesion_rows(bad_view="noise_q2"), reps=200)
    assert res["noise_q2"]["delta_aurc"] == pytest.approx(0.0)
    assert res["noise_q2"]["pass"] is False
    assert res["pass"] is False


def test_h2_verdict_ignores_the_clean_bin():
    # clean is made uninformative (head == peak there); every degraded bin still wins.
    res = h2(_lesion_rows(bad_view="clean"), reps=200)
    assert res["clean"]["pass"] is False
    assert all(res[v]["pass"] for v in GATE_BINS)
    assert res["pass"] is True


def test_h2_fails_rather_than_passes_when_a_bin_has_no_usable_data():
    # one degraded bin is all-correct: AURC there is undefined, not a free win.
    res = h2(_lesion_rows(degenerate_view="us8"), reps=200)
    assert not np.isfinite(res["us8"]["delta_aurc"])
    assert res["us8"]["pass"] is False
    assert res["pass"] is False


def _scan_rows(n_patients, correct_frac, head_fn, min_fn=None, seed=0):
    """One synthetic scan row per (patient, view). correct_frac[view] is the fraction of
    patients correct in that view (the first that many patients, by index, are correct).
    head_fn/min_fn(view, correct, patient_index, rng) -> score."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        patient = f"p{i}"
        for v in VIEW_ORDER:
            correct = 1.0 if i < round(correct_frac.get(v, 1.0) * n_patients) else 0.0
            head = head_fn(v, correct, i, rng)
            rows.append({"view": v, "patient": patient, "correct": correct, "head_score": head,
                         "min_lesion": min_fn(v, correct, i, rng) if min_fn else head})
    return rows


# clean..us4 are perfectly correct; us8 is a coin flip; us16 is mostly wrong. A blunt
# acceleration cutoff can only keep or drop a whole view, so at risk<=0.05 it must drop
# both us8 and us16 even though most of us16 is salvageable scan by scan.
CORRECT_FRAC = {"clean": 1.0, "noise_q1": 1.0, "noise_q2": 1.0, "noise_q3": 1.0,
                "us4": 1.0, "us8": 0.5, "us16": 0.1}


def test_h3_passes_when_the_learned_gate_beats_the_acceleration_rule():
    rows = _scan_rows(40, CORRECT_FRAC, head_fn=lambda v, c, i, rng: c + rng.uniform(0, 0.05))
    res = h3(rows, reps=200)
    assert res["coverage_head"] > res["coverage_rule"]
    assert res["vs_rule"]["pass"] is True
    assert res["pass"] is True


def test_h3_fails_when_the_learned_gate_does_not_beat_the_acceleration_rule():
    rows = _scan_rows(40, CORRECT_FRAC, head_fn=lambda v, c, i, rng: 1.0 - c)
    res = h3(rows, reps=200)
    assert res["coverage_head"] < res["coverage_rule"]
    assert res["vs_rule"]["pass"] is False
    assert res["pass"] is False


def test_h3_verdict_is_decided_by_the_rule_comparison_not_the_min_baseline():
    """head clearly beats the acceleration rule but is narrowly beaten by the
    min-of-lesion-scores baseline: the overall verdict must follow vs_rule and pass,
    even though vs_min on its own would fail."""
    def head_fn(v, c, i, rng):
        head = c + rng.uniform(0, 0.05)
        if v == "us16" and c == 0.0 and 4 <= i < 9:    # 5 wrong us16 scans boosted high
            head = 2.0 + rng.uniform(0, 0.05)
        if v == "clean" and c == 1.0 and i < 5:        # 5 correct clean scans pushed low
            head = -1.0 + rng.uniform(0, 0.05)
        return head

    rows = _scan_rows(40, CORRECT_FRAC, head_fn=head_fn,
                       min_fn=lambda v, c, i, rng: c + rng.uniform(0, 0.01))
    res = h3(rows, reps=200)
    assert res["coverage_head"] > res["coverage_rule"]
    assert res["vs_rule"]["pass"] is True
    assert res["coverage_min_baseline"] > res["coverage_head"]
    assert res["vs_min"]["pass"] is False
    assert res["pass"] is True
