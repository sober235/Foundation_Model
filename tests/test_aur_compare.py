# tests/test_aur_compare.py
"""Paired comparison of two evaluated arms on the same rows (C2 against C0; SSL-first plan §6, C0 is the control)."""
import json

import numpy as np
import pytest

import anatobind.aur.compare as C
from anatobind.aur.labels import HOST_NAMES

LWM, RWM = HOST_NAMES.index("white_matter_left"), HOST_NAMES.index("white_matter_right")


def _case(case, patient, source, dice, inst, dets_hit=True, sequence="FLAIR", a_supervised=True):
    gt = [{"instance": i + 1, "box": [i * 4, 0, 0, i * 4 + 2, 2, 2], "n_voxels": 8, "volume_mm3": 8.0, "stratum": "<5"} for i in range(len(inst))]
    dets = [{"instance": i + 1, "box": g["box"], "score": 0.9, "volume_mm3": 8.0} for i, g in enumerate(gt)] if dets_hit else []
    return {"case": case, "patient": patient, "source": source, "sequence": sequence, "a_supervised": a_supervised,
            "a": {"host_macro": dice, "hosts": {}, "entities": {}, "entity_macro": dice},
            "u": {"gt": gt, "dets": dets} if inst else None,
            "r": {"instances": [{"instance": i + 1, "truth": t, "r": r, "b0": b, "stratum": "<5", "volume_mm3": 8.0} for i, (t, r, b) in enumerate(inst)]} if inst else None}


def _arm(tmp_path, name, cases, thresholds):
    d = tmp_path / name
    d.mkdir()
    (d / "cases.json").write_text(json.dumps(cases))
    (d / "aggregate.json").write_text(json.dumps({"u": {"per_source": {s: {"thr": t} for s, t in thresholds.items()}}, "run": {"checkpoint": name}}))
    return d


def test_paired_comparison_of_the_three_tracks(tmp_path):
    a = [_case("a1", "p1", "pdgm", 0.9, [(LWM, LWM, RWM)]), _case("a2", "p2", "pdgm", 0.8, [(RWM, RWM, RWM)]),
         _case("b1", "p3", "bmsr", 0.7, [(LWM, LWM, LWM)], sequence="T1c"), _case("x", "p4", "isles", 0.1, [], a_supervised=False)]
    b = [_case("a1", "p1", "pdgm", 0.8, [(LWM, RWM, RWM)]), _case("a2", "p2", "pdgm", 0.8, [(RWM, RWM, RWM)], dets_hit=False),
         _case("b1", "p3", "bmsr", 0.6, [(LWM, LWM, LWM)], sequence="T1c"), _case("x", "p4", "isles", 0.5, [], a_supervised=False)]
    res = C.compare(_arm(tmp_path, "c2", a, {"pdgm": 0.5, "bmsr": 0.5}), _arm(tmp_path, "c0", b, {"pdgm": 0.5, "bmsr": 0.5}), n_boot=200, seed=0)
    assert res["n_rows"] == 4 and res["a"]["n"] == 3                         # the A-unsupervised row is left out
    assert res["a"]["mean_a"] == pytest.approx(0.8) and res["a"]["mean_b"] == pytest.approx(0.7333, abs=1e-4)
    assert res["a"]["diff"] == pytest.approx(0.0667, abs=1e-4) and res["a"]["ci95"][0] <= res["a"]["diff"] <= res["a"]["ci95"][1]
    u = res["u"]["pdgm/FLAIR"]
    assert u["sens_a"] == pytest.approx(1.0) and u["sens_b"] == pytest.approx(0.5) and u["n_gt"] == 2
    r = res["r"]
    assert r["n"] == 3 and r["aba_a"] == pytest.approx(1.0) and r["aba_b"] == pytest.approx(2 / 3)
    assert r["paired"] == {"both_right": 2, "a_only": 1, "b_only": 0, "both_wrong": 0}
    assert res["evidence"].startswith("NOT_EVIDENCE")
    text = C.markdown(res, "C2", "C0")
    assert "C2" in text and "C0" in text and "pdgm/FLAIR" in text
    json.dumps(res)


def test_the_arms_must_hold_the_same_rows(tmp_path):
    a = [_case("a1", "p1", "pdgm", 0.9, [])]
    b = [_case("a2", "p2", "pdgm", 0.9, [])]
    with pytest.raises(ValueError, match="same rows"):
        C.compare(_arm(tmp_path, "c2", a, {}), _arm(tmp_path, "c0", b, {}), n_boot=10)


def test_patient_bootstrap_moves_the_rows_of_a_patient_together():
    vals = np.array([1.0, 1.0, 0.0])
    lo, hi = C.patient_bootstrap(vals, ["p1", "p1", "p2"], n_boot=500, seed=0)
    assert 0.0 <= lo <= hi <= 1.0 and lo < 0.5 < hi
