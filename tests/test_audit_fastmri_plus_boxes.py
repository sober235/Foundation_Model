# tests/test_audit_fastmri_plus_boxes.py
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_fastmri_plus_boxes.py"
    spec = importlib.util.spec_from_file_location("audit_boxes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_box_contrast_is_high_on_the_bright_block_and_about_one_elsewhere():
    m = _load()
    img = np.ones((100, 100), np.float32)
    img[40:50, 60:70] = 10.0
    assert m.box_contrast(img, 40, 50, 60, 70) > 5.0
    assert m.box_contrast(img, 10, 20, 10, 20) == pytest.approx(1.0)


def test_volume_verdict_and_series():
    m = _load()
    assert m.volume_verdict([1.0, 1.1], [1.6, 1.4]) == "converted"
    assert m.volume_verdict([1.6, 1.4], [1.0, 1.1]) == "as_is"
    assert m.volume_verdict([1.0], [1.0]) == "tie"
    assert m.series_of("file_brain_AXFLAIR_203_6000123") == "203"


def test_decide_applies_the_organ_and_series_thresholds():
    m = _load()
    ok = {"knee": {"n": 100, "converted": 95, "per_series": {}},
          "brain": {"n": 100, "converted": 92, "per_series": {"200": {"n": 50, "converted": 45}, "203": {"n": 10, "converted": 8},
                                                             "205": {"n": 1, "converted": 0}}}}
    assert m.decide(ok) is True
    bad_series = {"knee": ok["knee"], "brain": {**ok["brain"], "per_series": {**ok["brain"]["per_series"], "203": {"n": 10, "converted": 5}}}}
    assert m.decide(bad_series) is False
    bad_knee = {**ok, "knee": {"n": 100, "converted": 80, "per_series": {}}}
    assert m.decide(bad_knee) is False
