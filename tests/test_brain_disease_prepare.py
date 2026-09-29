import importlib.util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/brain_disease_prepare.py"
    spec = importlib.util.spec_from_file_location("brain_disease_prepare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_surgery_flags_read_yes_and_no_and_refuse_anything_else(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("SubjectID,Sex,Prior Craniotomy/Biopsy/Resection\n100101A,Male,No\n100202C,Female,Yes\n")
    assert _load().surgery_flags(p) == {"100101A": {"prior_surgery": False}, "100202C": {"prior_surgery": True}}
    p.write_text("SubjectID,Prior Craniotomy/Biopsy/Resection\n100101A,maybe\n")
    with pytest.raises(ValueError, match="100101A: unexpected surgery value 'maybe'"):
        _load().surgery_flags(p)


def test_expected_counts_match_the_spec():
    assert _load().EXPECTED == {"glioma": (501, 495), "metastasis": (461, 314), "infarct": (250, 250)}
