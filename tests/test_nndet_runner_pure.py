import importlib.util
from pathlib import Path

import numpy as np
import pytest

RUNNER = Path(__file__).resolve().parents[1] / "scripts/nndet_runner.py"


def _runner():
    spec = importlib.util.spec_from_file_location("nndet_runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_undo_margin_shifts_only_the_low_ends():
    r = _runner()
    b = np.array([[0.0, 1.0, 5.0, 6.0, 2.0, 4.0]])
    assert r.undo_margin(b).tolist() == [[1.0, 2.0, 5.0, 6.0, 3.0, 4.0]]
    assert b.tolist() == [[0.0, 1.0, 5.0, 6.0, 2.0, 4.0]]
    assert r.undo_margin(np.zeros((0, 6))).shape == (0, 6)


def test_runner_imports_neither_anatobind_nor_nndetection_at_module_level():
    top = [l for l in RUNNER.read_text().splitlines() if l.startswith(("import ", "from "))]
    assert not any(w in l for l in top for w in ("anatobind", "nndet", "torch", "omegaconf"))


def test_cli_refuses_an_existing_out_before_touching_nndetection(tmp_path):
    out = tmp_path / "x.json"
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        _runner().main(["gt", "--prep", str(tmp_path), "--out", str(out)])
