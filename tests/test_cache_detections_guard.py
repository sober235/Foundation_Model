import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/cache_detections.py"
    spec = importlib.util.spec_from_file_location("cache_detections", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_run_dir_is_the_gate_0_model_for_the_given_fold():
    m = _load()
    assert m.default_run_dir(0) == Path("runs/detector_gate0_fold0")
    assert m.default_run_dir(3) == Path("runs/detector_gate0_fold3")


def test_refuse_existing_cache_passes_on_a_fresh_directory(tmp_path):
    m = _load()
    m.refuse_existing_cache(tmp_path / "fold0", overwrite=False)      # does not exist yet: fine


def test_refuse_existing_cache_passes_on_an_empty_existing_directory(tmp_path):
    m = _load()
    d = tmp_path / "fold0"
    d.mkdir()
    m.refuse_existing_cache(d, overwrite=False)


def test_refuse_existing_cache_refuses_when_pkl_files_are_already_there(tmp_path):
    m = _load()
    d = tmp_path / "fold0"
    d.mkdir()
    (d / "file1000000__clean.pkl").write_bytes(b"")
    with pytest.raises(SystemExit):
        m.refuse_existing_cache(d, overwrite=False)


def test_refuse_existing_cache_allows_overwrite(tmp_path):
    m = _load()
    d = tmp_path / "fold0"
    d.mkdir()
    (d / "file1000000__clean.pkl").write_bytes(b"")
    m.refuse_existing_cache(d, overwrite=True)                        # no exception
