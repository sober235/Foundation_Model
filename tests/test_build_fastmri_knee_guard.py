import importlib.util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/build_fastmri_knee.py"
    spec = importlib.util.spec_from_file_location("build_fastmri_knee", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_refuse_existing_export_passes_on_a_fresh_directory(tmp_path):
    m = _load()
    m.refuse_existing_export(tmp_path / "fresh", ["file1", "file2"])          # does not exist yet: fine


def test_refuse_existing_export_passes_on_an_empty_existing_directory(tmp_path):
    m = _load()
    out = tmp_path / "out"
    out.mkdir()
    m.refuse_existing_export(out, ["file1", "file2"])


def test_refuse_existing_export_refuses_when_manifest_csv_already_exists(tmp_path):
    m = _load()
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.csv").write_text("file,out_dir\n")
    with pytest.raises(SystemExit):
        m.refuse_existing_export(out, ["file1"])


def test_refuse_existing_export_refuses_when_a_volume_dir_is_a_symlink(tmp_path):
    m = _load()
    out = tmp_path / "out"
    out.mkdir()
    target = tmp_path / "legacy" / "file1"
    target.mkdir(parents=True)
    (out / "file1").symlink_to(target)
    with pytest.raises(SystemExit):
        m.refuse_existing_export(out, ["file1", "file2"])


def test_refuse_existing_export_allows_a_real_directory_that_is_not_a_symlink(tmp_path):
    m = _load()
    out = tmp_path / "out"
    (out / "file1").mkdir(parents=True)                                       # a resumable direct export, not a Gate 0 relink
    m.refuse_existing_export(out, ["file1", "file2"])
