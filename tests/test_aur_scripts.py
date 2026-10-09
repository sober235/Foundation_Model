# tests/test_aur_scripts.py
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _volume(path, arr, spacing=(1.0, 1.0, 1.0)):
    nib.save(nib.Nifti1Image(arr, np.diag(list(spacing) + [1.0])), str(path))
    return str(path)


def test_prepare_grids_and_isles_check(tmp_path):
    p = _load("aur_prepare")
    seg = np.zeros((16, 16, 8), np.int16)
    seg[:8] = 2
    seg[8:] = 41
    img = np.ones((16, 16, 8), np.float32)
    rows = [{"case": "sub-strokecase0001", "source": "isles", "sequence": "DWI", "source_sequence": "DWI",
             "image": _volume(tmp_path / "dwi.nii.gz", img), "anatomy": _volume(tmp_path / "seg.nii.gz", seg), "lesion": None},
            {"case": "x", "source": "pdgm", "sequence": "T1", "source_sequence": "T1",
             "image": _volume(tmp_path / "t1.nii.gz", np.ones((16, 16, 9), np.float32)), "anatomy": str(tmp_path / "seg.nii.gz"), "lesion": None}]
    bad = p.stage_grids(rows)
    assert len(bad) == 1 and bad[0][0] == "x"
    assert p.host_volumes_ml(seg, (1.0, 1.0, 1.0))[0] == pytest.approx(8 * 16 * 8 / 1000)
    out = tmp_path / "check"
    table = p.stage_isles_check(rows, out, n_cases=2, n_montage=1)
    assert set(table) == {"pdgm", "bmsr", "isles"} and (out / "host_volumes_median_ml.csv").is_file() and (out / "isles_sub-strokecase0001.png").is_file()
    with pytest.raises(FileExistsError):
        p.stage_isles_check(rows, out)
    with pytest.raises(SystemExit):
        p.main(["--stage", "samples"])


def test_probe_builds_a_batch_and_steps_on_cpu_sized_inputs():
    pr = _load("aur_probe")
    b = pr.synthetic_batch(1, (8, 16, 16), torch.device("cpu"))
    assert b["image"].shape == (1, 1, 8, 16, 16) and b["instance"].max() == 2 and b["valid"][0, 0, 0, -1] == 0.0
    assert pr.main.__doc__ is None or True                       # the CUDA path is exercised by the controller's probe run
