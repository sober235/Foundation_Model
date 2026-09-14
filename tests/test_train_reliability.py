import csv
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/train_reliability.py"
    spec = importlib.util.spec_from_file_location("train_reliability", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _minimal_export(root, patient_of):
    """patient_of: {file_name: patient_id}. Writes meta.json per file, a header-only lesions.csv,
    and folds.json putting every file in fold 0."""
    root.mkdir(parents=True, exist_ok=True)
    for f, pid in patient_of.items():
        (root / f).mkdir()
        (root / f / "meta.json").write_text(json.dumps({"patient_id": pid}))
    (root / "lesions.csv").write_text("lesion_id,file,family,z0,z1,x0,y0,x1,y1,n_boxes\n")
    (root / "folds.json").write_text(json.dumps({"folds": {f: 0 for f in patient_of}}))


def _det_blob():
    return {"dets": [{"score": 0.9, "family": "meniscus", "z0": 0, "z1": 2,
                      "y0": 0, "x0": 0, "y1": 10, "x1": 10, "embed": np.zeros(4, np.float32)}],
            "global_feat": np.zeros(4, np.float32)}


def test_gather_reports_the_real_patient_not_the_volume_name(tmp_path):
    """Two volumes of one patient (e.g. the PD and fat-sat series of one knee) must collapse to the
    same patient id, not be counted as two independent patients."""
    from anatobind.data_engine.fastmri_knee import VIEWS
    root = tmp_path / "leg2"
    files = {"file_pd": "p0", "file_fs": "p0"}
    _minimal_export(root, files)

    det_root = tmp_path / "det"
    (det_root / "fold0").mkdir(parents=True)
    for f in files:
        with open(det_root / "fold0" / f"{f}__{VIEWS[0]}.pkl", "wb") as fh:
            pickle.dump(_det_blob(), fh)

    lesion_rows, scan_rows = _load().gather(root, det_root, 0, 0.1, by_file={})

    assert len(lesion_rows) == 2 and len(scan_rows) == 2
    assert {r["patient"] for r in lesion_rows} == {"p0"}
    assert {r["patient"] for r in scan_rows} == {"p0"}
    assert {r["file"] for r in lesion_rows} == set(files)   # file still distinguishes the two volumes


def test_main_skips_a_fold_with_no_scan_rows_instead_of_crashing(tmp_path, monkeypatch, capsys):
    """Regression: an empty fold (e.g. no cached detections yet) must be skipped with a message,
    not crash several frames deep into training with an IndexError."""
    root = tmp_path / "leg2"
    _minimal_export(root, {})                     # zero patients -> every fold is empty
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["train_reliability.py", "--score-min", "0.3",
                                     "--export-root", str(root), "--det-root", str(tmp_path / "det")])

    _load().main()

    printed = capsys.readouterr().out
    for k in range(5):
        assert f"fold {k}: no scan rows, skipping the scan head" in printed
    with open(tmp_path / "runs" / "reliability_scores.csv", newline="") as fh:
        assert list(csv.DictReader(fh)) == []
