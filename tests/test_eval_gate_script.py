import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/eval_gate.py"
    spec = importlib.util.spec_from_file_location("eval_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path, informative):
    """informative=True: head_score 与正确性强相关，peak_score 与之无关。"""
    from anatobind.data_engine.fastmri_knee import VIEWS
    rng = np.random.default_rng(0)
    root = tmp_path / "leg2"
    rows = []
    for i in range(60):
        f = f"file{i}"
        (root / f).mkdir(parents=True)
        (root / f / "meta.json").write_text(json.dumps({"patient_id": f"p{i // 2}"}))
        for v in VIEWS:
            correct = float(rng.integers(0, 2))
            peak = float(rng.uniform(0, 1))
            head = correct * 0.5 + rng.uniform(0, 0.5) if informative else peak
            for kind in ("lesion", "scan"):
                rows.append({"kind": kind, "file": f, "view": v, "fold": i % 5, "patient": "",
                             "head_score": head, "peak_score": peak, "correct": correct,
                             "min_lesion": peak})
    path = tmp_path / "scores.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path, root


def test_h2_passes_when_the_head_knows_something_the_peak_score_does_not(tmp_path):
    scores, root = _write(tmp_path, informative=True)
    out = tmp_path / "report"
    _load().main(["--scores", str(scores), "--export-root", str(root), "--out", str(out),
                  "--figs", str(tmp_path / "figs"), "--reps", "200"])
    s = json.loads((out / "summary.json").read_text())
    assert s["h2"]["pass"] is True
    assert (tmp_path / "figs/risk_coverage.png").exists()
    assert (tmp_path / "figs/coverage_at_risk.png").exists()


def test_h2_fails_when_the_head_only_repeats_the_peak_score(tmp_path):
    scores, root = _write(tmp_path, informative=False)
    out = tmp_path / "report"
    _load().main(["--scores", str(scores), "--export-root", str(root), "--out", str(out),
                  "--figs", str(tmp_path / "figs"), "--reps", "200"])
    assert json.loads((out / "summary.json").read_text())["h2"]["pass"] is False
