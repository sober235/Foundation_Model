import importlib.util
import json
from pathlib import Path

from anatobind.level_r.store import Store


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/level_r_report.py"
    spec = importlib.util.spec_from_file_location("level_r_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lab(host, acc=None, t=20.0):
    return {"primary_host": host, "acceptable_hosts": acc or [host], "topography": "deep_white_matter", "adjacency": ["none"],
            "ambiguity": "certain", "not_a_lesion": False, "local_quality": "good", "confidence": 4, "comment": "", "time_seconds": t}


def _registry():
    return [{"lesion_id": i, "patient_id": f"p{i // 2}", "band": "0" if i < 4 else ">4", "stratum_geometry": "inplane_0.69_slice_5" if i != 5 else "inplane_0.62_slice_3"}
            for i in range(6)]


def test_build_summary_and_render(tmp_path):
    m = _load()
    s = Store(tmp_path / "db.sqlite")
    s.add_reader("r2", "reader", "fedcba9876543210", "读者 2")
    s.add_reader("r1", "reader", "0123456789abcdef", "读者 1")
    s.add_reader("adj", "adjudicator", "aaaaaaaaaaaaaaaa", "裁定")
    s.load_lesions([{"lesion_id": i, "code": f"c{i}", "volume_code": "v", "z0": 0, "z1": 0, "boxes": {}} for i in range(6)])
    for i in range(6):
        s.submit_label("r1", i, _lab("white_matter", t=10.0 * (i + 1)))
        s.submit_label("r2", i, _lab("white_matter" if i < 5 else "cortex"))
    s.submit_label("r1", 0, _lab("white_matter", t=25.0))                # a revisit: lesion 0 took r1 10 + 25 s
    summary = m.build_summary(s, _registry(), lesion_ids=[0, 1, 2, 3, 4, 5], n_boot=100, seed=0)
    assert summary["readers"] == ["r1", "r2"] and summary["n_pairs"] == 6 and summary["raw"] == 5 / 6 and summary["pilot_size"] == 6
    assert summary["time"]["reader_a"]["median_s"] == 37.5                # 20 30 35 40 50 60 from the full history
    md = m.render_markdown(summary, title="pilot", command="python scripts/level_r_report.py --db x")
    for piece in ("# pilot", "GATE_R7:", "raw agreement", "positive agreement", "confusion", "band", "slice_3mm", "hours_for_1297",
                  "python scripts/level_r_report.py --db x", '"n_pairs": 6', "- r1 (reader A): n 6, median 37.5 s", "- r2 (reader B): n 6",
                  "## confusion (rows r1 = reader A, columns r2 = reader B)", "between r1 and r2", "Gwet AC1 (K = 8)",
                  "0 mm band raw 95% patient-bootstrap CI"):
        assert piece in md, piece
    assert ("GATE_R7: PASS" in md) == summary["gate_r7"]["single_host_endpoint_allowed"] and "⚠️" not in md
    sub = m.build_summary(s, _registry(), lesion_ids=[0, 1], n_boot=10, seed=0)
    assert sub["n_pairs"] == 2 and sub["raw"] == 1.0 and sub["time"]["reader_a"]["n"] == 2


def test_render_warns_when_pilot_lesions_lack_a_pair_and_appends_strict_json(tmp_path):
    m = _load()
    s = Store(tmp_path / "db.sqlite")
    s.add_reader("r1", "reader", "0123456789abcdef", "读者 1")
    s.add_reader("r2", "reader", "fedcba9876543210", "读者 2")
    s.load_lesions([{"lesion_id": i, "code": f"c{i}", "volume_code": "v", "z0": 0, "z1": 0, "boxes": {}} for i in range(6)])
    for i in (4, 5):
        s.submit_label("r1", i, _lab("cortex"))
    s.submit_label("r2", 4, _lab("cortex"))                              # r2 has not read pilot lesion 5 yet
    summary = m.build_summary(s, _registry(), lesion_ids=[4, 5], n_boot=10, seed=0)
    assert summary["n_pairs"] == 1 and summary["pilot_size"] == 2
    md = m.render_markdown(summary, title="pilot", command="x")
    assert "⚠️ pairs < pilot size" in md
    block = md.split("```json\n", 1)[1].split("\n```", 1)[0]
    assert "NaN" not in block and json.loads(block)["gate_r7"]["band0_raw"] is None     # no 0 mm pair: null, not NaN


def test_main_writes_the_report_and_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    m = _load()
    s = Store(tmp_path / "db.sqlite")
    s.add_reader("r1", "reader", "0123456789abcdef", "读者 1")
    s.add_reader("r2", "reader", "fedcba9876543210", "读者 2")
    s.load_lesions([{"lesion_id": i, "code": f"c{i}", "volume_code": "v", "z0": 0, "z1": 0, "boxes": {}} for i in range(2)])
    for i in range(2):
        s.submit_label("r1", i, _lab("cortex"))
        s.submit_label("r2", i, _lab("cortex"))
    reg = tmp_path / "reg.csv"
    import csv
    from test_level_r_registry import HEADER, _row      # tests/ is on sys.path under pytest, like conftest's `from synth import ...`
    with open(reg, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows([_row(0), _row(1, patient="P2")])
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"lesion_ids": [0, 1]}))
    out = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", ["level_r_report.py", "--db", str(tmp_path / "db.sqlite"), "--registry", str(reg), "--pilot", str(pilot),
                                     "--out", str(out), "--n-boot", "20"])
    m.main()
    assert out.exists() and "GATE_R7:" in out.read_text()
    import pytest
    with pytest.raises(SystemExit):
        m.main()


def test_main_refuses_a_missing_database(tmp_path, monkeypatch):
    import sqlite3

    import pytest
    m = _load()
    monkeypatch.setattr("sys.argv", ["level_r_report.py", "--db", str(tmp_path / "typo.sqlite"), "--out", str(tmp_path / "report.md")])
    with pytest.raises(sqlite3.OperationalError):
        m.main()
    assert list(tmp_path.iterdir()) == []
