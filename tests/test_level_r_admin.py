import csv
import json
from collections import Counter

import pytest

from anatobind.level_r.admin import (
    FINAL_COLUMNS, backup_db, export_csvs, lesion_folds, make_order, make_patient_folds, seal, sha256_file,
)
from anatobind.level_r.store import Store

WM = {"primary_host": "white_matter", "acceptable_hosts": ["white_matter"], "topography": "deep_white_matter", "adjacency": ["none"],
      "ambiguity": "certain", "not_a_lesion": False, "local_quality": "good", "confidence": 5, "comment": "a,b"}
CX = {**WM, "primary_host": "cortex", "acceptable_hosts": ["cortex"]}


def test_patient_folds_are_balanced_seeded_and_patient_disjoint():
    pats = [f"p{i:03d}" for i in range(165)]
    f = make_patient_folds(pats, k=5, seed=0)
    assert set(f) == set(pats) and Counter(f.values()) == {0: 33, 1: 33, 2: 33, 3: 33, 4: 33}
    assert f == make_patient_folds(list(reversed(pats)), k=5, seed=0) and f != make_patient_folds(pats, k=5, seed=1)
    registry = [{"lesion_id": i, "patient_id": pats[i % 165]} for i in range(400)]
    lf = lesion_folds(registry, f)
    assert len(lf) == 400 and all(lf[i] == f[pats[i % 165]] for i in range(400))


def test_lesion_folds_calls_the_patient_disjoint_assertion(monkeypatch):
    """The cross-fold check itself is tested in tests/test_fastmri_knee_manifest.py; here we only require that
    lesion_folds routes through it with string keys."""
    import anatobind.level_r.admin as admin
    calls = []
    monkeypatch.setattr(admin, "assert_folds_by_patient", lambda folds, patients: calls.append((folds, patients)))
    lesion_folds([{"lesion_id": 0, "patient_id": "a"}, {"lesion_id": 1, "patient_id": "b"}], {"a": 0, "b": 1})
    assert calls == [({"0": 0, "1": 1}, {"0": "a", "1": "b"})]


def test_make_order_puts_pilot_first_and_is_a_seeded_permutation():
    ids = list(range(20))
    o = make_order(ids, pilot_ids={3, 7, 11}, seed=1)
    assert sorted(o) == ids and set(o[:3]) == {3, 7, 11} and o == make_order(ids, {3, 7, 11}, 1) and o != make_order(ids, {3, 7, 11}, 2)


def _filled_store(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.add_reader("r1", "reader", "0123456789abcdef", "读者 1")
    s.add_reader("r2", "reader", "fedcba9876543210", "读者 2")
    s.add_reader("adj", "adjudicator", "aaaaaaaaaaaaaaaa", "裁定")
    s.load_lesions([{"lesion_id": i, "code": f"c{i}", "volume_code": "v", "z0": 0, "z1": 0, "boxes": {}} for i in range(4)])
    for lid, (a, b) in {0: (WM, WM), 1: (WM, CX), 2: (CX, CX), 3: (WM, CX)}.items():
        s.submit_label("r1", lid, {**a, "time_seconds": 10.0, "window": [1, 2]})
        s.submit_label("r2", lid, b)
    s.submit_label("r1", 0, CX)                                       # history: r1 changed lesion 0 -> now a disagreement
    s.submit_adjudication("adj", 1, {**CX, "reason": "皮层"})
    return s


def test_export_csvs_writes_history_adjudications_and_final_status(tmp_path):
    s = _filled_store(tmp_path)
    files = export_csvs(s, tmp_path / "export")
    assert sorted(p.name for p in files) == ["adjudications.csv", "final_labels.csv", "labels_r1.csv", "labels_r2.csv"]
    r1 = list(csv.DictReader(open(tmp_path / "export/labels_r1.csv", newline="", encoding="utf-8")))
    assert len(r1) == 5 and r1[0]["comment"] == "a,b" and json.loads(r1[0]["acceptable_hosts"]) == ["white_matter"] and r1[0]["window"] == "[1, 2]"
    final = {int(r["lesion_id"]): r for r in csv.DictReader(open(tmp_path / "export/final_labels.csv", newline="", encoding="utf-8"))}
    assert final[0]["status"] == "pending" and final[1]["status"] == "adjudicated" and final[2]["status"] == "agreed" and final[3]["status"] == "pending"
    assert list(final[2]) == list(FINAL_COLUMNS) and final[2]["primary_host"] == "cortex"
    adj = list(csv.DictReader(open(tmp_path / "export/adjudications.csv", newline="", encoding="utf-8")))
    assert len(adj) == 1 and adj[0]["reason"] == "皮层"
    assert "皮层".encode("utf-8") in (tmp_path / "export/adjudications.csv").read_bytes()


def test_export_without_two_readers_still_writes_reader_files(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.add_reader("r1", "reader", "0123456789abcdef", "读者 1")
    files = export_csvs(s, tmp_path / "export")
    assert sorted(p.name for p in files) == ["adjudications.csv", "final_labels.csv", "labels_r1.csv"]
    assert open(tmp_path / "export/final_labels.csv", encoding="utf-8").read().strip() == ",".join(FINAL_COLUMNS)


def _final_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_seal_splits_by_fold_writes_sha256_and_is_one_shot(tmp_path):
    rows = [{"lesion_id": i, "status": "agreed", "primary_host": "white_matter", "acceptable_hosts": '["white_matter"]', "not_a_lesion": False} for i in range(10)]
    final = _final_csv(tmp_path / "final.csv", rows)
    lesion_fold = {i: i % 5 for i in range(10)}
    man = seal(final, lesion_fold, tmp_path / "sealed", tmp_path / "manifest.json", now="2026-10-01T00:00:00+00:00")
    assert set(man) == {f"fold{k}" for k in range(5)} and all(m["rows"] == 2 and m["sealed_at"] == "2026-10-01T00:00:00+00:00" for m in man.values())
    for k in range(5):
        p = tmp_path / "sealed" / f"labels_fold{k}.csv"
        assert man[f"fold{k}"]["sha256"] == sha256_file(p) and man[f"fold{k}"]["path"] == str(p)
        assert [int(r["lesion_id"]) for r in csv.DictReader(open(p, newline="", encoding="utf-8"))] == [k, k + 5]
    assert json.loads((tmp_path / "manifest.json").read_text()) == man
    with pytest.raises(FileExistsError):
        seal(final, lesion_fold, tmp_path / "sealed", tmp_path / "manifest2.json")
    with pytest.raises(FileExistsError):
        seal(final, lesion_fold, tmp_path / "sealed2", tmp_path / "manifest.json")


def test_seal_refuses_pending_rows(tmp_path):
    final = _final_csv(tmp_path / "final.csv", [{"lesion_id": 0, "status": "pending", "primary_host": "", "acceptable_hosts": "[]", "not_a_lesion": ""}])
    with pytest.raises(ValueError):
        seal(final, {0: 0}, tmp_path / "sealed", tmp_path / "manifest.json")


def test_backup_copies_the_live_database_and_never_overwrites(tmp_path):
    s = _filled_store(tmp_path)
    p = backup_db(tmp_path / "db.sqlite", tmp_path / "backup", now="2026-10-01T08:00:00+00:00")
    assert p.name == "level_r_2026-10-01T08-00-00.sqlite" and Store(p).label_rows() == s.label_rows()
    with pytest.raises(FileExistsError):
        backup_db(tmp_path / "db.sqlite", tmp_path / "backup", now="2026-10-01T08:00:00+00:00")
