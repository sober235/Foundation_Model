import csv
import importlib.util
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from anatobind.level_r.admin import (
    ADJ_COLUMNS, FINAL_COLUMNS, LABEL_COLUMNS, backup_db, export_csvs, lesion_folds, make_order, make_patient_folds, seal, sha256_file,
)
from anatobind.level_r.store import Store

WM = {"primary_host": "white_matter", "acceptable_hosts": ["white_matter"], "topography": "deep_white_matter", "adjacency": ["none"],
      "ambiguity": "certain", "not_a_lesion": False, "lesion_type": "nonspecific_wm_lesion", "side": "image_left", "lobe": "frontal",
      "local_quality": "good", "confidence": 5, "comment": "a,b"}
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
    for lid, (a, b) in {0: (WM, WM), 1: (WM, CX), 2: (CX, {**CX, "lobe": "parietal"}), 3: (WM, CX)}.items():
        s.submit_label("r1", lid, {**a, "time_seconds": 10.0, "window": [1, 2]})
        s.submit_label("r2", lid, b)
    s.submit_label("r1", 0, CX)                                       # history: r1 changed lesion 0 -> now a disagreement
    s.submit_adjudication("adj", 1, {**CX, "reason": "皮层"})
    return s


NOW = "2026-10-01T08:00:00+00:00"
STAMP = "2026-10-01T08-00-00"


def test_export_csvs_writes_history_adjudications_and_final_status(tmp_path):
    s = _filled_store(tmp_path)
    files = export_csvs(s, tmp_path / "export", now=NOW)
    out = tmp_path / "export" / STAMP                                  # a fresh timestamped directory per export
    assert sorted(p.name for p in files) == ["adjudications.csv", "final_labels.csv", "labels_r1.csv", "labels_r2.csv"]
    assert all(p.parent == out for p in files)
    r1 = list(csv.DictReader(open(out / "labels_r1.csv", newline="", encoding="utf-8")))
    assert len(r1) == 5 and r1[0]["comment"] == "a,b" and json.loads(r1[0]["acceptable_hosts"]) == ["white_matter"] and r1[0]["window"] == "[1, 2]"
    assert list(r1[0]) == list(LABEL_COLUMNS) and (r1[0]["lesion_type"], r1[0]["side"], r1[0]["lobe"]) == ("nonspecific_wm_lesion", "image_left", "frontal")
    final = {int(r["lesion_id"]): r for r in csv.DictReader(open(out / "final_labels.csv", newline="", encoding="utf-8"))}
    assert final[0]["status"] == "pending" and final[1]["status"] == "adjudicated" and final[2]["status"] == "agreed" and final[3]["status"] == "pending"
    assert list(final[2]) == list(FINAL_COLUMNS) and final[2]["primary_host"] == "cortex"
    assert (final[2]["lesion_type"], final[2]["side"], final[2]["lobe"]) == ("nonspecific_wm_lesion", "image_left", "")   # lobes differ
    assert (final[1]["lesion_type"], final[1]["side"], final[1]["lobe"]) == ("nonspecific_wm_lesion", "image_left", "frontal")
    assert final[0]["lesion_type"] == final[0]["side"] == final[0]["lobe"] == ""                                  # pending
    adj = list(csv.DictReader(open(out / "adjudications.csv", newline="", encoding="utf-8")))
    assert len(adj) == 1 and adj[0]["reason"] == "皮层"
    assert list(adj[0]) == list(ADJ_COLUMNS) and (adj[0]["lesion_type"], adj[0]["side"], adj[0]["lobe"]) == ("nonspecific_wm_lesion", "image_left", "frontal")
    assert "皮层".encode("utf-8") in (out / "adjudications.csv").read_bytes()


def test_export_refuses_an_existing_stamp_and_never_touches_earlier_exports(tmp_path):
    s = _filled_store(tmp_path)
    export_csvs(s, tmp_path / "export", now=NOW)
    before = (tmp_path / "export" / STAMP / "labels_r1.csv").read_bytes()
    s.submit_label("r1", 2, WM)
    with pytest.raises(FileExistsError):
        export_csvs(s, tmp_path / "export", now=NOW)
    assert (tmp_path / "export" / STAMP / "labels_r1.csv").read_bytes() == before
    later = export_csvs(s, tmp_path / "export", now="2026-10-01T09:30:00+00:00")
    assert later[0].parent.name == "2026-10-01T09-30-00" and len(list(csv.DictReader(open(later[0], newline="", encoding="utf-8")))) == 6


def test_export_without_two_readers_raises_and_writes_nothing(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.add_reader("r1", "reader", "0123456789abcdef", "读者 1")
    with pytest.raises(ValueError):
        export_csvs(s, tmp_path / "export", now=NOW)
    assert not (tmp_path / "export").exists()


def _final_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def _agreed(ids):
    return [{"lesion_id": i, "status": "agreed", "primary_host": "white_matter", "acceptable_hosts": '["white_matter"]', "not_a_lesion": False,
             "lesion_type": "nonspecific_wm_lesion", "side": "image_left", "lobe": "frontal"} for i in ids]


def test_seal_splits_by_fold_writes_sha256_and_is_one_shot(tmp_path):
    final = _final_csv(tmp_path / "final.csv", _agreed(range(10)))
    lesion_fold = {i: i % 5 for i in range(10)}
    man = seal(final, lesion_fold, tmp_path / "sealed", tmp_path / "manifest.json", k=5, now="2026-10-01T00:00:00+00:00")
    assert set(man) == {f"fold{k}" for k in range(5)} and all(m["rows"] == 2 and m["sealed_at"] == "2026-10-01T00:00:00+00:00" for m in man.values())
    for k in range(5):
        p = tmp_path / "sealed" / f"labels_fold{k}.csv"
        assert man[f"fold{k}"]["sha256"] == sha256_file(p) and man[f"fold{k}"]["path"] == str(p)
        rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
        assert [int(r["lesion_id"]) for r in rows] == [k, k + 5]
        assert all((r["lesion_type"], r["side"], r["lobe"]) == ("nonspecific_wm_lesion", "image_left", "frontal") for r in rows)
    assert json.loads((tmp_path / "manifest.json").read_text()) == man
    with pytest.raises(FileExistsError):
        seal(final, lesion_fold, tmp_path / "sealed", tmp_path / "manifest2.json", k=5)
    with pytest.raises(FileExistsError):
        seal(final, lesion_fold, tmp_path / "sealed2", tmp_path / "manifest.json", k=5)


def test_seal_writes_every_fold_of_the_table_even_an_empty_one(tmp_path):
    final = _final_csv(tmp_path / "final.csv", _agreed(range(8)))
    man = seal(final, {i: i % 4 for i in range(8)}, tmp_path / "sealed", tmp_path / "manifest.json", k=5)
    assert [man[f"fold{f}"]["rows"] for f in range(5)] == [2, 2, 2, 2, 0]              # k = 5 from the table, not max fold + 1
    assert (tmp_path / "sealed" / "labels_fold4.csv").read_text(encoding="utf-8").strip() == ",".join(FINAL_COLUMNS)


def test_seal_refuses_pending_rows(tmp_path):
    final = _final_csv(tmp_path / "final.csv", [{"lesion_id": 0, "status": "pending", "primary_host": "", "acceptable_hosts": "[]", "not_a_lesion": ""}])
    with pytest.raises(ValueError):
        seal(final, {0: 0}, tmp_path / "sealed", tmp_path / "manifest.json", k=1)


@pytest.mark.parametrize("ids, message", [
    (range(8), "2 missing, 0 extra, 0 duplicated"),                    # partial: lesions 8 and 9 absent
    ([], "10 missing, 0 extra, 0 duplicated"),                         # empty: header only
    (list(range(10)) + [3], "0 missing, 0 extra, 1 duplicated"),       # lesion 3 twice
    (range(11), "0 missing, 1 extra, 0 duplicated"),                   # lesion 10 is not in the fold table
])
def test_seal_refuses_anything_but_each_lesion_exactly_once_and_writes_nothing(tmp_path, ids, message):
    final = _final_csv(tmp_path / "final.csv", _agreed(ids))
    with pytest.raises(ValueError, match=message):
        seal(final, {i: i % 5 for i in range(10)}, tmp_path / "sealed", tmp_path / "manifest.json", k=5)
    assert not (tmp_path / "sealed").exists() and not (tmp_path / "manifest.json").exists()


def _admin_script():
    path = Path(__file__).resolve().parents[1] / "scripts/level_r_admin.py"
    spec = importlib.util.spec_from_file_location("level_r_admin_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("argv", [
    ["add-reader", "--reader-id", "r1", "--role", "reader", "--display", "读者 1"],
    ["order", "--reader-id", "r1", "--seed", "1"],
    ["export", "--out", "EXPORT"],
    ["backup", "--out", "BACKUP"],
    ["release", "--reader-id", "r1"],
])
def test_admin_subcommands_other_than_init_refuse_a_missing_database(tmp_path, monkeypatch, argv):
    db = tmp_path / "typo.sqlite"
    argv = [str(tmp_path / a) if a in ("EXPORT", "BACKUP") else a for a in argv]
    monkeypatch.setattr("sys.argv", ["level_r_admin.py", argv[0], "--db", str(db), *argv[1:]])
    with pytest.raises(sqlite3.OperationalError):
        _admin_script().main()
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_release_command_lets_a_reader_continue_past_the_pilot(tmp_path, monkeypatch, capsys):
    s = _filled_store(tmp_path)
    monkeypatch.setattr("sys.argv", ["level_r_admin.py", "release", "--db", str(tmp_path / "db.sqlite"), "--reader-id", "r2"])
    _admin_script().main()
    assert s.is_released("r2") and not s.is_released("r1") and "r2" in capsys.readouterr().out


def test_seal_command_takes_the_fold_count_from_folds_json(tmp_path, monkeypatch):
    from test_level_r_registry import _row, write_registry     # tests/ is on sys.path under pytest
    reg = write_registry(tmp_path / "reg.csv", [_row(0, patient="P1"), _row(1, patient="P2")])
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({"k": 5, "seed": 0, "patient_fold": {"P1": 0, "P2": 1}}))
    final = _final_csv(tmp_path / "final.csv", _agreed(range(2)))
    monkeypatch.setattr("sys.argv", ["level_r_admin.py", "seal", "--final", str(final), "--folds", str(folds), "--registry", str(reg),
                                     "--out", str(tmp_path / "sealed"), "--manifest", str(tmp_path / "manifest.json")])
    _admin_script().main()
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert [man[f"fold{f}"]["rows"] for f in range(5)] == [1, 1, 0, 0, 0] and len(man) == 5


def test_backup_copies_the_live_database_and_never_overwrites(tmp_path):
    s = _filled_store(tmp_path)
    p = backup_db(tmp_path / "db.sqlite", tmp_path / "backup", now="2026-10-01T08:00:00+00:00")
    assert p.name == "level_r_2026-10-01T08-00-00.sqlite" and Store(p).label_rows() == s.label_rows()
    with pytest.raises(FileExistsError):
        backup_db(tmp_path / "db.sqlite", tmp_path / "backup", now="2026-10-01T08:00:00+00:00")


def test_backup_of_a_missing_database_raises_and_creates_nothing(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        backup_db(tmp_path / "typo.sqlite", tmp_path / "backup", now="2026-10-01T08:00:00+00:00")
    assert not (tmp_path / "typo.sqlite").exists() and not (tmp_path / "backup").exists()
