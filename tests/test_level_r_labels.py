import csv
import sys

import pytest

from anatobind.eval.level_r_labels import SealIntegrityError, SealedAccessError, load_fold, load_test_labels, load_train_labels
from anatobind.level_r.admin import FINAL_COLUMNS, export_csvs, seal
from anatobind.level_r.store import Store


@pytest.fixture
def sealed(tmp_path):
    rows = [{"lesion_id": i, "status": "agreed" if i % 2 else "adjudicated", "primary_host": "white_matter" if i < 8 else None,
             "acceptable_hosts": '["white_matter"]' if i < 8 else "[]", "not_a_lesion": i >= 8,
             "lesion_type": "nonspecific_wm_lesion" if i < 8 else None, "side": "image_left" if i < 8 else None,
             "lobe": "frontal" if i < 8 and i != 6 else None} for i in range(10)]
    final = tmp_path / "final.csv"
    with open(final, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    seal(final, {i: i % 5 for i in range(10)}, tmp_path / "sealed", tmp_path / "manifest.json", k=5, now="2026-10-01T00:00:00+00:00")
    return tmp_path / "sealed", tmp_path / "manifest.json"


def test_load_fold_parses_and_verifies(sealed):
    d, m = sealed
    rows = load_fold(3, d, m)
    assert [r["lesion_id"] for r in rows] == [3, 8]
    assert rows[0] == {"lesion_id": 3, "status": "agreed", "primary_host": "white_matter", "acceptable_hosts": ["white_matter"], "not_a_lesion": False,
                       "lesion_type": "nonspecific_wm_lesion", "side": "image_left", "lobe": "frontal"}
    assert rows[1]["not_a_lesion"] is True and rows[1]["primary_host"] is None and rows[1]["acceptable_hosts"] == []
    assert rows[1]["lesion_type"] is None and rows[1]["side"] is None and rows[1]["lobe"] is None
    assert [r["lobe"] for r in load_fold(1, d, m)] == ["frontal", None]            # lesion 6: an empty lobe reads back as None


def test_lesion_type_side_and_lobe_survive_export_seal_and_load(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.add_reader("r1", "reader", "0123456789abcdef", "读者 1")
    s.add_reader("r2", "reader", "fedcba9876543210", "读者 2")
    s.load_lesions([{"lesion_id": i, "code": f"c{i}", "volume_code": "v", "z0": 0, "z1": 0, "boxes": {}} for i in range(4)])
    wm = {"primary_host": "white_matter", "acceptable_hosts": ["white_matter"], "topography": "deep_white_matter", "adjacency": ["none"],
          "ambiguity": "certain", "lesion_type": "lacunar_infarct", "side": "image_right", "lobe": "temporal", "local_quality": "good", "confidence": 4}
    nal = {"not_a_lesion": True, "local_quality": "poor", "confidence": 3}
    for lid, (a, b) in {0: (wm, wm), 1: (wm, {**wm, "lobe": "insular"}), 2: (wm, {**wm, "side": "midline"}), 3: (nal, nal)}.items():
        s.submit_label("r1", lid, a)
        s.submit_label("r2", lid, b)
    s.submit_adjudication("adj", 2, {**wm, "side": "midline", "lobe": "not_applicable", "reason": "跨中线"})
    final = next(p for p in export_csvs(s, tmp_path / "export", now="2026-10-01T08:00:00+00:00") if p.name == "final_labels.csv")
    seal(final, {0: 0, 1: 0, 2: 1, 3: 1}, tmp_path / "sealed", tmp_path / "manifest.json", k=2, now="2026-10-01T08:00:00+00:00")
    rows = {r["lesion_id"]: r for f in (0, 1) for r in load_fold(f, tmp_path / "sealed", tmp_path / "manifest.json")}
    assert [(rows[i]["status"], rows[i]["lesion_type"], rows[i]["side"], rows[i]["lobe"]) for i in range(4)] == [
        ("agreed", "lacunar_infarct", "image_right", "temporal"), ("agreed", "lacunar_infarct", "image_right", None),
        ("adjudicated", "lacunar_infarct", "midline", "not_applicable"), ("agreed", None, None, None)]


def test_train_labels_exclude_the_held_out_fold(sealed):
    d, m = sealed
    train = load_train_labels(0, d, m)
    assert sorted(r["lesion_id"] for r in train) == [1, 2, 3, 4, 6, 7, 8, 9]


BAD_FOLD_IDS = [5, -1, "0", True]           # out of range, negative, a string that formats as fold0, a bool


@pytest.mark.parametrize("k", BAD_FOLD_IDS)
def test_train_labels_refuse_fold_ids_that_do_not_name_a_sealed_fold(sealed, k):
    d, m = sealed
    with pytest.raises(SealedAccessError):
        load_train_labels(k, d, m)


@pytest.mark.parametrize("k", BAD_FOLD_IDS)
def test_load_fold_refuses_fold_ids_that_do_not_name_a_sealed_fold(sealed, k):
    d, m = sealed
    with pytest.raises(SealedAccessError):
        load_fold(k, d, m)


@pytest.mark.parametrize("k", BAD_FOLD_IDS)
def test_test_labels_refuse_a_bad_fold_id_before_writing_the_log(sealed, k):
    d, m = sealed
    with pytest.raises(SealedAccessError):
        load_test_labels(k, unblind=True, sealed_dir=d, manifest_path=m)
    assert not (d / "access_log.txt").exists()


def test_tampering_is_detected(sealed):
    d, m = sealed
    p = d / "labels_fold1.csv"
    p.write_bytes(p.read_bytes().replace(b"white_matter", b"cortex______", 1))
    with pytest.raises(SealIntegrityError):
        load_fold(1, d, m)
    with pytest.raises(SealIntegrityError):
        load_train_labels(0, d, m)


def test_test_fold_needs_the_explicit_flag_and_logs_every_access(sealed):
    d, m = sealed
    with pytest.raises(SealedAccessError):
        load_test_labels(2, sealed_dir=d, manifest_path=m)
    with pytest.raises(SealedAccessError):
        load_test_labels(2, unblind=1, sealed_dir=d, manifest_path=m)               # only the literal True
    assert not (d / "access_log.txt").exists()
    rows = load_test_labels(2, unblind=True, sealed_dir=d, manifest_path=m)
    assert [r["lesion_id"] for r in rows] == [2, 7]
    log = (d / "access_log.txt").read_text(encoding="utf-8").strip().splitlines()
    ts, fold, caller, entry = log[0].split("\t")                                # time, fold, calling file, entry script
    assert len(log) == 1 and ts and fold == "fold2" and caller.endswith("test_level_r_labels.py") and entry == sys.argv[0]
    load_test_labels(2, unblind=True, sealed_dir=d, manifest_path=m, log_path=d / "other.log")
    assert (d / "other.log").read_text().count("\n") == 1
