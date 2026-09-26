import csv

import pytest

from anatobind.eval.level_r_labels import SealIntegrityError, SealedAccessError, load_fold, load_test_labels, load_train_labels
from anatobind.level_r.admin import FINAL_COLUMNS, seal


@pytest.fixture
def sealed(tmp_path):
    rows = [{"lesion_id": i, "status": "agreed" if i % 2 else "adjudicated", "primary_host": "white_matter" if i < 8 else None,
             "acceptable_hosts": '["white_matter"]' if i < 8 else "[]", "not_a_lesion": i >= 8} for i in range(10)]
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
    assert rows[0] == {"lesion_id": 3, "status": "agreed", "primary_host": "white_matter", "acceptable_hosts": ["white_matter"], "not_a_lesion": False}
    assert rows[1]["not_a_lesion"] is True and rows[1]["primary_host"] is None and rows[1]["acceptable_hosts"] == []


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
    log = (d / "access_log.txt").read_text().strip().splitlines()
    assert len(log) == 1 and "\tfold2\t" in log[0] and log[0].endswith("test_level_r_labels.py")
    load_test_labels(2, unblind=True, sealed_dir=d, manifest_path=m, log_path=d / "other.log")
    assert (d / "other.log").read_text().count("\n") == 1
