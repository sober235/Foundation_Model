import csv

import numpy as np
import pytest

from anatobind.eval.level_r_labels import SealedAccessError
from anatobind.level_r.admin import FINAL_COLUMNS, seal
from anatobind.relation.labels import acceptable_matrix, c1_labels, from_level_r_rows, r_test_labels, r_train_labels
from synth_relation import synthetic_table


@pytest.fixture
def sealed(tmp_path):
    rows = [{"lesion_id": i, "status": "agreed", "primary_host": "white_matter" if i < 8 else None,
             "acceptable_hosts": '["white_matter", "cortex"]' if i == 1 else '["other"]' if i == 2 else '["white_matter"]' if i < 8 else "[]",
             "not_a_lesion": i >= 8, "lesion_type": None, "side": None, "lobe": None} for i in range(10)]
    final = tmp_path / "final.csv"
    with open(final, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    seal(final, {i: i % 5 for i in range(10)}, tmp_path / "sealed", tmp_path / "manifest.json", k=5, now="2026-10-01T00:00:00+00:00")
    return tmp_path / "sealed", tmp_path / "manifest.json"


def test_c1_labels_are_singletons_from_the_table():
    t, _ = synthetic_table(2, 3)
    lab = c1_labels(t)
    assert len(lab) == 6 and all(len(v) == 1 for v in lab.values()) and lab[0] == frozenset({t.rows[0]["c1_class"]})


def test_level_r_rows_map_other_and_exclude_not_a_lesion(sealed):
    d, m = sealed
    labels, excluded = r_test_labels(d, m, unblind=True)
    assert sorted(excluded) == [8, 9] and labels[1] == frozenset({"white_matter", "cortex"}) and labels[2] == frozenset({"other_deep_grey"})
    train, ex = r_train_labels(0, d, m)
    assert 0 not in train and 5 not in train and 1 in train and ex == [8, 9]      # fold 0 holds lesions 0 and 5


def test_test_labels_need_the_literal_unblind_flag(sealed):
    d, m = sealed
    with pytest.raises(SealedAccessError):
        r_test_labels(d, m)
    with pytest.raises(SealedAccessError):
        r_test_labels(d, m, unblind=1)


def test_acceptable_matrix_marks_slots_and_missing_labels():
    labels = {0: frozenset({"white_matter"}), 2: frozenset({"cortex", "thalamus"})}
    has, acc = acceptable_matrix(labels, np.array([0, 1, 2]))
    assert has.tolist() == [True, False, True] and acc.shape == (3, 7)
    assert acc[0].tolist() == [True] + [False] * 6 and acc[2].tolist() == [False, True, True, False, False, False, False]
    assert not acc[1].any()


def _r_row(lid, hosts, not_a_lesion=False):
    return {"lesion_id": lid, "status": "agreed", "primary_host": hosts[0] if hosts else None, "acceptable_hosts": hosts,
            "not_a_lesion": not_a_lesion, "lesion_type": None, "side": None, "lobe": None}


def test_level_r_rows_reject_an_unknown_host_and_an_empty_set_on_a_real_lesion():
    with pytest.raises(ValueError, match="lesion 3"):
        from_level_r_rows([_r_row(0, ["white_matter"]), _r_row(3, ["white_matter", "ventricle"])])
    with pytest.raises(ValueError, match="lesion 4"):
        from_level_r_rows([_r_row(4, [])])
    labels, excluded = from_level_r_rows([_r_row(5, [], not_a_lesion=True), _r_row(6, ["other"])])   # empty is fine when excluded
    assert excluded == [5] and labels == {6: frozenset({"other_deep_grey"})}
