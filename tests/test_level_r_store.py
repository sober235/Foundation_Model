import itertools
import sqlite3
from pathlib import Path

import pytest

import anatobind.level_r.store as store_module
from anatobind.level_r.schema import InvalidLabel
from anatobind.level_r.store import Store, needs_adjudication, token_hash

T1, T2, TA = "0123456789abcdef", "fedcba9876543210", "aaaaaaaaaaaaaaaa"
LESIONS = [{"lesion_id": i, "code": f"c{i:07d}", "volume_code": "v0000000", "z0": 1, "z1": 1, "boxes": {"1": [[2, 6, 3, 9]]}} for i in range(5)]
WM = {"primary_host": "white_matter", "acceptable_hosts": ["white_matter"], "topography": "deep_white_matter", "adjacency": ["none"],
      "ambiguity": "certain", "not_a_lesion": False, "lesion_type": "nonspecific_wm_lesion", "side": "image_left", "lobe": "frontal",
      "local_quality": "good", "confidence": 5, "comment": ""}
WM_CX = {**WM, "acceptable_hosts": ["white_matter", "cortex"], "ambiguity": "two_host"}
CX = {**WM, "primary_host": "cortex", "acceptable_hosts": ["cortex"]}
CX_TH = {**CX, "acceptable_hosts": ["cortex", "thalamus"]}
NAL = {"not_a_lesion": True, "local_quality": "fair", "confidence": 3, "adjacency": []}
NO_TYPE = {"lesion_type": None, "side": None, "lobe": None}
# The schema as it stood before the lesion_type / side / lobe columns (a14b022), for the migration test.
OLD_SCHEMA = """
CREATE TABLE IF NOT EXISTS readers(
    reader_id TEXT PRIMARY KEY, role TEXT NOT NULL CHECK(role IN ('reader', 'adjudicator')),
    token_hash TEXT NOT NULL UNIQUE, display TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lesions(
    lesion_id INTEGER PRIMARY KEY, code TEXT NOT NULL, volume_code TEXT NOT NULL,
    z0 INTEGER NOT NULL, z1 INTEGER NOT NULL, boxes_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS orders(
    reader_id TEXT NOT NULL, position INTEGER NOT NULL, lesion_id INTEGER NOT NULL, is_pilot INTEGER NOT NULL,
    PRIMARY KEY(reader_id, position));
CREATE TABLE IF NOT EXISTS labels(
    row_id INTEGER PRIMARY KEY, reader_id TEXT NOT NULL, lesion_id INTEGER NOT NULL,
    primary_host TEXT, acceptable_json TEXT NOT NULL, topography TEXT, adjacency_json TEXT NOT NULL, ambiguity TEXT,
    not_a_lesion INTEGER NOT NULL, local_quality TEXT, confidence INTEGER, comment TEXT NOT NULL,
    time_seconds REAL, window_json TEXT, ts TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS adjudications(
    row_id INTEGER PRIMARY KEY, adjudicator_id TEXT NOT NULL, lesion_id INTEGER NOT NULL,
    primary_host TEXT, acceptable_json TEXT NOT NULL, topography TEXT, adjacency_json TEXT NOT NULL, ambiguity TEXT,
    not_a_lesion INTEGER NOT NULL, reason TEXT NOT NULL, ts TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS releases(
    row_id INTEGER PRIMARY KEY, reader_id TEXT NOT NULL, ts TEXT NOT NULL);
"""


def _store_at(path):
    s = Store(path)
    s.add_reader("r1", "reader", T1, "读者 1")
    s.add_reader("r2", "reader", T2, "读者 2")
    s.add_reader("adj", "adjudicator", TA, "裁定")
    s.load_lesions(LESIONS)
    return s


def _store(tmp_path):
    return _store_at(tmp_path / "level_r.sqlite")


def test_store_source_never_modifies_or_removes_rows():
    src = Path(store_module.__file__).read_text().upper()
    assert "UPDATE" not in src and "DELETE" not in src


def test_tokens_are_stored_hashed_and_looked_up(tmp_path):
    s = _store(tmp_path)
    assert s.reader_for_token(T1) == {"reader_id": "r1", "role": "reader", "display": "读者 1"}
    assert s.reader_for_token("0000000000000000") is None
    raw = sqlite3.connect(str(tmp_path / "level_r.sqlite")).execute("SELECT token_hash FROM readers WHERE reader_id='r1'").fetchone()[0]
    assert raw == token_hash(T1) and raw != T1 and len(raw) == 64
    assert [r["reader_id"] for r in s.readers()] == ["adj", "r1", "r2"]


def test_lesions_load_once_and_keep_boxes(tmp_path):
    s = _store(tmp_path)
    s.load_lesions(LESIONS)                                  # second load is a no-op
    assert s.lesion_ids() == [0, 1, 2, 3, 4]
    assert s.lesion(3)["boxes"] == {"1": [[2, 6, 3, 9]]} and s.lesion(3)["code"] == "c0000003" and s.lesion(9) is None


def test_order_is_set_once_and_marks_pilot(tmp_path):
    s = _store(tmp_path)
    s.set_order("r1", [3, 1, 4, 0, 2], pilot_ids={3, 1})
    assert [(o["position"], o["lesion_id"], o["is_pilot"]) for o in s.order("r1")] == [(0, 3, 1), (1, 1, 1), (2, 4, 0), (3, 0, 0), (4, 2, 0)]
    with pytest.raises(ValueError):
        s.set_order("r1", [0, 1, 2, 3, 4], set())
    assert s.order("r2") == []


def test_labels_append_latest_wins_and_progress_moves(tmp_path):
    s = _store(tmp_path)
    s.set_order("r1", [3, 1, 4, 0, 2], {3, 1})
    assert s.progress("r1") == {"done": 0, "total": 5, "next": 3, "held": False}
    r1 = s.submit_label("r1", 3, {**WM, "time_seconds": 41.5, "window": [100, 900]})
    r2 = s.submit_label("r1", 3, {**CX, "time_seconds": 12.0})
    assert r2 > r1
    rows = s.label_rows("r1")
    assert [r["primary_host"] for r in rows] == ["white_matter", "cortex"] and rows[0]["window"] == [100, 900] and rows[1]["window"] is None
    (latest,) = s.latest_labels("r1")
    assert latest["primary_host"] == "cortex" and latest["acceptable_hosts"] == ["cortex"] and latest["adjacency"] == ["none"]
    assert latest["not_a_lesion"] is False and latest["time_seconds"] == 12.0 and latest["ts"]
    assert (latest["lesion_type"], latest["side"], latest["lobe"]) == ("nonspecific_wm_lesion", "image_left", "frontal")
    assert s.progress("r1") == {"done": 1, "total": 5, "next": 1, "held": False}
    s.submit_label("r1", 4, NAL)                                       # a not_a_lesion answer stores no type, side or lobe
    assert {k: s.label_rows("r1")[-1][k] for k in NO_TYPE} == NO_TYPE


def test_progress_holds_at_the_end_of_the_pilot_until_the_reader_is_released(tmp_path):
    s = _store(tmp_path)
    s.set_order("r1", [3, 1, 4, 0, 2], {3, 1})
    s.set_order("r2", [0, 1, 2, 3, 4], set())
    s.submit_label("r1", 1, WM)                                        # pilot answered out of order: still inside the pilot
    assert s.progress("r1") == {"done": 1, "total": 5, "next": 3, "held": False}
    s.submit_label("r1", 3, WM)
    assert s.progress("r1") == {"done": 2, "total": 5, "next": None, "held": True} and s.is_released("r1") is False
    s.release("r1")
    assert s.is_released("r1") is True and s.is_released("r2") is False
    assert s.progress("r1") == {"done": 2, "total": 5, "next": 4, "held": False}
    for lid in range(5):                                               # an order without pilot lesions never holds
        assert s.progress("r2")["held"] is False
        s.submit_label("r2", lid, WM)
    assert s.progress("r2") == {"done": 5, "total": 5, "next": None, "held": False}
    for who in ("nobody", "adj"):                                      # only an existing reader can be released
        with pytest.raises(ValueError):
            s.release(who)
    rows = sqlite3.connect(str(tmp_path / "level_r.sqlite")).execute("SELECT reader_id, ts FROM releases").fetchall()
    assert len(rows) == 1 and rows[0][0] == "r1" and rows[0][1]


def test_submit_rejects_invalid_or_unknown_and_stores_nothing(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(InvalidLabel):
        s.submit_label("r1", 0, {**WM, "confidence": 9})
    with pytest.raises(KeyError):
        s.submit_label("r1", 99, WM)
    assert s.label_rows() == []


@pytest.mark.parametrize("extra", [
    {"time_seconds": -1}, {"time_seconds": float("nan")}, {"time_seconds": float("inf")}, {"time_seconds": True},
    {"time_seconds": "abc"}, {"time_seconds": {}}, {"time_seconds": [3]}, {"time_seconds": 10 ** 400},
    {"window": "x"}, {"window": [1]}, {"window": [1, 2, 3]}, {"window": [1, "a"]}, {"window": [True, 2]},
    {"window": [float("nan"), 1]}, {"window": {"lo": 1, "hi": 2}},
])
def test_submit_rejects_malformed_time_or_window_and_stores_nothing(tmp_path, extra):
    s = _store(tmp_path)
    with pytest.raises(InvalidLabel):
        s.submit_label("r1", 0, {**WM, **extra})
    assert s.label_rows() == []


def test_submit_accepts_well_formed_time_and_window(tmp_path):
    s = _store(tmp_path)
    for extra in ({}, {"time_seconds": None, "window": None}, {"time_seconds": 0}, {"time_seconds": 12.5, "window": [-40, 900]},
                  {"time_seconds": 3, "window": [1.5, 2.5]}):
        s.submit_label("r1", 0, {**WM, **extra})
    assert [(r["time_seconds"], r["window"]) for r in s.label_rows("r1")] == [
        (None, None), (None, None), (0, None), (12.5, [-40, 900]), (3, [1.5, 2.5])]


@pytest.mark.parametrize("a, b, expected", [
    (WM, WM, False), (WM, WM_CX, False),                     # same host, union {wm, cortex} <= 2
    (WM, CX, True),                                          # different primary host
    (WM, NAL, True), (NAL, NAL, False),                      # one says not a lesion; both do
    (WM_CX, CX_TH, True),                                    # union {wm, cortex, thalamus} > 2 even though... hosts differ too
    ({**WM_CX, "primary_host": "cortex"}, CX_TH, True),      # same primary host, union of three
    (WM, {**WM, "lesion_type": "lacunar_infarct"}, True),    # same host, different lesion type
    (WM, {**WM, "side": "image_right"}, True),               # same host, different screen side
    (WM, {**WM, "side": "midline"}, True),
    (WM, {**WM, "lobe": "parietal"}, False),                 # a different lobe alone is not sent to the adjudicator
    ({**WM, "lobe": "not_applicable"}, {**WM_CX, "lobe": "insular"}, False),
])
def test_needs_adjudication_covers_the_r9_cases(tmp_path, a, b, expected):
    from anatobind.level_r.schema import validate_label
    assert needs_adjudication(validate_label(a), validate_label(b)) is expected


def test_disagreements_and_final_labels_follow_r9(tmp_path):
    s = _store(tmp_path)
    for lid, (a, b) in {0: (WM, WM_CX), 1: (WM, CX), 2: (NAL, WM), 3: (NAL, NAL)}.items():
        s.submit_label("r1", lid, a)
        s.submit_label("r2", lid, b)
    s.submit_label("r1", 4, WM)                              # r2 has not read lesion 4 yet
    ids, readers = s.disagreements()
    assert ids == [1, 2] and readers == ["r1", "r2"]
    final = {f["lesion_id"]: f for f in s.final_labels()}
    assert set(final) == {0, 1, 2, 3}
    assert final[0] == {"lesion_id": 0, "status": "agreed", "primary_host": "white_matter", "acceptable_hosts": ["cortex", "white_matter"], "not_a_lesion": False,
                        "lesion_type": "nonspecific_wm_lesion", "side": "image_left", "lobe": "frontal"}
    assert final[3] == {"lesion_id": 3, "status": "agreed", "primary_host": None, "acceptable_hosts": [], "not_a_lesion": True, **NO_TYPE}
    assert final[1] == {"lesion_id": 1, "status": "pending", "primary_host": None, "acceptable_hosts": [], "not_a_lesion": None, **NO_TYPE}
    assert final[2]["status"] == "pending"
    with pytest.raises(InvalidLabel):
        s.submit_adjudication("adj", 1, {**CX, "reason": ""})
    s.submit_adjudication("adj", 1, {"primary_host": "cortex", "acceptable_hosts": ["cortex"], "topography": "cortical", "adjacency": [],
                                     "ambiguity": "certain", "lesion_type": "lacunar_infarct", "side": "image_right", "lobe": "temporal",
                                     "reason": "皮层内信号"})
    final = {f["lesion_id"]: f for f in s.final_labels()}
    assert final[1] == {"lesion_id": 1, "status": "adjudicated", "primary_host": "cortex", "acceptable_hosts": ["cortex"], "not_a_lesion": False,
                        "lesion_type": "lacunar_infarct", "side": "image_right", "lobe": "temporal"}
    (adj,) = s.latest_adjudications()
    assert adj["adjudicator_id"] == "adj" and adj["reason"] == "皮层内信号" and adj["lesion_id"] == 1
    assert (adj["lesion_type"], adj["side"], adj["lobe"]) == ("lacunar_infarct", "image_right", "temporal")


def test_type_or_side_disagreements_go_to_the_adjudicator_and_the_lobe_is_kept_only_when_both_agree(tmp_path):
    s = _store(tmp_path)
    for lid, (a, b) in {0: (WM, WM_CX),                                # everything the same
                        1: (WM, {**WM, "lobe": "parietal"}),           # lobes differ: still agreed, final lobe unknown
                        2: (WM, {**WM, "lesion_type": "perivascular_space"}),
                        3: (WM, {**WM, "side": "midline"})}.items():
        s.submit_label("r1", lid, a)
        s.submit_label("r2", lid, b)
    assert s.disagreements()[0] == [2, 3]
    final = {f["lesion_id"]: f for f in s.final_labels()}
    assert [(final[i]["status"], final[i]["lesion_type"], final[i]["side"], final[i]["lobe"]) for i in range(4)] == [
        ("agreed", "nonspecific_wm_lesion", "image_left", "frontal"), ("agreed", "nonspecific_wm_lesion", "image_left", None),
        ("pending", None, None, None), ("pending", None, None, None)]
    s.submit_adjudication("adj", 3, {**WM, "side": "midline", "lobe": "not_applicable", "reason": "跨中线"})
    f3 = {f["lesion_id"]: f for f in s.final_labels()}[3]
    assert (f3["status"], f3["lesion_type"], f3["side"], f3["lobe"]) == ("adjudicated", "nonspecific_wm_lesion", "midline", "not_applicable")


def _ticking_clock(monkeypatch):
    """now_iso has second resolution; give every stored row its own second so the order of events is unambiguous."""
    ticks = itertools.count()
    monkeypatch.setattr(store_module, "now_iso", lambda: f"2026-10-01T00:{(t := next(ticks)) // 60:02d}:{t % 60:02d}+00:00")


def _status(s, lid):
    return {f["lesion_id"]: f for f in s.final_labels()}[lid]["status"]


def test_an_adjudication_goes_stale_when_a_reader_revises_after_it(tmp_path, monkeypatch):
    _ticking_clock(monkeypatch)
    s = _store(tmp_path)
    s.submit_label("r1", 1, WM)
    s.submit_label("r2", 1, CX)
    s.submit_adjudication("adj", 1, {**CX, "reason": "皮层"})
    assert _status(s, 1) == "adjudicated" and s.adjudicated_lesion_ids() == {1}
    s.submit_label("r2", 1, CX_TH)                             # revised after the ruling, still a disagreement
    assert _status(s, 1) == "pending" and s.adjudicated_lesion_ids() == set() and s.disagreements()[0] == [1]
    s.submit_adjudication("adj", 1, {**CX, "reason": "仍是皮层"})
    assert _status(s, 1) == "adjudicated" and s.adjudicated_lesion_ids() == {1}
    assert {f["lesion_id"]: f for f in s.final_labels()}[1]["primary_host"] == "cortex"


def test_an_adjudication_in_the_same_second_as_a_revision_still_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "now_iso", lambda: "2026-10-01T00:00:00+00:00")
    s = _store(tmp_path)
    s.submit_label("r1", 1, WM)
    s.submit_label("r2", 1, CX)
    s.submit_adjudication("adj", 1, {**CX, "reason": "皮层"})
    s.submit_label("r2", 1, CX_TH)
    assert _status(s, 1) == "adjudicated" and s.adjudicated_lesion_ids() == {1}


def test_disagreements_need_exactly_two_readers(tmp_path):
    s = Store(tmp_path / "x.sqlite")
    s.add_reader("r1", "reader", T1, "读者 1")
    with pytest.raises(ValueError):
        s.disagreements()


def test_a_database_from_before_the_releases_table_is_refused_until_init_adds_it(tmp_path):
    p = tmp_path / "old.sqlite"
    c = sqlite3.connect(str(p))
    c.executescript(store_module.SCHEMA.split("CREATE TABLE IF NOT EXISTS releases")[0])
    c.execute("INSERT INTO readers VALUES ('r1', 'reader', 'h', '读者 1')")
    c.commit()
    c.close()
    with pytest.raises(sqlite3.OperationalError, match="releases"):
        Store(p, create=False)
    Store(p)                                                     # what `level_r_admin.py init` does: add missing tables
    s = Store(p, create=False)
    assert [r["reader_id"] for r in s.readers()] == ["r1"] and s.is_released("r1") is False


def _columns(path, table):
    c = sqlite3.connect(str(path))
    try:
        return [r[1] for r in c.execute(f"PRAGMA table_info({table})")]
    finally:
        c.close()


def test_a_database_from_before_the_lesion_type_side_and_lobe_columns_is_refused_until_init_adds_them(tmp_path):
    p = tmp_path / "old.sqlite"
    c = sqlite3.connect(str(p))
    c.executescript(OLD_SCHEMA)
    c.executemany("INSERT INTO readers VALUES (?, 'reader', ?, ?)", [("r1", token_hash(T1), "读者 1"), ("r2", token_hash(T2), "读者 2")])
    c.execute("INSERT INTO lesions VALUES (0, 'c0000000', 'v0000000', 1, 1, '{}')")
    c.execute("INSERT INTO labels(reader_id, lesion_id, primary_host, acceptable_json, topography, adjacency_json, ambiguity, not_a_lesion,"
              " local_quality, confidence, comment, ts) VALUES ('r1', 0, 'white_matter', '[\"white_matter\"]', 'deep_white_matter', '[]',"
              " 'certain', 0, 'good', 5, '', '2026-09-26T00:00:00+00:00')")
    c.commit()
    c.close()
    with pytest.raises(sqlite3.OperationalError, match="level_r_admin.py init"):
        Store(p, create=False)
    assert "lesion_type" not in _columns(p, "labels")            # refusing leaves the file as it was
    Store(p)                                                     # what `level_r_admin.py init` does: add the missing columns
    Store(p)                                                     # and running it again is harmless
    Store(tmp_path / "new.sqlite")
    for table in ("labels", "adjudications"):
        assert _columns(p, table)[-3:] == ["lesion_type", "side", "lobe"]
        assert _columns(p, table) == _columns(tmp_path / "new.sqlite", table)      # same layout as a database made today
    s = Store(p, create=False)
    (old,) = s.label_rows("r1")
    assert old["primary_host"] == "white_matter" and {k: old[k] for k in NO_TYPE} == NO_TYPE
    s.submit_label("r2", 0, {**WM, "side": "image_right"})
    s.submit_adjudication("adj", 0, {**WM, "reason": "迁移后可写"})
    assert s.latest_labels("r2")[0]["side"] == "image_right" and s.adjudication_rows()[0]["lesion_type"] == "nonspecific_wm_lesion"


def test_a_third_reader_is_refused(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(ValueError):
        s.add_reader("r3", "reader", "1111111111111111", "读者 3")
    assert [r["reader_id"] for r in s.readers()] == ["adj", "r1", "r2"] and s.reader_for_token("1111111111111111") is None


def test_opening_without_create_needs_an_existing_level_r_database(tmp_path):
    missing = tmp_path / "typo.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        Store(missing, create=False)
    assert not missing.exists()                                  # a mistyped path never leaves an empty database behind
    other = tmp_path / "other.sqlite"
    sqlite3.connect(str(other)).execute("CREATE TABLE t(x)")
    with pytest.raises(sqlite3.OperationalError, match="lacks the Level R tables"):
        Store(other, create=False)
    odd = tmp_path / "a b#c?d" / "level_r.sqlite"                # URI-special characters in the path
    odd.parent.mkdir()
    _store_at(odd)
    s = Store(odd, create=False)
    assert [r["reader_id"] for r in s.readers()] == ["adj", "r1", "r2"] and s.lesion_ids() == [0, 1, 2, 3, 4]
    s.submit_label("r1", 0, WM)
    assert len(Store(odd, create=False).label_rows("r1")) == 1
