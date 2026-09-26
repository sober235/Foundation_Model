"""SQLite store for Level R (spec §7). labels, adjudications and pilot releases are append-only: a reader who changes an
answer adds a row, and statistics take the last row per (reader, lesion). No statement in this module ever modifies or removes a
row, and the test suite greps this file to keep it that way."""
import hashlib
import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from urllib.parse import quote

from anatobind.level_r.schema import MAX_ACCEPTABLE, InvalidLabel, validate_adjudication, validate_label

SCHEMA = """
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
TABLES = tuple(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)\(", SCHEMA))
LABEL_COLUMNS = ("reader_id", "lesion_id", "primary_host", "acceptable_json", "topography", "adjacency_json", "ambiguity",
                 "not_a_lesion", "local_quality", "confidence", "comment", "time_seconds", "window_json", "ts")
ADJ_COLUMNS = ("adjudicator_id", "lesion_id", "primary_host", "acceptable_json", "topography", "adjacency_json", "ambiguity",
               "not_a_lesion", "reason", "ts")


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite_number(v):
    """A JSON number sqlite and json.dumps keep as it is: not a bool, not NaN or +-Infinity (both comparisons fail), not
    an integer too large for a sqlite INTEGER."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and -1e15 < v < 1e15


def needs_adjudication(a, b):
    """Spec R9: different primary host, or one says not_a_lesion, or the union of acceptable sets exceeds 2."""
    if a["not_a_lesion"] != b["not_a_lesion"]:
        return True
    if a["not_a_lesion"]:
        return False
    if a["primary_host"] != b["primary_host"]:
        return True
    return len(set(a["acceptable_hosts"]) | set(b["acceptable_hosts"])) > MAX_ACCEPTABLE


def _parse(row):
    d = dict(row)
    d["acceptable_hosts"] = json.loads(d.pop("acceptable_json"))
    d["adjacency"] = json.loads(d.pop("adjacency_json"))
    d["not_a_lesion"] = bool(d["not_a_lesion"])
    if "window_json" in d:
        w = d.pop("window_json")
        d["window"] = json.loads(w) if w is not None else None
    return d


class Store:
    def __init__(self, path, create=True):
        """create=True (init, tests) creates the file and any missing table. create=False (every other entry point)
        opens an existing Level R database read-write and raises sqlite3.OperationalError when the file is missing or
        lacks a table, so a mistyped path never leaves an empty database behind."""
        self.path, self.create = str(path), create
        self._lock = threading.Lock()
        with self._conn() as c:
            if create:
                c.executescript(SCHEMA)
            have = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        missing = [t for t in TABLES if t not in have]
        if missing:
            raise sqlite3.OperationalError(f"{self.path} lacks the Level R tables {missing}; "
                                           "scripts/level_r_admin.py init adds missing tables to an existing database")

    def _conn(self):
        if self.create:
            c = sqlite3.connect(self.path, timeout=30)
        else:
            c = sqlite3.connect(f"file:{quote(self.path)}?mode=rw", uri=True, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # readers -----------------------------------------------------------------------------------------------------
    def add_reader(self, reader_id, role, token, display):
        with self._lock, self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM readers WHERE role = 'reader'").fetchone()[0]
            if role == "reader" and n >= 2:
                raise ValueError(f"Level R has exactly two readers and already has {n}; a third reader is refused")
            c.execute("INSERT INTO readers VALUES (?, ?, ?, ?)", (reader_id, role, token_hash(token), display))

    def reader_for_token(self, token):
        with self._conn() as c:
            r = c.execute("SELECT reader_id, role, display FROM readers WHERE token_hash = ?", (token_hash(token),)).fetchone()
        return dict(r) if r else None

    def readers(self):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT reader_id, role, display FROM readers ORDER BY reader_id")]

    def _reader_ids(self):
        ids = [r["reader_id"] for r in self.readers() if r["role"] == "reader"]
        if len(ids) != 2:
            raise ValueError(f"Level R needs exactly two readers, have {ids}")
        return ids

    # lesions -----------------------------------------------------------------------------------------------------
    def load_lesions(self, records):
        with self._lock, self._conn() as c:
            c.executemany("INSERT OR IGNORE INTO lesions VALUES (?, ?, ?, ?, ?, ?)",
                          [(r["lesion_id"], r["code"], r["volume_code"], r["z0"], r["z1"], json.dumps(r["boxes"])) for r in records])

    def lesion(self, lesion_id):
        with self._conn() as c:
            r = c.execute("SELECT * FROM lesions WHERE lesion_id = ?", (int(lesion_id),)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["boxes"] = json.loads(d.pop("boxes_json"))
        return d

    def lesion_ids(self):
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT lesion_id FROM lesions ORDER BY lesion_id")]

    # orders ------------------------------------------------------------------------------------------------------
    def set_order(self, reader_id, lesion_ids, pilot_ids):
        with self._lock, self._conn() as c:
            if c.execute("SELECT 1 FROM orders WHERE reader_id = ? LIMIT 1", (reader_id,)).fetchone():
                raise ValueError(f"{reader_id} already has an order; orders are set once")
            c.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)",
                          [(reader_id, i, int(lid), int(lid in pilot_ids)) for i, lid in enumerate(lesion_ids)])

    def order(self, reader_id):
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT position, lesion_id, is_pilot FROM orders WHERE reader_id = ? ORDER BY position", (reader_id,))]

    # labels ------------------------------------------------------------------------------------------------------
    def submit_label(self, reader_id, lesion_id, payload):
        lab = validate_label(payload)
        t, window = payload.get("time_seconds"), payload.get("window")
        if t is not None and not (_finite_number(t) and t >= 0):
            raise InvalidLabel(f"time_seconds {t!r} must be a finite number >= 0")
        if window is not None and not (isinstance(window, list) and len(window) == 2 and all(_finite_number(v) for v in window)):
            raise InvalidLabel(f"window {window!r} must be a list of two numbers")
        row = (reader_id, int(lesion_id), lab["primary_host"], json.dumps(lab["acceptable_hosts"]), lab["topography"],
               json.dumps(lab["adjacency"]), lab["ambiguity"], int(lab["not_a_lesion"]), lab["local_quality"],
               lab["confidence"], lab["comment"], t, json.dumps(window) if window is not None else None, now_iso())
        with self._lock, self._conn() as c:
            if c.execute("SELECT 1 FROM lesions WHERE lesion_id = ?", (int(lesion_id),)).fetchone() is None:
                raise KeyError(lesion_id)
            cur = c.execute(f"INSERT INTO labels({', '.join(LABEL_COLUMNS)}) VALUES ({', '.join('?' * len(LABEL_COLUMNS))})", row)
            return cur.lastrowid

    def label_rows(self, reader_id=None):
        q = "SELECT * FROM labels" + (" WHERE reader_id = ?" if reader_id else "") + " ORDER BY row_id"
        with self._conn() as c:
            return [_parse(r) for r in c.execute(q, (reader_id,) if reader_id else ())]

    def latest_labels(self, reader_id=None):
        last = {}
        for r in self.label_rows(reader_id):
            last[(r["reader_id"], r["lesion_id"])] = r
        return list(last.values())

    def progress(self, reader_id):
        """held: the reader has answered every pilot lesion of their order and has not been released (spec §9: both
        readers finish the pilot, the report runs, only then does reading continue); next is None while held."""
        done = {r["lesion_id"] for r in self.latest_labels(reader_id)}
        order = self.order(reader_id)
        pilot = [o["lesion_id"] for o in order if o["is_pilot"]]
        held = bool(pilot) and all(l in done for l in pilot) and not self.is_released(reader_id)
        return {"done": sum(o["lesion_id"] in done for o in order), "total": len(order),
                "next": None if held else next((o["lesion_id"] for o in order if o["lesion_id"] not in done), None),
                "held": held}

    # pilot release (append-only) ---------------------------------------------------------------------------------
    def release(self, reader_id):
        with self._lock, self._conn() as c:
            if c.execute("SELECT 1 FROM readers WHERE reader_id = ? AND role = 'reader'", (reader_id,)).fetchone() is None:
                raise ValueError(f"{reader_id!r} is not a reader")
            c.execute("INSERT INTO releases(reader_id, ts) VALUES (?, ?)", (reader_id, now_iso()))

    def is_released(self, reader_id):
        with self._conn() as c:
            return c.execute("SELECT 1 FROM releases WHERE reader_id = ? LIMIT 1", (reader_id,)).fetchone() is not None

    # adjudication ------------------------------------------------------------------------------------------------
    def submit_adjudication(self, adjudicator_id, lesion_id, payload):
        lab = validate_adjudication(payload)
        row = (adjudicator_id, int(lesion_id), lab["primary_host"], json.dumps(lab["acceptable_hosts"]), lab["topography"],
               json.dumps(lab["adjacency"]), lab["ambiguity"], int(lab["not_a_lesion"]), lab["reason"], now_iso())
        with self._lock, self._conn() as c:
            if c.execute("SELECT 1 FROM lesions WHERE lesion_id = ?", (int(lesion_id),)).fetchone() is None:
                raise KeyError(lesion_id)
            cur = c.execute(f"INSERT INTO adjudications({', '.join(ADJ_COLUMNS)}) VALUES ({', '.join('?' * len(ADJ_COLUMNS))})", row)
            return cur.lastrowid

    def adjudication_rows(self):
        with self._conn() as c:
            return [_parse(r) for r in c.execute("SELECT * FROM adjudications ORDER BY row_id")]

    def latest_adjudications(self):
        last = {}
        for r in self.adjudication_rows():
            last[r["lesion_id"]] = r
        return list(last.values())

    def disagreements(self):
        """Lesions both readers have answered where R9 sends them to adjudication, plus the two reader ids in order."""
        readers = self._reader_ids()
        a, b = ({l["lesion_id"]: l for l in self.latest_labels(r)} for r in readers)
        return sorted(lid for lid in set(a) & set(b) if needs_adjudication(a[lid], b[lid])), readers

    def _current_adjudications(self, a, b):
        """The latest adjudication per lesion, kept only while its ts is >= both readers' latest answer ts: a reader who
        revises after the ruling sends the lesion back to the adjudicator. now_iso() strings (UTC, whole seconds) compare
        lexicographically, so an answer in the same second as the ruling leaves the ruling current."""
        return {z["lesion_id"]: z for z in self.latest_adjudications()
                if all(z["ts"] >= side[z["lesion_id"]]["ts"] for side in (a, b) if z["lesion_id"] in side)}

    def adjudicated_lesion_ids(self):
        readers = self._reader_ids()
        a, b = ({l["lesion_id"]: l for l in self.latest_labels(r)} for r in readers)
        return set(self._current_adjudications(a, b))

    def final_labels(self):
        """R9: agreed -> union of the acceptable sets; disagreement -> the last adjudication while it is current, else
        pending."""
        readers = self._reader_ids()
        a, b = ({l["lesion_id"]: l for l in self.latest_labels(r)} for r in readers)
        adj = self._current_adjudications(a, b)
        out = []
        for lid in sorted(set(a) & set(b)):
            x, y = a[lid], b[lid]
            if not needs_adjudication(x, y):
                out.append({"lesion_id": lid, "status": "agreed", "primary_host": x["primary_host"],
                            "acceptable_hosts": sorted(set(x["acceptable_hosts"]) | set(y["acceptable_hosts"])),
                            "not_a_lesion": x["not_a_lesion"]})
            elif lid in adj:
                z = adj[lid]
                out.append({"lesion_id": lid, "status": "adjudicated", "primary_host": z["primary_host"],
                            "acceptable_hosts": sorted(z["acceptable_hosts"]), "not_a_lesion": z["not_a_lesion"]})
            else:
                out.append({"lesion_id": lid, "status": "pending", "primary_host": None, "acceptable_hosts": [], "not_a_lesion": None})
        return out
