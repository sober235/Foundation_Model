"""SQLite store for Level R (spec §7). labels and adjudications are append-only: a reader who changes an answer adds
a row, and statistics take the last row per (reader, lesion). No statement in this module ever modifies or removes a
row, and the test suite greps this file to keep it that way."""
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone

from anatobind.level_r.schema import MAX_ACCEPTABLE, validate_adjudication, validate_label

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
"""
LABEL_COLUMNS = ("reader_id", "lesion_id", "primary_host", "acceptable_json", "topography", "adjacency_json", "ambiguity",
                 "not_a_lesion", "local_quality", "confidence", "comment", "time_seconds", "window_json", "ts")
ADJ_COLUMNS = ("adjudicator_id", "lesion_id", "primary_host", "acceptable_json", "topography", "adjacency_json", "ambiguity",
               "not_a_lesion", "reason", "ts")


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    def __init__(self, path):
        self.path = str(path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # readers -----------------------------------------------------------------------------------------------------
    def add_reader(self, reader_id, role, token, display):
        with self._lock, self._conn() as c:
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
        window = payload.get("window")
        row = (reader_id, int(lesion_id), lab["primary_host"], json.dumps(lab["acceptable_hosts"]), lab["topography"],
               json.dumps(lab["adjacency"]), lab["ambiguity"], int(lab["not_a_lesion"]), lab["local_quality"],
               lab["confidence"], lab["comment"], payload.get("time_seconds"),
               json.dumps(window) if window is not None else None, now_iso())
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
        done = {r["lesion_id"] for r in self.latest_labels(reader_id)}
        order = self.order(reader_id)
        return {"done": sum(o["lesion_id"] in done for o in order), "total": len(order),
                "next": next((o["lesion_id"] for o in order if o["lesion_id"] not in done), None)}

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

    def final_labels(self):
        """R9: agreed -> union of the acceptable sets; disagreement -> the last adjudication, else pending."""
        readers = self._reader_ids()
        a, b = ({l["lesion_id"]: l for l in self.latest_labels(r)} for r in readers)
        adj = {x["lesion_id"]: x for x in self.latest_adjudications()}
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
