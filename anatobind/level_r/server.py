"""Level R HTTP service (spec §5, §6, §10): stdlib ThreadingHTTPServer, a JSON API plus the static app. Every call
except the static pages and /api/enums carries ?token=<16 hex>; the caller is looked up by token hash and a reader
only ever sees their own order and answers. Every JSON body passes assert_blind before it leaves."""
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from anatobind.level_r.blind import assert_blind, blind_lesion, blind_volume_meta
from anatobind.level_r.schema import InvalidLabel, enums

APP_DIR = Path(__file__).resolve().parent / "app"
STATIC = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css", "/guide.html": "guide.html"}
ANSWER_KEYS = ("primary_host", "acceptable_hosts", "topography", "adjacency", "ambiguity", "not_a_lesion", "local_quality",
               "confidence", "comment", "reason", "ts")
TOKEN_RE = re.compile(r"[0-9a-f]{16}")
mimetypes.add_type("text/javascript", ".js")


def answer_view(row):
    """A label or adjudication row as the browser may see it: the answer fields only, no ids."""
    return {k: row[k] for k in ANSWER_KEYS if k in row}


def make_handler(store, data_root, app_dir=APP_DIR):
    data_root, app_dir = Path(data_root), Path(app_dir)

    class Handler(BaseHTTPRequestHandler):
        server_version = "LevelR/1"

        def log_message(self, fmt, *args):      # the launcher's log captures stdout; per-request lines are noise
            pass

        def _send(self, code, body, ctype="application/json"):
            if ctype == "application/json":
                body = json.dumps(assert_blind(body)).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _caller(self, query):
            tok = (query.get("token") or [""])[0]
            r = store.reader_for_token(tok) if TOKEN_RE.fullmatch(tok) else None
            if r is None:
                self._send(403, {"error": "invalid token"})
            return r

        def _may_see(self, caller, lesion_id):
            return caller["role"] == "adjudicator" or any(o["lesion_id"] == lesion_id for o in store.order(caller["reader_id"]))

        def do_GET(self):
            u = urlparse(self.path)
            q, p = parse_qs(u.query), u.path
            if p in STATIC:
                f = app_dir / STATIC[p]
                if not f.exists():
                    return self._send(404, {"error": "no such page"})
                return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            if p == "/api/enums":
                return self._send(200, enums())
            caller = self._caller(q)
            if caller is None:
                return
            try:
                return self._get(caller, p)
            except ValueError as e:                     # fewer than two readers
                return self._send(409, {"error": str(e)})

        def _get(self, caller, p):
            rid, role = caller["reader_id"], caller["role"]
            if p == "/api/me":
                extra = store.progress(rid) if role == "reader" else {"disagreements": len(store.disagreements()[0])}
                return self._send(200, {**caller, **extra})
            if p == "/api/list":
                done = {l["lesion_id"] for l in store.latest_labels(rid)}
                return self._send(200, [{**o, "done": o["lesion_id"] in done} for o in store.order(rid)])
            m = re.fullmatch(r"/api/lesion/(\d+)", p)
            if m:
                lid = int(m.group(1))
                if not self._may_see(caller, lid):
                    return self._send(403, {"error": "not in your list"})
                L = store.lesion(lid)
                if L is None:
                    return self._send(404, {"error": "no such lesion"})
                mine = [l for l in store.latest_labels(rid) if l["lesion_id"] == lid]
                return self._send(200, {"lesion": blind_lesion(L), "answer": answer_view(mine[0]) if mine else None})
            m = re.fullmatch(r"/api/volume/([0-9a-f]{8})\.(json|u16)", p)
            if m:
                code, ext = m.groups()
                f = data_root / "volumes" / f"{code}.{ext}"
                if not f.exists():
                    return self._send(404, {"error": "no such volume"})
                if ext == "json":
                    return self._send(200, blind_volume_meta(json.loads(f.read_text())))
                return self._send(200, f.read_bytes(), "application/octet-stream")
            if p == "/api/disagreements":
                if role != "adjudicator":
                    return self._send(403, {"error": "adjudicator only"})
                ids, _ = store.disagreements()
                done = {a["lesion_id"] for a in store.latest_adjudications()}
                return self._send(200, [{"lesion_id": i, "done": i in done} for i in ids])
            m = re.fullmatch(r"/api/adjudicate/(\d+)", p)
            if m:
                if role != "adjudicator":
                    return self._send(403, {"error": "adjudicator only"})
                lid = int(m.group(1))
                ids, readers = store.disagreements()
                if lid not in ids:
                    return self._send(404, {"error": "not a disagreement"})
                views = [answer_view(next(l for l in store.latest_labels(r) if l["lesion_id"] == lid)) for r in readers]
                mine = [a for a in store.latest_adjudications() if a["lesion_id"] == lid]
                return self._send(200, {"lesion": blind_lesion(store.lesion(lid)), "readers": views,
                                        "answer": answer_view(mine[0]) if mine else None})
            return self._send(404, {"error": "no such route"})

        def do_POST(self):
            u = urlparse(self.path)
            caller = self._caller(parse_qs(u.query))
            if caller is None:
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                lid = int(body.get("lesion_id", -1))
            except (ValueError, TypeError):
                return self._send(400, {"error": "body must be JSON with an integer lesion_id"})
            try:
                if u.path == "/api/label":
                    if caller["role"] != "reader":
                        return self._send(403, {"error": "readers only"})
                    if not self._may_see(caller, lid):
                        return self._send(403, {"error": "not in your list"})
                    return self._send(200, {"row_id": store.submit_label(caller["reader_id"], lid, body)})
                if u.path == "/api/adjudication":
                    if caller["role"] != "adjudicator":
                        return self._send(403, {"error": "adjudicator only"})
                    if lid not in store.disagreements()[0]:
                        return self._send(400, {"error": "not a disagreement"})
                    return self._send(200, {"row_id": store.submit_adjudication(caller["reader_id"], lid, body)})
            except InvalidLabel as e:
                return self._send(400, {"error": str(e)})
            except KeyError:
                return self._send(404, {"error": "no such lesion"})
            except ValueError as e:
                return self._send(409, {"error": str(e)})
            return self._send(404, {"error": "no such route"})

    return Handler


def make_server(store, data_root, bind="127.0.0.1", port=8790, app_dir=APP_DIR):
    return ThreadingHTTPServer((bind, port), make_handler(store, data_root, app_dir))
