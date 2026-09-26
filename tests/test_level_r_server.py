import http.client
import importlib.util
import itertools
import json
import sqlite3
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import anatobind.level_r.store as store_module
from anatobind.level_r.blind import assert_blind
from anatobind.level_r.export import write_volume
from anatobind.level_r.schema import LESION_TYPES, LOBES, SIDES, enums
from anatobind.level_r.server import ANSWER_KEYS, make_server
from anatobind.level_r.store import Store

T1, T2, TA, BAD = "0123456789abcdef", "fedcba9876543210", "aaaaaaaaaaaaaaaa", "ffffffffffffffff"
CODE = "0f0f0f0f"
LESIONS = [{"lesion_id": i, "code": f"c{i:07d}", "volume_code": CODE, "z0": 1, "z1": 2, "boxes": {"1": [[2, 6, 3, 9]], "2": [[2, 6, 3, 9]]}} for i in range(4)]
WM = {"primary_host": "white_matter", "acceptable_hosts": ["white_matter"], "topography": "deep_white_matter", "adjacency": ["none"],
      "ambiguity": "certain", "not_a_lesion": False, "lesion_type": "nonspecific_wm_lesion", "side": "image_left", "lobe": "frontal",
      "local_quality": "good", "confidence": 5, "comment": "", "time_seconds": 30.0, "window": [10, 900]}
CX = {**WM, "primary_host": "cortex", "acceptable_hosts": ["cortex"]}


@pytest.fixture
def served(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    store.add_reader("r1", "reader", T1, "读者 1")
    store.add_reader("r2", "reader", T2, "读者 2")
    store.add_reader("adj", "adjudicator", TA, "裁定")
    store.load_lesions(LESIONS)
    store.set_order("r1", [2, 0, 1], {2})
    store.set_order("r2", [0, 1, 2], set())
    u = (np.arange(4 * 8 * 8) % 1000).astype("<u2").reshape(4, 8, 8)
    write_volume(tmp_path / "data" / "volumes", CODE, u, {"spacing_slice_mm": 5.0, "spacing_row_mm": 0.7, "spacing_col_mm": 0.7}, [0, 999])
    app = tmp_path / "app"
    app.mkdir()
    (app / "index.html").write_text("<html><body>Level R</body></html>")
    (app / "app.js").write_text("// app")
    srv = make_server(store, tmp_path / "data", "127.0.0.1", 0, app_dir=app)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield store, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(base, path, token=None, raw=False):
    url = base + path + (("&" if "?" in path else "?") + f"token={token}" if token else "")
    try:
        with urllib.request.urlopen(url) as r:
            body = r.read()
            return r.status, (body if raw else json.loads(body)), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), e.headers


def post(base, path, token, payload):
    req = urllib.request.Request(base + path + f"?token={token}", data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def post_raw(base, path, headers, body):
    """Like post(), but over a raw http.client connection so a caller-supplied (possibly bogus) Content-Length
    header reaches the server as-is; urllib always computes a correct one itself."""
    host, port = base[len("http://"):].split(":")
    conn = http.client.HTTPConnection(host, int(port))
    try:
        conn.request("POST", path, body=body, headers=headers)
        r = conn.getresponse()
        return r.status, json.loads(r.read() or b"{}")
    finally:
        conn.close()


def test_static_pages_and_enums_need_no_token(served):
    _, base = served
    status, body, headers = get(base, "/", raw=True)
    assert status == 200 and b"Level R" in body and headers["Content-Type"].startswith("text/html")
    assert get(base, "/app.js", raw=True)[2]["Content-Type"].startswith(("text/javascript", "application/javascript"))
    status, en, _ = get(base, "/api/enums")
    assert status == 200 and en["max_acceptable"] == 2 and "white_matter" in en["primary_hosts"]
    assert (en["lesion_types"], en["sides"], en["lobes"]) == (list(LESION_TYPES), list(SIDES), list(LOBES))


def test_invalid_or_missing_token_is_403_everywhere(served):
    store, base = served
    for path in ("/api/me", "/api/list", "/api/lesion/0", f"/api/volume/{CODE}.json", "/api/disagreements", "/api/adjudicate/1"):
        assert get(base, path, BAD)[0] == 403 and get(base, path)[0] == 403, path
    assert post(base, "/api/label", BAD, {"lesion_id": 0, **WM})[0] == 403
    assert post(base, "/api/adjudication", BAD, {"lesion_id": 1, **CX, "reason": "x"})[0] == 403
    assert store.label_rows() == [] and store.adjudication_rows() == []


def test_reader_sees_only_their_own_order_and_answers(served):
    store, base = served
    status, me, _ = get(base, "/api/me", T1)
    assert status == 200 and me == {"reader_id": "r1", "role": "reader", "display": "读者 1", "done": 0, "total": 3, "next": 2, "held": False}
    status, lst, _ = get(base, "/api/list", T1)
    assert [(o["position"], o["lesion_id"], o["is_pilot"], o["done"]) for o in lst] == [(0, 2, 1, False), (1, 0, 0, False), (2, 1, 0, False)]
    assert get(base, "/api/lesion/3", T1)[0] == 403                     # not in r1's list
    status, L, _ = get(base, "/api/lesion/2", T1)
    assert status == 200 and set(L["lesion"]) == {"lesion_id", "code", "volume_code", "z0", "z1", "boxes"} and L["answer"] is None
    store.submit_label("r2", 2, CX)
    assert get(base, "/api/lesion/2", T1)[1]["answer"] is None          # r2's answer is invisible to r1


def test_volume_routes_serve_blind_meta_and_raw_bytes(served):
    _, base = served
    status, meta, _ = get(base, f"/api/volume/{CODE}.json", T1)
    assert status == 200 and meta == {"shape": [4, 8, 8], "spacing_slice_mm": 5.0, "spacing_row_mm": 0.7, "spacing_col_mm": 0.7, "window": [0, 999]}
    status, body, headers = get(base, f"/api/volume/{CODE}.u16", T1, raw=True)
    assert status == 200 and len(body) == 4 * 8 * 8 * 2 and headers["Content-Type"] == "application/octet-stream"
    assert np.frombuffer(body, "<u2")[9] == 9
    assert get(base, "/api/volume/deadbeef.json", T1)[0] == 404


def test_label_submission_validates_appends_and_moves_progress(served):
    store, base = served
    assert post(base, "/api/label", T1, {"lesion_id": 2, **WM, "confidence": 0})[0] == 400
    assert post(base, "/api/label", T1, {"lesion_id": 3, **WM})[0] == 403
    assert store.label_rows() == []
    status, out = post(base, "/api/label", T1, {"lesion_id": 2, **WM})
    assert status == 200 and out["row_id"] >= 1
    status, out2 = post(base, "/api/label", T1, {"lesion_id": 2, **CX})
    assert out2["row_id"] > out["row_id"] and len(store.label_rows("r1")) == 2
    assert get(base, "/api/me", T1)[1]["done"] == 1
    assert get(base, "/api/lesion/2", T1)[1]["answer"]["primary_host"] == "cortex"
    assert post(base, "/api/label", TA, {"lesion_id": 2, **WM})[0] == 403    # adjudicator cannot label


def test_answers_carry_lesion_type_side_and_lobe_back_to_the_page(served):
    store, base = served
    assert post(base, "/api/label", T1, {"lesion_id": 2, **WM, "side": None})[0] == 400              # required unless 不是病灶
    assert post(base, "/api/label", T1, {"lesion_id": 2, **WM, "side": "midline", "lobe": "not_applicable"})[0] == 200
    ans = get(base, "/api/lesion/2", T1)[1]["answer"]
    assert (ans["lesion_type"], ans["side"], ans["lobe"]) == ("nonspecific_wm_lesion", "midline", "not_applicable")
    store.submit_label("r2", 2, {**WM, "lesion_type": "lacunar_infarct", "side": "midline"})         # same host, other type
    assert get(base, "/api/disagreements", TA)[1] == [{"lesion_id": 2, "done": False}]
    view = get(base, "/api/adjudicate/2", TA)[1]
    assert [(v["lesion_type"], v["side"], v["lobe"]) for v in view["readers"]] == [
        ("nonspecific_wm_lesion", "midline", "not_applicable"), ("lacunar_infarct", "midline", "frontal")]
    ruling = {"lesion_id": 2, **WM, "lesion_type": "lacunar_infarct", "side": "midline", "lobe": "not_applicable", "reason": "中心低信号"}
    assert post(base, "/api/adjudication", TA, ruling)[0] == 200
    ans = get(base, "/api/adjudicate/2", TA)[1]["answer"]
    assert (ans["lesion_type"], ans["side"], ans["lobe"]) == ("lacunar_infarct", "midline", "not_applicable")


def test_answer_keys_and_enums_pass_the_blinding_check():
    assert {"lesion_type", "side", "lobe"} <= set(ANSWER_KEYS)
    assert_blind({k: None for k in ANSWER_KEYS})
    assert_blind(enums())


def test_label_submission_validates_time_and_window(served):
    store, base = served
    for bad in ({"time_seconds": {}}, {"time_seconds": "abc"}, {"window": "x"}):
        status, out = post(base, "/api/label", T1, {"lesion_id": 2, **WM, **bad})
        assert status == 400 and "error" in out, bad
    assert store.label_rows() == []
    assert post(base, "/api/label", T1, {"lesion_id": 2, **WM})[0] == 200


def test_unexpected_errors_are_a_bare_500_and_the_traceback_goes_to_stderr(served, monkeypatch, capsys):
    store, base = served

    def boom(*args, **kwargs):
        raise RuntimeError("secret detail")
    monkeypatch.setattr(store, "progress", boom)
    monkeypatch.setattr(store, "submit_label", boom)
    status, body, _ = get(base, "/api/me", T1)
    assert (status, body) == (500, {"error": "internal error"})
    assert post(base, "/api/label", T1, {"lesion_id": 2, **WM}) == (500, {"error": "internal error"})
    assert get(base, "/api/me", BAD)[0] == 403 and get(base, "/api/lesion/99", TA)[0] == 404      # mapped errors unchanged
    err = capsys.readouterr().err
    assert err.count("RuntimeError: secret detail") == 2


def test_me_holds_a_reader_after_the_pilot_until_release(served):
    store, base = served
    assert post(base, "/api/label", T1, {"lesion_id": 2, **WM})[0] == 200    # lesion 2 is r1's whole pilot
    me = get(base, "/api/me", T1)[1]
    assert (me["done"], me["next"], me["held"]) == (1, None, True)
    assert get(base, "/api/me", T2)[1]["held"] is False                      # r2 has no pilot lesions
    store.release("r1")
    me = get(base, "/api/me", T1)[1]
    assert (me["done"], me["next"], me["held"]) == (1, 0, False)


def test_adjudicator_sees_disagreements_anonymously_and_can_rule(served):
    store, base = served
    for lid, (a, b) in {0: (WM, WM), 1: (WM, CX), 2: (CX, WM)}.items():
        store.submit_label("r1", lid, a)
        store.submit_label("r2", lid, b)
    assert get(base, "/api/disagreements", T1)[0] == 403
    for tok in (T1, T2):                                               # lesion 1 is a real disagreement: 403 is the role check
        assert get(base, "/api/adjudicate/1", tok)[0] == 403
        assert post(base, "/api/adjudication", tok, {"lesion_id": 1, **CX, "reason": "x"})[0] == 403
    assert store.adjudication_rows() == []
    status, dis, _ = get(base, "/api/disagreements", TA)
    assert status == 200 and dis == [{"lesion_id": 1, "done": False}, {"lesion_id": 2, "done": False}]
    assert get(base, "/api/me", TA)[1]["disagreements"] == 2
    status, view, _ = get(base, "/api/adjudicate/1", TA)
    assert status == 200 and [v["primary_host"] for v in view["readers"]] == ["white_matter", "cortex"]
    assert all("reader_id" not in v and "row_id" not in v for v in view["readers"]) and view["answer"] is None
    assert get(base, "/api/adjudicate/0", TA)[0] == 404                # agreed lesions are not adjudicated
    assert post(base, "/api/adjudication", TA, {"lesion_id": 1, **CX, "reason": ""})[0] == 400
    assert post(base, "/api/adjudication", TA, {"lesion_id": 0, **CX, "reason": "x"})[0] == 400
    status, out = post(base, "/api/adjudication", TA, {"lesion_id": 1, "primary_host": "cortex", "acceptable_hosts": ["cortex"],
                                                       "topography": "cortical", "adjacency": [], "ambiguity": "certain",
                                                       "lesion_type": "nonspecific_wm_lesion", "side": "image_left", "lobe": "frontal",
                                                       "reason": "皮层内"})
    assert status == 200 and out["row_id"] == 1
    assert get(base, "/api/disagreements", TA)[1][0] == {"lesion_id": 1, "done": True}
    assert get(base, "/api/adjudicate/1", TA)[1]["answer"]["reason"] == "皮层内"
    assert get(base, "/api/lesion/1", TA)[0] == 200                     # adjudicator may view any lesion


def test_disagreement_is_open_again_when_a_reader_revises_after_the_ruling(served, monkeypatch):
    store, base = served
    ticks = itertools.count()                                          # one second per stored row, see test_level_r_store
    monkeypatch.setattr(store_module, "now_iso", lambda: f"2026-10-01T00:00:{next(ticks):02d}+00:00")
    store.submit_label("r1", 1, WM)
    store.submit_label("r2", 1, CX)
    ruling = {"lesion_id": 1, "primary_host": "cortex", "acceptable_hosts": ["cortex"], "topography": "cortical", "adjacency": [],
              "ambiguity": "certain", "lesion_type": "nonspecific_wm_lesion", "side": "image_left", "lobe": "frontal", "reason": "皮层内"}
    assert post(base, "/api/adjudication", TA, ruling)[0] == 200
    assert get(base, "/api/disagreements", TA)[1] == [{"lesion_id": 1, "done": True}]
    store.submit_label("r1", 1, {**WM, "comment": "又看了一遍"})
    assert get(base, "/api/disagreements", TA)[1] == [{"lesion_id": 1, "done": False}]
    assert post(base, "/api/adjudication", TA, ruling)[0] == 200
    assert get(base, "/api/disagreements", TA)[1] == [{"lesion_id": 1, "done": True}]


def test_adjudicator_routes_report_409_without_two_readers(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    store.add_reader("adj", "adjudicator", TA, "裁定")
    app = tmp_path / "app"
    app.mkdir()
    (app / "index.html").write_text("x")
    srv = make_server(store, tmp_path / "data", "127.0.0.1", 0, app_dir=app)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        assert get(base, "/api/disagreements", TA)[0] == 409 and get(base, "/api/me", TA)[0] == 409
    finally:
        srv.shutdown()
        srv.server_close()


def test_server_script_refuses_a_missing_database(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/level_r_server.py"
    spec = importlib.util.spec_from_file_location("level_r_server_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "data" / "volumes").mkdir(parents=True)
    monkeypatch.setattr("sys.argv", ["level_r_server.py", "--db", str(tmp_path / "typo.sqlite"), "--data-root", str(tmp_path / "data"), "--port", "0"])
    with pytest.raises(sqlite3.OperationalError):
        mod.main()
    assert not (tmp_path / "typo.sqlite").exists()


def test_bad_content_length_header_is_400_not_a_dropped_connection(served):
    store, base = served
    body = json.dumps({"lesion_id": 2, **WM}).encode()
    status, out = post_raw(base, f"/api/label?token={T1}", {"Content-Type": "application/json", "Content-Length": "abc"}, body)
    assert status == 400 and "error" in out
    assert store.label_rows() == []


def test_non_dict_json_body_is_400_not_a_dropped_connection(served):
    store, base = served
    body = json.dumps([1, 2, 3]).encode()
    status, out = post_raw(base, f"/api/label?token={T1}", {"Content-Type": "application/json"}, body)
    assert status == 400 and "error" in out
    assert store.label_rows() == []
