import json
import re
from pathlib import Path

from anatobind.level_r.schema import ADJACENCY, AMBIGUITY, LOCAL_QUALITY, NOT_A_LESION, PRIMARY_HOSTS, TOPOGRAPHY
from anatobind.level_r.server import APP_DIR, STATIC

ROUTES = ("/api/enums", "/api/me", "/api/list", "/api/lesion/", "/api/volume/", "/api/label", "/api/disagreements",
          "/api/adjudicate/", "/api/adjudication")
IDS = ("view", "off", "prev", "next", "zslider", "zlabel", "wlabel", "wreset", "zoom1", "zoom2", "zoom4", "primary_host",
       "acceptable_hosts", "topography", "adjacency", "ambiguity", "not_a_lesion", "local_quality", "confidence", "comment",
       "reason", "submit", "list", "readers", "status", "who", "progress", "lcode", "btn-list", "btn-next", "sec-list", "sec-lesion")


def _read(name):
    return (APP_DIR / name).read_text(encoding="utf-8")


def test_every_static_page_the_server_maps_exists():
    for name in set(STATIC.values()):
        assert (APP_DIR / name).is_file(), name


def test_index_wires_script_style_guide_and_every_element_the_script_uses():
    idx = _read("index.html")
    assert 'src="app.js"' in idx and 'href="style.css"' in idx and 'href="guide.html"' in idx and 'lang="zh-CN"' in idx
    for i in IDS:
        assert f'id="{i}"' in idx, i


def test_zh_labels_cover_every_enum_key_and_are_json():
    js = _read("app.js")
    m = re.search(r"const ZH = (\{.*?\});", js, re.S)
    zh = json.loads(m.group(1))
    for group in (PRIMARY_HOSTS, TOPOGRAPHY, ADJACENCY, AMBIGUITY, LOCAL_QUALITY):
        for k in group:
            assert k in zh and zh[k], k
    assert NOT_A_LESION in zh


def test_app_uses_the_routes_the_server_serves_and_reads_the_token_from_the_url():
    js = _read("app.js")
    for r in ROUTES:
        assert r in js, r
    assert "URLSearchParams" in js and '"token"' in js
    assert "time_seconds" in js and "imageSmoothingEnabled = false" in js and "Uint16Array" in js


def test_reader_facing_sources_never_mention_what_readers_must_not_see():
    text = _read("index.html") + _read("app.js") + _read("style.css")
    for word in ("SynthSeg", "synthseg", "patient", "series", "d_interface", "stratum", "Nonspecific", "Lacunar", "fastMRI+"):
        assert word not in text, word


def test_guide_defines_primary_host_operationally():
    g = _read("guide.html")
    for phrase in ("最主要、最合理的解剖宿主组织", "不是框内重叠最大的结构", "可接受", "近皮层", "不是病灶"):
        assert phrase in g, phrase


def test_app_escapes_server_strings_and_keeps_arrow_keys_inside_form_controls():
    js = _read("app.js")
    assert "function esc(" in js
    assert "esc(fmt(" in js
    assert '["TEXTAREA", "SELECT", "INPUT", "BUTTON"]' in js
    assert "已保存，但加载下一例失败" in js
