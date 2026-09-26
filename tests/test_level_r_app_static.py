import json
import re
from pathlib import Path

from anatobind.level_r.schema import (
    ADJACENCY, AMBIGUITY, LESION_TYPES, LOBES, LOCAL_QUALITY, NOT_A_LESION, PRIMARY_HOSTS, SIDES, TOPOGRAPHY,
)
from anatobind.level_r.server import APP_DIR, STATIC

ROUTES = ("/api/enums", "/api/me", "/api/list", "/api/lesion/", "/api/volume/", "/api/label", "/api/disagreements",
          "/api/adjudicate/", "/api/adjudication")
IDS = ("view", "off", "prev", "next", "zslider", "zlabel", "wlabel", "wreset", "zoom1", "zoom2", "zoom4", "primary_host",
       "acceptable_hosts", "topography", "adjacency", "ambiguity", "not_a_lesion", "local_quality", "confidence", "comment",
       "reason", "submit", "list", "readers", "status", "who", "progress", "lcode", "btn-list", "btn-next", "sec-list", "sec-lesion",
       "lesion_type", "side", "lobe")
NEW_FIELDS = {"lesion_type": ("lesion_types", "病灶类型", "请选择病灶类型"), "side": ("sides", "侧别", "请选择侧别"),
              "lobe": ("lobes", "脑叶", "请选择脑叶")}


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


def _js_object(js, name):
    return json.loads(re.search(rf"const {name} = (\{{.*?\}});", js, re.S).group(1))


def _js_function(js, name):
    """Source of one top-level function of app.js, up to the next top-level declaration."""
    start = js.index(f"function {name}(")
    return js[start:start + re.search(r"\n(?:async )?function |\n\$\(|\nconst ", js[start + 1:]).start() + 1]


def test_zh_labels_cover_every_enum_key_and_are_json():
    zh = _js_object(_read("app.js"), "ZH")
    for group in (PRIMARY_HOSTS, TOPOGRAPHY, ADJACENCY, AMBIGUITY, LOCAL_QUALITY, LESION_TYPES, SIDES, LOBES):
        for k in group:
            assert k in zh and zh[k], k
    assert NOT_A_LESION in zh
    assert zh["other"] == "其他"                     # shared by host and lesion type; the host select says 脑室内 goes there
    assert zh["image_left"] == "图像左侧" and zh["image_right"] == "图像右侧" and zh["lacunar_infarct"] == "腔隙性梗死"
    assert "胼胝体" in zh["not_applicable"]           # 不适用 covers everything that belongs to no lobe, not only deep grey / infratentorial


def test_lesion_type_side_and_lobe_selects_sit_between_not_a_lesion_and_the_host():
    idx = _read("index.html")
    pos = [idx.index(f'id="{i}"') for i in ("not_a_lesion", "lesion_type", "side", "lobe", "primary_host")]
    assert pos == sorted(pos)
    for label in ('病灶类型 <select id="lesion_type">', '侧别（以屏幕左右为准） <select id="side">', '脑叶 <select id="lobe">',
                  '主宿主结构（脑室内病灶选"其他"） <select id="primary_host">'):
        assert label in idx, label
    assert "血管周围间隙不算" in idx                  # the checkbox itself says a perivascular space is a lesion type, not 不是病灶


def test_app_fills_disables_sends_checks_and_restores_lesion_type_side_and_lobe():
    js = _read("app.js")
    field_zh = _js_object(js, "FIELD_ZH")
    build, toggle, read, check, fill_form, readers = (_js_function(js, f) for f in
                                                      ("buildForm", "toggleNal", "readForm", "validate", "fillForm", "showReaders"))
    for k, (enum_key, name, message) in NEW_FIELDS.items():
        assert f'fill("{k}", en.{enum_key})' in build, k
        assert f'"{k}"' in toggle, k                                          # disabled when 不是病灶 is ticked
        assert f'{k}: nal ? null : ($("{k}").value || null)' in read, k       # null for a not_a_lesion answer
        assert f'if (!p.{k}) return "{message}";' in check, k
        assert f'$("{k}").value = a.{k} || "";' in fill_form, k
        assert f'"{k}"' in readers and field_zh[k] == name, k                  # a row of the adjudicator's table


def test_app_uses_the_routes_the_server_serves_and_reads_the_token_from_the_url():
    js = _read("app.js")
    for r in ROUTES:
        assert r in js, r
    assert "URLSearchParams" in js and '"token"' in js
    assert "time_seconds" in js and "imageSmoothingEnabled = false" in js and "Uint16Array" in js


def test_reader_facing_sources_never_mention_what_readers_must_not_see():
    """The guide is reader-facing too; it may say 公开数据集 but not the field strength (22 of the 165 volumes are 1.5 T),
    series, patient, SynthSeg, the fastMRI+ label strings or fastMRI+. Since follow-up A1 the readers answer the lesion
    type themselves, so the Chinese clinical terms (非特异性白质病灶, 腔隙性梗死) are answer options offered for every
    lesion alike; the English label strings of the dataset stay out."""
    text = _read("index.html") + _read("app.js") + _read("style.css") + _read("guide.html")
    for word in ("SynthSeg", "synthseg", "patient", "series", "d_interface", "stratum", "Nonspecific", "Lacunar", "white matter lesion",
                 "fastMRI+", "3T", "3 T", "1.5T", "1.5 T", "Tesla", "特斯拉"):
        assert word not in text, word


def test_adjudicator_gets_no_comment_box_since_adjudications_store_none():
    assert '<label class="reader-only">备注 <textarea id="comment"' in _read("index.html")


def test_zoom_centre_is_fixed_per_lesion_thumbnails_keep_the_aspect_ratio_and_volume_errors_surface():
    js = _read("app.js")
    assert "state.center = lesionCenter(" in js and "[cy, cx] = state.center" in js
    assert "Math.round(200 * R / C)" in js
    assert "if (!r.ok) throw" in js


def test_guide_defines_primary_host_operationally():
    g = _read("guide.html")
    for phrase in ("最主要、最合理的解剖宿主组织", "不是框内重叠最大的结构", "可接受", "近皮层", "不是病灶"):
        assert phrase in g, phrase


def test_guide_defines_lesion_type_screen_side_and_lobe():
    g = _read("guide.html")
    for phrase in ("病灶类型", "非特异性白质病灶", "腔隙性梗死", "血管周围间隙", "中心 CSF 样低信号", "无高信号环",
                   "侧别", "以屏幕左右为准", "不是患者左右", "统一换算", "中线",
                   "脑叶", "额叶", "顶叶", "颞叶", "枕叶", "岛叶", "不适用", "深部灰质"):
        assert phrase in g, phrase


def test_guide_settles_two_lobe_lesions_non_lobar_white_matter_and_the_scope_of_confidence():
    g = _read("guide.html")
    assert "占多的那个脑叶" in g          # a lesion spanning two lobes takes the lobe holding more of it
    assert "病灶中心所在的脑叶" in g      # and when that cannot be judged, the lobe holding the lesion's centre
    assert "半卵圆中心" in g              # corona radiata / centrum semiovale take the lobe of the cortex above them
    assert "胼胝体" in g                  # corpus callosum and other non-lobar white matter: 不适用, structure named in the comment
    assert "最没把握" in g                # confidence covers host, lesion type and side: the least certain of the three


def test_app_stops_at_the_end_of_the_pilot_and_the_guide_says_so():
    js = _read("app.js")
    assert "me.held" in js and "pilot 已完成，请等待通知再继续" in js
    g = _read("guide.html")
    assert "等我们通知" in g and "之后在同一链接里继续读完全部" not in g


def test_app_escapes_server_strings_and_keeps_arrow_keys_inside_form_controls():
    js = _read("app.js")
    assert "function esc(" in js
    assert "esc(fmt(" in js
    assert '["TEXTAREA", "SELECT", "INPUT", "BUTTON"]' in js
    assert "已保存，但加载下一例失败" in js


def test_open_lesion_assigns_state_only_after_the_volume_loaded():
    js = _read("app.js")
    src = js[js.index("async function openLesion"):]
    assert src.index("await loadVolume(") < src.index("state.lesion =")
