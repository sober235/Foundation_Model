# Level R 读片协议与工具（PR-B）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让两位放射科医生在浏览器里独立回答 1297 个 fastMRI+ 脑 FLAIR 小病灶"在哪块脑区"，第三位裁分歧，答案落在服务器 sqlite 里，按患者五折封存，并给出 pilot 150 的一致率与用时报告。

**Architecture:** 一条导出脚本把 Gate 0.5 注册表里的 165 卷 RSS 写成 16 位小端数组 + 几何 JSON，病灶逐层框沿 Gate 0.5 同一路径重算并与注册表逐条核对。服务端是 Python 标准库 `http.server` + `sqlite3` 的单文件服务，只追加写；发给浏览器的每个 JSON 都先过字段白名单。前端一页纯 HTML/JS（canvas 调窗、翻层、放大、表单）。统计与封存是两个纯函数模块，报告脚本把 pilot 的一致率、κ、AC1、分层与用时写成 markdown。

**Tech Stack:** Python 3.11（`~/anaconda3/envs/nvgen`），只用标准库 + numpy + h5py；前端 vanilla JS，无构建；不加任何新依赖。

**Spec:** `docs/superpowers/specs/2026-09-25-level-r-annotation-tooling-design.md`（决定 R1–R10）。上游：v2.6 §3（本体）、§7（读片协议）、§12.7（封存）、§18（工具最低功能）、§25 第 9 项（四档分层、全集标注）。`docs/handoff/2026-09-25-assessment-review-corrections.md` §8 给出 `primary_host` 的临床操作定义，写进读片说明页。

## Global Constraints

- 数据只从 `/data2/congcong/data/FM_data` 读；服务器数据根 `/data2/congcong/data/FM_data/derived/level_r/`（不入库）。fastMRI+ 脑标注 CSV：`fastMRI_lh_brain_knee/Annotations/brain.csv`；h5 在 `fastMRI_lh_brain_knee/kspace/brain/{multicoil_train,multicoil_val}/`。
- 病灶清单 = `docs/verification/2026-09-24/gate05/lesions.csv`（1297 行，全部 `status=ok`，165 卷 = 165 名患者），`lesion_id` 是全局连接键（spec R6）。列：`lesion_id,file,patient_id,series,stratum_series,stratum_geometry,label,z0,z1,n_slices,x0,y0,x1,y1,inplane_mm,spacing_row_mm,spacing_col_mm,spacing_slice_mm,host_lookup_all,host_lookup_parenchyma,host_class_nearest,d1_mm,d_interface_mm,delta_d_mm,status`。
- 本文所有 `python` 指 `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python`；测试命令 `python -m pytest tests/<file> -q -p no:cacheprovider`；全套 `python -m pytest tests/ -q -p no:cacheprovider`（基线 413 passed）。测试不得读 `/data2`（GitHub Actions 也跑）。
- 每个函数先写测试再写实现（CLAUDE.md）。CPU 任务 `nice -n 19`。
- 医生看不到：SynthSeg、模型输出、距离档、采集系列、患者号、fastMRI+ 原标签文字、h5 stem（spec R5）。白名单在 `anatobind/level_r/blind.py`，服务端每个 JSON 响应都过 `assert_blind`。
- `labels` / `adjudications` 两表只追加，`store.py` 源码里不得出现 UPDATE / DELETE 两个词（连注释也不行，测试用 grep 守）。
- 表单枚举（spec §5，英文键入库、中文界面）：`primary_host ∈ {white_matter, cortex, thalamus, basal_ganglia, brainstem, cerebellum, other}`；`acceptable_hosts` 多选 ≤ 2 且含 `primary_host`；`topography ∈ {periventricular, juxtacortical, cortical, deep_white_matter, infratentorial}`；`adjacency ⊆ {adjacent_to_cortex, adjacent_to_ventricle, crosses_boundary, none}`；`ambiguity ∈ {certain, two_host, multi_structure, insufficient_resolution}`；`not_a_lesion` 勾选后宿主可空；`local_quality ∈ {good, fair, poor}`；`confidence ∈ 1..5`；`comment` 自由文字。
- 折：165 名患者按 `patient_id`、种子 0 分五折，断言患者不跨折（复用 `anatobind.data_engine.fastmri_knee.assert_folds_by_patient`）。封存一次性，永不覆盖已有文件（用户规矩：任何情况不删不覆盖数据）。
- pilot：150 例，格子 = 距离四档 `0 / (0, 2] / (2, 4] / > 4 mm` × `stratum_geometry` 四层，非空格子至少 8（不足 8 全取），每患者 ≤ 3，种子 0。
- 一致率门 R7：全体 raw 的 95% 患者 bootstrap 区间下限 ≥ 0.80，且 0 mm 档 raw ≥ 0.70。
- 服务默认 `127.0.0.1:8790`；`--bind 0.0.0.0` 须显式打开。token 16 位十六进制，库里只存 sha256。
- 提交：作者用仓库本地配置；消息英文、句首大写、描述做了什么；不写任何 AI 痕迹。分支 `plan/level-r-tooling-2026-09-25`，工作树 `/data0/congcong/code/Project_Doing/foundation_model-levelr`。

## 文件结构

新建：

| 文件 | 职责 |
|---|---|
| `anatobind/level_r/__init__.py` | 空 |
| `anatobind/level_r/schema.py` | 枚举、`validate_label` / `validate_adjudication`、`enums()` |
| `anatobind/level_r/registry.py` | 读 Gate 0.5 注册表、距离四档、`volume_code` / `lesion_code` |
| `anatobind/level_r/blind.py` | 白名单、`assert_blind` |
| `anatobind/level_r/export.py` | 逐层框重算与核对、u16 转换、写卷、`export_all` |
| `anatobind/level_r/store.py` | sqlite `Store`：读者、病灶、顺序、标签、裁定、分歧、最终标签 |
| `anatobind/level_r/server.py` | `make_handler` / `make_server`（API + 静态页） |
| `anatobind/level_r/app/{index.html,app.js,style.css,guide.html}` | 读者页、裁定页、读片说明 |
| `anatobind/level_r/admin.py` | 患者折、读者顺序、CSV 导出、封存、备份 |
| `anatobind/level_r/pilot.py` | pilot 150 抽样 |
| `anatobind/eval/level_r_stats.py` | raw、κ、AC1、positive agreement、混淆、集合值、患者 bootstrap、分层、用时、门 |
| `anatobind/eval/level_r_labels.py` | `load_train_labels(k)` / `load_test_labels(k, unblind)`，sha256 核对，访问日志 |
| `scripts/level_r_export.py` | 导出入口 |
| `scripts/level_r_admin.py` | `folds / init / add-reader / order / export / seal / backup` |
| `scripts/level_r_server.py` | 起服务 |
| `scripts/level_r_pilot_sample.py` | 抽 pilot → `data/level_r/pilot_150.json` |
| `scripts/level_r_report.py` | pilot 报告 → `docs/verification/<日期>/level_r_pilot.md` |
| `data/level_r/folds.json`、`data/level_r/pilot_150.json` | 入库的折表与 pilot 清单（Task 7、8 生成） |
| `docs/level_r_tool.md` | 部署与运维步骤 |
| `tests/test_level_r_*.py` | 每个模块一份 |

修改：`CLAUDE.md`（代码地图加 Level R 一段；Task 12）、`STATUS.md`（Task 13 交接时整体重写）。

依赖顺序：Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13。Task 9/10 只依赖 Task 1/7，可与 5/6 并行。

---

### Task 1: 标注 schema 与注册表读取

**Files:**
- Create: `anatobind/level_r/__init__.py`（空文件）
- Create: `anatobind/level_r/schema.py`
- Create: `anatobind/level_r/registry.py`
- Test: `tests/test_level_r_schema.py`、`tests/test_level_r_registry.py`

**Interfaces:**
- Produces: `schema.PRIMARY_HOSTS / TOPOGRAPHY / ADJACENCY / AMBIGUITY / LOCAL_QUALITY: tuple[str]`、`MAX_ACCEPTABLE = 2`、`NOT_A_LESION = "not_a_lesion"`、`class InvalidLabel(ValueError)`、`validate_label(p: dict, require_quality=True) -> dict`（键 `primary_host, acceptable_hosts, topography, adjacency, ambiguity, not_a_lesion, local_quality, confidence, comment`）、`validate_adjudication(p) -> dict`（多一个 `reason`）、`enums() -> dict`。
- Produces: `registry.REGISTRY: Path`、`BANDS = ("0", "0-2", "2-4", ">4")`、`load_registry(path=REGISTRY) -> list[dict]`（整数/浮点已解析，多一个 `band` 键）、`distance_band(d_mm) -> str`、`is_3mm(stratum_geometry) -> bool`、`volume_code(stem) -> str`（8 位十六进制）、`lesion_code(lesion_id) -> str`。

- [ ] **Step 1: 写 schema 的失败测试**

```python
# tests/test_level_r_schema.py
import pytest

from anatobind.level_r.schema import (
    ADJACENCY, AMBIGUITY, LOCAL_QUALITY, MAX_ACCEPTABLE, PRIMARY_HOSTS, TOPOGRAPHY, InvalidLabel, enums,
    validate_adjudication, validate_label,
)

GOOD = {"primary_host": "white_matter", "acceptable_hosts": ["white_matter", "cortex"], "topography": "juxtacortical",
        "adjacency": ["adjacent_to_cortex"], "ambiguity": "two_host", "not_a_lesion": False, "local_quality": "good",
        "confidence": 4, "comment": "贴皮层"}


def test_valid_label_is_normalised_and_keeps_every_field():
    out = validate_label(GOOD)
    assert out == {**GOOD, "comment": "贴皮层"}


@pytest.mark.parametrize("bad", [
    {**GOOD, "primary_host": None},                                    # 缺主宿主
    {**GOOD, "primary_host": "ventricle"},                             # 不在枚举里
    {**GOOD, "acceptable_hosts": ["white_matter", "cortex", "thalamus"]},   # 集合超 2
    {**GOOD, "acceptable_hosts": ["cortex"]},                          # 集合不含主宿主
    {**GOOD, "acceptable_hosts": ["white_matter", "white_matter"]},    # 重复
    {**GOOD, "confidence": 0}, {**GOOD, "confidence": 6}, {**GOOD, "confidence": "4"}, {**GOOD, "confidence": True},
    {**GOOD, "topography": "lobar"}, {**GOOD, "ambiguity": "unsure"}, {**GOOD, "local_quality": "ok"},
    {**GOOD, "adjacency": ["none", "adjacent_to_cortex"]},             # none 排斥其他
    {**GOOD, "adjacency": ["near_cortex"]},
    {**GOOD, "not_a_lesion": True},                                    # 不是病灶却给了宿主
])
def test_invalid_labels_raise(bad):
    with pytest.raises(InvalidLabel):
        validate_label(bad)


def test_not_a_lesion_answer_needs_no_host_topography_or_ambiguity():
    out = validate_label({"not_a_lesion": True, "local_quality": "poor", "confidence": 2, "adjacency": []})
    assert out["primary_host"] is None and out["acceptable_hosts"] == [] and out["topography"] is None
    assert out["ambiguity"] is None and out["not_a_lesion"] is True and out["comment"] == ""


def test_adjudication_requires_a_reason_and_not_quality_or_confidence():
    p = {k: v for k, v in GOOD.items() if k not in ("local_quality", "confidence")}
    with pytest.raises(InvalidLabel):
        validate_adjudication(p)
    out = validate_adjudication({**p, "reason": "皮层信号连续"})
    assert out["reason"] == "皮层信号连续" and out["local_quality"] is None and out["confidence"] is None


def test_enums_expose_every_list_and_the_cap():
    e = enums()
    assert e == {"primary_hosts": list(PRIMARY_HOSTS), "topography": list(TOPOGRAPHY), "adjacency": list(ADJACENCY),
                 "ambiguity": list(AMBIGUITY), "local_quality": list(LOCAL_QUALITY), "max_acceptable": MAX_ACCEPTABLE}
    assert "other" in PRIMARY_HOSTS and "none" in ADJACENCY and MAX_ACCEPTABLE == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_level_r_schema.py -q -p no:cacheprovider`
Expected: FAIL，`ModuleNotFoundError: No module named 'anatobind.level_r'`

- [ ] **Step 3: 写 schema.py**

```python
# anatobind/level_r/schema.py
"""Level R annotation schema (spec §5; v2.6 §3, §7.5). English keys go into the database, the Chinese labels live in
the app. validate_label is the single validation both the server and the tests rely on; the browser repeats the same
rules for immediate feedback but is never trusted."""

PRIMARY_HOSTS = ("white_matter", "cortex", "thalamus", "basal_ganglia", "brainstem", "cerebellum", "other")
TOPOGRAPHY = ("periventricular", "juxtacortical", "cortical", "deep_white_matter", "infratentorial")
ADJACENCY = ("adjacent_to_cortex", "adjacent_to_ventricle", "crosses_boundary", "none")
AMBIGUITY = ("certain", "two_host", "multi_structure", "insufficient_resolution")
LOCAL_QUALITY = ("good", "fair", "poor")
MAX_ACCEPTABLE = 2
NOT_A_LESION = "not_a_lesion"      # the class a not_a_lesion answer takes in agreement statistics
MAX_COMMENT = 2000


class InvalidLabel(ValueError):
    pass


def _choice(p, key, allowed, optional):
    v = p.get(key)
    if optional and v in (None, ""):
        return None
    if v not in allowed:
        raise InvalidLabel(f"{key} {v!r} must be one of {allowed}")
    return v


def validate_label(p, require_quality=True):
    """Return the normalised answer or raise InvalidLabel. p: the JSON body of one submission.
    require_quality=False (adjudications) drops local_quality and confidence."""
    out = {"not_a_lesion": bool(p.get("not_a_lesion", False))}
    acc = p.get("acceptable_hosts") or []
    if not isinstance(acc, list) or len(acc) != len(set(acc)):
        raise InvalidLabel("acceptable_hosts must be a list without repeats")
    if len(acc) > MAX_ACCEPTABLE:
        raise InvalidLabel(f"acceptable_hosts has {len(acc)} entries; at most {MAX_ACCEPTABLE}")
    for h in acc:
        if h not in PRIMARY_HOSTS:
            raise InvalidLabel(f"unknown host {h!r}")
    host = p.get("primary_host") or None
    if out["not_a_lesion"]:
        if host is not None or acc:
            raise InvalidLabel("a not_a_lesion answer carries no host")
    else:
        if host not in PRIMARY_HOSTS:
            raise InvalidLabel(f"primary_host {host!r} is required and must be one of {PRIMARY_HOSTS}")
        if host not in acc:
            raise InvalidLabel("acceptable_hosts must contain primary_host")
    out["primary_host"], out["acceptable_hosts"] = host, list(acc)
    out["topography"] = _choice(p, "topography", TOPOGRAPHY, optional=out["not_a_lesion"])
    out["ambiguity"] = _choice(p, "ambiguity", AMBIGUITY, optional=out["not_a_lesion"])
    adj = p.get("adjacency") or []
    if not isinstance(adj, list) or len(adj) != len(set(adj)) or any(a not in ADJACENCY for a in adj):
        raise InvalidLabel(f"adjacency must be a subset of {ADJACENCY}")
    if "none" in adj and len(adj) > 1:
        raise InvalidLabel("adjacency 'none' excludes the other choices")
    out["adjacency"] = list(adj)
    if require_quality:
        out["local_quality"] = _choice(p, "local_quality", LOCAL_QUALITY, optional=False)
        c = p.get("confidence")
        if isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= 5:
            raise InvalidLabel("confidence must be an integer 1..5")
        out["confidence"] = c
    else:
        out["local_quality"], out["confidence"] = None, None
    out["comment"] = str(p.get("comment") or "")[:MAX_COMMENT]
    return out


def validate_adjudication(p):
    out = validate_label(p, require_quality=False)
    reason = str(p.get("reason") or "").strip()
    if not reason:
        raise InvalidLabel("reason is required for an adjudication")
    out["reason"] = reason[:MAX_COMMENT]
    return out


def enums():
    return {"primary_hosts": list(PRIMARY_HOSTS), "topography": list(TOPOGRAPHY), "adjacency": list(ADJACENCY),
            "ambiguity": list(AMBIGUITY), "local_quality": list(LOCAL_QUALITY), "max_acceptable": MAX_ACCEPTABLE}
```

- [ ] **Step 4: 跑 schema 测试确认通过**

Run: `python -m pytest tests/test_level_r_schema.py -q -p no:cacheprovider`
Expected: 全部 PASS（1 + 15 参数化 + 3）

- [ ] **Step 5: 写 registry 的失败测试**

```python
# tests/test_level_r_registry.py
import csv

import pytest

from anatobind.level_r.registry import BANDS, distance_band, is_3mm, lesion_code, load_registry, volume_code

HEADER = ["lesion_id", "file", "patient_id", "series", "stratum_series", "stratum_geometry", "label", "z0", "z1", "n_slices",
          "x0", "y0", "x1", "y1", "inplane_mm", "spacing_row_mm", "spacing_col_mm", "spacing_slice_mm", "host_lookup_all",
          "host_lookup_parenchyma", "host_class_nearest", "d1_mm", "d_interface_mm", "delta_d_mm", "status"]


def _row(lesion_id, file="file_brain_AXFLAIR_200_1", patient="P1", d_interface=0.0, stratum="inplane_0.69_slice_5", status="ok"):
    return dict(zip(HEADER, [lesion_id, file, patient, "200", "200_201", stratum, "Nonspecific white matter lesion", 2, 2, 1,
                             91, 182, 98, 188, 4.8125, 0.6875, 0.6875, 5.0, 41, 41, "white_matter", 0.0, d_interface,
                             d_interface, status]))


def write_registry(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    return path


def test_load_registry_parses_numbers_and_adds_the_band(tmp_path):
    p = write_registry(tmp_path / "lesions.csv", [_row(0, d_interface=0.0), _row(1, d_interface=5.09), _row(2, status="outside")])
    reg = load_registry(p)
    assert [r["lesion_id"] for r in reg] == [0, 1]                     # non-ok rows are dropped
    assert reg[0]["z0"] == 2 and isinstance(reg[0]["x1"], int) and reg[0]["spacing_row_mm"] == 0.6875
    assert reg[0]["band"] == "0" and reg[1]["band"] == ">4"
    assert reg[0]["patient_id"] == "P1" and reg[0]["file"] == "file_brain_AXFLAIR_200_1"


def test_load_registry_refuses_duplicate_ids(tmp_path):
    p = write_registry(tmp_path / "lesions.csv", [_row(3), _row(3)])
    with pytest.raises(ValueError):
        load_registry(p)


@pytest.mark.parametrize("d, band", [(0.0, "0"), (-0.0, "0"), (0.001, "0-2"), (2.0, "0-2"), (2.01, "2-4"), (4.0, "2-4"), (4.01, ">4"), (30.0, ">4")])
def test_distance_band_edges_follow_v26_item_9(d, band):
    assert distance_band(d) == band and band in BANDS


def test_is_3mm_reads_the_geometry_stratum():
    assert is_3mm("inplane_0.62_slice_3") and is_3mm("inplane_0.86_slice_3") and not is_3mm("inplane_0.69_slice_5")


def test_codes_are_eight_hex_chars_stable_and_distinct():
    a, b = volume_code("file_brain_AXFLAIR_200_6002425"), volume_code("file_brain_AXFLAIR_200_6002426")
    assert len(a) == 8 and int(a, 16) >= 0 and a != b and a == volume_code("file_brain_AXFLAIR_200_6002425")
    assert lesion_code(7) == lesion_code("7") and lesion_code(7) != lesion_code(8) and len(lesion_code(7)) == 8
    assert "6002425" not in a                                            # the stem is not recoverable by eye
```

- [ ] **Step 6: 跑测试确认失败**

Run: `python -m pytest tests/test_level_r_registry.py -q -p no:cacheprovider`
Expected: FAIL，`ImportError`（registry 模块不存在）

- [ ] **Step 7: 写 registry.py**

```python
# anatobind/level_r/registry.py
"""The Level R lesion registry is Gate 0.5's lesions.csv (docs/verification/2026-09-24/gate05/lesions.csv): 1297
fastMRI+ FLAIR small lesions in the RSS frame with patient_id, d_interface and the measured-geometry stratum.
lesion_id is the join key for orders, labels, statistics and sealing (spec R6). Readers never see this file."""
import csv
import hashlib
from pathlib import Path

REGISTRY = Path("docs/verification/2026-09-24/gate05/lesions.csv")
INT_FIELDS = ("lesion_id", "z0", "z1", "n_slices", "x0", "y0", "x1", "y1")
FLOAT_FIELDS = ("inplane_mm", "spacing_row_mm", "spacing_col_mm", "spacing_slice_mm", "d1_mm", "d_interface_mm", "delta_d_mm")
BANDS = ("0", "0-2", "2-4", ">4")          # v2.6 §25 item 9: d_interface 0 / (0, 2] / (2, 4] / > 4 mm


def distance_band(d_mm):
    if d_mm <= 0:
        return "0"
    if d_mm <= 2:
        return "0-2"
    if d_mm <= 4:
        return "2-4"
    return ">4"


def is_3mm(stratum_geometry):
    return stratum_geometry.endswith("_slice_3")


def load_registry(path=REGISTRY):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if r["status"] != "ok":
            continue
        rec = dict(r)
        for k in INT_FIELDS:
            rec[k] = int(r[k])
        for k in FLOAT_FIELDS:
            rec[k] = float(r[k])
        rec["band"] = distance_band(rec["d_interface_mm"])
        out.append(rec)
    ids = [r["lesion_id"] for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate lesion_id")
    return out


def volume_code(stem):
    """Stable 8-hex code shown to readers instead of the h5 stem (spec §4)."""
    return hashlib.sha256(stem.encode()).hexdigest()[:8]


def lesion_code(lesion_id):
    return hashlib.sha256(f"lesion:{int(lesion_id)}".encode()).hexdigest()[:8]
```

- [ ] **Step 8: 跑两份测试确认通过**

Run: `python -m pytest tests/test_level_r_schema.py tests/test_level_r_registry.py -q -p no:cacheprovider`
Expected: 全部 PASS

- [ ] **Step 9: 提交**

```bash
git add anatobind/level_r/__init__.py anatobind/level_r/schema.py anatobind/level_r/registry.py tests/test_level_r_schema.py tests/test_level_r_registry.py
git commit -m "Level R: annotation schema with validation and the Gate 0.5 registry loader"
```

---

### Task 2: 盲化白名单

**Files:**
- Create: `anatobind/level_r/blind.py`
- Test: `tests/test_level_r_blind.py`

**Interfaces:**
- Produces: `LESION_KEYS = ("lesion_id", "code", "volume_code", "z0", "z1", "boxes")`、`VOLUME_KEYS = ("shape", "spacing_slice_mm", "spacing_row_mm", "spacing_col_mm", "window")`、`FORBIDDEN: tuple[str]`（键名子串黑名单，只供测试与 `assert_blind` 复核）、`class BlindingError(ValueError)`、`blind_lesion(d) -> dict`、`blind_volume_meta(d) -> dict`、`assert_blind(obj) -> obj`（递归检查任意 JSON 结构的字典键，命中黑名单就抛 `BlindingError`）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_level_r_blind.py
import pytest

from anatobind.level_r.blind import FORBIDDEN, BlindingError, assert_blind, blind_lesion, blind_volume_meta

FULL_LESION = {"lesion_id": 3, "code": "ab12cd34", "volume_code": "0f0f0f0f", "z0": 2, "z1": 3, "boxes": {"2": [[10, 14, 20, 26]]},
               "label": "Lacunar infarct", "patient_id": "dcfc", "series": "200", "stratum_geometry": "inplane_0.69_slice_5",
               "d_interface_mm": 0.0, "delta_d_mm": 0.0, "file": "file_brain_AXFLAIR_200_6002425", "host_lookup_all": 41, "band": "0"}


def test_blind_lesion_keeps_only_the_whitelist():
    out = blind_lesion(FULL_LESION)
    assert out == {"lesion_id": 3, "code": "ab12cd34", "volume_code": "0f0f0f0f", "z0": 2, "z1": 3, "boxes": {"2": [[10, 14, 20, 26]]}}
    assert out is not FULL_LESION and "label" not in out


def test_blind_volume_meta_keeps_geometry_and_window_only():
    meta = {"shape": [16, 320, 320], "spacing_slice_mm": 5.0, "spacing_row_mm": 0.6875, "spacing_col_mm": 0.6875,
            "window": [1200, 48000], "scale_p999": 0.0002, "stem": "file_brain_AXFLAIR_200_6002425", "patient_id": "x"}
    assert blind_volume_meta(meta) == {"shape": [16, 320, 320], "spacing_slice_mm": 5.0, "spacing_row_mm": 0.6875,
                                       "spacing_col_mm": 0.6875, "window": [1200, 48000]}


@pytest.mark.parametrize("key", ["label", "d_interface_mm", "delta_d_mm", "d1_mm", "stratum_geometry", "series", "patient_id",
                                 "stem", "file", "host_lookup_all", "host_class_nearest", "band"])
def test_assert_blind_catches_a_forbidden_key_anywhere(key):
    with pytest.raises(BlindingError):
        assert_blind({"lesion": {"lesion_id": 1}, "extra": [{"ok": 1}, {key: 5}]})
    assert any(f in key for f in FORBIDDEN)


def test_assert_blind_passes_clean_structures_and_returns_them():
    obj = {"lesion": blind_lesion(FULL_LESION), "answer": {"primary_host": "white_matter", "acceptable_hosts": ["white_matter"]},
           "readers": [{"primary_host": "cortex"}], "done": 3, "total": 10, "next": None}
    assert assert_blind(obj) is obj
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_level_r_blind.py -q -p no:cacheprovider`
Expected: FAIL，`ImportError`

- [ ] **Step 3: 写 blind.py**

```python
# anatobind/level_r/blind.py
"""What the browser may receive (spec R5). Whitelists, not blacklists: a new registry column can never leak by
accident. FORBIDDEN is the list of key substrings that must never appear in anything sent out; assert_blind walks a
whole response and is applied to every JSON body by the server, so the whitelists and the blacklist guard each other."""
LESION_KEYS = ("lesion_id", "code", "volume_code", "z0", "z1", "boxes")
VOLUME_KEYS = ("shape", "spacing_slice_mm", "spacing_row_mm", "spacing_col_mm", "window")
FORBIDDEN = ("label", "d_interface", "delta_d", "d1_", "stratum", "series", "patient", "stem", "file", "host_lookup",
             "host_class", "band", "synthseg", "lookup")


class BlindingError(ValueError):
    pass


def _check_key(k):
    low = str(k).lower()
    for f in FORBIDDEN:
        if f in low:
            raise BlindingError(f"key {k!r} matches forbidden substring {f!r}")


def _pick(d, keys):
    out = {k: d[k] for k in keys if k in d}
    for k in out:
        _check_key(k)
    return out


def blind_lesion(d):
    return _pick(d, LESION_KEYS)


def blind_volume_meta(d):
    return _pick(d, VOLUME_KEYS)


def assert_blind(obj):
    """Raise BlindingError if any dict key (at any depth) matches FORBIDDEN; return obj unchanged otherwise."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_key(k)
            assert_blind(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            assert_blind(v)
    return obj
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_level_r_blind.py -q -p no:cacheprovider`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add anatobind/level_r/blind.py tests/test_level_r_blind.py
git commit -m "Level R: field whitelist and recursive blinding check for everything sent to the browser"
```

---

### Task 3: 导出（卷 u16 + 逐层框，与注册表核对）

**Files:**
- Create: `anatobind/level_r/export.py`
- Create: `scripts/level_r_export.py`
- Test: `tests/test_level_r_export.py`

**Interfaces:**
- Consumes: `anatobind.data_engine.fastmri.{MIN_BOX_SIDE, merge_boxes_3d, rows_to_rss_frame, read_fastmri_plus_rows}`、`anatobind.data_engine.fastmri_knee.volume_geometry(h5_path) -> {patient_id, n_rows, n_cols, slices, spacing_slice_mm, spacing_row_mm, spacing_col_mm}`、Task 1 的 `volume_code / lesion_code / load_registry`。
- Produces: `SMALL_LABELS`、`small_lesion_rows(rows)`、`merged_lesions(rows_of_file, n_rows)`、`boxes_by_slice(members) -> {str(slice): [[row0, row1, col0, col1], ...]}`、`match_registry(registry_rows, lesions) -> {lesion_id: lesion}`（按 `(label, z0, z1, x0, y0, x1, y1)` 一一对应，否则 `ValueError`）、`to_u16(rss) -> (np.ndarray('<u2'), {"scale_p999": float, "window": [lo, hi]})`、`write_volume(out_dir, code, u16, geometry, window) -> meta`、`read_volume(out_dir, code) -> (u16, meta)`、`lesion_record(registry_row, lesion) -> dict`、`export_all(registry, csv_rows, out, h5_of, log=print) -> list[dict]`（写 `out/volumes/<code>.{u16,json}` 与 `out/lesions.json`）。
- 文件格式：`<code>.u16` 小端 uint16，形状 `slices × rows × cols` C 序；`<code>.json` = `{"shape": [S, R, C], "spacing_slice_mm", "spacing_row_mm", "spacing_col_mm", "window": [lo, hi]}`；`lesions.json` = `[{"lesion_id", "code", "volume_code", "z0", "z1", "boxes"}]`，按 `lesion_id` 升序。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_level_r_export.py
import json

import h5py
import numpy as np
import pytest

from anatobind.level_r.blind import assert_blind
from anatobind.level_r.export import (
    boxes_by_slice, export_all, match_registry, merged_lesions, read_volume, small_lesion_rows, to_u16,
)
from anatobind.level_r.registry import lesion_code, volume_code

HEADER = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding>'
    "<encodedSpace><matrixSize><x>64</x><y>40</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>28</x><y>17.5</y><z>4.5</z></fieldOfView_mm></encodedSpace>"
    "<reconSpace><matrixSize><x>32</x><y>32</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>14</x><y>14</y><z>3</z></fieldOfView_mm></reconSpace>"
    "</encoding></ismrmrdHeader>"
)
STEM = "file_brain_AXFLAIR_200_1"


def _h5(path, patient="P1", slices=4, size=32, seed=0):
    rss = np.random.default_rng(seed).uniform(1e-5, 4e-4, (slices, size, size)).astype(np.float32)
    with h5py.File(path, "w") as h:
        h.create_dataset("reconstruction_rss", data=rss)
        h.create_dataset("ismrmrd_header", data=np.bytes_(HEADER.encode()))
        h.attrs["patient_id"] = patient
        h.attrs["acquisition"] = "AXFLAIR"
    return path, rss


def _csv_row(s, x, y, w, h, label="Lacunar infarct", file=STEM):
    return {"file": file, "slice": s, "x": x, "y": y, "width": w, "height": h, "label": label}


# CSV frame y counts from the bottom of a 32-row image: y=6, h=4 -> RSS rows [32-6-4, 32-6) = [22, 26)
CSV_ROWS = [_csv_row(1, 10, 6, 6, 4), _csv_row(2, 10, 6, 6, 4)]
REGISTRY = [{"lesion_id": 7, "file": STEM, "patient_id": "P1", "label": "Lacunar infarct",
             "z0": 1, "z1": 2, "x0": 10, "y0": 22, "x1": 16, "y1": 26}]


def test_small_lesion_filter_matches_gate05():
    rows = [_csv_row(1, 0, 0, 5, 5), _csv_row(1, 0, 0, 2, 5), _csv_row(1, 0, 0, 5, 5, label="Mass"),
            _csv_row(1, 0, 0, 5, 5, file="file_brain_AXT1_200_1"), _csv_row(1, 0, 0, 5, 5, label="Nonspecific white matter lesion")]
    assert small_lesion_rows(rows) == [rows[0], rows[4]]


def test_merge_and_boxes_by_slice_are_in_the_rss_frame():
    (L,) = merged_lesions(CSV_ROWS, n_rows=32)
    assert (L["z0"], L["z1"], L["x0"], L["y0"], L["x1"], L["y1"]) == (1, 2, 10, 22, 16, 26)
    assert boxes_by_slice(L["members"]) == {"1": [[22, 26, 10, 16]], "2": [[22, 26, 10, 16]]}


def test_match_registry_is_one_to_one_on_label_and_box():
    lesions = merged_lesions(CSV_ROWS, 32)
    assert match_registry(REGISTRY, lesions)[7] is lesions[0]
    with pytest.raises(ValueError):
        match_registry([{**REGISTRY[0], "x0": 11}], lesions)                       # box differs
    with pytest.raises(ValueError):
        match_registry([{**REGISTRY[0], "label": "Nonspecific white matter lesion"}], lesions)   # label differs


def test_to_u16_maps_p999_to_full_scale_and_reports_the_window():
    v = np.linspace(0.0, 1.0, 1001, dtype=np.float32).reshape(1, 7, 143)
    u, info = to_u16(v)
    assert u.dtype == np.dtype("<u2") and u.shape == v.shape
    assert info["scale_p999"] == pytest.approx(0.999) and u.max() == 65535 and u.min() == 0
    assert u.ravel()[500] == round(0.5 / 0.999 * 65535)
    lo, hi = info["window"]
    assert 0 <= lo < hi <= 65535 and lo == round(0.01 / 0.999 * 65535) and hi == round(0.995 / 0.999 * 65535)


def test_export_all_writes_blind_volumes_and_lesions_that_match_the_registry(tmp_path):
    h5, rss = _h5(tmp_path / f"{STEM}.h5")
    out = tmp_path / "level_r"
    recs = export_all(REGISTRY, CSV_ROWS + [_csv_row(1, 0, 0, 5, 5, label="Mass")], out, h5_of=lambda stem: h5, log=lambda *a: None)
    code = volume_code(STEM)
    assert recs == [{"lesion_id": 7, "code": lesion_code(7), "volume_code": code, "z0": 1, "z1": 2,
                     "boxes": {"1": [[22, 26, 10, 16]], "2": [[22, 26, 10, 16]]}}]
    assert json.loads((out / "lesions.json").read_text()) == recs
    u, meta = read_volume(out / "volumes", code)
    assert u.shape == (4, 32, 32) and u.dtype == np.dtype("<u2") and u.max() == 65535
    assert meta["shape"] == [4, 32, 32] and meta["spacing_slice_mm"] == 3.0 and meta["spacing_row_mm"] == pytest.approx(0.4375)
    assert len(meta["window"]) == 2 and (out / "volumes" / f"{code}.u16").stat().st_size == 4 * 32 * 32 * 2
    text = (out / "lesions.json").read_text() + (out / "volumes" / f"{code}.json").read_text()
    for leak in ("Lacunar", "label", "P1", "patient", STEM, "stratum", "series"):
        assert leak not in text
    assert_blind(recs)
    assert_blind(meta)


def test_export_all_refuses_a_registry_that_does_not_match_the_csv(tmp_path):
    h5, _ = _h5(tmp_path / f"{STEM}.h5")
    with pytest.raises(ValueError):
        export_all([{**REGISTRY[0], "z1": 3}], CSV_ROWS, tmp_path / "out", h5_of=lambda stem: h5, log=lambda *a: None)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_level_r_export.py -q -p no:cacheprovider`
Expected: FAIL，`ImportError`

- [ ] **Step 3: 写 export.py**

```python
# anatobind/level_r/export.py
"""Level R export (spec §4): per volume a little-endian uint16 array + JSON geometry, per lesion its per-slice boxes.
Boxes are recomputed along Gate 0.5's exact path (read_fastmri_plus_rows -> rows_to_rss_frame -> merge_boxes_3d) and
asserted equal to the registry so lesion_id lines up. Nothing written here carries a label string, a distance, a
stratum, a series or a patient id: those stay in the registry, which readers never see."""
import json
from pathlib import Path

import h5py
import numpy as np

from anatobind.data_engine.fastmri import MIN_BOX_SIDE, merge_boxes_3d, rows_to_rss_frame
from anatobind.data_engine.fastmri_knee import volume_geometry
from anatobind.level_r.registry import lesion_code, volume_code

SMALL_LABELS = ("Nonspecific white matter lesion", "Lacunar infarct")     # as scripts/brain_frame.py
U16_MAX = 65535
P_SCALE = 99.9
WINDOW_PERCENTILES = (1.0, 99.5)


def small_lesion_rows(rows):
    """The Gate 0.5 filter on fastMRI+ rows: FLAIR, the two small-lesion labels, both sides >= MIN_BOX_SIDE."""
    return [r for r in rows if "AXFLAIR" in r["file"] and r["label"] in SMALL_LABELS
            and r["width"] >= MIN_BOX_SIDE and r["height"] >= MIN_BOX_SIDE]


def merged_lesions(rows_of_file, n_rows):
    return merge_boxes_3d(rows_to_rss_frame(rows_of_file, n_rows), "label")


def boxes_by_slice(members):
    """{slice: [[row0, row1, col0, col1], ...]} in the RSS frame (rows from the top, half-open)."""
    out = {}
    for m in members:
        out.setdefault(str(m["slice"]), []).append([m["y"], m["y"] + m["height"], m["x"], m["x"] + m["width"]])
    return out


def _key(d):
    return (d["label"], d["z0"], d["z1"], d["x0"], d["y0"], d["x1"], d["y1"])


def match_registry(registry_rows, lesions):
    """Registry rows of one file <-> merged lesions of that file, one-to-one on (label, z0, z1, x0, y0, x1, y1)."""
    by_key = {}
    for L in lesions:
        by_key.setdefault(_key(L), []).append(L)
    out = {}
    for r in registry_rows:
        hits = by_key.get(_key(r), [])
        if len(hits) != 1:
            raise ValueError(f"{r['file']} lesion {r['lesion_id']}: {len(hits)} merged lesions match {_key(r)}; "
                             f"registry and CSV disagree, refusing to export")
        out[r["lesion_id"]] = hits[0]
    return out


def to_u16(rss):
    """float RSS volume -> (uint16 volume, info): p99.9 -> 65535, clipped; default window [p1, p99.5] in u16 units."""
    v = np.asarray(rss, np.float32)
    scale = float(np.percentile(v, P_SCALE))
    if not scale > 0:
        raise ValueError("volume has no positive intensities")
    u = np.clip(np.rint(v / scale * U16_MAX), 0, U16_MAX).astype("<u2")
    lo, hi = (int(np.clip(np.rint(np.percentile(v, q) / scale * U16_MAX), 0, U16_MAX)) for q in WINDOW_PERCENTILES)
    return u, {"scale_p999": scale, "window": [lo, hi]}


def write_volume(out_dir, code, u16, geometry, window):
    """volumes/<code>.u16 (little-endian, slices x rows x cols) and volumes/<code>.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{code}.u16").write_bytes(np.ascontiguousarray(u16, dtype="<u2").tobytes())
    meta = {"shape": [int(s) for s in u16.shape], "spacing_slice_mm": geometry["spacing_slice_mm"],
            "spacing_row_mm": geometry["spacing_row_mm"], "spacing_col_mm": geometry["spacing_col_mm"], "window": list(window)}
    (out_dir / f"{code}.json").write_text(json.dumps(meta))
    return meta


def read_volume(out_dir, code):
    meta = json.loads((Path(out_dir) / f"{code}.json").read_text())
    u = np.frombuffer((Path(out_dir) / f"{code}.u16").read_bytes(), dtype="<u2").reshape(meta["shape"])
    return u, meta


def lesion_record(registry_row, lesion):
    return {"lesion_id": registry_row["lesion_id"], "code": lesion_code(registry_row["lesion_id"]),
            "volume_code": volume_code(registry_row["file"]), "z0": lesion["z0"], "z1": lesion["z1"],
            "boxes": boxes_by_slice(lesion["members"])}


def export_all(registry, csv_rows, out, h5_of, log=print):
    """registry: load_registry rows; csv_rows: read_fastmri_plus_rows(brain.csv); h5_of: stem -> h5 path."""
    out = Path(out)
    by_file_reg, by_file_csv = {}, {}
    for r in registry:
        by_file_reg.setdefault(r["file"], []).append(r)
    for r in small_lesion_rows(csv_rows):
        by_file_csv.setdefault(r["file"], []).append(r)
    records = []
    for i, f in enumerate(sorted(by_file_reg)):
        path = h5_of(f)
        g = volume_geometry(path)
        with h5py.File(path) as h:
            rss = h["reconstruction_rss"][()]
        matched = match_registry(by_file_reg[f], merged_lesions(by_file_csv.get(f, []), g["n_rows"]))
        u16, info = to_u16(rss)
        write_volume(out / "volumes", volume_code(f), u16, g, info["window"])
        records += [lesion_record(r, matched[r["lesion_id"]]) for r in by_file_reg[f]]
        log(f"{i + 1}/{len(by_file_reg)} {volume_code(f)}: {len(by_file_reg[f])} lesions, shape {u16.shape}")
    records.sort(key=lambda r: r["lesion_id"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "lesions.json").write_text(json.dumps(records))
    return records
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_level_r_export.py -q -p no:cacheprovider`
Expected: 全部 PASS

- [ ] **Step 5: 写导出脚本**

```python
#!/usr/bin/env python
# scripts/level_r_export.py
"""Level R export (spec §4): the Gate 0.5 registry's 165 FLAIR volumes as uint16 arrays + geometry, and every
lesion's per-slice boxes recomputed along the Gate 0.5 path and asserted equal to the registry.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/level_r_export.py \
      --out /data2/congcong/data/FM_data/derived/level_r
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import read_fastmri_plus_rows  # noqa: E402
from anatobind.level_r.export import export_all  # noqa: E402
from anatobind.level_r.registry import REGISTRY, load_registry  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
CSV = FM / "fastMRI_lh_brain_knee/Annotations/brain.csv"
KROOT = FM / "fastMRI_lh_brain_knee/kspace/brain"


def h5_of(stem):
    for split in ("multicoil_train", "multicoil_val"):
        p = KROOT / split / f"{stem}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--limit", type=int, default=0, help="only the first N volumes (smoke test)")
    a = ap.parse_args()
    if (a.out / "lesions.json").exists():
        sys.exit(f"{a.out / 'lesions.json'} exists; this export never overwrites (choose another --out)")
    registry = load_registry(a.registry)
    if a.limit:
        keep = sorted({r["file"] for r in registry})[:a.limit]
        registry = [r for r in registry if r["file"] in keep]
    recs = export_all(registry, read_fastmri_plus_rows(CSV), a.out, h5_of)
    print(f"exported {len(recs)} lesions from {len({r['volume_code'] for r in recs})} volumes to {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 语法与帮助检查**

Run: `python scripts/level_r_export.py --help`
Expected: 打印参数说明，退出码 0

- [ ] **Step 7: 提交**

```bash
git add anatobind/level_r/export.py scripts/level_r_export.py tests/test_level_r_export.py
git commit -m "Level R: export uint16 volumes and per-slice lesion boxes checked against the Gate 0.5 registry"
```

---

### Task 4: sqlite 存储（只追加）

**Files:**
- Create: `anatobind/level_r/store.py`
- Test: `tests/test_level_r_store.py`

**Interfaces:**
- Consumes: Task 1 的 `validate_label / validate_adjudication / MAX_ACCEPTABLE`。
- Produces: `token_hash(token) -> str`、`now_iso() -> str`、`needs_adjudication(a, b) -> bool`、`class Store(path)`：
  - `add_reader(reader_id, role, token, display)`；`reader_for_token(token) -> {reader_id, role, display} | None`；`readers() -> list[dict]`
  - `load_lesions(records)`（`lesions.json` 的记录，`INSERT OR IGNORE`）；`lesion(lesion_id) -> dict | None`（含 `boxes`）；`lesion_ids() -> list[int]`
  - `set_order(reader_id, lesion_ids, pilot_ids)`（一位读者只能设一次）；`order(reader_id) -> [{position, lesion_id, is_pilot}]`
  - `submit_label(reader_id, lesion_id, payload) -> row_id`；`label_rows(reader_id=None) -> list[dict]`（全部历史行，列表字段已解析）；`latest_labels(reader_id=None) -> list[dict]`（每 (reader, lesion) 最后一行）；`progress(reader_id) -> {done, total, next}`
  - `submit_adjudication(adjudicator_id, lesion_id, payload) -> row_id`；`adjudication_rows() -> list[dict]`；`latest_adjudications() -> list[dict]`
  - `disagreements() -> (sorted lesion_ids, [reader_id_1, reader_id_2])`（恰好两位 role=reader，否则 `ValueError`）；`final_labels() -> [{lesion_id, status ∈ agreed|adjudicated|pending, primary_host, acceptable_hosts, not_a_lesion}]`
- 解析后的标签 dict 键：`row_id, reader_id, lesion_id, primary_host, acceptable_hosts(list), topography, adjacency(list), ambiguity, not_a_lesion(bool), local_quality, confidence, comment, time_seconds, window(list|None), ts`；裁定 dict 多 `adjudicator_id, reason`，无 `local_quality/confidence/time_seconds/window`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_level_r_store.py
import sqlite3
from pathlib import Path

import pytest

import anatobind.level_r.store as store_module
from anatobind.level_r.schema import InvalidLabel
from anatobind.level_r.store import Store, needs_adjudication, token_hash

T1, T2, TA = "0123456789abcdef", "fedcba9876543210", "aaaaaaaaaaaaaaaa"
LESIONS = [{"lesion_id": i, "code": f"c{i:07d}", "volume_code": "v0000000", "z0": 1, "z1": 1, "boxes": {"1": [[2, 6, 3, 9]]}} for i in range(5)]
WM = {"primary_host": "white_matter", "acceptable_hosts": ["white_matter"], "topography": "deep_white_matter", "adjacency": ["none"],
      "ambiguity": "certain", "not_a_lesion": False, "local_quality": "good", "confidence": 5, "comment": ""}
WM_CX = {**WM, "acceptable_hosts": ["white_matter", "cortex"], "ambiguity": "two_host"}
CX = {**WM, "primary_host": "cortex", "acceptable_hosts": ["cortex"]}
CX_TH = {**CX, "acceptable_hosts": ["cortex", "thalamus"]}
NAL = {"not_a_lesion": True, "local_quality": "fair", "confidence": 3, "adjacency": []}


def _store(tmp_path):
    s = Store(tmp_path / "level_r.sqlite")
    s.add_reader("r1", "reader", T1, "读者 1")
    s.add_reader("r2", "reader", T2, "读者 2")
    s.add_reader("adj", "adjudicator", TA, "裁定")
    s.load_lesions(LESIONS)
    return s


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
    assert s.progress("r1") == {"done": 0, "total": 5, "next": 3}
    r1 = s.submit_label("r1", 3, {**WM, "time_seconds": 41.5, "window": [100, 900]})
    r2 = s.submit_label("r1", 3, {**CX, "time_seconds": 12.0})
    assert r2 > r1
    rows = s.label_rows("r1")
    assert [r["primary_host"] for r in rows] == ["white_matter", "cortex"] and rows[0]["window"] == [100, 900] and rows[1]["window"] is None
    (latest,) = s.latest_labels("r1")
    assert latest["primary_host"] == "cortex" and latest["acceptable_hosts"] == ["cortex"] and latest["adjacency"] == ["none"]
    assert latest["not_a_lesion"] is False and latest["time_seconds"] == 12.0 and latest["ts"]
    assert s.progress("r1") == {"done": 1, "total": 5, "next": 1}


def test_submit_rejects_invalid_or_unknown_and_stores_nothing(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(InvalidLabel):
        s.submit_label("r1", 0, {**WM, "confidence": 9})
    with pytest.raises(KeyError):
        s.submit_label("r1", 99, WM)
    assert s.label_rows() == []


@pytest.mark.parametrize("a, b, expected", [
    (WM, WM, False), (WM, WM_CX, False),                     # same host, union {wm, cortex} <= 2
    (WM, CX, True),                                          # different primary host
    (WM, NAL, True), (NAL, NAL, False),                      # one says not a lesion; both do
    (WM_CX, CX_TH, True),                                    # union {wm, cortex, thalamus} > 2 even though... hosts differ too
    ({**WM_CX, "primary_host": "cortex"}, CX_TH, True),      # same primary host, union of three
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
    assert final[0] == {"lesion_id": 0, "status": "agreed", "primary_host": "white_matter", "acceptable_hosts": ["cortex", "white_matter"], "not_a_lesion": False}
    assert final[3] == {"lesion_id": 3, "status": "agreed", "primary_host": None, "acceptable_hosts": [], "not_a_lesion": True}
    assert final[1]["status"] == "pending" and final[2]["status"] == "pending"
    with pytest.raises(InvalidLabel):
        s.submit_adjudication("adj", 1, {**CX, "reason": ""})
    s.submit_adjudication("adj", 1, {"primary_host": "cortex", "acceptable_hosts": ["cortex"], "topography": "cortical",
                                     "adjacency": [], "ambiguity": "certain", "reason": "皮层内信号"})
    final = {f["lesion_id"]: f for f in s.final_labels()}
    assert final[1] == {"lesion_id": 1, "status": "adjudicated", "primary_host": "cortex", "acceptable_hosts": ["cortex"], "not_a_lesion": False}
    (adj,) = s.latest_adjudications()
    assert adj["adjudicator_id"] == "adj" and adj["reason"] == "皮层内信号" and adj["lesion_id"] == 1


def test_disagreements_need_exactly_two_readers(tmp_path):
    s = Store(tmp_path / "x.sqlite")
    s.add_reader("r1", "reader", T1, "读者 1")
    with pytest.raises(ValueError):
        s.disagreements()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_level_r_store.py -q -p no:cacheprovider`
Expected: FAIL，`ImportError`

- [ ] **Step 3: 写 store.py**（注意：文件里任何地方都不能出现 UPDATE / DELETE 两个词）

```python
# anatobind/level_r/store.py
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_level_r_store.py -q -p no:cacheprovider`
Expected: 全部 PASS（含源码 grep 那条）

- [ ] **Step 5: 提交**

```bash
git add anatobind/level_r/store.py tests/test_level_r_store.py
git commit -m "Level R: append-only sqlite store with readers, orders, labels, adjudications and the R9 final-label rule"
```

---

### Task 5: HTTP 服务（API + 静态页）

**Files:**
- Create: `anatobind/level_r/server.py`
- Create: `scripts/level_r_server.py`
- Test: `tests/test_level_r_server.py`

**Interfaces:**
- Consumes: Task 4 的 `Store`；Task 2 的 `assert_blind / blind_lesion / blind_volume_meta`；Task 1 的 `enums / InvalidLabel`。
- Produces: `APP_DIR = Path(__file__).parent / "app"`、`make_handler(store, data_root, app_dir=APP_DIR) -> type[BaseHTTPRequestHandler]`、`make_server(store, data_root, bind="127.0.0.1", port=8790, app_dir=APP_DIR) -> ThreadingHTTPServer`。
- 路由（除静态页与 `/api/enums` 外都要 `?token=<16 hex>`，无效 → 403）：

| 方法 路径 | 角色 | 返回 |
|---|---|---|
| GET `/`、`/index.html`、`/app.js`、`/style.css`、`/guide.html` | 任何人 | 静态文件 |
| GET `/api/enums` | 任何人 | `schema.enums()` |
| GET `/api/me` | reader | `{reader_id, role, display, done, total, next}`；adjudicator：`{reader_id, role, display, disagreements}` |
| GET `/api/list` | reader | `[{position, lesion_id, is_pilot, done}]` 只有本人顺序 |
| GET `/api/lesion/<id>` | reader（须在本人顺序里，否则 403）/ adjudicator | `{lesion: blind_lesion, answer: 本人最后一次答案 | null}` |
| GET `/api/volume/<code>.json` | 已登录 | `blind_volume_meta` |
| GET `/api/volume/<code>.u16` | 已登录 | `application/octet-stream` 原始字节 |
| POST `/api/label` | reader | body `{lesion_id, ...表单, time_seconds, window}` → `{row_id}`；校验失败 400 `{error}` |
| GET `/api/disagreements` | adjudicator | `[{lesion_id, done}]` |
| GET `/api/adjudicate/<id>` | adjudicator | `{lesion, readers: [答案视图 ×2（匿名，按 reader_id 排序）], answer}` |
| POST `/api/adjudication` | adjudicator | body `{lesion_id, ...表单, reason}` → `{row_id}` |

  读者不足两位时 `/api/me`（adjudicator）、`/api/disagreements` 返回 409 `{error}`。所有 JSON 响应经 `assert_blind`；`Cache-Control: no-store`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_level_r_server.py
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from anatobind.level_r.export import write_volume
from anatobind.level_r.server import make_server
from anatobind.level_r.store import Store

T1, T2, TA, BAD = "0123456789abcdef", "fedcba9876543210", "aaaaaaaaaaaaaaaa", "ffffffffffffffff"
CODE = "0f0f0f0f"
LESIONS = [{"lesion_id": i, "code": f"c{i:07d}", "volume_code": CODE, "z0": 1, "z1": 2, "boxes": {"1": [[2, 6, 3, 9]], "2": [[2, 6, 3, 9]]}} for i in range(4)]
WM = {"primary_host": "white_matter", "acceptable_hosts": ["white_matter"], "topography": "deep_white_matter", "adjacency": ["none"],
      "ambiguity": "certain", "not_a_lesion": False, "local_quality": "good", "confidence": 5, "comment": "", "time_seconds": 30.0, "window": [10, 900]}
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


def test_static_pages_and_enums_need_no_token(served):
    _, base = served
    status, body, headers = get(base, "/", raw=True)
    assert status == 200 and b"Level R" in body and headers["Content-Type"].startswith("text/html")
    assert get(base, "/app.js", raw=True)[2]["Content-Type"].startswith(("text/javascript", "application/javascript"))
    status, en, _ = get(base, "/api/enums")
    assert status == 200 and en["max_acceptable"] == 2 and "white_matter" in en["primary_hosts"]


def test_invalid_or_missing_token_is_403_everywhere(served):
    _, base = served
    for path in ("/api/me", "/api/list", "/api/lesion/0", f"/api/volume/{CODE}.json", "/api/disagreements"):
        assert get(base, path, BAD)[0] == 403 and get(base, path)[0] == 403
    assert post(base, "/api/label", BAD, {"lesion_id": 0, **WM})[0] == 403


def test_reader_sees_only_their_own_order_and_answers(served):
    store, base = served
    status, me, _ = get(base, "/api/me", T1)
    assert status == 200 and me == {"reader_id": "r1", "role": "reader", "display": "读者 1", "done": 0, "total": 3, "next": 2}
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
    assert get(base, "/api/me", T1)[1]["done"] == 1 and get(base, "/api/me", T1)[1]["next"] == 0
    assert get(base, "/api/lesion/2", T1)[1]["answer"]["primary_host"] == "cortex"
    assert post(base, "/api/label", TA, {"lesion_id": 2, **WM})[0] == 403    # adjudicator cannot label


def test_adjudicator_sees_disagreements_anonymously_and_can_rule(served):
    store, base = served
    for lid, (a, b) in {0: (WM, WM), 1: (WM, CX), 2: (CX, WM)}.items():
        store.submit_label("r1", lid, a)
        store.submit_label("r2", lid, b)
    assert get(base, "/api/disagreements", T1)[0] == 403
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
                                                       "topography": "cortical", "adjacency": [], "ambiguity": "certain", "reason": "皮层内"})
    assert status == 200 and out["row_id"] == 1
    assert get(base, "/api/disagreements", TA)[1][0] == {"lesion_id": 1, "done": True}
    assert get(base, "/api/adjudicate/1", TA)[1]["answer"]["reason"] == "皮层内"
    assert get(base, "/api/lesion/1", TA)[0] == 200                     # adjudicator may view any lesion


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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_level_r_server.py -q -p no:cacheprovider`
Expected: FAIL，`ImportError`

- [ ] **Step 3: 写 server.py**

```python
# anatobind/level_r/server.py
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_level_r_server.py -q -p no:cacheprovider`
Expected: 全部 PASS。若 `app.js` 的 Content-Type 断言失败，检查 `mimetypes.add_type` 是否在 import 时执行。

- [ ] **Step 5: 写起服务脚本**

```python
#!/usr/bin/env python
# scripts/level_r_server.py
"""Level R service (spec §10). Binds 127.0.0.1:8790 by default; pass --bind 0.0.0.0 explicitly for LAN access.

  D=/data2/congcong/data/FM_data/derived/level_r
  setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/level_r_server.py \
      --db $D/level_r.sqlite --data-root $D --pid-file $D/server.pid > $D/server.log 2>&1 &
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.level_r.server import make_server  # noqa: E402
from anatobind.level_r.store import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True, help="directory holding volumes/ from level_r_export.py")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--pid-file", type=Path)
    a = ap.parse_args()
    if not (a.data_root / "volumes").is_dir():
        sys.exit(f"{a.data_root}/volumes is missing; run scripts/level_r_export.py first")
    srv = make_server(Store(a.db), a.data_root, a.bind, a.port)
    if a.pid_file:
        a.pid_file.write_text(str(os.getpid()))
    print(f"Level R serving {a.data_root} on http://{a.bind}:{srv.server_address[1]}/ (db {a.db})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 帮助检查**

Run: `python scripts/level_r_server.py --help`
Expected: 参数说明，退出码 0

- [ ] **Step 7: 提交**

```bash
git add anatobind/level_r/server.py scripts/level_r_server.py tests/test_level_r_server.py
git commit -m "Level R: token-scoped HTTP API with blinded lesion, volume and adjudication routes"
```

---

### Task 6: 前端（读者页、裁定页、读片说明）

**Files:**
- Create: `anatobind/level_r/app/index.html`、`anatobind/level_r/app/app.js`、`anatobind/level_r/app/style.css`、`anatobind/level_r/app/guide.html`
- Test: `tests/test_level_r_app_static.py`

**Interfaces:**
- Consumes: Task 5 的全部路由；`/api/enums` 决定下拉内容，前端不写死枚举列表，只写中文标签表 `ZH`（JSON 字面量，测试解析它并与 `schema` 比对）。
- 行为（spec §5–§6）：入口 `/?token=…`；启动读 `/api/enums`、`/api/me`，然后打开下一例；canvas 画当前层（16 位数组按窗宽窗位映射为灰度，`imageSmoothingEnabled=false`），框绿色；↑↓/←→ 或滑条翻整卷；1×/2×/4× 放大以病灶框中心为中心；拖动横向改窗宽、纵向改窗位，"复位窗"回默认；右侧两张小图为上一层/下一层（同窗、画框）。表单校验与 `schema.validate_label` 同规则；提交带 `time_seconds`（页面打开到提交）与 `window`。列表页可回看并重提。裁定人身份自动切换：列表为分歧病灶，病灶页多一张两位读者答案表（匿名，读者 1/读者 2）与必填"裁定理由"，隐藏局部质量与信心。
- 浏览器测试不进 CI：静态测试守文件存在、引用关系、`ZH` 覆盖全部枚举键、用到的路由与服务端一致、前端源码不含禁词。真正的交互验收在 Task 12 的冒烟部署里由人（用户）在浏览器点一遍。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_level_r_app_static.py
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_level_r_app_static.py -q -p no:cacheprovider`
Expected: FAIL（`FileNotFoundError` / 断言失败）

- [ ] **Step 3: 写 index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Level R 读片</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header>
  <span id="who">…</span> · <span id="progress">…</span> · <span id="lcode"></span>
  <span class="right"><a href="#" id="btn-list">列表</a> · <a href="#" id="btn-next">下一个</a> · <a href="guide.html" target="_blank">读片说明</a></span>
  <div id="status"></div>
</header>

<section id="sec-list" class="hidden">
  <ul id="list"></ul>
</section>

<section id="sec-lesion" class="hidden">
  <div class="viewer">
    <canvas id="view" width="640" height="640"></canvas>
    <canvas id="off" class="hidden"></canvas>
    <div class="controls">
      <span id="zlabel"></span>
      <input type="range" id="zslider" min="0" max="15" value="0">
      <button type="button" id="zoom1">1×</button><button type="button" id="zoom2">2×</button><button type="button" id="zoom4">4×</button>
      <span id="wlabel"></span> <button type="button" id="wreset">复位窗</button>
      <div class="hint">↑↓ 或 ←→ 翻层；在图上拖动：横向改窗宽、纵向改窗位；绿框是待判的病灶</div>
    </div>
    <div class="thumbs">
      <div>上一层<br><canvas id="prev" width="200" height="200"></canvas></div>
      <div>下一层<br><canvas id="next" width="200" height="200"></canvas></div>
    </div>
  </div>

  <form id="form" onsubmit="return false">
    <div id="readers" class="adj-only"></div>
    <label class="row"><input type="checkbox" id="not_a_lesion"> 不是病灶（框内没有病灶）</label>
    <label>主宿主结构 <select id="primary_host"></select></label>
    <fieldset><legend>可接受的宿主（最多 2 个，必须包含主宿主）</legend><div id="acceptable_hosts"></div></fieldset>
    <label>拓扑位置 <select id="topography"></select></label>
    <fieldset><legend>邻接（可多选）</legend><div id="adjacency"></div></fieldset>
    <label>不确定性 <select id="ambiguity"></select></label>
    <label class="reader-only">局部图像质量（病灶附近能不能看清） <select id="local_quality"></select></label>
    <fieldset class="reader-only"><legend>信心（1 低 – 5 高）</legend><div id="confidence"></div></fieldset>
    <label>备注 <textarea id="comment" rows="2"></textarea></label>
    <label class="adj-only">裁定理由（必填） <textarea id="reason" rows="2"></textarea></label>
    <button type="button" id="submit">提交并下一个</button>
  </form>
</section>

<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 4: 写 style.css**

```css
/* anatobind/level_r/app/style.css */
body { margin: 0; font: 15px/1.5 -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; background: #111; color: #ddd; }
header { padding: 8px 16px; background: #1c1c1c; border-bottom: 1px solid #333; }
header .right { float: right; }
header a { color: #8ab4f8; text-decoration: none; }
#status { color: #ffb74d; min-height: 1.4em; }
.hidden { display: none !important; }
body.adjudicator .reader-only { display: none; }
body:not(.adjudicator) .adj-only { display: none; }
#sec-lesion { display: flex; gap: 24px; padding: 12px 16px; align-items: flex-start; flex-wrap: wrap; }
.viewer { display: grid; grid-template-columns: auto 220px; gap: 12px; }
#view { background: #000; border: 1px solid #444; cursor: crosshair; max-width: 70vh; max-height: 70vh; grid-row: 1 / span 2; }
.controls { grid-column: 1; }
.controls input[type=range] { width: 260px; vertical-align: middle; }
.controls button, form button { margin: 0 2px; padding: 4px 10px; background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; cursor: pointer; }
.hint { color: #888; font-size: 13px; }
.thumbs { grid-column: 2; grid-row: 1; display: flex; flex-direction: column; gap: 8px; font-size: 13px; color: #aaa; }
.thumbs canvas { background: #000; border: 1px solid #444; display: block; }
form { min-width: 340px; max-width: 460px; display: flex; flex-direction: column; gap: 8px; }
form label { display: flex; flex-direction: column; gap: 2px; }
form label.row { flex-direction: row; align-items: center; gap: 6px; }
form select, form textarea { background: #222; color: #eee; border: 1px solid #555; padding: 4px; }
fieldset { border: 1px solid #444; padding: 6px 10px; }
fieldset label { display: inline-flex; flex-direction: row; align-items: center; gap: 4px; margin-right: 12px; }
#submit { padding: 8px; font-size: 16px; background: #2e7d32; }
#readers table { border-collapse: collapse; margin-bottom: 8px; }
#readers td, #readers th { border: 1px solid #444; padding: 2px 8px; }
#list { list-style: none; padding: 12px 16px; columns: 3; }
#list li.done a { color: #7cb342; }
#list a { color: #8ab4f8; text-decoration: none; }
```

- [ ] **Step 5: 写 app.js**

```javascript
// anatobind/level_r/app/app.js — Level R reader/adjudicator page (spec §5–§6). Vanilla JS, no build step.
"use strict";

const TOKEN = new URLSearchParams(location.search).get("token") || "";
const ZH = {"white_matter": "白质", "cortex": "皮层", "thalamus": "丘脑", "basal_ganglia": "基底节", "brainstem": "脑干", "cerebellum": "小脑", "other": "其他（含脑室内）",
  "periventricular": "脑室旁", "juxtacortical": "近皮层", "cortical": "皮层内", "deep_white_matter": "深部白质", "infratentorial": "幕下",
  "adjacent_to_cortex": "邻近皮层", "adjacent_to_ventricle": "邻近脑室", "crosses_boundary": "跨越边界", "none": "无",
  "certain": "确定", "two_host": "两个宿主难分", "multi_structure": "多结构", "insufficient_resolution": "分辨率不足",
  "good": "好", "fair": "一般", "poor": "差", "not_a_lesion": "不是病灶"};
const FIELD_ZH = {"primary_host": "主宿主", "acceptable_hosts": "可接受集合", "topography": "拓扑位置", "adjacency": "邻接",
  "ambiguity": "不确定性", "not_a_lesion": "不是病灶", "comment": "备注"};

const state = {me: null, enums: null, mode: "reader", lesion: null, vol: null, z: 0, zoom: 2, win: null, opened: 0};
const volCache = new Map();
const $ = (id) => document.getElementById(id);

// ---- API --------------------------------------------------------------------------------------------------------
async function api(path, opts) {
  const sep = path.includes("?") ? "&" : "?";
  const r = await fetch(path + sep + "token=" + encodeURIComponent(TOKEN), opts);
  if (r.status === 403) { $("status").textContent = "令牌无效或无权限，请检查链接"; throw new Error("403"); }
  return r;
}
async function apiJSON(path, opts) {
  const r = await api(path, opts);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || String(r.status));
  return j;
}
async function loadVolume(code) {
  if (volCache.has(code)) return volCache.get(code);
  const meta = await apiJSON(`/api/volume/${code}.json`);
  const buf = await (await api(`/api/volume/${code}.u16`)).arrayBuffer();
  const v = {meta, data: new Uint16Array(buf)};
  volCache.set(code, v);
  return v;
}

// ---- rendering --------------------------------------------------------------------------------------------------
function sliceImage(v, z, lo, hi) {
  const [S, R, C] = v.meta.shape, img = new ImageData(C, R), d = img.data, base = z * R * C, span = Math.max(1, hi - lo);
  for (let i = 0; i < R * C; i++) {
    let g = (v.data[base + i] - lo) * 255 / span;
    g = g < 0 ? 0 : g > 255 ? 255 : g;
    const o = i * 4; d[o] = d[o + 1] = d[o + 2] = g; d[o + 3] = 255;
  }
  return img;
}
function boxesOn(z) { return state.lesion.boxes[String(z)] || []; }
function lesionCenter() {
  const bs = boxesOn(state.z).length ? boxesOn(state.z) : Object.values(state.lesion.boxes).flat();
  let r = 0, c = 0;
  for (const [r0, r1, c0, c1] of bs) { r += (r0 + r1) / 2; c += (c0 + c1) / 2; }
  return [r / bs.length, c / bs.length];
}
function draw() {
  if (!state.vol) return;
  const v = state.vol, [S, R, C] = v.meta.shape, z = state.z, [lo, hi] = state.win;
  const off = $("off"); off.width = C; off.height = R;
  off.getContext("2d").putImageData(sliceImage(v, z, lo, hi), 0, 0);
  const cv = $("view"), ctx = cv.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  const k = state.zoom, sw = C / k, sh = R / k, [cy, cx] = lesionCenter();
  const ox = k === 1 ? 0 : Math.min(Math.max(0, cx - sw / 2), C - sw), oy = k === 1 ? 0 : Math.min(Math.max(0, cy - sh / 2), R - sh);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(off, ox, oy, sw, sh, 0, 0, cv.width, cv.height);
  const fx = cv.width / sw, fy = cv.height / sh;
  ctx.strokeStyle = "#00e676"; ctx.lineWidth = 2;
  for (const [r0, r1, c0, c1] of boxesOn(z)) ctx.strokeRect((c0 - ox) * fx, (r0 - oy) * fy, (c1 - c0) * fx, (r1 - r0) * fy);
  $("zlabel").textContent = `层 ${z + 1} / ${S}` + (z >= state.lesion.z0 && z <= state.lesion.z1 ? "（病灶层）" : "");
  $("wlabel").textContent = `窗 [${lo}, ${hi}]`;
  $("zslider").value = z;
  thumb("prev", z - 1); thumb("next", z + 1);
}
function thumb(id, z) {
  const cv = $(id), ctx = cv.getContext("2d"), [S, R, C] = state.vol.meta.shape;
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (z < 0 || z >= S) return;
  const off = document.createElement("canvas"); off.width = C; off.height = R;
  off.getContext("2d").putImageData(sliceImage(state.vol, z, state.win[0], state.win[1]), 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, cv.width, cv.height);
  ctx.strokeStyle = "#00e676"; ctx.lineWidth = 1;
  const fx = cv.width / C, fy = cv.height / R;
  for (const [r0, r1, c0, c1] of boxesOn(z)) ctx.strokeRect(c0 * fx, r0 * fy, (c1 - c0) * fx, (r1 - r0) * fy);
}
function setZ(z) { const S = state.vol.meta.shape[0]; state.z = Math.min(Math.max(0, z), S - 1); draw(); }

// ---- viewer controls ---------------------------------------------------------------------------------------------
window.addEventListener("keydown", (e) => {
  if (!state.vol || e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowUp" || e.key === "ArrowRight") { e.preventDefault(); setZ(state.z + 1); }
  else if (e.key === "ArrowDown" || e.key === "ArrowLeft") { e.preventDefault(); setZ(state.z - 1); }
});
$("zslider").oninput = (e) => setZ(+e.target.value);
for (const k of [1, 2, 4]) $("zoom" + k).onclick = () => { state.zoom = k; draw(); };
$("wreset").onclick = () => { state.win = [...state.vol.meta.window]; draw(); };
let drag = null;
$("view").addEventListener("mousedown", (e) => { drag = {x: e.clientX, y: e.clientY, win: [...state.win]}; });
window.addEventListener("mousemove", (e) => {
  if (!drag) return;
  const [lo, hi] = drag.win, width = hi - lo, level = (hi + lo) / 2;
  const nw = Math.max(16, width + (e.clientX - drag.x) * width / 200), nl = level - (e.clientY - drag.y) * width / 200;
  state.win = [Math.round(nl - nw / 2), Math.round(nl + nw / 2)];
  draw();
});
window.addEventListener("mouseup", () => { drag = null; });

// ---- form -------------------------------------------------------------------------------------------------------
function fill(id, keys) { $(id).innerHTML = '<option value="">请选择</option>' + keys.map(k => `<option value="${k}">${ZH[k] || k}</option>`).join(""); }
function checkboxes(id, keys) { $(id).innerHTML = keys.map(k => `<label><input type="checkbox" name="${id}" value="${k}"> ${ZH[k] || k}</label>`).join(""); }
function radios(id, vals) { $(id).innerHTML = vals.map(v => `<label><input type="radio" name="${id}" value="${v}"> ${v}</label>`).join(""); }
function checked(name) { return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map(i => i.value); }
function setChecked(name, vals) { document.querySelectorAll(`input[name="${name}"]`).forEach(i => { i.checked = vals.includes(i.value); }); }
function buildForm(en) {
  fill("primary_host", en.primary_hosts); checkboxes("acceptable_hosts", en.primary_hosts);
  fill("topography", en.topography); checkboxes("adjacency", en.adjacency); fill("ambiguity", en.ambiguity);
  fill("local_quality", en.local_quality); radios("confidence", [1, 2, 3, 4, 5]);
}
function toggleNal() {
  const nal = $("not_a_lesion").checked;
  for (const id of ["primary_host", "topography", "ambiguity"]) $(id).disabled = nal;
  document.querySelectorAll('input[name="acceptable_hosts"]').forEach(i => { i.disabled = nal; });
}
$("not_a_lesion").onchange = toggleNal;
$("primary_host").onchange = () => {
  const v = $("primary_host").value;
  document.querySelectorAll('input[name="acceptable_hosts"]').forEach(i => { if (i.value === v) i.checked = true; });
};
function resetForm() { $("form").reset(); toggleNal(); $("readers").innerHTML = ""; }
function fillForm(a) {
  $("not_a_lesion").checked = !!a.not_a_lesion; toggleNal();
  $("primary_host").value = a.primary_host || ""; setChecked("acceptable_hosts", a.acceptable_hosts || []);
  $("topography").value = a.topography || ""; setChecked("adjacency", a.adjacency || []); $("ambiguity").value = a.ambiguity || "";
  $("local_quality").value = a.local_quality || ""; setChecked("confidence", a.confidence != null ? [String(a.confidence)] : []);
  $("comment").value = a.comment || ""; $("reason").value = a.reason || "";
}
function readForm() {
  const nal = $("not_a_lesion").checked, adj = state.mode === "adjudicator";
  const p = {lesion_id: state.lesion.lesion_id, not_a_lesion: nal,
    primary_host: nal ? null : ($("primary_host").value || null), acceptable_hosts: nal ? [] : checked("acceptable_hosts"),
    topography: nal ? null : ($("topography").value || null), adjacency: checked("adjacency"), ambiguity: nal ? null : ($("ambiguity").value || null),
    comment: $("comment").value, time_seconds: (Date.now() - state.opened) / 1000, window: state.win};
  if (adj) p.reason = $("reason").value; else { p.local_quality = $("local_quality").value || null; p.confidence = parseInt(checked("confidence")[0] || "0", 10); }
  const err = validate(p, adj);
  if (err) throw new Error(err);
  return p;
}
function validate(p, adj) {
  const M = state.enums.max_acceptable;
  if (!p.not_a_lesion) {
    if (!p.primary_host) return "请选择主宿主结构";
    if (!p.acceptable_hosts.includes(p.primary_host)) return "可接受集合必须包含主宿主";
    if (!p.topography) return "请选择拓扑位置";
    if (!p.ambiguity) return "请选择不确定性";
  }
  if (p.acceptable_hosts.length > M) return `可接受集合最多 ${M} 个`;
  if (p.adjacency.includes("none") && p.adjacency.length > 1) return "邻接选了“无”就不能再选其他";
  if (adj) { if (!p.reason.trim()) return "请填写裁定理由"; return null; }
  if (!p.local_quality) return "请选择局部图像质量";
  if (!(p.confidence >= 1 && p.confidence <= 5)) return "请选择信心 1–5";
  return null;
}
function fmt(v) {
  if (Array.isArray(v)) return v.length ? v.map(x => ZH[x] || x).join("、") : "—";
  if (v === true) return "是"; if (v === false || v == null || v === "") return "—";
  return ZH[v] || v;
}
function showReaders(views) {
  const keys = ["primary_host", "acceptable_hosts", "topography", "adjacency", "ambiguity", "not_a_lesion", "comment"];
  $("readers").innerHTML = "<table><tr><th></th><th>读者 1</th><th>读者 2</th></tr>" +
    keys.map(k => `<tr><td>${FIELD_ZH[k]}</td>${views.map(v => `<td>${fmt(v[k])}</td>`).join("")}</tr>`).join("") + "</table>";
}

// ---- navigation -------------------------------------------------------------------------------------------------
function show(name) { $("sec-list").classList.toggle("hidden", name !== "list"); $("sec-lesion").classList.toggle("hidden", name !== "lesion"); }
async function refreshMe() {
  const me = await apiJSON("/api/me");
  state.me = me; state.mode = me.role;
  document.body.classList.toggle("adjudicator", me.role === "adjudicator");
  $("who").textContent = me.display;
  $("progress").textContent = me.role === "reader" ? `已完成 ${me.done} / ${me.total}` : `待裁定 ${me.disagreements}`;
  return me;
}
async function openLesion(lid) {
  const j = await apiJSON(state.mode === "adjudicator" ? `/api/adjudicate/${lid}` : `/api/lesion/${lid}`);
  state.lesion = j.lesion;
  state.vol = await loadVolume(j.lesion.volume_code);
  state.win = [...state.vol.meta.window];
  state.z = Math.floor((j.lesion.z0 + j.lesion.z1) / 2);
  state.zoom = 2; state.opened = Date.now();
  const [S, R, C] = state.vol.meta.shape, cv = $("view");
  cv.width = C * 2; cv.height = R * 2; $("zslider").max = S - 1;
  $("lcode").textContent = `病灶 ${j.lesion.code}`;
  resetForm();
  if (j.answer) fillForm(j.answer);
  if (state.mode === "adjudicator") showReaders(j.readers);
  $("status").textContent = "";
  show("lesion"); draw();
}
async function showList() {
  const rows = await apiJSON(state.mode === "adjudicator" ? "/api/disagreements" : "/api/list");
  $("list").innerHTML = rows.map(r => `<li class="${r.done ? "done" : ""}"><a href="#" data-lid="${r.lesion_id}">` +
    (state.mode === "adjudicator" ? `病灶 #${r.lesion_id}` : `第 ${r.position + 1} 例${r.is_pilot ? "（pilot）" : ""}`) + (r.done ? " ✓" : "") + "</a></li>").join("");
  $("list").querySelectorAll("a").forEach(a => { a.onclick = (e) => { e.preventDefault(); openLesion(+a.dataset.lid); }; });
  show("list");
}
async function openNext() {
  const me = await refreshMe();
  if (state.mode === "adjudicator") {
    const rows = await apiJSON("/api/disagreements"), n = rows.find(r => !r.done);
    return n ? openLesion(n.lesion_id) : showList();
  }
  return me.next != null ? openLesion(me.next) : showList();
}
$("btn-list").onclick = (e) => { e.preventDefault(); showList(); };
$("btn-next").onclick = (e) => { e.preventDefault(); openNext(); };
$("submit").onclick = async () => {
  try {
    const p = readForm();
    await apiJSON(state.mode === "adjudicator" ? "/api/adjudication" : "/api/label",
      {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(p)});
    $("status").textContent = "已保存";
    await openNext();
  } catch (e) { $("status").textContent = "未保存：" + e.message; }
};

async function main() {
  state.enums = await apiJSON("/api/enums");
  buildForm(state.enums);
  await openNext();
}
main().catch(e => { $("status").textContent = "加载失败：" + e.message; });
```

- [ ] **Step 6: 写 guide.html（读片说明，含 primary_host 的操作定义）**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Level R 读片说明</title><link rel="stylesheet" href="style.css">
<style>main { max-width: 820px; margin: 0 auto; padding: 16px; } h2 { color: #8ab4f8; } code { background: #222; padding: 0 4px; }</style></head>
<body><main>
<h1>Level R 读片说明</h1>
<p>这项工作要回答的是：<b>每一个标出的脑内小病灶，长在哪块解剖结构上。</b>您看到的是 3T FLAIR 轴位图，绿框是待判的病灶（由公开数据集的标注者标出）。两位医生各自独立读，互不可见对方答案；第三位医生只裁分歧。工具里不显示任何自动分区或模型结果。</p>

<h2>怎么看图</h2>
<ul>
<li>↑↓ 或 ←→ 翻层，滑条也可以；病灶所在层在标题里标出"（病灶层）"。请上下多看几层再判。</li>
<li>1× / 2× / 4× 放大，以病灶框为中心；默认 2×。</li>
<li>在图上按住拖动：横向改窗宽、纵向改窗位；"复位窗"回默认。右侧小图是上一层、下一层。</li>
</ul>

<h2>各字段怎么填</h2>
<h3>主宿主结构（primary_host）</h3>
<p><b>定义：结合整个可见 MRI 体积，您判断该病灶最主要、最合理的解剖宿主组织。</b>它<b>不是框内重叠最大的结构</b>，也不是离框最近的结构，而是您作为放射科医生对"这个病灶长在哪"的判断。可选：白质、皮层、丘脑、基底节、脑干、小脑、其他（含脑室内病灶）。脑室与脑脊液不算宿主，脑室内病灶选"其他"，并在拓扑位置选"脑室旁"、邻接选"邻近脑室"。</p>
<h3>可接受的宿主（acceptable_hosts）</h3>
<p>最多 2 个，必须包含主宿主。5 mm 层厚下贴着边界的病灶常常分不清白质还是皮层：这时主宿主填最可能的一个，另一个合理的宿主加进可接受集合。例：主宿主 = 白质，拓扑 = 近皮层，可接受 = {白质, 皮层}。不要把"近皮层"当成宿主。</p>
<h3>拓扑位置（topography）</h3>
<p>脑室旁、近皮层、皮层内、深部白质、幕下。描述病灶在哪一带，和宿主是两个独立的问题。</p>
<h3>邻接（adjacency）</h3>
<p>可多选：邻近皮层、邻近脑室、跨越边界；都不是就选"无"（选了"无"不能再选其他）。</p>
<h3>不确定性（ambiguity）</h3>
<p>确定 / 两个宿主难分 / 多结构 / 分辨率不足。"两个宿主难分"时可接受集合通常应有 2 个。</p>
<h3>不是病灶</h3>
<p>框里没有您认可的病灶（标错、血管、伪影）时勾选，宿主可以不填。这类框会单独统计，不进主终点。</p>
<h3>局部图像质量、信心</h3>
<p>质量指病灶附近能否看清结构边界（好 / 一般 / 差）；信心 1（低）到 5（高）指您对主宿主判断的把握。</p>
<h3>备注</h3>
<p>自由填写。任何让您犹豫的地方都值得写一句。</p>

<h2>流程</h2>
<p>先读 150 例 pilot（顺序里已排在最前），我们据此看两位医生的一致率与每例用时；之后在同一链接里继续读完全部。已提交的病灶可以在"列表"里回看和修改，修改会作为新答案保存，统计取最后一次。每例用时会自动记录，不必赶。</p>
</main></body></html>
```

- [ ] **Step 7: 跑静态测试确认通过**

Run: `python -m pytest tests/test_level_r_app_static.py tests/test_level_r_server.py -q -p no:cacheprovider`
Expected: 全部 PASS

- [ ] **Step 8: 本机冒烟（无浏览器）**

用 Task 5 测试里的方式起一个临时服务，确认 `/`、`/app.js`、`/guide.html` 都返回 200 且 JS 无语法错误（node 若不在机器上则跳过语法检查，写明）：

```bash
which node && node --check anatobind/level_r/app/app.js && echo JS_SYNTAX_OK || echo "node absent: syntax check skipped"
```

- [ ] **Step 9: 提交**

```bash
git add anatobind/level_r/app tests/test_level_r_app_static.py
git commit -m "Level R: browser reader and adjudicator page with 16-bit windowing, slice navigation and the reading guide"
```

---

### Task 7: 管理操作（患者折、读者顺序、CSV 导出、封存、备份）+ 生成 `data/level_r/folds.json`

**Files:**
- Create: `anatobind/level_r/admin.py`
- Create: `scripts/level_r_admin.py`
- Create: `data/level_r/folds.json`（由脚本生成后入库）
- Test: `tests/test_level_r_admin.py`

**Interfaces:**
- Consumes: `anatobind.data_engine.fastmri_knee.assert_folds_by_patient(folds: {key: fold}, patients: {key: patient_id})`；Task 4 的 `Store`；Task 1 的 `load_registry`。
- Produces: `make_patient_folds(patient_ids, k=5, seed=0) -> {patient_id: fold}`、`lesion_folds(registry, patient_fold) -> {lesion_id: fold}`（断言患者不跨折）、`make_order(lesion_ids, pilot_ids, seed) -> list[int]`（pilot 打乱排最前，其余打乱排后）、`sha256_file(path) -> str`、`export_csvs(store, out_dir) -> list[Path]`（`labels_<reader>.csv` 全部历史行、`adjudications.csv`、`final_labels.csv`）、`seal(final_labels_csv, lesion_fold, out_dir, manifest_path, now=None) -> manifest`（一次性，任何目标已存在或有 pending 就拒绝）、`backup_db(db_path, out_dir, now=None) -> Path`（sqlite 在线备份，文件名带时间戳，永不覆盖）。
- `folds.json` 格式：`{"k": 5, "seed": 0, "n_patients": 165, "patient_fold": {patient_id: fold}, "lesions_per_fold": {"0": n, ...}}`。
- `sealed_manifest.json` 格式：`{"fold0": {"path", "sha256", "rows", "sealed_at"}, ...}`。
- CSV 列：`LABEL_COLUMNS = (row_id, reader_id, lesion_id, primary_host, acceptable_hosts, topography, adjacency, ambiguity, not_a_lesion, local_quality, confidence, comment, time_seconds, window, ts)`（列表字段写 JSON 字符串）；`ADJ_COLUMNS = (row_id, adjudicator_id, lesion_id, primary_host, acceptable_hosts, topography, adjacency, ambiguity, not_a_lesion, reason, ts)`；`FINAL_COLUMNS = (lesion_id, status, primary_host, acceptable_hosts, not_a_lesion)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_level_r_admin.py
import csv
import json
from collections import Counter

import pytest

from anatobind.level_r.admin import (
    FINAL_COLUMNS, backup_db, export_csvs, lesion_folds, make_order, make_patient_folds, seal, sha256_file,
)
from anatobind.level_r.store import Store

WM = {"primary_host": "white_matter", "acceptable_hosts": ["white_matter"], "topography": "deep_white_matter", "adjacency": ["none"],
      "ambiguity": "certain", "not_a_lesion": False, "local_quality": "good", "confidence": 5, "comment": "a,b"}
CX = {**WM, "primary_host": "cortex", "acceptable_hosts": ["cortex"]}


def test_patient_folds_are_balanced_seeded_and_patient_disjoint():
    pats = [f"p{i:03d}" for i in range(165)]
    f = make_patient_folds(pats, k=5, seed=0)
    assert set(f) == set(pats) and Counter(f.values()) == {0: 33, 1: 33, 2: 33, 3: 33, 4: 33}
    assert f == make_patient_folds(list(reversed(pats)), k=5, seed=0) and f != make_patient_folds(pats, k=5, seed=1)
    registry = [{"lesion_id": i, "patient_id": pats[i % 165]} for i in range(400)]
    lf = lesion_folds(registry, f)
    assert len(lf) == 400 and all(lf[i] == f[pats[i % 165]] for i in range(400))


def test_lesion_folds_calls_the_patient_disjoint_assertion(monkeypatch):
    """The cross-fold check itself is tested in tests/test_fastmri_knee_manifest.py; here we only require that
    lesion_folds routes through it with string keys."""
    import anatobind.level_r.admin as admin
    calls = []
    monkeypatch.setattr(admin, "assert_folds_by_patient", lambda folds, patients: calls.append((folds, patients)))
    lesion_folds([{"lesion_id": 0, "patient_id": "a"}, {"lesion_id": 1, "patient_id": "b"}], {"a": 0, "b": 1})
    assert calls == [({"0": 0, "1": 1}, {"0": "a", "1": "b"})]


def test_make_order_puts_pilot_first_and_is_a_seeded_permutation():
    ids = list(range(20))
    o = make_order(ids, pilot_ids={3, 7, 11}, seed=1)
    assert sorted(o) == ids and set(o[:3]) == {3, 7, 11} and o == make_order(ids, {3, 7, 11}, 1) and o != make_order(ids, {3, 7, 11}, 2)


def _filled_store(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.add_reader("r1", "reader", "0123456789abcdef", "读者 1")
    s.add_reader("r2", "reader", "fedcba9876543210", "读者 2")
    s.add_reader("adj", "adjudicator", "aaaaaaaaaaaaaaaa", "裁定")
    s.load_lesions([{"lesion_id": i, "code": f"c{i}", "volume_code": "v", "z0": 0, "z1": 0, "boxes": {}} for i in range(4)])
    for lid, (a, b) in {0: (WM, WM), 1: (WM, CX), 2: (CX, CX), 3: (WM, CX)}.items():
        s.submit_label("r1", lid, {**a, "time_seconds": 10.0, "window": [1, 2]})
        s.submit_label("r2", lid, b)
    s.submit_label("r1", 0, CX)                                       # history: r1 changed lesion 0 -> now a disagreement
    s.submit_adjudication("adj", 1, {**CX, "reason": "皮层"})
    return s


def test_export_csvs_writes_history_adjudications_and_final_status(tmp_path):
    s = _filled_store(tmp_path)
    files = export_csvs(s, tmp_path / "export")
    assert sorted(p.name for p in files) == ["adjudications.csv", "final_labels.csv", "labels_r1.csv", "labels_r2.csv"]
    r1 = list(csv.DictReader(open(tmp_path / "export/labels_r1.csv", newline="")))
    assert len(r1) == 5 and r1[0]["comment"] == "a,b" and json.loads(r1[0]["acceptable_hosts"]) == ["white_matter"] and r1[0]["window"] == "[1, 2]"
    final = {int(r["lesion_id"]): r for r in csv.DictReader(open(tmp_path / "export/final_labels.csv", newline=""))}
    assert final[0]["status"] == "pending" and final[1]["status"] == "adjudicated" and final[2]["status"] == "agreed" and final[3]["status"] == "pending"
    assert list(final[2]) == list(FINAL_COLUMNS) and final[2]["primary_host"] == "cortex"
    adj = list(csv.DictReader(open(tmp_path / "export/adjudications.csv", newline="")))
    assert len(adj) == 1 and adj[0]["reason"] == "皮层"


def test_export_without_two_readers_still_writes_reader_files(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.add_reader("r1", "reader", "0123456789abcdef", "读者 1")
    files = export_csvs(s, tmp_path / "export")
    assert sorted(p.name for p in files) == ["adjudications.csv", "final_labels.csv", "labels_r1.csv"]
    assert open(tmp_path / "export/final_labels.csv").read().strip() == ",".join(FINAL_COLUMNS)


def _final_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_seal_splits_by_fold_writes_sha256_and_is_one_shot(tmp_path):
    rows = [{"lesion_id": i, "status": "agreed", "primary_host": "white_matter", "acceptable_hosts": '["white_matter"]', "not_a_lesion": False} for i in range(10)]
    final = _final_csv(tmp_path / "final.csv", rows)
    lesion_fold = {i: i % 5 for i in range(10)}
    man = seal(final, lesion_fold, tmp_path / "sealed", tmp_path / "manifest.json", now="2026-10-01T00:00:00+00:00")
    assert set(man) == {f"fold{k}" for k in range(5)} and all(m["rows"] == 2 and m["sealed_at"] == "2026-10-01T00:00:00+00:00" for m in man.values())
    for k in range(5):
        p = tmp_path / "sealed" / f"labels_fold{k}.csv"
        assert man[f"fold{k}"]["sha256"] == sha256_file(p) and man[f"fold{k}"]["path"] == str(p)
        assert [int(r["lesion_id"]) for r in csv.DictReader(open(p, newline=""))] == [k, k + 5]
    assert json.loads((tmp_path / "manifest.json").read_text()) == man
    with pytest.raises(FileExistsError):
        seal(final, lesion_fold, tmp_path / "sealed", tmp_path / "manifest2.json")
    with pytest.raises(FileExistsError):
        seal(final, lesion_fold, tmp_path / "sealed2", tmp_path / "manifest.json")


def test_seal_refuses_pending_rows(tmp_path):
    final = _final_csv(tmp_path / "final.csv", [{"lesion_id": 0, "status": "pending", "primary_host": "", "acceptable_hosts": "[]", "not_a_lesion": ""}])
    with pytest.raises(ValueError):
        seal(final, {0: 0}, tmp_path / "sealed", tmp_path / "manifest.json")


def test_backup_copies_the_live_database_and_never_overwrites(tmp_path):
    s = _filled_store(tmp_path)
    p = backup_db(tmp_path / "db.sqlite", tmp_path / "backup", now="2026-10-01T08:00:00+00:00")
    assert p.name == "level_r_2026-10-01T08-00-00.sqlite" and Store(p).label_rows() == s.label_rows()
    with pytest.raises(FileExistsError):
        backup_db(tmp_path / "db.sqlite", tmp_path / "backup", now="2026-10-01T08:00:00+00:00")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_level_r_admin.py -q -p no:cacheprovider`
Expected: FAIL，`ImportError`

- [ ] **Step 3: 写 admin.py**

```python
# anatobind/level_r/admin.py
"""Admin operations behind scripts/level_r_admin.py: patient folds (v2.6 §4.4 / §12.6), per-reader orders, CSV export,
one-shot sealing with a sha256 manifest (v2.6 §12.7) and online sqlite backups. Nothing here ever overwrites a file."""
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from anatobind.data_engine.fastmri_knee import assert_folds_by_patient
from anatobind.level_r.store import now_iso

LABEL_COLUMNS = ("row_id", "reader_id", "lesion_id", "primary_host", "acceptable_hosts", "topography", "adjacency", "ambiguity",
                 "not_a_lesion", "local_quality", "confidence", "comment", "time_seconds", "window", "ts")
ADJ_COLUMNS = ("row_id", "adjudicator_id", "lesion_id", "primary_host", "acceptable_hosts", "topography", "adjacency", "ambiguity",
               "not_a_lesion", "reason", "ts")
FINAL_COLUMNS = ("lesion_id", "status", "primary_host", "acceptable_hosts", "not_a_lesion")


def make_patient_folds(patient_ids, k=5, seed=0):
    """Same rule as fastmri_knee.make_folds, keyed by patient: sorted unique ids, seeded permutation, round robin."""
    uniq = sorted(set(patient_ids))
    order = np.random.default_rng(seed).permutation(len(uniq))
    return {uniq[int(j)]: i % k for i, j in enumerate(order)}


def lesion_folds(registry, patient_fold):
    folds = {str(r["lesion_id"]): patient_fold[r["patient_id"]] for r in registry}
    assert_folds_by_patient(folds, {str(r["lesion_id"]): r["patient_id"] for r in registry})
    return {int(k): v for k, v in folds.items()}


def make_order(lesion_ids, pilot_ids, seed):
    rng = np.random.default_rng(seed)
    pilot = [l for l in lesion_ids if l in pilot_ids]
    rest = [l for l in lesion_ids if l not in pilot_ids]
    return [pilot[int(i)] for i in rng.permutation(len(pilot))] + [rest[int(i)] for i in rng.permutation(len(rest))]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, columns, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def export_csvs(store, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for r in store.readers():
        if r["role"] == "reader":
            p = out_dir / f"labels_{r['reader_id']}.csv"
            _write_csv(p, LABEL_COLUMNS, store.label_rows(r["reader_id"]))
            written.append(p)
    _write_csv(out_dir / "adjudications.csv", ADJ_COLUMNS, store.adjudication_rows())
    try:
        final = store.final_labels()
    except ValueError:                       # fewer than two readers yet
        final = []
    _write_csv(out_dir / "final_labels.csv", FINAL_COLUMNS, final)
    return written + [out_dir / "adjudications.csv", out_dir / "final_labels.csv"]


def seal(final_labels_csv, lesion_fold, out_dir, manifest_path, now=None):
    """Split final_labels.csv by outer fold into out_dir/labels_fold{k}.csv and write the sha256 manifest. One shot."""
    with open(final_labels_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    pending = [r for r in rows if r["status"] == "pending"]
    if pending:
        raise ValueError(f"{len(pending)} lesions are pending adjudication; seal after the adjudicator has finished")
    out_dir, manifest_path = Path(out_dir), Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; sealing is one-shot")
    k = max(lesion_fold.values()) + 1
    per = {f: [] for f in range(k)}
    for r in rows:
        per[lesion_fold[int(r["lesion_id"])]].append(r)
    targets = [out_dir / f"labels_fold{f}.csv" for f in range(k)]
    for p in targets:
        if p.exists():
            raise FileExistsError(f"{p} exists; sealing is one-shot, never overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now or now_iso()
    manifest = {}
    for f, p in enumerate(targets):
        _write_csv(p, FINAL_COLUMNS, per[f])
        manifest[f"fold{f}"] = {"path": str(p), "sha256": sha256_file(p), "rows": len(per[f]), "sealed_at": stamp}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest


def backup_db(db_path, out_dir, now=None):
    """Consistent copy of the live sqlite file (sqlite backup API, safe while the server runs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or now_iso()).split("+")[0].replace(":", "-")        # 2026-10-01T08-00-00
    dst = out_dir / f"level_r_{stamp}.sqlite"
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    src, dest = sqlite3.connect(str(db_path)), sqlite3.connect(str(dst))
    with dest:
        src.backup(dest)
    src.close()
    dest.close()
    return dst
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_level_r_admin.py -q -p no:cacheprovider`
Expected: 全部 PASS

- [ ] **Step 5: 写管理脚本**

```python
#!/usr/bin/env python
# scripts/level_r_admin.py
"""Level R administration (spec §3, §7, §10). Subcommands, all one-shot and none overwriting:

  folds       registry -> data/level_r/folds.json (165 patients, seed 0, five folds)
  init        create the sqlite database and load lesions.json into it
  add-reader  create a reader or adjudicator; the token is printed exactly once
  order       give one reader their randomised order (pilot first)
  export      labels_<reader>.csv / adjudications.csv / final_labels.csv
  seal        final_labels.csv -> sealed/labels_fold{k}.csv + data/level_r/sealed_manifest.json
  backup      timestamped copy of the live database

  D=/data2/congcong/data/FM_data/derived/level_r
  python scripts/level_r_admin.py folds --out data/level_r/folds.json
  python scripts/level_r_admin.py init --db $D/level_r.sqlite --lesions $D/lesions.json
  python scripts/level_r_admin.py add-reader --db $D/level_r.sqlite --reader-id r1 --role reader --display "读者 1"
  python scripts/level_r_admin.py order --db $D/level_r.sqlite --reader-id r1 --seed 1 --pilot data/level_r/pilot_150.json
  python scripts/level_r_admin.py export --db $D/level_r.sqlite --out $D/export
  python scripts/level_r_admin.py seal --final $D/export/final_labels.csv --folds data/level_r/folds.json --out $D/sealed
  python scripts/level_r_admin.py backup --db $D/level_r.sqlite --out $D/backup
"""
import argparse
import json
import secrets
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.level_r.admin import backup_db, export_csvs, lesion_folds, make_order, make_patient_folds, seal  # noqa: E402
from anatobind.level_r.registry import REGISTRY, load_registry  # noqa: E402
from anatobind.level_r.store import Store  # noqa: E402

MANIFEST = Path("data/level_r/sealed_manifest.json")


def cmd_folds(a):
    if a.out.exists():
        sys.exit(f"{a.out} exists; folds are generated once")
    registry = load_registry(a.registry)
    pf = make_patient_folds([r["patient_id"] for r in registry], k=a.k, seed=a.seed)
    lf = lesion_folds(registry, pf)
    out = {"k": a.k, "seed": a.seed, "n_patients": len(pf), "patient_fold": pf,
           "lesions_per_fold": {str(k): v for k, v in sorted(Counter(lf.values()).items())}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"{len(pf)} patients, lesions per fold {out['lesions_per_fold']} -> {a.out}")


def cmd_init(a):
    store = Store(a.db)
    records = json.loads(a.lesions.read_text())
    store.load_lesions(records)
    print(f"{a.db}: {len(store.lesion_ids())} lesions loaded from {a.lesions}")


def cmd_add_reader(a):
    token = secrets.token_hex(8)
    Store(a.db).add_reader(a.reader_id, a.role, token, a.display)
    print(f"reader {a.reader_id} ({a.role}, {a.display}) created. Link (shown once, not stored):\n  /?token={token}")


def cmd_order(a):
    store = Store(a.db)
    pilot = set(json.loads(a.pilot.read_text())["lesion_ids"]) if a.pilot else set()
    order = make_order(store.lesion_ids(), pilot, a.seed)
    store.set_order(a.reader_id, order, pilot)
    print(f"{a.reader_id}: {len(order)} lesions, {len(pilot)} pilot first, seed {a.seed}")


def cmd_export(a):
    for p in export_csvs(Store(a.db), a.out):
        print(p)


def cmd_seal(a):
    folds = json.loads(a.folds.read_text())
    registry = load_registry(a.registry)
    lf = lesion_folds(registry, folds["patient_fold"])
    man = seal(a.final, lf, a.out, a.manifest)
    for k, m in man.items():
        print(f"{k}: {m['rows']} rows sha256 {m['sha256'][:12]}… -> {m['path']}")
    print(f"manifest -> {a.manifest}")


def cmd_backup(a):
    print(backup_db(a.db, a.out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("folds"); p.add_argument("--registry", type=Path, default=REGISTRY); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=5); p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_folds)
    p = sub.add_parser("init"); p.add_argument("--db", type=Path, required=True); p.add_argument("--lesions", type=Path, required=True); p.set_defaults(fn=cmd_init)
    p = sub.add_parser("add-reader"); p.add_argument("--db", type=Path, required=True); p.add_argument("--reader-id", required=True)
    p.add_argument("--role", choices=("reader", "adjudicator"), required=True); p.add_argument("--display", required=True); p.set_defaults(fn=cmd_add_reader)
    p = sub.add_parser("order"); p.add_argument("--db", type=Path, required=True); p.add_argument("--reader-id", required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--pilot", type=Path); p.set_defaults(fn=cmd_order)
    p = sub.add_parser("export"); p.add_argument("--db", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.set_defaults(fn=cmd_export)
    p = sub.add_parser("seal"); p.add_argument("--final", type=Path, required=True); p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--registry", type=Path, default=REGISTRY); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST); p.set_defaults(fn=cmd_seal)
    p = sub.add_parser("backup"); p.add_argument("--db", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.set_defaults(fn=cmd_backup)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 生成并核对 folds.json**

Run:
```bash
python scripts/level_r_admin.py folds --out data/level_r/folds.json
python - <<'EOF'
import json; f = json.load(open("data/level_r/folds.json")); from collections import Counter
print(f["n_patients"], Counter(f["patient_fold"].values()), f["lesions_per_fold"])
EOF
```
Expected: `165 Counter({0: 33, 1: 33, 2: 33, 3: 33, 4: 33}) {...}`，五折病灶数之和 1297。把这两行输出原样贴进 Step 8 的提交说明。

- [ ] **Step 7: 脚本冒烟（临时目录）**

```bash
T=$(mktemp -d)
python scripts/level_r_admin.py init --db $T/db.sqlite --lesions <(echo '[{"lesion_id":0,"code":"c0","volume_code":"v0","z0":0,"z1":0,"boxes":{}}]')
python scripts/level_r_admin.py add-reader --db $T/db.sqlite --reader-id r1 --role reader --display "读者 1"
python scripts/level_r_admin.py order --db $T/db.sqlite --reader-id r1 --seed 1
python scripts/level_r_admin.py export --db $T/db.sqlite --out $T/export && ls $T/export
python scripts/level_r_admin.py backup --db $T/db.sqlite --out $T/backup
```
Expected: 每步打印一行，无 traceback（`init` 用进程替换不行就先写个临时 json 文件）。

- [ ] **Step 8: 提交**

```bash
git add anatobind/level_r/admin.py scripts/level_r_admin.py tests/test_level_r_admin.py data/level_r/folds.json
git commit -m "Level R: admin operations for patient folds, reader orders, CSV export, one-shot sealing and backups; commit the 165-patient fold table"
```

---

### Task 8: pilot 150 抽样 + 生成 `data/level_r/pilot_150.json`

**Files:**
- Create: `anatobind/level_r/pilot.py`
- Create: `scripts/level_r_pilot_sample.py`
- Create: `data/level_r/pilot_150.json`（脚本生成后入库）
- Test: `tests/test_level_r_pilot.py`

**Interfaces:**
- Consumes: Task 1 的 `load_registry`（行含 `band`、`stratum_geometry`、`patient_id`、`lesion_id`）。
- Produces: `cell_of(row) -> "band|stratum_geometry"`、`allocate(cell_sizes: {cell: n}, n, min_per_cell) -> {cell: alloc}`、`sample_pilot(registry, n=150, min_per_cell=8, max_per_patient=3, seed=0) -> {"n", "seed", "min_per_cell", "max_per_patient", "lesion_ids": sorted list, "cells": {cell: {"n", "alloc", "picked"}}}`。
- 规则（spec R8/§9）：不足 `min_per_cell` 的格子全取；比例份额不足 `min_per_cell` 的格子取 `min_per_cell`；其余预算按人口比例、最大余数法分给剩下的格子。抽取时格子按人数从小到大轮到，各格一份种子置换，患者已有 `max_per_patient` 例就跳过；缺口从最大的格子的剩余合格病灶回填。同种子结果相同。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_level_r_pilot.py
from collections import Counter

import pytest

from anatobind.level_r.pilot import allocate, cell_of, sample_pilot

SIZES = {"0|A": 500, "0-2|A": 200, "2-4|A": 200, ">4|A": 300, "0|B": 20, "0-2|B": 3, "2-4|B": 6, ">4|B": 3}


def _registry(sizes, patients_per_cell=None):
    rows, lid = [], 0
    for cell, n in sizes.items():
        band, stratum = cell.split("|")
        k = (patients_per_cell or {}).get(cell, n)                 # default: every lesion its own patient
        for i in range(n):
            rows.append({"lesion_id": lid, "band": band, "stratum_geometry": stratum, "patient_id": f"{cell}-p{i % k}"})
            lid += 1
    return rows


def test_allocate_takes_small_cells_whole_floors_at_min_and_splits_the_rest_by_largest_remainder():
    a = allocate(SIZES, n=150, min_per_cell=8)
    assert a == {"0|A": 54, "0-2|A": 22, "2-4|A": 22, ">4|A": 32, "0|B": 8, "0-2|B": 3, "2-4|B": 6, ">4|B": 3}
    assert sum(a.values()) == 150


def test_allocate_refuses_a_budget_that_cannot_honour_the_floor():
    with pytest.raises(ValueError):
        allocate(SIZES, n=20, min_per_cell=8)


def test_sample_pilot_hits_150_honours_cells_and_is_deterministic():
    reg = _registry(SIZES)
    p = sample_pilot(reg, n=150, min_per_cell=8, max_per_patient=3, seed=0)
    ids = p["lesion_ids"]
    assert len(ids) == 150 and ids == sorted(set(ids))
    by_id = {r["lesion_id"]: r for r in reg}
    counts = Counter(cell_of(by_id[i]) for i in ids)
    assert counts == {"0|A": 54, "0-2|A": 22, "2-4|A": 22, ">4|A": 32, "0|B": 8, "0-2|B": 3, "2-4|B": 6, ">4|B": 3}
    assert p["cells"]["0|B"] == {"n": 20, "alloc": 8, "picked": 8} and p["cells"]["0-2|B"] == {"n": 3, "alloc": 3, "picked": 3}
    assert p == sample_pilot(reg, 150, 8, 3, 0) and p["lesion_ids"] != sample_pilot(reg, 150, 8, 3, 1)["lesion_ids"]


def test_patient_cap_moves_the_shortfall_to_the_largest_cells():
    reg = _registry(SIZES, patients_per_cell={"0|A": 10})          # 500 lesions from 10 patients: at most 30 pickable
    p = sample_pilot(reg, n=150, min_per_cell=8, max_per_patient=3, seed=0)
    by_id = {r["lesion_id"]: r for r in reg}
    per_patient = Counter(by_id[i]["patient_id"] for i in p["lesion_ids"])
    assert len(p["lesion_ids"]) == 150 and max(per_patient.values()) <= 3
    assert p["cells"]["0|A"]["picked"] == 30 and p["cells"]["0|A"]["alloc"] == 54
    assert sum(c["picked"] for c in p["cells"].values()) == 150


def test_sample_pilot_fails_loudly_when_the_cap_makes_150_impossible():
    reg = _registry({"0|A": 300}, patients_per_cell={"0|A": 20})   # 20 patients x 3 = 60 < 150
    with pytest.raises(ValueError):
        sample_pilot(reg, n=150, min_per_cell=8, max_per_patient=3, seed=0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_level_r_pilot.py -q -p no:cacheprovider`
Expected: FAIL，`ImportError`

- [ ] **Step 3: 写 pilot.py**

```python
# anatobind/level_r/pilot.py
"""Pilot sample (spec R8 / §9): n lesions over the cells distance band x measured-geometry stratum, every non-empty
cell represented, no patient dominating. The pilot only estimates agreement and reading time (v2.6 §7.2); the
agreement gate is judged on the full set."""
from collections import Counter, defaultdict

import numpy as np


def cell_of(row):
    return f"{row['band']}|{row['stratum_geometry']}"


def allocate(cell_sizes, n, min_per_cell):
    """{cell: size} -> {cell: allocation}. Cells below min_per_cell are taken whole; cells whose proportional share
    is below min_per_cell get exactly min_per_cell; the remaining budget is split among the other cells in proportion
    to size, largest-remainder rounding."""
    total = sum(cell_sizes.values())
    fixed, flex = {}, {}
    for c, s in cell_sizes.items():
        if s < min_per_cell:
            fixed[c] = s
        elif n * s / total < min_per_cell:
            fixed[c] = min_per_cell
        else:
            flex[c] = s
    budget = n - sum(fixed.values())
    if budget < min_per_cell * len(flex):
        raise ValueError(f"budget {budget} after the fixed cells cannot give {len(flex)} cells {min_per_cell} each")
    if not flex:
        return fixed
    flex_total = sum(flex.values())
    raw = {c: budget * s / flex_total for c, s in flex.items()}
    alloc = {c: int(np.floor(v)) for c, v in raw.items()}
    for c in sorted(flex, key=lambda c: (-(raw[c] - alloc[c]), c))[: budget - sum(alloc.values())]:
        alloc[c] += 1
    for c, a in alloc.items():
        if a < min_per_cell or a > flex[c]:
            raise ValueError(f"cell {c}: allocation {a} outside [{min_per_cell}, {flex[c]}]")
    return {**fixed, **alloc}


def sample_pilot(registry, n=150, min_per_cell=8, max_per_patient=3, seed=0):
    cells = defaultdict(list)
    for r in registry:
        cells[cell_of(r)].append(r)
    alloc = allocate({c: len(v) for c, v in cells.items()}, n, min_per_cell)
    rng = np.random.default_rng(seed)
    per_patient, picked, leftovers = Counter(), [], {}
    for c in sorted(cells, key=lambda c: (len(cells[c]), c)):            # small cells draw first
        rows = sorted(cells[c], key=lambda r: r["lesion_id"])
        got, rest = [], []
        for r in (rows[int(i)] for i in rng.permutation(len(rows))):
            if len(got) < alloc[c] and per_patient[r["patient_id"]] < max_per_patient:
                got.append(r)
                per_patient[r["patient_id"]] += 1
            else:
                rest.append(r)
        picked += got
        leftovers[c] = rest
    short = n - len(picked)
    for c in sorted(cells, key=lambda c: (-len(cells[c]), c)):           # refill from the largest cells
        for r in leftovers[c]:
            if short <= 0:
                break
            if per_patient[r["patient_id"]] < max_per_patient:
                picked.append(r)
                per_patient[r["patient_id"]] += 1
                short -= 1
    if len(picked) != n:
        raise ValueError(f"could only draw {len(picked)} of {n} lesions under the {max_per_patient}-per-patient cap")
    counts = Counter(cell_of(r) for r in picked)
    return {"n": n, "seed": seed, "min_per_cell": min_per_cell, "max_per_patient": max_per_patient,
            "lesion_ids": sorted(r["lesion_id"] for r in picked),
            "cells": {c: {"n": len(cells[c]), "alloc": alloc[c], "picked": counts.get(c, 0)} for c in sorted(cells)}}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_level_r_pilot.py -q -p no:cacheprovider`
Expected: 全部 PASS

- [ ] **Step 5: 写抽样脚本**

```python
#!/usr/bin/env python
# scripts/level_r_pilot_sample.py
"""Draw the Level R pilot (spec R8 / §9) from the Gate 0.5 registry and write data/level_r/pilot_150.json.

  python scripts/level_r_pilot_sample.py --out data/level_r/pilot_150.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.level_r.pilot import sample_pilot  # noqa: E402
from anatobind.level_r.registry import REGISTRY, load_registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-per-cell", type=int, default=8)
    ap.add_argument("--max-per-patient", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; the pilot is drawn once")
    p = sample_pilot(load_registry(a.registry), a.n, a.min_per_cell, a.max_per_patient, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(p, indent=1))
    print(f"{'cell':32s} {'n':>5s} {'alloc':>5s} {'picked':>6s}")
    for c, v in p["cells"].items():
        print(f"{c:32s} {v['n']:5d} {v['alloc']:5d} {v['picked']:6d}")
    print(f"total picked {len(p['lesion_ids'])} (seed {a.seed}) -> {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 在真实注册表上抽样并核对**

Run: `python scripts/level_r_pilot_sample.py --out data/level_r/pilot_150.json`
Expected: 16 行格子表 + `total picked 150`。核对：`0|inplane_0.69_slice_5` 等四个大格子 alloc 之和 = 87；七个不足 8 的格子 `picked == n`；3 mm 层厚的两层（`*_slice_3`）只来自 7 名患者，patient 上限会让它们 picked < alloc，缺口回填到 0.69 mm 大格子——这是预期行为，把表原样贴进 `docs/verification/2026-09-26/level_r_pilot_sample.md`（命令 + 输出）。

再跑一次核对患者上限：
```bash
python - <<'EOF'
import json
from anatobind.level_r.registry import load_registry
p = json.load(open("data/level_r/pilot_150.json")); reg = {r["lesion_id"]: r for r in load_registry()}
from collections import Counter
pp = Counter(reg[i]["patient_id"] for i in p["lesion_ids"]); print("lesions", len(p["lesion_ids"]), "patients", len(pp), "max per patient", max(pp.values()))
EOF
```
Expected: `lesions 150 patients >= 50 max per patient 3`。

- [ ] **Step 7: 提交**

```bash
git add anatobind/level_r/pilot.py scripts/level_r_pilot_sample.py tests/test_level_r_pilot.py data/level_r/pilot_150.json docs/verification/2026-09-26/level_r_pilot_sample.md
git commit -m "Level R: stratified pilot sampler over distance band x geometry stratum with a per-patient cap; commit the 150-lesion pilot list"
```

---

### Task 9: 一致率统计

**Files:**
- Create: `anatobind/eval/level_r_stats.py`
- Test: `tests/test_level_r_stats.py`

**Interfaces:**
- Consumes: Task 4 的 `latest_labels()` 输出（dict 含 `lesion_id, primary_host, acceptable_hosts, not_a_lesion, time_seconds`）；Task 1 的注册表行（`patient_id, band, stratum_geometry`）与 `is_3mm`、`NOT_A_LESION`。
- Produces: `host_of(label) -> str`、`set_of(label) -> set`、`pairs(a, b, lesion_ids=None) -> [{"lesion_id", "x", "y", "sx", "sy"}]`、`raw_agreement(x, y) -> float`、`cohen_kappa(x, y) -> float`、`gwet_ac1(x, y, categories=None) -> float`、`positive_agreement(x, y) -> {class: {"n_x", "n_y", "positive_agreement"}}`、`confusion(x, y) -> {"categories": [...], "counts": [[...]]}`、`set_agreement(P) -> float`、`bootstrap_ci(outcomes_by_patient, n_boot=2000, seed=0, alpha=0.05) -> (lo, hi)`、`strata_report(P, reg_by_id) -> {"band": {...}, "stratum_geometry": {...}, "slice_3mm": {...}}`（每层 `{"n", "raw", "set_agreement"}`）、`time_summary(labels) -> {"n", "median_s", "q1_s", "q3_s", "hours_for_1297"}`、`gate_r7(raw_ci_low, raw_band0) -> dict`、`summarise(a, b, registry, lesion_ids=None, n_boot=2000, seed=0) -> dict`。
- 公式（spec §8）：raw = 主结构相同的比例；κ = (p_o − p_e)/(1 − p_e)，p_e = Σ_c n_x(c)·n_y(c)/N²；AC1 = (p_o − p_e)/(1 − p_e)，p_e = Σ_c π_c(1 − π_c)/(K − 1)，π_c = (n_x(c)+n_y(c))/(2N)，K = 类别数（默认观测到的类别）；positive agreement(c) = 2·n_agree(c)/(n_x(c)+n_y(c))；集合值一致 = 两个可接受集合有交集的比例；`not_a_lesion` 的答案在主结构统计里记为类 `not_a_lesion`，集合为 `{not_a_lesion}`。区间：按患者 bootstrap 2000 次、种子 0、百分位法。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_level_r_stats.py
import math

import pytest

from anatobind.eval.level_r_stats import (
    bootstrap_ci, cohen_kappa, confusion, gate_r7, gwet_ac1, pairs, positive_agreement, raw_agreement, set_agreement,
    strata_report, summarise, time_summary,
)

X = ["A", "A", "B", "B", "C"]
Y = ["A", "B", "B", "B", "C"]


def _lab(lid, host, acc=None, nal=False, t=None):
    return {"lesion_id": lid, "primary_host": None if nal else host, "acceptable_hosts": [] if nal else (acc or [host]),
            "not_a_lesion": nal, "time_seconds": t}


def test_hand_computed_three_class_table():
    assert raw_agreement(X, Y) == pytest.approx(0.8)
    assert cohen_kappa(X, Y) == pytest.approx((0.8 - 0.36) / 0.64)                      # p_e = (2*1 + 2*3 + 1*1)/25
    assert gwet_ac1(X, Y) == pytest.approx((0.8 - 0.31) / 0.69)                         # pi = .3/.5/.2 -> p_e = .62/2
    pa = positive_agreement(X, Y)
    assert pa["A"]["positive_agreement"] == pytest.approx(2 / 3) and pa["B"]["positive_agreement"] == pytest.approx(0.8)
    assert pa["C"] == {"n_x": 1, "n_y": 1, "positive_agreement": 1.0}
    assert confusion(X, Y) == {"categories": ["A", "B", "C"], "counts": [[1, 1, 0], [0, 2, 0], [0, 0, 1]]}
    assert gwet_ac1(X, Y, categories=["A", "B", "C", "D"]) == pytest.approx((0.8 - 0.62 / 3) / (1 - 0.62 / 3))


def test_pairs_use_latest_answers_of_both_readers_and_map_not_a_lesion_to_its_own_class():
    a = [_lab(0, "white_matter", ["white_matter", "cortex"]), _lab(1, None, nal=True), _lab(2, "cortex")]
    b = [_lab(0, "cortex", ["cortex"]), _lab(1, "white_matter"), _lab(3, "cortex")]
    P = pairs(a, b)
    assert [p["lesion_id"] for p in P] == [0, 1]
    assert (P[0]["x"], P[0]["y"], P[0]["sx"], P[0]["sy"]) == ("white_matter", "cortex", {"white_matter", "cortex"}, {"cortex"})
    assert (P[1]["x"], P[1]["sx"]) == ("not_a_lesion", {"not_a_lesion"})
    assert set_agreement(P) == pytest.approx(0.5)                                        # lesion 0 sets intersect, lesion 1 do not
    assert [p["lesion_id"] for p in pairs(a, b, lesion_ids=[1, 3])] == [1]


def test_bootstrap_ci_contains_the_point_estimate_and_narrows_with_more_patients():
    small = {f"p{i}": ([1.0, 1.0, 0.0] if i % 2 else [1.0, 0.0, 0.0]) for i in range(10)}      # pooled mean 0.5
    big = {f"p{i}": ([1.0, 1.0, 0.0] if i % 2 else [1.0, 0.0, 0.0]) for i in range(100)}
    lo, hi = bootstrap_ci(small, n_boot=500, seed=0)
    lo2, hi2 = bootstrap_ci(big, n_boot=500, seed=0)
    assert lo <= 0.5 <= hi and lo2 <= 0.5 <= hi2 and (hi2 - lo2) < (hi - lo)
    assert bootstrap_ci(small, n_boot=500, seed=0) == (lo, hi)
    assert bootstrap_ci({"p0": [1.0, 1.0]}, n_boot=50) == (1.0, 1.0)


def _reg(lid, patient, band, stratum):
    return {"lesion_id": lid, "patient_id": patient, "band": band, "stratum_geometry": stratum}


REG = {0: _reg(0, "p0", "0", "inplane_0.69_slice_5"), 1: _reg(1, "p0", "0", "inplane_0.69_slice_5"), 2: _reg(2, "p1", "0-2", "inplane_0.69_slice_5"),
       3: _reg(3, "p2", ">4", "inplane_0.62_slice_3"), 4: _reg(4, "p3", "2-4", "inplane_0.86_slice_5")}
A = [_lab(0, "white_matter", t=30), _lab(1, "cortex", ["cortex", "white_matter"], t=60), _lab(2, "white_matter", t=45), _lab(3, "cerebellum", t=20), _lab(4, "thalamus", t=90)]
B = [_lab(0, "white_matter"), _lab(1, "white_matter"), _lab(2, "cortex"), _lab(3, "cerebellum"), _lab(4, "basal_ganglia")]


def test_strata_report_sums_to_the_total_and_separates_3mm_volumes():
    P = pairs(A, B)
    s = strata_report(P, REG)
    assert sum(v["n"] for v in s["band"].values()) == 5 == sum(v["n"] for v in s["stratum_geometry"].values())
    assert s["band"]["0"] == {"n": 2, "raw": 0.5, "set_agreement": 1.0}                 # lesion 1: sets {cortex, wm} & {wm}
    assert s["band"][">4"]["raw"] == 1.0 and s["slice_3mm"] == {"n": 1, "raw": 1.0, "set_agreement": 1.0}
    assert s["stratum_geometry"]["inplane_0.86_slice_5"]["raw"] == 0.0


def test_time_summary_and_gate():
    t = time_summary(A)
    assert t["n"] == 5 and t["median_s"] == 45 and t["q1_s"] == 30 and t["q3_s"] == 60 and t["hours_for_1297"] == pytest.approx(45 * 1297 / 3600)
    assert time_summary([_lab(0, "cortex")]) == {"n": 0, "median_s": None, "q1_s": None, "q3_s": None, "hours_for_1297": None}
    assert gate_r7(0.81, 0.72) == {"all_ci_low": 0.81, "pass_all": True, "band0_raw": 0.72, "pass_band0": True, "single_host_endpoint_allowed": True}
    assert gate_r7(0.79, 0.72)["pass_all"] is False and gate_r7(0.85, 0.69)["single_host_endpoint_allowed"] is False


def test_summarise_assembles_everything_and_restricts_to_a_lesion_subset():
    s = summarise(A, B, list(REG.values()), n_boot=200, seed=0)
    assert s["n_pairs"] == 5 and s["n_patients"] == 4 and s["raw"] == pytest.approx(0.4)
    assert s["raw_ci95"][0] <= 0.4 <= s["raw_ci95"][1] and not math.isnan(s["kappa"]) and not math.isnan(s["ac1"])
    assert s["set_agreement"] == pytest.approx(0.6) and s["gate_r7"]["band0_raw"] == 0.5 and s["gate_r7"]["pass_all"] is False
    assert set(s["strata"]) == {"band", "stratum_geometry", "slice_3mm"} and s["time"]["reader_a"]["n"] == 5 and s["time"]["reader_b"]["n"] == 0
    sub = summarise(A, B, list(REG.values()), lesion_ids=[0, 3], n_boot=50)
    assert sub["n_pairs"] == 2 and sub["raw"] == 1.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_level_r_stats.py -q -p no:cacheprovider`
Expected: FAIL，`ImportError`

- [ ] **Step 3: 写 level_r_stats.py**

```python
# anatobind/eval/level_r_stats.py
"""Reader agreement for Level R (spec §8; v2.6 §7.5, §7.7). Inputs are the latest answers of the two readers (Store
.latest_labels) and the registry rows; nothing here reads the database or the sealed folds directly."""
import math
from collections import Counter, defaultdict

import numpy as np

from anatobind.level_r.registry import BANDS, is_3mm
from anatobind.level_r.schema import NOT_A_LESION

N_FULL = 1297
NAN = float("nan")


def host_of(label):
    return NOT_A_LESION if label["not_a_lesion"] else label["primary_host"]


def set_of(label):
    return {NOT_A_LESION} if label["not_a_lesion"] else set(label["acceptable_hosts"])


def pairs(a, b, lesion_ids=None):
    A = {l["lesion_id"]: l for l in a}
    B = {l["lesion_id"]: l for l in b}
    ids = set(A) & set(B)
    if lesion_ids is not None:
        ids &= set(lesion_ids)
    return [{"lesion_id": i, "x": host_of(A[i]), "y": host_of(B[i]), "sx": set_of(A[i]), "sy": set_of(B[i])} for i in sorted(ids)]


def raw_agreement(x, y):
    return sum(p == q for p, q in zip(x, y)) / len(x) if x else NAN


def cohen_kappa(x, y):
    n = len(x)
    if not n:
        return NAN
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[c] * cy[c] for c in set(cx) | set(cy)) / n ** 2
    po = raw_agreement(x, y)
    return (po - pe) / (1 - pe) if pe < 1 else NAN


def gwet_ac1(x, y, categories=None):
    n = len(x)
    if not n:
        return NAN
    cats = list(categories) if categories else sorted(set(x) | set(y))
    cx, cy = Counter(x), Counter(y)
    pi = {c: (cx[c] + cy[c]) / (2 * n) for c in cats}
    pe = sum(p * (1 - p) for p in pi.values()) / (len(cats) - 1) if len(cats) > 1 else 0.0
    po = raw_agreement(x, y)
    return (po - pe) / (1 - pe) if pe < 1 else NAN


def positive_agreement(x, y):
    out = {}
    cx, cy = Counter(x), Counter(y)
    for c in sorted(set(x) | set(y)):
        agree = sum(p == q == c for p, q in zip(x, y))
        denom = cx[c] + cy[c]
        out[c] = {"n_x": cx[c], "n_y": cy[c], "positive_agreement": 2 * agree / denom if denom else NAN}
    return out


def confusion(x, y):
    cats = sorted(set(x) | set(y))
    return {"categories": cats, "counts": [[sum(p == a and q == b for p, q in zip(x, y)) for b in cats] for a in cats]}


def set_agreement(P):
    return sum(bool(p["sx"] & p["sy"]) for p in P) / len(P) if P else NAN


def bootstrap_ci(outcomes_by_patient, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI of the pooled mean, resampling patients with replacement (v2.6 §12.3: patients are the units)."""
    pats = sorted(outcomes_by_patient)
    arrs = [np.asarray(outcomes_by_patient[p], float) for p in pats]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pats), len(pats))
        stats.append(float(np.concatenate([arrs[i] for i in idx]).mean()))
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def _layer(P):
    return {"n": len(P), "raw": raw_agreement([p["x"] for p in P], [p["y"] for p in P]), "set_agreement": set_agreement(P)}


def strata_report(P, reg_by_id):
    out = {"band": {}, "stratum_geometry": {}}
    for key in out:
        for value in sorted({reg_by_id[p["lesion_id"]][key] for p in P}, key=lambda v: (BANDS.index(v) if v in BANDS else 99, v)):
            out[key][value] = _layer([p for p in P if reg_by_id[p["lesion_id"]][key] == value])
    out["slice_3mm"] = _layer([p for p in P if is_3mm(reg_by_id[p["lesion_id"]]["stratum_geometry"])])
    return out


def time_summary(labels):
    t = np.array([l["time_seconds"] for l in labels if l.get("time_seconds") is not None], float)
    if not t.size:
        return {"n": 0, "median_s": None, "q1_s": None, "q3_s": None, "hours_for_1297": None}
    med = float(np.median(t))
    return {"n": int(t.size), "median_s": med, "q1_s": float(np.percentile(t, 25)), "q3_s": float(np.percentile(t, 75)),
            "hours_for_1297": med * N_FULL / 3600}


def gate_r7(raw_ci_low, raw_band0):
    pass_all = bool(raw_ci_low >= 0.80)
    pass_band0 = bool(raw_band0 >= 0.70)
    return {"all_ci_low": raw_ci_low, "pass_all": pass_all, "band0_raw": raw_band0, "pass_band0": pass_band0,
            "single_host_endpoint_allowed": pass_all and pass_band0}


def summarise(a, b, registry, lesion_ids=None, n_boot=2000, seed=0):
    reg = {r["lesion_id"]: r for r in registry}
    P = pairs(a, b, lesion_ids)
    x, y = [p["x"] for p in P], [p["y"] for p in P]
    outcomes = defaultdict(list)
    for p in P:
        outcomes[reg[p["lesion_id"]]["patient_id"]].append(float(p["x"] == p["y"]))
    lo, hi = bootstrap_ci(outcomes, n_boot, seed) if P else (NAN, NAN)
    band0 = [p for p in P if reg[p["lesion_id"]]["band"] == "0"]
    raw0 = raw_agreement([p["x"] for p in band0], [p["y"] for p in band0])
    return {"n_pairs": len(P), "n_patients": len(outcomes), "raw": raw_agreement(x, y), "raw_ci95": [lo, hi],
            "kappa": cohen_kappa(x, y), "ac1": gwet_ac1(x, y), "positive_agreement": positive_agreement(x, y),
            "confusion": confusion(x, y), "set_agreement": set_agreement(P), "strata": strata_report(P, reg),
            "gate_r7": gate_r7(lo, raw0), "time": {"reader_a": time_summary(a), "reader_b": time_summary(b)}}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_level_r_stats.py -q -p no:cacheprovider`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add anatobind/eval/level_r_stats.py tests/test_level_r_stats.py
git commit -m "Level R: reader agreement statistics with kappa, Gwet AC1, positive agreement, patient bootstrap, strata and the R7 gate"
```

---

### Task 10: 封存标签加载器（训练折 / 测试折 + 访问日志）

**Files:**
- Create: `anatobind/eval/level_r_labels.py`
- Test: `tests/test_level_r_labels.py`

**Interfaces:**
- Consumes: Task 7 的 `seal()` 产物（`labels_fold{k}.csv` + manifest）与 `sha256_file`。
- Produces: `SEALED_DIR = Path("/data2/congcong/data/FM_data/derived/level_r/sealed")`、`MANIFEST = Path("data/level_r/sealed_manifest.json")`、`class SealIntegrityError(RuntimeError)`、`class SealedAccessError(RuntimeError)`、`load_fold(k, sealed_dir=SEALED_DIR, manifest_path=MANIFEST) -> list[dict]`（sha256 与行数核对；行 dict：`lesion_id: int, status, primary_host, acceptable_hosts: list, not_a_lesion: bool`）、`load_train_labels(k, ...) -> list[dict]`（除第 k 折外的全部）、`load_test_labels(k, unblind=False, ..., log_path=None) -> list[dict]`（`unblind is not True` 就抛 `SealedAccessError`；否则追加一行 `ts\tfold{k}\t<调用方文件>` 到 `sealed_dir/access_log.txt` 或 `log_path`）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_level_r_labels.py
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
    seal(final, {i: i % 5 for i in range(10)}, tmp_path / "sealed", tmp_path / "manifest.json", now="2026-10-01T00:00:00+00:00")
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_level_r_labels.py -q -p no:cacheprovider`
Expected: FAIL，`ImportError`

- [ ] **Step 3: 写 level_r_labels.py**

```python
# anatobind/eval/level_r_labels.py
"""Sealed Level R labels (v2.6 §12.7). Development code reads training folds; the outer test fold opens only with the
literal unblind=True and every such read is logged. The sha256 manifest lives in the repository, the CSVs on the
server, so neither can drift without the other noticing."""
import csv
import inspect
import json
from pathlib import Path

from anatobind.level_r.admin import sha256_file
from anatobind.level_r.store import now_iso

SEALED_DIR = Path("/data2/congcong/data/FM_data/derived/level_r/sealed")
MANIFEST = Path("data/level_r/sealed_manifest.json")


class SealIntegrityError(RuntimeError):
    pass


class SealedAccessError(RuntimeError):
    pass


def _parse(r):
    return {"lesion_id": int(r["lesion_id"]), "status": r["status"], "primary_host": r["primary_host"] or None,
            "acceptable_hosts": json.loads(r["acceptable_hosts"] or "[]"), "not_a_lesion": r["not_a_lesion"] in ("True", "true", "1")}


def load_fold(k, sealed_dir=SEALED_DIR, manifest_path=MANIFEST):
    entry = json.loads(Path(manifest_path).read_text())[f"fold{k}"]
    p = Path(sealed_dir) / f"labels_fold{k}.csv"
    if sha256_file(p) != entry["sha256"]:
        raise SealIntegrityError(f"{p}: sha256 differs from {manifest_path}")
    with open(p, newline="") as fh:
        rows = [_parse(r) for r in csv.DictReader(fh)]
    if len(rows) != entry["rows"]:
        raise SealIntegrityError(f"{p}: {len(rows)} rows, manifest says {entry['rows']}")
    return rows


def load_train_labels(k, sealed_dir=SEALED_DIR, manifest_path=MANIFEST):
    folds = sorted(int(name[4:]) for name in json.loads(Path(manifest_path).read_text()))
    return [r for f in folds if f != k for r in load_fold(f, sealed_dir, manifest_path)]


def load_test_labels(k, unblind=False, sealed_dir=SEALED_DIR, manifest_path=MANIFEST, log_path=None):
    if unblind is not True:
        raise SealedAccessError(f"fold {k} is a sealed outer test fold (v2.6 §12.7); pass unblind=True only for the final evaluation")
    caller = inspect.stack()[1].filename
    log = Path(log_path) if log_path else Path(sealed_dir) / "access_log.txt"
    with open(log, "a") as fh:
        fh.write(f"{now_iso()}\tfold{k}\t{caller}\n")
    return load_fold(k, sealed_dir, manifest_path)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_level_r_labels.py -q -p no:cacheprovider`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add anatobind/eval/level_r_labels.py tests/test_level_r_labels.py
git commit -m "Level R: sealed fold loaders with sha256 verification, an explicit unblind flag and an access log"
```

---

### Task 11: pilot 报告脚本

**Files:**
- Create: `scripts/level_r_report.py`
- Test: `tests/test_level_r_report.py`

**Interfaces:**
- Consumes: Task 4 `Store.latest_labels(reader_id)`、`Store.readers()`；Task 9 `summarise`；Task 1 `load_registry`；`data/level_r/pilot_150.json` 的 `lesion_ids`。
- Produces（脚本内函数，测试用 importlib 加载，与 `tests/test_brain_frame.py` 同一写法）：`build_summary(store, registry, lesion_ids, n_boot, seed) -> dict`（两位 role=reader 的读者按 reader_id 排序）、`render_markdown(summary, title, command) -> str`、`main()`（参数 `--db --registry --pilot --out --n-boot --seed --title`；`--out` 已存在则拒绝）。
- 报告固定段：概况（n_pairs、n_patients、raw 与 95% 区间、κ、AC1、集合值一致）；`GATE_R7: PASS|FAIL`（全体区间下限 ≥ 0.80，0 mm 档 raw ≥ 0.70，逐项给数）；每类 positive agreement 表；混淆矩阵；距离四档 × 采集分层表，3 mm 卷单列；两位读者用时（中位、四分位、按 1297 折算工时）；末尾原样附命令与 `summary` JSON。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_level_r_report.py
import importlib.util
import json
from pathlib import Path

from anatobind.level_r.store import Store


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/level_r_report.py"
    spec = importlib.util.spec_from_file_location("level_r_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lab(host, acc=None, t=20.0):
    return {"primary_host": host, "acceptable_hosts": acc or [host], "topography": "deep_white_matter", "adjacency": ["none"],
            "ambiguity": "certain", "not_a_lesion": False, "local_quality": "good", "confidence": 4, "comment": "", "time_seconds": t}


def _registry():
    return [{"lesion_id": i, "patient_id": f"p{i // 2}", "band": "0" if i < 4 else ">4", "stratum_geometry": "inplane_0.69_slice_5" if i != 5 else "inplane_0.62_slice_3"}
            for i in range(6)]


def test_build_summary_and_render(tmp_path):
    m = _load()
    s = Store(tmp_path / "db.sqlite")
    s.add_reader("r2", "reader", "fedcba9876543210", "读者 2")
    s.add_reader("r1", "reader", "0123456789abcdef", "读者 1")
    s.add_reader("adj", "adjudicator", "aaaaaaaaaaaaaaaa", "裁定")
    s.load_lesions([{"lesion_id": i, "code": f"c{i}", "volume_code": "v", "z0": 0, "z1": 0, "boxes": {}} for i in range(6)])
    for i in range(6):
        s.submit_label("r1", i, _lab("white_matter", t=10.0 * (i + 1)))
        s.submit_label("r2", i, _lab("white_matter" if i < 5 else "cortex"))
    summary = m.build_summary(s, _registry(), lesion_ids=[0, 1, 2, 3, 4, 5], n_boot=100, seed=0)
    assert summary["readers"] == ["r1", "r2"] and summary["n_pairs"] == 6 and summary["raw"] == 5 / 6
    md = m.render_markdown(summary, title="pilot", command="python scripts/level_r_report.py --db x")
    for piece in ("# pilot", "GATE_R7:", "raw agreement", "positive agreement", "confusion", "band", "slice_3mm", "hours_for_1297",
                  "python scripts/level_r_report.py --db x", '"n_pairs": 6'):
        assert piece in md, piece
    assert ("GATE_R7: PASS" in md) == summary["gate_r7"]["single_host_endpoint_allowed"]
    sub = m.build_summary(s, _registry(), lesion_ids=[0, 1], n_boot=10, seed=0)
    assert sub["n_pairs"] == 2 and sub["raw"] == 1.0


def test_main_writes_the_report_and_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    m = _load()
    s = Store(tmp_path / "db.sqlite")
    s.add_reader("r1", "reader", "0123456789abcdef", "读者 1")
    s.add_reader("r2", "reader", "fedcba9876543210", "读者 2")
    s.load_lesions([{"lesion_id": i, "code": f"c{i}", "volume_code": "v", "z0": 0, "z1": 0, "boxes": {}} for i in range(2)])
    for i in range(2):
        s.submit_label("r1", i, _lab("cortex"))
        s.submit_label("r2", i, _lab("cortex"))
    reg = tmp_path / "reg.csv"
    import csv
    from test_level_r_registry import HEADER, _row      # tests/ is on sys.path under pytest, like conftest's `from synth import ...`
    with open(reg, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows([_row(0), _row(1, patient="P2")])
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"lesion_ids": [0, 1]}))
    out = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", ["level_r_report.py", "--db", str(tmp_path / "db.sqlite"), "--registry", str(reg), "--pilot", str(pilot),
                                     "--out", str(out), "--n-boot", "20"])
    m.main()
    assert out.exists() and "GATE_R7:" in out.read_text()
    import pytest
    with pytest.raises(SystemExit):
        m.main()
```

若 `from test_level_r_registry import ...` 在你的 pytest 配置下找不到模块，改成在本测试里复制 `HEADER` 与 `_row`（各十行），不要给 `tests/` 加 `__init__.py`。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_level_r_report.py -q -p no:cacheprovider`
Expected: FAIL（脚本不存在）

- [ ] **Step 3: 写报告脚本**

```python
#!/usr/bin/env python
# scripts/level_r_report.py
"""Level R pilot report (spec §8, §9): reader agreement, strata, reading time and the R7 gate, written as markdown with
the command and the raw summary attached.

  D=/data2/congcong/data/FM_data/derived/level_r
  python scripts/level_r_report.py --db $D/level_r.sqlite --pilot data/level_r/pilot_150.json \
      --out docs/verification/$(date +%F)/level_r_pilot.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.level_r_stats import summarise  # noqa: E402
from anatobind.level_r.registry import REGISTRY, load_registry  # noqa: E402
from anatobind.level_r.store import Store  # noqa: E402

FENCE = "`" * 3          # built at runtime so this file itself stays out of markdown fences


def build_summary(store, registry, lesion_ids, n_boot, seed):
    readers = sorted(r["reader_id"] for r in store.readers() if r["role"] == "reader")
    if len(readers) != 2:
        raise ValueError(f"need exactly two readers, have {readers}")
    a, b = (store.latest_labels(r) for r in readers)
    return {"readers": readers, **summarise(a, b, registry, lesion_ids, n_boot=n_boot, seed=seed)}


def _f(x):
    return "nan" if x is None or x != x else f"{x:.3f}"


def _table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def render_markdown(s, title, command):
    g = s["gate_r7"]
    lines = [f"# {title}", "",
             f"readers {s['readers']} · pairs {s['n_pairs']} · patients {s['n_patients']}", "",
             f"raw agreement {_f(s['raw'])} (95% patient-bootstrap CI {_f(s['raw_ci95'][0])}–{_f(s['raw_ci95'][1])}) · "
             f"Cohen κ {_f(s['kappa'])} · Gwet AC1 {_f(s['ac1'])} · set-valued agreement {_f(s['set_agreement'])}", "",
             f"GATE_R7: {'PASS' if g['single_host_endpoint_allowed'] else 'FAIL'} "
             f"(all CI low {_f(g['all_ci_low'])} vs 0.80 -> {g['pass_all']}; 0 mm band raw {_f(g['band0_raw'])} vs 0.70 -> {g['pass_band0']})",
             "", "## positive agreement per class", "",
             _table(["class", "n reader A", "n reader B", "positive agreement"],
                    [[c, v["n_x"], v["n_y"], _f(v["positive_agreement"])] for c, v in s["positive_agreement"].items()]),
             "", "## confusion (rows reader A, columns reader B)", "",
             _table([""] + s["confusion"]["categories"], [[c] + row for c, row in zip(s["confusion"]["categories"], s["confusion"]["counts"])]),
             "", "## strata", ""]
    for key in ("band", "stratum_geometry"):
        lines += [f"### {key}", "", _table([key, "n", "raw", "set agreement"],
                                          [[k, v["n"], _f(v["raw"]), _f(v["set_agreement"])] for k, v in s["strata"][key].items()]), ""]
    t3 = s["strata"]["slice_3mm"]
    lines += [f"slice_3mm volumes: n {t3['n']}, raw {_f(t3['raw'])}, set agreement {_f(t3['set_agreement'])}", "", "## reading time", ""]
    for name, t in s["time"].items():
        lines.append(f"- {name}: n {t['n']}, median {t['median_s']} s (IQR {t['q1_s']}–{t['q3_s']}), hours_for_1297 {t['hours_for_1297']}")
    lines += ["", "## command", "", FENCE, command, FENCE, "", "## summary (raw)", "", FENCE + "json", json.dumps(s, indent=1, default=list), FENCE, ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--pilot", type=Path, help="pilot json; restricts the report to its lesion_ids")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="Level R pilot agreement")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; reports are never overwritten")
    ids = json.loads(a.pilot.read_text())["lesion_ids"] if a.pilot else None
    s = build_summary(Store(a.db), load_registry(a.registry), ids, a.n_boot, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render_markdown(s, a.title, " ".join(sys.argv)))
    print(f"pairs {s['n_pairs']} raw {s['raw']:.3f} CI {s['raw_ci95']} GATE_R7 {'PASS' if s['gate_r7']['single_host_endpoint_allowed'] else 'FAIL'} -> {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_level_r_report.py -q -p no:cacheprovider`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add scripts/level_r_report.py tests/test_level_r_report.py
git commit -m "Level R: pilot agreement report with the R7 gate, strata, confusion and reading-time tables"
```

---

### Task 12: 运维文档、代码地图、真实数据导出与冒烟部署

**Files:**
- Create: `docs/level_r_tool.md`
- Create: `docs/verification/2026-09-26/level_r_smoke.md`
- Modify: `CLAUDE.md`（"代码地图"一节末尾加 Level R 一段；不改其他行）
- 服务器侧产物（不入库）：`/data2/congcong/data/FM_data/derived/level_r/{volumes/, lesions.json}`、`/data2/congcong/data/FM_data/derived/level_r_smoke/`

**Interfaces:**
- Consumes: Task 3/5/7/8 的脚本。
- Produces: 可复制粘贴的部署步骤；真实导出；一个在 8791 端口起过、curl 过、又停掉的冒烟服务的记录。真实读者的 token 不在本任务生成（读片人姓名待定，v2.6 §18）。

- [ ] **Step 1: 写 docs/level_r_tool.md**

```markdown
# Level R 读片工具：部署与运维

规格 `docs/superpowers/specs/2026-09-25-level-r-annotation-tooling-design.md`，实施计划 `docs/superpowers/plans/2026-09-26-level-r-annotation-tooling.md`。
所有命令在仓库根目录执行，`python` = `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python`，`D=/data2/congcong/data/FM_data/derived/level_r`。

## 一次性准备
1. 导出图像与病灶：`nice -n 19 python scripts/level_r_export.py --out $D`（165 卷，约 540 MB；已存在 `lesions.json` 会拒跑）。
2. 折表与 pilot 已入库：`data/level_r/folds.json`、`data/level_r/pilot_150.json`。重生成命令见两份脚本头部，脚本拒绝覆盖。
3. 建库：`python scripts/level_r_admin.py init --db $D/level_r.sqlite --lesions $D/lesions.json`。
4. 读者：`python scripts/level_r_admin.py add-reader --db $D/level_r.sqlite --reader-id r1 --role reader --display "读者 1"`（r2 同理；裁定人 `--reader-id adj --role adjudicator`）。token 只打印一次，记到用户手里，不写进仓库。
5. 顺序：`python scripts/level_r_admin.py order --db $D/level_r.sqlite --reader-id r1 --seed 1 --pilot data/level_r/pilot_150.json`，r2 用 `--seed 2`。

## 起服务
```
setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/level_r_server.py \
    --db $D/level_r.sqlite --data-root $D --pid-file $D/server.pid > $D/server.log 2>&1 &
```
默认只听 `127.0.0.1:8790`。医生访问方式两种：用户在自己机器上 `ssh -L 8790:127.0.0.1:8790 <server>` 后打开 `http://127.0.0.1:8790/?token=<token>`；或明确加 `--bind 0.0.0.0` 走内网地址。停服务：`kill $(cat $D/server.pid)`。sqlite 每次提交即落盘，崩溃不丢数据。

## 每次读片结束
- 导出：`python scripts/level_r_admin.py export --db $D/level_r.sqlite --out $D/export`
- 备份：`python scripts/level_r_admin.py backup --db $D/level_r.sqlite --out $D/backup`（带时间戳，不覆盖）

## pilot 报告
`python scripts/level_r_report.py --db $D/level_r.sqlite --pilot data/level_r/pilot_150.json --out docs/verification/$(date +%F)/level_r_pilot.md`
门（spec R7）：全体 raw 的 95% 区间下限 ≥ 0.80 且 0 mm 档 raw ≥ 0.70。门没过先回 v2.6 §3 本体与表单再议，不硬推。

## 封存（全集读完、裁定完之后，一次）
`python scripts/level_r_admin.py seal --final $D/export/final_labels.csv --folds data/level_r/folds.json --out $D/sealed`
写 `$D/sealed/labels_fold{k}.csv` 与仓库里的 `data/level_r/sealed_manifest.json`（提交它）。之后开发只用 `anatobind.eval.level_r_labels.load_train_labels(k)`；`load_test_labels(k, unblind=True)` 只在最终评估调用，每次都记到 `$D/sealed/access_log.txt`。

## 医生看不到什么
SynthSeg、任何模型输出、距离档、采集系列、患者号、fastMRI+ 原标签文字、h5 文件名。白名单在 `anatobind/level_r/blind.py`，服务端每个 JSON 都过 `assert_blind`。
```

- [ ] **Step 2: CLAUDE.md 代码地图加一段**

在"代码地图"列表末尾（`- 工作树：` 那一条之前）加：

```markdown
- `anatobind/level_r/`（PR-B，Level R 读片工具）：`schema.py`（枚举与校验）、`registry.py`（Gate 0.5 注册表、距离四档、匿名码）、`blind.py`（白名单）、`export.py`（u16 卷 + 逐层框，与注册表核对）、`store.py`（只追加 sqlite）、`server.py`（API）、`app/`（读者/裁定页、读片说明）、`admin.py`（折、顺序、导出、封存、备份）、`pilot.py`（150 例抽样）；`anatobind/eval/level_r_stats.py`（一致率、κ、AC1、分层、门）、`anatobind/eval/level_r_labels.py`（封存折加载、unblind 旗标）；脚本 `scripts/level_r_{export,admin,server,pilot_sample,report}.py`；入库数据 `data/level_r/{folds.json,pilot_150.json}`；运维 `docs/level_r_tool.md`。服务器数据在 `derived/level_r/`。
```

- [ ] **Step 3: 真实导出**

Run:
```bash
D=/data2/congcong/data/FM_data/derived/level_r
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/level_r_export.py --out $D 2>&1 | tail -3
ls $D/volumes | wc -l; du -sh $D; python -c "import json; r=json.load(open('$D/lesions.json')); print(len(r), len({x['volume_code'] for x in r}))"
```
Expected: 最后一行 `exported 1297 lesions from 165 volumes`；`volumes/` 330 个文件（165 × 2）；`1297 165`。若 `match_registry` 抛错，停下写明是哪个文件哪个病灶，不要改判据绕过——那意味着注册表与 CSV 的合并路径不一致，是 Gate 0.5 级别的问题。

- [ ] **Step 4: 冒烟部署（独立库，不碰正式库）**

```bash
D=/data2/congcong/data/FM_data/derived/level_r; S=/data2/congcong/data/FM_data/derived/level_r_smoke; mkdir -p $S
python scripts/level_r_admin.py init --db $S/level_r.sqlite --lesions $D/lesions.json
python scripts/level_r_admin.py add-reader --db $S/level_r.sqlite --reader-id smoke_r1 --role reader --display "冒烟读者 1" | tee $S/tokens.txt
python scripts/level_r_admin.py add-reader --db $S/level_r.sqlite --reader-id smoke_r2 --role reader --display "冒烟读者 2" | tee -a $S/tokens.txt
python scripts/level_r_admin.py add-reader --db $S/level_r.sqlite --reader-id smoke_adj --role adjudicator --display "冒烟裁定" | tee -a $S/tokens.txt
python scripts/level_r_admin.py order --db $S/level_r.sqlite --reader-id smoke_r1 --seed 1 --pilot data/level_r/pilot_150.json
python scripts/level_r_admin.py order --db $S/level_r.sqlite --reader-id smoke_r2 --seed 2 --pilot data/level_r/pilot_150.json
setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/level_r_server.py \
    --db $S/level_r.sqlite --data-root $D --port 8791 --pid-file $S/server.pid > $S/server.log 2>&1 &
sleep 2; T=$(grep -o 'token=[0-9a-f]*' $S/tokens.txt | head -1 | cut -d= -f2)
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" http://127.0.0.1:8791/
curl -s http://127.0.0.1:8791/api/enums | head -c 200; echo
curl -s "http://127.0.0.1:8791/api/me?token=$T"; echo
L=$(curl -s "http://127.0.0.1:8791/api/me?token=$T" | python -c "import json,sys; print(json.load(sys.stdin)['next'])")
curl -s "http://127.0.0.1:8791/api/lesion/$L?token=$T" | head -c 300; echo
V=$(curl -s "http://127.0.0.1:8791/api/lesion/$L?token=$T" | python -c "import json,sys; print(json.load(sys.stdin)['lesion']['volume_code'])")
curl -s "http://127.0.0.1:8791/api/volume/$V.json?token=$T"; echo
curl -s -o /dev/null -w "u16 bytes %{size_download}\n" "http://127.0.0.1:8791/api/volume/$V.u16?token=$T"
curl -s -o /dev/null -w "bad token -> %{http_code}\n" "http://127.0.0.1:8791/api/me?token=0000000000000000"
```
Expected: `200 text/html`；enums JSON；`/api/me` 给 `total 1297, done 0, next <id>`；病灶 JSON 只有 6 个键；卷 JSON 是 `shape [16, 320, 320]` 一类；u16 字节数 = shape 乘积 × 2；坏 token → 403。把全部输出原样写进 `docs/verification/2026-09-26/level_r_smoke.md`（命令 + 输出 + 服务日志首行）。

- [ ] **Step 5: 浏览器验收（用户做，计划里只登记）**

把 `$S/tokens.txt` 里冒烟读者 1 的链接与端口转发命令（`ssh -L 8791:127.0.0.1:8791 <server>`）交给用户，请用户在浏览器里：打开第一例、翻层、拖窗、放大、提交一次、回列表改一次、用冒烟读者 2 再提交一例让它与读者 1 分歧、用裁定 token 看到分歧并裁定。每步是否正常记入 `level_r_smoke.md` 的"浏览器验收"表（用户口述结果，标 `USER_REPORTED`）。这一步不阻塞后续任务；服务不停，等用户看完再 `kill $(cat $S/server.pid)`。

- [ ] **Step 6: 提交**

```bash
git add docs/level_r_tool.md docs/verification/2026-09-26/level_r_smoke.md CLAUDE.md
git commit -m "Level R: deployment runbook, smoke deployment record and the code-map entry"
```

---

### Task 13: 交接（STATUS.md、全套测试；push 与合 main 前问用户）

**Files:**
- Modify: `STATUS.md`（整体重写，五段固定）
- Modify: `CLAUDE.md`（"当前状态"那一行改为本轮状态与阅读顺序）

- [ ] **Step 1: 全套测试**

Run: `python -m pytest tests/ -q -p no:cacheprovider`
Expected: 413 + 本计划新增的全部通过，0 failed；把 `N passed in Ns` 原样写进 STATUS §1。

- [ ] **Step 2: 重写 STATUS.md**

五段：①已完成且已验证（导出 1297/165 的命令与输出、冒烟 curl 输出、测试数、pilot 格子表、folds 五折计数）；②待用户拍板（读片人姓名与 token 发放、裁定人、伦理备案、服务对外方式 A 端口转发/B `--bind 0.0.0.0`、level-r 分支合回 main 的时机、nnDetection 第二臂何时起）；③下一步（发 token → pilot 150 → `level_r_report.py` → 过门读全集 → 裁定 → `seal`；PR-C 关系基线在 pilot 期间可并行开工）；④坑（`derived/level_r_smoke/` 是冒烟库不是正式库；8790 正式 / 8791 冒烟；`lesions.json` 与 `pilot_150.json` 都拒绝覆盖；浏览器交互无 CI，改 `app.js` 后要人点一遍；测试不读 /data2）；⑤为什么（stdlib 服务不加依赖、u16 而非 PNG、白名单而非黑名单、只追加、pilot 缺口回填到大格子、封存一次性）。

- [ ] **Step 3: 改 CLAUDE.md 当前状态行**

把"**当前状态（2026-09-25 …）**"那一整条改成：PR-B 工具已实现并冒烟通过（分支 `plan/level-r-tooling-2026-09-25`，工作树 `../foundation_model-levelr`），等读者 token 与 pilot；先读 `STATUS.md`，再读 `docs/superpowers/specs/2026-09-25-level-r-annotation-tooling-design.md` 与 `docs/level_r_tool.md`。其余不动。

- [ ] **Step 4: 提交，然后停下问用户**

```bash
git add STATUS.md CLAUDE.md
git commit -m "Status 2026-09-26: Level R tooling implemented and smoke-tested; waiting for reader tokens and the pilot"
```

停下，向用户列出三件事等答复，不要自己做：① 是否 push 分支（push 是仓库外动作）；② 是否合回 main 并打 tag `handoff/2026-09-26-level-r-tooling`（会话边界规矩）；③ 冒烟服务是否可以停。

---

## 自审记录（写计划时做的三项检查）

1. **规格覆盖**：spec §3 系统组成的每个文件都有任务（export→3，admin→7，server→5，app→6，store→4，blind→2，stats→9，labels→10，pilot→8，report→11，folds.json→7，sealed_manifest.json→7/10 在封存时产生）；R1–R10 逐条：R1/R3/R4 → Task 5/6，R2 → 4/5/6 裁定角色，R5 → 2/5/6，R6 → 1/3，R7 → 9/11，R8 → 8，R9 → 4，R10 → 7/10。spec §11 的测试清单：导出（3）、服务（5）、存储无修改路径（4）、封存 sha256/篡改/旗标/日志（10）、统计手算表/bootstrap/分层/集合值（9）、pilot 下限/上限/总数/种子（8）。spec §12"明确不做"未被任何任务触碰。`docs/handoff/2026-09-25-assessment-review-corrections.md` §8 的 primary_host 操作定义进了 `guide.html`（Task 6）。
2. **占位符**：没有留待以后填的空位；每个代码步骤都有完整代码；读者姓名与 token 是规格本身的待定项（v2.6 §18），不在本计划范围，Task 12/13 只登记。
3. **名称一致性**：`Store.latest_labels / label_rows / adjudication_rows / latest_adjudications / disagreements / final_labels`（Task 4）被 Task 5/7/11 按同名调用；`answer_view` 只在 Task 5；`sha256_file` 定义在 Task 7 的 `admin.py`，Task 10 从那里导入；`FINAL_COLUMNS` 定义在 Task 7，Task 10 测试导入它；`distance_band / BANDS / is_3mm` 定义在 Task 1，Task 8/9 使用；`enums()` 定义在 Task 1，Task 5 路由 `/api/enums` 返回它，Task 6 前端消费它。

已知未覆盖（有意）：浏览器交互没有自动化测试（Task 6 静态检查 + Task 12 用户验收）；服务无 HTTPS（内网/端口转发，spec §10）；单实例 sqlite 写锁靠 `threading.Lock`，两位读者的并发量下够用。

