# Gate 0 + 膝 H1 重跑 + Gate 0.5 · 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 fastMRI+ 标注框的上下翻转修进数据引擎（Gate 0），用修好的框在 fastMRI+ 膝关节上原样重训 leg 2 的 2.5D 检测器并按预先登记的口径判"能不能迁到 SKM-TEA"（H1 重跑，VERDICT §4 选项 ① 的前半段），再对 1297 个 fastMRI+ 脑 FLAIR 小病灶做与模型无关的几何盘点并冻结困难组阈值 t（Gate 0.5）。

**Architecture:** 翻转只在一处实现（`anatobind/data_engine/fastmri.py` 的 `convert_box_csv_to_rss`），脑和膝都从它走；leg 2 的旧导出 56 GB 不重导，新建导出根用链接指向旧卷目录，只换坐标文件与 manifest，manifest 带 `transform_version`，加载器见不到版本 2 就拒绝；检测器训练脚本、缓存、H1 门一行不改逻辑，只换数据根；检测评估复用计划 1 的 `detection_metrics`（3D IoU ≥ 0.1、每卷假阳 ≤ 2 的工作点）；Gate 0.5 的几何量放进新模块 `anatobind/eval/geometry.py`，脑侧无类别查表放进 `anatobind/eval/lookup.py`，探针的读框/合并/翻转进包并加测试。

**Tech Stack:** numpy、scipy.ndimage（EDT）、h5py、nibabel、matplotlib、torch 2.5.1（检测器）、pytest；nvgen 环境。

**Spec:** `docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md` §2（Gate 0 / Gate 0.5 定义）、§4.1–4.5（修复、必需测试、旧导出处理、患者折、几何盘点）、§6.2（修复后重跑 H1 的报告项）、§7.3（H1 困难组定义）、§3 与 §25 第 4–5 项（脑室/CSF 不作宿主；t 的冻结规则）、§17（代码结构：不另起平行模块）、§23（PR-A / PR-A′ 内容）；`docs/verification/2026-09-23/knee_eval/VERDICT.md` §4（第二臂选项与推荐）；`docs/superpowers/specs/2026-09-23-aur-capability-system-design.md` §2（计划 3 含 Gate 0）。用户 2026-09-24 拍板：按推荐走；npz 不删；已合并分支不删。

## Global Constraints

- 数据只从 `/data2/congcong/data/FM_data` 读；新产物只写 `/data2/congcong/data/FM_data/derived/`。**旧导出 `derived/fastmri_knee/leg2/` 一个字节都不改、不删**（含其 `manifest.csv`、`lesions.csv`、`detections/`）。
- 解释器一律 `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python`；测试一律 `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`（基线 342 passed，约 40 s；每个任务结束时全套必须全过）。
- 工作树 `/data0/congcong/code/Project_Doing/foundation_model-aur`，分支 `build/aur-system`（先 `git merge --ff-only main` 快进到 7a889ed；不开新分支；交接时合回 main 打 tag）。提交作者用仓库本地身份（Congcong Liu），消息英文、句首大写，**不写任何 AI trailer**。
- GPU：四张 A800 `0 1 4 6`，启动前 `nvidia-smi` 确认空闲；每卡一个训练，`CUDA_VISIBLE_DEVICES` 钉死，后台 `setsid nohup … &`，日志进工作树的 `logs/`（未跟踪）。CPU 任务 `nice -n 19`，总线程 ≤ 48。
- 新训练写 `runs/detector_gate0_fold{k}`（工作树内，`runs/` 在 .gitignore），**不碰主目录 `runs/detector_fold{k}`**。新检出缓存写新导出根的 `detections/`。
- 坐标约定：fastMRI+ CSV 的 `y` 从 RSS 数组底部数起，RSS 帧里框占行 `[nr − y − h, nr − y)`、列 `[x, x + w)`；`nr` 是该卷 `reconstruction_rss` 的行数（膝全部 320；脑按卷不同：320/276/256/234/213）。所有导出、评估、盘点只认 RSS 帧（行从顶部数）。
- 任何数字都要附可粘贴命令与原始输出（样板 `docs/verification/2026-09-08/REPORT.md`）；以伪标签或重叠率为参照的数字标 `NOT_EVIDENCE`。

## 本计划预先登记的决定（执行前定，执行中不改）

| # | 决定 | 理由 |
|---|---|---|
| D1 | 翻转函数放 `anatobind/data_engine/fastmri.py`（脑膝共用的 fastMRI 模块），`fastmri_knee.py` 从它导入。v2.6 §17 写的是放 `fastmri_knee.py`，这里偏离一步 | 脑侧 Gate 0.5 也要用，放在膝模块里让脑代码反向依赖膝模块；§17 的本意是"不另起平行模块"，这里没有新模块 |
| D2 | 新导出根 `derived/fastmri_knee/leg2_gate0/`，卷目录是指向 `../leg2/<file>` 的相对符号链接；只写 `lesions.csv`、`folds.json`、`manifest.csv`、`README.txt` | v2.6 §4.3；图像与框约定无关，不重导 56 GB |
| D3 | 旧导出的判定靠 manifest 缺 `transform_version` 列（或值 ≠ 2）；旧 manifest 不改写 | 用户规矩：不覆盖已有数据文件；加载器拒绝即达到 §4.3 的目的 |
| D4 | 折从 h5 `patient_id` 重新生成并断言患者不跨折，且必须与旧 `folds.json` 完全一致 | H1 重跑只换框不换患者划分，前后可比 |
| D5 | Gate 0 审计通过线：膝（CORPDFS 上的积液与骨髓水肿框）转换后亮度胜出的卷 ≥ 90%；脑（FLAIR 小病灶框）整体 ≥ 90% 且每个 ≥ 3 卷的系列 ≥ 75%。不过就停，报告给用户 | 探针实测膝 30/30、脑 22/24；系列门是为了抓"某个系列标在别的网格上"的情况 |
| D6 | H1 重跑：训练配置与 09-15 完全相同（`FULL`、20000 步、批 16、lr 2e-4、wd 0.05、seed 0），只换导出根；检出缓存阈值降到 0.01（旧 0.05）以便扫描工作点 | 只改一个变量才能回答"塌陷是不是框的问题"；缓存阈值是评估参数不是训练参数 |
| D7 | 迁移判据（预先登记）：clean 视图、患者五折、3D IoU ≥ 0.1 一对一、每卷假阳 ≤ 2 的工作点上，**四个与 SKM-TEA 共享的族（meniscus / cartilage / ligament / effusion）大类正确灵敏度 ≥ 0.5** → 推荐迁到 SKM-TEA；否则推荐 VERDICT §4 的 ②（nnDetection）。五族版本与每卷假阳（含 198 卷正常膝单算）一并报告 | 与计划 1 的门同口径；bone 族 SKM-TEA 没有，迁移用不上 |
| D8 | Gate 0.5 的候选宿主本体：白质 {2,41}、皮层 {3,42}、丘脑 {10,49}、基底节 {11,50,12,51,13,52,26,58}、脑干 {16}、小脑 {7,46,8,47}、其他深部灰质 {17,53,18,54,28,60}；左右合并；脑室 {4,43,5,44,14,15} 与 CSF {24} 只作地标 | v2.6 §3 的 primary_host 不分左右；§25 第 4 项脑室/CSF 不作宿主 |
| D9 | `d_interface` = 病灶体素到第二近候选宿主类的最小距离（第一近为 0 时即到最近交界面的距离，误差一个体素对角线内）；`Δd` = 第二近 − 第一近；EDT 用该卷 NIfTI 头里的间距 | v2.6 §4.5 原文："它与'到最近其他候选结构的距离'是同一个量"；用类距离而不是 26 邻域交界体素，才能保证 5 mm 层厚下 t ≤ 4 是面内准则 |
| D10 | t 的冻结：{2,3,4,5} 中取困难组占比 ≥ 15% 的最小值；5 mm 仍不足 → Gate 0.5 不过。若冻结的 t 下困难组超过一半，不自动冻结，报给用户定 | v2.6 §25 第 5 项 + §4.5 的警示（89% 白质体素离交界面 < 3 mm） |
| D11 | 分层同时报两种：v2.6 的系列号分层（200/201 vs 其他）和按实测几何分层（面内 ≤ 0.70 mm / ≈ 0.86 mm × 层厚 5 / 3 mm）。09-24 实测 209/210/205 系列是 320×320@0.6875 mm，与 200 同分辨率，"低分辨率系列"按系列号分不准 | 2026-09-24 冷启动实测 165 卷（见本计划 Task 11 步骤 1 的表） |
| D12 | 加 GitHub Actions 跑 CPU 测试（v2.6 §4.4 与 PR-A 明写） | 折断言现在自然通过，加入其他序列后未必；CI 是唯一自动防线 |
| D13 | `anatobind/eval/geometry.py` 本轮只做距离变换、`d_interface`、`Δd`；v2.6 §17 提到的 16 维几何留给 PR-C | 本轮没有消费者 |

---

## 文件结构

```
anatobind/data_engine/fastmri.py        修改：框约定常量、convert_box_csv_to_rss / convert_box_rss_to_csv、rss_spacing_mm、
                                        voxel_to_world / world_to_voxel、read_fastmri_plus_rows、rows_to_rss_frame、
                                        box_iou_2d、merge_boxes_3d（脑膝共用）
anatobind/data_engine/fastmri_knee.py   修改：read_annotations / merge_to_3d 委托给上面的通用函数；EXPORT_ROOT 指向新根；
                                        LEGACY_EXPORT_ROOT；MANIFEST_FIELDS / manifest_row / write_manifest / load_manifest；
                                        LegacyBoxConvention；load_lesions / load_folds / assert_folds_by_patient；
                                        volume_geometry；LESION_FIELDS / write_lesions；patient_of 缺 patient_id 即报错
anatobind/train/train_detector.py       修改：load_fold 走 load_folds + load_lesions（拒绝旧导出）
scripts/build_fastmri_knee.py           修改：写 RSS 帧的 lesions.csv 与新 manifest
scripts/relink_fastmri_knee_gate0.py    新建：从旧导出建新导出根
scripts/audit_fastmri_plus_boxes.py     新建：转换前后亮度审计 + 叠图（膝、脑各系列）
scripts/check_h1.py                     修改：gather 用 load_lesions
scripts/train_reliability.py            修改：lesions_by_file 用 load_lesions
scripts/cache_detections.py             修改：--score-min
scripts/eval_fastmri_knee_detection.py  新建：H1 重跑的检测评估（工作点、族、大小、中心误差、IoU、患者覆盖）
anatobind/eval/fastmri_knee_detection.py 新建：上面脚本的可测逻辑
anatobind/eval/geometry.py              新建：宿主类映射、类距离图、病灶类距离、d_interface / Δd、member_rects
anatobind/eval/lookup.py                修改：BrainLookup（33 类无类别查表 + 最近兜底）
scripts/brain_frame.py                  新建：Gate 0.5 盘点
.github/workflows/tests.yml             新建：CI
requirements-ci.txt                     新建
tests/test_fastmri_box_convention.py    新建（Task 1）
tests/test_fastmri_plus_rows.py         新建（Task 2）
tests/test_fastmri_knee_manifest.py     新建（Task 3）
tests/test_train_detector.py            修改（Task 4：导出加 manifest；加拒绝旧导出的测试）
tests/test_train_reliability.py         修改（Task 4：_minimal_export 加 manifest）
tests/test_relink_fastmri_knee_gate0.py 新建（Task 5）
tests/test_audit_fastmri_plus_boxes.py  新建（Task 6）
tests/test_fastmri_knee_detection_eval.py 新建（Task 9）
tests/test_geometry.py                  新建（Task 10）
tests/test_brain_lookup.py              新建（Task 10）
tests/test_brain_frame.py               新建（Task 11）
docs/verification/2026-09-24/{gate0,h1_rerun,gate05}/  产物
docs/verification/2026-09-24/REPORT.md  总报告
```

---

### Task 0: 工作树就位

**Files:** 无代码改动。

- [ ] **Step 1: 把 build/aur-system 快进到 main**

```bash
cd /data0/congcong/code/Project_Doing/foundation_model-aur
git status -sb            # 期望：## build/aur-system，只有 ?? logs/
git merge --ff-only main  # 期望：Fast-forward 到 7a889ed
git log --oneline -1      # 期望：7a889ed ... Merge branch 'build/aur-system': knee capability system (plan 1)
mkdir -p logs
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider | tail -1   # 期望 342 passed
```

---

### Task 1: 框约定与物理坐标（`fastmri.py`）

**Files:**
- Modify: `anatobind/data_engine/fastmri.py`（在 `PAD_TO_SLICES` 之后加一段）
- Test: `tests/test_fastmri_box_convention.py`

**Interfaces:**
- Produces: `TRANSFORM_VERSION = 2`、`BOX_CONVENTION_CSV`、`BOX_CONVENTION_RSS`、`MIN_BOX_SIDE = 3`；`convert_box_csv_to_rss(x, y, width, height, n_rows) -> (row0, row1, col0, col1)`（半开）；`convert_box_rss_to_csv(row0, row1, col0, col1, n_rows) -> (x, y, width, height)`；`rss_spacing_mm(header_xml) -> (slice_mm, row_mm, col_mm)`；`voxel_to_world(index, spacing) -> tuple`；`world_to_voxel(point_mm, spacing) -> tuple`。

- [ ] **Step 1: 写测试**

```python
# tests/test_fastmri_box_convention.py
import numpy as np
import pytest

from anatobind.data_engine.fastmri import (
    BOX_CONVENTION_RSS, TRANSFORM_VERSION, convert_box_csv_to_rss, convert_box_rss_to_csv, rss_spacing_mm,
    voxel_to_world, world_to_voxel,
)

# a knee header: 640 x 368 acquired over 280 x 161.42 mm, reconSpace 320 x 320 over 140 mm, 3 mm slices
HEADER = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding>'
    "<encodedSpace><matrixSize><x>640</x><y>368</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>280</x><y>161.42</y><z>4.5</z></fieldOfView_mm></encodedSpace>"
    "<reconSpace><matrixSize><x>320</x><y>320</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>140</x><y>140</y><z>3</z></fieldOfView_mm></reconSpace>"
    "</encoding></ismrmrdHeader>"
)


def _label_as_fastmri_plus_did(rss_slice):
    """What the annotator saw was the RSS slice flipped up/down; the CSV box is drawn around the bright block
    in that flipped image: (x, y, width, height) with y counted from the top of the FLIPPED image."""
    flipped = rss_slice[::-1]
    rows, cols = np.nonzero(flipped > 0)
    return int(cols.min()), int(rows.min()), int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)


def test_known_bright_block_is_covered_exactly_after_conversion():
    n_rows = 320
    img = np.zeros((n_rows, 320), np.float32)
    row0, row1, col0, col1 = 40, 61, 200, 233               # the block lives in rows 40..60 of the RSS array
    img[row0:row1, col0:col1] = 1.0
    x, y, w, h = _label_as_fastmri_plus_did(img)
    assert (y, h) == (n_rows - row1, row1 - row0)           # i.e. the CSV y is measured from the bottom
    assert convert_box_csv_to_rss(x, y, w, h, n_rows) == (row0, row1, col0, col1)
    r0, r1, c0, c1 = convert_box_csv_to_rss(x, y, w, h, n_rows)
    assert img[r0:r1, c0:c1].all() and img.sum() == (r1 - r0) * (c1 - c0)


def test_using_the_csv_box_as_is_misses_the_block():
    img = np.zeros((320, 320), np.float32)
    img[40:61, 200:233] = 1.0
    x, y, w, h = _label_as_fastmri_plus_did(img)
    assert img[y:y + h, x:x + w].sum() == 0                 # what leg 2 trained on before Gate 0


@pytest.mark.parametrize("n_rows", [320, 276, 260, 213])
def test_csv_to_rss_to_csv_round_trip_is_exact(n_rows):
    rng = np.random.default_rng(n_rows)
    for _ in range(200):
        y = int(rng.integers(0, n_rows - 3))
        x = int(rng.integers(0, 300))
        h = int(rng.integers(3, n_rows - y + 1))
        w = int(rng.integers(3, 40))
        assert convert_box_rss_to_csv(*convert_box_csv_to_rss(x, y, w, h, n_rows), n_rows) == (x, y, w, h)


def test_rss_spacing_is_acquired_resolution_in_plane_and_recon_fov_z_through_plane():
    sp = rss_spacing_mm(HEADER)
    assert sp[0] == pytest.approx(3.0)
    assert sp[1] == pytest.approx(280 / 640)
    assert sp[2] == pytest.approx(161.42 / 368)


def test_voxel_and_world_are_index_times_spacing_and_invert_each_other():
    sp = (3.0, 0.4375, 0.4386)
    assert voxel_to_world((2, 10, 100), sp) == pytest.approx((6.0, 4.375, 43.86))
    assert world_to_voxel(voxel_to_world((2, 10, 100), sp), sp) == pytest.approx((2.0, 10.0, 100.0))


def test_the_convention_constants_are_what_the_manifest_will_record():
    assert TRANSFORM_VERSION == 2
    assert BOX_CONVENTION_RSS == "rss_rows_from_top"
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_fastmri_box_convention.py -q -p no:cacheprovider`
Expected: ImportError（`convert_box_csv_to_rss` 不存在）。

- [ ] **Step 3: 实现**

在 `anatobind/data_engine/fastmri.py` 的 `PAD_TO_SLICES = 12` 之后加：

```python
# --- fastMRI+ box convention (Gate 0; v2.6 §4.1) ---------------------------------------------------------------
# The fastMRI+ README: "In the process of converting the images to DICOM, the pixel arrays were flipped (up/down)
# to provide a view that was closer to DICOM orientation and assist with labeling." So the CSV y counts rows from
# the BOTTOM of the reconstruction_rss array. Every export from Gate 0 on stores rows counted from the top, and
# says so in its manifest (transform_version 2). Verified on 24 brain and 30 knee volumes on 2026-09-15
# (docs/verification/2026-09-16-brain-probe/REPORT.md §2) and on every annotated volume by
# scripts/audit_fastmri_plus_boxes.py.
BOX_CONVENTION_CSV = "fastmri_plus_csv_rows_from_bottom"
BOX_CONVENTION_RSS = "rss_rows_from_top"
TRANSFORM_VERSION = 2        # 1 = CSV boxes used as-is (leg 2 before 2026-09-24), 2 = convert_box_csv_to_rss applied
MIN_BOX_SIDE = 3


def convert_box_csv_to_rss(x, y, width, height, n_rows):
    """fastMRI+ CSV box -> half-open box on the RSS array, (row0, row1, col0, col1), rows counted from the top."""
    return n_rows - y - height, n_rows - y, x, x + width


def convert_box_rss_to_csv(row0, row1, col0, col1, n_rows):
    """Inverse of convert_box_csv_to_rss: (x, y, width, height) as fastMRI+ would have written it."""
    return col0, n_rows - row1, col1 - col0, row1 - row0


def rss_spacing_mm(header_xml):
    """(slice, row, col) voxel size in mm of the stored RSS: acquired resolution in plane (encodedSpace
    FOV / matrix, readout oversampling already folded in), reconSpace fov z through plane. Same rule as
    rss_h5_to_nifti."""
    g = parse_recon_geometry(header_xml)
    return (g["fov_z_mm"], g["enc_fov_x_mm"] / g["enc_nx"], g["enc_fov_y_mm"] / g["enc_ny"])


def voxel_to_world(index, spacing):
    """(slice, row, col) index -> mm. fastMRI h5 files carry no patient position, so the array origin is 0 mm and
    this is the only physical frame the data has."""
    return tuple(float(i) * float(s) for i, s in zip(index, spacing))


def world_to_voxel(point_mm, spacing):
    return tuple(float(p) / float(s) for p, s in zip(point_mm, spacing))
```

- [ ] **Step 4: 跑测试，确认通过**

Run: 同 Step 2。Expected: 9 passed。

- [ ] **Step 5: 提交**

```bash
git add anatobind/data_engine/fastmri.py tests/test_fastmri_box_convention.py
git commit -m "Data engine: fastMRI+ box convention, CSV to RSS conversion and physical spacing helpers"
```

---

### Task 2: fastMRI+ 通用读框与 3D 合并；膝模块委托

**Files:**
- Modify: `anatobind/data_engine/fastmri.py`（加 `read_fastmri_plus_rows`、`rows_to_rss_frame`、`box_iou_2d`、`merge_boxes_3d`）
- Modify: `anatobind/data_engine/fastmri_knee.py:39-111`（`read_annotations`、`_iou`、`merge_to_3d`）
- Test: `tests/test_fastmri_plus_rows.py`

**Interfaces:**
- Consumes: Task 1 的 `convert_box_csv_to_rss`。
- Produces: `read_fastmri_plus_rows(csv_path) -> list[dict(file, slice, x, y, width, height, label)]`（CSV 帧）；`rows_to_rss_frame(rows, n_rows_of) -> list[dict]`（同键，RSS 帧；`n_rows_of` 是 int 或 `{file: int}`）；`box_iou_2d(a, b)`；`merge_boxes_3d(rows, group_key, iou_min=0.3) -> list[dict(file, <group_key>, z0, z1, x0, y0, x1, y1, n_boxes, members)]`。膝：`read_annotations` 行多一个 `label` 键；`merge_to_3d` 输出多一个 `members` 键，其余不变。

- [ ] **Step 1: 写测试**

```python
# tests/test_fastmri_plus_rows.py
import csv

import numpy as np

from anatobind.data_engine.fastmri import merge_boxes_3d, read_fastmri_plus_rows, rows_to_rss_frame
from anatobind.data_engine.fastmri_knee import merge_to_3d, read_annotations

FIELDS = ["file", "slice", "study_level", "x", "y", "width", "height", "label"]


def _write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _csv_row(file, s, x, y, w, h, label, study="No"):
    return {"file": file, "slice": s, "study_level": study, "x": x, "y": y, "width": w, "height": h, "label": label}


def test_reader_strips_labels_and_skips_study_level_and_non_integer_rows(tmp_path):
    p = tmp_path / "knee.csv"
    _write_csv(p, [_csv_row("f1", 3, 10, 20, 5, 6, "Joint Effusion "),
                   _csv_row("f1", "", "", "", "", "", "artifact", study="Yes"),
                   _csv_row("f1", 4, "n/a", 1, 2, 2, "Meniscus Tear")])
    assert read_fastmri_plus_rows(p) == [
        {"file": "f1", "slice": 3, "x": 10, "y": 20, "width": 5, "height": 6, "label": "Joint Effusion"}]


def test_knee_reader_maps_labels_to_families_and_drops_the_rest(tmp_path):
    p = tmp_path / "knee.csv"
    _write_csv(p, [_csv_row("f1", 3, 1, 2, 5, 6, "Joint Effusion "), _csv_row("f1", 3, 1, 2, 5, 6, "Periarticular cysts")])
    rows = read_annotations(p)
    assert [(r["family"], r["label"]) for r in rows] == [("effusion", "Joint Effusion")]


def test_rows_to_rss_frame_flips_rows_per_file_and_keeps_width_and_height():
    rows = [{"file": "a", "slice": 0, "x": 10, "y": 20, "width": 5, "height": 6, "label": "L"},
            {"file": "b", "slice": 0, "x": 10, "y": 20, "width": 5, "height": 6, "label": "L"}]
    out = rows_to_rss_frame(rows, {"a": 320, "b": 276})
    assert (out[0]["x"], out[0]["y"], out[0]["width"], out[0]["height"]) == (10, 320 - 26, 5, 6)
    assert (out[1]["x"], out[1]["y"]) == (10, 276 - 26)
    assert rows_to_rss_frame(rows[:1], 320) == out[:1]
    assert rows[0]["y"] == 20                                    # input rows are not mutated


def _row(file, s, x, y, w, h, label="L"):
    return {"file": file, "slice": s, "x": x, "y": y, "width": w, "height": h, "label": label}


def test_merge_keeps_members_and_groups_by_the_given_key():
    rows = [_row("f", 3, 10, 10, 20, 20), _row("f", 4, 11, 10, 20, 20), _row("f", 4, 11, 10, 20, 20, "M")]
    lesions = merge_boxes_3d(rows, "label")
    assert sorted((L["label"], L["n_boxes"]) for L in lesions) == [("L", 2), ("M", 1)]
    L = next(L for L in lesions if L["label"] == "L")
    assert (L["z0"], L["z1"], L["x0"], L["y0"], L["x1"], L["y1"]) == (3, 4, 10, 10, 31, 30)
    assert [m["slice"] for m in L["members"]] == [3, 4]


def test_flipping_rows_commutes_with_merging():
    """A row flip is a reflection: in-plane IoU and slice adjacency are unchanged, so
    merge(flip(rows)) must equal flip(merge(rows)) box for box."""
    rng = np.random.default_rng(0)
    rows = [_row("f", int(rng.integers(0, 6)), int(rng.integers(0, 300)), int(rng.integers(0, 300)),
                 int(rng.integers(3, 20)), int(rng.integers(3, 20)), str(rng.choice(["L", "M"]))) for _ in range(120)]
    n = 320
    merged_after_flip = merge_boxes_3d(rows_to_rss_frame(rows, n), "label")
    merged_before_flip = merge_boxes_3d(rows, "label")

    def key(L):
        return (L["label"], L["z0"], L["z1"], L["x0"], L["x1"], L["n_boxes"])

    assert sorted((key(L), L["y0"], L["y1"]) for L in merged_after_flip) == \
        sorted((key(L), n - L["y1"], n - L["y0"]) for L in merged_before_flip)


def test_knee_merge_to_3d_is_the_generic_merge_keyed_by_family():
    rows = [{**_row("f", 3, 10, 10, 20, 20), "family": "meniscus"}, {**_row("f", 4, 10, 10, 20, 20), "family": "meniscus"}]
    lesions = merge_to_3d(rows)
    assert [L["family"] for L in lesions] == ["meniscus"] and lesions[0]["n_boxes"] == 2
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_fastmri_plus_rows.py -q -p no:cacheprovider`
Expected: ImportError（`read_fastmri_plus_rows` 不存在）。

- [ ] **Step 3: 实现通用函数**

`anatobind/data_engine/fastmri.py` 顶部加 `import csv` 与 `from collections import defaultdict`；在 Task 1 那段之后加：

```python
def read_fastmri_plus_rows(csv_path):
    """fastMRI+ CSV -> per-slice boxes in the CSV frame, one dict per row: file, slice, x, y, width, height, label.
    Study-level rows carry no box and are skipped, as are rows whose geometry is not integer; labels are stripped
    (the file has "Joint Effusion " with a trailing space)."""
    out = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["study_level"].strip() == "Yes":
                continue
            try:
                out.append({"file": r["file"], "slice": int(r["slice"]), "x": int(r["x"]), "y": int(r["y"]),
                            "width": int(r["width"]), "height": int(r["height"]), "label": r["label"].strip()})
            except ValueError:
                continue
    return out


def rows_to_rss_frame(rows, n_rows_of):
    """CSV-frame rows -> RSS-frame rows (x = col0, y = row0 counted from the top; width/height unchanged).
    n_rows_of: the row count of each file's reconstruction_rss, an int (same for every file) or {file: int}."""
    out = []
    for r in rows:
        n = n_rows_of[r["file"]] if isinstance(n_rows_of, dict) else int(n_rows_of)
        row0, row1, col0, col1 = convert_box_csv_to_rss(r["x"], r["y"], r["width"], r["height"], n)
        out.append({**r, "x": col0, "y": row0, "width": col1 - col0, "height": row1 - row0})
    return out


def box_iou_2d(a, b):
    """IoU of two boxes given as dicts with x, y, width, height (any frame, as long as both share it)."""
    ax1, ay1 = a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1 = b["x"] + b["width"], b["y"] + b["height"]
    iw = max(0, min(ax1, bx1) - max(a["x"], b["x"]))
    ih = max(0, min(ay1, by1) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union else 0.0


def merge_boxes_3d(rows, group_key, iou_min=0.3):
    """Boxes of one (file, group_key) on adjacent slices with in-plane IoU >= iou_min form one 3D lesion (the leg 2
    rule). Each lesion keeps its member rows and the enclosing box: x0/y0 inclusive, x1/y1 exclusive, z0..z1
    inclusive slice indices."""
    lesions = []
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["file"], r[group_key])].append(r)
    for (file, group), boxes in sorted(by_group.items()):
        parent = list(range(len(boxes)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, a in enumerate(boxes):
            for j, b in enumerate(boxes):
                if j <= i or abs(a["slice"] - b["slice"]) != 1:
                    continue
                if box_iou_2d(a, b) >= iou_min:
                    parent[find(i)] = find(j)
        comps = defaultdict(list)
        for i in range(len(boxes)):
            comps[find(i)].append(boxes[i])
        for members in comps.values():
            lesions.append({
                "file": file, group_key: group,
                "z0": min(m["slice"] for m in members), "z1": max(m["slice"] for m in members),
                "x0": min(m["x"] for m in members), "y0": min(m["y"] for m in members),
                "x1": max(m["x"] + m["width"] for m in members), "y1": max(m["y"] + m["height"] for m in members),
                "n_boxes": len(members), "members": members,
            })
    return lesions
```

- [ ] **Step 4: 膝模块委托**

`anatobind/data_engine/fastmri_knee.py`：在 import 区加 `from anatobind.data_engine.fastmri import box_iou_2d, merge_boxes_3d, read_fastmri_plus_rows`；把第 39–55 行的 `read_annotations`、第 68–75 行的 `_iou`、第 78–111 行的 `merge_to_3d` 整体换成：

```python
def read_annotations(csv_path):
    """CSV rows -> rows with a family (labels outside FAMILY_OF_LABEL are dropped). Still in the CSV frame:
    the row flip happens once, in rows_to_rss_frame, before merging."""
    out = []
    for r in read_fastmri_plus_rows(csv_path):
        fam = FAMILY_OF_LABEL.get(r["label"])
        if fam is not None:
            out.append({**r, "family": fam})
    return out


_iou = box_iou_2d       # old name, still imported by the probe scripts under docs/verification


def merge_to_3d(rows, iou_min=0.3):
    """Adjacent-slice boxes of one family with in-plane IoU >= iou_min are one lesion (fastmri.merge_boxes_3d)."""
    return merge_boxes_3d(rows, "family", iou_min)
```

`import csv` 与 `from collections import Counter, defaultdict` 里不再用的名字（`defaultdict`）删掉；`csv` 在 Task 3 还要用，保留。

- [ ] **Step 5: 跑测试，确认通过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_fastmri_plus_rows.py tests/test_fastmri_knee_boxes.py tests/test_fastmri_knee_export.py -q -p no:cacheprovider`
Expected: 全过（旧的膝测试 6 + 5 个也过）。

- [ ] **Step 6: 提交**

```bash
git add anatobind/data_engine/fastmri.py anatobind/data_engine/fastmri_knee.py tests/test_fastmri_plus_rows.py
git commit -m "Data engine: shared fastMRI+ row reader, RSS-frame conversion and 3D merge; knee delegates to them"
```

---

### Task 3: manifest 约定、旧导出拒载、患者折断言（`fastmri_knee.py`）

**Files:**
- Modify: `anatobind/data_engine/fastmri_knee.py`（`EXPORT_ROOT` 一段与 `patient_of`）
- Test: `tests/test_fastmri_knee_manifest.py`

**Interfaces:**
- Consumes: Task 1 的 `TRANSFORM_VERSION`、`BOX_CONVENTION_RSS`、`rss_spacing_mm`。
- Produces: `EXPORT_ROOT`（新根）、`LEGACY_EXPORT_ROOT`、`IMAGE_ORIENTATION`、`MANIFEST_FIELDS`、`LESION_FIELDS`、`class LegacyBoxConvention(ValueError)`、`volume_geometry(h5_path) -> dict(patient_id, n_rows, n_cols, slices, spacing_slice_mm, spacing_row_mm, spacing_col_mm)`、`manifest_row(name, h5_path, out_dir, n_lesions, status) -> dict`、`write_manifest(path, rows)`、`write_lesions(path, lesions)`、`load_manifest(export_root) -> {file: row}`（旧导出抛 `LegacyBoxConvention`）、`load_lesions(export_root) -> list[dict]`（整数已转）、`assert_folds_by_patient(folds, patients)`、`load_folds(export_root) -> {file: fold}`；`patient_of(paths)` 缺 `patient_id` 时抛 `KeyError`。

- [ ] **Step 1: 写测试**

```python
# tests/test_fastmri_knee_manifest.py
import csv
import json

import h5py
import numpy as np
import pytest

from anatobind.data_engine.fastmri import TRANSFORM_VERSION
from anatobind.data_engine.fastmri_knee import (
    LESION_FIELDS, MANIFEST_FIELDS, LegacyBoxConvention, assert_folds_by_patient, load_folds, load_lesions,
    load_manifest, manifest_row, patient_of, write_lesions, write_manifest,
)

HEADER = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding>'
    "<encodedSpace><matrixSize><x>640</x><y>368</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>280</x><y>161.42</y><z>4.5</z></fieldOfView_mm></encodedSpace>"
    "<reconSpace><matrixSize><x>320</x><y>320</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>140</x><y>140</y><z>3</z></fieldOfView_mm></reconSpace>"
    "</encoding></ismrmrdHeader>"
)


def _write_h5(path, patient="p1", rows=320, cols=320, slices=3, with_patient=True):
    with h5py.File(path, "w") as h:
        h.create_dataset("kspace", data=np.zeros((slices, 2, rows * 2, cols + 48), np.complex64))
        h.create_dataset("reconstruction_rss", data=np.zeros((slices, rows, cols), np.float32))
        h.create_dataset("ismrmrd_header", data=np.bytes_(HEADER.encode()))
        if with_patient:
            h.attrs["patient_id"] = patient
    return path


def _export(root, files, patients, version=TRANSFORM_VERSION):
    root.mkdir(parents=True, exist_ok=True)
    write_manifest(root / "manifest.csv",
                   [{"file": f, "patient_id": p, "transform_version": version, "status": "ok"} for f, p in zip(files, patients)])
    write_lesions(root / "lesions.csv", [{"file": files[0], "family": "meniscus", "z0": 1, "z1": 2,
                                          "x0": 4, "y0": 5, "x1": 12, "y1": 15, "n_boxes": 2, "members": []}])
    (root / "folds.json").write_text(json.dumps({"folds": {f: i % 2 for i, f in enumerate(files)}}))


def test_manifest_row_records_patient_grid_spacing_and_convention(tmp_path):
    h5 = _write_h5(tmp_path / "file1.h5")
    row = manifest_row("file1", h5, tmp_path / "out", n_lesions=2, status="ok")
    assert row["patient_id"] == "p1" and row["n_rows"] == 320 and row["n_cols"] == 320 and row["slices"] == 3
    assert row["spacing_slice_mm"] == pytest.approx(3.0) and row["spacing_row_mm"] == pytest.approx(0.4375)
    assert row["transform_version"] == TRANSFORM_VERSION and row["box_coordinate_convention"] == "rss_rows_from_top"
    assert set(row) == set(MANIFEST_FIELDS)


def test_write_lesions_numbers_them_and_drops_members(tmp_path):
    write_lesions(tmp_path / "lesions.csv", [
        {"file": "b", "family": "meniscus", "z0": 1, "z1": 2, "x0": 4, "y0": 5, "x1": 12, "y1": 15, "n_boxes": 2, "members": [1]},
        {"file": "a", "family": "effusion", "z0": 0, "z1": 0, "x0": 1, "y0": 1, "x1": 5, "y1": 5, "n_boxes": 1, "members": [1]}])
    with open(tmp_path / "lesions.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == LESION_FIELDS
    assert [(r["lesion_id"], r["file"]) for r in rows] == [("0", "a"), ("1", "b")]


def test_a_manifest_without_transform_version_is_refused(tmp_path):
    root = tmp_path / "leg2"
    root.mkdir()
    (root / "manifest.csv").write_text("file,out_dir,slices,n_lesions,status\nfile1,x,3,1,ok\n")   # the 2026-09-14 layout
    with pytest.raises(LegacyBoxConvention, match="transform_version"):
        load_manifest(root)


def test_version_one_is_refused_and_version_two_loads_with_integers(tmp_path):
    _export(tmp_path / "v1", ["file1"], ["p1"], version=1)
    with pytest.raises(LegacyBoxConvention):
        load_lesions(tmp_path / "v1")
    _export(tmp_path / "v2", ["file1"], ["p1"])
    lesions = load_lesions(tmp_path / "v2")
    assert lesions[0]["y0"] == 5 and isinstance(lesions[0]["z1"], int) and lesions[0]["n_boxes"] == 2


def test_folds_that_split_a_patient_are_rejected():
    with pytest.raises(ValueError, match="patient-disjoint"):
        assert_folds_by_patient({"a": 0, "b": 1}, {"a": "p", "b": "p"})
    assert_folds_by_patient({"a": 0, "b": 0, "c": 1}, {"a": "p", "b": "p", "c": "q"})   # no raise


def test_load_folds_checks_patients_from_the_manifest(tmp_path):
    _export(tmp_path / "ok", ["file1", "file2"], ["p1", "p2"])
    assert load_folds(tmp_path / "ok") == {"file1": 0, "file2": 1}
    _export(tmp_path / "bad", ["file1", "file2"], ["p1", "p1"])          # one patient, two folds
    with pytest.raises(ValueError, match="patient-disjoint"):
        load_folds(tmp_path / "bad")


def test_patient_of_refuses_a_file_without_patient_id(tmp_path):
    with_id = _write_h5(tmp_path / "a.h5", patient="pa")
    without = _write_h5(tmp_path / "b.h5", with_patient=False)
    assert patient_of({"a": with_id}) == {"a": "pa"}
    with pytest.raises(KeyError, match="patient_id"):
        patient_of({"b": without})
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_fastmri_knee_manifest.py -q -p no:cacheprovider`
Expected: ImportError（`LegacyBoxConvention` 等不存在）。

- [ ] **Step 3: 实现**

`anatobind/data_engine/fastmri_knee.py`：import 区补 `from anatobind.data_engine.fastmri import BOX_CONVENTION_RSS, TRANSFORM_VERSION, rss_spacing_mm`（与 Task 2 的 import 合并成一行）。把第 168–182 行（`KSPACE_ROOT` … `patient_of`）换成：

```python
KSPACE_ROOT = Path("/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/knee")
ANNOTATIONS = Path("/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/Annotations/knee.csv")
EXPORT_ROOT = Path("/data2/congcong/data/FM_data/derived/fastmri_knee/leg2_gate0")     # boxes in the RSS frame (Gate 0)
LEGACY_EXPORT_ROOT = Path("/data2/congcong/data/FM_data/derived/fastmri_knee/leg2")    # 2026-09-14, boxes as-is: read-only history
IMAGE_ORIENTATION = "reconstruction_rss order (slice, row, col); rows counted from the top"
MANIFEST_FIELDS = ["file", "out_dir", "slices", "n_lesions", "status", "patient_id", "n_rows", "n_cols",
                   "spacing_slice_mm", "spacing_row_mm", "spacing_col_mm", "image_orientation",
                   "box_coordinate_convention", "transform_version"]
LESION_FIELDS = ["lesion_id", "file", "family", "z0", "z1", "x0", "y0", "x1", "y1", "n_boxes"]


class LegacyBoxConvention(ValueError):
    """The export was written before Gate 0 (CSV rows used as-is, i.e. mirrored boxes); refuse to use it."""


def volume_paths(root=KSPACE_ROOT):
    return {p.stem: p for split in ("multicoil_train", "multicoil_val") for p in sorted((Path(root) / split).glob("*.h5"))}


def _patient_attr(h, path):
    if "patient_id" not in h.attrs:
        raise KeyError(f"{path}: no patient_id attribute; folds are split by patient (v2.6 §4.4)")
    pid = h.attrs["patient_id"]
    return pid.decode() if isinstance(pid, bytes) else str(pid)


def patient_of(paths):
    out = {}
    for name, p in paths.items():
        with h5py.File(p) as h:
            out[name] = _patient_attr(h, p)
    return out


def volume_geometry(h5_path):
    """patient_id, RSS grid and (slice, row, col) spacing of one fastMRI h5 file."""
    with h5py.File(h5_path) as h:
        pid = _patient_attr(h, h5_path)
        hdr = h["ismrmrd_header"][()]
        n_slices, n_rows, n_cols = h["reconstruction_rss"].shape
    sp = rss_spacing_mm(hdr.decode() if isinstance(hdr, bytes) else hdr)
    return {"patient_id": pid, "n_rows": int(n_rows), "n_cols": int(n_cols), "slices": int(n_slices),
            "spacing_slice_mm": sp[0], "spacing_row_mm": sp[1], "spacing_col_mm": sp[2]}


def manifest_row(name, h5_path, out_dir, n_lesions, status):
    g = volume_geometry(h5_path)
    return {"file": name, "out_dir": str(out_dir), "slices": g["slices"], "n_lesions": n_lesions, "status": status,
            "patient_id": g["patient_id"], "n_rows": g["n_rows"], "n_cols": g["n_cols"],
            "spacing_slice_mm": g["spacing_slice_mm"], "spacing_row_mm": g["spacing_row_mm"],
            "spacing_col_mm": g["spacing_col_mm"], "image_orientation": IMAGE_ORIENTATION,
            "box_coordinate_convention": BOX_CONVENTION_RSS, "transform_version": TRANSFORM_VERSION}


def write_manifest(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_lesions(path, lesions):
    """lesions.csv: one row per 3D lesion, numbered in (file, family, z0, x0) order; members are not written."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LESION_FIELDS, extrasaction="ignore")
        w.writeheader()
        for i, L in enumerate(sorted(lesions, key=lambda L: (L["file"], L["family"], L["z0"], L["x0"]))):
            w.writerow({"lesion_id": i, **L})


def load_manifest(export_root):
    """manifest.csv -> {file: row}; refuses exports whose boxes are not transform_version 2."""
    path = Path(export_root) / "manifest.csv"
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    bad = [r["file"] for r in rows if str(r.get("transform_version", "")) != str(TRANSFORM_VERSION)]
    if bad:
        raise LegacyBoxConvention(
            f"{path}: {len(bad)} of {len(rows)} volumes lack transform_version {TRANSFORM_VERSION} (first {bad[:3]}); "
            f"their boxes are the mirrored CSV boxes of leg 2 before Gate 0. Build a converted root with "
            f"scripts/relink_fastmri_knee_gate0.py.")
    return {r["file"]: r for r in rows}


def load_lesions(export_root):
    """lesions.csv of a Gate-0 export (RSS frame) with integers parsed; refuses legacy exports."""
    load_manifest(export_root)
    with open(Path(export_root) / "lesions.csv", newline="") as fh:
        return [{**r, **{k: int(r[k]) for k in ("z0", "z1", "x0", "y0", "x1", "y1", "n_boxes")}}
                for r in csv.DictReader(fh)]


def assert_folds_by_patient(folds, patients):
    """Raise if any patient has volumes in two folds. folds: {file: fold}; patients: {file: patient_id}."""
    fold_of_patient = {}
    for vol, fold in folds.items():
        p = patients[vol]
        if fold_of_patient.setdefault(p, fold) != fold:
            raise ValueError(f"patient {p} is in fold {fold_of_patient[p]} and fold {fold} ({vol}); "
                             f"folds must be patient-disjoint (v2.6 §4.4)")


def load_folds(export_root):
    """folds.json -> {file: fold}, asserted patient-disjoint against the manifest's patient_id column."""
    folds = json.loads((Path(export_root) / "folds.json").read_text())["folds"]
    manifest = load_manifest(export_root)
    assert_folds_by_patient(folds, {f: manifest[f]["patient_id"] for f in folds})
    return folds
```

`make_folds`、`_view_seed`、`export_volume` 原样保留在后面。

- [ ] **Step 4: 跑测试，确认通过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_fastmri_knee_manifest.py tests/test_fastmri_knee_export.py -q -p no:cacheprovider`
Expected: 全过。

- [ ] **Step 5: 提交**

```bash
git add anatobind/data_engine/fastmri_knee.py tests/test_fastmri_knee_manifest.py
git commit -m "Data engine: export manifest with box convention and spacing, legacy-export refusal, patient-disjoint folds"
```

---

### Task 4: 消费者改走新加载器（训练、H1 门、可靠性、缓存、导出脚本）

**Files:**
- Modify: `anatobind/train/train_detector.py:16-27,65-72`
- Modify: `scripts/check_h1.py:72-78`
- Modify: `scripts/train_reliability.py:31-37`
- Modify: `scripts/cache_detections.py:23-42`
- Modify: `scripts/build_fastmri_knee.py`
- Modify: `tests/test_train_detector.py:11-27`、`tests/test_train_reliability.py:21-29`

**Interfaces:**
- Consumes: Task 3 的 `load_folds`、`load_lesions`、`manifest_row`、`write_manifest`、`write_lesions`、`volume_geometry`；Task 2 的 `rows_to_rss_frame`。
- Produces: `load_fold(export_root, fold) -> (train, held, lesions)` 签名不变；`cache_detections.py --score-min`；`build_fastmri_knee.py` 产出 RSS 帧 lesions.csv + 新 manifest。

- [ ] **Step 1: 改测试里的导出夹具（先让它们代表新格式）**

`tests/test_train_detector.py` 的 `_export` 改成：

```python
def _export(root, files, slices=9, size=32, version=2):
    from anatobind.data_engine.fastmri_knee import write_lesions, write_manifest
    rng = np.random.default_rng(0)
    lesions, manifest = [], []
    for i, name in enumerate(files):
        d = root / name
        d.mkdir(parents=True)
        for v in VIEWS:
            np.save(d / f"{v}.npy", rng.normal(size=(slices, size, size)).astype(np.float16))
        (d / "meta.json").write_text(json.dumps({"slices": slices, "size": size, "patient_id": f"p{i}"}))
        lesions.append({"file": name, "family": "meniscus", "z0": 3, "z1": 5, "x0": 4, "y0": 6, "x1": 14, "y1": 18, "n_boxes": 3})
        manifest.append({"file": name, "out_dir": str(d), "slices": slices, "n_lesions": 1, "status": "ok",
                         "patient_id": f"p{i}", "n_rows": size, "n_cols": size, "transform_version": version})
    write_lesions(root / "lesions.csv", lesions)
    write_manifest(root / "manifest.csv", manifest)
    (root / "folds.json").write_text(json.dumps({"folds": {f: i % 5 for i, f in enumerate(files)}}))
```

并在文件末尾加：

```python
def test_load_fold_refuses_an_export_written_before_gate_0(tmp_path):
    from anatobind.data_engine.fastmri_knee import LegacyBoxConvention
    from anatobind.train.train_detector import load_fold
    root = tmp_path / "leg2"
    root.mkdir()
    _export(root, [f"file{i}" for i in range(5)], version=1)
    with pytest.raises(LegacyBoxConvention):
        load_fold(root, 0)
```

`tests/test_train_reliability.py` 的 `_minimal_export` 改成：

```python
def _minimal_export(root, patient_of):
    """patient_of: {file_name: patient_id}. Writes meta.json per file, a header-only lesions.csv, a Gate-0
    manifest and folds.json putting every file in fold 0."""
    from anatobind.data_engine.fastmri_knee import write_manifest
    root.mkdir(parents=True, exist_ok=True)
    for f, pid in patient_of.items():
        (root / f).mkdir()
        (root / f / "meta.json").write_text(json.dumps({"patient_id": pid}))
    (root / "lesions.csv").write_text("lesion_id,file,family,z0,z1,x0,y0,x1,y1,n_boxes\n")
    write_manifest(root / "manifest.csv",
                   [{"file": f, "patient_id": pid, "transform_version": 2, "status": "ok"} for f, pid in patient_of.items()])
    (root / "folds.json").write_text(json.dumps({"folds": {f: 0 for f in patient_of}}))
```

- [ ] **Step 2: 跑这两个测试文件，确认新测试失败、旧测试仍过**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_train_detector.py tests/test_train_reliability.py -q -p no:cacheprovider`
Expected: `test_load_fold_refuses_an_export_written_before_gate_0` FAIL（没抛异常），其余过。

- [ ] **Step 3: 改 `load_fold`**

`anatobind/train/train_detector.py`：把 `import csv` 删掉（`json` 仍用），import 改为 `from anatobind.data_engine.fastmri_knee import EXPORT_ROOT, load_folds, load_lesions`，`load_fold` 换成：

```python
def load_fold(export_root, fold):
    """Training and held-out files of one patient-disjoint fold plus the RSS-frame lesions; refuses exports
    written before Gate 0 (anatobind.data_engine.fastmri_knee.LegacyBoxConvention)."""
    folds = load_folds(export_root)
    train = sorted(f for f, k in folds.items() if k != fold)
    held = sorted(f for f, k in folds.items() if k == fold)
    return train, held, load_lesions(export_root)
```

- [ ] **Step 4: 改 `check_h1.py` 与 `train_reliability.py` 的读框**

`scripts/check_h1.py`：import 里加 `load_lesions`（`from anatobind.data_engine.fastmri_knee import EXPORT_ROOT, VIEWS, load_lesions`），`gather` 开头第 74–78 行换成：

```python
    by_file = {}
    for r in load_lesions(export_root):
        by_file.setdefault(r["file"], []).append(r)
```

`scripts/train_reliability.py` 第 31–37 行的 `lesions_by_file` 换成：

```python
def lesions_by_file(export_root):
    out = {}
    for r in load_lesions(export_root):
        out.setdefault(r["file"], []).append(r)
    return out
```

并在其 import 区加 `load_lesions`（`from anatobind.data_engine.fastmri_knee import EXPORT_ROOT, VIEWS, load_lesions`）；`csv` 若因此不再使用则删掉该 import。

- [ ] **Step 5: `cache_detections.py` 加 `--score-min`**

```python
    ap.add_argument("--score-min", type=float, default=0.05,
                    help="lowest per-slice score kept in the cache (0.01 for the Gate 0 rerun so the FROC sweep has room)")
```

并把 `dets, gfeat = detect_volume(model, vol, device)` 改为 `dets, gfeat = detect_volume(model, vol, device, score_min=a.score_min)`。

- [ ] **Step 6: `build_fastmri_knee.py` 写新格式**

整个 `main` 换成（`LESION_FIELDS`/`MANIFEST_FIELDS` 常量从脚本删掉，改从包导入）：

```python
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import rows_to_rss_frame  # noqa: E402
from anatobind.data_engine.fastmri_knee import (  # noqa: E402
    ANNOTATIONS, EXPORT_ROOT, assert_folds_by_patient, clean_boxes, export_volume, make_folds, manifest_row,
    merge_to_3d, read_annotations, volume_geometry, volume_paths, write_lesions, write_manifest,
)


def job(args):
    name, path, lesions, out_root, seed = args
    os.nice(19)
    return export_volume(path, lesions, Path(out_root) / name, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--limit", type=int, default=0, help="export only the first N volumes (smoke test)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    paths = volume_paths()
    names = sorted(paths)[:a.limit] if a.limit else sorted(paths)
    geo = {n: volume_geometry(paths[n]) for n in names}
    kept, dropped = clean_boxes(read_annotations(ANNOTATIONS))
    kept = [r for r in kept if r["file"] in geo]                       # boxes of volumes that are not on disk cannot be exported
    lesions = merge_to_3d(rows_to_rss_frame(kept, {n: g["n_rows"] for n, g in geo.items()}))
    write_lesions(a.out / "lesions.csv", lesions)
    print(f"boxes kept {len(kept)}, dropped {dict(dropped)}; 3D lesions {len(lesions)}; "
          f"by family {dict(Counter(L['family'] for L in lesions))}")

    patients = {n: geo[n]["patient_id"] for n in names}
    folds = make_folds(patients)
    assert_folds_by_patient(folds, patients)
    (a.out / "folds.json").write_text(json.dumps({"folds": folds}, indent=1))

    by_file = {}
    for L in lesions:
        by_file.setdefault(L["file"], []).append(L)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(job, [(n, paths[n], by_file.get(n, []), a.out, 1000 + i) for i, n in enumerate(names)]))
    rows = [manifest_row(r["file"], paths[r["file"]], r["out_dir"], r["n_lesions"], r["status"]) for r in results]
    write_manifest(a.out / "manifest.csv", rows)
    bad = [r["file"] for r in results if r["status"] != "ok"]
    print(f"{len(results)} volumes, {len(bad)} failed: {bad[:10]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
```

（`import json` 保留。）`export_volume` 里 `size=320` 与 `n_rows` 的关系由 Task 5 的 relink 断言守着；这里不再重导。

- [ ] **Step 7: 全套测试**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`
Expected: 全过（342 + 新增）。

- [ ] **Step 8: 提交**

```bash
git add anatobind/train/train_detector.py scripts/check_h1.py scripts/train_reliability.py scripts/cache_detections.py scripts/build_fastmri_knee.py tests/test_train_detector.py tests/test_train_reliability.py
git commit -m "Leg 2: every consumer loads folds and lesions through the Gate 0 manifest; cache threshold flag"
```

---

### Task 5: 从旧导出建新导出根（relink）并在真实数据上运行

**Files:**
- Create: `scripts/relink_fastmri_knee_gate0.py`
- Test: `tests/test_relink_fastmri_knee_gate0.py`

**Interfaces:**
- Consumes: Task 2–3 的函数。
- Produces: `relink(legacy_root, new_root, annotations, paths) -> dict(summary)`；磁盘上的 `derived/fastmri_knee/leg2_gate0/`。

- [ ] **Step 1: 写测试**

```python
# tests/test_relink_fastmri_knee_gate0.py
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from anatobind.data_engine.fastmri_knee import LegacyBoxConvention, VIEWS, load_folds, load_lesions

HEADER = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding>'
    "<encodedSpace><matrixSize><x>64</x><y>40</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>28</x><y>17.5</y><z>4.5</z></fieldOfView_mm></encodedSpace>"
    "<reconSpace><matrixSize><x>32</x><y>32</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>14</x><y>14</y><z>3</z></fieldOfView_mm></reconSpace>"
    "</encoding></ismrmrdHeader>"
)
SIZE = 32


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/relink_fastmri_knee_gate0.py"
    spec = importlib.util.spec_from_file_location("relink_gate0", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _h5(path, patient, slices=4):
    with h5py.File(path, "w") as h:
        h.create_dataset("kspace", data=np.zeros((slices, 2, 64, 40), np.complex64))
        h.create_dataset("reconstruction_rss", data=np.zeros((slices, SIZE, SIZE), np.float32))
        h.create_dataset("ismrmrd_header", data=np.bytes_(HEADER.encode()))
        h.attrs["patient_id"] = patient
        h.attrs["acquisition"] = "CORPDFS_FBK"
    return path


def _legacy_export(root, files, patients, slices=4):
    """The 2026-09-14 layout: images + meta.json per volume, as-is lesions.csv, folds.json, five-column manifest."""
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for f, p in zip(files, patients):
        d = root / f
        d.mkdir()
        for v in VIEWS:
            np.save(d / f"{v}.npy", rng.normal(size=(slices, SIZE, SIZE)).astype(np.float16))
        (d / "meta.json").write_text(json.dumps({"patient_id": p, "acquisition": "CORPDFS_FBK", "slices": slices,
                                                 "size": SIZE, "n_lesions": 1, "seed": 1}))
    (root / "lesions.csv").write_text("lesion_id,file,family,z0,z1,x0,y0,x1,y1,n_boxes\n"
                                      f"0,{files[0]},effusion,1,2,4,6,14,18,2\n")        # CSV frame: y from the bottom
    (root / "folds.json").write_text(json.dumps({"folds": {files[0]: 0, files[1]: 1}}))
    (root / "manifest.csv").write_text("file,out_dir,slices,n_lesions,status\n" +
                                       "".join(f"{f},{root / f},{slices},{int(f == files[0])},ok\n" for f in files))


def _annotations(path, file):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "slice", "study_level", "x", "y", "width", "height", "label"])
        w.writeheader()
        w.writerow({"file": file, "slice": 1, "study_level": "No", "x": 4, "y": 6, "width": 10, "height": 12, "label": "Joint Effusion "})
        w.writerow({"file": file, "slice": 2, "study_level": "No", "x": 4, "y": 6, "width": 10, "height": 12, "label": "Joint Effusion "})


def _tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_relink_flips_rows_links_volumes_and_leaves_the_legacy_root_untouched(tmp_path):
    files, patients = ["file1", "file2"], ["pa", "pb"]
    paths = {f: _h5(tmp_path / f"{f}.h5", p) for f, p in zip(files, patients)}
    legacy = tmp_path / "leg2"
    _legacy_export(legacy, files, patients)
    ann = tmp_path / "knee.csv"
    _annotations(ann, "file1")
    before = _tree_hash(legacy)

    new = tmp_path / "leg2_gate0"
    summary = _load().relink(legacy, new, ann, paths)

    assert _tree_hash(legacy) == before                                   # nothing in the legacy root changed
    assert (new / "file1").is_symlink() and (new / "file1" / "clean.npy").exists()
    lesions = load_lesions(new)                                           # loads: manifest says transform_version 2
    assert len(lesions) == 1
    L = lesions[0]
    assert (L["x0"], L["x1"]) == (4, 14)
    assert (L["y0"], L["y1"]) == (SIZE - 18, SIZE - 6)                    # rows flipped: [nr - y - h, nr - y)
    assert load_folds(new) == {"file1": 0, "file2": 1}
    assert summary["n_volumes"] == 2 and summary["n_lesions"] == 1 and summary["legacy_match"] == 1
    with pytest.raises(LegacyBoxConvention):
        load_lesions(legacy)


def test_relink_refuses_when_the_folds_would_change(tmp_path):
    files, patients = ["file1", "file2"], ["pa", "pb"]
    paths = {f: _h5(tmp_path / f"{f}.h5", p) for f, p in zip(files, patients)}
    legacy = tmp_path / "leg2"
    _legacy_export(legacy, files, patients)
    (legacy / "folds.json").write_text(json.dumps({"folds": {"file1": 1, "file2": 0}}))   # not what make_folds gives
    ann = tmp_path / "knee.csv"
    _annotations(ann, "file1")
    with pytest.raises(ValueError, match="folds"):
        _load().relink(legacy, tmp_path / "new", ann, paths)
```

（`make_folds({"file1": "pa", "file2": "pb"})` 在 seed 0 下给 `{file1: 0, file2: 1}`，2026-09-24 实测；第二个测试写的是相反的分法。）

- [ ] **Step 2: 跑测试，确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_relink_fastmri_knee_gate0.py -q -p no:cacheprovider`
Expected: FileNotFoundError（脚本不存在）。

- [ ] **Step 3: 写脚本**

```python
#!/usr/bin/env python
# scripts/relink_fastmri_knee_gate0.py
"""Gate 0 for the leg 2 knee export (v2.6 §4.3): a new export root whose volume directories are relative links to
the 2026-09-14 images and whose lesions.csv is in the RSS frame. The legacy root is read, never written.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/relink_fastmri_knee_gate0.py
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import TRANSFORM_VERSION, rows_to_rss_frame  # noqa: E402
from anatobind.data_engine.fastmri_knee import (  # noqa: E402
    ANNOTATIONS, EXPORT_ROOT, LEGACY_EXPORT_ROOT, assert_folds_by_patient, clean_boxes, make_folds, manifest_row,
    merge_to_3d, read_annotations, volume_geometry, volume_paths, write_lesions, write_manifest,
)


def check_against_legacy(legacy_csv, lesions, n_rows_of):
    """Every legacy lesion of a volume on disk must reappear with the same file/family/z/x/n_boxes and rows
    mapped by row0' = nr - row1, row1' = nr - row0. Returns the number of matched lesions."""
    with open(legacy_csv, newline="") as fh:
        old = [r for r in csv.DictReader(fh) if r["file"] in n_rows_of]

    def key(d):
        return (d["file"], d["family"], int(d["z0"]), int(d["z1"]), int(d["x0"]), int(d["x1"]), int(d["n_boxes"]))

    a = sorted((key(r), n_rows_of[r["file"]] - int(r["y1"]), n_rows_of[r["file"]] - int(r["y0"])) for r in old)
    b = sorted((key(L), L["y0"], L["y1"]) for L in lesions)
    if a != b:
        diff = next(((x, y) for x, y in zip(a, b) if x != y), None)
        raise ValueError(f"legacy/new lesion mismatch: {len(a)} legacy vs {len(b)} new; first difference {diff}")
    return len(b)


def relink(legacy_root, new_root, annotations, paths):
    legacy_root, new_root = Path(legacy_root), Path(new_root)
    with open(legacy_root / "manifest.csv", newline="") as fh:
        legacy = {r["file"]: r for r in csv.DictReader(fh)}
    names = sorted(legacy)
    geo = {n: volume_geometry(paths[n]) for n in names}
    for n in names:
        meta = json.loads((legacy_root / n / "meta.json").read_text())
        if geo[n]["n_rows"] != meta["size"] or geo[n]["slices"] != meta["slices"]:
            raise ValueError(f"{n}: exported grid {meta['size']}x{meta['slices']} slices != reconstruction_rss "
                             f"{geo[n]['n_rows']} rows x {geo[n]['slices']} slices; the CSV rows were labelled on the RSS grid")
    n_rows_of = {n: geo[n]["n_rows"] for n in names}

    kept, dropped = clean_boxes(read_annotations(annotations))
    missing = sorted({r["file"] for r in kept} - set(n_rows_of))
    kept = [r for r in kept if r["file"] in n_rows_of]
    lesions = merge_to_3d(rows_to_rss_frame(kept, n_rows_of))
    matched = check_against_legacy(legacy_root / "lesions.csv", lesions, n_rows_of)

    patients = {n: geo[n]["patient_id"] for n in names}
    folds = make_folds(patients)
    assert_folds_by_patient(folds, patients)
    legacy_folds = json.loads((legacy_root / "folds.json").read_text())["folds"]
    if folds != legacy_folds:
        raise ValueError("the regenerated folds differ from the legacy folds.json; the H1 rerun must keep the same "
                         "patients per fold, stop and look")

    new_root.mkdir(parents=True, exist_ok=True)
    for n in names:
        link = new_root / n
        if not link.exists():
            os.symlink(os.path.relpath(legacy_root / n, new_root), link)
    write_lesions(new_root / "lesions.csv", lesions)
    (new_root / "folds.json").write_text(json.dumps({"folds": folds}, indent=1))
    by_file = Counter(L["file"] for L in lesions)
    write_manifest(new_root / "manifest.csv",
                   [manifest_row(n, paths[n], new_root / n, by_file.get(n, 0), legacy[n]["status"]) for n in names])
    (new_root / "README.txt").write_text(
        f"Gate 0 export (transform_version {TRANSFORM_VERSION}): volume directories link to {legacy_root}; only "
        f"lesions.csv (RSS frame, rows from the top), folds.json and manifest.csv are new. Built by "
        f"scripts/relink_fastmri_knee_gate0.py from {annotations}.\n")
    return {"n_volumes": len(names), "n_lesions": len(lesions), "legacy_match": matched, "dropped": dict(dropped),
            "files_without_volume": len(missing), "by_family": dict(Counter(L["family"] for L in lesions))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-root", type=Path, default=LEGACY_EXPORT_ROOT)
    ap.add_argument("--out", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    a = ap.parse_args()
    print(json.dumps(relink(a.legacy_root, a.out, a.annotations, volume_paths()), indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试，确认通过**

Run: 同 Step 2。Expected: 2 passed。

- [ ] **Step 5: 在真实数据上跑**

```bash
cd /data0/congcong/code/Project_Doing/foundation_model-aur
mkdir -p docs/verification/2026-09-24/gate0
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/relink_fastmri_knee_gate0.py \
    | tee docs/verification/2026-09-24/gate0/relink_output.txt
```

期望：`n_volumes 1172`、`legacy_match` = `n_lesions`（旧 lesions.csv 4016 条里属于在盘卷的那部分；2026-09-24 实测 CSV 有 974 卷在盘，若 `files_without_volume` > 0 记进报告）、`by_family` 与旧的 {cartilage 1324, meniscus 1227, bone 615, ligament 523, effusion 327} 只差不在盘的卷。然后核验：

```bash
R=/data2/congcong/data/FM_data/derived/fastmri_knee
ls -la $R/leg2_gate0 | head -5; ls $R/leg2_gate0 | wc -l          # 期望 1172 个链接 + 4 个文件
readlink $R/leg2_gate0/file1000000                                  # 期望 ../leg2/file1000000
head -3 $R/leg2_gate0/lesions.csv; head -3 $R/leg2/lesions.csv     # 同一 lesion: y0'=320-y1, y1'=320-y0
diff <(python3 -c "import json;print(sorted(json.load(open('$R/leg2/folds.json'))['folds'].items()))") \
     <(python3 -c "import json;print(sorted(json.load(open('$R/leg2_gate0/folds.json'))['folds'].items()))") && echo FOLDS_IDENTICAL
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python - <<'EOF'
from anatobind.data_engine.fastmri_knee import EXPORT_ROOT, LEGACY_EXPORT_ROOT, load_lesions, LegacyBoxConvention
print("new:", len(load_lesions(EXPORT_ROOT)), "lesions")
try:
    load_lesions(LEGACY_EXPORT_ROOT); print("BUG: legacy loaded")
except LegacyBoxConvention as e:
    print("legacy refused:", str(e)[:80])
EOF
stat -c '%y %s' $R/leg2/manifest.csv $R/leg2/lesions.csv $R/leg2/folds.json    # 期望 mtime 仍是 2026-09-14/15
```

把上面全部输出追加进 `docs/verification/2026-09-24/gate0/relink_output.txt`。

- [ ] **Step 6: 提交**

```bash
git add scripts/relink_fastmri_knee_gate0.py tests/test_relink_fastmri_knee_gate0.py docs/verification/2026-09-24/gate0/relink_output.txt
git commit -m "Gate 0: relink the leg 2 knee export with RSS-frame boxes, identical folds and a versioned manifest"
```

---

### Task 6: 转换前后的亮度审计与叠图（Gate 0 的判门）

**Files:**
- Create: `scripts/audit_fastmri_plus_boxes.py`
- Test: `tests/test_audit_fastmri_plus_boxes.py`
- Output: `docs/verification/2026-09-24/gate0/{audit.md,audit.json,audit_output.txt}`、`~/figs/anatobind_gate0/*.png`

**Interfaces:**
- Consumes: Task 1–2 的 `convert_box_csv_to_rss`、`read_fastmri_plus_rows`。
- Produces: 函数 `box_contrast(img, row0, row1, col0, col1) -> float`、`series_of(file) -> str`、`volume_verdict(ratios_asis, ratios_converted) -> str`、`decide(summary) -> bool`；判门行 `GATE0_AUDIT: PASS|FAIL`。

- [ ] **Step 1: 写测试**

```python
# tests/test_audit_fastmri_plus_boxes.py
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_fastmri_plus_boxes.py"
    spec = importlib.util.spec_from_file_location("audit_boxes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_box_contrast_is_high_on_the_bright_block_and_about_one_elsewhere():
    m = _load()
    img = np.ones((100, 100), np.float32)
    img[40:50, 60:70] = 10.0
    assert m.box_contrast(img, 40, 50, 60, 70) > 5.0
    assert m.box_contrast(img, 10, 20, 10, 20) == pytest.approx(1.0)


def test_volume_verdict_and_series():
    m = _load()
    assert m.volume_verdict([1.0, 1.1], [1.6, 1.4]) == "converted"
    assert m.volume_verdict([1.6, 1.4], [1.0, 1.1]) == "as_is"
    assert m.volume_verdict([1.0], [1.0]) == "tie"
    assert m.series_of("file_brain_AXFLAIR_203_6000123") == "203"


def test_decide_applies_the_organ_and_series_thresholds():
    m = _load()
    ok = {"knee": {"n": 100, "converted": 95, "per_series": {}},
          "brain": {"n": 100, "converted": 92, "per_series": {"200": {"n": 50, "converted": 45}, "203": {"n": 10, "converted": 8},
                                                             "205": {"n": 1, "converted": 0}}}}
    assert m.decide(ok) is True
    bad_series = {"knee": ok["knee"], "brain": {**ok["brain"], "per_series": {**ok["brain"]["per_series"], "203": {"n": 10, "converted": 5}}}}
    assert m.decide(bad_series) is False
    bad_knee = {**ok, "knee": {"n": 100, "converted": 80, "per_series": {}}}
    assert m.decide(bad_knee) is False
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_audit_fastmri_plus_boxes.py -q -p no:cacheprovider`
Expected: FileNotFoundError。

- [ ] **Step 3: 写脚本**

```python
#!/usr/bin/env python
# scripts/audit_fastmri_plus_boxes.py
"""Gate 0 overlay / intensity audit (v2.6 §4.2 item 4) over every annotated volume: under the converted mapping a
fastMRI+ box must sit on the bright tissue it names, under the as-is mapping it must not.
Knee: joint effusion and subchondral oedema boxes on fat-saturated PD (CORPDFS). Brain: nonspecific white-matter
lesions and lacunar infarcts on FLAIR, reported per series. Contrast = 75th percentile inside the box / median of
a ring around it (the 2026-09-15 probe's statistic, probe0b_contrast.py).

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/audit_fastmri_plus_boxes.py \
      --out docs/verification/2026-09-24/gate0 --figs ~/figs/anatobind_gate0
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import convert_box_csv_to_rss, read_fastmri_plus_rows  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
KNEE_CSV = FM / "fastMRI_lh_brain_knee/Annotations/knee.csv"
BRAIN_CSV = FM / "fastMRI_lh_brain_knee/Annotations/brain.csv"
KNEE_ROOT = FM / "fastMRI_lh_brain_knee/kspace/knee"
BRAIN_ROOT = FM / "fastMRI_lh_brain_knee/kspace/brain"
KNEE_LABELS = {"Joint Effusion", "Bone- Subchondral edema"}
BRAIN_LABELS = {"Nonspecific white matter lesion", "Lacunar infarct"}
KNEE_MIN_WIN, BRAIN_MIN_WIN, SERIES_MIN_WIN, SERIES_MIN_N = 0.90, 0.90, 0.75, 3


def h5_path(root, stem):
    for split in ("multicoil_train", "multicoil_val"):
        p = root / split / f"{stem}.h5"
        if p.exists():
            return p
    return None


def series_of(file):
    m = re.search(r"AX[A-Z0-9]+_(\d+)_", file)
    return m.group(1) if m else "?"


def box_contrast(img, row0, row1, col0, col1):
    """75th percentile inside the box over the median of a ring 2..6 px around it; 1.0 on flat background."""
    inner = np.zeros(img.shape, bool)
    inner[row0:row1, col0:col1] = True
    ring = ndimage.binary_dilation(inner, iterations=6) & ~ndimage.binary_dilation(inner, iterations=2)
    return float(np.percentile(img[inner], 75) / (np.median(img[ring]) + 1e-9))


def volume_verdict(ratios_asis, ratios_converted):
    a, c = float(np.median(ratios_asis)), float(np.median(ratios_converted))
    return "converted" if c > a else "as_is" if a > c else "tie"


def audit_volume(rss, rows):
    """rss: (slice, row, col). Returns (as-is ratios, converted ratios) over the volume's boxes."""
    ns, nr, nc = rss.shape
    asis, conv = [], []
    for r in rows:
        if not 0 <= r["slice"] < ns:
            continue
        img = rss[r["slice"]]
        for kind, (row0, row1, col0, col1) in (("as_is", (r["y"], r["y"] + r["height"], r["x"], r["x"] + r["width"])),
                                                ("converted", convert_box_csv_to_rss(r["x"], r["y"], r["width"], r["height"], nr))):
            row0, row1, col0, col1 = max(0, row0), min(nr, row1), max(0, col0), min(nc, col1)
            if row1 <= row0 or col1 <= col0:
                continue
            (asis if kind == "as_is" else conv).append(box_contrast(img, row0, row1, col0, col1))
    return asis, conv


def audit_organ(csv_path, root, labels, acquisition=None):
    rows = [r for r in read_fastmri_plus_rows(csv_path) if r["label"] in labels and r["width"] >= 3 and r["height"] >= 3]
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)
    per_volume = {}
    for f in sorted(by_file):
        p = h5_path(root, f)
        if p is None:
            continue
        with h5py.File(p, "r") as h:
            if acquisition is not None and str(h.attrs.get("acquisition", "")) != acquisition:
                continue
            rss = h["reconstruction_rss"][()]
        asis, conv = audit_volume(rss, by_file[f])
        if asis and conv:
            per_volume[f] = {"n_boxes": len(conv), "median_as_is": float(np.median(asis)),
                             "median_converted": float(np.median(conv)), "verdict": volume_verdict(asis, conv),
                             "series": series_of(f), "rss_rows": int(rss.shape[1])}
    return per_volume


def summarise(per_volume):
    out = {"n": len(per_volume), "converted": sum(v["verdict"] == "converted" for v in per_volume.values()),
           "as_is": sum(v["verdict"] == "as_is" for v in per_volume.values()),
           "median_ratio_as_is": float(np.median([v["median_as_is"] for v in per_volume.values()])) if per_volume else None,
           "median_ratio_converted": float(np.median([v["median_converted"] for v in per_volume.values()])) if per_volume else None,
           "per_series": {}}
    for s in sorted({v["series"] for v in per_volume.values()}):
        sel = [v for v in per_volume.values() if v["series"] == s]
        out["per_series"][s] = {"n": len(sel), "converted": sum(v["verdict"] == "converted" for v in sel),
                                "rss_rows": sorted({v["rss_rows"] for v in sel})}
    return out


def decide(summary):
    knee, brain = summary["knee"], summary["brain"]
    if knee["n"] == 0 or brain["n"] == 0:
        return False
    if knee["converted"] / knee["n"] < KNEE_MIN_WIN or brain["converted"] / brain["n"] < BRAIN_MIN_WIN:
        return False
    for s in brain["per_series"].values():
        if s["n"] >= SERIES_MIN_N and s["converted"] / s["n"] < SERIES_MIN_WIN:
            return False
    return True


def overlays(per_volume, csv_path, root, labels, figs, name, per_series=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in read_fastmri_plus_rows(csv_path) if r["label"] in labels and r["file"] in per_volume]
    picks = []
    if per_series:
        for s in sorted({v["series"] for v in per_volume.values()}):
            picks.append(next(f for f, v in per_volume.items() if v["series"] == s))
    else:
        picks = sorted(per_volume)[:8]
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6), squeeze=False)
    for j, f in enumerate(picks):
        r = next(r for r in rows if r["file"] == f)
        with h5py.File(h5_path(root, f), "r") as h:
            img = h["reconstruction_rss"][r["slice"]]
        nr = img.shape[0]
        for i, (kind, (row0, row1, col0, col1)) in enumerate((("as-is", (r["y"], r["y"] + r["height"], r["x"], r["x"] + r["width"])),
                                                              ("converted", convert_box_csv_to_rss(r["x"], r["y"], r["width"], r["height"], nr)))):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.5))
            ax.add_patch(plt.Rectangle((col0, row0), col1 - col0, row1 - row0, fill=False, edgecolor="lime", linewidth=1.2))
            ax.set_title(f"{kind} {per_volume[f]['series']} {f[-7:]} s{r['slice']}", fontsize=7)
            ax.axis("off")
    plt.tight_layout()
    figs.mkdir(parents=True, exist_ok=True)
    plt.savefig(figs / f"{name}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figs", type=Path, default=Path.home() / "figs/anatobind_gate0")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    knee = audit_organ(KNEE_CSV, KNEE_ROOT, KNEE_LABELS, acquisition="CORPDFS_FBK")
    brain = audit_organ(BRAIN_CSV, BRAIN_ROOT, BRAIN_LABELS)
    summary = {"knee": summarise(knee), "brain": summarise(brain)}
    summary["pass"] = decide(summary)
    (a.out / "audit.json").write_text(json.dumps({"summary": summary, "knee": knee, "brain": brain}, indent=1))
    overlays(knee, KNEE_CSV, KNEE_ROOT, KNEE_LABELS, a.figs, "knee_effusion_asis_vs_converted")
    overlays(brain, BRAIN_CSV, BRAIN_ROOT, BRAIN_LABELS, a.figs, "brain_small_lesions_per_series", per_series=True)
    lines = ["# Gate 0 box audit", "",
             "| organ | volumes | converted wins | as-is wins | median ratio as-is | median ratio converted |", "|---|---|---|---|---|---|"]
    for organ in ("knee", "brain"):
        s = summary[organ]
        lines.append(f"| {organ} | {s['n']} | {s['converted']} | {s['as_is']} | {s['median_ratio_as_is']:.3f} | {s['median_ratio_converted']:.3f} |")
    lines += ["", "brain per series (n, converted wins, RSS rows):", ""]
    for s, v in summary["brain"]["per_series"].items():
        lines.append(f"- {s}: n={v['n']}, converted={v['converted']}, rows={v['rss_rows']}")
    lines += ["", f"GATE0_AUDIT: {'PASS' if summary['pass'] else 'FAIL'} (knee >= {KNEE_MIN_WIN:.0%}, brain >= {BRAIN_MIN_WIN:.0%}, "
              f"every series with >= {SERIES_MIN_N} volumes >= {SERIES_MIN_WIN:.0%})"]
    (a.out / "audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试，确认通过**

Run: 同 Step 2。Expected: 3 passed。

- [ ] **Step 5: 在真实数据上跑并判门**

```bash
cd /data0/congcong/code/Project_Doing/foundation_model-aur
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/audit_fastmri_plus_boxes.py \
    --out docs/verification/2026-09-24/gate0 --figs ~/figs/anatobind_gate0 | tee docs/verification/2026-09-24/gate0/audit_output.txt
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/anatobind_gate0/knee_effusion_asis_vs_converted.png
```

期望：`GATE0_AUDIT: PASS`；膝转换胜出 ≥ 90%（探针 30/30）、脑 ≥ 90%（探针 22/24）；每个系列（200/201/202/203/205/206/209/210）都记录 RSS 行数。**若 FAIL 或某系列（如 203 的 213/276 行卷）转换不胜出：停，不进 Task 7–11，把 audit.md 与图报给用户**（可能该系列标在别的网格上，需要另定映射）。看两张图：

- http://localhost:8765/anatobind_gate0/knee_effusion_asis_vs_converted.png ；/home/congcongliu/figs/anatobind_gate0/knee_effusion_asis_vs_converted.png
- http://localhost:8765/anatobind_gate0/brain_small_lesions_per_series.png ；/home/congcongliu/figs/anatobind_gate0/brain_small_lesions_per_series.png

- [ ] **Step 6: 提交**

```bash
git add scripts/audit_fastmri_plus_boxes.py tests/test_audit_fastmri_plus_boxes.py docs/verification/2026-09-24/gate0
git commit -m "Gate 0: intensity audit of the fastMRI+ box conversion on every annotated knee and brain volume"
```

---

### Task 7: GitHub Actions 跑 CPU 测试

**Files:**
- Create: `.github/workflows/tests.yml`、`requirements-ci.txt`

- [ ] **Step 1: 写文件**

```yaml
# .github/workflows/tests.yml
name: tests
on:
  push:
    branches: [main, "build/**", "plan/**"]
  pull_request:
jobs:
  cpu:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
      - run: pip install -r requirements-ci.txt
      - run: PYTHONPATH=. python -m pytest tests/ -q -p no:cacheprovider
```

```
# requirements-ci.txt — the CPU test suite's third-party imports (torch is installed from the CPU wheel index first)
numpy
scipy
nibabel
h5py
matplotlib
monai==1.5.2
einops
pytest
```

- [ ] **Step 2: 本地确认测试不依赖 /data2**

```bash
grep -rn "/data2\|/data0" tests/ | grep -v "^tests/.*#" | head    # 期望：没有输出
```

- [ ] **Step 3: 提交并推送分支，看 CI**

```bash
git add .github/workflows/tests.yml requirements-ci.txt
git commit -m "CI: run the CPU test suite on GitHub Actions"
git push origin build/aur-system || https_proxy=http://127.0.0.1:7897 git push origin build/aur-system
sleep 60; gh run list --repo sober235/Foundation_Model --limit 3
gh run watch --repo sober235/Foundation_Model $(gh run list --repo sober235/Foundation_Model --limit 1 --json databaseId --jq '.[0].databaseId') --exit-status
```

期望：conclusion `success`。若失败（多半是缺包或 torch/monai 版本），按日志补 `requirements-ci.txt` 再推；最多修两轮，仍不过就把 workflow 留着、把失败原因写进 STATUS §4 交用户。

---

### Task 8: H1 重跑——五折训练与检出缓存（GPU）

**Files:**
- 无代码改动；产物 `runs/detector_gate0_fold{0..4}/`、新根 `detections/fold{k}/`、`logs/detector_gate0_fold*.log`、`logs/cache_gate0_fold*.log`

- [ ] **Step 1: 确认卡空、配置对得上旧跑**

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader      # 0 1 4 6 须空
python3 -c "import json; c=json.load(open('/data0/congcong/code/Project_Doing/foundation_model/runs/detector_fold0/config.json')); print({k:c[k] for k in ('steps','warmup','batch','lr','wd','seed','model')})"
```

期望旧配置：steps 20000、warmup 500、batch 16、lr 2e-4、wd 0.05、seed 0、model = FULL。新跑除 `--out` 与导出根外一字不改。

- [ ] **Step 2: 起四折**

四条命令分别写（zsh 里 `set -- $pair` 不分词，别用循环拆字符串）：

```bash
cd /data0/congcong/code/Project_Doing/foundation_model-aur
PY=$HOME/anaconda3/envs/nvgen/bin/python
CUDA_VISIBLE_DEVICES=0 setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 $PY scripts/train_detector.py --fold 0 --out runs/detector_gate0_fold0 --resume > logs/detector_gate0_fold0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 $PY scripts/train_detector.py --fold 1 --out runs/detector_gate0_fold1 --resume > logs/detector_gate0_fold1.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 $PY scripts/train_detector.py --fold 2 --out runs/detector_gate0_fold2 --resume > logs/detector_gate0_fold2.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 $PY scripts/train_detector.py --fold 3 --out runs/detector_gate0_fold3 --resume > logs/detector_gate0_fold3.log 2>&1 &
pgrep -af "[t]rain_detector.py --fold" | cut -c1-120             # 期望四行
```

两分钟后 `tail -n 2 logs/detector_gate0_fold*.log`：每折已在打印 step 行，s/step 约 0.18–0.31。

- [ ] **Step 3: fold 4 在 fold 0 结束后自动起（GPU 0）**

```bash
cat > logs/launch_gate0_fold4.sh <<'EOF'
#!/bin/bash
cd /data0/congcong/code/Project_Doing/foundation_model-aur
until grep -q "^done:" logs/detector_gate0_fold0.log 2>/dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 $HOME/anaconda3/envs/nvgen/bin/python \
    scripts/train_detector.py --fold 4 --out runs/detector_gate0_fold4 --resume > logs/detector_gate0_fold4.log 2>&1
EOF
chmod +x logs/launch_gate0_fold4.sh; setsid nohup logs/launch_gate0_fold4.sh > logs/launch_gate0_fold4.out 2>&1 &
```

- [ ] **Step 4: 训练期间做 Task 9–11 的代码部分（CPU）；每折结束后缓存检出**

每折 `logs/detector_gate0_fold$f.log` 出现 `done:` 后：

```bash
f=0; g=0     # 用刚空出来的那张卡
CUDA_VISIBLE_DEVICES=$g setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 $HOME/anaconda3/envs/nvgen/bin/python \
    scripts/cache_detections.py --fold $f --run runs/detector_gate0_fold$f --score-min 0.01 > logs/cache_gate0_fold$f.log 2>&1 &
```

期望每折日志末行 `fold k: N (volume, view) detection files in .../leg2_gate0/detections/foldk`，N = 该折留出卷数 × 7（五折合计 1172 × 7 = 8204）。

- [ ] **Step 5: 训练事实记录**

五折都 `done:` 后：

```bash
for f in 0 1 2 3 4; do echo "fold $f: $(tail -n 2 logs/detector_gate0_fold$f.log | head -1 | cut -c1-160)"; done
for f in 0 1 2 3 4; do python3 - <<EOF
import json
rows=[json.loads(l) for l in open("runs/detector_gate0_fold$f/metrics.jsonl")]
last=rows[-50:]; print("fold $f steps", rows[-1]["step"], "s/step %.3f" % (rows[-1]["seconds"]/rows[-1]["step"]), "loss last50 %.3f heat %.3f" % (sum(r["loss"] for r in last)/50, sum(r["heat"] for r in last)/50))
EOF
done
ls /data2/congcong/data/FM_data/derived/fastmri_knee/leg2_gate0/detections/fold*/ | grep -c pkl     # 期望 8204
```

存到 `docs/verification/2026-09-24/h1_rerun/training_facts.txt`。

---

### Task 9: H1 重跑的评估（检测工作点、族、大小、几何、患者覆盖）与旧 H1 门

**Files:**
- Create: `anatobind/eval/fastmri_knee_detection.py`、`scripts/eval_fastmri_knee_detection.py`
- Test: `tests/test_fastmri_knee_detection_eval.py`
- Output: `docs/verification/2026-09-24/h1_rerun/{summary.md,gate.json,froc_*.csv,hits_clean.csv,output.txt,h1.txt}`

**Interfaces:**
- Consumes: `anatobind.eval.detection_metrics.{sweep, operating_point, per_family, gate, match_scan}`、`anatobind.eval.matching.iou3d`、Task 3 的 `load_manifest/load_lesions/load_folds`、检出 pickle（`{"dets": [ {family, score, z0, z1, y0, x0, y1, x1, ...} ]}`）。
- Produces: `SHARED_FAMILIES`、`THRESHOLDS`（0.01…0.99）、`corners(d)`、`scan_record(file, gt_lesions, dets, normal, families=None)`、`box_volume_ml(box, spacing)`、`hit_geometry(scan, thr, spacing)`、`size_terciles(scans, spacing_of, thr)`、`patient_coverage(scans, patient_of, thr)`、`fp_per_normal_scan(scans, thr)`；脚本输出 `TRANSFER_GATE: PASS|FAIL`。

- [ ] **Step 1: 写测试**

```python
# tests/test_fastmri_knee_detection_eval.py
import pytest

from anatobind.eval.detection_metrics import operating_point, sweep
from anatobind.eval.fastmri_knee_detection import (
    SHARED_FAMILIES, THRESHOLDS, box_volume_ml, corners, fp_per_normal_scan, hit_geometry, patient_coverage,
    scan_record, size_terciles,
)

SP = (3.0, 0.5, 0.5)


def _lesion(family, z0=2, z1=4, y0=10, x0=10, y1=30, x1=30):
    return {"family": family, "z0": z0, "z1": z1, "y0": y0, "x0": x0, "y1": y1, "x1": x1}


def _det(family, score, **kw):
    return {**_lesion(family, **kw), "score": score}


def test_corners_make_z1_exclusive_like_failure_labels():
    assert corners(_lesion("meniscus")) == [2, 10, 10, 5, 30, 30]


def test_scan_record_keeps_only_the_requested_families_for_truth_and_detections():
    gt = [_lesion("bone"), _lesion("meniscus", x0=100, x1=120)]
    dets = [_det("bone", 0.9), _det("cartilage", 0.8, x0=100, x1=120)]
    s = scan_record("f", gt, dets, normal=False, families=SHARED_FAMILIES)
    assert [g["family"] for g in s["gt"]] == ["meniscus"] and [d["family"] for d in s["dets"]] == ["cartilage"]
    assert scan_record("f", gt, dets, normal=False)["gt"][0]["family"] == "bone"


def test_box_volume_in_ml_uses_the_spacing():
    assert box_volume_ml([0, 0, 0, 2, 10, 10], SP) == pytest.approx(2 * 3.0 * 10 * 0.5 * 10 * 0.5 / 1000)


def test_hit_geometry_reports_centre_error_in_mm_and_iou():
    s = scan_record("f", [_lesion("meniscus")], [_det("meniscus", 0.9, y0=12, y1=32)], normal=False)   # shifted 2 rows
    hits = hit_geometry(s, thr=0.5, spacing=SP)
    assert len(hits) == 1 and hits[0]["centre_error_mm"] == pytest.approx(2 * 0.5) and 0.6 < hits[0]["iou"] < 1.0
    assert hits[0]["pred_family"] == "meniscus"


def test_size_terciles_split_each_family_into_three_bins_and_count_hits():
    scans = [scan_record(f"s{i}", [_lesion("meniscus", x1=10 + 5 * (i + 1))], [_det("meniscus", 0.9, x1=10 + 5 * (i + 1))] if i % 2 else [], normal=False)
             for i in range(6)]
    t = size_terciles(scans, {f"s{i}": SP for i in range(6)}, thr=0.5)
    assert [b["n"] for b in t["meniscus"]] == [2, 2, 2] and sum(b["hit"] for b in t["meniscus"]) == 3


def test_patient_coverage_counts_patients_with_a_family_correct_hit():
    scans = [scan_record("a", [_lesion("meniscus")], [_det("meniscus", 0.9)], normal=False),
             scan_record("b", [_lesion("meniscus")], [_det("cartilage", 0.9)], normal=False),     # localised, wrong family
             scan_record("c", [], [], normal=True)]
    cov = patient_coverage(scans, {"a": "p1", "b": "p2", "c": "p3"}, thr=0.5)
    assert cov == {"n_patients": 2, "covered": 1, "coverage": 0.5}


def test_fp_per_normal_scan_counts_only_scans_the_radiologist_called_normal():
    scans = [scan_record("n1", [], [_det("bone", 0.9), _det("bone", 0.3)], normal=True),
             scan_record("a", [], [_det("bone", 0.9)], normal=False)]     # annotated volume whose boxes were all dropped
    assert fp_per_normal_scan(scans, thr=0.5) == {"n_normal": 1, "fp_per_normal_scan": 1.0}


def test_sweep_thresholds_are_fine_enough_for_a_poorly_calibrated_detector():
    assert THRESHOLDS[0] == 0.01 and len(THRESHOLDS) == 99
    s = scan_record("f", [_lesion("meniscus")], [_det("meniscus", 0.03)], normal=False)
    rows = sweep([s], THRESHOLDS)
    assert operating_point(rows)["sensitivity_family"] == 1.0
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_fastmri_knee_detection_eval.py -q -p no:cacheprovider`
Expected: ModuleNotFoundError。

- [ ] **Step 3: 写模块**

```python
# anatobind/eval/fastmri_knee_detection.py
"""Held-out detection numbers of the leg 2 knee detector after Gate 0 (v2.6 §6.2) on the 3D detections cached by
scripts/cache_detections.py. Matching and the operating point are the SKM-TEA gate's
(anatobind.eval.detection_metrics: one-to-one 3D IoU >= 0.1, <= 2 false positives per scan)."""
import numpy as np

from anatobind.eval.detection_metrics import match_scan
from anatobind.eval.matching import iou3d

SHARED_FAMILIES = ("meniscus", "cartilage", "ligament", "effusion")   # the SKM-TEA families; bone exists only in fastMRI+
THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.01, 1.0, 0.01), 2))


def corners(d):
    """(z0, y0, x0, z1, y1, x1) with z1 exclusive, as anatobind.eval.detect3d.failure_labels builds them."""
    return [d["z0"], d["y0"], d["x0"], d["z1"] + 1, d["y1"], d["x1"]]


def scan_record(file, gt_lesions, dets, normal, families=None):
    """One scan for detection_metrics: truth and detections restricted to `families` (None = all)."""
    keep = (lambda f: True) if families is None else (lambda f: f in families)
    return {"file": file, "normal": bool(normal),
            "gt": [{"box": corners(L), "family": L["family"]} for L in gt_lesions if keep(L["family"])],
            "dets": [{"box": corners(d), "family": d["family"], "score": float(d["score"])} for d in dets if keep(d["family"])]}


def box_volume_ml(box, spacing):
    z0, y0, x0, z1, y1, x1 = box
    return float((z1 - z0) * spacing[0] * (y1 - y0) * spacing[1] * (x1 - x0) * spacing[2]) / 1000.0


def hit_geometry(scan, thr, spacing, iou=0.1):
    """Per hit at threshold thr: centre error in mm (box centres, physical spacing) and 3D IoU."""
    dets = [d for d in scan["dets"] if d["score"] >= thr]
    pairs = match_scan(scan["gt"], dets, iou)
    out = []
    for g, p in pairs.items():
        a, b = np.array(scan["gt"][g]["box"], float), np.array(dets[p]["box"], float)
        ca, cb = (a[:3] + a[3:]) / 2, (b[:3] + b[3:]) / 2
        out.append({"file": scan["file"], "family": scan["gt"][g]["family"], "pred_family": dets[p]["family"],
                    "centre_error_mm": float(np.sqrt((((ca - cb) * np.asarray(spacing, float)) ** 2).sum())),
                    "iou": float(iou3d(a[None], b[None])[0, 0]), "score": dets[p]["score"]})
    return out


def size_terciles(scans, spacing_of, thr, iou=0.1):
    """Localisation hits by family x size tercile (mL), like the SKM-TEA miss analysis."""
    rows = []
    for s in scans:
        dets = [d for d in s["dets"] if d["score"] >= thr]
        pairs = match_scan(s["gt"], dets, iou)
        for g, r in enumerate(s["gt"]):
            rows.append({"family": r["family"], "ml": box_volume_ml(r["box"], spacing_of[s["file"]]), "hit": int(g in pairs)})
    out = {}
    for fam in sorted({r["family"] for r in rows}):
        sel = [r for r in rows if r["family"] == fam]
        ml, hit = np.array([r["ml"] for r in sel]), np.array([r["hit"] for r in sel])
        bins = np.digitize(ml, np.quantile(ml, [1 / 3, 2 / 3]), right=True)
        out[fam] = [{"lo": float(ml[bins == b].min()) if (bins == b).any() else None,
                     "hi": float(ml[bins == b].max()) if (bins == b).any() else None,
                     "n": int((bins == b).sum()), "hit": int(hit[bins == b].sum())} for b in range(3)]
    return out


def patient_coverage(scans, patient_of, thr, iou=0.1):
    """Share of patients with >= 1 lesion that get >= 1 family-correct hit at thr."""
    has, got = set(), set()
    for s in scans:
        if not s["gt"]:
            continue
        p = patient_of[s["file"]]
        has.add(p)
        dets = [d for d in s["dets"] if d["score"] >= thr]
        if any(dets[q]["family"] == s["gt"][g]["family"] for g, q in match_scan(s["gt"], dets, iou).items()):
            got.add(p)
    return {"n_patients": len(has), "covered": len(got), "coverage": len(got) / len(has) if has else 0.0}


def fp_per_normal_scan(scans, thr):
    """Detections per scan on the volumes the radiologist left without any annotation (true negatives)."""
    normals = [s for s in scans if s["normal"]]
    n_fp = sum(sum(1 for d in s["dets"] if d["score"] >= thr) for s in normals)
    return {"n_normal": len(normals), "fp_per_normal_scan": n_fp / len(normals) if normals else 0.0}
```

（`np.digitize(..., right=True)` 把恰好落在分位点上的值归到低一档；测试里的六个等差体积分成 2/2/2，2026-09-24 实测。）

- [ ] **Step 4: 跑测试，确认通过**

Run: 同 Step 2。Expected: 8 passed。

- [ ] **Step 5: 写脚本**

```python
#!/usr/bin/env python
# scripts/eval_fastmri_knee_detection.py
"""H1 rerun evaluation (v2.6 §6.2): the leg 2 knee detector trained on Gate-0 boxes, held-out patients, every fold.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_fastmri_knee_detection.py \
      --out docs/verification/2026-09-24/h1_rerun
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri_knee import ANNOTATIONS, EXPORT_ROOT, VIEWS, load_folds, load_lesions, load_manifest  # noqa: E402
from anatobind.eval.detection_metrics import FP_MAX, GATE_SENSITIVITY, gate, operating_point, per_family, sweep  # noqa: E402
from anatobind.eval.fastmri_knee_detection import (  # noqa: E402
    SHARED_FAMILIES, THRESHOLDS, fp_per_normal_scan, hit_geometry, patient_coverage, scan_record, size_terciles,
)

FAMILY_SETS = {"shared4": SHARED_FAMILIES, "all5": None}
DECISION = ("clean", "shared4")


def load_view(export_root, det_root, view):
    """Raw material of one view: {file: (gt lesions, cached 3D detections, normal?)} over every held-out file
    (each file is held out in exactly one fold), plus spacing and patient per file and the files without a cache."""
    manifest, folds = load_manifest(export_root), load_folds(export_root)
    by_file = {}
    for r in load_lesions(export_root):
        by_file.setdefault(r["file"], []).append(r)
    with open(ANNOTATIONS, newline="") as fh:
        annotated = {r["file"] for r in csv.DictReader(fh)}        # every row, study-level ones included: a normal
    raw, missing = {}, []                                          # volume is one the radiologist wrote nothing about
    for f, k in sorted(folds.items()):
        p = Path(det_root) / f"fold{k}" / f"{f}__{view}.pkl"
        if not p.exists():
            missing.append(f)
            continue
        raw[f] = (by_file.get(f, []), pickle.load(open(p, "rb"))["dets"], f not in annotated)
    spacing_of = {f: (float(m["spacing_slice_mm"]), float(m["spacing_row_mm"]), float(m["spacing_col_mm"])) for f, m in manifest.items()}
    patient_of = {f: m["patient_id"] for f, m in manifest.items()}
    return raw, spacing_of, patient_of, missing


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--det-root", type=Path, default=None)
    ap.add_argument("--views", default=",".join(VIEWS))
    a = ap.parse_args()
    det_root = a.det_root or a.export_root / "detections"
    a.out.mkdir(parents=True, exist_ok=True)
    lines = ["# fastMRI+ knee detector after Gate 0 (held-out patients, five folds)", "",
             f"IoU >= 0.1, FP budget {FP_MAX}/scan, transfer gate on clean x shared4 family sensitivity >= {GATE_SENSITIVITY}", "",
             "| view | families | n_scans | n_gt | thr | sensitivity | sensitivity_family | fp_per_scan | fp_per_normal_scan | ceiling (sens_fam @ fp) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    decision = None
    for view in a.views.split(","):
        raw, spacing_of, patient_of, missing = load_view(a.export_root, det_root, view)
        for fam_name, fams in FAMILY_SETS.items():
            scans = [scan_record(f, gt, dets, normal, fams) for f, (gt, dets, normal) in raw.items()]
            rows = sweep(scans, THRESHOLDS)
            write_csv(a.out / f"froc_{view}_{fam_name}.csv", rows)
            op = operating_point(rows)
            ceiling = max(rows, key=lambda r: r["sensitivity_family"])
            n_gt = sum(len(s["gt"]) for s in scans)
            if op is None:
                lines.append(f"| {view} | {fam_name} | {len(scans)} | {n_gt} | - | - | - | - | - | {ceiling['sensitivity_family']:.3f} @ {ceiling['fp_per_scan']:.2f} |")
                continue
            fpn = fp_per_normal_scan(scans, op["thr"])
            lines.append(f"| {view} | {fam_name} | {len(scans)} | {n_gt} | {op['thr']:.2f} | {op['sensitivity']:.3f} | "
                         f"{op['sensitivity_family']:.3f} | {op['fp_per_scan']:.2f} | {fpn['fp_per_normal_scan']:.2f} (n={fpn['n_normal']}) | "
                         f"{ceiling['sensitivity_family']:.3f} @ {ceiling['fp_per_scan']:.2f} |")
            if (view, fam_name) == DECISION:
                fams_at_op = per_family(scans, op["thr"])
                terciles = size_terciles(scans, spacing_of, op["thr"])
                hits = [h for s in scans for h in hit_geometry(s, op["thr"], spacing_of[s["file"]])]
                cov = patient_coverage(scans, patient_of, op["thr"])
                decision = {**gate(rows), "view": view, "families": fam_name, "n_scans": len(scans), "n_gt": n_gt,
                            "missing_detection_files": len(missing), "fp_per_normal_scan": fpn,
                            "per_family": fams_at_op, "size_terciles": terciles, "patient_coverage": cov,
                            "n_hits": len(hits),
                            "centre_error_mm_median": float(np.median([h["centre_error_mm"] for h in hits])) if hits else None,
                            "iou_median": float(np.median([h["iou"] for h in hits])) if hits else None,
                            "family_correct_on_hits": float(np.mean([h["family"] == h["pred_family"] for h in hits])) if hits else None}
                if hits:
                    write_csv(a.out / f"hits_{view}.csv", hits)
    lines += ["", "## decision view (clean x shared4)", "", "```", json.dumps(decision, indent=1), "```", "",
              f"TRANSFER_GATE: {'PASS' if decision and decision['pass'] else 'FAIL'} "
              f"({json.dumps({k: decision[k] for k in ('thr', 'sensitivity_family', 'fp_per_scan')}) if decision else 'no operating point'})"]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    (a.out / "gate.json").write_text(json.dumps(decision, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 五折缓存齐后跑评估与旧 H1 门**

```bash
cd /data0/congcong/code/Project_Doing/foundation_model-aur
D=docs/verification/2026-09-24/h1_rerun; mkdir -p $D
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/eval_fastmri_knee_detection.py --out $D | tee $D/output.txt
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/check_h1.py > $D/h1.txt 2>&1; echo "rc=$?" >> $D/h1.txt; cat $D/h1.txt
```

期望：`summary.md` 有 7 视图 × 2 族集的工作点行、`gate.json` 有 `pass`、`TRANSFER_GATE:` 一行；`h1.txt` 有 `SCORE_MIN`/`H1 PASS|FAIL`（09-15 是"任何阈值都不行"，这次先看阈值搜索是否落入 [0.2, 0.5]）。**不解释、不调参**：结果原样进报告。

- [ ] **Step 7: 提交**

```bash
git add anatobind/eval/fastmri_knee_detection.py scripts/eval_fastmri_knee_detection.py tests/test_fastmri_knee_detection_eval.py docs/verification/2026-09-24/h1_rerun
git commit -m "H1 rerun: held-out detection evaluation of the Gate 0 knee detector with the pre-registered transfer gate"
```

---

### Task 10: Gate 0.5 的几何量与脑侧查表（`geometry.py`、`lookup.py`）

**Files:**
- Create: `anatobind/eval/geometry.py`
- Modify: `anatobind/eval/lookup.py`（末尾加 `BrainLookup`）
- Test: `tests/test_geometry.py`、`tests/test_brain_lookup.py`

**Interfaces:**
- Produces（geometry）: `HOST_CLASSES`（有序 dict：名 → SynthSeg 标签元组，类号 = 序号 + 1）、`CLASS_NAMES`、`LANDMARKS`、`host_class_map(seg) -> int8 array`、`class_distance_maps(class_map, spacing) -> {class_id: float array}`、`member_rects(members, shape) -> list[(c0, c1, r0, r1, s)]`、`lesion_class_distances(dist_maps, rects) -> {class_id: mm}`、`interface_margin(dists) -> (nearest_class, d1, d2)`。
- Produces（lookup）: `BRAIN_ALL`、`BRAIN_PARENCHYMA`、`class BrainLookup(seg, spacing, candidates)` 带 `host(rects) -> (label | None, overlap_fraction)`。

- [ ] **Step 1: 写测试**

```python
# tests/test_geometry.py
import numpy as np
import pytest

from anatobind.eval.geometry import (
    CLASS_NAMES, HOST_CLASSES, LANDMARKS, class_distance_maps, host_class_map, interface_margin,
    lesion_class_distances, member_rects,
)

SP = (0.5, 0.5, 5.0)          # (col, row, slice) mm, like a fastMRI FLAIR stack


def _seg():
    """(col, row, slice) = (40, 40, 4): left WM (2) on rows 0..20 and right WM (41) on rows 20..40 for cols 0..20,
    cortex (3) for cols 20..40, a lateral ventricle (4) inside the WM at cols 2..6, rows 2..6."""
    seg = np.zeros((40, 40, 4), np.int16)
    seg[:20, :20, :] = 2
    seg[:20, 20:, :] = 41
    seg[20:, :, :] = 3
    seg[2:6, 2:6, :] = 4
    return seg


def _member(x, w, y, h, s):
    return {"x": x, "width": w, "y": y, "height": h, "slice": s}


def test_host_class_map_merges_sides_and_drops_landmarks():
    m = host_class_map(_seg())
    wm, cortex = CLASS_NAMES.index("white_matter") + 1, CLASS_NAMES.index("cortex") + 1
    assert m[10, 10, 0] == wm and m[10, 30, 0] == wm          # left and right WM are one class
    assert m[30, 10, 0] == cortex
    assert m[3, 3, 0] == 0                                     # ventricle is a landmark, not a host
    assert set(LANDMARKS) == {4, 43, 5, 44, 14, 15, 24}
    assert sum(len(v) for v in HOST_CLASSES.values()) == 25


def test_a_lesion_inside_white_matter_has_d1_zero_and_d2_the_in_plane_distance_to_cortex():
    seg = _seg()
    dist = class_distance_maps(host_class_map(seg), SP)
    rects = member_rects([_member(10, 6, 10, 6, 0)], seg.shape)        # cols 10..15, rows 10..15, slice 0
    d = lesion_class_distances(dist, rects)
    c1, d1, d2 = interface_margin(d)
    assert c1 == CLASS_NAMES.index("white_matter") + 1 and d1 == 0.0
    assert d2 == pytest.approx((20 - 15) * 0.5)                         # 5 voxels to the first cortex column


def test_a_lesion_straddling_the_boundary_has_zero_interface_distance():
    seg = _seg()
    dist = class_distance_maps(host_class_map(seg), SP)
    _, d1, d2 = interface_margin(lesion_class_distances(dist, member_rects([_member(18, 4, 10, 4, 0)], seg.shape)))
    assert d1 == 0.0 and d2 == 0.0


def test_the_ventricle_does_not_create_an_interface():
    seg = _seg()
    dist = class_distance_maps(host_class_map(seg), SP)
    _, d1, d2 = interface_margin(lesion_class_distances(dist, member_rects([_member(7, 2, 7, 2, 0)], seg.shape)))
    assert d1 == 0.0 and d2 == pytest.approx((20 - 8) * 0.5)           # cortex, 12 voxels away; the ventricle two voxels away is ignored


def test_cross_slice_neighbours_are_a_slice_thickness_away():
    seg = np.zeros((40, 40, 4), np.int16)
    seg[:, :, :] = 2
    seg[:, :, 3] = 3                                                    # cortex only on the last slice
    dist = class_distance_maps(host_class_map(seg), SP)
    _, _, d2 = interface_margin(lesion_class_distances(dist, member_rects([_member(10, 4, 10, 4, 0)], seg.shape)))
    assert d2 == pytest.approx(3 * 5.0)                                 # never <= 4 mm: t <= 4 is an in-plane criterion


def test_a_single_class_gives_an_infinite_second_distance():
    seg = np.full((10, 10, 2), 2, np.int16)
    dist = class_distance_maps(host_class_map(seg), SP)
    c1, d1, d2 = interface_margin(lesion_class_distances(dist, member_rects([_member(1, 2, 1, 2, 0)], seg.shape)))
    assert d1 == 0.0 and d2 == float("inf")


def test_member_rects_clip_to_the_grid_and_drop_boxes_off_it():
    rects = member_rects([_member(-3, 6, 38, 6, 1), _member(5, 3, 5, 3, 9)], (40, 40, 4))
    assert rects == [(0, 3, 38, 40, 1)]
```

```python
# tests/test_brain_lookup.py
import numpy as np

from anatobind.eval.lookup import BRAIN_ALL, BRAIN_PARENCHYMA, BrainLookup


def _seg():
    seg = np.zeros((40, 40, 2), np.int16)
    seg[:20, :, :] = 2            # WM
    seg[20:30, :, :] = 3          # cortex
    seg[30:, :, :] = 24           # CSF; cols 30..40 (the outermost)
    return seg


def test_argmax_overlap_wins_and_reports_the_fraction():
    lk = BrainLookup(_seg(), (0.5, 0.5, 5.0), BRAIN_ALL)
    label, frac = lk.host([(16, 24, 10, 14, 0)])         # cols 16..23: 4 WM columns + 4 cortex columns
    assert label in (2, 3) and frac == 0.5
    label, frac = lk.host([(16, 22, 10, 14, 0)])         # 4 WM + 2 cortex
    assert label == 2 and frac == 4 / 6


def test_zero_overlap_falls_back_to_the_nearest_candidate_in_mm():
    seg = _seg()
    seg[30:, :, :] = 0                                    # background beyond the cortex
    lk = BrainLookup(seg, (0.5, 0.5, 5.0), BRAIN_PARENCHYMA)
    label, frac = lk.host([(35, 38, 10, 14, 0)])
    assert label == 3 and frac == 0.0


def test_parenchyma_candidates_exclude_ventricles_and_csf():
    assert 24 not in BRAIN_PARENCHYMA and 4 not in BRAIN_PARENCHYMA and 24 in BRAIN_ALL
    lk = BrainLookup(_seg(), (0.5, 0.5, 5.0), BRAIN_PARENCHYMA)
    assert lk.host([(32, 36, 10, 14, 0)]) == (3, 0.0)     # the CSF voxels count as no overlap; nearest parenchyma is cortex


def test_no_voxels_on_the_grid_gives_none():
    assert BrainLookup(_seg(), (0.5, 0.5, 5.0), BRAIN_ALL).host([]) == (None, 0.0)
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_geometry.py tests/test_brain_lookup.py -q -p no:cacheprovider`
Expected: ModuleNotFoundError / ImportError。

- [ ] **Step 3: 写 `geometry.py`**

```python
# anatobind/eval/geometry.py
"""Model-free lesion geometry on a SynthSeg label map (v2.6 §4.5, §7.3): the distance from a lesion to each
candidate host class, the interface distance d_interface and the margin Δd that define the hard group H1.

Host classes follow v2.6 §3: cerebral white matter, cortex, thalamus, basal ganglia, brainstem, cerebellum and the
other deep grey structures (hippocampus, amygdala, ventral DC). Left and right are one class because the reader's
primary_host is side-agnostic. Ventricles and CSF are landmarks, not hosts (§25 item 4), so no interface is drawn
against them. Distances are Euclidean distance transforms in mm on the volume's own spacing; with 5 mm slices a
voxel one slice away is >= 5 mm, so t <= 4 mm is an in-plane criterion (§4.5).

Grid convention: label maps are (col, row, slice) like derived/synthseg/*/seg_native (fastmri.rss_h5_to_nifti);
lesion members are RSS-frame boxes (x = col0, y = row0 from the top, width, height, slice).
"""
import numpy as np
from scipy import ndimage

HOST_CLASSES = {
    "white_matter": (2, 41),
    "cortex": (3, 42),
    "thalamus": (10, 49),
    "basal_ganglia": (11, 50, 12, 51, 13, 52, 26, 58),
    "brainstem": (16,),
    "cerebellum": (7, 46, 8, 47),
    "other_deep_grey": (17, 53, 18, 54, 28, 60),
}
CLASS_NAMES = tuple(HOST_CLASSES)                 # class id = index + 1; 0 = not a host
LANDMARKS = (4, 43, 5, 44, 14, 15, 24)            # lateral / inferior-lateral / 3rd / 4th ventricles, CSF


def host_class_map(seg):
    """SynthSeg labels -> host class ids 1..7; ventricles, CSF, background and everything else -> 0."""
    out = np.zeros(np.shape(seg), np.int8)
    for i, labels in enumerate(HOST_CLASSES.values(), start=1):
        out[np.isin(seg, labels)] = i
    return out


def class_distance_maps(class_map, spacing):
    """{class id: distance in mm from every voxel to the nearest voxel of that class}, for the classes present."""
    return {int(c): ndimage.distance_transform_edt(class_map != c, sampling=spacing)
            for c in np.unique(class_map) if c > 0}


def member_rects(members, shape):
    """RSS-frame member boxes -> clipped (col0, col1, row0, row1, slice) rectangles on a (col, row, slice) grid;
    boxes that leave nothing on the grid are dropped."""
    nc, nr, ns = shape
    out = []
    for m in members:
        c0, c1 = max(0, m["x"]), min(nc, m["x"] + m["width"])
        r0, r1 = max(0, m["y"]), min(nr, m["y"] + m["height"])
        if c1 > c0 and r1 > r0 and 0 <= m["slice"] < ns:
            out.append((c0, c1, r0, r1, m["slice"]))
    return out


def lesion_class_distances(dist_maps, rects):
    """Minimum over the lesion's voxels of each class distance map (0 when the lesion touches the class)."""
    return {c: float(min(d[c0:c1, r0:r1, s].min() for c0, c1, r0, r1, s in rects)) for c, d in dist_maps.items()}


def interface_margin(dists):
    """(nearest class, d1, d2): d1 = distance to the nearest host class, d2 = to the second nearest.
    d_interface = d2 (the distance to the nearest interface with another class, within one voxel diagonal when
    the lesion lies inside one class; 0 when it straddles two) and Δd = d2 - d1. inf when only one class exists."""
    order = sorted(dists.items(), key=lambda kv: kv[1])
    c1, d1 = order[0]
    d2 = order[1][1] if len(order) > 1 else float("inf")
    return int(c1), d1, d2
```

- [ ] **Step 4: `lookup.py` 末尾加 `BrainLookup`**

在 `import numpy as np` 下加 `from scipy import ndimage`，文件末尾加：

```python
# --- brain: category-free lookup over SynthSeg labels (v2.6 §4.5; the 2026-09-15 probe's rule) ------------------
BRAIN_ALL = (2, 41, 3, 42, 4, 43, 5, 44, 7, 46, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 14, 15, 16, 17, 53, 18, 54,
             24, 26, 58, 28, 60)
BRAIN_PARENCHYMA = tuple(l for l in BRAIN_ALL if l not in (4, 43, 5, 44, 14, 15, 24))


class BrainLookup:
    """argmax overlap over the candidate labels among the lesion's voxels; zero overlap -> the candidate nearest in
    mm (EDT on the volume spacing). No class restriction: the brain CSV has no host-implying categories.
    seg is a (col, row, slice) label map; rects are (col0, col1, row0, row1, slice) from geometry.member_rects."""

    def __init__(self, seg, spacing, candidates=BRAIN_ALL):
        self.seg = np.asarray(seg)
        self.cand = np.array(sorted(candidates))
        _, idx = ndimage.distance_transform_edt(~np.isin(self.seg, self.cand), sampling=spacing, return_indices=True)
        self.nearest = self.seg[tuple(idx)]

    def host(self, rects):
        """(label, overlap fraction); (None, 0.0) when the lesion has no voxel on the grid."""
        if not rects:
            return None, 0.0
        vals = np.concatenate([self.seg[c0:c1, r0:r1, s].ravel() for c0, c1, r0, r1, s in rects])
        cv = vals[np.isin(vals, self.cand)]
        if cv.size:
            counts = np.bincount(cv, minlength=int(self.cand.max()) + 1)
            return int(np.argmax(counts)), float(counts.max() / vals.size)
        near = np.concatenate([self.nearest[c0:c1, r0:r1, s].ravel() for c0, c1, r0, r1, s in rects])
        counts = np.bincount(near[near > 0], minlength=int(self.cand.max()) + 1)
        return int(np.argmax(counts)), 0.0
```

- [ ] **Step 5: 跑测试，确认通过**

Run: 同 Step 2。Expected: 11 passed。（`test_argmax_overlap_wins_and_reports_the_fraction` 第一段允许平局取任一。）

- [ ] **Step 6: 提交**

```bash
git add anatobind/eval/geometry.py anatobind/eval/lookup.py tests/test_geometry.py tests/test_brain_lookup.py
git commit -m "Eval: host-class distance geometry (d_interface, margin) and the category-free brain lookup"
```

---

### Task 11: Gate 0.5 盘点脚本，跑全集，冻结 t

**Files:**
- Create: `scripts/brain_frame.py`
- Test: `tests/test_brain_frame.py`
- Output: `docs/verification/2026-09-24/gate05/{brain_frame.md,lesions.csv,gate05.json,output.txt}`

**Interfaces:**
- Consumes: Task 2 的 `read_fastmri_plus_rows`、`rows_to_rss_frame`、`merge_boxes_3d`；Task 3 的 `volume_geometry`；Task 10 的 geometry 与 `BrainLookup`。
- Produces: `SMALL_LABELS`、`series_of(file)`、`geometry_stratum(spacing_row_mm, spacing_slice_mm) -> str`、`analyse_volume(seg, zooms, rows, n_rows, meta) -> list[dict]`、`freeze_t(share_at, min_share=0.15) -> int | None`、`summarise(rows) -> dict`；判门行 `GATE05: GO|NO-GO|DECIDE`。

- [ ] **Step 1: 2026-09-24 实测的分层事实（写进脚本 docstring 与报告）**

165 个有小病灶的 FLAIR 卷，按 (系列, RSS 行列, 面内间距, 层厚)：

```
59 x 200 (320x320) 0.6875 / 5.0        33 x 201 (320x320) 0.6875 / 5.0      5 x 201 (320x320) 0.625 / 3.0
 3 x 201 (320x260..290) 0.6875 / 5.0   12 x 202 (320x260) 0.6875 / 5.0      2 x 202 (256x256) 0.8594 / 3.0
 3 x 202 (256x208..224) 0.8594 / 5.0   19 x 203 (213/234/276 sq) 0.8594 / 5.0
 1 x 205 (320x320) 0.6875 / 5.0        4 x 206 (256x224) 0.8594 / 5.0
11 x 209 (320x320) 0.6875 / 5.0       13 x 210 (320x320) 0.6875 / 5.0
```

即 205/209/210 与 200 同分辨率；"低分辨率"的实体是 0.8594 mm 面内的 202(部分)/203/206；另有 7 卷是 3 mm 层厚。分层字符串：`geometry_stratum` 返回 `"inplane_0.69_slice_5"`、`"inplane_0.86_slice_5"`、`"inplane_0.63_slice_3"`、`"inplane_0.86_slice_3"`（面内四舍五入到 0.01，层厚取整）。

- [ ] **Step 2: 写测试**

```python
# tests/test_brain_frame.py
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/brain_frame.py"
    spec = importlib.util.spec_from_file_location("brain_frame", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_freeze_t_takes_the_smallest_t_reaching_the_share_or_none():
    m = _load()
    assert m.freeze_t({2: 0.10, 3: 0.20, 4: 0.30, 5: 0.40}) == 3
    assert m.freeze_t({2: 0.16, 3: 0.20, 4: 0.30, 5: 0.40}) == 2
    assert m.freeze_t({2: 0.01, 3: 0.05, 4: 0.10, 5: 0.12}) is None


def test_series_and_geometry_strata():
    m = _load()
    assert m.series_of("file_brain_AXFLAIR_209_6001234") == "209"
    assert m.geometry_stratum(0.6875, 5.0) == "inplane_0.69_slice_5"
    assert m.geometry_stratum(0.8594, 3.0) == "inplane_0.86_slice_3"


def test_analyse_volume_flips_rows_merges_and_measures_one_lesion():
    m = _load()
    seg = np.zeros((40, 40, 4), np.int16)     # (col, row, slice)
    seg[:20, :, :] = 2                        # WM in cols 0..20
    seg[20:, :, :] = 3                        # cortex beyond
    n_rows = 40
    # CSV frame: y from the bottom. rows 10..14 from the top <=> y = 40 - 14 = 26, height 4
    rows = [{"file": "f", "slice": 1, "x": 10, "y": 26, "width": 6, "height": 4, "label": "Lacunar infarct"},
            {"file": "f", "slice": 2, "x": 10, "y": 26, "width": 6, "height": 4, "label": "Lacunar infarct"}]
    out = m.analyse_volume(seg, (0.5, 0.5, 5.0), rows, n_rows, {"file": "f", "patient_id": "p", "series": "200",
                                                                "spacing_row_mm": 0.5, "spacing_col_mm": 0.5, "spacing_slice_mm": 5.0})
    assert len(out) == 1
    L = out[0]
    assert (L["z0"], L["z1"], L["n_slices"], L["y0"], L["y1"]) == (1, 2, 2, 10, 14)
    assert L["host_lookup_all"] == 2 and L["host_class_nearest"] == "white_matter"
    assert L["d1_mm"] == 0.0 and L["d_interface_mm"] == pytest.approx((20 - 15) * 0.5) and L["delta_d_mm"] == L["d_interface_mm"]
    assert L["status"] == "ok" and L["inplane_mm"] == pytest.approx(6 * 0.5)


def test_summarise_reports_shares_per_t_and_the_majority_flag():
    m = _load()
    rows = [{"d_interface_mm": d, "delta_d_mm": d, "status": "ok", "patient_id": f"p{i % 3}", "n_slices": 1,
             "stratum_series": "200_201", "stratum_geometry": "inplane_0.69_slice_5", "host_class_nearest": "white_matter",
             "host_lookup_all": 2, "host_lookup_parenchyma": 2} for i, d in enumerate([0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 9.0, 10.0])]
    s = m.summarise(rows)
    assert s["n_lesions"] == 10 and s["n_patients"] == 3
    assert s["share_at"][2] == pytest.approx(0.3) and s["share_at"][5] == pytest.approx(0.6)
    assert s["t_frozen"] == 2 and s["majority_flag"] is False
    rows_easy = [{**r, "d_interface_mm": 9.0} for r in rows]
    assert m.summarise(rows_easy)["t_frozen"] is None
```

- [ ] **Step 3: 跑测试，确认失败**

Run: `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/test_brain_frame.py -q -p no:cacheprovider`
Expected: FileNotFoundError。

- [ ] **Step 4: 写脚本**

```python
#!/usr/bin/env python
# scripts/brain_frame.py
"""Gate 0.5 (v2.6 §4.5): geometry of every fastMRI+ FLAIR small lesion (nonspecific white-matter lesion, lacunar
infarct; >= 3 px; the leg 2 merge rule) on the clean SynthSeg parcellation, before any reader sees an image.
Per lesion: the category-free lookup host (33 labels, with and without ventricles/CSF as candidates), the nearest
host class, d1, d_interface (= d2) and Δd (anatobind.eval.geometry), in-plane extent, slice count, series and
measured-geometry strata. Then the share of the hard group at t = 2/3/4/5 mm and the frozen t.

Strata measured on 2026-09-24 over the 165 volumes: 200/201/205/209/210 are 320x320 at 0.6875 mm; the 0.8594 mm
volumes are (part of) 202, 203 and 206; seven volumes have 3 mm slices. The series-number split of v2.6 §7.3
(200/201 vs the rest) is reported alongside the measured one.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_frame.py \
      --out docs/verification/2026-09-24/gate05
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import MIN_BOX_SIDE, merge_boxes_3d, read_fastmri_plus_rows, rows_to_rss_frame  # noqa: E402
from anatobind.data_engine.fastmri_knee import volume_geometry  # noqa: E402
from anatobind.eval.geometry import (  # noqa: E402
    CLASS_NAMES, class_distance_maps, host_class_map, interface_margin, lesion_class_distances, member_rects,
)
from anatobind.eval.lookup import BRAIN_ALL, BRAIN_PARENCHYMA, BrainLookup  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
CSV = FM / "fastMRI_lh_brain_knee/Annotations/brain.csv"
KROOT = FM / "fastMRI_lh_brain_knee/kspace/brain"
SEG_ROOT = FM / "derived/synthseg/fastmri_brain/seg_native"
SMALL_LABELS = ("Nonspecific white matter lesion", "Lacunar infarct")
T_GRID = (2, 3, 4, 5)
MIN_SHARE = 0.15
FIELDS = ["lesion_id", "file", "patient_id", "series", "stratum_series", "stratum_geometry", "label", "z0", "z1", "n_slices",
          "x0", "y0", "x1", "y1", "inplane_mm", "spacing_row_mm", "spacing_col_mm", "spacing_slice_mm",
          "host_lookup_all", "host_lookup_parenchyma", "host_class_nearest", "d1_mm", "d_interface_mm", "delta_d_mm", "status"]


def series_of(file):
    return re.search(r"AXFLAIR_(\d+)_", file).group(1)


def geometry_stratum(spacing_row_mm, spacing_slice_mm):
    return f"inplane_{spacing_row_mm:.2f}_slice_{int(round(spacing_slice_mm))}"


def h5_of(stem):
    for split in ("multicoil_train", "multicoil_val"):
        p = KROOT / split / f"{stem}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def analyse_volume(seg, zooms, rows, n_rows, meta):
    """seg: (col, row, slice) SynthSeg labels; zooms: (col, row, slice) mm; rows: this file's CSV-frame small-lesion
    boxes. One dict per merged lesion."""
    lesions = merge_boxes_3d(rows_to_rss_frame(rows, n_rows), "label")
    class_map = host_class_map(seg)
    dist = class_distance_maps(class_map, zooms)
    look_all, look_par = BrainLookup(seg, zooms, BRAIN_ALL), BrainLookup(seg, zooms, BRAIN_PARENCHYMA)
    out = []
    for L in lesions:
        rects = member_rects(L["members"], seg.shape)
        row = {"file": meta["file"], "patient_id": meta["patient_id"], "series": meta["series"],
               "stratum_series": "200_201" if meta["series"] in ("200", "201") else "other",
               "stratum_geometry": geometry_stratum(meta["spacing_row_mm"], meta["spacing_slice_mm"]),
               "label": L["label"], "z0": L["z0"], "z1": L["z1"], "n_slices": L["z1"] - L["z0"] + 1,
               "x0": L["x0"], "y0": L["y0"], "x1": L["x1"], "y1": L["y1"],
               "inplane_mm": max((L["x1"] - L["x0"]) * meta["spacing_col_mm"], (L["y1"] - L["y0"]) * meta["spacing_row_mm"]),
               "spacing_row_mm": meta["spacing_row_mm"], "spacing_col_mm": meta["spacing_col_mm"], "spacing_slice_mm": meta["spacing_slice_mm"]}
        if not rects or not dist:
            out.append({**row, "host_lookup_all": None, "host_lookup_parenchyma": None, "host_class_nearest": None,
                        "d1_mm": None, "d_interface_mm": None, "delta_d_mm": None, "status": "outside" if not rects else "no_host_class"})
            continue
        c1, d1, d2 = interface_margin(lesion_class_distances(dist, rects))
        out.append({**row, "host_lookup_all": look_all.host(rects)[0], "host_lookup_parenchyma": look_par.host(rects)[0],
                    "host_class_nearest": CLASS_NAMES[c1 - 1], "d1_mm": d1, "d_interface_mm": d2, "delta_d_mm": d2 - d1, "status": "ok"})
    return out


def freeze_t(share_at, min_share=MIN_SHARE):
    for t in T_GRID:
        if share_at[t] >= min_share:
            return t
    return None


def _quantiles(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    return {"n": int(v.size), "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90))} if v.size else {"n": 0}


def summarise(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    d = np.array([r["d_interface_mm"] for r in ok], float)
    share_at = {t: float((d <= t).mean()) if d.size else 0.0 for t in (0, 1) + T_GRID}
    t = freeze_t(share_at)
    per_patient = Counter(r["patient_id"] for r in rows)
    m = np.array(sorted(per_patient.values()), float)
    out = {"n_lesions": len(rows), "n_ok": len(ok), "n_status": dict(Counter(r["status"] for r in rows)),
           "n_patients": len(per_patient),
           "per_patient": {"min": int(m.min()), "median": float(np.median(m)), "max": int(m.max()),
                           "m_eff": float((m ** 2).sum() / m.sum())} if m.size else {},
           "single_slice_share": float(np.mean([r["n_slices"] == 1 for r in rows])) if rows else 0.0,
           "share_at": share_at, "t_frozen": t, "hard_share": share_at[t] if t else None,
           "majority_flag": bool(t is not None and share_at[t] > 0.5),
           "d_interface": _quantiles([r["d_interface_mm"] for r in ok]), "delta_d": _quantiles([r["delta_d_mm"] for r in ok]),
           "host_class_nearest": dict(Counter(r["host_class_nearest"] for r in ok)),
           "host_lookup_all": dict(Counter(r["host_lookup_all"] for r in ok)),
           "host_lookup_parenchyma": dict(Counter(r["host_lookup_parenchyma"] for r in ok)),
           "strata": {}}
    for key in ("stratum_series", "stratum_geometry"):
        for s in sorted({r[key] for r in ok}):
            sel = np.array([r["d_interface_mm"] for r in ok if r[key] == s], float)
            out["strata"][f"{key}={s}"] = {"n": int(sel.size), **{f"share_le_{t}": float((sel <= t).mean()) for t in T_GRID}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_fastmri_plus_rows(CSV) if "AXFLAIR" in r["file"] and r["label"] in SMALL_LABELS
            and r["width"] >= MIN_BOX_SIDE and r["height"] >= MIN_BOX_SIDE]
    by_file = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append(r)
    files = sorted(by_file)[:a.limit] if a.limit else sorted(by_file)
    records = []
    for i, f in enumerate(files):
        g = volume_geometry(h5_of(f))
        img = nib.load(str(SEG_ROOT / f"{f}_seg.nii.gz"))
        seg = np.asarray(img.dataobj).astype(np.int16)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        if seg.shape != (g["n_cols"], g["n_rows"], g["slices"]):
            raise ValueError(f"{f}: seg grid {seg.shape} != RSS (cols, rows, slices) {(g['n_cols'], g['n_rows'], g['slices'])}")
        meta = {"file": f, "patient_id": g["patient_id"], "series": series_of(f), "spacing_row_mm": g["spacing_row_mm"],
                "spacing_col_mm": g["spacing_col_mm"], "spacing_slice_mm": g["spacing_slice_mm"]}
        records += analyse_volume(seg, zooms, by_file[f], g["n_rows"], meta)
        print(f"{i + 1}/{len(files)} {f}: {len(by_file[f])} boxes -> {len(records)} lesions so far", flush=True)
    for k, r in enumerate(records):
        r["lesion_id"] = k
    with open(a.out / "lesions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)
    s = summarise(records)
    verdict = "DECIDE" if s["majority_flag"] else "GO" if s["t_frozen"] else "NO-GO"
    s["verdict"] = verdict
    (a.out / "gate05.json").write_text(json.dumps(s, indent=1))
    lines = ["# Gate 0.5 brain frame (fastMRI+ FLAIR small lesions on clean SynthSeg)", "",
             f"lesions {s['n_lesions']} (ok {s['n_ok']}, {s['n_status']}), patients {s['n_patients']}, per patient {s['per_patient']}",
             f"single-slice share {s['single_slice_share']:.3f}", "",
             "share of lesions with d_interface <= t (host classes of v2.6 §3, sides merged, ventricles/CSF landmarks):",
             "".join(f"  t={t}: {v:.3f}" for t, v in s["share_at"].items()), "",
             f"d_interface quantiles {s['d_interface']}", f"delta_d quantiles {s['delta_d']}", "",
             f"nearest host class: {s['host_class_nearest']}", f"lookup host (all 33): {s['host_lookup_all']}",
             f"lookup host (parenchyma only): {s['host_lookup_parenchyma']}", "", "strata:"]
    lines += [f"  {k}: {v}" for k, v in s["strata"].items()]
    lines += ["", f"t_frozen = {s['t_frozen']} (smallest t in {T_GRID} with share >= {MIN_SHARE}); hard share {s['hard_share']}; majority_flag {s['majority_flag']}",
              f"GATE05: {verdict}"]
    (a.out / "brain_frame.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 跑测试，确认通过**

Run: 同 Step 3。Expected: 4 passed。

- [ ] **Step 6: 先跑 3 卷冒烟，再跑全集**

```bash
cd /data0/congcong/code/Project_Doing/foundation_model-aur
SMOKE=<本会话 scratchpad 目录>/gate05_smoke      # 临时产物放会话 scratchpad，不进仓库
PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=4 nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_frame.py --out $SMOKE --limit 3 | tail -5
D=docs/verification/2026-09-24/gate05; mkdir -p $D
PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=4 nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_frame.py --out $D | tee $D/output.txt
```

期望：全集 `lesions 1297`、`patients 165`（与 09-22 的 `flair_counts_output.txt` 一致：1297 / 165 / 单层 79.3% / 200_201 层 1077）；每卷 2–5 s，共约 10 分钟；`GATE05:` 一行。**若 `DECIDE`（冻结的 t 下困难组过半）或 `NO-GO`：不冻结，报给用户**（D10）。

- [ ] **Step 7: 提交**

```bash
git add scripts/brain_frame.py tests/test_brain_frame.py docs/verification/2026-09-24/gate05
git commit -m "Gate 0.5: geometry inventory of the 1297 FLAIR small lesions and the frozen hard-group threshold"
```

---

### Task 12: 总报告、STATUS / CLAUDE、合回 main、打 tag、推送

**Files:**
- Create: `docs/verification/2026-09-24/REPORT.md`
- Modify: `STATUS.md`（整体重写五段）、`CLAUDE.md`（状态行、测试数、阅读顺序、代码地图、红线）
- Modify: `docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md` §4.5 / §7.3（只加一段"2026-09-24 实测：分层改按实测几何，理由见 brain_frame.md"，不改其余）

- [ ] **Step 1: 写 REPORT.md**

体例同 `docs/verification/2026-09-23/knee_eval/REPORT.md`（每节：结论 → 命令 → 原始输出 → 支撑的决策），章节固定：

1. 结论三行：Gate 0（审计 PASS/FAIL + 膝/脑胜出比例）、H1 重跑（clean × shared4 工作点：阈值、两种灵敏度、每卷假阳、正常卷假阳；`TRANSFER_GATE`；旧 H1 门结果；与 09-15 的对照：当时 300 卷 1067 病灶只检出 208 个、最高分 0.067）、Gate 0.5（1297/165、t 冻结值与困难组占比、`GATE05`）。
2. Gate 0：`relink_output.txt` 摘录、`audit.md` 原样、两张图的两行地址、旧导出未动的 `stat` 输出。
3. H1 重跑：`training_facts.txt`、`summary.md` 原样、`h1.txt` 原样、每族与大小三分位、中心误差/IoU 中位、患者覆盖。
4. Gate 0.5：`brain_frame.md` 原样；系列号分层与实测几何分层的对照；D8/D9 的定义原文。
5. 未做与限制：bone 族不进迁移判据；检出缓存阈值 0.01；16 维几何未做；CI 状态；旧 `runs/detector_fold*` 与旧 `detections/` 保留未删。
6. 下一步建议（按 D7 判据只写一条推荐）。

- [ ] **Step 2: 更新 STATUS.md 与 CLAUDE.md**

STATUS 五段：已完成且已验证（三道门各一段，带命令与输出摘录）；待用户拍板（迁移与否 / ② / Gate 0.5 的 t 若 DECIDE / 分层改按实测几何 / 60 GB npz 仍未删 / 分支仍未删 / 旧遗留）；下一步（迁移 = 写"fastMRI+ 预训练 → SKM-TEA 微调"计划；否则 nnDetection 计划；脑侧计划 2；Level R 协议 PR-B）；坑（fastMRI+ 系列号 ≠ 分辨率；旧导出靠 manifest 拒载；缓存阈值；zsh 数组）；为什么（D1–D13 指向本计划）。CLAUDE.md：状态行改为"Gate 0 已落地（新根 leg2_gate0）、H1 已重跑（结论…）、Gate 0.5 已盘点（t = …）"；测试数改实测；阅读顺序加本计划与 REPORT；代码地图加 `fastmri.py` 的框约定函数、`geometry.py`、`BrainLookup`、四个新脚本；红线里"fastMRI+ 的框 y 从底部数起"改为"只认 RSS 帧导出（transform_version 2），旧根 `leg2/` 拒载"。

- [ ] **Step 3: 全套测试、合回 main、打 tag、推送**

```bash
cd /data0/congcong/code/Project_Doing/foundation_model-aur
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider | tail -1   # 记下 N passed
git add docs/verification/2026-09-24 STATUS.md CLAUDE.md docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md
git commit -m "Status 2026-09-24: Gate 0 landed, knee H1 rerun on corrected boxes, Gate 0.5 inventory and frozen t"
cd /data0/congcong/code/Project_Doing/foundation_model
git merge --no-ff build/aur-system -m "Merge branch 'build/aur-system': Gate 0, knee H1 rerun, Gate 0.5"
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider | tail -1
git tag -a handoff/2026-09-24-gate0-h1-gate05 -m "Gate 0 box conversion, knee H1 rerun, Gate 0.5 brain frame"
git push origin main && git push origin handoff/2026-09-24-gate0-h1-gate05     # TLS 失败时前缀 https_proxy=http://127.0.0.1:7897
git ls-remote origin | grep -E 'main|gate0-h1-gate05'
```

分支 `build/aur-system` 与工作树按用户要求保留。

- [ ] **Step 4: 汇报**

一段话 + REPORT.md 路径 + 三个判门行原文（`GATE0_AUDIT`、`TRANSFER_GATE`、`GATE05`）+ 图的两行地址 + 等用户拍板的清单。

---

## 时间线

| 时段 | 内容 |
|---|---|
| 第 1 天上午 | Task 0–4（代码 + 测试，CPU） |
| 第 1 天中午 | Task 5 relink（分钟级）、Task 6 审计（约 10 分钟）→ Gate 0 判门；Task 7 CI |
| 第 1 天下午 | Task 8 起四折训练（每折约 1–1.7 h，GPU 0/1/4/6）；训练期间写 Task 9–11 的代码 |
| 第 1 天晚 | fold 4；每折结束即缓存检出；Task 11 跑 Gate 0.5 全集（CPU 约 10 分钟） |
| 第 2 天上午 | Task 9 评估 + 旧 H1 门；Task 12 报告、交接 |

止损点：Task 6 审计 FAIL → 停；Task 11 `DECIDE`/`NO-GO` → 不冻结 t，交用户；Task 9 无论 PASS/FAIL 都停下交用户定迁移与否（本计划不做迁移、不做 nnDetection）。
