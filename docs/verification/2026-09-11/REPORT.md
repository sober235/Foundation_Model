# 2026-09-11 验证报告：官方校正版分割、坐标系实测、m1r 导出与查表天花板复核

每一节：结论 → 怎么算的（可直接粘贴的命令）→ 原始输出 → 这个数支撑了哪个决策。所有命令从仓库根目录执行，前缀 `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python`（下文记作 `$PY`）。

## 复现环境

```bash
cd /data0/congcong/code/Project_Doing/foundation_model      # main
/data2/congcong/data/FM_data/SKM-TEA_ltr/segmentation_masks/raw-data-track/   # 官方校正版分割，155 个
/data2/congcong/data/FM_data/derived/skmtea/m1/     # 旧导出（dicom-track 分割），只读，保留
/data2/congcong/data/FM_data/derived/skmtea/m1r/    # 新导出（raw-data-track 分割），图像硬链接 m1
```

---

## 1. 校正版分割从哪儿来

**结论**：SKM-TEA 在 Redivis 上的归属是 AIMI 组织，引用 `aimi.skm_tea:5r8z:v1_0`。`segmentation_masks` 文件表 310 行 = `dicom-track/` 155 + `raw-data-track/` 155。用只读 API token（scope `data.data`，账号 `congcongliu_ai`，2026-09-11 获批数据访问权）逐个下载 raw-data-track 的 155 个 NIfTI，共 45,797,243 字节，每个文件的 md5 与 Redivis 报告的一致。

**怎么做的**

```bash
V=/tmp/claude-1002/-home-congcongliu--claude/ae80a0ab-ab7e-4342-8ff8-78d7232fe5e7/scratchpad/redivis_venv   # pip install redivis==0.20.14
PYTHONPATH=. $V/bin/python scripts/fetch_skmtea_rawtrack_api.py --list       # 只列文件
PYTHONPATH=. $V/bin/python scripts/fetch_skmtea_rawtrack_api.py --download   # 下载 + md5 校验
```

**原始输出（--download 末尾）**

```
downloaded 155 NIfTIs, md5 mismatches: []
raw-data-track files on disk: 155 (expected 155); missing []; unexpected []
```

**为什么必须换**：SKM-TEA 论文附录 A.3——DICOM 经厂商梯度畸变校正，在其上画的分割"与 SENSE 重建图的对应区域对不上"，作者用 b-spline 配准得到校正版；附录 A.6 要求原始数据线一律用校正版。m1 用的是 dicom-track，配的却是 SENSE 图。

---

## 2. 校正版分割的坐标系：恒等，155/155

**结论**：校正版 NIfTI 直接存在 h5 的 (x, y, z) 网格上。16 种"转置 × 翻转"候选里，155 卷（LR 137、RL 18）的最优都是恒等 `----`；最优得分最低 2.354（门槛 1.5），与次优的差距最低 0.260、中位 1.779。dicom-track 需要的转置和 RL 翻转在这里都不适用。

**怎么算的**：对每个候选把分割搬到 h5 网格上，用帧门槛同一个指标打分——回波 1 上髌骨软骨与股骨软骨各自"标签内均值 / 全部标签外均值"的较小者。

```bash
$PY scripts/discover_rawtrack_frame.py --workers 4 | tee docs/verification/2026-09-11/discover_rawtrack_frame_output.txt
```

**原始输出（节选）**

```
155 scans -> docs/verification/2026-09-11/rawtrack_frame.csv
LR: 137 scans, best-frame counts {'----': 137}, deviating []
RL: 18 scans, best-frame counts {'----': 18}, deviating []
RULE {"LR": "----", "RL": "----"}
score_best <= 1.5 (the dicom-track gate line): []
```

**支撑的决策**：`anatobind/data_engine/skmtea.py::RAW_TRACK_FRAMES = {"LR": "----", "RL": "----"}`；`tests/test_rawtrack_frames.py` 把常量和这张 CSV 绑在一起。

---

## 3. 旧导出的分割偏了多少

**结论**：dicom-track 分割相对校正版，逐标签的 Dice 中位 0.70（髌骨软骨）到 0.94（内侧半月板），质心偏移中位 0.3–0.9 mm，最大 3.8 mm。髌骨软骨最差：18 卷 Dice < 0.5，MTR_150 只有 0.002（偏 3.84 mm）。

| 标签 | 结构 | Dice 中位 | Dice 最小 | 偏移中位 mm | 偏移最大 mm | Dice<0.5 的卷数 |
|---|---|---|---|---|---|---|
| 1 | patellar cartilage | 0.700 | 0.002 | 0.92 | 3.84 | 18 |
| 2 | femoral cartilage | 0.850 | 0.325 | 0.58 | 2.72 | 1 |
| 3 | tibial cartilage medial | 0.883 | 0.517 | 0.33 | 1.92 | 0 |
| 4 | tibial cartilage lateral | 0.865 | 0.598 | 0.43 | 2.45 | 0 |
| 5 | meniscus medial | 0.937 | 0.661 | 0.30 | 1.63 | 0 |
| 6 | meniscus lateral | 0.896 | 0.577 | 0.46 | 2.47 | 0 |

**怎么算的**：同一次 `discover_rawtrack_frame.py` 运行，把 dicom-track（按旧规则 `T---`/`T--z` 搬到 h5 网格）与校正版（恒等）逐标签比较，逐卷结果在 `rawtrack_frame.csv` 的 `dice_*`、`shift_mm_*` 列。

**顺带破了一个旧案**：MTR_150 在旧帧门槛上得 1.408（< 1.5），当时归因为 16 线圈组和髌骨软骨全层缺损。换成校正版分割后它得 3.386。真正的原因是 dicom-track 的髌骨软骨 mask 在这卷上错位了 3.84 mm。`STATUS.md` 里"MTR_150 帧门槛失败已破案，不改"这条要更新。

**支撑的决策**：软骨只有 2–3 mm 厚，1 mm 上下的错位足以让查表在干净图上判错、也足以污染 nnU-Net 的训练标签。G2 只能在校正版上判。

---

## 4. m1r 导出：只换分割和宿主判定

**结论**：155 卷全部成功；图像与 m1 是同一批 inode（硬链接），只重做了 `seg.nii.gz` 和 `boxes.csv`。宿主标签只有 1 行变化：MTR_110 ann 15 从 medial（m1，靠外扩 4 体素才判出）变成 unresolved。组织族真值来自 tissue_id，不受影响。

```bash
$PY scripts/build_skmtea_m1r.py --workers 4 | tee docs/verification/2026-09-11/build_m1r_output.txt
stat -c '%i %n' /data2/congcong/data/FM_data/derived/skmtea/m1/MTR_110/image_us16_e1.nii.gz \
                /data2/congcong/data/FM_data/derived/skmtea/m1r/MTR_110/image_us16_e1.nii.gz
```

**原始输出**

```
155 scans, 0 failed: []
in_seg host_side m1 : {'medial': 76, 'single': 160, 'lateral': 68, 'single_no_overlap': 5, 'unresolved': 2}
in_seg host_side m1r: {'medial': 75, 'single': 159, 'lateral': 68, 'single_no_overlap': 6, 'unresolved': 3}
host_label changed on 1 rows (scan, ann_id, m1, m1r):
   MTR_110 15 5
236129999 .../m1/MTR_110/image_us16_e1.nii.gz
236129999 .../m1r/MTR_110/image_us16_e1.nii.gz
```

**支撑的决策**：训练缓存、nnU-Net 数据集和全部评估都建在 m1r 上；m1 保留，09-08 与 09-09 的报告仍可复现。

---

## 5. 查表天花板：m1 复现与 m1r 复核

**结论**：09-09 的数字在 m1 上逐行复现（`diff` 为空）。换成校正版分割后，类别感知查表（V1：按类别限定候选 + IoA 最大 + 零重叠取最近）的组织族级准确率从 0.968 降到 **0.958**（n=308），软骨病变子集从 0.952 降到 0.938，半月板撕裂仍是 1.000。可争空间约 4 个点，与 09-09 的"约 3 个点"同量级，§13.6 的判据设计不变。

| 口径（组织族级） | m1（dicom-track，09-09） | m1r（raw-data-track） |
|---|---|---|
| 可评分实例（宿主已解析） | 309 | 308 |
| V0 不看类别的 argmax IoA | 0.864 | 0.854 |
| **V1 类别感知 + 最近兜底** | **0.968** | **0.958** |
| V2 类别感知 + 外扩 IoA + 最近兜底 | 0.948 | 0.948 |
| V3 类别感知、只看最近 | 0.822 | 0.825 |
| 半月板撕裂 V1（n=101 / 100） | 1.000 | 1.000 |
| 软骨病变 V1（n=208） | 0.952 | 0.938 |

**怎么算的**

```bash
D=docs/verification/2026-09-11; FM=/data2/congcong/data/FM_data/derived/skmtea
$PY $D/lookup_ceiling_root.py $FM/m1  /tmp/lookup_m1.json          > $D/lookup_ceiling_m1_output.txt
diff $D/lookup_ceiling_m1_output.txt docs/verification/2026-09-09/lookup_ceiling_output.txt && echo "m1 reproduces 09-09"
$PY $D/lookup_ceiling_root.py $FM/m1r $D/lookup_ceiling_m1r.json    > $D/lookup_ceiling_m1r_output.txt
```

`lookup_ceiling_root.py` 是 09-09 脚本的逐字复制，只把导出根目录和输出路径改成了命令行参数。

**原始输出**：`lookup_ceiling_m1_output.txt`（与 09-09 一致）、`lookup_ceiling_m1r_output.txt`（完整）；逐实例结果在 `lookup_ceiling_m1r.json`。m1r 上 V2 仍答错的 16 个实例全是软骨病变，其中 MTR_020 ann 67、MTR_101 ann 241、MTR_150 ann 124 的框与标注宿主在校正版分割上零重叠，最近宿主体素分别在 3.75 / 9.38 / 2.62 mm 外。

**支撑的决策**：校正后天花板略降，说明旧分割的错位曾让查表"多对"了几个实例，但没有改变结论：干净数据上没有 10 个点可赢，G2 仍按 §13.6 在退化条件下、用 nnU-Net 分割判。

---

## 6. 待办

| # | 事项 | 状态 |
|---|---|---|
| 1 | m1r 训练缓存（`derived/skmtea/m1r_cache`） | 进行中 |
| 2 | nnU-Net 数据集（1860 例，硬链接）与预处理 | 进行中 |
| 3 | fold-0 上游试跑，48 h 预算核对 | 缓存完成后开始 |
| 4 | `STATUS.md` 里 MTR_150 那条按第 3 节更新 | 交接时 |
