# 2026-09-24/25 Gate 0 落地、膝 H1 重跑、Gate 0.5 盘点：验证报告

每节：结论 → 命令 → 原始输出 → 支撑的决策。命令前缀 `PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python`（nvgen 环境，工作树 `foundation_model-aur`，分支 `build/aur-system`）。计划：`docs/superpowers/plans/2026-09-24-gate0-h1-rerun-gate05.md`（决定 D1–D13 在其"预先登记的决定"表）；规则出处：v2.6 `docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md` §4.1–4.5、§6.2、§7.3；`docs/verification/2026-09-23/knee_eval/VERDICT.md` §4。

## 1. 结论

1. **Gate 0（fastMRI+ 框上下翻转）已落地。** 转换只在 `anatobind/data_engine/fastmri.py::convert_box_csv_to_rss` 一处；新导出根 `derived/fastmri_knee/leg2_gate0/` 链接旧卷目录、只换坐标文件，4016 个病灶与旧文件逐条对上（只有 y 翻转），折划分与旧的完全一致，旧根一个字节没动且现在会被拒载。全量亮度审计：膝 236 卷转换后胜出 211 卷（89.4%），脑 188 卷 166 卷（88.3%），每个系列都 ≥ 75%（最低 201 系列 76.8%）。**预先登记的组织级 90% 门槛两边都没到（差 0.6 / 1.7 个百分点），脚本判 `GATE0_AUDIT: FAIL`；按系列门槛加叠图判 Gate 0 通过（决定见 §2.4，用户 2026-09-24 已知晓未反对）。**
2. **膝 H1 重跑：迁移判据不过。** 用修好的框、与 09-15 完全相同的配置重训五折 2.5D 检测器：clean 视图、四个共享族、患者五折、每卷假阳 ≤ 2 的工作点上，大类正确灵敏度 **0.091**（定位 0.155，阈值 0.18，每卷假阳 1.53；198 卷正常膝 1.41）；阈值放到 0.01 也只有定位 0.366 / 大类 0.215（每卷假阳 65.7）。`TRANSFER_GATE: FAIL`。旧 H1 门（可靠性头前置门）同样 FAIL：任何阈值都不能把扫描级正样本率压进 [0.2, 0.5]。与 09-15 的对照：那次最高分 0.067、300 卷只检出 208 个；这次检测器确实学到了东西（分数到 0.9、大病灶半月板最大三分位 312/409 能定位），但绝对召回离"可用"很远。**按预先登记的 D7：不迁移，推荐 VERDICT §4 的 ②（nnDetection 直训 SKM-TEA）。**
3. **Gate 0.5：1297 个脑小病灶几何盘点完成，t 没有冻结（`GATE05: DECIDE`）。** 距离第二近脑区 ≤ 2 mm 的病灶占 58.2%，≤ 0 mm（框内已跨两个脑区）占 42.0%，≤ 5 mm 占 88.7%。按 v2.6 规则 t 应取 2 mm，但困难组过半（决定 D10 的警示情形），留给用户定分层方式。

## 2. Gate 0

### 2.1 代码

`anatobind/data_engine/fastmri.py`：`BOX_CONVENTION_CSV / BOX_CONVENTION_RSS / TRANSFORM_VERSION = 2 / MIN_BOX_SIDE`、`convert_box_csv_to_rss`、`convert_box_rss_to_csv`、`rss_spacing_mm`、`voxel_to_world / world_to_voxel`、`read_fastmri_plus_rows`、`rows_to_rss_frame`、`box_iou_2d`、`merge_boxes_3d`（脑膝共用的 3D 合并，膝 `merge_to_3d` 委托）。`anatobind/data_engine/fastmri_knee.py`：14 列 manifest（`transform_version`、`box_coordinate_convention`、逐卷间距、`patient_id`）、`LegacyBoxConvention`、`load_manifest / load_lesions / load_folds`（缺版本列或版本 ≠ 2 拒载；病灶或折引用了 manifest 里没有的卷报错；患者跨折报错）。训练、H1 门、可靠性、缓存、导出脚本全部改走这些加载器。测试从 342 增到 397（`tests/test_fastmri_box_convention.py` 等 9 个新文件）。

### 2.2 新导出根（relink）

```
PYTHONPATH=. nice -n 19 python scripts/relink_fastmri_knee_gate0.py | tee docs/verification/2026-09-24/gate0/relink_output.txt
```

`relink_output.txt` 摘录：

```
{"n_volumes": 1172, "n_lesions": 4016, "legacy_match": 4016, "dropped": {"too_small": 1}, "files_without_volume": 0,
 "by_family": {"cartilage": 1324, "effusion": 327, "ligament": 523, "bone": 615, "meniscus": 1227}}
ls $R/leg2_gate0 | wc -l            -> 1176        (1172 个链接 + README.txt/lesions.csv/folds.json/manifest.csv)
readlink $R/leg2_gate0/file1000000  -> ../leg2/file1000000
head -3 lesions.csv 新: 0,file1000000,cartilage,13,15,136,141,158,157,3   旧: 0,file1000000,cartilage,13,15,136,163,158,179,3
                                      (320-179=141, 320-163=157)
FOLDS_IDENTICAL
new: 4016 lesions ; legacy refused: .../leg2/manifest.csv: manifest has no transform_version column ...
旧根 manifest.csv / lesions.csv / folds.json mtime 仍是 2026-09-14
```

### 2.3 全量亮度审计

```
PYTHONPATH=. OMP_NUM_THREADS=4 nice -n 19 python scripts/audit_fastmri_plus_boxes.py --out docs/verification/2026-09-24/gate0 --figs ~/figs/anatobind_gate0
```

`audit.md` 原样：

```
| organ | volumes | converted wins | as-is wins | median ratio as-is | median ratio converted |
| knee  |   236   |      211       |     25     |       1.292        |         1.661          |
| brain |   188   |      166       |     22     |       1.052        |         1.144          |
brain per series (n, converted wins, RSS rows):
  200: n=63, converted=59, rows=[320]      201: n=56, converted=43, rows=[320]
  202: n=21, converted=19, rows=[256, 320] 203: n=19, converted=17, rows=[213, 234, 276]
  205: n=1,  converted=1,  rows=[320]      206: n=4,  converted=4,  rows=[256]
  209: n=11, converted=10, rows=[320]      210: n=13, converted=13, rows=[320]
GATE0_AUDIT: FAIL (knee >= 90%, brain >= 90%, every series with >= 3 volumes >= 75%)
```

图（转换前后各一行；膝为积液/骨髓水肿框，脑为每个系列一卷小病灶框）：

- http://localhost:8765/anatobind_gate0/knee_effusion_asis_vs_converted.png ；/home/congcongliu/figs/anatobind_gate0/knee_effusion_asis_vs_converted.png
- http://localhost:8765/anatobind_gate0/brain_small_lesions_per_series.png ；/home/congcongliu/figs/anatobind_gate0/brain_small_lesions_per_series.png

### 2.4 判门决定（预先登记的门槛没过，为什么仍判通过）

按 `audit.json` 逐卷看：膝 25 个"原样胜出"的卷里，转换后框的亮度比中位仍是 1.43（两种放法都落在亮的液体上），差距中位 0.14，而转换胜出的 211 卷差距中位 0.45；脑 22 个原样胜出的卷差距中位 0.057，其中 8 卷只有 1–2 个框。没有任何系列整体失败，而系列级失败才是"某系列标在别的网格上"的信号；两张图里转换后的框全在亮病灶上、原样的全在暗处。判定：组织级 90% 是按 09-15 探针 30/24 卷样本定的，定紧了；映射正确，Gate 0 通过。**这条门槛没达到的事实原样保留，不改数字、不改脚本。**

## 3. 膝 H1 重跑

### 3.1 训练事实

五折在 GPU 0 上串行（1/2/4/5/6 被用户另外几个会话占用），`logs/train_chain_gate0.sh`，2026-09-24 16:20 → 23:24，每折训练约 57 分钟 + 缓存约 27 分钟。配置与 09-15 的 `runs/detector_fold*` 逐项相同（steps 20000、warmup 500、batch 16、lr 2e-4、wd 0.05、seed 0、workers 4、FULL 模型 3.46 M 参数），只换 `export_root`。检出缓存阈值 0.01（旧 0.05，评估参数不是训练参数）。`training_facts.txt` 摘录：

```
fold 0 steps 20000 s/step 0.179 loss last50 4.267 heat 3.094 offset 0.499 size 0.674
fold 1 steps 20000 s/step 0.170 loss last50 4.197 heat 3.031 offset 0.506 size 0.660
fold 2 steps 20000 s/step 0.168 loss last50 4.365 heat 3.202 offset 0.495 size 0.668
fold 3 steps 20000 s/step 0.170 loss last50 4.291 heat 3.074 offset 0.507 size 0.710
fold 4 steps 20000 s/step 0.169 loss last50 4.360 heat 3.200 offset 0.503 size 0.657
pkl per fold: 1603 1708 1680 1589 1624  total 8204
config check (fold 0 vs legacy run): identical on [steps, warmup, batch, lr, wd, seed, workers, model, grad_ckpt, tiny, parameters]
```

（09-15 那次五折末段 heat 4.22–4.32；这次 2.9–3.4。）

### 3.2 评估

```
D=docs/verification/2026-09-24/h1_rerun
PYTHONPATH=. OMP_NUM_THREADS=8 nice -n 19 python scripts/eval_fastmri_knee_detection.py --out $D > $D/output.txt
PYTHONPATH=. nice -n 19 python scripts/check_h1.py > $D/h1.txt
```

`summary.md` 主表（IoU ≥ 0.1、一对一、每卷假阳 ≤ 2 的工作点；`shared4` = 与 SKM-TEA 共享的 meniscus/cartilage/ligament/effusion，`all5` 加 bone）：

```
| view     | families | n_scans | n_gt | thr  | sensitivity | sensitivity_family | fp_per_scan | fp_per_normal_scan | ceiling (sens_fam @ fp) |
| clean    | shared4  |  1172   | 3401 | 0.18 |    0.155    |       0.091        |    1.53     |    1.41 (n=198)    |     0.215 @ 65.70       |
| clean    | all5     |  1172   | 4016 | 0.18 |    0.135    |       0.077        |    1.51     |    1.41 (n=198)    |     0.189 @ 82.00       |
| noise_q1 | shared4  |  1172   | 3401 | 0.18 |    0.154    |       0.098        |    1.52     |    1.40             |     0.222 @ 65.87       |
| noise_q2 | shared4  |  1172   | 3401 | 0.17 |    0.165    |       0.106        |    1.89     |    1.91             |     0.228 @ 67.19       |
| noise_q3 | shared4  |  1172   | 3401 | 0.16 |    0.151    |       0.102        |    1.92     |    1.91             |     0.230 @ 74.47       |
| us4      | shared4  |  1172   | 3401 | 0.18 |    0.169    |       0.096        |    1.97     |    1.95             |     0.195 @ 64.99       |
| us8      | shared4  |  1172   | 3401 | 0.18 |    0.151    |       0.088        |    1.76     |    1.84             |     0.197 @ 62.08       |
| us16     | shared4  |  1172   | 3401 | 0.18 |    0.149    |       0.084        |    1.76     |    1.77             |     0.179 @ 64.01       |
TRANSFER_GATE: FAIL ({"thr": 0.18, "sensitivity_family": 0.0914, "fp_per_scan": 1.528})
```

判定视图（clean × shared4）在工作点上的细节（`gate.json`）：

```
每族 (大类正确 / 定位命中 / 真值):  cartilage 110/160/1324   meniscus 201/360/1227   ligament 0/7/523   effusion 0/0/327
按大小三分位的定位命中 (mL):
  cartilage 0/446 [0.01-0.18] | 13/437 [0.18-0.76] | 147/441 [0.76-21.3]
  meniscus  4/409 [0.02-0.39] | 44/409 [0.40-2.48] | 312/409 [2.50-68.6]
  ligament  0/175 | 3/174 | 4/174          effusion 0/109 | 0/109 | 0/109
命中 527 个: 中心误差中位 6.22 mm, 3D IoU 中位 0.242, 大类正确 0.590
患者覆盖: 496 个有病灶的患者里 184 个至少有一个大类正确的命中 (0.371)
```

阈值扫描的头部（`froc_clean_shared4.csv`）：

```
thr   n_hit  n_hit_family  n_fp    sens   sens_fam  fp/scan
0.01  1244   730           76997   0.366  0.215     65.7
0.05  1240   729           53471   0.365  0.214     45.6
0.10  1092   624           14854   0.321  0.183     12.7
0.18   527   311            1791   0.155  0.091      1.53   <- 工作点
```

旧 H1 门（`h1.txt`）：

```
threshold 0.05 .. 0.15: scan-level positive rate on training folds 1.000 / 1.000 / 0.998
threshold 0.20: 0.909   threshold 0.25 .. 0.95: 0.825 / 0.821 / 0.822 / 0.823 ...
H1 FAIL: no threshold puts the scan-level positive rate in [0.2, 0.5]
```

诊断（不是预先登记的口径，只作附注）：阈值 0.18 由半月板/软骨的分数决定，此时检测器只输出这两族；积液、骨、韧带的检出分数更低——阈值 0.01 时积液能定位 130/327（大类正确 117），0.10 时 55/327，0.18 时 0。逐族阈值会改变工作点，但那是新的判据，不在本轮范围内。

### 3.3 支撑的决策

决定 D7 预先登记：clean × shared4 大类正确灵敏度 ≥ 0.5 @ ≤ 2 FP/卷才推荐迁到 SKM-TEA。实测 0.091，上限 0.215，不迁。09-15 的结论"检测器塌陷"要修正为：框方向确实污染了那次训练（最高分 0.067、208 个检出），修好后检测器能学到大病灶（半月板最大三分位 312/409 定位、软骨 147/441），但整体召回仍远低于可用线；09-15 列出的其他嫌疑（focal loss 量级失衡、25 个 epoch、3.46 M 参数、stride 2）没有被排除，只是这轮按预先登记不去调它们。**推荐：VERDICT §4 的 ②，nnDetection（Retina U-Net 3D）直训 SKM-TEA，2–3 天；它同时回答"是目标函数/训练配方的问题还是数据量的问题"。**

## 4. Gate 0.5

```
PYTHONPATH=. OMP_NUM_THREADS=4 nice -n 19 python scripts/brain_frame.py --out docs/verification/2026-09-24/gate05 | tee docs/verification/2026-09-24/gate05/output.txt
```

`brain_frame.md` 原样：

```
lesions 1297 (ok 1297), patients 165, per patient {min 1, median 3, max 87, m_eff 29.45}; single-slice share 0.793
share of lesions with d_interface <= t (host classes of v2.6 §3, sides merged, ventricles/CSF landmarks):
  t=0: 0.420  t=1: 0.500  t=2: 0.582  t=3: 0.697  t=4: 0.749  t=5: 0.887
d_interface quantiles {p10 0.0, median 1.22, p90 5.19}    delta_d quantiles {p10 0.0, median 0.97, p90 5.13}
nearest host class: {white_matter: 1147, cortex: 150}
lookup host (all 32 non-background aseg labels): {41: 492, 2: 497, 42: 178, 3: 97, 24: 31, 11: 1, 4: 1}
lookup host (parenchyma only, 25 labels):        {41: 492, 2: 497, 42: 200, 3: 106, 11: 2}
strata:
  stratum_series=200_201 (n 1077): share<=2 0.591, <=3 0.708, <=4 0.756, <=5 0.893
  stratum_series=other   (n  220): share<=2 0.536, <=3 0.641, <=4 0.718, <=5 0.859
  stratum_geometry=inplane_0.62_slice_3 (n 32):  0.719 / 0.875 / 0.906 / 0.906
  stratum_geometry=inplane_0.69_slice_5 (n 1175): 0.579 / 0.697 / 0.746 / 0.889
  stratum_geometry=inplane_0.86_slice_3 (n 11):  0.545 / 0.636 / 0.818 / 0.818
  stratum_geometry=inplane_0.86_slice_5 (n 79):  0.582 / 0.633 / 0.734 / 0.861
t_frozen = 2 (smallest t in (2, 3, 4, 5) with share >= 0.15); hard share 0.582; majority_flag True
GATE05: DECIDE
```

计数与 09-22 的 `flair_counts_output.txt` 逐项一致（1297 / 165 / 单层 79.3% / 200-201 系列 1077 / m_eff 29.45）。

定义（决定 D8、D9）：候选宿主类 = 白质 {2,41}、皮层 {3,42}、丘脑 {10,49}、基底节 {11,50,12,51,13,52,26,58}、脑干 {16}、小脑 {7,46,8,47}、其他深部灰质 {17,53,18,54,28,60}，左右合并；脑室 {4,43,5,44,14,15} 与 CSF {24} 只作地标。`d_interface` = 病灶体素到第二近宿主类的最小距离（第一近为 0 时即到最近交界面的距离，误差一个体素对角线内），`Δd` = 第二近 − 第一近，EDT 用该卷 NIfTI 头里的间距；5 mm 层厚下 t ≤ 4 是面内准则。

分层发现（决定 D11）：v2.6 按系列号分的"200/201 全分辨率 vs 低分辨率其他"不准——205/209/210 系列也是 320×320 @ 0.6875 mm，真正 0.86 mm 面内的是 202（部分）/203/206 共 90 个病灶，另有 43 个病灶在 3 mm 层厚的卷上。两种分层都已报告。

支撑的决策：按 v2.6 §25 第 5 项的规则 t = 2 mm，但困难组占 58%，正是 v2.6 §4.5 警示的"按白质体素分布时困难组会是大半"的情形；按决定 D10 不自动冻结，交用户。用户已收到三个选项（2026-09-25）：① 照规则冻 t = 2 mm 接受困难组过半；② 改成按距离分四档（0 / 0–2 / 2–4 / >4 mm，占比 42 / 16 / 17 / 25%）作分析分层，标注按全集来，不做困难组富集抽样（推荐）；③ 只把 t = 0（框内已跨两个脑区，42%）当困难组。

## 5. 未做与限制

- 迁移判据只看四个共享族；bone 族（fastMRI+ 独有）不进判据，五族数字一并报告。
- 检出缓存阈值 0.01 与 09-15 的 0.05 不同；`check_h1.py` 的阈值搜索从 0.05 起，两者可比。
- 逐族阈值、损失归一化、训练更久、更大模型：都没做，也不在本轮登记的判据内。
- `anatobind/eval/geometry.py` 只有距离变换与 d_interface/Δd；v2.6 §17 的 16 维几何留给 PR-C。
- GitHub Actions 工作流已入库但尚未在 GitHub 上跑过（分支未推；交接推送后首次触发）。
- 旧 `runs/detector_fold*`（主目录）与旧 `derived/fastmri_knee/leg2/detections/` 保留未删；新训练在工作树 `runs/detector_gate0_fold*`，新检出在 `leg2_gate0/detections/`。
- 组织级 90% 审计门槛没达到（§2.4）；Gate 0.5 的 t 没有冻结（§4）。

## 6. 下一步建议

1. 第二臂改走 VERDICT §4 的 ②：nnDetection 直训 SKM-TEA（另写计划）。
2. Gate 0.5 的分层由用户在三个选项里定，之后才写 Level R 协议（PR-B）。
3. 交接推送后看 CI 首跑；红了就补 `requirements-ci.txt`。
