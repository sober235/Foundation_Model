# 2026-09-23 膝侧 A/U/R 能力系统（计划 1）验证报告

每节：结论 → 命令 → 原始输出 → 支撑的决策。前缀 `source scripts/nnunet_env.sh; PYTHONPATH=. python`（nvgen 环境）。计划：`docs/superpowers/plans/2026-09-23-knee-capability-system.md`；规格：`docs/superpowers/specs/2026-09-23-aur-capability-system-design.md`。

## 1. 结论

三条目标在 SKM-TEA 膝关节、按患者五折、留出扫描上的数字（clean 视图，`summary.md` / `gate.json`）：

```
| view  | n_scans | n_gt | thr  | sensitivity | sensitivity_family | fp_per_scan |
| clean |   155   | 465  | 0.75 |   0.282     |       0.254        |    1.57     |
每族(工作点): 软骨 25/208 (0.12), 积液 64/116 (0.55), 韧带 9/38 (0.24), 半月板 20/103 (0.19)
绑定(命中上): n=54, 组织族正确 0.796, 侧别 0.808 (n_side=26)
绑定(给定标注框对照): n=311, 组织族正确 0.961, 侧别 0.979 (n_side=143)
解剖 Dice (Dataset901, 六类): 0.886 / 0.873 / 0.852 / 0.858 / 0.845 / 0.839
GATE: FAIL (thr 0.75, sensitivity_family 0.254, fp_per_scan 1.57, n_scans 155, n_gt 465)
```

- **目标 1（解剖分割）**：Dataset901 nnU-Net，六类留出 Dice 见上表，与 09-13 的记录（0.841–0.862 均值）一致。
- **目标 2（病灶框 + 大类）**：Dataset902 箱填 nnU-Net，**门不过**：工作点大类正确灵敏度 0.254、每卷假阳 1.57，阈值上限 0.394，解码变体不改善；小病灶（软骨 < 1.8 mL、半月板 < 0.4 mL）几乎全漏，积液可用（0.55）。裁决与第二臂选项在 `VERDICT.md`。
- **目标 3（所在结构）**：在命中的病灶上组织族正确 0.796（大类判对时 0.96）、侧别 0.81；给定标注框的对照 0.961 / 0.979，与 09-09 查表天花板一致。目标 3 的实现（类别限定查表 + 集合值 + 侧别）没有问题，它的数字受目标 2 的召回限制。
- **入口** `scripts/infer_knee.py`：与折末验证逐体素一致；DICOM 世界帧路径可用（§5）。

## 2. 训练事实

Dataset902（`anatobind/nnunet/prepare_lesion.py`）：1860 例 = 155 扫描 × 12 视图（干净 ×6 + 退化 6 档），与 Dataset901 同折同名；标签 = 箱填掩膜（`lesion_labels.py`，积液先、大框先小框后）。预处理 12 分钟（`logs/nnunet902_preprocess.log`）。

训练：`nnUNetv2_train 902 3d_fullres <fold> -tr nnUNetTrainer_250epochs --npz`，patch 96×160×160、batch 2、间距 0.8/0.625/0.625。fold 1→GPU 0、fold 2→GPU 1、fold 3→GPU 4（11:41 起）、fold 0→GPU 6（11:51 起，首次因 zsh 数组下标从 1 起数拿到空卡号而崩，重启）、fold 4→GPU 0（15:54 起，fold 1 验证齐后由 `logs/launch_fold4_when_fold1_done.sh` 自动启动）。

```
for f in 0 1 2 3 4; do L=$(ls -t $nnUNet_results/Dataset902_SKMTEAlesion/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_$f/training_log_*.txt | head -1); echo "fold $f epochs $(grep -c 'Epoch time' $L) mean $(grep -o 'Epoch time: [0-9.]*' $L | awk '{s+=$3} END {printf "%.1f", s/NR}')s pseudo-dice $(grep 'Pseudo dice' $L | tail -1 | grep -o '\[.*\]')"; done
```

```
fold 0: epochs 250, mean epoch time 52.6s, final pseudo dice [0.0981, 0.0477, 0.1601, 0.5413]   (软骨, 半月板, 韧带, 积液)
fold 1: epochs 250, mean epoch time 52.2s, final pseudo dice [0.1210, 0.1710, 0.1480, 0.5529]
fold 2: epochs 250, mean epoch time 52.7s, final pseudo dice [0.1439, 0.1395, 0.1050, 0.5618]
fold 3: epochs 250, mean epoch time 52.9s, final pseudo dice [0.1243, 0.1807, 0.1230, 0.6649]
fold 4: epochs 250, mean epoch time 50.7s, final pseudo dice [0.1142, 0.1020, 0.2720, 0.6491]
```

每折约 3.7 h 训练 + 约 20 分钟折末验证（217 例，含 npz 概率）。结果目录 `derived/nnunet/results/Dataset902_SKMTEAlesion/` 约 179 GB（npz 概率占大头）。烟雾门：第 20 轮只有积液伪 Dice > 0.05，第 50 轮四折四族全部 > 0.03（`logs/nnunet902_fold*.log` 因 stdout 缓冲滞后，看 `training_log_*.txt`）。

## 3. 五折评估（clean 视图，判门）

```
D=docs/verification/2026-09-23/knee_eval
PYTHONPATH=. python scripts/eval_knee_folds.py --out $D --views clean | tee $D/output.txt
PYTHONPATH=. python $D/miss_analysis.py $D clean
PYTHONPATH=. python $D/decode_variants.py --folds 0,1,2,3,4
```

`summary.md` 原样：

```
# Knee capability evaluation

IoU >= 0.1, FP budget 2.0/scan, gate on `clean` family sensitivity >= 0.5

| view | n_scans | n_gt | thr | sensitivity | sensitivity_family | fp_per_scan |
|---|---|---|---|---|---|---|
| clean | 155 | 465 | 0.75 | 0.282 | 0.254 | 1.57 |

## clean

per family at the operating point: Cartilage Lesion: 25/208 (0.12), Effusion: 64/116 (0.55), Ligament Tear: 9/38 (0.24), Meniscal Tear: 20/103 (0.19)

binding on hits (system): n=54, family correct 0.796, side correct 0.808 (n_side=26)
binding on annotated boxes (given-box control): n=311, family correct 0.961, side correct 0.979 (n_side=143)
anatomy Dice (Dataset901 vs export seg): 1: 0.886, 2: 0.873, 3: 0.852, 4: 0.858, 5: 0.845, 6: 0.839

GATE: FAIL ({"view": "clean", "pass": false, "thr": 0.75, "sensitivity_family": 0.2537634408602151, "fp_per_scan": 1.5677419354838709, "n_scans": 155, "n_gt": 465})
```

阈值扫描 `froc_clean.csv`：

```
 0.05 hit 183 fam 155 fp 670 sens 0.394 sens_fam 0.333 fp/scan 4.32
  0.1 hit 183 fam 155 fp 670 sens 0.394 sens_fam 0.333 fp/scan 4.32
 0.15 hit 183 fam 155 fp 670 sens 0.394 sens_fam 0.333 fp/scan 4.32
  0.2 hit 183 fam 155 fp 670 sens 0.394 sens_fam 0.333 fp/scan 4.32
 0.25 hit 183 fam 155 fp 670 sens 0.394 sens_fam 0.333 fp/scan 4.32
  0.3 hit 183 fam 155 fp 670 sens 0.394 sens_fam 0.333 fp/scan 4.32
 0.35 hit 183 fam 155 fp 670 sens 0.394 sens_fam 0.333 fp/scan 4.32
  0.4 hit 183 fam 155 fp 663 sens 0.394 sens_fam 0.333 fp/scan 4.28
 0.45 hit 183 fam 155 fp 658 sens 0.394 sens_fam 0.333 fp/scan 4.25
  0.5 hit 182 fam 154 fp 637 sens 0.391 sens_fam 0.331 fp/scan 4.11
 0.55 hit 181 fam 153 fp 571 sens 0.389 sens_fam 0.329 fp/scan 3.68
  0.6 hit 178 fam 151 fp 502 sens 0.383 sens_fam 0.325 fp/scan 3.24
 0.65 hit 166 fam 143 fp 419 sens 0.357 sens_fam 0.308 fp/scan 2.70
  0.7 hit 157 fam 137 fp 341 sens 0.338 sens_fam 0.295 fp/scan 2.20
 0.75 hit 131 fam 118 fp 243 sens 0.282 sens_fam 0.254 fp/scan 1.57
  0.8 hit 102 fam  91 fp 153 sens 0.219 sens_fam 0.196 fp/scan 0.99
 0.85 hit  53 fam  51 fp  55 sens 0.114 sens_fam 0.110 fp/scan 0.35
  0.9 hit   2 fam   2 fp   3 sens 0.004 sens_fam 0.004 fp/scan 0.02
 0.95 hit   0 fam   0 fp   0 sens 0.000 sens_fam 0.000 fp/scan 0.00
```

漏检分层（`miss_analysis_clean.txt`）：

```
clean: 465 annotated boxes in 130 scans, operating threshold 0.75

localisation hits (IoU >= 0.1 at the operating point) by family x size tercile [mL range]:
  Cartilage Lesion  31/208 (0.15)  terciles: 2/69 [0.0-1.8] | 11/69 [1.8-6.0] | 18/70 [6.1-68.6]
  Effusion          68/116 (0.59)  terciles: 9/38 [0.1-58.2] | 25/39 [59.5-170.1] | 34/39 [173.2-637.9]
  Ligament Tear     9/38 (0.24)  terciles: 1/12 [0.0-3.4] | 1/13 [4.9-13.0] | 7/13 [13.1-40.2]
  Meniscal Tear     23/103 (0.22)  terciles: 1/34 [0.0-0.4] | 9/34 [0.5-2.7] | 13/35 [3.1-31.3]

hits: predicted family = truth?  118/131
hit scores: min 0.75, median 0.84, max 0.90
scans with no hit at all: 40/130

binding on hits (in-seg only): correct: 43, wrong_class: 9, wrong_host: 2
```

解码变体（`decode_variants_clean.txt`）：

```
clean: 155 scans, folds [0, 1, 2, 3, 4]
p_level | thr | sensitivity | sensitivity_family | fp_per_scan | ceiling(sens @ thr 0.05, fp/scan) | per family at op
 0.50 | 0.75 | 0.303 | 0.269 | 1.72 | 0.391 @ 4.24 | Cart 32/208, Effu 64/116, Liga 9/38, Meni 20/103
 0.30 | 0.65 | 0.295 | 0.260 | 1.58 | 0.430 @ 4.77 | Cart 28/208, Effu 62/116, Liga 9/38, Meni 22/103
 0.20 | 0.55 | 0.314 | 0.269 | 1.81 | 0.441 @ 5.33 | Cart 31/208, Effu 63/116, Liga 9/38, Meni 22/103
 0.10 | 0.45 | 0.301 | 0.249 | 1.70 | 0.458 @ 6.13 | Cart 28/208, Effu 61/116, Liga 7/38, Meni 20/103
```

## 4. 退化视图附注（`knee_eval_allviews/summary.md`）

七个视图各自的工作点(命令 `scripts/eval_knee_folds.py --out docs/verification/2026-09-23/knee_eval_allviews`,records/froc 见该目录):

```
| view | n_scans | n_gt | thr | sensitivity | sensitivity_family | fp_per_scan |
|---|---|---|---|---|---|---|
| clean | 155 | 465 | 0.75 | 0.282 | 0.254 | 1.57 |
| noise_q1 | 155 | 465 | 0.75 | 0.284 | 0.256 | 1.59 |
| noise_q2 | 155 | 465 | 0.75 | 0.284 | 0.252 | 1.61 |
| noise_q3 | 155 | 465 | 0.75 | 0.299 | 0.262 | 1.61 |
| us4 | 155 | 465 | 0.75 | 0.277 | 0.241 | 1.74 |
| us8 | 155 | 465 | 0.75 | 0.284 | 0.243 | 1.75 |
| us16 | 155 | 465 | 0.75 | 0.280 | 0.245 | 1.77 |
```

退化视图的灵敏度与 clean 在同一水平(噪声、欠采都没有把本就找不到的小病灶找回来,也没有明显丢掉大病灶);门只按 clean 判。

## 5. 入口回归与世界帧核查

- `infer_regression.txt`：MTR_010（fold 0 留出）经 `scripts/infer_knee.py --frame h5 --folds 0` 与 nnU-Net 折末验证输出比较：解剖六类 Dice ≥ 0.9999、逐体素一致率 1.0；病灶标签图逐体素一致率 1.0（29 452 vs 29 449 体素）。
- `dicom_world_frame.md`：SKM-TEA DICOM 一卷（两回波同一 SeriesInstanceUID，按 EchoNumbers 拆分后 dicom2nifti 转出，轴码 (P, I, L)）经 `--frame world` 转到 (I, P, R)，解剖六类全出、软骨亮度分数 3.895（门线 1.5），病灶表与 h5 帧逐条对应（面内坐标 ×2）。
- 图：http://localhost:8765/anatobind_knee/MTR_010_dicom_world_overlay.png ；/home/congcongliu/figs/anatobind_knee/MTR_010_dicom_world_overlay.png（h5 帧对照 MTR_010_clean_fold0_overlay.png）。

## 6. 未做与限制

- 韧带撕裂与积液没有分割结构，所在结构分别记 `unknown` / `none`（spec §3.3）。
- "给定标注框"一列是对照，不是系统能力；伪 Dice 与训练日志里的数字不是证据。
- 第二臂未跑；脑侧（计划 2）未开始；(b) 阶段网络未动。
- 179 GB 结果目录里的 npz 概率只在评估时用到，是否删除由用户定。
