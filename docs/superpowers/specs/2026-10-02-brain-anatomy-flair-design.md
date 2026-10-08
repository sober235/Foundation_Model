# S4 脑部解剖层：在 fastMRI 样式的 FLAIR 上输出绑定用的解剖（设计规格，2026-10-02）

用户 2026-09-30 分段点头（概览 → 数据 → 仿真 → 去颅骨 → 验证与达标线 → 推理），2026-10-02 "yes"。
上位文件：`docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md`（v2.6，§0 目标、§5 Stage A、§15 数据集角色、§25 决定记录）。
相关记录：`docs/verification/2026-09-29/s4_probe/`（SynthSeg 在 fastMRI FLAIR 上的轮廓）、`docs/verification/2026-09-30/s4_sibbms_synthseg/`（SibBMS 老师标签）、`docs/verification/2026-09-30/s4_probe2/`（fastMRI FLAIR 栈的几何与覆盖）。

## 1. 目标与边界

目标 ①（感知解剖）在目标域上的补课：S7 的绑定与整句用的解剖 A 是 SynthSeg 在 1 mm T1 上的伪标签；fastMRI 的病例没有 T1，SynthSeg 直接跑 5 mm、16 层的 FLAIR 栈时最低两层和颅顶不可靠（`s4_probe`），而 fastMRI+ 的 1297 个病灶里 264 个（20%）整个落在最低两层。Level R 医生标签将在 fastMRI 上标"病灶在哪个结构"，是 R 唯一的人类真值；没有可靠的 A，这些病灶无法与医生比。

S4 产出两个模型和一条推理链：
- **脑轮廓模型**：fastMRI FLAIR 单层 → 脑/非脑，推理前去颅骨。
- **学生解剖模型**：去了颅骨的 fastMRI 样式 FLAIR 栈 → 7 类宿主结构分左右（脑干不分）+ 脑室 + 背景（§4），即 S7 绑定查表（`anatobind/bind/brain_lookup.py`，`anatobind/eval/geometry.py::HOST_CLASSES`）用到的那一套，不多不少。
- **推理入口**：fastMRI h5（或 NIfTI 栈）→ RSS → 去颅骨 → 学生 → SynthSeg 标签值的解剖图 → 直接喂给 `BrainBinder`。

不做：通用 FLAIR 解剖模型、别的粒度、脑叶、SibBMS 上的精度追求、序列物理仿真、颅骨合成、多器官。所有数字都是与伪标签的一致率，**NOT_EVIDENCE**；S4 的最终判定推迟到 Level R 医生标签可用时（A12）。

## 2. 决定（A1–A17）

| 编号 | 决定 |
|---|---|
| A1 | 老师 = SynthSeg robust 2.0 在同病人 1 mm T1 上的标签（已有 `derived/synthseg/{pdgm,bmsr,sibbms}/seg_native`），搬到同空间的 FLAIR 上；不新跑 SynthSeg，不手工标注。 |
| A2 | 训练来源三个都用：SibBMS（358 例 / 185 人；2026-10-08 更正：可用老师图 362 张里有 4 次 MS 检查没有 FLAIR，实际入组 358，见 `build/sources.txt`）、UCSF-PDGM（501）、UCSF-BMSR（461）。排除名单：`s4_sibbms_synthseg/README.md` 的 8 张失败图与 1 个二维文件（`sources.py::EXCLUDED_SIBBMS`）。fastMRI 的 447 卷只做验证，不进训练（半监督留给以后）。 |
| A3 | 划分按病人：每个来源各留 20% 病人做测试（SibBMS 的多次检查不跨集；PDGM 的随访 `_FU` 跟基线同人；BMSR 字母后缀同人），其余 80% 给 nnU-Net（它自己再分五折，只训 fold 0）。测试病人不进 nnU-Net 的数据集目录。 |
| A4 | 学生标签空间 16 类（§4）：背景、7 类宿主 × 左右（脑干一类，不分侧）、脑室。训练用紧凑编号 0–15，推理时映射回 SynthSeg 标签值，`BrainBinder` 不改。 |
| A5 | 仿真离线预生成（nnU-Net 从预处理数据训练，不改它的数据加载）：每个训练卷生成 K = 4 个随机样板，测试卷各 1 个（固定种子）。 |
| A6 | 样板几何按 `s4_probe2` 的实测：16 层 × 5 mm（10% 的样本 14 层）；**样板从颅顶往下放**：栈顶在"最后一层有脑（>5 cm²）"之上空 e 层，e 从实测分布抽（1/2/3/4/5 层各占 4.5%/35%/43%/14%/3.5%），底层 = 栈顶 − 16 × 5 mm（不低于体积底）；底层的脑截面占比只作统计量记录（实测中位数 0.83 受最低层伪标签漏标影响，偏低，不拿它定位，计划写作时 2026-10-02 改）；绕左右轴 ±10°、绕前后轴 ±5° 倾斜；每 5 mm 内 5 层取平均（不加层间隙）；面内 0.69 mm（82%）或 0.86 mm（18%）；矩阵按实测三种 FOV（320×320、260×320、276×276）随机取。 |
| A7 | 强度只做增广，不做序列物理：随机对比度/伽马、低频偏置场、Rician 噪声、轻微模糊，每个样本一套；加上 nnU-Net 自带的增广。 |
| A8 | 标签同步变换：几何与图像相同；1 mm → 5 mm 用每 5 层多数票（平局取靠近样板中心的那层），面内最近邻；病灶区（PDGM 整瘤标注含水肿、BMSR 转移瘤标注）标为 nnU-Net 的 `ignore` 标签（编号 16，最高），不计损失；SibBMS 的 MS 病灶没有同网格标注（`Output/Annotation` 只有 10 个掩膜且在另一网格上），不做忽略。 |
| A9 | 学生模型 = nnU-Net v2 `3d_fullres`，`nnUNetTrainer_250epochs`，patch 由规划决定（预期 [16, 320, 320]，与 Dataset903 同），只训 fold 0；数据集 `Dataset907_BrainAnatomyFLAIR`。 |
| A10 | 脑轮廓模型 = nnU-Net v2 `2d`，`nnUNetTrainer_250epochs`，fold 0；数据集 `Dataset908_FastMRIBrainOutline`：fastMRI 447 卷里排除 14 卷失败卷（`s4_probe` 的"轮廓不到 300 mL"名单）后的 433 卷，标签 = SynthSeg 标签 > 0 填实取最大连通块；每卷第 2 层到"面积 > 5 cm² 的最高层减 1"之间的层是监督层，其余层整层标 `ignore`；按 fastMRI 的 `patient_id` 分，留 20% 病人测试。 |
| A11 | 达标线（写死，不调参救线，M5/D9 同理）：在 fastMRI 的 447 卷（2026-10-08 注：实际是其中轮廓可用的 433 卷，14 卷 SynthSeg 轮廓 < 300 mL 的失败卷没有可用参照，其 30 个框不入分母）、第 2 层到颅顶下一层：① 用 fastMRI+ 的 1297 个框查表，学生 A 与 SynthSeg A 的主结构一致率 ≥ 0.90；② 13 个宿主类（6 对分侧 + 脑干；脑室是地标，不入门；原文误写 14）对 SynthSeg 的平均 Dice ≥ 0.80；③ 脑轮廓模型在监督层对 SynthSeg 轮廓的 Dice ≥ 0.97。三条都过才算过；不过就报告并停。最低两层与颅顶只报告（输出的体积分布 + 蒙太奇，USER_REPORTED），不判。 |
| A12 | S4 的最终判定推迟到 Level R 医生标签可用时：到时比较"学生 A 查表的主结构"与"医生标的主结构"在最低两层上的一致率，与现在的伪标签 A 对照。本规格只到 A11。 |
| A13 | 仿真域测试集（三来源各 20% 病人，各 1 个样板）只报告 Dice，不设线：它只证明学生学会了老师。 |
| A14 | 去颅骨不装 HD-BET 进主链；HD-BET 只作可选抽查（20 卷）对照，装不装、何时装由控制方在计划里定。 |
| A15 | 算力：只用空卡（`nvidia-smi` 当时无人用的卡），每个训练一张卡；CPU 任务 `nice -n 19`、总线程 ≤ 48；预估 > 24 h 的任务先问用户。不删任何文件：每个输出目录都"先检查、后建"，重跑换新名字。 |
| A16 | 记录：`docs/verification/2026-10-02/brain_anatomy_flair/`（建数据集、仿真样例蒙太奇、训练启动与时长、评估报告、冒烟、README 索引）；绑定数字全部标 NOT_EVIDENCE。 |
| A17 | （2026-10-03 增补，A9 的训练器以本条为准）学生模型不做镜像增强。学生的类别分左右；nnU-Net 默认的镜像把图像和标签一起翻转、标签值不变，左侧结构会以同一个标签出现在两侧，左右学不出来。第一次训练（`nnUNetTrainer_250epochs`）前 8 轮的伪 Dice 正是这个样子：白质 左 0.18 / 右 0.50，皮层 左 0.09 / 右 0.45（记录见 `launch.md`）。训练器改为 nnU-Net 自带的 `nnUNetTrainer_250epochs_NoMirroring`：训练不镜像，推理也不做镜像平均；其余增强（含面内旋转，旋转不改变左右手性）保持默认。轮廓模型不分左右，仍用 `nnUNetTrainer_250epochs`。 |

## 3. 数据

| 来源 | 配对 FLAIR | 老师图 | 网格 | 病灶忽略区 | 病人键 |
|---|---|---|---|---|---|
| SibBMS（`SibBMS_ms/sibbms/Output/{MS,Norm}`） | 358（老师图 362 张，4 次检查无 FLAIR） | `derived/synthseg/sibbms/seg_native/<Cohort>_sub-XXX_ses-YYY_T1w_seg.nii.gz` | 197×233×189，1 mm，RAS，已去颅骨，模板空间 | 无 | `<Cohort>_sub-XXX` |
| UCSF-PDGM（`UCSF-PDGM_lh/…/UCSF-PDGM-XXXX_nifti/*_FLAIR.nii.gz`） | 501 | `derived/synthseg/pdgm/seg_native/UCSF-PDGM-XXXX_T1_seg.nii.gz` | 1 mm 各向同性 | 整瘤标注（`anatobind.nnunet.brain_disease.label_path("glioma", case)`，值 1/2/4） | `UCSF-PDGM-XXXX`（随访同人） |
| UCSF-BMSR（`UCSF-BMSR_cbb/…/<case>/<case>_FLAIR.nii.gz`） | 461 | `derived/synthseg/bmsr/seg_native/<case>_T1pre_seg.nii.gz` | 1.5 × 0.86 × 0.86 mm | 转移瘤标注（`label_path("metastasis", case)`） | 数字前缀（字母后缀同人） |

通道文件与老师图同网格这一点 S7 已核对（`checks/channel_grid`：3887 个通道文件与解剖图仿射完全相同）；SibBMS 的 FLAIR 与 T1w 同形状同仿射（本规格的建数据集脚本再断言一次）。BMSR 的 FLAIR 不是 1 mm：仿真前先把图像与标签重采样到 1 mm 各向同性（图像三线性、标签最近邻），再走统一的样板流程。

## 4. 标签空间

| 紧凑编号 | 名称 | SynthSeg 标签 | 推理输出的 SynthSeg 值 |
|---|---|---|---|
| 0 | background | 其余全部（含 CSF 24、未列的小结构、背景） | 0 |
| 1 / 2 | white_matter 左 / 右 | 2 / 41 | 2 / 41 |
| 3 / 4 | cortex 左 / 右 | 3 / 42 | 3 / 42 |
| 5 / 6 | thalamus 左 / 右 | 10 / 49 | 10 / 49 |
| 7 / 8 | basal_ganglia 左 / 右 | 11, 12, 13, 26 / 50, 51, 52, 58 | 11 / 50 |
| 9 | brainstem | 16 | 16 |
| 10 / 11 | cerebellum 左 / 右 | 7, 8 / 46, 47 | 7 / 46 |
| 12 / 13 | other_deep_grey 左 / 右 | 17, 18, 28 / 53, 54, 60 | 17 / 53 |
| 14 | ventricles | 4, 43, 5, 44, 14, 15 | 4 |
| 15 | ignore（只在训练标签里） | 病灶区 | — |

左右按 `geometry.LEFT_LABELS / RIGHT_LABELS`。推理输出用每类的代表值，`host_class_map` 与 `LANDMARKS` 对它们的解释与原图一致（基底节四个标签合成一个代表值不改变宿主类；脑室四个标签合成 4，`BrainBinder` 只把它当地标）。CSF（24）归背景：它在 `LANDMARKS` 里只用于"不是宿主"，背景同样不是宿主。

## 5. 仿真（每个样板一次抽样，种子 = 哈希(病例, 样板号)）

1. 载入 1 mm 的 FLAIR 与老师图（BMSR 先重采样到 1 mm）；病灶掩膜重采样到同网格（最近邻）。
2. 旋转：绕左右轴 θ₁ ~ U(−10°, 10°)、绕前后轴 θ₂ ~ U(−5°, 5°)，以脑的质心为中心；图像三线性、标签与掩膜最近邻。
3. 栈的位置：算每个轴向 1 mm 层的脑截面积（老师图 > 0 且非忽略），找最高的 > 5 cm² 的层 z_top；抽空层数 e（A6 的分布）；栈顶 = z_top + 1 + 5e，底层 z0 = 栈顶 − 5n（n = 层数），不低于 0。底层截面占最大截面的比例记进样板参数（`bottom_area_share`），与 `s4_probe2` 的 0.83 对照。
4. 层数 n = 16（概率 0.9）或 14（0.1）；从底层起每 5 层合一层，共 n 层；超出体积顶部的层填零。
5. 图像：5 层平均 → 面内重采样到目标间距 s ∈ {0.6875（0.82）, 0.86（0.18）} → 放进目标矩阵（{320×320（0.6），260×320（0.2），276×276（0.2）}；FOV 以脑质心为中心，超出裁掉、不足补零）。
6. 标签：每 5 层多数票（任何平局，背景也算在内，取靠近样板中心的那层的标签；2026-10-08 按代码 `simulate.py::vote_labels` 把原来含糊的一句写清）→ 最近邻到同一网格；病灶掩膜任一层命中即该体素 = ignore（15）。
7. 强度：乘以低频偏置场（2 阶多项式六项 X、Y、Z、X²、Y²、XY，幅度 ±20%；2026-10-08 更正：原文写 3 阶，代码 `simulate.py::bias_field` 实现的是 2 阶，只是增广，不改）、伽马 ∈ [0.7, 1.4]、高斯模糊 σ ∈ [0, 0.7] 像素、Rician 噪声（σ = 1–4% 的脑内中位数）；最后按脑内 1–99 分位线性拉到 [0, 1000]（nnU-Net 再做自己的 z-score）。
8. 输出 `<case>_s<k>_0000.nii.gz` 与标签，仿射写成轴向 5 mm 栈（与 fastMRI 的 RSS NIfTI 同约定：(col, row, slice)，层间距 5 mm）。

记录：每来源各 2 个样板的蒙太奇（图像 + 标签叠加，16 层）进 `docs/verification/2026-10-02/brain_anatomy_flair/simulation/`，人眼看一遍（USER_REPORTED）。

## 6. 脑轮廓模型（Dataset908）

- 输入：fastMRI 447 卷 AXFLAIR 的 RSS NIfTI（`rss_h5_to_nifti(..., pad_to_slices=0)`，与 S2 相同），排除 14 卷失败卷 → 433 卷。
- 标签：SynthSeg 图 > 0 → 逐层二值填洞 → 取最大三维连通块；监督层 = 第 2 层到"面积 > 5 cm² 的最高层减 1"；其他层整层 = ignore（2）。标签名 {background: 0, brain: 1, ignore: 2}。
- 训练：nnU-Net `2d`，fold 0；按 `patient_id` 划分 80/20，测试病人不进数据集。
- 推理：逐层 → 三维最大连通块 → 逐层填洞 → 掩膜。评估：监督层对 SynthSeg 轮廓的 Dice（A11 ③）；最低两层与颅顶出蒙太奇。

## 7. 评估（`scripts/eval_brain_anatomy.py`）

1. **fastMRI 目标域**（447 卷，学生 + 轮廓模型的完整推理链）：按层取"可靠层"（第 2 层到顶层减 1）；13 个宿主类（6 对分侧 + 脑干，见 A11）对 SynthSeg 的 Dice（逐卷再平均，类在两边都空的卷跳过）；1297 个框的主结构一致率（`anatobind.eval.lookup.BrainLookup(seg, spacing, BRAIN_PARENCHYMA).host(rects)` 对学生 A 与 SynthSeg A 各查一次，比较宿主类；只算框完全在可靠层内的病灶，其余单列）；最低两层：学生输出各类体积分布与蒙太奇。
2. **仿真域测试**（A13）：16 类 Dice。
3. **轮廓模型**：监督层 Dice；蒙太奇。
4. 报告 `REPORT.md` + `verdict.json`（三条线各自通过与否、总判定），命令与代码版本写进报告。

## 8. 推理入口（`scripts/infer_brain_anatomy.py`）

`--h5 <fastMRI h5>` 或 `--nifti <栈>`，`--out <新目录>`，`--gpu`；流程：RSS NIfTI → 轮廓模型（2d，fold 0）→ 掩膜外置零 → 学生模型（3d_fullres，fold 0）→ 紧凑编号映射回 SynthSeg 值 → `anatomy.nii.gz`（int16，原网格）；并演示绑定：对给定的框（`--box x0 y0 z0 x1 y1 z1`，可选）用 `BrainBinder` 输出主结构与侧别。冒烟用 S2 冒烟同一卷 `file_brain_AXFLAIR_201_6002917.h5`。

## 9. 代码布局

```
anatobind/anatomy/labels.py       紧凑编号 ↔ SynthSeg 标签；老师图 → 学生标签；左右；忽略区合并
anatobind/anatomy/simulate.py     §5 的样板仿真（几何、合层、重采样、强度、标签多数票）；纯函数，可测
anatobind/anatomy/outline.py      §6 的轮廓标签（填洞、最大连通块、监督层）与推理后处理
anatobind/anatomy/sources.py      三来源的病例枚举、病人键、路径、排除名单、按病人划分
anatobind/eval/brain_anatomy.py   §7 的 Dice、框主结构一致率、可靠层、判定
anatobind/infer/brain_anatomy.py  §8 的推理链（RSS → 轮廓 → 学生 → 标签映射 → 绑定演示）
scripts/brain_anatomy_prepare.py  --stage sources | simulate | dataset907 | dataset908 | splits
scripts/brain_anatomy_train.py    在空卡上启动 907（3d_fullres）与 908（2d）的 fold 0（复用 brain_detector_train 的空卡判断）
scripts/eval_brain_anatomy.py     §7
scripts/infer_brain_anatomy.py    §8
tests/test_brain_anatomy_*.py     每个模块一份，合成小体数据，不读 /data2
```

## 10. 时间与算力预估（不是承诺）

仿真预生成：约 1300 卷 × 4 + 测试 ≈ 5500 个样板，每个几秒（CPU，8 进程）→ 1–2 h。nnU-Net 规划与预处理：3d_fullres 上 5500 例 16×320×320 ≈ 1 h。学生训练：250 轮、一张卡，按 S7 的经验 8–12 h。轮廓模型：2d、433 卷约 6500 层，2–4 h。评估：447 卷推理两套模型，GPU 约 1 h。总计约两天的机器时间，人工在旁的时间以计划的任务数为准。

## 11. 已知偏差与风险

- 老师是伪标签，肿瘤内部与被挤压的结构本就不准（S7 终审的提醒同样适用）。
- 训练 FLAIR 是 3D FLAIR，fastMRI 是 2D TSE FLAIR；A7 只做增广。过不了 A11 ① ② 时先看是不是这一条。
- SibBMS 在模板空间、已去颅骨、脑体积偏大（老师图 > 0 的中位数 2013 mL）；学生在真实大小的脑上要靠 PDGM/BMSR 与几何增广。
- fastMRI 的最低两层没有可靠参照，A11 不判那里；真正的裁判是 Level R（A12）。
- 一个来源的病人键解析错会让同一人跨训练与测试：`sources.py` 的测试要覆盖三种键。
