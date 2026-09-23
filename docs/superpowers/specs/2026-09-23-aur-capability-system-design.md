# A/U/R 能力系统（第一阶段）· 设计规格

> 状态：2026-09-23 定稿，来自当日的拷问式设计会（grill）与两次可行性 spike。它记录**已经拍板的决定**，不再讨论备选。
> 与 v2.6 的关系：v2.6（`docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md`）定义"什么算证据"；本规格定义"先把三条能力做出来"。两者冲突时，证据规则以 v2.6 为准，网络目标以 `docs/plans/2026-09-22-anatobind-mri-network-target.md` 为准（决定 Q8）。

## 0. 一句话

给一个 3D MRI 体积，系统输出三样东西：① 解剖结构的逐体素分割图；② 病灶的 3D 框加大类；③ 每个病灶所在的解剖结构（主结构、全部重叠结构及占比、侧别）。先膝后脑，先能力后网络。

## 1. 决定记录

| # | 决定 | 选项 | 一句话理由 |
|---|---|---|---|
| Q1 | 先做能力 (a)，网络 (b) 之后作为可替换关系模块接在 (a) 的输出后面，与查表对比 | (c) | 膝上目标 1、3 已有数字，缺的是病灶检测；自研 mask 头留出 Dice 0.38 对 nnU-Net 0.86 |
| Q1 | S（序列 token）、E（可观测性）、k 空间干预 | 不做 | 三条核心目标里没有它们 |
| Q2 | 解剖输出 = 逐体素分割图（NIfTI） | (a) | 真值与伪标签都是分割图；查表只能靠它 |
| Q3 | 病灶输出 = 3D 框 + 大类；有掩膜真值的数据集顺带出掩膜 | 框+大类 | 全库标注绝大多数是框；细类样本太散 |
| Q4 | 第三条目标的对错标准：膝用标注者 `tissue_id`，脑只报与查表的一致率并标 NOT_EVIDENCE | (b) | 重叠率当真值会让查表按定义满分（v2.6 红线） |
| Q5 | 脑膝都做，膝先；前列腺不做 | (c) | 膝是唯一三条目标都能算准确率的地方；PI-CAI 不在盘上 |
| Q6 | 在本仓库 main 开分支 `build/aur-system`，交接时合回 main 打 tag | (b) | 数据引擎与已训模型都在这里 |
| Q7 | 四张 A800，单次 ≤ 24 h 不问，CPU ≤ 48 线程 nice 19，两周出膝的五折数字 | — | 卡号见 §6 |
| Q8 | FM_MRI 文档定网络，v2.6 定证据 | (c) | 见文首 |
| Q9 | 膝病灶检测器：框填成掩膜训 nnU-Net 3D（Dataset902）；不达标再上 fastMRI+ 预训练的 2.5D 臂 | (a) | 复用已跑通的 Dataset901 流程，一天出结果 |
| Q10 | 脑病灶：先做 PDGM 肿瘤、BMSR 转移、ISLES 梗死三个大类（3D 掩膜真值），fastMRI+ 小病灶第二步 | (c) 先 (b) 后 | 掩膜是 1 mm 各向同性 3D 真值；fastMRI+ 是 5 mm 层厚 2D 框且要先修翻转 |
| Q11 | 脑解剖：SynthSeg robust 加 `--parc` 重跑会用到的卷，可报脑叶 | (b) | 方案自己的例子就是"左额叶" |
| Q12 | 验收口径见 §4；达标线只设在目标 2 | — | 目标 1、3 只报数 |
| N1 | 膝检测器四族都检；输入只用回波 1，干净 ×6 + 退化 6 档共 12 视图当训练样本 | (a)+(i) | 与解剖模型同配方；韧带、积液没有分割但必须检出 |
| N2 | 单命令入口吃 NIfTI 图像体积；DICOM 先用转换器转 | (a) | k 空间重建留在数据引擎 |
| N3 | 脑三个 nnU-Net 各用本数据集原生序列；每个数据集按患者留 20% 训一次 | (a)+(i) | 序列不同无法共模；FLAIR 单通道会丢 T1 增强 |
| N4 | 所在结构报主结构 + 全部重叠结构及占比 + 侧别 | (b) | 脑肿瘤跨脑叶时只有集合说得清 |
| N5 | 膝两周之后：脑掩膜三大类一周，再 fastMRI+ 小病灶一周（含 Gate 0 翻转修复），四周交付 | (a) | fastMRI+ 那一周顺手做掉 v2.6 的 Gate 0 |
| N6 | 每过一道门报一次：烟雾测试、膝 fold 0、膝五折、脑留出 | (a) | 每次附可粘贴命令与原始输出 |
| N7 | FM_MRI 文档拷入 `docs/plans/2026-09-22-anatobind-mri-network-target.md` 入库 | (a) | 不入库以后的会话找不到 |
| N8 | (b) 阶段（关系模块）的设计推迟到 (a) 出数之后再拷问 | (a) | 检测器质量决定关系模块的输入 |

## 2. 范围拆分（三份实施计划）

每份计划单独产出可测的软件；顺序固定。

1. **计划 1：膝**（`docs/superpowers/plans/2026-09-23-knee-capability-system.md`）：Dataset902 病灶检测器五折、检测与绑定评估、单命令入口、验证报告。**两周**。
2. **计划 2：脑掩膜三大类**：SynthSeg `--parc` 重跑（权重取自 FreeSurfer 7.4.1 安装包）、PDGM/BMSR/ISLES 三个 nnU-Net、脑侧入口与一致率报告。**一周**。计划 1 交付后再写。
3. **计划 3：fastMRI+ 脑小病灶**：Gate 0 翻转修复（v2.6 PR-A）、2D/2.5D 小病灶检测、接入脑侧入口。**一周**。计划 2 交付后再写。

(b) 阶段（FM_MRI 文档 §9–§12 的实体解码器与关系 Transformer）不在这三份计划内（决定 N8）。

## 3. 系统接口

### 3.1 病灶大类与标签号

膝（Dataset902 与所有输出统一）：

```
label  family            boxes.csv supercategory   dataset_v2.CLASSES id（查表用）
1      cartilage_lesion  "Cartilage Lesion"        1
2      meniscal_tear     "Meniscal Tear"           0
3      ligament_tear     "Ligament Tear"           3
4      effusion          "Effusion"                2
```

两套编号并存是历史原因：`anatobind/eval/lookup.py` 与 `dataset_v2.CLASSES` 用右列，nnU-Net 标签用左列。转换只在一处做（`anatobind/nnunet/lesion_labels.py`），任何新代码不得再手写映射。

脑（计划 2）：`1 tumor（PDGM 整瘤）、2 metastasis（BMSR）、3 infarct（ISLES）`，三个模型各自二类，大类由哪个模型响决定。

### 3.2 解剖本体

膝：Dataset901 的 6 类（1 髌骨软骨、2 股骨软骨、3 胫骨软骨内侧、4 胫骨软骨外侧、5 内侧半月板、6 外侧半月板）。侧别由标签号决定：3、5 内侧，4、6 外侧，1、2 单一。

脑：SynthSeg robust 2.0 的 33 类 aseg 加 `--parc` 的 Desikan 皮层分区；脑叶由分区合并得到。脑室与 CSF 可以出现在"重叠结构"列表里，但不作主结构（v2.6 §3）。

### 3.3 目标 3 的输出定义

对每个病灶（预测框 + 预测大类）：

- `host_label` / `host_name`：主结构 = `anatobind.eval.lookup.b0_host`（类别限定候选、框内重叠率最大者，零重叠取最近结构）。积液记 `none`，韧带撕裂记 `unknown`（没有韧带分割）。
- `host_fractions`：类别限定候选中每个结构在框内的体素占比（JSON 串），决定 N4。
- `side`：`medial` / `lateral` / `single` / `-`。

### 3.4 单命令入口（计划 1 交付膝版）

```
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/infer_knee.py \
    --image <volume.nii.gz> --out <dir> [--frame h5|world] [--folds 0 1 2 3 4] [--gpu 0]
```

输入约定：已训模型只认 SKM-TEA 导出帧，即数组轴 (X, Y, Z) 分别沿 上→下、前→后、左→右，体素 0.625×0.625×0.8 mm。`--frame h5` 表示输入已在该帧（导出文件、缓存）；`--frame world` 表示信任 NIfTI 仿射，入口把轴重排到轴码 `('I', 'P', 'R')` 再喂模型。nnU-Net 自己按头信息重采样间距，入口不重采样。

输出目录：

```
anatomy.nii.gz      6 类解剖标签图，与输入同帧同仿射
lesions.nii.gz      4 类病灶标签图（nnU-Net 原始输出）
lesions.csv         lesion_id, family, score, x0,y0,z0,x1,y1,z1（体素）, host_label, host_name, side, host_fractions
overlay.png         三面图：分数最高的病灶所在层，解剖叠色，病灶框
```

## 4. 验收口径（决定 Q12）

### 4.1 膝（计划 1）

全部按患者五折（`derived/skmtea/m1r/splits.json`，155 扫描各 31，与 Dataset901 同折同名）。评估在 nnU-Net 自带的折末验证输出上做（`fold_k/validation/<scan>_<view>.nii.gz` 与 `--npz` 的概率），不另跑推理。

| 目标 | 指标 | 真值 | 设线 |
|---|---|---|---|
| 1 解剖 | 六类留出 Dice（Dataset901 已有，重报） | raw-data-track 校正分割 | 不设线 |
| 2 病灶 | 留出灵敏度与每卷假阳；匹配 = 3D IoU ≥ 0.1、一对一（`anatobind.eval.matching.match`）；灵敏度分"定位命中"与"定位命中且大类正确"两种；工作点 = 每卷假阳 ≤ 2 下灵敏度最高的分数阈值 | `boxes.csv` 全部 465 框（四族） | **干净视图、大类正确的灵敏度 ≥ 0.5 且每卷假阳 ≤ 2** |
| 3 所在结构 | 在工作点的命中检出上：组织族正确率（`matching.bucket`，对照 `tissue_id`）与侧别正确率（对照 `host_side ∈ {medial, lateral}`）；同时报"给定标注框"的数字作对照 | 标注者 `tissue_id` / `host_side` | 不设线 |

退化视图（noise q1–q3、us 4/8/16）同样报，只作附注。

止损：干净视图不过线 → 停下写裁决交用户，不自行换第三条路；第二臂（fastMRI+ 预训练 2.5D）另写计划。

### 4.2 脑（计划 2）

目标 1 用 SynthSeg，不报数；目标 2 按三个大类报留出 Dice 与病灶级检出率（按患者留 20%）；目标 3 只报与查表的一致率和弃权率，全部标 NOT_EVIDENCE。

### 4.3 NOT_EVIDENCE 规则

凡以伪标签或重叠率为参照的数字，表头与文件名带 `NOT_EVIDENCE`。膝侧绑定数字中，"给定标注框"一列是对照，不是系统能力。

## 5. 可行性条件（2026-09-23 spike 实测）与处置

| 条件 | 实测 | 处置 |
|---|---|---|
| 箱填冲突 | 465 框中跨族重叠 183 对，冲突体素占框体素 3.3%；积液框中位 99 mL、最大 638 mL（占全卷 19.5%） | 涂色顺序：积液最先，其余按体积从大到小，小框覆盖大框（`lesion_labels.paint_order`）；只输出框不输出病灶掩膜 |
| 样本量 | 130 卷 465 框，与 09-12 塌掉的检测头同一批 | 赌 nnU-Net 的 patch 加重增广；烟雾门：20 轮内伪 Dice 不动即停 |
| 算力 | nnU-Net 一折 3.8–6.3 h（轮时 63–75 s，250 轮） | 四卡各一任务，`nnUNet_n_proc_DA=8`，合计 ≤ 48 线程 |
| 脑叶分区权重 | `synthseg_parc_2.0.h5` 不在盘上，官方链接 404；FreeSurfer 7.4.1 tarball 9.48 GB 可达 | 计划 2 下载并只取该文件 |
| ISLES FLAIR | 250/250 不在 DWI 网格 | 计划 2 只用 DWI + ADC |
| 入口轴序 | 训练图仿射是单位阵，nnU-Net 不按仿射重定向 | `--frame` 开关 + 规范化步骤（§3.4） |
| 达标线 | 无文献锚点（SKM-TEA 原文无检测基线；fastMRI+ 膝 2D 最新 mAP@0.5 = 37.1%） | 只作里程碑门，不作科学主张 |

## 6. 资源与路径

- GPU：四张 A800，默认编号 `0 1 4 6`（3、5 是 A100，2 偶被占用）；用户可改，所有脚本用 `CUDA_VISIBLE_DEVICES` 钉一张。
- CPU：`nice -n 19`，总线程 ≤ 48；nnU-Net 同时最多 4 个训练。
- 磁盘：新产物一律写 `/data2/congcong/data/FM_data/derived/`（`nnunet/` 下 Dataset902，`knee_infer/` 下入口输出）；`/data0` 已用 90%，不写。
- 长任务 `setsid nohup … &`，日志进 `logs/`（未跟踪）。
- 测试：`PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider`。
- 提交：作者 Congcong Liu，英文信息，无 AI 痕迹（仓库 CLAUDE.md）。

## 7. 明确不做

S、E、k 空间干预；前列腺；细类病灶；疾病分级与诊断；脑侧医生标注（用户另定时间）；(b) 阶段网络。
