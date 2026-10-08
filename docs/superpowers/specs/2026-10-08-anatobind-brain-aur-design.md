# AnatoBind-MRI 脑部 A/U/R 实现：设计规格（2026-10-08）

> 出处：用户 2026-09-04 的研究方案 `AnatoBind-MRI_cui.md`（下称"方案"）与 2026-09-16 的 V7 修订稿。用户 2026-10-08 的决定：按方案的架构实现三个子目标，先做脑；三个子目标改为 A（解剖）不变、U（找出所有异常，不分病种）、R（每个异常一个分侧主宿主）；不做疾病头；输入单序列；A 用 SynthSeg 全部结构而 R 用 13 类宿主；跳过方案的 Stage I 自监督预训练。本规格只覆盖这一段；方案的第三个贡献（可观测性 E、k 空间干预）整块后移，见 §10。

## 0. 一句话

用方案的可变尺寸 3D Swin Transformer + 实体/事件解码器 + Relation Token，在脑部四个来源的 1 mm 数据上训一个模型，输入任意一个序列卷，输出 32 个解剖结构的 mask、所有异常实例的 mask、每个实例的分侧主宿主和一句"xxx 存在异常"。全部评估对伪标签（SynthSeg 解剖、mask 推出的关系），标 NOT_EVIDENCE；对照组是现在跑通的 nnU-Net + 几何查表那条链（方案的 B0）。临床对错仍由 Level R 读片判。

## 1. 决定（N1–N18）

| # | 决定 |
|---|---|
| N1 | 部位只做脑。膝是第二部位，另开规格。 |
| N2 | 三个子目标：A = 解剖实体 + mask（SynthSeg 的 32 个非背景结构，`anatobind.eval.lookup.BRAIN_ALL`）；U = 一个"异常"类，所有来源里被标注的病灶都是正样本，不分病种，输出实例 mask；R = 每个 U 实例一个主宿主，词表 13 类分侧宿主（6 对分侧组织 + 脑干，`anatobind.anatomy.labels.HOST_IDS` 的类）+ 1 类"无宿主"（§4.3）。不做疾病头 D、不做"累及/邻近"次级关系、不做 E。 |
| N3 | 输入单序列：一个样本 = 一个序列卷 `X ∈ R^{1×D×H×W}`，S token 预测序列类型（T1 / T1c / T2 / FLAIR / DWI / ADC）。病灶标签只挂在能看见它的序列上（§3.2）；看不见的序列只监督 A 和 S。不做序列融合。 |
| N4 | 不统一 resize：原生网格，pad 到窗口倍数并带 `M_valid`；训练用 1 mm 下 128×160×160 体素的裁块（保留物理坐标），推理整卷滑窗。裁块不是 resize，方案 §3 允许。 |
| N5 | 骨干 = 方案 §4 的配置：通道 [64,128,256,512]、深度 [2,2,6,2]、头 [2,4,8,16]、窗口 (4,8,8)、patch stem Conv3D (2,4,4)；位置编码 = 物理坐标 3D RoPE + 归一化局部坐标（方案 §7）。现有 `anatobind/model/backbone.py`（embed 32、无 M_valid、无 RoPE、MONAI 相对位置偏置）升到这个配置并补齐两项。 |
| N6 | A 解码器：K = 32 个固定身份 query，读 F2–F4，输出身份存在性 + mask（F1 分辨率的 mask 头，再上采到原分辨率）。 |
| N7 | U 解码器：M = 64 个匿名 query，读 F1–F3，DETR 式匈牙利匹配到病灶连通块，输出存在性 + mask。退路预先登记（§11）：若 U 的门不过且原因是实例解码器收敛（判据见 §11），U 改为语义分割头 + 连通块成实例，事件 token 由连通块内特征池化得到；这仍是"找出所有异常"。 |
| N8 | R = 每个 U 实例对 13 个候选宿主建 token `r_ij = φ[A_i, U_j, S, G_ij]`，G 含三轴物理位移、最近表面距离、重叠比、侧别标志、层级（组织族），只在该实例的候选之间做自注意力（V7 §8 的候选竞争），输出 14 类 softmax（13 宿主 + 无宿主）。方案 §12 的全局 K×M 关系 Transformer 作消融。 |
| N9 | R 真值按方案 §13：病灶连通块与各宿主 mask 的重叠比给软分布，主宿主取最大；零重叠时取 10 mm 内最近宿主（与 `BrainBinder` 同规则），再无则"无宿主"。脑室/CSF 是地标不作宿主（V7 §5）。不需要人工标关系。 |
| N10 | Hard negative（方案 §14）：对侧同组织、重叠比第二大的邻接宿主，进 `L_hard` 的对比项。 |
| N11 | 训练两阶段：Stage II 联合训 A + S + U；Stage III 加 R（骨干低学习率），`L = L_A + L_S + L_U + λ_R (L_bind + λ_h L_hard)`。跳过方案 Stage I；Stage IV 不做。 |
| N12 | 数据按病人划分，每个来源留 20% 测试；PDGM / BMSR / SibBMS 沿用 S4 的划分（`derived/brain_anatomy/cases.json`），ISLES 用同一函数新划。测试病人不进任何阶段的训练。 |
| N13 | 对照组 B0 = 现有链：S7 的 nnU-Net 检测器折外预测（U 的对照）+ 在同一张预测解剖图上几何查表（R 的对照）。A 的对照 = SynthSeg 本身（真值）与 S4 学生。 |
| N14 | 门预先写死（§7），全部对伪标签：A 13 类宿主平均 Dice ≥ 0.80；U 每个来源在 nnU-Net 的工作点误报下灵敏度不低于 nnU-Net 0.05 以上；R 受控轨道 ABA ≥ B0。不过就报告并停，不调参救线。 |
| N15 | 推理输出写回 SynthSeg 标签值，下游 `BrainBinder` 与记录代码不改；句子简化为"<侧><宿主>存在异常，体积约 xx"，没有"累及"、"邻近"、疾病印象。 |
| N16 | 算力（用户 2026-10-08 批准 4 × A800 80 GB）：每个训练阶段用 4 张卡数据并行（`torchrun` + DDP），每卡 batch 2、有效 batch 8，AMP + 梯度检查点；评估与推理用其中一张。CPU `nice -n 19`、总线程 ≤ 48。训练预算按“看过的裁块数”定，不按迭代数（§6）；预计 Stage II 10–21 h、Stage III 3–7 h，每步耗时以 P3 探针实测为准（§12），实测后重算并写进记录；超过 24 h 的阶段先告知。 |
| N17 | 不删任何文件：每个输出目录先检查后建，存在即拒绝；重跑换新名。记录在 `docs/verification/2026-10-08/anatobind_brain_aur/`。 |
| N18 | 规格里每个数字标出处；评估报告每个数字可追到文件；比较两张伪标签的任何数字都标 NOT_EVIDENCE。 |

## 2. 范围与成功标准

输入：一个脑部 3D 序列卷（NIfTI；DICOM 读入属于演示那条线，不在此）。输出：`anatomy.nii.gz`（SynthSeg 值）、`lesions.nii.gz`（实例 id）、`record.json`（每个实例：体积、框、主宿主、侧别、宿主分布、句子）。成功 = §7 的三个门都过；门不过则报告、停、交用户。

## 3. 数据

### 3.1 来源

| 来源 | 例数 | 序列（网格） | A 监督 | U 监督 | 进训练 | 出处 |
|---|---|---|---|---|---|---|
| UCSF-PDGM | 501 | T1 / T1c / T2 / FLAIR，1 mm 同网格 | SynthSeg on T1（`derived/synthseg/pdgm/seg_native`，501） | 整瘤 mask，值 1（坏死）/ 2（水肿）/ 4（强化） | 是 | `anatobind/nnunet/brain_disease.py` |
| UCSF-BMSR | 461 | T1pre / T1post / FLAIR，同网格 | SynthSeg on T1pre（461） | 转移瘤 mask，值 1 | 是 | 同上 |
| ISLES-2022 | 250 | DWI / ADC 同网格（FLAIR 另网格，不用） | SynthSeg on DWI（250；质量要先核，§12 P1） | 梗死 mask，值 1 | 是 | 同上 |
| SibBMS | 358 次检查 / 185 人 | T1 / T2 / FLAIR，1 mm 模板空间 | SynthSeg on T1（`derived/synthseg/sibbms/seg_native`） | **无**：MS 斑块标注只有 10 个受试者且在原生网格（201×261×261），与模板空间 FLAIR 不同网格 | 只监督 A 和 S，U 损失屏蔽（§3.2） | `anatobind/anatomy/sources.py`；2026-10-08 核查 |
| fastMRI 脑 FLAIR | 433 卷 | 厚层 2D 轴位，0.6875 × 0.6875 × 5 mm | SynthSeg 伪标签（不可靠层已知） | fastMRI+ 框 1297 个 | 否，只做外部一致率 | S4 记录 |
| SibBMS 标注子集 | 10 人 | FLAIR / T1 / T1c / T2 原生网格 + 斑块 mask | 无（可现跑 SynthSeg） | MS 斑块 | 否，只作 U 的小规模外部核对 | `SibBMS_ms/sibbms/Output/Annotation` |

HCP（`derived/synthseg/hcp/seg_native`，1113 张 T1）是 A 的备用来源：只有在 A 的门不过且原因是解剖样本不足时才加入，加入要作为规格增补记录。第一篇的数据范围维持 v2.6 §15（用户 "approve 1"）。

### 3.2 样本与监督标志

一个样本 = 一个序列卷。每个样本带三个标志：

- `a_supervised`：恒为真（SynthSeg 图在同一网格；ISLES 的 ADC、BMSR 的 FLAIR 等与 SynthSeg 输入序列同网格，S7 的 `check_grid` 已逐例验证过 PDGM / BMSR / ISLES）。
- `u_supervised` 与 `u_values`：病灶在该序列上看得见才为真。PDGM：FLAIR 用 {1,2,4}，T1c 用 {1,4}，T1 / T2 为假；BMSR：T1post 用 {1}，T1pre / FLAIR 为假；ISLES：DWI 用 {1}，ADC 为假；SibBMS 全部为假（斑块未标注，不能当负样本）。为假的样本 U 的损失整体屏蔽，不是当作"无病灶"。
- `seq_type`：S 的标签，六类。

训练样本数（上限）：PDGM 4 × 400 + BMSR 3 × 359 + ISLES 2 × 200 + SibBMS 3 × 276 ≈ 3900 卷；其中 U 有监督的约 400 × 2 + 359 + 200 ≈ 1360 卷。

### 3.3 划分

按病人，每来源 20% 测试（`sources.split_by_patient`，seed 0）。PDGM / BMSR / SibBMS 直接用 `derived/brain_anatomy/cases.json` 的划分；ISLES 新划并写进同一格式的 `cases_aur.json`。测试病人的任何序列都不进训练。U 的对照 nnU-Net（S7）是五折全训、每例都有折外预测，所以任何测试集上都能公平比较。

## 4. 标签与真值

### 4.1 A

32 个 SynthSeg 非背景标签各一个实体（`BRAIN_ALL` 的顺序定 query 下标）。监督：每个实体的二值 mask（BCE + Dice，只在 `M_valid` 内）+ 存在性（该结构在裁块内是否有体素）。推理：逐体素取存在实体里 mask 概率最大者，写回 SynthSeg 值；全部为 0 的体素为背景。

### 4.2 U

病灶二值 mask（`u_values` 的并集）做 26 连通块 → 实例；体积 < 10 mm³ 的块忽略（与 S7 的 `lesion_components` 同规则）。匈牙利匹配代价 = 存在性 + mask Dice + mask 焦点损失（Mask2Former 的常用组合）；未匹配 query 学"无"。裁块边界切开的病灶按裁块内的部分算一个实例。

### 4.3 R

对每个真值实例：与 13 个宿主类（SynthSeg 图映射到 `HOST_CLASS_OF`）的重叠比 → 软分布 `p_host`；主宿主 = argmax；全部为 0 时取 10 mm 内最近宿主（`BrainBinder` 的 `NEAR_MM`），再无则第 14 类"无宿主"。侧别包含在类里（左 / 右 / 脑干无侧）。真值由 SynthSeg 图算，不由模型的 A 预测算；训练时 R 的输入 A_i 是模型自己的实体 embedding，这样 R 学的是"在自己的感知上绑定"。

Hard negative：对侧同组织（存在时）、`p_host` 第二大的宿主。

## 5. 模型

```
X (1×D×H×W, 任意尺寸)
 → pad 到 (2,4,4)×窗口倍数, M_valid
 → Conv3D (2,4,4) patch stem → Z0
 → Swin stage1..3 (通道 64/128/256/512, 深度 2/2/6/2, 头 2/4/8/16, 窗口 4/8/8)
   注意力里: 物理坐标 3D RoPE(x_mm, y_mm, z_mm) + 归一化局部坐标 [-1,1]^3; key 掩蔽 M_valid
 → F1 (stride 2,4,4), F2, F3, F4
 → A 解码器: 32 个身份 query, 3 层 cross-attn 到 F2..F4, d_model 256 → 存在性 + mask embedding
   mask 头: F1 像素解码 → mask = sigmoid(⟨embedding, pixel feature⟩), 上采到原分辨率
 → S token: F4 全局池化 → 6 类
 → U 解码器: 64 个匿名 query, 3 层 cross-attn 到 F1..F3, d_model 256 → 存在性 + mask embedding (同一 mask 头)
 → R: 每个 U 实例 j: 13 个候选 token r_ij = MLP([A_i; U_j; S; G_ij]), G_ij 从 A_i 的预测 mask 与 U_j 的预测 mask 算
      2 层自注意力 (只在 j 的 13 个候选之间) → 14 类 softmax (13 宿主 + 无宿主)
```

参数量预估：骨干约 25–30 M（MONAI SwinTransformer embed 64 去掉 layers4），解码器约 10 M。消融项（不进门）：方案 §12 的全局 K×M 关系 Transformer（现有 `anatobind/model/relation.py::RelationModule`）；去掉 RoPE 用相对位置偏置。

复用：`anatobind/model/backbone.py`（改 embed_dim、加 M_valid 与 RoPE）、`decoders.py` 的 `_QueryStack`/`ADecoder`/`UBDecoder`/`FullResMaskHead`/`PixelDecoder`（改 K / M / d_model、加 mask 监督）、`relation.py` 的 `geometry_features`（扩到 §4.3 的 G）与 `IndependentCandidateHead`（作 B1 消融）、`losses.py`。新代码放 `anatobind/aur/`。

## 6. 训练

- 裁块：128 × 160 × 160 体素（ISLES 的 2 mm DWI 也按体素裁，RoPE 用各自的物理坐标），每卷每 epoch 随机 2 块，含病灶的块按 1:1 过采样（U 有监督的卷）。强度增广同 S4 的 `intensity_augment`（对比度 / 伽马 / 偏置场 / 噪声 / 模糊）；**不做镜像**（宿主分侧，教训见 S4 A17）；面内小角度旋转 ± 10°。
- 预算按裁块数：Stage II 看 240 k 个裁块（≈ 3900 卷 × 2 块 / epoch ≈ 31 epoch），Stage III 看 80 k 个。4 卡 × batch 2 = 每步 8 块 → Stage II 30 k 步、Stage III 10 k 步。
- Stage II：A + S + U，AdamW，lr 3e-4（batch 8 下按平方根缩放自单卡的 2e-4），1 k 步线性预热后余弦，AMP + 梯度检查点，4 卡 DDP（`torchrun --nproc_per_node 4`，每卡一个采样器分片；模型无 BatchNorm，LayerNorm 不需同步）。
- Stage III：加 R，骨干 lr 3e-5、解码器 1.5e-4、R 头 3e-4，10 k 步。`λ_R = 1`，`λ_h = 0.2`（Stage III 开始前写死，不扫）。
- 时间：每步耗时估计 1.2–2.5 s（128 × 160 × 160 裁块、embed 64、AMP、检查点，含 DDP 同步约 10%），即 Stage II 10–21 h、Stage III 3–7 h；P3 探针实测后重算。单卡回退（4 卡不齐时）：同样的裁块预算，步数 × 4，时间约 1.7–3.5 天 + 0.5–1.2 天。
- 每 5 k 迭代在验证集（训练来源里再留 10% 病人作验证，不是测试集）记 A Dice、U 匹配灵敏度、R 准确率；只用它选 checkpoint，不改任何门。
- 显存探针（§12 P3）决定 batch 与检查点策略；若 128 × 160 × 160 在 80 GB 上 batch 1 都不行，裁块降到 96 × 160 × 160 并记录。

## 7. 评估与门（开训前写死）

全部在 §3.3 的测试病人上，对伪标签，NOT_EVIDENCE。

| 子目标 | 指标 | 对照 | 门 |
|---|---|---|---|
| A | 1 mm 测试卷上整卷滑窗预测 vs SynthSeg：13 类宿主各自 Dice 与平均；32 结构全部报告；按来源分报 | SynthSeg（真值本身）；S4 学生（只在 fastMRI 口径下可比） | 13 类平均 Dice ≥ 0.80 |
| U | 逐病灶灵敏度 @ 每例误报，按来源分报；匹配规则 = S2/S7 的（总 IoU 最大的一对一指派，去掉 < 0.1）；工作点 = S7 各检测器的阈值对应的每例误报（0.349 / 0.575 / 1.556） | 同一测试病人上 S7 nnU-Net 的折外预测 | 每个来源：灵敏度 ≥ nnU-Net 同误报下的灵敏度 − 0.05 |
| R 受控轨道 | 给定真值病灶 mask，R 的主宿主 vs §4.3 伪真值 → ABA（分侧）、侧别准确率、组织族准确率 | B0 = 在模型自己的预测解剖图上几何查表；B0* = 在 SynthSeg 图上查表（上限） | ABA(R) ≥ ABA(B0) |
| R 端到端 | 只在 U 匹配上的病灶上算同样指标 | 同上 | 只报告 |
| 方案实验 1 | 识别–绑定 gap：A 局部 Dice ≥ 0.8 且 U 匹配上的病灶里，B0 主宿主错的比例，以及 R 把其中多少改对 / 改错（rescue / harm，V7 §11） | — | 只报告 |

另报：< 5 mm 病灶灵敏度单独一行；SibBMS 10 例标注子集上 U 的灵敏度（外部，只报告）；fastMRI 433 卷上 A 的可靠层 Dice 与 1297 框的宿主一致率（与 S4 同口径，只报告）。

门不过的处理：报告、停、用户决定；§11 的退路只在用户点头后启用。

## 8. 推理与输出

`scripts/infer_anatobind_brain.py --nifti <卷> --out <新目录> --gpu <空卡> [--seq T1|T1c|T2|FLAIR|DWI|ADC]`：整卷滑窗（重叠 1/2，高斯加权）→ `anatomy.nii.gz`（SynthSeg 值）、`lesions.nii.gz`（实例 id）、`record.json`。不给 `--seq` 时用 S 的预测并写进记录。句子："<侧><宿主>存在异常，体积约 xx mL"（按体积从大到小，最多 5 处，其余计数；复用 `anatobind/infer/brain_disease.py` 的 `volume_text` 与去重逻辑，去掉"累及 / 邻近 / 印象"）。记录字段：`host`, `host_side`, `host_probs`（14 类）, `volume_mm3`, `box`, `sequence`, `anatomy_source = "AnatoBind A head (NOT_EVIDENCE)"`。

## 9. 工程、测试、记录

- 新包 `anatobind/aur/`：`labels.py`（32 实体 / 13 宿主映射）、`samples.py`（样本表与监督标志、划分）、`targets.py`（连通块实例、R 软真值、hard negative）、`model.py`（骨干 + 三个解码器 + R 头组装）、`rope.py`、`losses.py`、`train.py`、`eval.py`、`infer.py`；脚本 `scripts/aur_{prepare,train,eval}.py`、`infer_anatobind_brain.py`。
- 每个函数先写测试：标签映射双向一致；M_valid 不泄漏（pad 区改值不改输出）；RoPE 对整体平移的不变性与相对位置的敏感性；匈牙利匹配在合成实例上的正确性；R 真值在合成 mask 上等于手算；裁块坐标回填；输出目录存在即拒绝。
- 记录目录 `docs/verification/2026-10-08/anatobind_brain_aur/`：数据表、显存探针、训练启动与速率、评估报告、门、蒙太奇、推理冒烟、README 索引。
- 子代理逐任务实现（SDD），控制器逐字节比对，终审用最强模型；主线在 main，tag `handoff/<日期>-anatobind-brain-aur`。

## 10. 与 09-04 方案的对应与偏离

| 方案 | 本规格 | 性质 |
|---|---|---|
| X → A,S,U → R → E | X → A,S,U → R | E 后移 |
| 可变尺寸 3D Swin，§4 配置，pad + M_valid，物理坐标 RoPE | 原样 | 保留 |
| 不统一 resize | 原生网格，训练裁块、推理整卷 | 形式变、意图保留 |
| A 固定身份 query + 轻量 mask 头 | 原样，K = 32 | 保留 |
| S token | 原样，6 类 | 保留 |
| U 事件解码器，分 U_B / U_Q | 只做 U_B，M = 64 | U_Q 随干预后移 |
| Relation Token r_ij = φ[A_i,U_j,S,G_ij]，全局关系 Transformer | token 原样；注意力改为每病灶内候选竞争（V7 §8），全局版作消融 | 形式变（出自 V7） |
| R 真值 = mask 重叠比 | 原样 | 保留 |
| Hard relational negatives | 原样 | 保留 |
| 宿主 = 全部解剖实体 | 两级 ontology：A 32 实体，R 13 宿主 + 地标（V7 §5） | 形式变（出自 V7） |
| 四阶段训练 | 只做 II、III | I、IV 后移 |
| 损失 L_semantic + λ_R L_relation + λ_I L_intervention | 前两项 | 第三项后移 |
| 主指标 ABA、laterality | 原样；加 U 的灵敏度 / 误报（因为 U 是子目标） | 保留 |
| 实验 1、2 | 做（§7） | 保留 |
| 实验 3、4、5；Knee、Prostate；KMAR、MR-ART、PI-CAI | 不做 | 后移 |
| 疾病印象 | 不做（用户 2026-10-08 拿掉） | 删除 |

## 11. 风险与退路

- **U 的 DETR 式实例解码器**在 3D 上收敛慢、小目标弱。判据：Stage II 到 60 k 迭代时验证集 U 匹配灵敏度 < 0.5，即视为实例解码器问题；退路 = 语义分割头 + 连通块（N7），事件 token 由连通块池化。启用前先报用户。
- **A 追不上 SynthSeg**：门是 0.80 不是 0.90；若不过且样本不足，HCP 作增补（规格增补）。
- **U 追不上 nnU-Net**：通才 Swin 第一版很可能略输专才 nnU-Net，门留了 0.05 的余量；R 才是方案要证明的。
- **SibBMS 的斑块未标注**：U 损失屏蔽；若 A 在 SibBMS 上因斑块混淆变差，分来源报。
- **ISLES 的 SynthSeg 在 DWI 上**质量未核（P1）；不行就 ISLES 只监督 U，不监督 A。
- **显存**：P3 探针定；降裁块或 embed 到 48 要记录为偏离。
- **算力被占**：卡常满，训练排队；每个训练 > 24 h 已在 N16 告知。

## 12. 开工前的核查（P0，只读或只写记录）

1. ISLES：SynthSeg(DWI) 的 13 类宿主体积分布与 PDGM / BMSR 的比，加 6 例蒙太奇人工看（USER_REPORTED）。
2. 三来源每例各序列与 SynthSeg 图同网格（复用 S7 的 `check_grid`，预期全过）。
3. 显存与速度探针：embed 64 骨干 + 三个解码器，裁块 128 × 160 × 160，AMP + 检查点，单卡 batch 1 / 2 的峰值显存与每步耗时（10 分钟）；再用 4 卡 DDP 跑 200 步量每步耗时与扩展效率（10 分钟）。用实测值重算 §6 的时间并写进记录。
4. SibBMS 10 例标注子集：FLAIR 与 mask 同网格确认（已核 1 例：标注 201×261×261，1 mm）。
5. 匹配规则与 S7 折外预测文件的位置核对，确保 U 的对照能逐例对上。

核查都写进记录目录后，再写实施计划。
