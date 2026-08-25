# AnatoBind:解剖–异常绑定的医学视觉表征学习

**研究方案 v1.0**(2026-08-25)
基于 2026-08-09 初始提案,经四轮评审修正(方法正确性审查 → venue 判定 → 数据盘点)后成稿。

---

## 1. 项目定位

把医学视觉编码器的目标从"识别异常"升级为输出**结构化场景图**:

> **什么异常 × 绑在哪个解剖实体 × 是病理还是伪影**

并用 MRI 独有的 k-space 物理可干预性作为训练信号与验证手段,以组合泛化(compositional generalization)作为评估口径。

核心命题:现有编码器即使能分别识别解剖与异常,也不保证把异常**绑定**到正确的解剖实体;我们将 anatomy–abnormality binding 形式化为结构化表征学习问题,学习解剖/病理/伪影三分解的表征,其绑定关系可通过受控干预验证。

**主投稿目标:Medical Image Analysis(MedIA)**;MICCAI 2027 为顺路选项;三大会(NeurIPS/ICLR)为备选叙事;NBE 留作二阶段临床化目标(见 §8)。

---

## 2. 背景与研究空白

已有工作覆盖四条线,均不解决 binding 问题:

1. **解剖感知编码器**:Anatomy-VLM(WACV'26)做 ROI 定位 + 多尺度视觉语言建模;MRI-CORE 在 11 万+ MRI volumes 上预训练,已能识别 body location 与 sequence type。"知道是哪个器官"已不构成贡献。
2. **医学视觉 grounding**:MIMO(CVPR'25)等解决 text finding ↔ 病灶区域对应。
3. **通用 object–attribute binding**:NeurIPS'25 已明确指出 CLIP 类模型会把属性绑到错误对象上。
4. **医学解剖–病理解耦**(初始提案遗漏、评审补上的一条线,MedIA 审稿人主场):SDNet/Chartsias(MedIA 2019)、pseudo-healthy synthesis(MedIA 2020)、结构化变分先验的脑病理–解剖解耦(arXiv 2211.07820)、PathoSyn 等。**anatomy–pathology 解耦本身不新。**

**本工作的新颖性收窄为如下组合**(检索未见直接前作):

> 伪影作为第一类因子(三分解 A/P/Q,而非 A/P 二分解)
> + k-space 物理干预作为训练与验证信号
> + 实体级显式 binding 矩阵(而非图像级解耦)
> + 组合泛化基准

与第 4 条线的差异化必须在 intro 与 related work 里一句话立住:**他们做图像级解耦,我们做实体级绑定,且伪影因子有采集物理的干预 ground truth。**

---

## 3. 问题形式化

### 3.1 生成模型视角

```
X = R_S(A ⊕ P; Q)

A   = 解剖(anatomy)
P   = 病理(pathology)
Q   = 伪影(artifact / acquisition corruption)
S   = 序列 / 采集协议
R_S = MRI 成像过程
```

### 3.2 核心不对称先验

```
病理 P:发生在采集之前  →  天然依附解剖实体(anatomy-bound)  →  稀疏绑定(低熵)
伪影 Q:发生在采集/重建  →  场状影响多个结构(acquisition-induced)→  多实体弥散绑定
```

**适用边界(评审修正)**:上述"伪影=场状"只对 motion / noise / aliasing / spike 成立。susceptibility(气–组织界面锚定)、Gibbs(锐利边缘锚定)、chemical shift(脂水边界)、flow ghost(血管源)都是解剖锚定的伪影。因此:

- 第一篇**显式限定伪影集合**为 {motion, noise, aliasing, spike},不声称对全部伪影类型普适;
- 绑定结构先验(病理低熵/伪影高熵)实现为**可学习的软先验**,不做硬约束。

---

## 4. 方法设计(修正版)

五组件管线。标注 ⚠ 处为评审发现的原提案问题及修正。

### 4.1 3D/2.5D Backbone

```
X → E_backbone → F ∈ R^(N×d)
```

### 4.2 Anatomy Slots(监督锚定)

```
Z_A = {A_1, ..., A_K},  A_i = CrossAttn(q_i^A, F)
```

解剖 queries 与固定解剖本体(named structures)一一对应,由分割标签监督。**槽身份被监督锚定**——这是 4.5 中干预等变性能摆脱自指的前提。表征基元从 patch-centric 变为 entity-centric,但不以 mask 为最终输出。

### 4.3 Anatomical Residual(异常 = 对解剖期望的违背)

```
Â_i = P_θ(A_{¬i}, c_i, S)     由邻近/对侧解剖、解剖身份、序列预测该结构的正常期望
R_i = A_i − Â_i               残差即异常证据
```

⚠ **残差塌缩风险(原提案未处理)**:若 P_θ 与编码器联合训练,编码器可把异常信息藏进邻近槽(泄漏),使残差消失。**修正:P_θ 只在 healthy-only 人群(HCP + OASIS,见 §6)上训练,冻结后用于病理数据;P_θ 输入侧加 stop-gradient。**

⚠ **新颖性定位**:此模块与 normative modeling / 重建式无监督异常检测(UAD)相邻,贡献必须定位在 **entity/slot 级残差**(非 pixel 级),否则会被归类为已有 UAD。

### 4.4 Abnormality Slots

在残差场 {R_i} 上做实体分解,得病理槽 Z_P 与伪影槽 Z_Q。

⚠ **工程风险**:unsupervised slot attention 在 3D 医学细病灶上未经证明(尺度差异大:小病灶 vs 全局伪影)。**主方案采用 DETR 式监督查询**(有 mask/bbox 监督可用),unsupervised slot 作为消融/扩展。

### 4.5 Binding Matrix(核心创新)

```
B_ji = P(u_j ∈ A_i) ∝ exp[ s(u_j, A_i) + λ · s_spatial(u_j, A_i) ]
```

Sinkhorn/OT 归一。病理绑定 B^P 低熵稀疏、伪影绑定 B^Q 允许场状多实体——均为软先验(见 §3.2)。

监督 B* 来自病灶 mask/bbox 与解剖 mask 的空间重叠。**由此产生全方案的核心经验性赌注**(见 §7.1)。

### 4.6 输出

```
T = {Z_A, Z_P, Z_Q, B_P, B_Q}   →   结构化医学 token
```

第一篇**不做完整 VLM/LLM 段**;仅保留一个 frozen-LLM probe 作为附加实验(§7.4 实验三)。

---

## 5. 训练目标与干预方案(修正版)

### 5.1 总损失

```
L = L_entity + λ_b·L_bind + λ_f·L_factor + λ_i·L_intervention (+ λ_v·L_probe)
```

- `L_entity`:解剖/病理/伪影身份(CE);
- `L_bind`:CE(B, B*);
- `L_factor`:因子泄漏惩罚(adversarial predictor / 正交 / 交叉协方差,不估 MI);
- `L_intervention = L_inv + L_equiv`:最重要一项,见下。

**训练分阶段**(五项从零联训不可行):解剖槽 → 冻结 → 残差/异常槽 → binding → 干预精调。

### 5.2 干预轴一:do(Q) k-space 物理干预

在 fastMRI raw multicoil k-space 上施加:

```
motion:  逐 shot 图像域刚体变换 + 固定线圈图重编码 + k-space 分段拼接
spike:   y(k_0) ← y(k_0) + δ
alias:   y' = M·y   (欠采)
noise:   y' = y + n
```

同一真实病理下只改伪影 → ground-truth intervention。

⚠ **实现正确性要求**:朴素相位斜坡 `e^(−i2πk·Δr)` 只模拟刚体平移;旋转与逐 shot 运动必须走图像域变换+重编码路线,否则"物理干预"名不副实。这部分是本组的核心能力栈,也是方案的护城河——不为赶进度牺牲实现质量。

⚠ **不变性目标分级(原提案 Z_A 严格不变有误)**:重度损坏真实销毁解剖信息,强拉 `Z_A(X) ≈ Z_A(X^Q)` 逼模型幻觉。修正:不变性按损坏程度分级施加(仅中度以内),且距离项用 contrastive/有界形式(原提案 `−α·D(Z_Q, Z_Q^Q)` 无界最大化会导致范数爆炸)。

### 5.3 干预轴二:伪影类型跨解剖不变

⚠ 伪影的"类型"解剖无关,但伪影的"场"与解剖纠缠(脑运动鬼影 ≠ 膝运动鬼影的具体表现)。修正:**在类型级投影头上做跨器官不变性**,不对全表征施加。

### 5.4 干预轴三:location intervention(绑定等变性)

目标:异常换宿主解剖时,`Z_P` 不变、只有 B 翻转(what ≠ where,但 what is bound to where)。

```
L_equiv = || E(T_{i→j}(X)) − T_{i→j}(E(X)) ||
```

⚠ **自指修正(原提案最弱一环)**:现实中不存在 do(B)(病灶搬不了家);若 T 由模型自身 slot 定义,损失存在平凡解。修正:

- **表征侧**:T 的作用定义为监督锚定的**槽身份固定置换**(解剖槽身份由 4.2 锚定;异常槽经 Hungarian 匹配),打破 E 与 T 的循环依赖;
- **图像侧**:只用**对侧同源替换**(contralateral swap)作为保信号物理性的 counterfactual;非同源移植(如肝病灶放进肾)信号不自洽,仅作 stress test,不作训练 GT。

### 5.5 理论定位(收窄)

真实可得的单因子干预只有 do(Q)。诚实的理论上限:配对 `(X, X^Q)` 两视图 → content(A,P) / style(Q) 块分离,是 von Kügelgen et al. 2021 多视图可识别性的直接推论。**只写一个 Proposition**,不做 identifiability 定理性声明;A/P 分离靠监督,B 是 (A,P) 关系的确定函数,均无需单独理论。

---

## 6. 数据方案(实际库存版,2026-08-25 盘点)

### 6.1 在库资产 → 角色映射

库存位置:`/data0/congcong/data/FM_Data`(~480G)+ 共享盘 `/data0/Dataset/fastMRI`。

| 资产 | 内容 | 方案角色 |
|---|---|---|
| fastMRI raw(共享盘) | **脑 multicoil 全量**(train 1.8T + val 514G + test/challenge,.h5 已验证) | do(Q) 物理干预引擎(脑)——完备 |
| fastMRI DICOM + fastMRI+ | 脑 4469 + 膝 1172 卷幅值图;标注 CSV 脑 8.2k / 膝 16.2k 框 | 快速原型 + 病理弱标注 |
| UCSF-PDGM v5 | 501 例胶质瘤,9 序列 + 肿瘤 3D mask + 脑实质分割 | 病理绑定主力(多序列证据链) |
| UCSF-BMSR v1.3 | 脑转移瘤多发小病灶 + mask | **binding 天然考场**(一脑多灶考多实体绑定) |
| ISLES-2022 | 250 卒中,DWI/ADC/FLAIR + mask | 病理绑定(卒中) |
| SKM-TEA | 155 例膝 qDESS + 组织分割 + 病理 bbox | 膝解剖 + 膝病理 |
| KMAR-50K | 配对 Artifact↔GroundTruth 膝运动 | **真实伪影 OOD 测试格** |
| HCP(1113+ 人) | T1w/T2w 结构像(brain-extracted) | healthy-only 人群 → P_θ 训练集(§4.3 修法落地) |
| OASIS-1/2 | 横断 + 纵向老化/痴呆脑 | 正常谱系扩展 |
| AMOS22 / TotalSeg-MRI v2 | 腹部多器官 / 全身解剖 | 解剖轴 |
| SPIDER | 腰椎 MRI + mask + 退变分级 | 新解剖格(脊柱),held-out 候选 |
| CMRxRecon | 占位待下载 | 心脏 raw k-space(可选第四器官) |

### 6.2 三个缺口与责任动作

1. **膝 raw k-space 不在本机**(FM_Data 内膝 DICOM 由 hongli 在另一台机器从 HDF5 转出;共享盘 knee/ 目录是无关数据)。动作:向 hongli 索取源文件搬运,或从 NYU 重下(~1.1TB)。**这是唯一影响论文骨架的缺口**——没有它,do(Q) 只剩脑一个器官,干预轴二(§5.3)断裂。
2. **MR-ART 未下载**(脑真实配对运动,OpenNeuro,几十 GB)。没有它,脑侧只有仿真伪影。便宜且高价值,立即补。
3. **脑细分结构标签缺失**(HCP 本地无 aseg,OASIS freesurfer 目录为空)。动作:自建 SynthSeg/FastSurfer 伪标签管线。纯工程。

另:五个 zip 未解压、格式不统一(DICOM/nii/h5/tar)——data-engine 阶段任务;PI-CAI 前列腺不下(第一篇裁剪)。

### 6.3 域差管理

PDGM/BMSR(剥颅+配准域)、HCP(研究级各向同性域)、fastMRI(临床 2D 轴位域)之间存在系统性域差。风险:解剖槽偷学"域身份/数据集来源"而非解剖。对策:Factor Leakage 指标(§7.5)增加 `Z_A → 数据集来源` probe,全程监控。

---

## 7. 实验设计

### 7.1 赌注实验(第一优先,先于一切全量训练)

**命题**:显式 binding 必须在(a)伪影污染下、(b)held-out 解剖×异常组合上,显著胜过 "分割解剖 + 分割/检测病灶 + 查表重叠"(seg-then-lookup,nnU-Net 管线实现)。

- 干净数据上 seg-then-lookup 几乎必然不输——比较必须设在伪影污染与组合迁移条件下;
- 主战场:UCSF-BMSR 多灶转移瘤(多实体绑定)+ fastMRI raw 仿真伪影梯度;
- **判据:若显著优势不成立,方向重议,不进入 M2。**

这是初始提案 13 项 baseline 里唯一漏掉、而审稿人必问的对照。

### 7.2 组合矩阵与 split

实际可填格(比初始提案更密):

```
脑   × { 胶质瘤(PDGM), 转移瘤(BMSR), 卒中(ISLES), WM病灶(fastMRI+),
         motion/spike/alias/noise(raw 物理仿真), 真实 motion(MR-ART,待补) }
膝   × { SKM-TEA 病理+解剖, fastMRI+ 病灶框, 真实配对 motion(KMAR),
         物理仿真(待膝 raw) }
脊柱 × { 退变分级(SPIDER) }          ← held-out 解剖格候选
腹部 × { AMOS 解剖 }
心脏 × { CMRxRecon raw }             ← 可选第四器官
```

Split 原则:hold out 的是**组合**而非样本(训 brain+motion、knee+aliasing → 测 knee+motion),各因子在训练中均单独出现过。

### 7.3 sim-to-real 伪影迁移(评审新增,证据力最强的一条)

fastMRI raw 物理仿真伪影上训练 → KMAR(膝)/ MR-ART(脑)**真实**运动上测试。仿真训、真实测的迁移证据比"脑学膝测"更有说服力。

### 7.4 三个 killer experiments

1. **Pathology or Artifact?** 真病灶 + 递增 motion(X_0 → X_m3):要求 Z_A/Z_P/B_P 稳定、Z_Q 响应、B_Q 随 motion 增强向多结构扩散;普通编码器整体表征漂移甚至把伪影当病灶。
2. **异常换位**(对侧同源替换):Z_P 不变、B 翻转([P→A_i] → [P→A_j])。
3. **Frozen-LLM probe**:同一冻结 LLM 下比较 patch tokens / anatomy tokens / bound tokens 对困难 QA(在哪个结构?病理还是伪影?左还是右?)的增益。轻量附加实验,非完整 VLM 段。

### 7.5 指标

- **ABA**(anatomy–abnormality binding accuracy):异常归属解剖的准确率;
- **Factor Leakage**:probe `Z_A→artifact`、`Z_Q→organ`、`Z_A→数据集来源` 应接近 chance,`Z_A→anatomy` 应高;
- **BIC**(binding intervention consistency):`Δ_target − Δ_nuisance`;
- 标准口径:检测 AP、分类 AUC + 显著性检验/置信区间(MedIA 审稿惯例)。

### 7.6 Baseline 清单(医学系为主)

1. **seg-then-lookup(nnU-Net 管线)——最重要对照**;
2. vanilla 3D ViT / DINO 特征;
3. MRI foundation encoder(如 MRI-CORE 类权重);
4. UAD/normative 系(pixel 级残差异常检测);
5. 伪影增广鲁棒训练(同数据、无因子分解);
6. SDNet 系图像级解耦;
7. Slot Attention 系代表 1–2 个(防"Slot Attention + 医学标签"质疑);
8. 消融:w/o intervention、w/o explicit binding、w/o artifact factor、unsupervised slots。

---

## 8. 投稿策略

| Venue | 判定 | 依据 |
|---|---|---|
| **MedIA(主目标)** | 口径最匹配,估计录用率(含 major revision)五到七成 | 医学特异 inductive bias + k-space 物理是一等贡献;理论不承重;多数据集验证广度已超常规证据量。风险:SDNet 线作者是审稿人主场,差异化必须锐利;审稿周期长 |
| MICCAI 2027(顺路) | 截稿约 2027 年 2 月底;核心版(binding + do(Q) 鲁棒性 + 赌注实验)可赶 | 会议版 → MedIA 扩展是经典管线;不为赶会牺牲物理干预实现质量 |
| NeurIPS/ICLR(备选) | 需重新抽象为一般 ML 叙事;理论只有 Proposition 级,ICLR 理论路线降级 | 单轮抽签,期望值低于 MedIA |
| **NBE(二阶段)** | 按现设计不能投:NBE 2025–26 口径 = health-system-scale(如 Prima:22 万 study/52 诊断)+ 临床终点 + REAL-FM 框架;本方案的表征指标与 10^4 公开数据错位 | 二阶段改造:临床可信性叙事(伪影诱发假阳性率、定位/侧别错误率、读者研究、多中心外部队列)。现在起在实验里**同报临床错误率口径**,预留钩子 |

---

## 9. 风险清单(评审结论汇总)

| # | 风险 | 修法/对策 | 状态 |
|---|---|---|---|
| 1 | seg-then-lookup 平凡管线胜出 | 赌注实验前置(§7.1),不成立则重议 | 待验证 |
| 2 | location intervention 自指/塌缩 | 监督锚定槽身份固定置换 + 对侧同源替换(§5.4) | 已设计 |
| 3 | Z_A 不变性在重度损坏下不成立 | 按损坏分级 + 有界距离项(§5.2) | 已设计 |
| 4 | Z_Q 跨器官全表征不变过强 | 类型级投影头(§5.3) | 已设计 |
| 5 | identifiability 声明过强 | 收窄为 Proposition(§5.5) | 已定 |
| 6 | anatomical residual 被归类为 UAD | entity 级残差定位 + P_θ healthy-only 训练(§4.3) | 已设计 |
| 7 | 解剖槽偷学域身份 | 域来源 leakage probe 全程监控(§6.3) | 已设计 |
| 8 | 伪影先验反例(susceptibility/Gibbs) | 显式限定伪影集合 + 软先验(§3.2) | 已定 |
| 9 | 膝 raw 缺失断掉干预轴二 | 向 hongli 索取或重下(§6.2) | **待办** |
| 10 | unsupervised slot 在 3D 细病灶失效 | DETR 式监督查询为主方案(§4.4) | 已定 |

---

## 10. 路线图与范围裁剪

### 里程碑

- **M0(即刻)**:补数三动作——膝 raw 搬运/重下、MR-ART 下载、CMRxRecon 跟进;zip 解压 + 格式统一 + SynthSeg 伪标签管线(data-engine)。
- **M1(≤2 个月)**:赌注实验(§7.1)。判据通过 → M2;不通过 → 方向重议。
- **M2**:全管线分阶段训练(§5.1)+ 三轴干预;中期决策是否赶 MICCAI 2027(截稿约 2 月底)。
- **M3**:组合基准 + killer experiments + 全部 baseline/消融 + 论文(MedIA)。

### 第一篇明确不做

- gaze(留第二篇:human visual search vs binding 表征);
- active search / where-to-look-next;
- 完整 VLM/LLM 段(只留 frozen-LLM probe);
- 前列腺(PI-CAI);
- 全伪影类型普适性声明(限定 motion/noise/aliasing/spike)。

---

## 附:关键参考

- fastMRI / fastMRI+(raw k-space + 病理标注);MR-ART(配对真实运动,Sci Data 2022);paired knee motion(Sci Data 2025)
- TotalSegmentator-MRI(Radiology 2025);AMOS22;SPIDER
- UCSF-PDGM v5;UCSF-BMSR v1.3;ISLES-2022;SKM-TEA;HCP;OASIS
- SDNet(MedIA 2019);pseudo-healthy synthesis(MedIA 2020);结构化变分先验病理–解剖解耦(arXiv 2211.07820)
- von Kügelgen et al., NeurIPS 2021(多视图 content/style 可识别性);ICLR 2024/2025 interventional CRL 与 compositional generalization 理论
- Anatomy-VLM(WACV 2026);MRI-CORE;MIMO(CVPR 2025);NeurIPS 2025 object-centric binding
