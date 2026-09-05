# AnatoBind-MRI:解剖–异常关系绑定与关系可观测性的 3D MRI 结构化感知编码器

*Evidence-Aware Relational Representation Learning with a Variable-Size 3D MRI Transformer*

**研究方案 v2.0**(2026-09-05)

v2.0 是两份前稿的合并稿:

- **骨架**来自 `AnatoBind-MRI_cui.md`(2026-09-04):架构叙事 `X → (A, S, U) → R → E`、变尺寸 3D Swin、显式关系 token、关系可观测性 E、四阶段训练、五个带证伪条件的实验;
- **血肉**来自 `RESEARCH_PLAN.md` v1.1(2026-08-31):研究空白与新颖性收窄、数据方案、赌注实验、投稿策略、风险清单;
- **修订**来自 `REVIEW_feasibility_cui_2026-09-04.md`:必改清单 A/B/C 三级全部吸收;评审 §7 十项待拍板事项按推荐项写入(§13),可改;
- **数据状态**更新到 2026-09-05:全库副本在 `/data2/congcong/data/FM_data`,脑 val 缺口已关,一切读取走 /data2。

记号变更(相对 v1.1):病理 P → 生物异常事件 U_B;伪影 Q → 采集诱发退化 U_Q;Binding Matrix B → 关系 R;BIC → RIC。文中标 **(默认)** 的数值与层数是建议默认值,两份前稿都未给出,未经实验验证。

---

## 0. 一页纸

**一句话**:训练一个接受任意尺寸 3D MRI 的编码器,对每张图输出解剖实体 A、采集条件 S、异常视觉事件 U(生物 U_B / 采集诱发 U_Q)、每一对"事件 → 解剖"的关系 R,以及这条关系在当前图像下是否仍有足够视觉证据 E;用 raw k-space 上的受控物理干预来训练它、验证它。

**回答三个问题**:

```
WHAT                  异常事件是什么                    → U
WHERE                 它绑在哪个解剖实体上              → R
CAN WE STILL SEE IT   当前图像还够不够支撑这条关系      → E
```

**核心命题**:识别 ≠ 关系绑定。编码器即使能分别识别 lesion 和 left frontal lobe,也不保证建立 `lesion → left frontal` 这条关系;运动、噪声、欠采之后,即使真实病灶未变,当前图像也可能已经不足以支撑这条关系。

**三个贡献**(§1.2):MRI 关系绑定问题;关系中心的变尺寸 3D MRI Transformer;证据感知、干预等变的关系学习。

**第一道门**(M1,6–8 周,§9.1):SKM-TEA 155 例按 scan 五折交叉验证,分割内 319 个病理实例,损坏条件下,显式关系模型 vs seg-then-lookup,ABA 提升 ≥ 10 个百分点且配对检验显著;不过则方向重议,不进 M2。

**主投** MedIA;MICCAI 2027 顺路;NeurIPS/ICLR 备选;NBE 二阶段(§10)。

**措辞边界**:不称 foundation model。SSL 语料约 9k 卷,称"结构化感知编码器"。

---

## 1. 项目定位

### 1.1 主线

不写成"我们训练了一个 MRI foundation model,可以做很多任务"。主问题是:

> 现有 MRI 编码器能识别实体,但是否真正形成可靠、可组合的异常–解剖关系?

围绕它建立关系中心的 MRI 表征学习框架:变尺寸 3D ViT + 关系 token + 受控 MRI 干预 + 关系可观测性。第一篇把脑 + 膝做扎实;跨器官外测用 SPIDER 脊柱(前列腺在库只有解剖、无病灶,§7.3)。

### 1.2 三个贡献

1. **MRI 关系绑定问题**:分别识别 WHAT 与 WHERE ≠ 知道 WHAT IS WHERE。现有 MRI 表征学习分别关注解剖识别、病灶检测、图像质量,没有研究异常视觉事件与解剖实体之间是否形成可靠、可组合的关系表示。
2. **关系中心的变尺寸 3D MRI Transformer**:`Patch → Entity/Event → Relation → Observability`。变尺寸层级 3D Swin、物理坐标位置编码、解剖实体 token、异常事件 token、显式关系 token、关系 Transformer;不同 D×H×W 的 MRI 不需要固定 resize。
3. **证据感知、干预等变的关系学习**:利用公开 raw k-space 可做物理可控干预这一 MRI 独有条件,不要求整个表征对损坏完全不变,而要求 `R(TX) ≈ π_T(R(X))`,并用 E 判断哪些关系在当前观测下仍有充分视觉证据。一句话:选择性不变 + 关系等变 + 证据感知的不确定性。

### 1.3 最终模型具备的五种底层感知能力

```
A   识别并定位解剖
S   理解序列 / 采集条件
U   检测异常视觉事件(生物候选 U_B + 采集诱发退化 U_Q)
R   知道异常属于 / 影响哪个解剖
E   知道当前 MRI 是否足以支撑这条关系
```

同一骨干后续可接自动 QC、异常候选检测、分割、定量、VLM。第一篇只证明这五种能力。

---

## 2. 背景与研究空白

已有工作覆盖七条线,均不解决关系绑定 + 可观测性问题(第 5、6 条为 2026-08-31 第二轮核查补充,第 7 条为 v2.0 因 E 新增):

1. **解剖感知编码器**:Anatomy-VLM(WACV'26)做 ROI 定位 + 多尺度视觉语言建模;MRI-CORE 在 11 万+ MRI volumes 上预训练,已能识别 body location 与 sequence type。"知道是哪个器官"已不构成贡献。
2. **医学视觉 grounding**:MIMO(CVPR'25)等解决 text finding ↔ 病灶区域对应。
3. **通用 object–attribute binding**:NeurIPS'25 已明确指出 CLIP 类模型会把属性绑到错误对象上。
4. **医学解耦表征**(MedIA 审稿人主场):SDNet/Chartsias(MedIA 2019)、pseudo-healthy synthesis(MedIA 2020)、结构化变分先验的脑病理–解剖解耦(arXiv 2211.07820)、PathoSyn 等。**anatomy–pathology 解耦本身不新。** 2025–26 邻近新作(均已核查,无碰撞):bio-vs-technical 因子事后线性旋转解耦(arXiv 2509.11436,域/扫描仪级风格因子,非实体级、无物理干预);MoViD(arXiv 2606.04414,"motion"为生理性心动定位信号而非伪影);SSRL-MAR(arXiv 2608.10170,图像域伪影自监督,目标是去伪影而非表征分解)。
5. **CXR 解剖–发现绑定监督**:Chest ImaGenome(2021,242k CXR 解剖–发现场景图)及其上的 anatomical grounding 预训练、AnatomiX(arXiv 2601.03191)、RadGenome-Anatomy(arXiv 2605.17368)。**"把发现绑到解剖"作为监督信号在 2D CXR 已存在**——但全部是文本–区域对齐/VLM grounding 路线:无伪影因子、无采集物理、无表征级关系建模、无组合泛化评估。
6. **医学组合泛化评估**:CrossMed(arXiv 2511.11034)以 Modality–Anatomy–Task 三元组考医学 MLLM 的组合泛化(VQA 任务级),报告 held-out 组合下 83% → 49% 的崩塌;另有 arXiv 2412.20070。**"医学影像组合泛化 benchmark"的名头已被占**,但其组合单元是"任务",不是"解剖 × 异常 × 退化"的实体关系;其崩塌数据反而是本方案动机的现成证据。
7. **学习式失效预测与图像质量评估**:E 的近邻是 learned failure prediction(ConfidNet 一族,"预测自己会不会错")与全局 IQA/QC 分数。差异:E 是关系级(每一对 `U_j → A_i` 一个值)、物理定标(真值来自 k-space 干预的局部保真度,§5.2),既不是全局质量分,也不只是模型自信度。Exp 5 必须把两者都作 baseline,否则 E 不构成独立贡献。

**新颖性收窄为如下组合**(两轮检索未见直接前作):

```
采集诱发退化作为第一类异常事件(U_B / U_Q 二分,而非只有病理)
+ k-space 物理干预作为训练与验证信号
+ 实体级显式关系 token(而非图像级解耦、亦非文本–区域对齐)
+ 关系可观测性 E(关系级、物理定标的证据量)
+ 实体级组合泛化基准(解剖 × 异常 × 退化,区别于 CrossMed 的任务级)
```

三处差异化必须在 intro 与 related work 里各一句话立住:对线 4——**他们做图像级解耦,我们做实体级关系,且退化因子有采集物理的干预 ground truth**;对线 5——**他们在 2D CXR 上做文本对齐式 grounding,我们在 3D MRI 上做关系表征,且唯有 MRI 有 raw k-space 可施加真干预**;对线 6——**他们考 MLLM 的任务组合,我们考编码器的实体关系组合**。

---

## 3. 问题形式化

### 3.1 变量与前向

```
X → (A, S, U) → R → E

A  = 解剖实体 {A_i}, i = 1..K            身份 + 空间支撑 + 实体嵌入
S  = 序列 / 采集条件(全局 token)          对比度、场强、方向、脂肪抑制
U  = 异常视觉事件 {U_j}, j = 1..M,  U = U_B ∪ U_Q
     U_B = 生物异常候选(病理)
     U_Q = 采集诱发、与解剖无关的退化 {motion, noise, aliasing}
R_ij = P(U_j 绑定 / 影响 A_i | S)          事件–解剖关系
E_ij = 当前图像是否足以支撑 U_j → A_i      关系可观测性
```

### 3.2 生成模型视角与不对称先验

```
X = R_S(A ⊕ U_B ; U_Q)          R_S = 序列 S 下的 MRI 成像过程

U_B:发生在采集之前  → 天然依附解剖实体  → 稀疏绑定(低熵)
U_Q:发生在采集/重建 → 场状影响多个结构  → 多实体弥散
```

**适用边界**(v1.1 §3.2,评审要求补回):上述"退化 = 场状、与解剖无关"只对 motion / noise / aliasing 成立。susceptibility(气–组织界面锚定)、Gibbs(锐利边缘锚定)、chemical shift(脂水边界)、flow ghost(血管源)都是解剖锚定的伪影。因此:

- U_Q 显式限定为三类;称"采集诱发、与解剖无关的退化",不用 artifact 全称;
- 稀疏/弥散先验实现为软先验(损失中的熵项与硬负样本),不做硬约束;
- 退化类型覆盖面按 Elster 分类分三档处理(§6.4),第三档写进 limitations。

### 3.3 干预的角色与理论边界

真实可得的单因子干预只有 do(U_Q),在 k-space 上施加。诚实的理论上限:配对 `(X, X^q)` 两视图 → content (A, U_B) / style (U_Q) 块分离,是 von Kügelgen et al. 2021 多视图可识别性的直接推论。**只写一个 Proposition**,不做 identifiability 定理性声明;A 与 U_B 的分离靠监督;R 是 (A, U_B) 关系的确定函数,无需单独理论。图像层的 do(R)(把病灶搬家)现实中不存在,不作训练目标;v1.1 的对侧同源替换降为可选压力测试(§9.6)。

---

## 4. 模型架构

### 4.1 预处理:不统一 resize

- 不做 `X → 128³` 类固定矩阵 resize:它会改变病灶/器官相对尺寸、解剖几何形态、伪影空间模式。
- 统一做:方向归一 + 强度归一(逐体数据百分位裁剪后 z-score,**默认**)+ 物理坐标保留。每个 patch 同时保留物理坐标 `(x, y, z)_mm` 与归一化局部坐标 `(x̃, ỹ, z̃) ∈ [−1, 1]³`。
- 温和重采样只用于极端 spacing:SKM-TEA 面内 0.3125 mm → 0.625 mm(与 KMAR 对齐;全分辨率 131 万 token/卷不可持续,评审 #7);训练裁块,推理滑窗。
- 只 pad 到 patch 与窗口所需的整数倍,同时生成 `M_valid`(1 真实 / 0 填充):注意力屏蔽 M=0 的 key,patch merging 用 max 传播 mask,池化用 masked mean,损失只算 M=1 的位置。模型没有固定的 D、H、W。
- size bucket:相近尺寸同批。> 30 万 token 的体数据 batch 1–2,< 12 万 token batch 4–8,全程 gradient checkpointing(**默认**)。
- fastMRI 脑 16 层 × 5 mm ≈ 80 mm(下限)部分覆盖,S4 处 z 只剩 1,**明说 2.5D**;解剖实体每卷随机缺席,A 解码器带 no-object,实体存在性也算指标(评审 #8)。

### 4.2 骨干:变尺寸层级 3D Swin

不用 flat vanilla ViT:shifted-window 层级结构对高分辨率输入效率更高,并天然给多尺度特征。

| 级 | stride(体素) | 通道 C | 块数 | 头 | 输出与分工 |
|---|---|---|---|---|---|
| stem | (2, 4, 4) | 64 | — | — | Conv3D,k = s = (2, 4, 4),1 通道输入 |
| S1 | (2, 4, 4) | 64 | 2 | 2 | F1:局部纹理、小病灶 |
| S2 | (4, 8, 8) | 128 | 2 | 4 | F2:局部结构 |
| S3 | (8, 16, 16) | 256 | 6 | 8 | F3:解剖区域 |
| S4 | (16, 32, 32) | 512 | 2 | 16 | F4:器官、全局质量 |

- 窗口 `W = (4, 8, 8)`,shifted window 交替;级间 patch merging 把 2×2×2 邻域拼接后线性降到下一级通道数。
- patch stem 取 (2, 4, 4) 而非 4³:临床 MRI 多有明显 through-plane 各向异性(5 mm / 3 mm 层厚占多数);论文做 (2,4,4) vs 4³ 消融。
- 参数量约 12M(**默认**;Swin-T 28M 按通道比 (64/96)² 折算,RoPE 无参数)。
- 不同任务不必都依赖最后一层:事件解码器读 F1–F3,解剖解码器读 F2–F4,S 只读 F4。

**位置编码:物理坐标 3D RoPE + 局部坐标。** 不用固定的 `E_pos ∈ R^(N₀×C)`(输入尺寸一变就要插值)。3D RoPE 按 x、y、z 三轴分别作用在 q、k 上,坐标用相对体数据中心的物理 mm,通道按三轴均分,频带覆盖 4–512 mm 波长(**默认**);局部坐标 `(x̃, ỹ, z̃)` 经两层 MLP 加到 token 上:`PE = PE_physical + PE_local`。"物理坐标对跨器官尤其重要"是断言不是证据,须消融(评审)。

**各数据集进骨干后的 token 数**(patch stem 2×4×4;S4 为三次 ×2 下采样后):

| 数据 | 矩阵 D×H×W | 体素 mm | S1 token | S4 token | 备注 |
|---|---|---|---|---|---|
| SKM-TEA 原始 | 160×512×512 | 0.8×0.31×0.31 | 1,310,720 | 2,560 | 不直接用 |
| SKM-TEA 面内 0.625 mm | 160×256×256 | 0.8×0.625×0.625 | 327,680 | 640 | 训练裁块,推理滑窗 |
| fastMRI 脑 | 16×320×320 | 5×0.69×0.69 | 51,200 | 100 | z 在 S4 只剩 1,实为 2.5D |
| fastMRI 脑 AXT2 | 16×384×384 | 5×0.57×0.57 | 73,728 | 144 | |
| fastMRI 膝 | 36×320×320 | 3×0.44×0.44 | 115,200 | 300 | |
| PDGM | 155×240×240 | 1×1×1 | 280,800 | 640 | 剥颅 BraTS 空间 |
| HCP | 260×311×260 | 0.7 各向同性 | 659,100 | 1,560 | SSL 用 |
| KMAR | 23×256×256 | 3.6×0.625×0.625 | 49,152 | 128 | 只测试 |

跨度 27 倍,size bucket 不是可选项;大体积 batch 只能 1–2。

### 4.3 解剖实体解码器(A)

```
Q_A = {q_i^A}_{i=1..K},   A_i = Decoder_A(q_i^A, F^{2:4})
输出:身份 logits(K+1,含 no-object) · mask M_i^A · 实体嵌入 A_i ∈ R^256
```

- **身份锚定**:query i 固定对应解剖本体中的结构 i,由分割标签监督,不做 Hungarian 匹配(v1.1 §4.2)。这是干预等变性不陷入自指的前提:T 对表征的作用可以定义为槽身份的固定置换,而不是由模型自己定义。
- 结构:6 层 Mask2Former 式 masked cross-attention,d = 256(**默认**);mask 由 query 嵌入与像素解码特征点积得到;质心与包围框由 mask 导出。
- 监督:身份 CE + mask Dice/BCE。mask 只用于 spatial grounding 与关系真值构建,不是论文最终输出。
- **K 按域定**:膝 6(髌骨软骨、股骨软骨、胫骨内/外侧软骨、内/外侧半月板;SKM-TEA 分割标签 1–6,直接支持侧别绑定);脑按 SynthSeg 粒度(半球 × 皮层/白质/脑室/深部核团/脑干/小脑,约 30)——**全库脑侧细分解剖标签为零,须先建伪标签管线**(§7.3),脑叶级只在 PDGM/BMSR/HCP 各向同性库上用 ANTsPy atlas 配准作扩展;躯干/骨盆/脊柱用 TotalSeg-MRI 51 类与 SPIDER。
- fastMRI 膝与 KMAR 无解剖标签,用 SKM-TEA 训练的分割模型迁移(qDESS → PD TSE 域差,需人工抽检)。

### 4.4 采集条件 token(S)

一个全局 token,输入 F4 的 masked attention pooling + 可用元数据嵌入(有则用),2 层 MLP 头(**默认**);预测对比度 {T1, T1post, T2, FLAIR, DWI, ADC, PD, qDESS}、场强 {1.5T, 3T}、方向、脂肪抑制;真值来自 ISMRMRD 头与 DICOM 标签。关系模块建模的是 `R(U, A | S)`,不忽略对比度。S 的主监督在 fastMRI 脑(5 对比度 + 1.5T/3T);SKM-TEA 单协议(3T qDESS,2 台扫描仪),S 在膝侧退化,条件化效果只能在脑侧检验。

### 4.5 异常事件解码器(U)

```
Q_U = {q_j^U}_{j=1..M},   U_j = Decoder_U(q_j^U, F^{1:3})
输出:类型 logits · 3D 框(中心, 尺寸) · 事件嵌入 U_j ∈ R^256
类型 = U_B 各族 + U_Q {motion, noise, aliasing, 其他退化} + no-object
```

- M = 20 query(**默认**),6 层 cross-attention,d = 256(**默认**)。
- DETR 式 Hungarian 匹配 GT 实例,CE + L1 + GIoU(**默认**)。有 mask/bbox 监督可用,故主方案为监督查询;unsupervised slot attention 在 3D 细病灶上未经证明,只作消融(v1.1 §4.4)。
- U_Q 真值来自干预引擎施加的类型与强度(§6);真实伪影卷(fastMRI+ 85 卷)只做测试。"其他退化"类 + 弃权用于真实端分类外伪影(评审 §3.8)。

### 4.6 关系 token 与关系 Transformer(R)

不用 `MLP(A_i, U_j) → R_ij` 直接出分,而是给每一对 (A_i, U_j) 建显式 token,再让所有对之间互相看:

```
r_ij⁰ = φ[ A_i ; U_j ; S ; g(G_ij) ]         φ: MLP (256·3 + 64) → 256      (默认)
G_ij  = { Δx, Δy, Δz (mm), ‖Δ‖, IoA(B_j^U, M_i^A), 侧别, 层级 }
g     : MLP → 64                                                            (默认)
{ r_ij⁰ } → Transformer_R (L = 2, d = 256, 8 头, 全对自注意) → { r_ij }     (默认)

R_ij   = σ( h_R(r_ij) )                        软关系,跨多结构的病灶用
主宿主 = softmax_{i ∈ {1..K} ∪ {none}} ( h_R'(r_·j) )    硬决策;积液这类无宿主事件落到 none
```

- 侧别:同侧/对侧半球,或半月板、胫骨软骨的内/外侧;层级:解剖本体中的父子关系。几何来自框与 mask。
- 规模:膝 K×M = 6×20,脑约 30×20,全对自注意足够轻,不需要分解注意力。
- 架构因此从"ViT + 多个独立 task head"变为 `Patch → Entity/Event → Relation`。
- **硬关系负样本**:真实关系 `lesion → left frontal`,构造 `lesion → right frontal`、`lesion → left parietal` 作负样本;膝部内侧半月板 ↔ 外侧半月板。`L_rel = L_bind + λ_h·L_hard`,迫使模型学 WHAT IS WHERE,而不是两个独立识别器。

### 4.7 关系可观测性(E)

```
F_i^local = masked-mean( F2 , dilate(M_i^A) )          (默认)
E_ij      = σ( h_E( [ r_ij ; F_i^local ] ) )
```

E 不是全局图像质量;它回答"当前这张 MRI 是否足以支撑 `U_j → A_i`"。同一病灶干净图上 E ≈ 0.95,加重度运动后可降到 0.18。真值定义见 §5.2。

### 4.8 输出

```
T = { {A_i}, S, {U_j}, R, E }        结构化 MRI token 集
```

### 4.9 相对 v1.1 的取舍(合并决策)

| v1.1 组件 | v2.0 处置 | 理由 |
|---|---|---|
| P_θ 正常模型 + 解剖残差 `R_i = A_i − Â_i` | 移出主线;可作 U 解码器输入的消融(残差 vs 直接特征) | 事件解码器直接检测 U;残差线与 UAD/normative modeling 相邻,需额外辩护(v1.1 风险 6);HCP/OASIS 保留为 SSL 正常谱 |
| Sinkhorn/OT 归一的 Binding Matrix B | 由关系 token + softmax(K + none) 替代 | 显式 token 可携带几何与 S 条件;OT 归一对无宿主事件(积液)不自然 |
| L_factor 因子泄漏损失 | 不作损失;保留为诊断探针(`A_i → 数据集来源` 应接近 chance,§7.4) | 评审要求保留域来源泄漏监控 |
| 对侧同源替换(图像层 location intervention) | 不进训练;可选压力测试(§9.6) | do(R) 现实不存在;硬关系负样本已在关系层提供 what ≠ where 的压力 |
| spike 作核心退化 | 移至 held-out 退化类型(§6.4) | 核心 U_Q 守三类 |
| frozen-LLM probe | 第一篇不做 | _cui 与评审一致:不加 LLM |
| 类型级投影头的跨器官不变(干预轴二) | 归入 U_Q 分类头(类型级)+ 组合泛化 Exp 4 | 同一目标(退化类型解剖无关)由 held-out Knee+Motion 检验 |
| 不变性按损坏分级 + 有界距离项 | 由 E 门控 + KL 形式的 L_equiv 实现(§5.3) | 同一修法,换了载体 |
| 指标 ABA / BIC / Factor Leakage | ABA 保留;BIC → RIC;Factor Leakage → 探针 | |

---

## 5. 真值与损失

### 5.1 关系真值 R*

- **SKM-TEA**:每条病理标注自带标注者指定的 `tissue_id`(1 半月板 / 2 ACL / 3 PCL / 4 股骨软骨 / 5 髌骨软骨 / 6 胫骨软骨 / −1 积液无宿主),直接作主宿主真值,**不用重叠率反推**(评审 #3:重叠率定义把真值送给平凡管线,又浪费了现成的显式关系标签)。476 例分三层:分割内 319(软骨病变 + 半月板撕裂,宿主在 6 类分割内,主判据);积液 117(无宿主 → none 的正样本,弃权分析);韧带 40(宿主 ACL/PCL 不在分割类里、亦无独立解剖监督 → 扩展分析,需外部韧带分割)。
- **有 mask 的脑病灶库**(PDGM / BMSR / ISLES):软真值 `R*_ji = |M_j^U ∩ M_i^A| / |M_j^U|`,例如 `lesion → 0.8 × left frontal + 0.2 × adjacent`;解剖 mask 是 SynthSeg 伪标签,须披露。
- **fastMRI+ 逐层 2D 框**:按层算 `IoA(B_j, M_i^A)` 再聚合。
- 不需要人工标几十万条关系。

### 5.2 可观测性真值 E*

```
Ω_ij      = dilate( M_i^A ∪ B_j^U , r = 8 mm )                   (默认)
E*_ij(q)  = exp( − NRMSE( X^q , X^0 ; Ω_ij ) / τ_E )              取值 (0, 1];motion / noise / aliasing 通用
   或     = 局部保留谱能量 / 局部 SNR                                aliasing / noise 可解析
X^q, X^0 为线圈合并后的幅值图;τ_E 按干净重复扫描的 NRMSE 分布定标
```

- **主口径**:模型无关的物理保真度(上式)。
- **第二口径**:模型相对——冻结的干净模型在 X^q 上这条关系是否仍正确。
- 评估时以第二口径为失效事件,报 AURC / ECE / Brier。若 E 只训成"我的 R 会不会错",就是 learned failure prediction(ConfidNet 一族),Exp 5 只比 softmax/entropy 说明不了独立贡献,故 baseline 必含 learned failure prediction(评审 #4)。

### 5.3 总损失

```
L        = L_sem + λ_R · L_rel + λ_I · L_int

L_sem    = L_A + L_S + L_U                       身份 CE + mask Dice/BCE;S 的 CE;U 的 CE + L1 + GIoU
L_rel    = L_bind + λ_h · L_hard                  BCE(R, R*) + 主宿主 CE;硬关系负样本对比项
L_int    = L_equiv + λ_E · L_obs + λ_m · L_rank

L_equiv  : E* > τ 的关系对上  R_B(TX) ≈ R_B(X)   (KL)
           E* < τ 的关系对上  不再强拉一致,改为 H(R_B) ↑
           U_Q 头必须预测施加的干预类型与强度      (R_Q(TX) 随 T 响应,即 R(TX) ≈ π_T(R(X)))
L_obs    : E 回归 E*
L_rank   : margin ranking  E^{q0} > E^{q1} > E^{q2} > E^{q3}   沿退化强度单调
```

| 权重 | 默认值 | 说明 |
|---|---|---|
| λ_R | 1.0 | 关系项 |
| λ_h | 0.5 | 硬负样本对比项 |
| λ_I | 1.0 | 干预项,阶段 IV 才开启 |
| λ_E | 1.0 | E 回归 |
| λ_m | 0.5 | E 排序 |
| τ | 0.5 | 选择性不变的门限 |

六个值全是**默认**。整体目标:选择性不变 + 关系等变 + 证据感知的不确定性。v1.1 的"不变性按损坏分级、距离项有界"在此由 E 门控与 KL 形式实现,不再单独列。

---

## 6. 受控 k-space 干预引擎

在多线圈 raw k-space 上施加物理可控的退化。同一病例的 A 与 U_B 不变,只有 U_Q 与 E 随损坏改变——这是干预区别于普通增广的根据。

### 6.1 三类算子(多线圈实现)

| 算子 | 实现 | 强度梯度 q1 → q3 | 数据来源 |
|---|---|---|---|
| 运动 T_m | 图像域逐 shot 刚体变换(平移 + 旋转)→ 用固定线圈图重新编码 → 按 shot 把 k-space 行拼接回去。shot 结构按 ISMRMRD 头的 echo train 取;SKM-TEA 3D qDESS 按 ky–kz 分段。**单线圈相位斜坡公式 `y'(k) = exp(−i2πk·Δr)·y(Rk)` 不用**:三协议全是多线圈(脑 16–20、膝 15、SKM 8/16 线圈),物体动而线圈不动,该式只对单线圈刚体成立(评审 #2) | 位移幅度与受影响 shot 数递增 | fastMRI 脑、膝;SKM-TEA。线圈图:SKM-TEA 自带 ESPIRiT;fastMRI 用 BART 从 ACS 估 |
| 噪声 T_n | 逐线圈复高斯 `y' = y + n` | σ 按目标 SNR 三档 | 同上 |
| 欠采 T_a | `y' = M_R · y`;2D 数据用 Cartesian 随机或等距掩膜,3D 用 SKM-TEA 自带 Poisson-disc 掩膜;重建零填充 RSS 或 SENSE(**默认**) | 4x、8x、16x | 同上 |

每个 clean case 生成 `X⁰, X^{q1}, X^{q2}, X^{q3}`;四张图过同一模型。三种采集协议(fastMRI 脑 2D 多线圈多对比度、fastMRI 膝 2D 15ch TSE、SKM-TEA 3D qDESS 双回波)各验一次仿真 vs 真实伪影的 k-space 统计对比(评审 A1)。这部分是本组核心能力栈,也是方案护城河,不为赶进度牺牲实现质量。

### 6.2 干预 ≠ 增广

普通增广只要求预测对损坏不变。干预学习要求 `R(TX) ≈ π_T(R(X))`:生物关系在证据充分时保持、退化事件必须按 T 的类型与强度响应、可观测性沿梯度单调下降。Exp 3 用同样的三类损坏训练两组模型,只有一组加等变约束,检验差别。

### 6.3 存储预算

仿真产物存线圈合并后的幅值图,不存损坏后的 k-space:fastMRI 脑 4469 卷 × 9 变体 × 3.3 MB ≈ 133 GB;膝 1172 × 9 × 7.4 MB ≈ 78 GB;SKM-TEA(0.625 mm)155 × 9 × 42 MB ≈ 58 GB。合计 < 300 GB,/data2 余 22 T。

### 6.4 退化类型覆盖面(对照 Elster 伪影分类,分三档)

| 档 | 退化类型 | 能否从 raw k-space 做带真值的干预 | 盘上真实样本 | 处置 |
|---|---|---|---|---|
| 核心 | 运动鬼影、噪声、欠采混叠/卷褶 | 能 | KMAR 运动配对;fastMRI+ 33 卷运动标记 | U_Q 分类头 + 干预训练 |
| 可廉价补入 | spike/herringbone、zipper 射频干扰、Gibbs 截断、部分傅里叶模糊、B1/bias field 不均匀 | 能,均为 k 空间或线圈合并层面的确定性操作 | 无 | **held-out 退化类型**,只进 Exp 4 / Exp 5 的测试,不进训练、不进主分类头 |
| 解剖锚定,方法论不覆盖 | 磁敏感、化学位移、流动/搏动、层间串扰、介电效应、部分容积 | 不能,依赖物体的场图、脂水分布或血流 | 无;fastMRI+ 505 个 "Possible artifact" 框类型未标,可能混有 | 写进 limitations,不声称覆盖 |

后果:真实测试端会撞上分类外伪影,fastMRI+ 505 框须先人工标子类型,或给模型加"其他退化"类与弃权。E 对未见退化类型的泛化是现成加分证据(§9.2 Exp 5)。

---

## 7. 数据方案(2026-09-05 库存版)

### 7.1 读取规则

**本项目一切数据读取走 `/data2/congcong/data/FM_data`**(用户指令,2026-09-05);/data0 原件只作冷备份;共享盘 `/data0/Dataset/fastMRI` 不再依赖——脑 multicoil_val 1378 卷与 train 缺的 1 卷已入副本,fastMRI+ 有标注的 997 卷脑数据 100% 在副本有 raw。仍不在副本、方法论也用不上:脑 test/challenge 欠采、官方脑 DICOM 71G、共享盘 knee/ 目录(34 个改名重复卷 + 线圈图)。

### 7.2 资产 → 变量 → 路径(路径相对 FM_data,2026-09-05 实测存在)

| 变量 | 资产 | 规模(实测) | 路径 | 状态 |
|---|---|---|---|---|
| A 膝 · U_B 膝 · R 真值 · T | SKM-TEA raw track | 155 个 `MTR_*.h5`:kspace 512×512×160×2 echo×8 coil complex64(8 例 16 coil),ESPIRiT maps,Poisson 4x–16x 掩膜,target;全 3T qDESS,2 台扫描仪;155 scan = 155 受试者 | `SKM-TEA/files_recon_calib-24/` | 已解压;目录内 2.4G truncated 残留与 820G 原 tar 可删(待授权) |
| R 显式真值 | SKM-TEA 标注 v1.0.0 | train/val/test.json 86/33/36 scans;242/104/130 = 476 个 3D 框 `[x,y,z,w,h,d]`,16 细类 4 族,每条带 tissue_id / confidence / labeler | `SKM-TEA_ltr/annotations/v1.0.0/` | 就位 |
| A 膝 | SKM-TEA 分割 | 155 nii,512×512×160,标签 0–6,与 raw target 同网格 | `SKM-TEA_ltr/segmentation_masks/dicom-track/` | 就位 |
| S · T 规模引擎 | fastMRI raw | 脑 multicoil train 4469 / val 1378(16×20×640×320 级;AXT2 2678 / T1POST 949 / FLAIR 344 / T1PRE 250 / T1 248;1.5T/3T);膝 train 973 / val 199(36×15×640×372) | `fastMRI_lh_brain_knee/kspace/brain/{multicoil_train,multicoil_val}`,`…/kspace/knee/{multicoil_train,multicoil_val}` | 就位 |
| U_B 2D 框 · U_Q 真实 | fastMRI+ | `brain.csv`:8213 框 / 997 卷 / 30 类(只标 FLAIR/T1/T1POST);`knee.csv`:16167 / 974 / 22 类;脑 85 卷带真实伪影标签(Possible artifact 505 定位框 + Motion artifact 33 study-level),膝 13 卷;**全部标注卷有 raw** | `fastMRI_lh_brain_knee/Annotations/` | 就位 |
| U_B 脑 3D mask | UCSF-PDGM v5 | 501 例 / 495 完整,9 序列(T1/T1c/T2/FLAIR/SWI/DWI/ADC/ASL/DTI),tumor seg 0/1/2/4,240×240×155 @ 1 mm 剥颅 BraTS 空间;WHO 4/3/2 = 402/43/56;**298 例与 BraTS21 重叠(262 train / 36 val),非子集**;parenchyma seg 只是二值 | `UCSF-PDGM_lh/` | 已解压;无 raw |
| U_B 脑多灶 | UCSF-BMSR v1.3 | 461 例 TRAIN(T1pre/T1post/FLAIR/T2Synth/subtraction/seg,324 例 BraTS-seg) | `UCSF-BMSR_cbb/` | zip 未解压 |
| U_B 脑卒中 | ISLES-2022 | 250 例 DWI/ADC/FLAIR + mask | `ISLES_ltr/` | zip 未解压 |
| U_Q 真实测试端 | KMAR-50K | part2 已解压:641 artifact / 642 GT 配对 + 204 测试;part1 zip 698/700;2D 多层 PD TSE fs 膝 256×256×23,0.625×0.625×3.6 mm,Siemens 1.5T/3T;抽查 120 对约 5% 头信息不一致 | `KMAR-50K_ty/` | 多数配对可直接用,少数配准 |
| A 躯干/骨盆/脊柱 · held-out 解剖 | TotalSeg-MRI v2.0.0;SPIDER;AMOS22 | 616 例 51 类(脑仅整体 1 类;有 prostate/femur/hip/vertebrae/discs;无膝软骨半月板);447 图 + 447 mask + 逐椎间盘退变分级;60 例 MRI 有标签 | `TotalSegmentor_MRI_cbb/`(同名异拼 `TotalSegmentator_MRI_cbb/` 为空占位);`SPIDER_cbb/`;`AMOS_cbb/` | 均 zip 未解压 |
| SSL 正常谱 | HCP;OASIS | 1113 例 T1w+T2w 0.7 mm(260×311×260)剥颅、无 aseg;OASIS-1 12 tar + OASIS-2 2 tar,freesurfer 空 | `HCP_lh_T1T2/`;`OASIS_cbb/` | HCP 就位;OASIS 未解压 |
| 占位空 | CMRxRecon;rawdata | — | `CMRxRecon_ty/`;`fastMR_lh_bran_knee_rawdata/` | 空 |
| 不在盘 | MR-ART、BraTS 全量、PI-CAI、K2S | — | — | 见 §7.3 替代 |

### 7.3 缺口与替代(评审 B 级)

1. **脑侧细分解剖标签 = 零**(TotalSeg-MRI 脑 1 类、PDGM parenchyma 二值、HCP 无 aseg、OASIS freesurfer 空);SynthSeg / FreeSurfer / FastSurfer 全机未装(ANTsPy 0.6.3 在 BBDM 环境,可做 atlas 配准)。动作:装 SynthSeg,建伪标签管线,解剖粒度降到 SynthSeg 能给的并如实披露。**M0 第一阻塞项**;脑侧任何 R / E 实验以此为前置。
2. BraTS 2021 → **UCSF-PDGM v5**(重叠 298 例,非子集;评估 PDGM 时不用 BraTS 训练过的权重,或剔除那 262 例)。
3. PI-CAI 前列腺 → **SPIDER 脊柱**作跨器官外测(在库,有 mask 与退变分级);PI-CAI 留二阶段。
4. MR-ART → 移出必需项;脑侧 sim-to-real 外验改为**膝侧 KMAR**;脑侧真实运动只有 fastMRI+ 33 卷 study-level 标记。
5. fastMRI+ 505 个 Possible artifact 框无子类型 → 先人工标子类型,或"其他退化"类 + 弃权。
6. 待解压:KMAR part1、BMSR、ISLES、TotalSeg、SPIDER、AMOS、OASIS(data-engine 阶段任务);待删(**不可逆,须用户点头**):SKM-TEA 2.4G truncated 残留、820G 原 tar,两处副本都有。

### 7.4 域差管理

PDGM/BMSR(剥颅配准域)、HCP(研究级各向同性域)、fastMRI(临床 2D 带颅骨域)之间有系统性域差。风险:解剖实体 query 学到"数据集身份"而非解剖。对策:保留探针 `A_i → 数据集来源`(应接近 chance)与 `A_i → anatomy`(应高),全程监控。SKM-TEA 训练的膝分割迁移到 fastMRI 膝 / KMAR(qDESS → PD TSE)需人工抽检。

### 7.5 算力与工具(2026-09-04 实测)

- GPU 8 × 80 GB(A800 ×6、A100 ×2),评审时 5 张空闲;CPU 112 核;RAM 1 TB;/data2 已用 5.9 T、余 22 T。
- 在库:BART(ESPIRiT/重建)、sigpy、MONAI 1.2(GR_pt1.10)/1.5(nvgen)、nnunetv2 与 dicom2nifti(nvgen)、ANTsPy 0.6.3(BBDM)、SimpleITK、pydicom;h5py + nibabel 环境须 `PYTHONNOUSERSITE=1`。
- 缺:SynthSeg / FreeSurfer / FastSurfer。
- SSL 预训练约 9k 卷、3D Swin-T 级,4 张 A800 上数天到两周(粗估,依裁块尺寸而定);赌注实验单卡即可,瓶颈是仿真管线与分割 baseline 的搭建,不是算力。

---

## 8. 四阶段训练

不做一步端到端硬训。四个阶段按依赖顺序排列,每一阶段只打开它需要的损失。

| 阶段 | 训练什么 | 数据 | 损失 | 产出与备注 |
|---|---|---|---|---|
| I · SSL 预训练 | 骨干网 | 约 9k 卷 MRI 图像:fastMRI 脑膝、HCP、PDGM、BMSR、ISLES、TotalSeg-MRI、SKM-TEA | `L_MIM + L_contrast` | 变尺寸 3D 骨干。规模是"结构化感知编码器"级,不是 foundation model 级,措辞要收 |
| II · 结构化感知 | 三个解码器(A、S、U);骨干低学习率微调 | SKM-TEA 分割与框;fastMRI+ 框;PDGM/BMSR/ISLES mask + SynthSeg 伪标签;fastMRI 头信息;U_Q 类型用仿真数据作普通监督 | `L_sem = L_A + L_S + L_U` | 先让模型知道 what exists、在哪里 |
| III · 关系绑定 | 关系模块(φ、Transformer_R、h_R);解码器冻结或低学习率 | SKM-TEA tissue_id 真值;脑病灶库重叠软真值;大量对侧与邻区硬负样本 | `L_rel` | 稳定的解剖–事件绑定 |
| IV · 干预与可观测性 | 关系模块、E 头、U_Q 类型-强度头;骨干与解码器冻结(**默认**) | fastMRI、SKM-TEA raw 上的仿真梯度 `X⁰…X^{q3}` | `L_int = L_equiv + λ_E L_obs + λ_m L_rank` | **KMAR 与 fastMRI+ 伪影卷只做测试,不进训练**(评审 #6:KMAR 参与训练会泄漏 Exp 4 的 held-out Knee+Motion) |

模块 × 阶段状态:

```
                    I        II        III        IV
骨干 Swin           训练     低 lr     冻结       冻结(默认)
A 解码器            —        训练      冻结/低lr  冻结
S 头                —        训练      冻结       冻结
U 解码器 (U_B)      —        训练      冻结/低lr  冻结
U_Q 类型-强度头     —        类型(仿真) 冻结      训练
关系模块 R          —        —         训练       训练
E 头                —        —         —          训练
```

---

## 9. 实验设计

### 9.1 M1 赌注实验(第一道门,先于一切全量训练)

**命题**:显式关系建模 + 物理干预必须在(a)退化条件下、(b)held-out 解剖 × 异常组合上,显著胜过"分割解剖 + 分割/检测病灶 + 查表重叠"(seg-then-lookup,nnU-Net 管线实现)。这是初始提案 13 项 baseline 里唯一漏掉、`_cui` 版再次漏掉、而审稿人必问的对照。

- 干净数据上 seg-then-lookup 几乎必然不输——分割内实例的识别–绑定 gap 按构造接近零(baseline 只要有 mask 头,重叠就能读出绑定)。比较必须设在退化条件与组合迁移下。
- **主战场 SKM-TEA**:真解剖分割 + 真病理 3D 框 + 标注者指定 tissue_id + 真 raw k-space 同批扫描四齐,do(U_Q) 物理保真,**彻底消除"解剖伪标签本身就是分割管线产物"这一 confound**(v1.0 的 fastMRI + SynthSeg 战场无法回避的审稿质疑)。
- **两臂同监督、同参数量级**:臂 A = nnU-Net 分割 + 3D 检测 + 查表;臂 B = 同骨干的实体/事件 query + 关系 token。
- **判据口径**:全 155 例按 scan 五折 CV(155 scan = 155 受试者,无泄漏),分割内 319 例实例,三级退化条件,ABA 提升 ≥ 10 个百分点且 instance 级 paired bootstrap / McNemar 显著;官方 split 86/33/36 只作报告口径。积液 117 例作弃权分析,韧带 40 例作扩展分析,三者分开报。
- **第二战场**:UCSF-BMSR 多灶转移瘤(多实体关系)+ fastMRI 脑 raw 仿真退化梯度(规模验证,解剖用 SynthSeg 伪标签并如实披露)。
- **判据不成立 → 方向重议,不进 M2。**

建议排期(6–8 周):数据引擎 1–2 周(SKM-TEA 重采样 0.625 mm;seg / bbox / tissue_id 对齐;多线圈运动/噪声/欠采仿真,用自带 ESPIRiT 图与 Poisson 掩膜;五折 split);两臂 2–3 周;评估 1 周(干净 / 三级退化 × ABA 三层、侧别准确率、RIC;paired bootstrap + McNemar)。

### 9.2 五个主实验与证伪条件

| 实验 | 设计 | 证伪条件 |
|---|---|---|
| **Exp 1 识别–绑定 gap** | 普通强 3D Swin backbone,Acc(A)、Acc(U) 很高时 ABA(U→A) 是否明显下降。SKM-TEA 用 tissue_id 直接算 ABA;只在(a)退化条件下、(b)对无 mask 头的纯分类/检测 baseline 测;分割外 157 例不拿来制造 gap | gap 不存在 → 整个课题的动机重审 |
| **Exp 2 关系中心 vs multi-task** | 同骨干、同数据、同标签、同参数量级:普通 multi-task ViT、attention alignment、entity-centric、关系 Transformer,**加 seg-then-lookup** | 显式关系不赢 → 只是加参数 |
| **Exp 3 干预 vs 普通增广** | 两组都见过 motion/noise/undersampling,只有 AnatoBind 用 `R(TX) ≈ π_T(R(X))`;真实端 fastMRI+ 85 卷伪影脑图(有 raw)与 KMAR;前提是多线圈仿真实现正确 | 等变约束不赢 → 干预只是换名的增广 |
| **Exp 4 组合泛化** | 训 Brain+Motion、Brain+Noise、Knee+Noise,held-out **Knee+Motion**(SKM 仿真 + KMAR 真实);第二维度 held-out 退化类型(spike/zipper/Gibbs/部分傅里叶/bias field);病理 × 组织组合(如 held-out 软骨病变 × 髌骨软骨,全库 56 例)只作次级分析 | 未见组合上崩塌 |
| **Exp 5 可观测性失效预测** | E 预测某条关系是否即将失效;报 AURC、ECE、Brier、AUROC;对照 maximum softmax、predictive entropy、关系置信、全局质量分、**learned failure prediction(ConfidNet 类)**;E 只在三类干预上训练,在五种 held-out 退化上测 AURC/ECE 是否保持 | E 不显著优于置信度/熵 → 不构成独立贡献 |

### 9.3 组合矩阵与 split

```
脑   × { 胶质瘤(PDGM), 转移瘤(BMSR), 卒中(ISLES), WM 病灶(fastMRI+),
         motion / noise / aliasing(fastMRI 脑 raw 物理仿真),
         真实 motion(fastMRI+ 33 卷 study-level;MR-ART 不在盘) }
膝   × { SKM-TEA 病理 × 组织(tissue_id 真值), fastMRI+ 病灶框,
         motion / noise / aliasing(SKM-TEA qDESS raw + fastMRI 膝 raw 物理仿真),
         真实配对 motion(KMAR,只测) }
脊柱 × { 退变分级(SPIDER) }          ← held-out 解剖格
腹部 × { AMOS 解剖(60 例,小) }
心脏 × { CMRxRecon raw }             ← 占位空,可选第四器官
```

Split 原则:hold out 的是**组合**而非样本(训 brain+motion、knee+aliasing → 测 knee+motion),各因子在训练中均单独出现过。KMAR 与 fastMRI+ 伪影卷永不进训练。

### 9.4 sim-to-real 退化迁移与跨采集协议泛化

- **主通道:膝**——SKM-TEA raw 上 3D 物理仿真运动(qDESS 按 ky–kz 分段,段间运动建模比 2D 更真实)训练 → KMAR-50K **真实**配对膝运动上测试。两端在库,即刻可闭环;配对前做头信息审计,约 5% 不一致的配准或剔除。
- 脑通道(fastMRI raw 仿真训 → MR-ART 真实测)待 MR-ART 补下后加入,现为可选。
- **跨采集协议泛化**:同一 do(motion) 机制横跨三种采集物理(fastMRI 脑 2D 多线圈多对比度、fastMRI 膝 2D 15ch TSE、SKM-TEA 3D qDESS 双回波)。干预等变性若在三种协议下同时成立,即证明学到的是干预机制而非单一协议的过拟合;成本只是仿真管线的协议适配。

### 9.5 指标

主表不以 Dice 为主。

- **ABA**(anatomy–abnormality binding accuracy):事件归属解剖的准确率,SKM-TEA 用 tissue_id;
- **Relation mAP / F1**;
- **Laterality binding accuracy**(侧别);
- **RIC**(relational intervention consistency):`Δ_target − Δ_nuisance`,干预后目标量该变的变、不该变的不变;
- **可观测性校准**:AURC / ECE / Brier / AUROC;
- **泄漏探针**:`A_i → 数据集来源`、`U_Q 类型头 → organ` 应接近 chance;`A_i → anatomy` 应高;
- 辅助:分割 Dice、检测 AP;所有主结果带显著性检验与置信区间(MedIA 审稿惯例);同报临床错误率口径(伪影诱发假阳性率、定位/侧别错误率),为 NBE 二阶段预留钩子。

### 9.6 baseline 与消融(医学系为主)

1. **seg-then-lookup(nnU-Net 分割 + 检测 + 查表)——最重要对照**;
2. vanilla 3D ViT / DINO 特征;multi-task ViT;attention alignment;entity-centric 无关系 token;
3. MRI foundation encoder(如 MRI-CORE 类权重);
4. UAD / normative 系(pixel 级残差异常检测);
5. 伪影增广鲁棒训练(同数据、同三类损坏、无等变约束);
6. SDNet 系图像级解耦;
7. Slot Attention 系代表 1–2 个(防"Slot Attention + 医学标签"质疑);
8. 不确定性 baseline:softmax、entropy、全局 IQA 分、learned failure prediction;
9. 消融:w/o intervention、w/o relation token(MLP 出分)、w/o U_Q 因子、w/o S 条件、w/o 物理坐标 RoPE、stem (2,4,4) vs 4³、unsupervised slots、U 解码器输入残差 vs 直接特征;
10. 可选压力测试:对侧同源替换(U_B 不变、主宿主翻转),不作训练目标。

### 9.7 统计功效(SKM-TEA 赌注实验)

配对 McNemar,α = 0.05 双侧,功效 0.8:

```
n ≈ (1.96 + 0.84)² × p_disc / d² = 7.84 × p_disc / d²
d = 两法 ABA 之差,p_disc = 不一致率
```

| 想检出的 ABA 提升 d | 假设 p_disc | 需要实例数 |
|---|---|---|
| 0.05 | 0.15 | 470 |
| 0.08 | 0.18 | 220 |
| 0.10 | 0.20 | 157 |
| 0.15 | 0.25 | 87 |

可用实例:官方 test 分割内 84、全部 130;五折 CV 分割内 319、全部 476。折扣:实例按扫描聚簇(平均每例约 3 个),设计效应 1.3–1.5,有效样本约为名义值 70%;CV 汇总各折预测再做配对检验略偏乐观。折算后官方 test 只能检出 ≥ 15 个百分点;全 155 例 CV 分割内 319 例(有效约 220–245)可检出 ≥ 9–10 个百分点。**门槛定 10 个百分点。**

---

## 10. 投稿策略

| Venue | 判定 | 依据 |
|---|---|---|
| **MedIA(主目标)** | 口径最匹配,估计录用率(含 major revision)五到七成 | 医学特异 inductive bias + k-space 物理干预是一等贡献;理论不承重;多数据集验证广度已超常规证据量。风险:SDNet 线作者是审稿人主场,差异化必须锐利;审稿周期长 |
| MICCAI 2027(顺路) | 截稿约 2027 年 2 月底;核心版(关系绑定 + do(U_Q) 鲁棒性 + 赌注实验)可赶 | 会议版 → MedIA 扩展是经典管线;不以会议为纲,10 月底看 M1 结果再定;不为赶会牺牲物理干预实现质量 |
| NeurIPS/ICLR(备选) | 需重新抽象为一般 ML 叙事;理论只有 Proposition 级 | 单轮抽签,期望值低于 MedIA |
| **NBE(二阶段)** | 按现设计不能投:NBE 2025–26 口径 = health-system-scale(如 Prima:22 万 study / 52 诊断)+ 临床终点 + REAL-FM 框架;本方案的表征指标与 10^4 公开数据错位 | 二阶段改造:临床可信性叙事(伪影诱发假阳性率、定位/侧别错误率、读者研究、多中心外部队列);现在起同报临床错误率口径 |

---

## 11. 风险清单

| # | 风险 | 修法 / 对策 | 状态 |
|---|---|---|---|
| 1 | seg-then-lookup 平凡管线胜出 | 赌注实验前置(§9.1,SKM-TEA tissue_id 真值,退化条件),不成立则重议 | 待验证 |
| 2 | 脑侧细分解剖标签为零,脑侧 R / E 实验没有 A | SynthSeg 伪标签管线;粒度降级并披露;脑叶级只在各向同性库 atlas 配准 | **M0 阻塞项** |
| 3 | 运动仿真用单线圈公式,"物理干预"名不副实 | 多线圈图像域重编码路线;三协议各验一次仿真 vs 真实 k-space 统计 | 已设计 |
| 4 | E 无真值,训成 ConfidNet 一族 | 物理保真度主真值 + 模型相对第二口径;baseline 含 learned failure prediction | 已设计 |
| 5 | KMAR 参与训练泄漏 held-out Knee+Motion | KMAR 只做测试;配对前头信息审计 | 已定 |
| 6 | 关系真值用重叠率把真值送给平凡管线 | R* 以 tissue_id 为主;分割内 / 积液 / 韧带三层分开报 | 已定 |
| 7 | 干预等变的自指 / 塌缩(T 由模型自身 slot 定义) | A query 身份锚定的固定置换;不做图像层 do(R) 训练目标 | 已设计 |
| 8 | 重度退化下强拉不变逼模型幻觉;无界距离项爆炸 | E 门控的选择性不变 + KL 形式;E < τ 时改为熵增 | 已设计 |
| 9 | 解剖 query 偷学域身份(剥颅域 vs 带颅骨域) | 域来源泄漏探针全程监控(§7.4) | 已设计 |
| 10 | 退化先验反例(susceptibility / Gibbs 等解剖锚定伪影) | U_Q 显式守三类 + 软先验;三档覆盖面;第三档写 limitations | 已定 |
| 11 | 真实端撞上分类外伪影,U_Q 结果解释不清 | fastMRI+ 505 框标子类型,或"其他退化"类 + 弃权 | 待做 |
| 12 | SKM-TEA 全分辨率 131 万 token 显存不可持续 | 面内重采样 0.625 mm;训练裁块,推理滑窗;size bucket | 已定 |
| 13 | fastMRI 脑实为 2.5D,解剖实体随机缺席 | 明说 2.5D;A 解码器 no-object;实体存在性也算指标 | 已定 |
| 14 | SKM-TEA 仅 155 例,统计力有限 | 全 155 例五折 CV + 分割内 319 例 + 10 个百分点门槛;脑侧第二战场规模佐证;可选补 K2S 300 例 | 已设计 |
| 15 | PDGM 与 BraTS21 重叠 298 例,预训练权重污染 | 评估 PDGM 不用 BraTS 训练过的权重,或剔除 262 例 | 已定 |
| 16 | unsupervised slot 在 3D 细病灶失效 | DETR 式监督查询为主方案 | 已定 |
| 17 | identifiability 声明过强 | 收窄为一个 Proposition(§3.3) | 已定 |
| 18 | "组合泛化 benchmark"名头已被 CrossMed 占用 | 定位为实体级(解剖 × 异常 × 退化)基准,引用其崩塌数据作动机 | 已定 |
| 19 | "foundation model"措辞与 9k 卷语料错位 | 称"结构化感知编码器" | 已定 |
| 20 | ~~膝 raw 缺失~~;~~SKM-TEA 解压被磁盘卡住~~;~~脑 val 不在本地~~ | 膝 raw 全量到位(08-28);SKM-TEA 已解压(09-04);全库副本 + 脑 val 入 /data2(09-05) | **已关闭** |

---

## 12. 路线图与范围裁剪

### 里程碑

- **M0(2026-09-05 状态)**:~~膝 raw~~ ✓;~~SKM-TEA 解压~~ ✓;~~全库副本到 /data2 + 脑 val~~ ✓。**待做**:装 SynthSeg + 脑侧伪标签管线(第一阻塞项);zip 解压 + 格式统一(DICOM / nii / h5 / tar);SKM-TEA 残留与原 tar 清理(须授权);MR-ART / K2S / CMRxRecon 跟进为可选。
- **M1(6–8 周)**:赌注实验(§9.1,主战场 SKM-TEA)。判据通过 → M2;不通过 → 方向重议。
- **M2**:四阶段全训练(§8)+ 三协议干预引擎(§6);中期决策是否赶 MICCAI 2027(10 月底看 M1)。
- **M3**:实体级组合基准 + 五个主实验(§9.2)+ 全部 baseline / 消融 + 论文(MedIA)。

### 第一篇明确不做

- gaze(留第二篇:human visual search vs 关系表征);
- active search / where-to-look-next;
- LLM / VLM / report generation(含 v1.1 的 frozen-LLM probe);
- 前列腺(PI-CAI)与大量额外器官;
- 全伪影类型普适性声明(核心 U_Q 限定 motion / noise / aliasing);
- 图像层 do(R) 作训练目标(对侧同源替换只作压力测试);
- "foundation model"措辞。

---

## 13. v2.0 采纳的决策(评审 §7 十项,按推荐项写入;未另行拍板者视为默认,可改)

| # | 事项 | v2.0 采纳 | 出处 |
|---|---|---|---|
| Q1 | 主干文档 | 合并成 v2.0:_cui 架构叙事为骨,v1.1 数据 / 赌注实验 / 风险表为肉 | 本稿 |
| Q2 | 前列腺外测 | 换 SPIDER 脊柱;PI-CAI 留二阶段 | §7.3 |
| Q3 | 脑侧解剖粒度 | SynthSeg 粒度 + 侧别为主;脑叶级只在 PDGM/BMSR/HCP 上用 ANTsPy atlas 配准作扩展 | §4.3 |
| Q4 | SKM-TEA 分辨率 | 面内 0.625 mm 与 KMAR 对齐;训练裁块,推理滑窗 | §4.1 |
| Q5 | E 的真值 | 物理保真度为主,模型相对为第二口径;baseline 必含 learned failure prediction | §5.2 |
| Q6 | 赌注实验判据 | 全 155 例五折 CV、分割内实例、退化条件、门槛 10 个百分点;官方 split 只作报告 | §9.1、§9.7 |
| Q7 | KMAR 角色 | 只做测试;头信息审计,约 5% 配准或剔除 | §8、§9.4 |
| Q8 | MICCAI 2027 | 不以会议为纲;M1 按 §9.1 跑,10 月底看结果再定 | §10 |
| Q9 | 清理授权 | **未执行**:删 2.4G truncated 残留与 820G 原 tar(两处副本都有);解压 KMAR part1、BMSR、ISLES、TotalSeg、SPIDER、AMOS、OASIS。删除不可逆,须用户点头 | §7.3 |
| Q10 | 退化类型覆盖面 | 三类为核心并补回适用边界论证;spike / zipper / Gibbs / 部分傅里叶 / bias field 作 held-out 退化类型进 Exp 4/5;解剖锚定类写 limitations | §3.2、§6.4 |

---

## 附:关键参考

- fastMRI / fastMRI+(raw k-space + 病理与伪影标注);MR-ART(配对真实运动,Sci Data 2022,未下);KMAR-50K(配对膝运动,Sci Data 2025)
- SKM-TEA(NeurIPS D&B 2021,raw track = qDESS k-space + ESPIRiT maps + target;标注 v1.0.0 官方 split 86/33/36,每条带 tissue_id);K2S challenge(MICCAI 2022,300 膝 raw + 分割,可选未下)
- TotalSegmentator-MRI(Radiology 2025);AMOS22;SPIDER
- UCSF-PDGM v5;UCSF-BMSR v1.3;ISLES-2022;HCP;OASIS
- SDNet(MedIA 2019);pseudo-healthy synthesis(MedIA 2020);结构化变分先验病理–解剖解耦(arXiv 2211.07820)
- 2025–26 邻近核查:bio-vs-technical 因子旋转(arXiv 2509.11436);MoViD(arXiv 2606.04414);SSRL-MAR(arXiv 2608.10170)
- CXR 绑定监督线:Chest ImaGenome(arXiv 2108.00316);AnatomiX(arXiv 2601.03191);RadGenome-Anatomy(arXiv 2605.17368)
- 组合泛化评估线:CrossMed(arXiv 2511.11034);medical MLLM compositional generalization(arXiv 2412.20070)
- 失效预测 / 校准线:ConfidNet(Corbière et al., NeurIPS 2019);AURC / ECE / Brier 标准口径
- von Kügelgen et al., NeurIPS 2021(多视图 content/style 可识别性);ICLR 2024/2025 interventional CRL 与 compositional generalization 理论
- Swin Transformer(ICCV 2021)/ Video Swin;RoPE(Su et al., 2021);Mask2Former(CVPR 2022);DETR(ECCV 2020)
- Anatomy-VLM(WACV 2026);MRI-CORE;MIMO(CVPR 2025);NeurIPS 2025 object-centric binding
- Elster,《Questions and Answers in MRI》伪影分类(退化覆盖面对照)

---

## 版本记录

- **v2.0(2026-09-05)**:合并稿。骨架换为 `_cui` 的 `X → (A, S, U) → R → E`(§3–§5、§8),新增变尺寸 3D Swin 骨干与物理坐标 RoPE(§4.1–4.2)、显式关系 token 与关系 Transformer(§4.6)、关系可观测性 E 及其物理定标真值(§4.7、§5.2)、四阶段训练(§8)、五个带证伪条件的主实验(§9.2);吸收 09-04 评审必改清单:多线圈运动仿真(§6.1)、R* 以 tissue_id 为主与三层报告(§5.1)、seg-then-lookup 与 learned failure prediction 进 baseline(§9.6)、SynthSeg 伪标签管线为 M0 阻塞项(§7.3)、KMAR 只测(§8)、BraTS → PDGM / PI-CAI → SPIDER / MR-ART 移出必需项(§7.3)、SKM-TEA 0.625 mm 与 fastMRI 脑 2.5D(§4.1)、赌注实验 155 例 CV 与 10 个百分点门槛(§9.1、§9.7)、退化类型三档(§6.4)、"foundation model"措辞收窄;v1.1 的 P_θ 残差、Sinkhorn 绑定矩阵、L_factor、对侧同源替换、frozen-LLM probe 的处置见 §4.9;数据表改为 /data2 副本实测路径(§7.2),脑 val 缺口关闭;风险表重排为 20 项;韧带实例数按标注 JSON 实测为 40(评审稿写 39;319 + 117 + 40 = 476);评审 §7 十项决策按推荐写入(§13)。
- **v1.1(2026-08-31)**:第二次数据盘点——膝 raw 全量与 SKM-TEA raw track 到位;赌注实验主战场换 SKM-TEA;sim-to-real 主通道换膝;新增跨采集协议泛化;第二轮 novelty 核查补两条前作线;风险表 +3 项。
- **v1.0(2026-08-25)**:初版成稿(方法修正版 + 首轮数据盘点 + venue 判定)。
