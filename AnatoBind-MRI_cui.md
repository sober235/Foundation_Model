# AnatoBind-MRI

## Evidence-Aware Relational Representation Learning with a Variable-Size 3D MRI Transformer

核心不是做一个“大而全”的 MRI foundation model，而是训练一个具有结构化 MRI perception 能力的统一视觉编码器：

\[
\boxed{
X\rightarrow A,S,U\rightarrow R\rightarrow E
}
\]

其中：

\[
A=\text{Anatomy},\quad
S=\text{Sequence/Acquisition}
\]

\[
U=\text{Abnormal Visual Event}
\]

\[
R=\text{Event--Anatomy Relation}
\]

\[
E=\text{Relation Observability}
\]

论文真正的创新集中在：

\[
\boxed{
R+E+\text{Controlled Intervention}
}
\]

而不是简单做 anatomy / lesion / artifact 多任务分类。

---

## 一、论文解决什么问题

普通 MRI ViT 即使能够分别识别：

\[
\text{lesion}
\]

和：

\[
\text{left frontal lobe}
\]

也不保证其真正建立：

\[
\boxed{
\text{lesion}\rightarrow\text{left frontal lobe}
}
\]

因此，第一个核心问题是：

\[
\boxed{
\text{Recognition}\neq\text{Relational Binding}
}
\]

进一步，在 motion、noise、undersampling 后，即使真实病灶没有改变，当前图像也可能已经不足以支撑：

\[
\text{lesion}\rightarrow A_i
\]

所以第二个核心问题是：

\[
\boxed{
\text{Can the current MRI still support this relation?}
}
\]

最终论文回答：

\[
\boxed{
\textbf{WHAT}
\rightarrow
\textbf{WHERE}
\rightarrow
\textbf{CAN WE STILL SEE IT?}
}
\]

---

## 二、数据集：Brain + Knee + Prostate

建议 **Brain 和 Knee 为主训练域，Prostate 只做跨器官泛化测试**。

| 数据集 | 作用 | 对应变量 |
|---|---|---|
| TotalSegmentator MRI | 多器官 anatomy supervision | \(A\) |
| fastMRI Brain/Knee | 大规模 raw k-space、自监督、物理 intervention | \(S,I\) |
| fastMRI+ | 脑/膝 abnormality bounding boxes | \(U,R\) |
| BraTS 2021 | 脑肿瘤 multi-sequence + segmentation | \(U,R\) |
| ISLES 2022 | DWI/ADC/FLAIR stroke + segmentation | \(S,U,R\) |
| SKM-TEA | knee raw + DICOM + anatomy + pathology | \(A,S,U,R,I\) |
| KMAR-50K | paired real motion / rescan | \(U_Q,E\) |
| MR-ART | real brain motion external validation | \(U_Q,E\) |
| PI-CAI | prostate cross-organ external validation | \(S,U,R\) |

TotalSegmentator MRI 有 616 个经过人工审核的临床 MRI，并提供 80 个 anatomical structures，数据跨不同序列、field strength、30 台 scanner、3 个 manufacturer，因此很适合作为 broad anatomy supervision。

fastMRI 是最核心的数据源：目前公开 raw data 包括约 1,398 个 knee scans、约 7,002 个 brain scans，以及 312 个 prostate examinations；brain 覆盖 T1、T1 post-contrast、T2、FLAIR。

fastMRI+ 又为 knee 和约 1,000 个 brain datasets 增加了大量 radiologist pathology bounding-box annotations，非常适合我们的“abnormality candidate”而非最终临床诊断定位。

SKM-TEA 虽然只有 155 例，但单个数据集同时具有 raw k-space、DICOM、6 类组织 segmentation 和 16 类 pathology bounding boxes，是训练 \(A,U,R\) 和 controlled intervention 的关键闭环数据。

Brain lesion supervision 使用 BraTS + ISLES：

- BraTS 2021 提供多机构 multiparametric brain MRI 与 tumor compartment segmentation；
- ISLES 2022 的公开训练集有 250 例 DWI、ADC、FLAIR 和 stroke lesion masks。

真实 QC 使用 KMAR-50K：1,190 名患者、1,444 组 paired knee MRI、62,506 slices，每组包括 motion-corrupted scan 与 rescan reference，并覆盖 1.5T/3T、多方向、多序列。

MR-ART 则提供 148 名受试者 matched motion-free / 两级 motion T1 MRI，留作 synthetic-to-real motion 外部验证。

PI-CAI 的公开部分有 1,500 例 prostate MRI，其中 425 例 csPCa-positive，并具有 lesion annotation，可用于最终 cross-organ transfer。

需要注意：PI-CAI 公开数据采用 CC BY-NC 4.0，因此第一篇科研使用没有明显问题，但未来商业化需要重新进行 license audit。

---

## 三、数据预处理原则：不同尺寸 MRI 不统一 Resize

不采用：

\[
X\rightarrow128^3
\]

这类固定矩阵 resize。

原因是固定 resize 会改变：

- lesion / organ 相对尺寸；
- anatomy 几何形态；
- artifact spatial pattern。

统一采用：

\[
\boxed{
\text{orientation normalization}
+
\text{intensity normalization}
+
\text{physical coordinate preservation}
}
\]

MRI 矩阵：

\[
D\times H\times W
\]

允许每例不同。

对于极端 spacing 可以做温和 resampling，但**不要求所有病例得到相同 matrix size**。

每个 patch 同时保留：

\[
(x,y,z)_{\rm physical}
\]

以及：

\[
(\tilde x,\tilde y,\tilde z)\in[-1,1]^3
\]

---

## 四、Backbone：Variable-Size 3D Hierarchical Swin

不采用 flat vanilla ViT。

原因是 Swin 的 shifted-window hierarchical architecture 对高分辨率输入计算效率更高，并天然提供 multi-scale features。

### 推荐配置

Channels：

\[
\boxed{
C=[64,128,256,512]
}
\]

Depth：

\[
\boxed{
[2,2,6,2]
}
\]

Heads：

\[
\boxed{
[2,4,8,16]
}
\]

Window：

\[
\boxed{
W=(4,8,8)
}
\]

作为第一版配置。

---

## 五、3D Patch Stem

输入：

\[
X\in\mathbb R^{1\times D\times H\times W}
\]

采用 `Conv3D` 进行 patch embedding。

第一版推荐：

\[
\boxed{
P=(2,4,4)
}
\]

即：

\[
Z^0=
Conv3D_{k=s=(2,4,4)}(X)
\]

原因是很多临床 MRI 存在明显 through-plane anisotropy，\(2\times4\times4\) 比粗暴使用 \(4^3\) 更保险。

论文中进一步进行：

\[
2\times4\times4
\quad vs\quad
4^3
\]

ablation。

---

## 六、Variable-Size 如何真正实现

假设输入：

\[
X_1=155\times187\times143
\]

和：

\[
X_2=201\times224\times96
\]

模型均直接接受。

只 pad 到层级下采样和 window 所要求的最近整数倍：

\[
X\rightarrow\tilde X
\]

同时建立：

\[
M_{\rm valid}
\]

其中：

\[
M=1
\]

表示真实 MRI；

\[
M=0
\]

表示 padding。

Attention、pooling 和 loss 均忽略：

\[
M=0
\]

因此：

\[
\boxed{
\text{模型没有固定 }D,H,W
}
\]

训练时采用 **size bucket**：

> 相近尺寸病例进入同一 batch，以减少 padding waste。

---

## 七、位置编码：Physical-Coordinate 3D RoPE

不使用固定：

\[
E_{\rm pos}\in\mathbb R^{N_0\times C}
\]

否则输入尺寸变化后需要进行位置编码插值。

采用：

\[
\boxed{
3D\ RoPE
}
\]

分别编码：

\[
x,\quad y,\quad z
\]

对于第 \(n\) 个 patch：

\[
c_n=
(x_n^{mm},y_n^{mm},z_n^{mm})
\]

位置编码：

\[
PE_n
=
RoPE_x(x_n)
\oplus
RoPE_y(y_n)
\oplus
RoPE_z(z_n)
\]

同时加入 normalized local coordinate：

\[
\tilde c_n\in[-1,1]^3
\]

最终：

\[
\boxed{
PE=
PE_{\rm physical}
+
PE_{\rm local}
}
\]

该设计对于：

\[
Brain\rightarrow Knee\rightarrow Prostate
\]

的跨器官建模尤其重要。

---

## 八、Backbone 输出四层特征

得到：

\[
F^1,F^2,F^3,F^4
\]

分别对应：

\[
F^1:\text{局部 texture / small lesion}
\]

\[
F^2:\text{局部 structure}
\]

\[
F^3:\text{anatomical region}
\]

\[
F^4:\text{organ/global quality}
\]

因此，不同任务不需要全部依赖最后一层特征。

---

## 九、Anatomy Entity Decoder

建立：

\[
Q_A=\{q_i^A\}_{i=1}^{K}
\]

使用：

\[
[F^2,F^3,F^4]
\]

进行 multi-scale cross-attention：

\[
A_i
=
Decoder_A
(q_i^A,F^{2:4})
\]

每个：

\[
A_i
\]

输出：

\[
\boxed{
\text{anatomy identity}
+
\text{spatial support}
+
\text{entity embedding}
}
\]

例如：

- brainstem；
- left frontal；
- ventricle；
- medial meniscus；
- 等。

同时设置轻量 mask head：

\[
A_i\rightarrow M_i^A
\]

Mask 主要用于：

1. spatial grounding；
2. relation GT 构建。

并非论文最终目的。

---

## 十、Sequence / Acquisition Token

引入一个 global acquisition token：

\[
S
\]

输入：

\[
F^4+\text{available metadata}
\]

预测：

- T1；
- T2；
- FLAIR；
- DWI；
- ADC；
- PD；
- field strength；
- orientation；
- 其他 acquisition attributes。

因此后续 relation 实际建模的是：

\[
\boxed{
R(U,A|S)
}
\]

而不是忽略 MRI contrast。

---

## 十一、Abnormal Event Decoder

建立：

\[
Q_U=\{q_j^U\}_{j=1}^{M}
\]

使用：

\[
[F^1,F^2,F^3]
\]

得到：

\[
U_j=
Decoder_U(q_j^U,F^{1:3})
\]

统一定义：

\[
\boxed{
U=\text{Abnormal Visual Event}
}
\]

第一篇只区分两类：

\[
U_B=\text{Biological abnormality candidate}
\]

和：

\[
U_Q=\text{Imaging artifact}
\]

其中 artifact 第一篇只做：

\[
\boxed{
Motion+Noise+Aliasing
}
\]

不要扩展到所有 MRI artifact。

---

## 十二、核心模块：Relation Tokens

不直接采用：

\[
MLP(A_i,U_j)\rightarrow R_{ij}
\]

而是对每一个：

\[
(A_i,U_j)
\]

建立显式 **Relation Token**：

\[
r_{ij}^{0}
=
\phi
\left[
A_i,
U_j,
S,
G_{ij}
\right]
\]

其中：

\[
G_{ij}
\]

编码：

- \(\Delta x,\Delta y,\Delta z\)；
- distance；
- overlap；
- laterality；
- hierarchy。

随后，所有 relation tokens 输入一个轻量关系 Transformer：

### Relational Transformer

\[
\{r_{ij}^{0}\}
\rightarrow
Transformer_R
\rightarrow
\{r_{ij}\}
\]

最终：

\[
\boxed{
R_{ij}
=
\sigma(h_R(r_{ij}))
}
\]

整个 architecture 因而从：

> ViT + 多个独立 task heads

转变为：

\[
\boxed{
Patch
\rightarrow
Entity/Event
\rightarrow
Relation
}
\]

这是论文 architecture 的核心。

---

## 十三、Relation GT 如何生成

对于具有 lesion mask 的数据：

\[
M_j^U
\]

和 anatomy mask：

\[
M_i^A
\]

直接定义：

\[
\boxed{
R_{ji}^*
=
\frac{
|M_j^U\cap M_i^A|
}{
|M_j^U|
}
}
\]

因此，可以得到 soft relation。

例如：

\[
lesion
\rightarrow
0.8\times left\ frontal
+
0.2\times adjacent\ region
\]

对于 fastMRI+ / SKM-TEA bounding box，则使用：

\[
IoA(B_j^U,M_i^A)
\]

构造 relation。

因此，无需重新人工标注几十万条 abnormality–anatomy relations。

---

## 十四、Hard Relational Negatives

真实关系：

\[
lesion\rightarrow left\ frontal
\]

构造：

\[
lesion\rightarrow right\ frontal
\]

以及：

\[
lesion\rightarrow left\ parietal
\]

作为 hard negatives。

Knee 中例如：

\[
medial\ meniscus
\leftrightarrow
lateral\ meniscus
\]

因此：

\[
\mathcal L_R
=
\mathcal L_{\rm BCE}
+
\lambda_h\mathcal L_{\rm contrast}
\]

真正迫使模型学习：

\[
\boxed{
\text{WHAT IS WHERE}
}
\]

---

## 十五、第二个核心模块：Relation Observability \(E\)

由 relation token：

\[
r_{ij}
\]

和 anatomy-local feature：

\[
F_i^{local}
\]

预测：

\[
\boxed{
E_{ij}
=
\sigma
\left(
h_E(r_{ij},F_i^{local})
\right)
}
\]

它不是 global image quality。

其定义是：

\[
\boxed{
\text{当前 MRI 是否足够支持 }
U_j\rightarrow A_i
}
\]

例如：

\[
lesion\rightarrow left\ frontal:
E=0.95
\]

加入严重 motion 后：

\[
E=0.18
\]

---

## 十六、Controlled k-Space Intervention

主要在 fastMRI 和 SKM-TEA raw data 上实施。

### 1. Motion

\[
y_t'(k)
=
e^{-i2\pi k^\top\Delta r_t}
y_t(R_tk)
\]

### 2. Noise

\[
y'=y+n
\]

### 3. Undersampling

\[
y'=M_Ry
\]

每个 clean case 生成：

\[
X^0,
X^{q_1},
X^{q_2},
X^{q_3}
\]

此时：

\[
A
\]

以及 underlying biological event：

\[
U_B
\]

并未变化。

但是：

\[
U_Q
\]

和：

\[
E
\]

会随 corruption 改变。

---

## 十七、Intervention Learning 不等于普通 Augmentation

核心约束：

\[
\boxed{
R(TX)\approx\pi_T(R(X))
}
\]

对于轻度 corruption 且：

\[
E_{ij}>\tau
\]

要求 biological relation：

\[
R_B(TX)\approx R_B(X)
\]

同时 artifact relation：

\[
R_Q(TX)
\]

应根据 corruption effect 发生相应变化。

当：

\[
E\downarrow
\]

不再强迫 pathology / abnormality relation 保持高 confidence，而应该：

\[
H(R_B)\uparrow
\]

因此整体学习目标可以概括为：

\[
\boxed{
\text{Selective Invariance}
+
\text{Relational Equivariance}
+
\text{Evidence-aware Uncertainty}
}
\]

---

## 十八、最终 Loss

正文只保留三个大项：

\[
\boxed{
\mathcal L
=
\mathcal L_{\rm semantic}
+
\lambda_R\mathcal L_{\rm relation}
+
\lambda_I\mathcal L_{\rm intervention}
}
\]

其中：

\[
\mathcal L_{\rm semantic}
=
\mathcal L_A
+
\mathcal L_S
+
\mathcal L_U
\]

\[
\mathcal L_{\rm relation}
=
\mathcal L_{\rm bind}
+
\lambda_h\mathcal L_{\rm hard}
\]

\[
\mathcal L_{\rm intervention}
=
\mathcal L_{\rm equiv}
+
\lambda_E\mathcal L_{\rm obs}
+
\lambda_m\mathcal L_{\rm rank}
\]

其中 Observability ranking：

\[
E^{q_0}
>
E^{q_1}
>
E^{q_2}
>
E^{q_3}
\]

对应 artifact severity 逐渐增大。

---

## 十九、训练流程

建议采用四阶段训练，而不是一步端到端硬训。

### Stage I：MRI SSL Pretraining

使用：

\[
fastMRI
+
TotalSegMRI
+
BraTS
+
ISLES
+
SKM
\]

中的 MRI 图像。

训练：

\[
\mathcal L_{\rm MIM}
+
\mathcal L_{\rm contrast}
\]

得到 variable-size 3D MRI backbone。

---

### Stage II：Structured Perception

训练：

\[
A,S,U
\]

目标是先让模型知道：

\[
\text{what exists}
\]

---

### Stage III：Relational Binding

重点训练：

\[
R
\]

大量使用：

\[
\text{wrong-anatomy hard negatives}
\]

迫使模型建立稳定 anatomy–event binding。

---

### Stage IV：Intervention + Observability

在 raw k-space 上生成：

\[
motion/noise/undersampling
\]

训练：

\[
R,E
\]

KMAR-50K 用于真实 paired motion calibration。

MR-ART 不参与训练，专门保留用于：

\[
\boxed{
synthetic\rightarrow real
}
\]

external validation。

---

## 二十、论文最重要的五个实验

### Experiment 1：Recognition–Binding Gap

首先使用一个普通强 3D Swin backbone，证明即使：

\[
Acc(A),Acc(U)
\]

很高，

\[
ABA(U\rightarrow A)
\]

仍可能明显下降。

如果该 gap 不存在，则整个课题的核心 motivation 需要重新审视。

---

### Experiment 2：Relation-Centric vs Multi-task ViT

所有模型使用相同：

- backbone；
- training data；
- labels。

比较：

1. 普通 multi-task ViT；
2. attention alignment；
3. entity-centric model；
4. Relation Transformer。

核心问题是：

> 显式关系建模是否真正提高 anatomy–abnormality binding，而不仅是增加参数量？

---

### Experiment 3：Intervention vs Ordinary Augmentation

两组模型均见过：

- motion；
- noise；
- undersampling。

但是只有 AnatoBind 使用：

\[
R(TX)\approx\pi_T(R(X))
\]

如果后者才能显著提升 relation robustness，才能说明：

> intervention learning 不是将 data augmentation 换一个名字。

---

### Experiment 4：Compositional Split

训练组合：

\[
Brain+Motion
\]

\[
Brain+Noise
\]

\[
Knee+Noise
\]

hold out：

\[
\boxed{
Knee+Motion
}
\]

测试 unseen：

\[
Anatomy\times Event
\]

composition。

---

### Experiment 5：Observability Failure Prediction

使用：

\[
E
\]

预测某条 relation 是否即将失效。

报告：

\[
AURC
\]

\[
ECE
\]

\[
Brier
\]

\[
AUROC
\]

并与以下 uncertainty baselines 比较：

- maximum softmax confidence；
- predictive entropy；
- relation confidence；
- global image quality score。

如果 \(E\) 不能显著优于普通 confidence / entropy，则 \(E\) 不足以构成独立贡献。

---

## 二十一、核心评价指标

最终主表不要以 Dice 为主。

### 1. Anatomy–Abnormality Binding Accuracy

\[
\boxed{
ABA=\text{Anatomy–Abnormality Binding Accuracy}
}
\]

### 2. Relation Detection

\[
\boxed{
Relation\ mAP/F1
}
\]

### 3. Laterality

\[
\boxed{
Laterality\ Binding\ Accuracy
}
\]

### 4. Intervention Robustness

\[
\boxed{
RIC=\text{Relational Intervention Consistency}
}
\]

### 5. Observability Calibration

\[
\boxed{
AURC/ECE/Brier
}
\]

Segmentation Dice 和 lesion detection AP 只作为辅助指标。

---

## 二十二、最终模型具备什么能力

训练完成的 ViT 具备五种底层 perception capability。

### 1. Anatomy

\[
\boxed{
A:\text{识别并定位 anatomy}
}
\]

### 2. Sequence / Acquisition

\[
\boxed{
S:\text{理解 MRI sequence/acquisition}
}
\]

### 3. Abnormal Event

\[
\boxed{
U:\text{检测异常视觉事件}
}
\]

包括：

- biological abnormality candidate；
- imaging artifact。

### 4. Relation

\[
\boxed{
R:\text{知道异常属于/影响哪个 anatomy}
}
\]

### 5. Observability

\[
\boxed{
E:\text{知道当前 MRI 是否足以支持这条关系}
}
\]

因此，同一个 backbone 后续可以连接：

- Automatic QC；
- Abnormality Candidate Detection；
- Segmentation；
- Quantification；
- VLM。

第一篇论文不需要全部展开，主要证明前五种 perception capabilities。

---

## 二十三、论文真正的三个贡献

### Contribution 1：MRI Relational Binding Problem

提出：

\[
\boxed{
\text{recognizing WHAT and WHERE separately}
\neq
\text{knowing WHAT IS WHERE}
}
\]

即现有 MRI representation learning 通常分别关注：

- anatomy recognition；
- lesion detection；
- image quality。

但并没有明确研究异常视觉事件与解剖实体之间是否形成可靠、可组合的关系表示。

---

### Contribution 2：Relation-Centric Variable-Size 3D MRI Transformer

提出：

\[
\boxed{
Patch
\rightarrow
Entity/Event
\rightarrow
Relation
\rightarrow
Observability
}
\]

采用：

- variable-size hierarchical 3D Swin；
- physical-coordinate positional encoding；
- Anatomy Entity Tokens；
- Abnormal Event Tokens；
- explicit Relation Tokens；
- Relation Transformer。

使不同：

\[
D\times H\times W
\]

的 MRI 不需要固定 resize。

---

### Contribution 3：Evidence-Aware Intervention-Equivariant Relational Learning

提出：

\[
\boxed{
\textbf{Evidence-Aware Intervention-Equivariant Relational Learning}
}
\]

利用 MRI 公开 raw k-space 数据能够进行物理可控干预这一特点，不要求整个 representation 对 corruption 完全 invariant，而要求：

\[
\boxed{
R(TX)\approx\pi_T(R(X))
}
\]

并通过：

\[
E
\]

判断哪些 relation 在当前 observation 下仍然具有充分视觉证据。

本质上实现：

\[
\boxed{
\text{Selective Invariance}
+
\text{Relational Equivariance}
+
\text{Evidence-Aware Uncertainty}
}
\]

---

# 最终论文主线

不要把论文写成：

> “我们训练了一个 MRI foundation model，可以做很多任务。”

更合适的主问题是：

> **现有 MRI encoder 能识别实体，但是否真正形成可靠、可组合的异常—解剖关系？**

围绕这一问题，通过：

\[
\boxed{
\text{Variable-size 3D ViT}
+
\text{Relation Tokens}
+
\text{Controlled MRI Intervention}
+
\text{Relation Observability}
}
\]

建立关系中心的 MRI representation learning framework。

第一篇论文应优先把：

\[
\boxed{
Brain+Knee
}
\]

做扎实，并使用：

\[
\boxed{
Prostate
}
\]

完成 cross-organ external generalization。

不建议第一篇继续加入：

- gaze；
- LLM；
- active search；
- 大量额外器官；
- report generation。

否则会稀释最核心的机器学习贡献：

\[
\boxed{
\textbf{MRI Relational Binding}
}
\]

以及：

\[
\boxed{
\textbf{Evidence-Aware Relational Robustness}
}
\]