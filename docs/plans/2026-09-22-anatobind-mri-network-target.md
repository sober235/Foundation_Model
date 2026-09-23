> **来源与用途（2026-09-23 拷入）**：原文件 `/data1/data1_congcong/code/FM_MRI/AnatoBind-MRI_research_plan.md`（2026-09-22 22:54 写入，md5 `c8d0cb99105c809e7a2220443e073ea7`，是 09-04 `AnatoBind-MRI_cui.md` 的改写版，正文相似度 0.87）。按 2026-09-23 决定 Q8(c)：这份文档定义"最终要搭什么网络"（§4–§12 的架构）；"什么算证据"（真值来源、Gate、统计口径、NOT_EVIDENCE 规则）以 `docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md` 为准。它引用的 BraTS 2021、MR-ART、PI-CAI 都不在 `/data2/congcong/data/FM_data`，TotalSegmentator MRI 实为 50 类；数据事实以 `docs/superpowers/specs/2026-09-23-aur-capability-system-design.md` §5 的实测为准。

# AnatoBind-MRI

## Evidence-Aware Relational Representation Learning with a Variable-Size 3D MRI Transformer

> 核心目标：训练一个具有结构化 MRI perception 能力的统一视觉编码器，而不是做一个“大而全”的 MRI foundation model。

整体建模流程：

\[
\boxed{
X\rightarrow A,S,U\rightarrow R\rightarrow E
}
\]

其中：

\[
A=\text{Anatomy},\quad
S=\text{Sequence/Acquisition},
\]

\[
U=\text{Abnormal Visual Event},
\]

\[
R=\text{Event--Anatomy Relation},
\]

\[
E=\text{Relation Observability}.
\]

论文真正的创新集中在：

\[
\boxed{
R+E+\text{Controlled Intervention}
}
\]

而不是简单做 anatomy / lesion / artifact 多任务分类。

---

## 1. 论文解决什么问题

普通 MRI ViT 即使能够分别识别：

\[
\text{lesion}
\]

和：

\[
\text{left frontal lobe},
\]

也不保证其真正建立：

\[
\boxed{
\text{lesion}\rightarrow\text{left frontal lobe}.
}
\]

因此第一核心问题是：

\[
\boxed{
\text{Recognition}\neq\text{Relational Binding}.
}
\]

进一步，在 motion、noise、undersampling 后，即使真实病灶没有改变，当前图像也可能已经不足以支撑：

\[
\text{lesion}\rightarrow A_i.
\]

所以第二核心问题是：

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

## 2. 数据集设计：Brain + Knee + Prostate

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

### 2.1 TotalSegmentator MRI

TotalSegmentator MRI 有 616 个经过人工审核的临床 MRI，并提供 80 个 anatomical structures。数据跨不同序列、field strength、30 台 scanner、3 个 manufacturer，因此适合作为 broad anatomy supervision。

### 2.2 fastMRI 与 fastMRI+

fastMRI 是最核心的数据源之一。公开 raw data 包括约：

- 1,398 个 knee scans；
- 约 7,002 个 brain scans；
- 312 个 prostate examinations。

Brain 数据覆盖：

- T1；
- T1 post-contrast；
- T2；
- FLAIR。

fastMRI+ 为 knee 和约 1,000 个 brain datasets 增加 radiologist pathology bounding-box annotations，适合训练“abnormality candidate”而不是最终临床诊断定位。

### 2.3 SKM-TEA

SKM-TEA 虽然规模较小，但单个数据集同时具有：

- raw k-space；
- DICOM；
- 6 类组织 segmentation；
- 16 类 pathology bounding boxes。

因此它是训练 \(A,U,R\) 以及 controlled intervention 的关键闭环数据源。

### 2.4 BraTS 与 ISLES

Brain lesion supervision 主要使用 BraTS 与 ISLES。

BraTS 2021 提供多机构 multiparametric brain MRI 与 tumor compartment segmentation。

ISLES 2022 的公开训练集包含 250 例：

- DWI；
- ADC；
- FLAIR；
- stroke lesion masks。

### 2.5 KMAR-50K 与 MR-ART

真实 QC 使用 KMAR-50K。该数据集包含：

- 1,190 名患者；
- 1,444 组 paired knee MRI；
- 62,506 slices；
- motion-corrupted scan；
- rescan reference。

数据覆盖 1.5T/3T、多方向、多序列。

MR-ART 提供 148 名受试者的 matched motion-free / 两级 motion T1 MRI，建议保留为 synthetic-to-real motion 外部验证集。

### 2.6 PI-CAI

PI-CAI 的公开部分包含约 1,500 例 prostate MRI，其中约 425 例为 csPCa-positive，并具有 lesion annotation。

推荐将其用于最终 cross-organ transfer，而不是参与主模型训练。

注意其公开数据为 CC BY-NC 4.0，适用于科研，但未来商业化需要重新进行 license audit。

---

## 3. 数据预处理原则：不同尺寸 MRI 不统一 resize

不采用：

\[
X\rightarrow128^3
\]

这种固定矩阵 resize。

原因是它可能改变：

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

允许输入矩阵：

\[
D\times H\times W
\]

在不同病例之间保持不同尺寸。

对于极端 spacing，可以做温和 resampling，但**不要求所有病例具有相同 matrix size**。

每个 patch 同时保留：

\[
(x,y,z)_{\rm physical}
\]

以及：

\[
(\tilde x,\tilde y,\tilde z)\in[-1,1]^3.
\]

---

## 4. Backbone：Variable-Size 3D Hierarchical Swin

不采用 flat vanilla ViT。

原因是 hierarchical shifted-window architecture 更适合高分辨率 3D MRI，并天然提供 multi-scale features。

推荐第一版配置：

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
[2,4,8,16].
}
\]

Window：

\[
\boxed{
W=(4,8,8)
}
\]

---

## 5. 3D Patch Stem

输入：

\[
X\in\mathbb R^{1\times D\times H\times W}.
\]

采用：

\[
Conv3D
\]

进行 patch embedding。

第一版推荐：

\[
\boxed{
P=(2,4,4)
}
\]

即：

\[
Z^0=
Conv3D_{k=s=(2,4,4)}(X).
\]

原因是许多临床 MRI 存在明显 through-plane anisotropy，\(2\times4\times4\) 相比粗暴 \(4^3\) 更稳妥。

论文中建议增加如下 ablation：

\[
2\times4\times4
\quad vs\quad
4^3.
\]

---

## 6. Variable-size 输入的具体实现

假设输入：

\[
X_1=155\times187\times143
\]

和：

\[
X_2=201\times224\times96.
\]

模型均直接接受。

仅 pad 到层级下采样和 window 所需的最近整数倍：

\[
X\rightarrow\tilde X.
\]

同时建立：

\[
M_{\rm valid}.
\]

其中：

\[
M=1
\]

表示真实 MRI 区域；

\[
M=0
\]

表示 padding。

Attention、pooling 和 loss 均忽略：

\[
M=0.
\]

因此：

\[
\boxed{
\text{模型没有固定 }D,H,W.
}
\]

训练时采用 size bucket，将相近尺寸的病例划入同一 batch，以减少 padding waste。

---

## 7. 位置编码：Physical-coordinate 3D RoPE

不使用固定：

\[
E_{\rm pos}\in\mathbb R^{N_0\times C}.
\]

否则输入尺寸改变时需要插值。

采用：

\[
\boxed{
3D\ RoPE
}
\]

分别编码：

\[
x,\quad y,\quad z.
\]

对第 \(n\) 个 patch：

\[
c_n=
(x_n^{mm},y_n^{mm},z_n^{mm}).
\]

位置编码为：

\[
PE_n
=
RoPE_x(x_n)
\oplus
RoPE_y(y_n)
\oplus
RoPE_z(z_n).
\]

再加入 normalized local coordinate：

\[
\tilde c_n\in[-1,1]^3.
\]

最终：

\[
\boxed{
PE=
PE_{\rm physical}
+
PE_{\rm local}.
}
\]

该设计对于跨：

\[
Brain\rightarrow Knee\rightarrow Prostate
\]

尤其重要。

---

## 8. Backbone 多尺度输出

得到：

\[
F^1,F^2,F^3,F^4.
\]

其中：

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
F^4:\text{organ/global quality}.
\]

因此不同任务不必全部依赖最后一层特征。

---

## 9. Anatomy Entity Decoder

建立：

\[
Q_A=\{q_i^A\}_{i=1}^{K}.
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
(q_i^A,F^{2:4}).
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
\text{entity embedding}.
}
\]

例如：

- brainstem；
- left frontal；
- ventricle；
- medial meniscus。

同时增加轻量 mask head：

\[
A_i\rightarrow M_i^A.
\]

Mask 主要用于 spatial grounding 和 relation GT，并不是论文的最终目标。

---

## 10. Sequence / Acquisition Token

引入 global acquisition token：

\[
S.
\]

输入：

\[
F^4+\text{available metadata}.
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
- 其他可用 acquisition attributes。

因此后续 relation 实际建模为：

\[
\boxed{
R(U,A|S).
}
\]

而不是忽略 MRI contrast。

---

## 11. Abnormal Event Decoder

建立：

\[
Q_U=\{q_j^U\}_{j=1}^{M}.
\]

使用：

\[
[F^1,F^2,F^3]
\]

得到：

\[
U_j=
Decoder_U(q_j^U,F^{1:3}).
\]

统一定义：

\[
\boxed{
U=\text{Abnormal Visual Event}.
}
\]

第一篇论文只划分为：

\[
U_B=\text{Biological abnormality candidate}
\]

和：

\[
U_Q=\text{Imaging artifact}.
\]

Artifact 第一篇只做：

\[
\boxed{
Motion+Noise+Aliasing.
}
\]

不要扩张到所有 MRI artifact。

---

## 12. 核心模块：Relation Tokens

不直接使用：

\[
MLP(A_i,U_j)\rightarrow R_{ij}.
\]

对每一个：

\[
(A_i,U_j)
\]

建立显式 Relation Token：

\[
r_{ij}^{0}
=
\phi
\left[
A_i,
U_j,
S,
G_{ij}
\right].
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

然后所有 relation tokens 输入一个轻量 Relational Transformer：

\[
\{r_{ij}^{0}\}
\rightarrow
Transformer_R
\rightarrow
\{r_{ij}\}.
\]

最终：

\[
\boxed{
R_{ij}
=
\sigma(h_R(r_{ij})).
}
\]

因此整个 architecture 不再是：

> ViT + 多个独立 prediction heads

而是：

\[
\boxed{
Patch
\rightarrow
Entity/Event
\rightarrow
Relation.
}
\]

这才是论文 architecture 的核心。

---

## 13. Relation GT 的构建

对于具有 lesion mask 的数据：

\[
M_j^U
\]

和 anatomy mask：

\[
M_i^A,
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
}.
}
\]

从而得到 soft relation，例如：

\[
lesion
\rightarrow
0.8\times left\ frontal
+
0.2\times adjacent\ region.
\]

对于 fastMRI+ / SKM-TEA bounding box，可以使用：

\[
IoA(B_j^U,M_i^A)
\]

构造 relation。

这样无需重新人工标注大量 relation labels。

---

## 14. Hard Relational Negatives

例如真实关系：

\[
lesion\rightarrow left\ frontal.
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

Knee 中可以构造：

\[
medial\ meniscus
\leftrightarrow
lateral\ meniscus.
\]

Relation loss：

\[
\mathcal L_R
=
\mathcal L_{\rm BCE}
+
\lambda_h\mathcal L_{\rm contrast}.
\]

其目标是真正逼迫模型学习：

\[
\boxed{
\text{WHAT IS WHERE}.
}
\]

---

## 15. Relation Observability \(E\)

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
\right).
}
\]

其含义不是 global image quality，而是：

\[
\boxed{
\text{当前 MRI 是否足够支持 }
U_j\rightarrow A_i.
}
\]

例如：

\[
lesion\rightarrow left\ frontal:
E=0.95.
\]

加入严重 motion 后：

\[
E=0.18.
\]

---

## 16. Controlled k-space Intervention

主要在 fastMRI 和 SKM-TEA raw data 上执行。

### 16.1 Motion

\[
y_t'(k)
=
e^{-i2\pi k^\top\Delta r_t}
y_t(R_tk).
\]

### 16.2 Noise

\[
y'=y+n.
\]

### 16.3 Undersampling

\[
y'=M_Ry.
\]

每个 clean case 生成：

\[
X^0,
X^{q_1},
X^{q_2},
X^{q_3}.
\]

此时 anatomy：

\[
A
\]

以及 underlying biological event：

\[
U_B
\]

不发生变化。

但：

\[
U_Q
\]

和：

\[
E
\]

会发生变化。

---

## 17. Intervention Learning 不等于普通 Data Augmentation

核心约束：

\[
\boxed{
R(TX)\approx\pi_T(R(X)).
}
\]

对于轻度 corruption 且：

\[
E_{ij}>\tau,
\]

要求 biological relation：

\[
R_B(TX)\approx R_B(X).
\]

同时 artifact relation：

\[
R_Q(TX)
\]

应根据 corruption effect 发生变化。

当：

\[
E\downarrow,
\]

不再强迫 pathology / abnormality relation 保持高 confidence，而应该满足：

\[
H(R_B)\uparrow.
\]

因此训练目标可以概括为：

\[
\boxed{
\text{Selective Invariance}
+
\text{Relational Equivariance}
+
\text{Evidence-aware Uncertainty}.
}
\]

---

## 18. 最终 Loss

正文建议只保留三个一级项：

\[
\boxed{
\mathcal L
=
\mathcal L_{\rm semantic}
+
\lambda_R\mathcal L_{\rm relation}
+
\lambda_I\mathcal L_{\rm intervention}.
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
\mathcal L_U.
\]

\[
\mathcal L_{\rm relation}
=
\mathcal L_{\rm bind}
+
\lambda_h\mathcal L_{\rm hard}.
\]

\[
\mathcal L_{\rm intervention}
=
\mathcal L_{\rm equiv}
+
\lambda_E\mathcal L_{\rm obs}
+
\lambda_m\mathcal L_{\rm rank}.
\]

Observability ranking：

\[
E^{q_0}
>
E^{q_1}
>
E^{q_2}
>
E^{q_3}
\]

对应 artifact severity 持续增大。

---

## 19. 训练流程

建议采用四阶段训练，而不是一步端到端硬训。

### Stage I：MRI SSL Pretraining

使用：

\[
fastMRI+TotalSegMRI+BraTS+ISLES+SKM
\]

图像进行自监督预训练。

训练：

\[
\mathcal L_{\rm MIM}
+
\mathcal L_{\rm contrast}.
\]

得到 variable-size 3D MRI backbone。

### Stage II：Structured Perception

训练：

\[
A,S,U.
\]

目标是先让模型知道：

\[
\text{what exists}.
\]

### Stage III：Relational Binding

重点训练：

\[
R.
\]

大量使用：

\[
\text{wrong-anatomy hard negatives}.
\]

### Stage IV：Intervention + Observability

在 raw k-space 上生成：

\[
motion/noise/undersampling.
\]

训练：

\[
R,E.
\]

KMAR-50K 用于真实 paired motion calibration。

MR-ART 不参与训练，保留为 synthetic-to-real external test。

---

## 20. 最关键的五个实验

### 20.1 Recognition–Binding Gap

使用普通强 3D Swin backbone，验证即使：

\[
Acc(A),Acc(U)
\]

较高，

\[
ABA(U\rightarrow A)
\]

仍可能明显下降。

如果该 gap 不存在，则整个课题的 motivation 需要重新审视。

### 20.2 Relation-Centric vs Multi-task ViT

所有模型使用相同 backbone、数据和标签，比较：

- 普通 multi-task；
- attention alignment；
- entity-centric；
- Relation Transformer。

目标是验证显式 relation modeling 是否真正带来收益。

### 20.3 Intervention vs Ordinary Augmentation

两组模型都见过 motion/noise。

只有 AnatoBind 使用：

\[
R(TX)\approx\pi_T(R(X)).
\]

如果后者才能显著提升 relation robustness，才说明 intervention learning 并不是简单给 data augmentation 换名字。

### 20.4 Compositional Split

训练组合：

\[
Brain+Motion,
Brain+Noise,
Knee+Noise
\]

hold out：

\[
\boxed{
Knee+Motion.
}
\]

测试 unseen Anatomy × Event composition。

### 20.5 Observability Failure Prediction

使用：

\[
E
\]

预测 relation 是否会失败。

报告：

\[
AURC,\ ECE,\ Brier,\ AUROC.
\]

如果 \(E\) 不能优于普通 softmax confidence / entropy，则 \(E\) 不能作为独立贡献。

---

## 21. 核心评价指标

最终主表不应以 Dice 为主。

建议主指标包括：

\[
\boxed{
ABA=\text{Anatomy–Abnormality Binding Accuracy}
}
\]

\[
\boxed{
Relation\ mAP/F1
}
\]

\[
\boxed{
Laterality\ Binding\ Accuracy
}
\]

\[
\boxed{
RIC=\text{Relational Intervention Consistency}
}
\]

以及：

\[
\boxed{
AURC/ECE/Brier
}
\]

用于评价 Observability。

Segmentation Dice、lesion detection AP 只作为辅助指标。

---

## 22. 最终模型具备的底层能力

训练完成的 ViT 具备五种核心能力：

\[
\boxed{
A:\text{识别并定位 anatomy}
}
\]

\[
\boxed{
S:\text{理解 MRI sequence/acquisition}
}
\]

\[
\boxed{
U:\text{检测异常视觉事件}
}
\]

其中包括：

- biological abnormality candidate；
- imaging artifact。

同时：

\[
\boxed{
R:\text{知道异常属于/影响哪个 anatomy}
}
\]

以及：

\[
\boxed{
E:\text{知道当前 MRI 是否足以支持这条关系}
}
\]

因此同一个 backbone 后续可以接：

- Automatic QC；
- Abnormality Candidate Detection；
- Segmentation；
- Quantification；
- VLM。

第一篇论文不需要全部实现，主要证明前五种 perception capability。

---

## 23. 论文真正的三个贡献

### Contribution 1：MRI Relational Binding Problem

提出：

\[
\boxed{
\text{recognizing WHAT and WHERE separately}
\neq
\text{knowing WHAT IS WHERE}.
}
\]

核心观点是：单独识别 anatomy 和 abnormality，并不代表模型真正理解二者的空间关系。

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
Observability.
}
\]

模型结合：

- variable-size hierarchical Swin；
- 3D physical-coordinate positional encoding；
- Anatomy Entity Decoder；
- Abnormal Event Decoder；
- Relation Tokens；
- Relational Transformer；
- Relation Observability。

不同 \(D\times H\times W\) MRI 不需要统一到固定 matrix size。

### Contribution 3：Evidence-Aware Intervention-Equivariant Relational Learning

利用 MRI raw k-space 的物理可干预性，不要求整个 feature invariant，而要求：

\[
\boxed{
R(TX)\approx\pi_T(R(X))
}
\]

并进一步由：

\[
E
\]

判断哪些 relation 在当前 observation 下仍值得相信。

---

## 24. 最终论文主线

不要把论文写成：

> “我们训练了一个 MRI foundation model，可以做很多任务。”

更适合的叙事是：

\[
\boxed{
\textbf{现有 MRI encoder 能识别实体，但是否真正形成可靠、可组合的异常—解剖关系？}
}
\]

我们通过：

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

解决该问题。

第一版建议只把 Brain/Knee 做扎实，并使用 Prostate 做外部泛化测试。

不建议继续加入：

- gaze；
- LLM；
- active search；
- 更多器官。

否则会稀释论文最核心的 machine learning contribution。

---

## 25. 一句话总结

AnatoBind-MRI 的核心不是“让 MRI 模型同时做更多任务”，而是让模型从：

\[
\text{识别 anatomy}
+
\text{识别 abnormality}
\]

进一步走向：

\[
\boxed{
\text{识别异常是什么}
\rightarrow
\text{异常在哪里}
\rightarrow
\text{异常属于哪个解剖结构}
\rightarrow
\text{当前图像是否仍足以支持这条关系}
}
\]

最终形成一个面向 MRI 的 **evidence-aware relational representation learning framework**。
