# Core-target reframing review: Anatomy → Lesion → Binding

日期：2026-09-22  
对象：当前 `main@d653c3d`  
性质：研究定位与执行优先级复审；这是建议稿，不代表已经修改 `RESEARCH_PLAN.md` 或已经验证完整模型。

---

## 0. 结论

当前项目应把第一原则重新定义为：

> **从 MRI 中建立解剖实体，识别病灶实体及其类型与空间位置，并显式判断每个病灶位于哪个解剖结构。**

最小核心输出应收缩为：

$
f_\theta(X)=\{A,U,R\}
$

其中：

- (A)：anatomical entities，解剖结构；
- (U)：lesion / abnormality entities，病灶事件；
- (R)：lesion–anatomy binding，病灶与解剖结构之间的宿主关系。

推荐把当前研究主线重构为：

$
X \rightarrow (A,U) \rightarrow R
$

而不是继续让 acquisition degradation、reliability、motion、scanner shift 或大型 Relation Transformer 定义整个项目。

这些模块仍然有价值，但应降为：

1. 对核心 structured perception 的 stress test；
2. error propagation 研究；
3. 第二阶段 robustness / reliability 扩展。

---

## 1. 用户核心目标的精确定义

项目最终应能够回答三个层次的问题。

### 1.1 解剖结构识别：What anatomy is present?

输入 MRI (X)，输出：

$
A=\{A_1,\dots,A_K\}
$

每个解剖实体建议表示为：

$
A_i=(c_i^A,M_i,b_i,p_i,a_i)
$

其中：

- (c_i^A)：解剖类别；
- (M_i)：mask；
- (b_i)：bounding box；
- (p_i)：物理坐标中心；
- (a_i)：anatomy token / embedding。

关键点：模型首先建立一个显式 anatomical coordinate system，而不是只产生一个语义分割图。

---

### 1.2 病灶识别：What lesion is present and where is it?

输出：

$
U=\{U_1,\dots,U_M\}
$

每个 lesion entity 建议表示为：

$
U_j=(c_j^U,b_j,m_j,p_j,s_j,u_j)
$

其中：

- (c_j^U)：病灶类型；
- (b_j)：3D box；
- (m_j)：可选 lesion mask；
- (p_j=(x,y,z)_{mm})：物理空间位置；
- (s_j)：size / extent；
- (u_j)：lesion token / embedding。

因此“病灶在哪里”不应只等价于 voxel-space bounding box。至少应在输出接口中保留毫米坐标、病灶大小以及空间范围。

---

### 1.3 病灶—解剖绑定：Which anatomy hosts this lesion?

对每个 lesion (U_j)，在所有候选 anatomy (A_i) 上计算：

$
P(A_i\mid U_j,X)
$

最终：

$
\hat A_j=\arg\max_i P(A_i\mid U_j,X)
$

模型输出应能形成类似：

$
\text{Lesion}_j
\rightarrow
\text{lesion type}
\rightarrow
\text{3D location}
\rightarrow
\text{host anatomy}
$

的结构化结果。

例如脑 MRI：

> WMH → left cerebral white matter

进一步可以扩展：

> WMH → left cerebral white matter → adjacent to left lateral ventricle

但“邻接”“压迫”“侵犯”等更丰富 relation 暂时不是第一阶段必需项。

---

## 2. 对当前 V7 定位的修正

当前 V7 把主要论证收缩到：

$
\text{degradation}
\rightarrow
\text{geometry failure}
\rightarrow
\text{binding failure}
\rightarrow
\text{relation rescue}
$

这条链本身仍有研究价值，但不应再作为项目的定义。

原因是用户真正需要的能力在 clean MRI 上也成立：

$
X\rightarrow A,quad
X\rightarrow U,quad
(A,U)\rightarrow R
$

Relation 不应只因为 segmentation 在 q3 噪声下会错才存在。即使在 clean MRI 中，显式输出 lesion-to-anatomy assignment 仍然是核心任务。

因此建议新的层级关系为：

### Core

$
X\rightarrow(A,U)\rightarrow R
$

### Robustness study

$
X^q\rightarrow(A^q,U^q,R^q)
$

### Reliability extension

$
R^q\rightarrow E
$

也就是说，degradation 是用于检验 structured perception 的干预变量，而不是定义 structured perception 本身。

---

## 3. 建议的核心模型

建议第一版核心模型只保留三个模块。

### Stage I — Anatomy Parsing

$
X\rightarrow\{A_i\}_{i=1}^{K}
$

目标：

- anatomy segmentation；
- anatomy identity；
- physical position；
- anatomy token。

现有 identity-anchored anatomy branch 可以继续保留，但应明确其角色是产生 anatomy entities，而不仅是给后续 relation 提供 mask。

---

### Stage II — Lesion Parsing

$
X\rightarrow\{U_j\}_{j=1}^{M}
$

目标：

- lesion detection / segmentation；
- lesion classification；
- 3D spatial localization；
- lesion token。

当前 dense centre-heatmap 路线比旧 DETR 查询更符合现有实验证据，可以继续作为一个实现选择。

但必须把“病灶分类”和“病灶空间位置”都写进正式输出接口，而不是只服务于后续 binding。

---

### Stage III — Anatomical Binding

对每个 lesion 单独进行 host competition：

$
r_{ij}
=
f(
a_i,
u_j,
g_{ij},
l_{ij}
)
$

其中：

- (a_i)：anatomy embedding；
- (u_j)：lesion embedding；
- (g_{ij})：physical geometry；
- (l_{ij})：local image / boundary evidence。

然后：

$
P(host_i\mid U_j)
=
\operatorname{softmax}_i h(r_{ij})
$

推荐使用 **per-lesion host competition**，而不是继续默认当前 `relation.py` 的 global (K\times M) flattened pair Transformer。

原因：当前任务本质上是：

$
\text{one lesion}
+
\text{candidate anatomies}
\rightarrow
\text{one host distribution}
$

首先应证明这一最小问题本身需要 learned relational evidence，再决定是否引入更复杂的全局 graph / Transformer。

---

## 4. Geometry 必须加强，但 geometry 不是 relation 的全部

当前 `relation.py` 中：

$
G_{ij}=(\Delta z,\Delta y,\Delta x,\|\Delta\|,IoA)
$

过于简单。

建议强几何基线和 relation model 至少共享：

- centroid displacement in mm；
- signed distance to candidate surface；
- minimum distance to candidate structure；
- soft / hard IoA；
- lesion-to-boundary distance；
- lesion size / extent；
- spacing / slice thickness；
- coarse normalized position；
- side / hemisphere（若适用）。

这样可以构建真正公平的 (B_{geo+})。

只有 relation model 在共享这些 geometry 后仍然从 (l_{ij}) 或 joint representation 中获得增益，才能支持：

> model learned information beyond geometry lookup.

否则 relation module 很可能只是复杂地实现了一个 distance classifier。

---

## 5. 当前最关键的基线矩阵

第一阶段不需要很多花哨模块，建议固定如下。

### B0 — geometry lookup

segmentation + overlap / nearest structure。

### Bprior — anatomical prior

$
P(host\mid lesion\ type,side,coarse\ location)
$

用来排除“模型只是记住病灶通常出现在哪”的 shortcut。

### Bgeo+ — strong geometry classifier

例如 logistic regression / XGBoost / 2-layer MLP，输入完整 geometry feature。

### Broi — direct local ROI classifier

直接从 lesion ROI / local context 预测 host，不显式进行 relation pair modeling。

### Brel — proposed per-lesion binding model

输入 anatomy token + lesion token + geometry + local boundary/image evidence。

### Boracle

clean / GT geometry，仅作为上限，不参与 superiority claim。

---

## 6. 第一篇论文真正要证明的三项能力

新的主实验应该首先回答：

### E1. Anatomy perception

$
X\rightarrow A
$

指标：

- Dice；
- HD95；
- anatomy detection / presence；
- identity consistency。

---

### E2. Lesion perception

$
X\rightarrow U
$

指标：

- lesion recall / FROC；
- lesion classification；
- box IoU / mask Dice；
- physical localization error。

---

### E3. Lesion–anatomy binding

$
(A,U)\rightarrow R
$

指标：

- host accuracy；
- macro-F1；
- top-k host accuracy；
- per-anatomy / per-lesion-type breakdown。

这三个结果共同构成项目最基本的“structured MRI perception”证据。

---

## 7. 真正决定 relation 是否必要的实验

核心问题不是：

> Transformer 是否比 lookup 高几个点？

而是：

> 在公平、强大的替代方法存在时，显式 learned lesion–anatomy binding 是否仍提供独立信息？

因此需要比较：

$
B_{rel}
\quad vs.\quad
B_0,,
B_{prior},,
B_{geo+},,
B_{roi}
$

推荐继续使用当前 V7 中定义的：

$
\text{rescue}
=
P(B_0\ wrong,B_{rel}\ correct)
$

$
\text{harm}
=
P(B_0\ correct,B_{rel}\ wrong)
$

$
\text{net rescue}
=
\text{rescue}-\text{harm}
$

但应把这个指标从“退化专用指标”提升为 binding module 的通用评价之一。

如果 (B_{rel}) 不能稳定优于 (B_{geo+})、(B_{prior}) 和 (B_{roi})，不应继续把 Relation Transformer / relation module 作为论文主要贡献。

---

## 8. Degradation 实验的新角色：研究 error propagation

已有 brain probe 仍然很重要，但它应重新解释为：

> structured perception 的 stress test。

对同一患者：

$
X\rightarrow(A,U,R)
$

$
X^q\rightarrow(A^q,U^q,R^q)
$

分别追踪三个层面的变化：

$
\Delta A,quad
\Delta U,quad
\Delta R
$

真正有价值的问题是：

> relation-level error 是否会被很小的 anatomy / lesion perception error 放大？

例如如果：

$
Dice(A)\downarrow 2\%
$

但：

$
Accuracy(R)\downarrow 15\%
$

则可以提出比“noise hurts segmentation”更强的科学结论：

> small local anatomical perturbations can be amplified into clinically meaningful lesion–anatomy relation errors.

因此现有 q1/q2/q3 probe 不应删除，而应从“项目存在的理由”改为“结构化感知的鲁棒性机制实验”。

---

## 9. Gate A′ 仍然必须保留

虽然 degradation 不再定义项目，但 robust segmentation 仍然是必须击败的反事实。

需要检验：

$
\text{corruption-trained robust segmentation}
\rightarrow
\text{does it remove most binding failures?}
$

如果是，则说明 degradation 下的大部分 relation error 源于 perception error。

但这不会否定 clean MRI 上的 (A/U/R) structured perception 目标；它只会否定：

> degradation rescue 必须依赖 relation module

这一更强的 claim。

因此 Gate A′ 的意义从“整个项目生死门”调整为：

> robustness mechanism claim 的生死门。

---

## 10. Level R 人工 relation truth 仍然必要

当前 brain probe 的 clean SynthSeg assignment 存在 reference instability。

所以：

$
\text{return to clean pseudo-reference}
\neq
\text{return to clinical truth}
$

这一判断仍然成立。

如果要声称：

> relation model gives more correct host assignment

最终必须在人标 relation reference 上验证，而不能只把 clean segmentation lookup 当 truth。

建议：

- 优先标 geometry-conflict lesions；
- 同时加入 2–3 倍 matched stable controls；
- 双阅片 + adjudication；
- 记录 uncertain / ambiguous cases，而不是强制单标签。

---

## 11. 第一阶段应主动移出的模块

在核心 (A/U/R) 没有被验证前，不建议继续投入：

- (U_Q) acquisition degradation head；
- scanner / vendor factor；
- motion modeling；
- aggressive undersampling；
- complex reliability (E_{ij})；
- global (K\times M) relation Transformer；
- full SSL / foundation-model pretraining；
- cross-organ generalization；
- end-to-end joint retraining。

这些内容不是错误，而是当前优先级不对。

---

## 12. Reliability E 的新位置

只有在 (R) 本身被证明有独立价值以后，再增加：

$
E_j=P(\hat R_j\ correct\mid X)
$

或 event-level selective confidence。

要求至少比较：

- softmax confidence；
- entropy；
- anatomy uncertainty；
- lesion detection uncertainty；
- local NRMSE / degradation proxies；
- ConfidNet-like failure predictor。

而且必须在同一 degradation severity 内比较，防止 (E) 只学“q3 比 q1 危险”。

---

## 13. 对当前代码的直接影响

### 13.1 `upstream.py`

方向基本兼容新目标。

应明确输出接口：

- anatomy class / presence / masks / embeddings；
- lesion type；
- lesion box / mask；
- lesion physical coordinates；
- lesion embeddings。

---

### 13.2 `relation.py`

不建议直接作为下一阶段 Gate 主模型继续扩写。

当前实现：

- flatten all (K\times M) pairs；
- global self-attention；
- 5D geometry；
- edge-level relation logit。

建议增加新的最小实现，例如：

`HostCompetitionHead`

逻辑：

$
(A_1,\dots,A_K,U_j)
\rightarrow
\{r_{1j},\dots,r_{Kj}\}
\rightarrow
softmax_i
$

先验证 per-lesion binding，再决定是否保留全局 RelationModule 作为消融。

---

### 13.3 `RESEARCH_PLAN.md`

建议后续正式修订时把第一原则从：

> degradation-driven relation rescue

改为：

> anatomy-centered structured MRI perception

并将 degradation / E / motion 移至 robustness / extension section。

---

## 14. 推荐的新执行顺序

1. 修复并冻结数据几何约定，包括 fastMRI+ box flip。
2. 明确 (A/U/R) 统一数据接口。
3. 先把 anatomy parsing 做可靠。
4. 把 lesion type + 3D location 做可靠。
5. 建立公平的 B0 / Bprior / Bgeo+ / Broi。
6. 实现最小 per-lesion HostCompetition relation head。
7. 在 clean / standard condition 下验证 (R) 是否有独立价值。
8. 建立 Level R human relation truth。
9. 再做 q1/q2/q3 degradation，分析 (Delta A,Delta U,Delta R)。
10. 只有 relation 在强基线和人标 truth 上成立，才增加 E / abstention。
11. motion / protocol / scanner / cross-organ 之后再扩展。

---

## 15. 推荐的第一篇论文叙事

不建议：

> MRI Foundation Model

也不建议：

> Robust Relation Transformer

建议围绕：

> **Anatomy-centered structured perception for MRI**

核心科学问题：

> Can an MRI model jointly represent anatomical entities and lesion events, and explicitly bind each lesion to its anatomical host beyond what can be explained by segmentation lookup, spatial priors, and strong geometry classifiers?

robustness 部分再问：

> How do acquisition perturbations propagate through anatomy perception, lesion perception, and lesion–anatomy binding?

这样可以把方法贡献与机制研究统一起来，而不是让 robustness 反过来定义整个模型。

---

## 16. Go / No-Go 标准

### Go：继续发展 relation

至少满足：

1. (A) 和 (U) 本身达到可用水平；
2. (B_{rel}) 在 patient-level split 上稳定优于 (B_{prior})、(B_{geo+})、(B_{roi})；
3. human Level R 上 net rescue 的 95% CI > 0；
4. 增益不能只由 lesion category / coarse location shortcut 解释。

### No-Go：停止把 relation 作为主要贡献

任一情况成立：

1. (B_{geo+}) 与 (B_{rel}) 打平；
2. (B_{roi}) 已经达到同等水平；
3. relation 仅在 pseudo-reference 上提升，人标 truth 上没有净收益；
4. clean / standard condition 下没有可重复的 binding value；
5. degradation 下的所谓 rescue 完全可由 robust perception 消除。

No-Go 不意味着项目失败；此时仍可保留：

$
X\rightarrow(A,U)
$

作为 structured anatomy + lesion perception 方向，只是不应继续包装为 relational reasoning。

---

## 17. 最终建议

从现在开始，项目所有设计决策都应先问：

> **它是否直接提高了 anatomy identification、lesion type/location identification、或者 lesion-to-anatomy binding？**

如果答案是否定的，应默认延后。

因此当前推荐的核心架构可以压缩成一句：

$
\boxed{
\text{MRI}
\rightarrow
\text{Anatomical Entities}
+
\text{Lesion Entities}
\rightarrow
\text{Explicit Anatomical Binding}
}
$

这应作为后续修改 `RESEARCH_PLAN.md`、架构图、模型接口、实验矩阵和论文叙事时的第一原则。
