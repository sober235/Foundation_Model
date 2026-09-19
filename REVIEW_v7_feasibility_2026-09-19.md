# V7 可行性复审与执行收敛（2026-09-19）

对象：当前 `main`，重点依据：
- `docs/verification/2026-09-16-brain-probe/REPORT.md`
- `REVIEW_v7_feasibility_2026-09-16.md`
- 提交 `4a7a612`、`2bd5147`、`1a16aca`
- 当前 `RESEARCH_PLAN.md`、`anatobind/model/relation.py`、`anatobind/model/upstream.py`、`anatobind/data_engine/fastmri_knee.py`

## 0. 最新结论

V7 的核心方向值得继续，但应进一步收缩为一个明确、可证伪的问题：

> 在 MRI 采集退化尚未导致全局解剖感知崩溃、但已使局部几何边界发生偏移时，显式 lesion–anatomy relation modeling 能否比几何查表、解剖先验、强几何分类器和局部 ROI 分类器更稳定地恢复病灶宿主，并在证据不足时拒绝输出。

目前已有证据支持“这个 failure regime 真实存在”，但尚无证据支持“Relation Transformer 已经有优越性”。因此不应继续把主要资源投入完整 AnatoBind 系统，而应先完成两个生死实验：Gate A′（更稳健 perception 是否直接消灭冲突）与 Gate B（relation 是否真的优于简单替代）。

如果 Gate A′ 失败，即 noise-trained segmentation 将冲突率压回 reference instability floor，关系主线应停止；如果 Gate B 不能稳定超过 prior / strong geometry / direct ROI baseline，也应停止把 relation 作为主要贡献。

---

## 1. 当前最重要的正面证据

24 卷 fastMRI+ FLAIR、780 个小病灶的脑侧 probe 已经观察到一个非崩溃的几何失效窗口：

| 条件 | lesion-host 改答率 | 95% CI | WM Dice | Cortex Dice |
|---|---:|---:|---:|---:|
| noise q1 | 3.1% | [1.6, 4.7]% | 0.972 | 0.971 |
| noise q2 | 4.6% | [2.9, 6.6]% | 0.952 | 0.949 |
| noise q3 | **7.4%** | **[5.5, 9.5]%** | **0.911** | **0.903** |

q3 的关键意义不是“图像很差”，而是：

```
global anatomy still usable
+
local anatomy boundary drifts
→
geometry lookup changes the lesion host
```

改答主要发生在近皮层病灶的同侧 white matter ↔ cortex 竞争，而不是整体 background collapse。因此这一现象可以被解释为 representation corruption，而不是 information destruction。

这正是 relation rescue 可能有价值的工作区间。

---

## 2. 简单去噪没有关掉窗口

09-16 Stage 1.5 的免训练预处理实验否定了一个原本非常危险的反证。

q3 原始改答率为 7.4%。做预处理后：

- 噪声底校正：q3 约 6.9%
- 噪声底校正 + NLM：q3 约 9.1%

尽管图像 NRMSE 明显下降，host assignment 的不稳定性并没有下降。也就是说：

```
image-level restoration improvement
!=
anatomical binding stability
```

因此“先去噪再分割就足够”目前没有得到支持。

但这里只排除了 training-free preprocessing，尚未排除 noise-trained segmentation。

---

## 3. 当前最大的科学漏洞：clean reference 自己会移动

更重要的新发现是：当前 pseudo-reference 不稳定。

良性预处理本身就会让 clean 图的 host assignment 改变：

- 仅噪声底校正：约 2.8%
- 噪声底校正 + NLM：约 6.0%

这个 3–6% 的 reference instability 与 q3 想证明的 7.4% 效应处在同一量级。

因此：

```
return to clean SynthSeg assignment
!=
return to clinical truth
```

后续不能再把“恢复 clean lookup”直接解释为“更正确”。否则可能出现 clean pseudo-label 本身错误，而模型只是把 degraded view 拉回错误参考的情况。

所以 Level R 放射科医生 relation reference 必须提前到 Gate B，并与 relation model 并行建立，而不是等到后期再补。

建议监督分成三层：

1. **C1：raw clean pseudo-reference**
2. **C2：reference stability**：在若干不会改变生物学内容的良性 processing 下测 host 一致性，得到每个 lesion 的 reference stability
3. **R：radiologist relation reference**：最终 superiority 只在人标 relation truth 上成立

pseudo-reference 可以用于训练，但不能继续承担最终 correctness 的定义。

---

## 4. 数据规模：够训练小 relation head，不够从零训练完整系统

当前 FM_data 实测：

- fastMRI+ 脑有框卷：476
- FLAIR：252
- FLAIR 合并后 3D 病灶：1825
- 目标小病灶：1297，分布在 165 卷
- 按 q3 7.4% 外推，geometry-conflict lesion 约 96 个

约 `1297 × 4` 个 clean/q1/q2/q3 lesion-view pair 足以训练一个轻量 relation classifier / candidate competition head。

但这个规模不足以支持从零联合训练：

```
3D Swin
+ anatomy decoder
+ lesion detector
+ relation transformer
+ reliability
```

因此正式 Gate B 应冻结 perception，先回答 relation 本身是否必要。

---

## 5. Relation 模型不应先做大，先证明它不是 shortcut

primary host 候选预计只有约 10 个。问题本质上是：

```
one lesion
+
~10 host candidates
→
one host distribution
```

这不需要大型 Transformer。

主模型可先采用非常小的 per-lesion candidate competition：

```
r_ij = Fuse(
  anatomy identity,
  lesion embedding,
  physical geometry,
  local image / boundary evidence
)

P(host_i | lesion_j) = softmax_i h(r_ij)
```

如果 1–2 层 attention 或 candidate MLP 已够，就不应为了方法外观继续放大全局 K×M Transformer。

---

## 6. Gate B 必须加入两个原方案缺失的危险 baseline

### 6.1 B_prior：宿主先验

probe 显示约 79% 小病灶在 clean 时完全落在 white matter 内。

因此一个很简单的：

```
P(host | lesion type, side, coarse location)
```

就可能“救回”大量 WM → cortex conflict。

如果 relation model 只是在学习“这类 lesion 通常在 WM”，那不属于 relation reasoning。

必须加入：

- majority host
- lesion/type/location-conditioned prior

### 6.2 B_geo+：强几何分类器

当前 B0 的 overlap / nearest-structure lookup 太弱。

需要把下列几何证据全部交给一个简单模型：

- signed distance to candidate surface
- distance to cortex
- distance to ventricle
- Δx / Δy / Δz in mm
- IoA / soft overlap
- lesion size
- spacing / slice thickness

然后使用 logistic regression、XGBoost 或 2-layer MLP。

如果 B_geo+ 与 relation model 打平，则 relation 模型只是复杂地实现了一个距离分类器。

---

## 7. 建议固定的 Gate B 方法矩阵

后续不要只比较 B0–B4，建议固定为：

| 方法 | 作用 |
|---|---|
| B0 | degraded overlap / nearest lookup |
| Bprior | majority / lesion-location prior |
| Bgeo+ | strong hand-designed geometry classifier |
| B1 | independent candidate MLP |
| B2 | local ROI → host classifier |
| B3 | per-lesion Host-Competition relation model |
| B4 | B3 + local boundary evidence |
| B5 | clean/oracle geometry，仅作为上限 |

所有方法必须共享完全相同的 lesion boxes、anatomy outputs、patient split 和 candidate ontology。

主指标：

```
rescue = B0 wrong and model correct
harm   = B0 correct and model wrong
net rescue = rescue - harm
```

主分析放 geometry-conflict subset，但 net rescue 必须在全部 lesion 上统计，patient-level bootstrap。

建议 Gate B 至少满足：

```
95% CI(net rescue) > 0
B4 > Bprior
B4 > Bgeo+
B4 > B2
```

否则不能主张 relation module 的独立必要性。

---

## 8. 当前真正的两个生死实验

### Gate A′：noise-trained robust segmentation

training-free denoising 已经回答，不需要重复。

下一步应使用 clean pseudo masks，训练见过 q1/q2/q3 的 nnU-Net 或同等级轻量 robust segmenter，然后重新测 q2/q3 host conflict。

如果 q3 conflict：

```
drops below the preregistered effect threshold
and approaches the reference-instability floor
```

则说明更稳健 perception 已经解决主要问题，应停止 relation 主线。

如果 q3 仍稳定超过门槛，则 relation line 才有充分必要性。

### Gate B：relation 是否真的不是 shortcut

只有在 Gate A′ 后继续。

最终 superiority 必须在人标 Level R 上判断，不再只以 clean SynthSeg pseudo-reference 为终点。

---

## 9. 论文新颖性应重新定义

不要把 novelty 写成：

- Relation Transformer
- anatomy-aware lesion modeling
- physics-aware robustness
- calibration / abstention
- denoising + downstream task

这些方向都有大量近邻工作。

当前更可信的新颖性是完整问题链：

```
controlled MRI acquisition corruption
→
anatomical geometry failure
→
lesion–anatomy binding failure
→
relation-level rescue
→
selective abstention
```

即：保持患者和病灶 biology 不变，只改变 acquisition evidence，显式测量 lesion–anatomy relation 在什么条件下失效、是否可以恢复、什么时候应拒绝输出。

这比“提出一个新的 Transformer”更有科学价值，也更符合 MedIA 的论证方式。

---

## 10. 第一篇论文应主动收缩主张

brain FLAIR 当前最适合回答：

```
small brain lesion anatomical binding
```

不适合支持：

```
general lesion understanding
```

因为 FLAIR lesion ontology 极不均衡，大量病例集中在 non-specific white matter lesions。

因此第一篇不要把“病灶名称识别”写成主要贡献，也不要继续强调 foundation model。

更合理的主问题：

> Acquisition degradation can corrupt lesion-to-anatomy binding before global anatomical perception collapses. Can evidence-aware relational modeling recover these host-assignment failures without harming stable cases?

---

## 11. 当前代码 / 文档存在三个必须同步的问题

### 11.1 `RESEARCH_PLAN.md` 已落后于当前决策

当前仍保留 DETR/Hungarian U_B 描述，但仓库真实上游已经使用 dense centre-heatmap head，且此前 synthetic 定位实验已经否定 DETR 方案。

正式进入下一阶段前，应把权威方案更新到当前真实设计。

### 11.2 `relation.py` 仍是旧的全局 K×M RelationModule

当前实现仍然：

- flatten 全部 K×M pair
- 全局 pair attention
- 5D geometry：Δz/Δy/Δx/||d||/IoA
- 保留 edge-level `h_R`

尚未实现 V7 建议的：

- per-lesion host competition
- landmark / signed-distance geometry
- local boundary evidence
- event-level host reliability

不要直接把旧 RelationModule 当作 Gate B 主模型。

### 11.3 fastMRI+ box flip 的正式 data-engine 修复尚未进入主线代码

`anatobind/data_engine/fastmri_knee.py` 当前仍直接使用 CSV 的 y 坐标，没有显式执行 DICOM labeling 所需的 up/down inverse transform。

脑侧 probe 已经证明这一问题会污染 detector 训练，所以 Gate 0 科学结论已经成立，但工程修复还需要正式落地并加单元测试。

此前 H1 detector collapse 在修复 box direction 前不能作为最终 detector 结论。

---

## 12. 现阶段应冻结的内容

在 Gate A′ / Gate B 之前，不建议继续投入：

- motion
- aggressive undersampling
- U_Q
- scanner / vendor shift
- cross-organ generalization
- full SSL / foundation-model route
- global K×M relation transformer
- complex edge-level E_ij
- end-to-end joint retraining

这些模块会增加工程复杂度，但不会回答当前最核心的科学问题。

---

## 13. 推荐的最短执行顺序

1. 正式修复 fastMRI+ box flip，加入 overlay/unit test。
2. 更新权威方案文档，使其与 heatmap detector、脑侧 V7 结论一致。
3. 完成 Gate A′：noise-trained robust segmentation。
4. 同步准备 Level R：优先标 geometry-conflict lesions + 2–3 倍 matched non-conflict controls，建议双阅片 / adjudication。
5. 只有 Gate A′ 仍成立，才实现轻量 B0/Bprior/Bgeo+/B1/B2/B3/B4。
6. 只有 Gate B 在 Level R 上有正 net rescue，才做 event-level reliability。
7. proper undersampling / motion / protocol shift 全部放到上述链条成立之后。

---

## 14. 最终裁决

当前项目不是“完整 AnatoBind 已被证明可行”，而是：

> **已经发现一个真实、可测量、值得继续验证的 brain small-lesion binding failure regime。**

当前支持继续的证据：

1. q3 geometry conflict 达 7.4%，且大结构 segmentation 尚未整体崩溃；
2. 改答随 q1→q2→q3 单调增加；
3. training-free denoising 没有消除这一窗口。

当前仍决定生死的缺口：

1. noise-trained robust segmentation 是否直接消灭 conflict；
2. relation 是否真正优于 prior / strong geometry / direct ROI classifier；
3. 在 radiologist relation truth 上是否真的提高 correctness，而不是只拟合不稳定的 clean SynthSeg reference。

只有这三点同时成立，才建议继续扩展为完整 AnatoBind；否则应按停止规则收缩或终止 relation 主线。

现阶段最值得做的不是更大的 Relation Transformer，而是把 Gate A′ 和 Gate B 做成足够强、足够公平、足够难被简单方法替代的实验。
