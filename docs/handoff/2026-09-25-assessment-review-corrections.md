# 2026-09-25 对上一轮评估的复核与修正

> 分支：`plan/level-r-tooling-2026-09-25`
>
> 目的：复核上一轮对 AnatoBind-MRI / Foundation_Model 当前研究状态的判断，明确哪些属于仓库已有事实、哪些属于已经冻结的设计决定、哪些只是后续方法学建议。本文不修改既有 Level R 规格，只记录需要在后续研究与论文表述中遵守的修正。

## 1. 总体结论

上一轮评估的主线方向基本正确：

- Level R 独立医生真值是必要的；
- 当前脑侧第一篇论文主实验应保持为“给定病灶实例 + 预测解剖 → R”；
- 脑侧 U 不应承担第一篇论文的主线；
- FLAIR 小病灶队列不适合声称丰富的 lesion-type recognition；
- Bgeo+ / B2 / B3 / B4 的核心问题是：学习到的 relation 是否包含超越纯几何和局部图像证据的信息。

但上一轮有若干表述把“仓库事实”“冻结决定”“评估建议”混在了一起。以下 4 项必须修正。

## 2. 关键修正 1：42% 不是“真实病灶跨两个脑区”

当前 Gate 0.5 的 `d_interface=0` 比例约为 42%。这个数字来自：

- fastMRI+ 提供的逐层 bounding boxes；
- `scripts/brain_frame.py` 将每个病灶的逐层 box 转成矩形 `member_rects`；
- `anatobind/eval/geometry.py::lesion_class_distances()` 在整个矩形 box 区域内计算对各 SynthSeg host class 的最小距离；
- 当 box 同时接触至少两个 host class 时，第二近 host 的距离 `d2=0`，因此 `d_interface=0`。

因此正确表述是：

> 约 42% 的 fastMRI+ 小病灶 **bounding box** 在当前 SynthSeg host-class 几何下同时接触至少两个候选宿主类别。

不能表述为：

> 42% 的真实病灶跨越两个脑区。

fastMRI+ 没有提供这些小病灶的真实 segmentation mask，box 中包含正常组织与部分容积，因此 box-level conflict 不等价于 lesion tissue occupancy。

后续论文、图表与文档应使用：

- “bounding-box anatomical conflict”
- “zero-distance dual-host conflict under SynthSeg geometry”
- “geometry-conflict candidates”

而不应把它写成真实病灶的跨结构比例。

这一修正反而进一步支持 Level R 的必要性：Gate 0.5 是用来发现“几何上存在歧义的候选病灶”，并不能定义真实 anatomical host。

## 3. 关键修正 2：脑侧 A 尚未被独立验证

需要区分膝侧与脑侧 A。

### 膝侧

膝侧 anatomy 已经有独立验证证据：

- nnU-Net 六类 held-out Dice 约 0.84–0.89；
- 该部分可以表述为 anatomy recognition 已经达到可用水平。

### 脑侧

脑侧目前使用 SynthSeg 33 类 pseudo-label / parcellation，但没有独立 anatomy ground truth 对 5 mm FLAIR 上的分区准确性做系统验证。

因此脑侧 A 的准确状态应写为：

> Brain A is available as a working anatomical representation, but is not independently validated on the Level R FLAIR cohort.

不应直接写成“脑侧解剖识别已经实现”。

当前更严格的状态是：

- Knee A：validated；
- Brain A：available but not independently validated；
- U：当前主要系统瓶颈；
- R：尚待 Level R 独立真值检验。

## 4. 关键修正 3：B2 先行是资源建议，不是已经冻结的实验路线

上一轮建议先跑：

`B0 → Bgeo+ → B2 → 再决定 B3/B4`

这个建议的出发点是正确的：已有评审明确指出 B2 可能已经吸收绝大多数图像信息，如果 B2 与 B3/B4 打平，则 relation Transformer / candidate competition 很可能只是模型包装，而不是独立科学贡献。

但必须明确：

- 当前 v2.6 仍预注册了 B0 / Bprior / Bgeo+ / B1 / B2 / B3 / B4 / B5；
- “先看 B2 再决定是否实现 B3/B4”只是工程资源优化建议；
- 若以后正式采用 early gate，只能在 inner train/validation folds 上做决定；
- 绝不能根据 outer Level R test folds 的结果再决定是否训练/报告 B3/B4，否则会发生 test-set information leakage。

Level R 当前的 fold sealing 与 `unblind=True` 访问控制应继续保留。

推荐的科学问题仍然是：

> 在相同 lesion instance 与相同 predicted anatomy 条件下，B4 是否提供了超越强 Bgeo+ 和 B2 的独立增益？

而不是“B4 是否比一个弱 lookup baseline 更好”。

## 5. 关键修正 4：FLAIR-only 是合理建议，但不是仓库事实本身

fastMRI+ 脑标注包含 FLAIR、T1 与 T1POST；当前 Level R 选择的是 165 名患者、1297 个 FLAIR 小病灶。

因此：

> “第一篇主 Level R cohort 保持 FLAIR，T1/T1POST 后续作为 cross-contrast generalization / robustness”

是一个合理的研究设计建议，但不应被写成已经由数据本身证明的唯一方案。

当前 FLAIR-only 的优势在于：

- Gate 0.5、1297 lesion registry、165-patient folds、distance strata 都已经围绕该队列建立；
- 当前主要问题是小病灶 anatomical binding，而不是多序列疾病分类；
- 直接把 T1/T1POST 混入 primary cohort 会改变 lesion visibility、病灶分布和 acquisition geometry；
- 加入其他序列后必须重新检查 patient-level duplication 与 split independence。

因此建议：第一篇 primary Level R 仍保持 FLAIR，但把这一点明确写成设计选择，而非数据集天然要求。

## 6. 仍然成立的核心判断

以下结论经复核仍然成立。

### 6.1 Level R 必须先于关系模型结论

SynthSeg + geometry lookup 与待比较的几何基线共享信息源，因此不能作为独立 correctness truth。

Level R 的两位独立读者 + 第三位裁定、隐藏 SynthSeg / model prediction / fastMRI+ label string 的设计是合理的。

### 6.2 脑侧 U 降为次要系统任务是正确的

v2.6 已经明确：

> 第一篇主实验是“给定病灶实例 + 预测解剖 → R”。

因此第一篇论文不能再宣称脑侧 fully end-to-end A/U/R 已经完成。

U 可以作为后续 system-level extension；当前 R 机制论文不应被小病灶 detector 拖死。

### 6.3 FLAIR 1297 小病灶不能承担丰富的 lesion-type learning

当前小病灶类别高度集中于：

- nonspecific white-matter lesion；
- lacunar infarct。

因此这部分 U 更适合定义为 lesion instance / lesion query，而不是广义多病种分类。

真正的脑 U disease-type supervision 应主要由具有 3D mask 的病灶数据承担，例如肿瘤、转移、梗死等数据。

### 6.4 Gate A′ 仍是 robustness claim 的必要强基线

当前 clean Level R / Gate R1 可以先做。

但在论文声称：

- degradation-aware relation；
- relation-specific robustness；
- observability E；

之前，必须加入 noise/degradation-trained robust anatomy segmentation strong baseline。

若 robust A 本身消除了大部分 relation conflict，则不能再声称第二层 relation mechanism 提供独立 robustness。

## 7. 新发现的 ontology 风险：当前 Level R 只解决 coarse host binding

当前 `primary_host` 是：

- white_matter
- cortex
- thalamus
- basal_ganglia
- brainstem
- cerebellum
- other

而且左右侧已经合并。

这能够回答：

> lesion → white matter

但不能直接回答：

> lesion → left frontal white matter

因此必须区分两个目标：

1. coarse anatomical host binding；
2. fine-grained anatomical localization。

当前 v2.6 + Level R 的主问题属于第 1 类。

如果论文中使用“模型知道病灶具体在哪个解剖结构上”这样的强表述，需要额外的 side / lobe / named-structure localization 层，否则 ontology 粒度不足。

第一篇建议把 claim 限定为：

> clinically valid coarse anatomical host binding

而不要直接泛化成 fully fine-grained anatomical localization。

## 8. 需要补入 reader protocol 的 primary_host 操作定义

当前工具枚举了 `primary_host`，但还需要补一个临床语义定义，否则两个放射科医生可能把“host”理解成不同概念。

建议正式定义为：

> primary_host = 放射科医生结合整个可见 MRI volume 后，判断该病灶最主要、最合理的解剖宿主组织；不是 bounding box 最大重叠结构，也不是 SynthSeg 最近结构。

对于边界/partial-volume 情况：

- 选择最可能的主宿主作为 `primary_host`；
- 另一个合理宿主加入 `acceptable_hosts`；
- 用 `topography` 与 `adjacency` 单独描述 juxtacortical / periventricular / crosses-boundary 等关系。

示例：

- `primary_host = white_matter`
- `topography = juxtacortical`
- `acceptable_hosts = {white_matter, cortex}`

这样可以避免把“juxtacortical”错误地当成 anatomy host。

## 9. 当前最准确的第一篇科学问题

现阶段不应把第一篇核心问题定义为“能否完整实现 A/U/R end-to-end”。

更准确的问题是：

> Given a lesion instance and the same predicted anatomy, does a learned relation model produce clinically valid anatomical binding beyond what can be explained by geometry alone or by a local image patch alone?

对应比较：

- Bgeo+：强几何证据；
- B2：局部图像证据；
- B3/B4：显式 candidate relation / image + geometry relation。

若 B4 在独立 Level R truth 上显著优于 Bgeo+ 且优于 B2，才支持“relation modeling 包含独立信息”的主张。

若 B2 已经与 B4 持平，则应弱化 relation-Transformer 叙事。

若 Bgeo+ 已经与学习模型持平，则论文更适合转向：

- independent radiologist relation benchmark；
- geometry failure analysis；
- ambiguity / boundary stratification；
- error propagation and reliability analysis。

## 10. 后续执行顺序建议

当前建议保持：

1. 完成 Level R pilot 150；
2. 验证 reader agreement、primary_host 语义可重复性和读片时间；
3. 通过后完成独立 Level R truth；
4. 在严格封存的外层五折下比较 Bgeo+ / B2 / B3 / B4；
5. 只有当 learned relation 在 clean truth 上确有独立增益，再进入 degradation robustness / Gate A′ / observability E；
6. 后续再扩展 fine-grained anatomy、T1/T1POST cross-contrast 和 fully end-to-end U。

## 11. 一句话版本

当前最需要避免的错误表述是：

> “42% 的真实病灶跨两个脑区，因此 relation learning 必然必要。”

正确版本是：

> “42% 的 fastMRI+ 小病灶 bounding boxes 在当前 SynthSeg 几何下产生零距离双宿主冲突，这证明纯 box-level geometry 存在大量潜在歧义，但不能说明真实病灶跨结构；是否需要 learned relation，必须由独立 Level R 医生真值和 Bgeo+/B2/B4 对照实验回答。”
