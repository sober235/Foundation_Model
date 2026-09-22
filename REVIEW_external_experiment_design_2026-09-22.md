# 外部 review：v2.5 可行性评审之后的实验设计意见（2026-09-22）及其核验

来源：用户 2026-09-22 粘贴给本仓库的外部 review 原文。评审对象是分支 `summary/2026-09-22-v2.5-feasibility-review` 上的 `REVIEW_v2.5_feasibility_2026-09-22.md`、`STATUS.md`、`docs/verification/2026-09-22-feasibility-review/flair_counts.py` 与 v2.5 方案。作者未署名。原文含 LaTeX 记号，原样保留作审计记录。

第二部分是逐条核验（2026-09-22，脚本与原始输出在 `docs/verification/2026-09-22-external-review-check/`），第三部分是取舍与在 v2.6 中的落地位置（`docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md`）。

---

## 第一部分：原文

我把 `summary/2026-09-22-v2.5-feasibility-review` 里的 `REVIEW_v2.5_feasibility_2026-09-22.md`、`STATUS.md`、新重算脚本 `flair_counts.py` 和 v2.5 当前方案重新串起来看了一遍。

这份 review 的质量很高，而且它指出了一个比前几轮更本质的问题：

$$
\boxed{\text{当前最大风险已经不是"能不能实现"，而是"真实增益是否足够大到能被统计学检测出来"}}
$$

我赞同它的"**有条件 GO**"，但有几处我不会原样采纳。尤其是"必须标完整 1297 个病灶""raw agreement 一个指标就能作为 R0 门""H1 用到非宿主边界距离定义"这三点，还可以进一步改进。

## 一、先说总体判断

现在这条路线已经不是架构设计阶段了，而进入真正的实验设计阶段。

已有证据大致变成：

$$
\text{A/U/R concept}
\quad \checkmark
$$

$$
\text{brain geometry ambiguity exists}
\quad \checkmark
$$

$$
\text{independent human R truth}
\quad ?
$$

$$
B_4>B_{\mathrm{geo+}}
\quad ?
$$

最后这两个问号决定论文。

这份新 review 最有价值的发现是脑侧的 cluster structure：

$$
1297\ \text{lesions}
$$

实际上集中在：

$$
165\ \text{FLAIR volumes}
$$

并且：

$$
m_{\mathrm{eff}}
=
\frac{\sum_i m_i^2}{\sum_i m_i}
=
29.45.
$$

所以原来从膝侧继承的：

$$
DE\approx1.3\sim1.5
$$

确实过于乐观。

如果：

$$
ICC=0.05,
$$

则：

$$
DE
=
1+(29.45-1)\times0.05
\approx2.42.
$$

因此"1297 个 lesion 看起来很多"是一个错觉。

真正有效信息量可能只有：

$$
\frac{1297}{2.42}
\approx536.
$$

这点我认为评审抓得非常准。

不过这个设计效应公式仍然只是**近似功效计算**。正式 sample size 不应该最终靠它决定，而应该在 pilot 后直接以 patient cluster 为单位做 simulation-based power：

$$
Z_j=
\begin{cases}
+1,&B_4\ correct,\ B_{\mathrm{geo+}}\ wrong\\
-1,&B_4\ wrong,\ B_{\mathrm{geo+}}\ correct\\
0,&otherwise
\end{cases}
$$

然后对真实 patient lesion count distribution、真实 \(p_{\mathrm{disc}}\)、真实 ICC 和 H1/H2 比例模拟：

$$
\bar Z = NetRescue
$$

的 patient-bootstrap CI。

也就是说：

> 当前 DE=2.42 非常适合作为"警报"，但不应该成为最终 power calculator。

---

# 二、§25 五个问题，我建议这样正式拍板

| 问题               | 新 review 建议                       | 我的建议                                |
| ---------------- | --------------------------------- | ----------------------------------- |
| Reader agreement | overall raw CI lower ≥0.80        | **基本同意，但不能只看 overall raw**          |
| CV vs holdout    | 1297 全标 + patient 5-fold CV       | **排除单一 20% holdout，但暂不承诺必须全标 1297** |
| 主 endpoint       | all lesions + acceptable host set | **同意**                              |
| ventricle        | 不作为 primary host                  | **同意**                              |
| H1 定义            | 到最近非宿主结构边界 ≤t                     | **需要修改定义**                          |

下面分别解释。

---

## 三、一致率门槛：raw agreement 比 AC1≥0.6 合理，但 raw 也不能单独做 Gate

这份 review 对这一点的数学判断是正确的。

当前类别高度不平衡：

$$
P(WM)\approx0.79,
$$

因此 Cohen's \(\kappa\) 会受到 prevalence 的明显影响。医学 observer studies 中确实存在高 observed agreement、但 \(\kappa\) 偏低的经典现象；Gwet AC1 可以作为补充，但它与 \(\kappa\) 不是简单可以互换的量。([PubMed Central (PMC)][1])

所以我支持：

$$
\text{raw agreement}
$$

作为最直观的门。

但我不建议只写：

$$
CI_{lower}(raw)>0.80
$$

就算 R0 通过。

因为存在一个明显漏洞：

如果两个 reader 都倾向于标：

$$
WM,
$$

即使 cortex 判断很差，overall raw 仍然可能非常漂亮。

而 B4 真正可能赢的恰好是：

$$
WM\leftrightarrow cortex
$$

这些 minority / H1 cases。

所以 Gate R0 应该变成"两层一致性"：

$$
\boxed{
\text{Global agreement}
+
\text{H1 / minority-class agreement}
}
$$

我会建议预注册：

$$
CI_{lower}(Raw_{all})\ge0.80
$$

同时必须报告：

$$
Raw_{H1},
$$

以及 WM、cortex 的 class-specific positive agreement / confusion matrix。

Gwet AC1 和 \(\kappa\) 都报告，但都不单独当生死门。

这比单独用 raw、\(\kappa\) 或 AC1 都稳。

---

# 四、1297 全标 + 5-fold CV：方向对，但我不会现在就拍板"必须全标"

review 说：

> 单一 20% holdout 基本注定检不出 3–4 个点。

我同意。

例如：

$$
n\approx250
$$

且：

$$
DE\approx2,
$$

有效样本只有约：

$$
125,
$$

想检测：

$$
d=0.03\sim0.04
$$

基本不现实。

所以：

$$
\boxed{\text{不建议单一 20\% holdout}}
$$

这一点可以直接定。

但是 review 进一步跳到：

$$
\boxed{\text{必须标完 1297}}
$$

我认为还太早。

因为它的 700-lesion 计算是假定：

> 简单/random subset。

而我们实际上可以做：

$$
\textbf{two-phase stratified sampling}.
$$

例如假设全集：

$$
20\%\ H1
+
80\%\ H2.
$$

不需要按这个比例标。

完全可以：

$$
260\ H1
+
300\ H2
$$

总共约：

$$
560
$$

个病灶。

然后使用：

$$
\hat\theta
=
P(H1)\hat\theta_{H1}
+
P(H2)\hat\theta_{H2}.
$$

或者 IPW：

$$
w_i=\frac{1}{\pi_i}.
$$

聚簇数据下的 two-phase / IPW 本身有成熟的统计处理方式，关键是 inference 必须同时考虑 sampling design 和 cluster correlation。([PubMed][2])

因此我现在更推荐：

> **pilot 之后同时模拟两个设计，而不是今天直接决定标 1297。**

方案 A：

$$
1297
$$

全标。

方案 B：

$$
\text{all/most H1}
+
\text{probability sample H2}
$$

大约：

$$
500\sim800
$$

个。

使用 pilot 实测：

$$
P(H1),
\quad
p_{\mathrm{disc},H1},
\quad
p_{\mathrm{disc},H2},
\quad
ICC,
\quad
d_{H1},
\quad
d_{H2}
$$

跑 Monte Carlo power。

哪一个达到：

$$
80\%\ power
$$

且医生工时最低，就选哪个。

**这一步可能直接省掉几十小时甚至上百小时的读片。**

---

# 五、如果采用五折 CV，必须升级成 grouped nested CV

review 写"患者级五折 CV"，这是对的，但还不够。

因为 Bgeo+、B2、B3、B4 都存在：

* architecture selection；
* feature selection；
* hyperparameter；
* threshold；
* local ROI size；
* attention depth；
* training epoch；

如果你看五折结果以后再调这些东西，再重新跑五折：

实际上就已经：

$$
\text{tuned on test folds}.
$$

医疗 AI 中这是典型 optimistic bias 来源；更规范的是 outer CV 负责性能估计、inner CV 负责 preprocessing / hyperparameter / model selection。([PubMed Central (PMC)][3])

所以最终应该是：

$$
\boxed{
\text{Outer 5-fold patient CV}
}
$$

每个 outer fold：

$$
80\%\ train
\rightarrow
\text{inner patient CV}
\rightarrow
\text{model selection}
$$

然后：

$$
20\%\ outer test
$$

只使用一次。

最终收集所有：

$$
\text{out-of-fold predictions}
$$

计算：

$$
NetRescue.
$$

所有：

* normalization；
* feature scaling；
* HGB parameters；
* B3 depth；
* B4 ROI size；
* threshold selection；

都必须在 outer training 内完成。

这是我认为新 review 里**没有写够清楚的一点**。

---

# 六、还要加一个硬规则：fold 必须按真正 patient ID，而不是 file name

新脚本里现在写的是：

```python
all_flair_patients = flair_files
```

并注释：

> one FLAIR volume == one patient.

旧 review 的确有这一结论，所以当前计算不是凭空来的。

但是正式实验不要靠注释保证这一点。

应该在生成 fold manifest 时做：

$$
file\rightarrow patient\_id
$$

显式 mapping，并 assert：

$$
\text{intersection}
(
Patient_{train},
Patient_{test}
)
=
\varnothing.
$$

如果确实：

$$
1\ volume=1\ patient,
$$

那这个 assertion 应该自然通过。

这只是一个很小的工程改动，但能避免未来加入 T1/T1POST 后立刻出现 patient leakage。

---

# 七、主终点 = all lesions + acceptable_host_set：我完全赞成

这一点新 review 比最早版本明显更成熟。

对于：

$$
Y_j=\{WM,Cortex\},
$$

如果：

$$
\hat y_j\in Y_j,
$$

就算正确。

这样不会强迫 5 mm FLAIR 给出不存在的亚体素"精确真值"。

而且正如 review 所说：

如果：

$$
Y_j=\{WM,Cortex\},
$$

Bgeo+ 和 B4 都答其中之一，

二者均正确，

所以：

$$
NetRescue=0.
$$

不会人为帮助 B4。

我建议再加一个报告量：

$$
SingletonRate
=
\frac{N(|Y|=1)}{N_{all}}.
$$

也就是实际有多少病灶能形成单一 host truth。

这比笼统叫 coverage 更清楚。

---

# 八、ventricle / CSF 不作为 primary host：赞成

对于目前研究对象：

* nonspecific WM lesion；
* lacunar lesion；

脑室和 CSF 更应该是：

$$
\text{landmark}
$$

而不是：

$$
\text{host tissue}.
$$

例如：

$$
primary\_host=WM
$$

$$
topography=periventricular
$$

$$
adjacent\_to=ventricle.
$$

这比：

$$
host=ventricle
$$

在医学概念上干净很多。

真正 intraventricular lesion：

$$
host=other/out\_of\_scope
$$

即可。

这一项我建议直接拍板，不需要继续拖。

---

# 九、我不完全同意当前 H1 定义

review 建议：

> lesion center 到最近的"非宿主候选边界" ≤ t mm。

问题是：

$$
\text{non-host}
$$

本身就需要先知道 host。

如果这个 host 又来自：

$$
SynthSeg+B0,
$$

那 H1 sampling definition 中仍然偷偷用了 pseudo-host。

更干净的定义应该完全不依赖"谁是 host"。

我推荐：

$$
\boxed{
H1(t):
d(
lesion,
nearest\ anatomical\ interface
)
\le t
}
$$

即：

> 病灶 centroid 或 lesion surface 距任意两个候选 anatomy 之间的界面 ≤ t mm。

例如 WM–cortex interface：

$$
d_j^{interface}
=
\min_{x\in lesion_j}
d(x,\partial(WM,Cortex)).
$$

然后：

$$
H1=
\mathbb{1}
[
d_j^{interface}\le3\text{ mm}
].
$$

完全不需要：

* model prediction；
* Bgeo+；
* B4；
* pseudo host。

这是更纯粹的 pre-model geometry definition。

我还建议增加一个连续指标：

$$
Margin_j
=
Score_{geo}^{(1)}
-
Score_{geo}^{(2)}
$$

或者距离版：

$$
\Delta d_j
=
d_{2nd}-d_{1st}.
$$

不要只靠二元 H1/H2。

后面可以直接画：

$$
NetRescue
\quad vs\quad
GeometryMargin.
$$

这个结果可能比"困难组提高几个百分点"更有论文价值。

---

# 十、先算完整 1297 个 lesion 的 geometry frame：我非常赞成，而且应该成为下一步第一优先级之一

现在：

$$
78.7\% WM
$$

这个数字来自：

$$
24
$$

个按照 lesion count 挑出来的 volume。

它不是 population estimate。

所以在任何医生读片前，我赞成 review 建议立即对全部 1297 个 lesion 计算：

$$
P(host),
$$

$$
d_{interface},
$$

$$
\Delta d,
$$

$$
P(H1(t=2)),
P(H1(t=3)),
P(H1(t=5)),
$$

$$
lesions/patient,
$$

以及 acquisition-series 分布。

这是整个项目当前**信息增益最高、成本最低**的一步。

因为如果最后发现：

$$
P(H1_{3mm})=6\%,
$$

relation 主线几乎可以直接降级。

如果：

$$
P(H1_{3mm})=25\%,
$$

那非常值得继续。

所以我甚至会把它放到：

$$
\boxed{\text{Gate 0.5}}
$$

---

# 十一、200/201 与 low-resolution variants 必须分层

新 review 又发现：

$$
1077
$$

个 small lesions 来自 200/201，

另有：

$$
220
$$

来自 low-resolution variants。

而 probe 实际主要建立在 200/201。

所以不建议把：

$$
1297
$$

直接当完全同质 population。

我建议在完整 frame audit 中同时算：

$$
H1_{200/201}
$$

和：

$$
H1_{low-res}.
$$

Level R pilot 也必须包含两个 strata。

如果 low-res：

$$
RawAgreement
$$

明显更低，

第一篇 primary population 可以定义为：

$$
200/201
$$

而 low-res 作为：

$$
secondary robustness cohort.
$$

虽然样本会从：

$$
1297\rightarrow1077,
$$

但 label quality 和 scientific interpretability 可能反而明显上升。

---

# 十二、我会降低 brain U 在第一篇里的优先级

新 review 对 brain detector 的担忧是正确的。

脑侧：

$$
1240/1297
$$

基本都是同一个：

> nonspecific white matter lesion。

所以"lesion type classification"在这个 first-paper dataset 里并不是真正有内容的任务。

再加上：

$$
79.3\%
$$

single-slice，

$$
42\%
$$

≤6 mm，

强行把：

$$
X\rightarrow U
$$

做成一个完整 detector，很可能消耗很多时间，却与 R 的核心科学问题关系很弱。

所以我建议第一篇正式拆成：

### Primary scientific experiment

$$
\boxed{
\text{given lesion instance}
+
\text{predicted anatomy}
\rightarrow
R
}
$$

### Secondary system experiment

$$
X
\rightarrow
U
\rightarrow
R.
$$

也就是说：

**不要让 brain detector 成为 relation paper 的前置生死门。**

长期系统当然仍然必须做 U。

但第一篇先证明：

$$
R
$$

到底有没有存在价值。

否则如果半年以后发现：

$$
B_4=B_{\mathrm{geo+}},
$$

前面花几个月把 detector 做得很漂亮也救不了 relation 主张。

---

# 十三、SynthSeg 作为 A 的问题，新 review 还有一句话我想修正

review 说：

> 基线与模型共用同一个 A，所以 A error 是共模，不偏袒任何一方。

严格来说不完全成立。

因为：

$$
B_{\mathrm{geo+}}
$$

只能看到 SynthSeg 产生的 geometry。

但：

$$
B_4
$$

还能看到：

$$
raw\ MRI/local\ boundary\ evidence.
$$

所以当 SynthSeg 错的时候，B4 有机会从图像中把它纠回来，而 Bgeo+ 没有。

如果 B4 因此赢了，这仍然是真实有价值的增益，但科学解释应该是：

> **local image evidence corrects anatomy-parsing / geometric errors**

而不一定是：

> relational reasoning.

所以我建议在 Level R protocol 增加一个很简单的字段：

$$
A\_local\_quality
=
\{acceptable,\ suspicious\}.
$$

reader 不需要重画 anatomy mask，只回答：

> "病灶附近的 anatomy boundary 看起来可靠吗？"

最后分别算：

$$
NetRescue_{A-good}
$$

和：

$$
NetRescue_{A-bad}.
$$

如果 B4 只在：

$$
A-bad
$$

上赢，

论文的机制解释就应该变成：

> boundary-aware correction。

如果在：

$$
A-good
$$

上仍然赢，

才是更强的 beyond-geometry evidence。

这会显著提升论文说服力。

---

# 十四、700 例 futility analysis：可以做，但必须更严格

review 提议：

$$
N\ge700
$$

时，如果：

$$
CI_{upper}(NetRescue)<0.02,
$$

停止 PR-D。

方向合理。

但前提必须是：

$$
B_4
$$

已经完全冻结。

不能：

> 看 700 例 → 发现差一点 → 改 B4 → 再看剩余 597。

否则 final inference 已经被 adaptive model development 污染。

我建议：

$$
\boxed{
\text{freeze model before interim}
}
$$

并把它定义成：

> **non-binding futility analysis**

只允许：

$$
stop
$$

不允许：

$$
declare\ success.
$$

如果想继续优化模型：

那么 interim cohort 必须退回 development set，后面需要新的独立 evaluation cohort。

---

# 十五、还有一个比 review 更重要的建议：annotation 和 model development 要解耦

我建议：

$$
\text{Level R annotations}
$$

完成后先封存：

$$
Outer-fold\ labels.
$$

模型开发者只拿：

$$
outer training fold
$$

的标签。

outer test：

$$
R
$$

标签由 evaluation script 在最终评估时读取。

否则在研究过程中不断看：

> 哪些病例 B4 错了、哪些 Bgeo+ 错了，

很容易产生人工 overfitting。

医疗 imaging 的 CV 指南也特别强调：一旦 test-fold 信息影响了模型选择，CV 会变得乐观；模型选择与最终性能估计应严格分离。([PubMed Central (PMC)][4])

---

# 十六、我现在建议的最终技术顺序

结合这份 review，我会把目前路线进一步压缩成下面 **10 步**：

1. **Gate 0**：修 box flip + manifest + tests + CI。
2. **Gate 0.5**：全集 1297 lesion geometry-frame audit。
3. 冻结 candidate anatomy ontology：ventricle/CSF 不作 host。
4. 冻结 H1：model-independent anatomy-interface distance。
5. 设计 Level R protocol：`primary_host + acceptable_host_set + topography + adjacency + not_a_lesion + A_local_quality`。
6. 100–150 lesion 双阅片 pilot，得到真实 agreement、ICC、p_disc、reading time。
7. 用 pilot 做 **simulation-based sample design**，比较 full-1297 vs stratified 500–800 的成本/功效。
8. 冻结 grouped nested CV + B0/Bprior/Bgeo+/B2 + B3/B4 specification。
9. 做 Level R 主实验。
10. **只有 Gate R1 为 GO**，再做 robustness / detector / E 的大规模扩展。

---

## 最终对这份新 review 的评价

我会给它这样的判断：

**正确的核心结论：**

$$
\boxed{\text{Conditional GO}}
$$

是对的。

"阴性结果风险高"也是对的。

"20% 单一 holdout 不值得做"我认为基本可以正式拍板。

"先算全集 geometry frame 再投入医生工时"非常重要，应该立刻执行。

但我会修改三个地方：

$$
\boxed{
1297\ full\ annotation
\text{ 不是现在就必须拍板}
}
$$

应该先与 stratified two-phase design 做 pilot-based power simulation。

$$
\boxed{
RawAgreement
\text{ 不能成为唯一 R0 gate}
}
$$

需要全体 + H1 + minority-class agreement。

以及：

$$
\boxed{
H1
\text{ 应定义为 model-independent anatomical-interface ambiguity}
}
$$

而不是"到非宿主候选边界"。

如果这三处修掉，我认为当前研究方案已经基本到达**可以冻结预注册并真正开实验**的程度。

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10820331/?utm_source=chatgpt.com "Quantifying Interrater Agreement and Reliability Between Thoracic Pathologists: Paradoxical Behavior of Cohen's Kappa in the Presence of a High Prevalence of the Histopathologic Feature in Lung Cancer - PMC"
[2]: https://pubmed.ncbi.nlm.nih.gov/31359448/?utm_source=chatgpt.com "On the analysis of two-phase designs in cluster-correlated data settings."
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11041453/?utm_source=chatgpt.com "Practical Considerations and Applied Examples of Cross-Validation for Model Development and Evaluation in Health Care: Tutorial - PMC"
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10388213/?utm_source=chatgpt.com "A Guide to Cross-Validation for Artificial Intelligence in Medical Imaging - PMC"

---

## 第二部分：核验（2026-09-22）

核验方式：重算算术；用真实的 165 卷病灶数分布做患者聚簇模拟；读 252 卷 FLAIR h5 的属性；在一卷真实 SynthSeg 分区上比较两种困难组定义；抓取四篇引文。脚本与原始输出：`docs/verification/2026-09-22-external-review-check/{power_sim.py, patient_id_check.py, interface_check.py}` 及对应 `*_output.txt`。

### 结论

数学与统计判断基本正确，四篇引文都存在且内容相符，所有提议在本仓库与数据上都做得到。三处需要修正或降温，最重要的一处是"分层抽样能省几十到上百小时读片"这个预期不成立。

### 逐条

| 原文 | 正确性 | 可行性 | 证据 |
|---|---|---|---|
| m_eff 29.45、ICC 0.05 时 DE 2.42、有效样本约 536 | 对 | 无关 | `power_sim_output.txt` [1]：29.45 / 2.42 / 535 |
| DE 公式只当警报，最终样本量按患者聚簇模拟 | 对 | 20 秒一轮 | 同上 [3]；公式法的"最小可检 d 约 0.03"与模拟一致 |
| 只看全体 raw 有漏洞 | 对，且更严重 | 无关 | 同上 [2]：白质全一致、皮层一半一致时，全体 raw 0.893、κ 0.62、AC1 0.88 全部过线；皮层 positive agreement 0.67，困难组内 raw 0.52。三种全局指标全部失守 |
| 排除单一 20% 留出 | 对 | 无关 | 与冷启动评审一致 |
| 分层两阶段抽样（约 560 例）可能省大量工时 | 方向可试，预期过高 | 做得到 | 见"三处修正"第一条 |
| 五折 CV 升级为嵌套 CV | 对，措辞要分级 | 做得到 | 引文 [3] 同时说样本大、超参搜索小的场景收益只有 1–2 个点；引文 [4] 主张"先选定算法再做 CV"。两篇都支持"不在测试折调参、按患者划分" |
| 折按真实患者 ID 分并断言 | 对 | 立刻能做 | `patient_id_check_output.txt`：252 卷有框 FLAIR 全带 `patient_id`，对应 252 个患者；165 卷小病灶卷对应 165 个患者；训练/验证集无重叠患者 |
| 主终点全部病灶 + 可接受集合；加 SingletonRate | 对 | 无关 | SingletonRate = v2.5 §7.8 的 coverage，同一个量 |
| 脑室、CSF 只作地标 | 对 | 无关 | 与 V7、v2.4 一致 |
| 困难组改用交界面距离 | 对，补三个细节 | 252 卷约 15 CPU 分钟 | `interface_check_output.txt`：两种定义差值 100% 在一个体素对角线之内（median 1.4 mm，max 5.09 mm），是同一个量 |
| 全集几何盘点提为 Gate 0.5 | 对 | 单线程 15 分钟 | 同上 |
| 200/201 与低分辨率分层 | 对 | 无关 | 1077 / 220 已核（`docs/verification/2026-09-22-feasibility-review/flair_counts_output.txt`） |
| 脑侧检测降为次要实验 | 对 | 无关 | v2.5 §6.3 已留口子 |
| A 误差共模说法不严格；加 A_local_quality | 对 | 字段零成本 | 无需核验 |
| 700 例中期分析前冻结模型、只许停 | 对 | 无关 | 标准做法 |
| 引文 [1] | 存在，相符 | | JTO Clin Res Rep 2023：88% 一致但 κ 0.43，AC1 0.85 |
| 引文 [2] | 存在 | | Rivera-Rodriguez，Stat Med 2019 |
| 引文 [3] | 存在，相符 | | JMIR AI 2023（原文未写年份） |
| 引文 [4] | 存在，相符 | | Radiology: AI 2023 |

### 三处修正

1. **分层抽样省不了那么多。** 主终点是全部病灶的净差距，简单组 H2 占约八成权重，它的均值再"容易"也得估准。解析计算（无聚簇）：H1 全标 + H2 抽 300 的标准误是全标的 1.38 倍，抽 600 是 1.12 倍；Neyman 最优分配应把约六成名额给 H2，与"简单组少标"相反。模拟（ICC 约 0.02，80% 功效对应的最小可检 d）：全标 1297 约 0.03（108 h）；H1 全标 + H2 600 约 0.035（72 h）；H1 全标 + H2 300 约 0.04（47 h）；随机 700 约 0.045（58 h）。每小时读片买到的信息量四种设计相近。简单病灶恰是标得最快、最少需要裁定的，砍掉它们省下的时间最少。"试标后模拟两个设计再选"的流程可以采纳，但工时要按分层实测读片时间算，且不应预期省上百小时。真正能省工时的只有把主终点改成困难组子组，这与原文第七节自己的主张矛盾，不采用。
2. **嵌套 CV 要按方法分级。** 对 B0、Bprior 这类没有超参的方法，外层五折加固定配置就够；对 Bgeo+ 的树模型和 B2、B3、B4，必须内层选参。写成规则而不是口号。
3. **交界面距离要写清三件事。** 距离从病灶表面算而不是从中心算，单层小病灶的中心在层内，表面定义才有意义。层厚 5 mm 意味着相邻层的结构中心至少 5 mm 远，t 取 2 到 4 mm 时困难组实际上是面内准则，跨层邻接进不来。阈值 t 不能凭感觉定：那一卷里 89% 的白质体素离某个交界面不到 3 mm，若病灶像白质体素那样分布，困难组就不是两成而是大半，"困难组"就失去意义。t 必须在 Gate 0.5 看到全集分布后再定，并在任何模型存在前冻结。原文"6% 就降级、25% 就继续"两个数只能当示例。

### 模拟的一处自我修正

第一版模拟让同一个患者随机效应同时抬高分歧率和模型胜率，两者同向变化使实际效应量随聚簇增大而膨胀，功效反而随 ICC 上升。已改为分歧率与胜率各自独立的 Beta 异质性，保证边际效应量不变；入库的 `power_sim.py` 与输出是修正后的版本。pilot 后应用实测的患者异质性替换这个假设。

---

## 第三部分：取舍与在 v2.6 中的落地

| 原文节 | 取舍 | v2.6 位置 |
|---|---|---|
| 一、模拟功效替代 DE 公式 | 采纳 | §9 |
| 三、两层一致率门 | 采纳，补困难组 raw ≥ 0.70 的下限 | §7.7、§2 Gate R0 |
| 四、排除留出；两设计试标后择一 | 采纳，预期改写为"每小时信息量相近" | §8.4、§9 |
| 五、嵌套 CV | 采纳，按方法分级 | §12.6 |
| 六、患者 ID 折断言 | 采纳 | §4.4 |
| 七、集合值终点 + SingletonRate | 采纳，与 coverage 统一命名 | §7.8、§12.1–12.2 |
| 八、脑室不作宿主 | 采纳 | §3 |
| 九、交界面距离 + Δd | 采纳，补表面、面内、t 后定三个细节 | §4.5、§7.3、§12.9 |
| 十、Gate 0.5 | 采纳，加 15% 占比的 GO 规则与 t 冻结规则 | §2、§4.5、§21 |
| 十一、200/201 与低分辨率分层 | 采纳 | §7.3、§15 |
| 十二、脑侧 U 降为次要 | 采纳 | §2、§6.3、§20 |
| 十三、A_local_quality 与 A-good / A-bad 分报 | 采纳 | §7.5、§12.9 |
| 十四、中期分析规则 | 采纳 | §12.8 |
| 十五、标签封存 | 采纳 | §12.7、§17 |
| 十六、十步顺序 | 采纳 | §19、§23 |

用户同日追加的三项模型决定（B4 两种证据都用并由初值起步、训练标签先用伪标签、B2 与 B4 绑定）记录在 v2.6 §10.1 与 §25。
