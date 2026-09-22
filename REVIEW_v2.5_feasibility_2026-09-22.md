# v2.5 方案可行性评估（2026-09-22，独立评审）

评审对象：`main @ 6848e1b`（附录 [C1]）。性质：冷启动的全新评估，不继承任何对话上下文；上一轮评审与 STATUS 的结论只当线索，能重跑的都重跑了。入库前由第二人逐条复核了数字、行号引用与 [C5] 脚本；[C5] 脚本与原始输出入库于 `docs/verification/2026-09-22-feasibility-review/`。

依据文件：`CLAUDE.md`、`STATUS.md`、`docs/plans/2026-09-22-aur-v2.5-complete-technical-route.md`（下称 v2.5）、`RESEARCH_PLAN.md` §0/§1/§9.1/§9.6/§9.7/§13、`docs/verification/2026-09-13/G2_verdict.md`、`docs/verification/2026-09-15/H1_verdict.md` 与 `h1.txt`、`docs/verification/2026-09-16-brain-probe/REPORT.md` 及其 json、`REVIEW_v7_feasibility_2026-09-16.md`、`REVIEW_v7_feasibility_2026-09-19.md`、`REVIEW_core_target_AUR_2026-09-22.md`、`REVIEW_expert_comments_audit_2026-09-09.md`、`docs/data_engine_synthseg.md`、`docs/superpowers/specs/2026-09-14-leg2-fastmri-knee-detection-gate-design.md`，以及 `anatobind/model/relation.py`、`anatobind/eval/{lookup,g2,matching}.py`、`anatobind/data_engine/fastmri_knee.py`、`anatobind/train/{dataset_knee,train_detector}.py`、`anatobind/model/{dense_head_2d,detector2d}.py`、`tests/`。

本次重跑：测试套件 [C2]、代码事实 grep [C3]、fastMRI+ 脑标注 CSV 计数 [C4]、用探针同款规则重算 FLAIR 病灶并做功效/一致率/工时计算 [C5]、依赖与 GPU 现状 [C6]、检测器参数量 [C7]、SynthSeg 覆盖 [C8]、旧导出目录 [C9]。数字后的 `(v2.5:117)` 表示文件与行号；`[C5]` 表示附录命令。不是证据的数字标 `NOT_EVIDENCE`。

---

## 0. 结论

**裁决：有条件 GO。** 工程路线清楚、代码底子干净（311 个测试全过 [C2]），Gate 0 与膝侧 H1 重跑两周内能出结果。但第一篇论文的核心主张"学习到的关系模型 B4 在医生标注的真值上超过强几何基线 Bgeo+"，科学余量很薄，而且 v2.5 自己的功效估算低估了脑数据的患者聚簇效应。这个项目现在最大的风险不是做不出来，而是做出来是阴性。

理由分五点：

1. **余量薄。** 探针 24 卷里 78.7% 的小病灶查表宿主就是白质（614/780，`probe2a_displacement.json` host_distribution），这部分几何和学习模型必然同答白质，产生不了分歧。能产生分歧的只剩皮层宿主的 146 个加靠边界的白质病灶，占两成上下。两法答案不同的病灶比例 p_disc 现实上大约 0.05 到 0.12（`NOT_EVIDENCE`，推算见 §2），净救回 d 大约 0.01 到 0.04。
2. **功效被高估。** v2.5 §25 第 2 项用 SKM-TEA 沿袭下来的设计效应 1.3 到 1.5（v2.5:931）。脑侧每卷小病灶数极度偏斜：中位 3、最大 87、前 10 卷占 38.3%，聚簇尺寸 m_eff = 29.45 [C5]。哪怕患者内相关 ICC 只有 0.02，设计效应也是 1.57；ICC 0.05 就是 2.42。全部 1297 个小病灶做五折 CV，在 p_disc 0.10 下能检出的最小 d 是 0.031 到 0.038；单一 20% 留出只有约 250 个病灶，最小可检 d 是 0.056 到 0.087，等于注定不显著。
3. **Level R 是长杆。** 读片人已确认是医生、可以找到（STATUS:22），但姓名、时间、裁定人、工具全空（v2.5:709–714）。全集双阅片按每例 1.5 / 2.5 / 4 分钟算是 65 / 108 / 173 小时读片总工时，另加裁定约 16 小时 [C5]。医生业余时间投入，这是以月计的事。
4. **U 未证。** 膝侧 H1 失败被翻转框污染，修复后重跑结果未知；脑侧检测器根本还没有设计（现有检测器是膝的五类，`train_detector.py:23`）。脑小病灶 79.3% 只占一层、42% 面内不超过 6 mm [C5]，检测难度高于膝。第一篇很可能只能做"给定病灶框"的 R 机制实验，v2.5 §6.3 已为此留了口子，是对的。
5. **工程缺口具体且可控。** 框翻转修复、B3、B4、Bprior、Bgeo+、B2、Level R 统计模块、实体 schema、标注工具全部不存在 [C3]，但每一块都是几百行的小模块，PR-A 到 PR-D 合计约 8 到 12 周工程时间（§6）。

条件与两周动作见 §9。§25 五项拍板的独立建议见 §7，其中第 1 项现文的两个备选（Gwet AC1 ≥ 0.6 或 raw ≥ 0.90）在这个宿主分布下差得很远，AC1 0.6 只相当于 raw 约 0.65，比现在的 raw 0.80 还松，不应采纳。

---

## 1. Gate 链能否执行

结论：Gate 0 与 Gate A/U（膝侧）两周内可执行；Gate R0 卡在人与工具；Gate R1 卡在样本设计与余量；Robustness 与 E 在第一篇之外。逐门如下。

| 门 | 前置条件 | 仓库已有 | 缺什么 | 工作量（估） | 失败风险 | 卡住它的拍板 |
|---|---|---|---|---|---|---|
| Gate 0 | 无 | 翻转方向已被探针实测（脑 22/24、膝 30/30，REPORT:30–31）；`reconstruct_rss`、`merge_to_3d` 可用 | `convert_box_csv_to_rss` 等函数、四类测试、overlay、manifest 版本、加载器拒旧 [C3] | 1 到 2 天 | 低 | 无 |
| Gate A/U 膝 | Gate 0 | 2.5D 检测器、五折训练脚本、`check_h1.py`、五折检测缓存目录 [C9] | 修复后重训五折 | 五折各约 1 小时空卡（H1_verdict:26–30 的 0.18 s/步 × 20000 步）；今天 8 张卡全空 [C6] | 中：修复后仍可能受损失量级失衡影响（H1_verdict:66–75） | 无 |
| Gate A/U 脑 | Gate 0 | A：SynthSeg 干净图伪标签，447 个 FLAIR 卷已算好 [C8]。U：什么都没有 | 脑侧检测器（类别、数据集、负样本规则）、A 的质控口径（脑侧没有解剖真值，REVIEW 09-16:76） | U 约 2 周起 | 高（§4） | 无 |
| Gate R0 | Gate 0（框要对才能给医生看，CLAUDE.md:31）；抽样框需 SynthSeg 距离变换 | §7 协议草案 | 读片人姓名/时间、裁定人、工具、一致率统计代码、抽样权重代码、伦理备案 | 工具 3 天到 2 周；pilot 读片每人 4 到 10 小时 [C5]；统计代码 3 到 5 天 | 高：5 mm FLAIR 近皮层病灶正是人也难定的那类（§3） | §25 第 1、3、4、5 项 |
| Gate R1 | R0 过、最终队列标完、B0 到 B4 建好、几何一致 | B0 膝版查表、B1 头（未接线）、G2 的 bootstrap 骨架 | Bprior、Bgeo+、B2、B3、B4、16 维几何、`relation_metrics.py`、`level_r_stats.py` [C3] | PR-C 3 到 5 周、PR-D 3 到 4 周、统计 1 周；读片以月计 | 高：余量薄（§2） | §25 第 2、3 项 |
| Robustness | R1 GO | SKM-TEA 退化引擎、nnU-Net 管线（膝）、24 卷噪声视图探针 | 252 卷噪声视图 SynthSeg（约 2 小时 CPU，REVIEW 09-16:52）、噪声训练的脑分割器、ΔA/ΔU/ΔR 代码 | 2 到 3 周 | 中：09-19 评审认为噪声训练分割器很可能把窗口关掉（REVIEW 09-19:243–258），仍未测 | 无 |
| E | R1 GO 且有残余失败 | `reliability.py` 是膝检测可靠性头，不是 R 的 E | 全部 | 第一篇不做（v2.5:820–827） | 不适用 | 无 |

两点补充。第一，Gate A/U 在脑侧无法成为真正的门：FM_data 里没有任何脑解剖人标（REVIEW 09-16:76），A 只能用 SynthSeg 自己的一致性和抽样目检当质控，v2.5 §5 列的 Dice、HD95 在脑侧算不出来。第二，Gate 0 不需要重做 56 GB 图像导出：图像数组与框约定无关，只有 `lesions.csv` 的坐标要换。建一个新导出根（硬链接或软链接到现有各卷目录，加新的 `lesions.csv` 与带 `transform_version` 的 manifest），旧根一个字节不动，加载器按 manifest 拒旧。这满足 v2.5:182 的要求，也省下上次导出耗掉的约 15 小时（目录时间戳 09-14 11:21 到 09-15 02:39 [C9]，`NOT_EVIDENCE` 级别的推断）。

---

## 2. 主张的科学余量（最关键的一节）

结论：**在全部 1297 个小病灶上做五折 CV，勉强够检出 d ≈ 0.03 到 0.04；单一留出注定检不出。** 而现实的 d 很可能就在 0.01 到 0.04 这个区间的下半段。这是整个方案最大的科学风险。

### 2.1 几何基线的天花板可能有多高

黑话先说清：宿主，也就是病灶所在的那块脑区；查表，也就是先分区再看病灶框压在哪个区上。

探针（24 卷、780 个小病灶，REPORT:17–18）干净图查表结果：白质 614、皮层 146、CSF 19、尾状核 1（`probe2a_displacement.json` host_distribution）。白质占 78.7%，与宿主重叠比例中位 1.00。这 614 个病灶整个泡在白质里，Bgeo+ 拿到"与白质重叠 100%、离皮层 x mm"这样的特征后只会答白质，B4 也不会去反对。分歧只可能出现在其余 21.3%，加上那些查表判白质但靠近边界的病灶。靠边界的比例可以用位移实验反推：面内挪 2 mm 有 5.7% 改答，3 mm 8.2%（REPORT:43–44），也就是离边界 3 mm 以内的病灶约 8%。两块合起来，能产生分歧的病灶上限大约 25% 到 30%（推算，`NOT_EVIDENCE`；全集上的实数见 §9 动作 3）。

再想医生会怎么标。小血管病的非特异白质病灶绝大多数在临床上就是白质病灶，近皮层的多数是"近皮层白质病灶"（v2.5 §3 自己也推荐 primary_host = white matter、topography = juxtacortical，v2.5:145–150）。也就是说在皮层宿主的 146 个里，人标很可能大部分仍是白质。Bgeo+ 只要学到"离皮层近但重叠在皮层的仍答白质"就能拿到这部分分。B4 要赢，只能靠局部图像证据分辨真正累及皮层的少数病灶。5 mm 层厚上这类证据很弱。

所以我的估计（`NOT_EVIDENCE`，推算）：两法答案不同的病灶比例 p_disc 在 0.05 到 0.12；分歧里 B4 答对的比例 q 在 0.55 到 0.70。净救回：

```
d = p_disc * (2q - 1)
p_disc 0.05: q 0.60 -> d 0.010   q 0.70 -> d 0.020
p_disc 0.10: q 0.60 -> d 0.020   q 0.65 -> d 0.030   q 0.70 -> d 0.040
p_disc 0.20: q 0.60 -> d 0.040   q 0.70 -> d 0.080
```

膝侧作对照：类别感知查表天花板 0.968，余量约 3 点，311 例检不出（REVIEW 09-09:11、108）。脑侧余量比膝大，但不是大一个量级。

### 2.2 功效估算：公式、假设、结果

公式与 v2.5 §25 一致（配对 McNemar，双侧 0.05，功效 0.8），但设计效应用实测聚簇尺寸重算：

```
n_needed = (1.96 + 0.84)^2 * p_disc / d^2 = 7.84 * p_disc / d^2
DE = 1 + (m_eff - 1) * ICC          设计效应：同一患者的多个病灶不独立，样本要打折
m_eff = sum(m_i^2) / sum(m_i)        m_i = 第 i 个患者的小病灶数；实测 29.45 [C5]
n_eff = n_available / DE
d_min = sqrt(7.84 * p_disc / n_eff)  给定样本能检出的最小净救回
```

实测输入 [C5]：1297 个小病灶分布在 165 卷；每卷 min 1、q25 2、中位 3、q75 7、max 87；前 10 卷 497 个（38.3%）。ICC 未知，须在 pilot 估（v2.5:387 已列），这里给 0.02 / 0.05 / 0.10 三档。

```
ICC   0.02   0.05   0.10
DE    1.57   2.42   3.85
```

两种样本设计下的最小可检 d（p_disc = 0.10）：

```
设计                 n     DE 1.0   DE 1.57   DE 2.42
20% 单一留出（中位）  248    0.056    0.070     0.087
≥700 加权子集         700    0.033    0.042     0.052
五折 CV 全集         1297    0.025    0.031     0.038
```

20% 留出的病灶数本身就不稳：随机抽 50 名 FLAIR 患者，小病灶数 5%/50%/95% 分位是 149 / 248 / 380，其中只有约 33 名患者有小病灶 [C5]。抽到哪几个重病患者决定了样本量。

读法：v2.5:931 写"p_disc 0.10、d 0.04 需 640 到 735"，那是 DE 1.3 到 1.5 的结果；换成脑侧实测聚簇，ICC 0.05 时同一目标要 490 × 2.42 ≈ 1190 个病灶，已接近全集。若真实 d 是 0.03，全集在 ICC 0.05 下也不够（需 870 × 2.42 ≈ 2100）。

### 2.3 对两种样本设计的回答

单一留出：在任何合理假设下都检不出 d ≤ 0.05，结果必然是"CI 跨零"，论文写不出主张。五折 CV 全集：在 ICC ≤ 0.05、d ≥ 0.035 时可检出；d 在 0.03 附近时是掷硬币。因此 CV 全集是唯一有机会的设计，而且要接受"有机会"的意思是五五开。

一个能提前减少不确定性的动作：在任何人标开始前，先在全部 1297 个病灶上算清楚干净 SynthSeg 下的宿主分布、每个病灶到最近其他候选边界的距离、单层病灶比例。这一步只要 CPU 几分钟，直接给出分歧预算的上限和 H1 分层的真实大小（§9 动作 3）。探针的 79% 来自按病灶数最多挑出的 24 卷（REPORT:17），重度小血管病为主，全集上的比例未测。

---

## 3. Level R 这条外部依赖

结论：**人已确认存在，但工时被低估，协议里有四处会在真标时出问题。**

### 3.1 工时

按 v2.5 §7.4 双阅片、§7.5 十个字段。每例 1.5 分钟是 STATUS:28 的假设；一个病灶要翻上下层、选宿主、选拓扑、多选邻接、填模糊与置信，我认为 2.5 到 4 分钟更接近（`NOT_EVIDENCE`，应在 pilot 计时）。两位读者合计 [C5]：

```
队列              1.5 min   2.5 min   4 min    裁定（25% × 3 min）
pilot 150          7.5 h    12.5 h    20 h       1.9 h
留出约 260        13.0 h    21.7 h    35 h       3.2 h
700 加权子集      35.0 h    58.3 h    93 h       8.8 h
CV 全集 1297      64.8 h   108.1 h   173 h      16.2 h
```

若两位医生各投入每周 3 小时，全集在 2.5 分钟/例下每人约 54 小时，即 18 周。这决定了 Gate R1 最早在第五到第六个月出结果（§8）。

### 3.2 协议里会出问题的地方

1. **5 mm 层厚上的白质/皮层交界。** 1297 个小病灶里 79.3% 只占一层，42% 面内最大边不超过 6 mm [C5]（后一数按 0.6875 mm 像素算，低分辨率系列的 220 个病灶实际更大，`data_engine_synthseg.md:17`）。一层 5 mm 的部分容积效应下，"这个 4 mm 的亮点是在皮层下白质还是已经碰到皮层"很多时候看不出来。而这恰恰是 H1 分层，也就是 B4 唯一可能有余量的地方。要预期 H1 分层的一致率明显低于全体。
2. **模糊规则会吞掉困难病例。** §3 给了 insufficient-resolution 这一档（v2.5:143）。读者用得越多，覆盖率越低，主终点越向容易病例收缩。建议：允许 ambiguous 但必须给出 acceptable_host_set，且集合最多两个元素；不设"无法判断"这一逃生口。
3. **"这不是病灶"没有出口。** fastMRI+ 的框是单人标注、无重复标注（REVIEW 09-16:29），FLAIR 里还有 57 个"Possible artifact"（[C4]，合并后计数）。读者会遇到不认可的框，字段里要有 `not_a_lesion` 并预注册其处理。
4. **盲化与裁定的可操作性。** 读者不能看到 SynthSeg 叠图、模型输出（v2.5:295–299），这容易做到。难的是裁定：第三读者未定；若改为两人当面共识，独立性就没了。另外读片顺序按患者分组会让读者对同一患者的病灶形成惯性，建议按病灶打乱但保证同一患者的上下文可回看。工具最低功能清单（v2.5:720–733）合理，用 3D Slicer 加一个外部表单最省事，自建网页工具至少一到两周。
5. **时间戳与计时。** v2.5:733 要求 timestamp，pilot 里要顺手拿到每例用时，替换 1.5 分钟的假设。

一致率的量纲问题见 §7 第 1 项。

---

## 4. U（病灶检测）能否成立

结论：**膝侧修复后有一半以上机会过 H1；脑侧 U 成立的机会明显更低。U 不成立时，用数据集自带框做 R 机制实验站得住，但论文标题不能再叫端到端 A/U/R。**

证据。膝侧 H1 失败是任何阈值下扫描级正率恒 0.823（`h1.txt`），模型把热图整体压到最大 0.067、低于初始化 0.1（H1_verdict:47），但真值中心处仍有 2.31 倍对比（H1_verdict:55–57）。探针证明训练框是上下镜像的（REPORT:33），也就是检测器一直在学错误位置，塌陷是应有的结局。修复后：1175 卷 [C9]、16167 行标注（`knee.csv` 去表头，含 study-level 行 [C4]）、3.46 M 参数（[C7] 实测 3458195）、20000 步批 16（`train_detector.py:29–31`），这个规模对五类膝病灶不算小。剩余风险是 H1_verdict:66–75 指出的损失量级失衡（160×160 格上负样本项约为正样本项 50 倍）与 25 个 epoch 偏少，修复标签后这两条仍在。我的判断：过 H1 的机会六成上下（`NOT_EVIDENCE`）。

脑侧比膝难三件事：样本少四倍（FLAIR 框行 3941，≥3 px 过滤后 3926，对膝 16167 行 [C4][C5]）；病灶小得多（79.3% 单层、42% 不超过 6 mm）；类别塌缩，1297 个小病灶里 1240 个是同一个标签"非特异白质病灶"（[C5]，REVIEW 09-16:70 同），"病灶名称"这一项在脑上没有内容。而且脑侧检测器连设计都没有：`detector2d.py` 是膝的五类 2.5D 头，`dataset_knee.py` 绑定膝导出目录。做一个脑侧版本两周起，能否达到"可用"（例如每卷 1 个假阳性下召回 0.7）我没有依据下判断。

若 U 不成立，第一篇用参考病灶（数据集自带框）做 R 机制实验是否站得住：站得住，前提是三条。R 的定义本来就是条件于病灶实例的（v2.5:243–249 已把两级实验分开）；所有基线共用同一批参考病灶，比较公平；论文明写"给定病灶实例"，标题与贡献里不称端到端。代价是 REVIEW core 09-22 §6 的 E2（病灶感知）只能报膝侧结果，脑侧 U 作为附带报告。这在 MedIA 是可以接受的诚实写法，但会削弱"结构化感知"的整体叙事。

---

## 5. A（解剖分区）的来源是否够用

结论：**第一篇不需要先训脑侧 A 模型，SynthSeg 伪标签够用；但要把"伪标签是共模误差"这件事说清，并给 A 加一道每卷质控。**

伪标签，也就是由算法自动生成、没人核对过的标签。脑侧 A 现状：SynthSeg-robust 2.0 的 33 类 aseg 标签（REPORT:19），447 个 FLAIR 卷的干净图已算好 [C8]，覆盖全部 252 个有框卷。已知缺陷：5 mm FLAIR 上干净图就有 19/780 病灶被判给 CSF，深部灰质在 q3 噪声下 Dice 只有 0.49（REPORT:77）；良性预处理就让查表答案改 2.8% 到 6.0%（REPORT:118）。

这对 R 公平比较意味着什么。基线与学习模型共用同一份 A（v2.5:211、402–408），所以 A 的错误是共模的：它同时扭曲 Bgeo+ 的特征和 B4 的输入，不偏袒任何一方。但有两个后果。第一，A 的系统性偏差（比如 5 mm 层上皮层被分厚）会变成一个可学的规律，Bgeo+ 用距离特征、B4 用图像证据都能学到"查表说皮层其实是白质"，这部分"纠错"不算关系推理，论文要把它和真正的关系增益分开报，Bprior 与 Bgeo+ 的存在正是为此。第二，A 的随机性错误（19 个 CSF 那类）压低了所有方法的天花板，也压低了 p_disc 里有意义的部分。这两条都是要在 pilot 里量的，不是要训模型解决的。

要不要先训脑侧 A。不要。用 SynthSeg 伪标签训 nnU-Net 只是蒸馏，干净图上的准确率不会高于老师；它唯一的价值是对噪声更稳，那是 Robustness 阶段（v2.5 §13.1）的事。工作量供参考：252 卷 × 4 视图约 1000 例，2 到 3 个 GPU 天（REVIEW 09-16:52）。第一篇该做的是便宜的两件事：每卷 SynthSeg 质控（脑掩膜体积、左右体积比、有无空标签，几十行代码），以及让读者在标宿主时顺手给"病灶附近分区是否可信"打一个是/否。

---

## 6. 工程就绪度

结论：**底子干净但关系主线几乎从零开始；PR-A 到 PR-D 合计约 8 到 12 周；没有 CI 已经造成过规格与代码漂移。**

### 6.1 现在有什么、没什么

有：`anatobind/` 下 40 个 .py（含 6 个 `__init__`）、`tests/` 下 50 个 `test_*.py`、311 个测试 33 秒跑完 [C2]；SKM-TEA 全套数据引擎与 G2 评估；膝侧 2.5D 检测器与五折训练；`lookup.py` 的类别感知 B0（膝专用，候选表写死为膝的六个标签，`lookup.py:11`）；`relation.py` 的 B1 `IndependentCandidateHead`，7 个测试 [C3]；G2 的按卷 bootstrap（`g2.py:35–42`）可以改造成按患者 bootstrap。

没有（[C3] 逐项 grep 为空）：框翻转修复（`nr - y`、`convert_box_csv_to_rss`、`voxel_to_world`、`transform_version` 在 `anatobind/`、`scripts/`、`tests/` 均无匹配）；`CandidateCompetitionHead`（B3）、`BoundaryAwareBindingHead`（B4）；Bprior、Bgeo+、B2；`sklearn`、`net rescue`、`kappa`、`gwet` 在代码里零出现；`entity_manifest.py`、`relation_metrics.py`、`level_r_stats.py` 三个文件不存在。B1 只被测试引用，没有接进任何训练或评估脚本。脑侧病灶的读框、合并、翻转、查表代码只在探针脚本副本 `docs/verification/2026-09-16-brain-probe/probe_common.py` 里（文件头自称"Not part of the repo"，不在 `anatobind` 包内，也没有测试）。几何仍是 5 维（`relation.py:19`），且 IoA 用 Python 双重循环（`relation.py:45–51`）。依赖：sklearn 1.9.0 可用，xgboost、statsmodels 未装 [C6]，与 v2.5:431 一致。

### 6.2 各 PR 工作量

以下天数全部是估计（`NOT_EVIDENCE`）。依据是各模块的代码量（现有对应模块都在 40 到 220 行之间，[C3] 前的 `wc -l`）和这个仓库的实际节奏：四周 146 次提交，leg 1 从数据引擎到 G2 裁决用了约一周，leg 2 从规格到 H1 裁决用了两天 [C10]。

- PR-A Gate 0：转换函数与四类测试 1 天；新导出根（链接 + 新 `lesions.csv` + manifest）半天；脑/膝 overlay 半天。共 1 到 2 天。`test_dataset_knee.py:67` 直接喂合成病灶，不受影响；`test_fastmri_knee_boxes.py` 不涉及 y 方向，只需新增测试。
- 膝 H1 重跑：五折各约 1 小时空卡，加缓存检出与 `check_h1.py`，1 天。
- PR-B Level R：协议文档 2 到 3 天；工具用 3D Slicer 加表单 2 到 3 天，自建网页 1 到 2 周；一致率与 IPW 估计量模块（含 bootstrap CI）3 到 5 天；抽样框脚本（要对 252 卷算距离变换）2 到 3 天。共 2 到 3 周，不含读片。
- PR-C 基线：把探针的脑侧查表移入仓库并加测试 2 到 3 天；16 维几何（每卷每候选的距离变换，缓存）1 周；Bprior、Bgeo+（sklearn 的 LR、HistGradientBoosting、MLP，嵌套 CV）3 到 4 天；B1 接线需要解剖与病灶的 embedding，脑侧没有训练好的编码器，得用解剖 id 嵌入加病灶 ROI 小 CNN，1 周；B2 局部 ROI 分类器 1 周。共 3 到 5 周。
- PR-D 学习关系：B3 在约 10 到 15 个候选上做一两层自注意力，3 到 4 天；B4 的逐对局部边界证据 l_ij 是最不确定的一块，1 到 2 周；五折固定超参训练与预注册 1 周。共 3 到 4 周。
- PR-E 鲁棒性：252 卷噪声视图 SynthSeg 约 2 小时 CPU；噪声训练分割器 2 到 3 GPU 天；ΔA/ΔU/ΔR 1 周。共 2 到 3 周。

### 6.3 没有 CI 的风险

仓库没有任何 CI 配置（`.github`、`pyproject.toml`、`tox.ini` 均不存在 [C3]），测试只在有人记得时跑。已有一个漂移实例：leg 2 规格把 H1 定义为"3D 检出 ≥ 300 且逐病灶正样本率 20% 到 80%"（spec:165），而 `scripts/check_h1.py` 实现的是"训练折上扫描级正率落在 [0.2, 0.5]"，裁决按后者写。这次没出事，下次就未必。测试全是 CPU、33 秒，加一个 GitHub Actions 工作流是二十行 YAML 的事，建议 PR-A 顺手加上。

### 6.4 最容易返工的地方

1. 在 Gate 0 之前碰任何用图像的模块（B2、B4 的 ROI 裁剪）：框错，特征全错。
2. 在 §25 第 4 项（脑室是否宿主）定案前写 §16 的 schema 和候选本体：K 会变。
3. 在 §25 第 5 项定案前写抽样权重：分层定义变了，π_i 全部重算。
4. 把 `lookup.py` 的膝常量泛化到脑侧时弄坏 G2 测试：建议脑侧另建函数、共用最近兜底逻辑，而不是改膝表。

---

## 7. §25 的五项待拍板决定

### 第 1 项：一致率门槛的量纲

建议：**用 raw agreement（两位医生直接一致的比例），最终队列全体的 95% CI 下限 ≥ 0.80，分层单报 raw、κ、Gwet AC1；某分层 raw 点估计 < 0.80 则该分层降为集合值终点。不采纳 AC1 ≥ 0.6。**

理由。用探针宿主分布（白质 0.787、皮层 0.187、CSF 0.024）算 [C5]：

```
raw 0.80 -> kappa 0.42, Gwet AC1 0.77     raw 0.90 -> kappa 0.71, AC1 0.89
Gwet AC1 = 0.6 <-> raw ≈ 0.65   （p_e_gwet = 0.115）
```

v2.5:930 把"AC1 ≥ 0.6"和"raw ≥ 0.90"并列为备选，两者在这个分布下差了 25 个百分点的 raw。AC1 0.6 会把一个三分之一时间意见相左的分层认证为可标注，之后 3 到 4 个点的 d 会被约三成的标签噪声淹没。κ 在 79/19 的分布下被机会一致率 0.655 拉得很低，也不适合作门。raw 直接就是"任何模型能被验证到的准确率上限"，量纲最诚实。选错的后果：门太松，主终点建立在噪声上，B4 与 Bgeo+ 的差异被衰减到检不出；门太严（raw 0.90），近皮层分层几乎必然被降级，B4 唯一有余量的地方消失。

### 第 2 项：五折 CV 全集还是单一留出

建议：**五折患者级 CV、全部 1297 个小病灶标注；读片按折顺序推进，预注册一次 ≥ 700 例时的无效性中期分析（只许停不许宣称）。** 单一留出直接排除（§2.2：最小可检 d 0.056 到 0.087）。样本量表用 pilot 实测 ICC 与 m_eff = 29.45 重算，不再沿用 1.3 到 1.5。选错的后果：留出等于花几十小时读片换一个必然跨零的 CI；全集的代价是 65 到 173 小时读片，但这批标注本身是可复用的资产（§8 退路）。

### 第 3 项：主终点人群

建议：**同意 §25 的建议：主终点 = 全部病灶、集合值正确性（predicted_host ∈ acceptable_host_set，确定病灶的集合只有一个元素）；确定病灶子集单列；报告 coverage。** 补一条：模糊病灶的集合若是 {白质, 皮层}，Bgeo+ 和 B4 都"答对"，对净救回贡献为零，所以集合值评分对 B4 是保守的，审稿人不会认为占了便宜。只算确定病灶的后果是把困难病例剔掉，B4 可能赢的地方正好被抹掉，且 coverage 会随读者习惯漂移。

### 第 4 项：脑室是否作 primary_host

建议：**不作宿主。脑室与 CSF 只作地标（topography = periventricular、adjacency = adjacent_to_ventricle）；真正的脑室内病灶记 host = other 并计入 coverage 报告，不进宿主准确率。** 理由：探针干净图就有 19/780 病灶被查表判给 CSF（REPORT:77），那是 5 mm 层上的部分容积伪标签错误。若脑室是宿主，任何学过"CSF 附近其实是白质"的模型都能在这 2.4% 上白得分数，看起来像救回，实际是修伪标签。选错的后果：K 变大、Bprior 变强、net rescue 被伪标签修正污染。

### 第 5 项：H1 分层的定义

建议：**纯几何准则：病灶中心到最近的非宿主候选边界的距离 ≤ t mm，在干净 SynthSeg 距离变换上算，对全部病灶可算，t 在任何模型存在前冻结。** t 的取值参考位移实验：2 mm 改答 5.7%、3 mm 8.2%、5 mm 15.6%（REPORT:43–45），t = 3 mm 时 H1 约占两成上下（`NOT_EVIDENCE`，应在全集上实算，§9 动作 3）。用"方法不一致"定义分层的后果：抽样概率与结局相关，IPW 失效；而且 B4 存在前算不出来，pilot 无法启动。

---

## 8. 第一篇论文的可行版本

结论：**三个月能走到 Gate R0 出结果与 PR-C 建完；六个月能否拿到 Gate R1 答案取决于两位医生从第二个月起每周各投入 3 小时以上。**

对照 v2.5 §20 的 14 条最小要求（v2.5:804–818）。以下时间线是估计（`NOT_EVIDENCE`），工程部分按 §6.2、读片部分按 §3.1 的每周 3 小时假设。

三个月（到 2026-12 下旬）：第 1 条 Gate 0 完成；第 2 条 A/U 可用只在膝侧成立，脑侧 A 为伪标签加质控、U 为初版检测器附带报告；第 3、4 条 Level R 的 pilot 完成、协议冻结、最终 N 定下，全集读片进行中；第 5 到 8 条 population 统计代码、Bprior、Bgeo+、B2 在伪标签与 pilot 标注上建好并冻结；第 12 条膝对照直接引用 G2；第 14 条伪参照与人标分开的口径写入协议。做不到：第 9 到 11 条（B3/B4 消融、全病灶净救回、患者级 CI），因为最终队列没标完。

六个月（到 2027-03 下旬）：若读片在第五个月末完成，B3/B4 在此之前已按预注册冻结，则第 9 到 11 条可以出结果，Gate R1 有答案。第 13 条脑边界子组同期出。Robustness 与 E 不在六个月内。工程时间线不是瓶颈，读片是。

Gate R1 No-Go 时的退路论文：题目大意是"5 mm 临床 FLAIR 小病灶的独立病灶–解剖关系真值：几何绑定的准确率、失败模式与误差传播"。内容是 1297 个病灶的双阅片 Level R 标注（可发布）、B0/Bprior/Bgeo+/B2 在人标上的准确率与失败分类（近皮层、单层、低分辨率系列）、膝侧对照、噪声下的 ΔA/ΔU/ΔR。值不值得投 MedIA：值得做，因为标注是资产、结论是负也是有信息量的；但作为方法论文它没有方法贡献，MedIA 的把握不大，更合适的是 Scientific Data（标注发布）加一篇 MELBA 或 MICCAI 数据/基准类的分析。建议现在就把这条退路写进预注册，让读片投入在两种结局下都不白费。

---

## 9. 最终裁决、三大风险、两周动作

### 9.1 裁决：有条件 GO

条件：

1. §25 第 2 项定为五折 CV 全集，功效表用 m_eff = 29.45 与 pilot 实测 ICC 重算，写进 v2.5。
2. Gate 0 与膝侧 H1 重跑两周内完成并落盘；脑侧读框、合并、翻转、查表代码从探针移入仓库并加测试。
3. 读片人、裁定人、工具在 PR-C 开工前定名；pilot 六周内完成，并按分层报告 raw 一致率与 H1 分层占比。
4. 预注册文档在任何人标与模型输出对接前冻结：Bgeo+ 特征清单、B4 结构与超参、主终点（全部病灶、集合值）、分层定义（几何 t）、中期无效性规则。
5. 关系主线的停止规则写死：pilot 里 H1 分层 raw 一致率 < 0.75，或 H1 分层占全体 < 15%，或 ≥ 700 例中期分析净救回 CI 上限 < 0.02，则停 PR-D，第一篇改为 §8 的退路版本。

### 9.2 前三大风险

1. **余量薄导致阴性。** 78.7% 病灶两法必然同答；现实 d 约 0.01 到 0.04；全集 CV 在 ICC 0.05 下只能检到 0.038。这是最可能的结局。
2. **Level R 交付。** 65 到 173 小时读片、人未定名、工具不存在、近皮层分层一致率大概率偏低。任何一环拖延，六个月内没有 R1。
3. **脑侧 U 不成立。** 样本比膝少四倍、79% 单层、类别塌缩；论文退为"给定病灶实例"的 R 研究，叙事减弱。

### 9.3 接下来两周最有价值的动作

第一周：

1. PR-A（1 到 2 天）。在 `anatobind/data_engine/fastmri_knee.py` 加 `convert_box_csv_to_rss(y_csv, h, nr) -> (y0, y1)`，公式 `[nr - y - h, nr - y)`，`merge_to_3d` 前调用；测试：合成已知框、往返误差 0、体素 × 间距物理坐标、manifest 含 `box_coordinate_convention = "rss_row_from_top"` 与 `transform_version`；`load_fold` 与 `SlabDataset` 读 manifest，缺键或旧版本抛异常。新导出根 `derived/fastmri_knee/leg2_rss/`（各卷目录用链接指向旧根，新 `lesions.csv`、`folds.json` 复制、新 manifest），旧根不动。overlay 各抽 5 卷脑、5 卷膝到 `~/figs/anatobind_gate0/`。验证：
   ```
   PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
   ```
2. 膝 H1 重跑（1 天，今天 8 张卡全空 [C6]）：
   ```
   for k in 0 1 2 3 4; do
     CUDA_VISIBLE_DEVICES=$k setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. \
       ~/anaconda3/envs/nvgen/bin/python scripts/train_detector.py --fold $k \
       --export-root /data2/congcong/data/FM_data/derived/fastmri_knee/leg2_rss \
       --out runs/detector_rss_fold$k > logs/detector_rss_fold$k.log 2>&1 &
   done
   ```
   之后 `scripts/cache_detections.py` 与 `scripts/check_h1.py`（参数看各自 `--help`，导出根同上），裁决写 `docs/verification/<日期>/H1_rerun.md`，把 spec:165 与 `check_h1.py` 的 H1 定义差异一并修正。
3. 脑侧抽样框（2 到 3 天，CPU 几分钟）。把 `probe_common.py` 的 `read_boxes`、`merge_lesions`、`region_mask`、`Lookup` 移入仓库（查表用 33 类无类别限定，与膝表分开），对全部 1297 个小病灶在干净 SynthSeg 上算：宿主分布、到最近其他候选边界的距离（给出 t = 2/3/5 mm 下的 H1 占比）、单层比例、每患者病灶数、200/201 与低分辨率系列分开。落盘 `docs/verification/<日期>/brain_frame.md`。这一步直接给出 §2 的分歧预算上限，决定要不要继续投入读片。

第二周：

4. §25 五项一次拍完（本文 §7 是建议），改 v2.5 正文。
5. 协议 v1 与预注册骨架：`docs/level_r/protocol_v1.md`（§7.5 字段 + `not_a_lesion` + 集合最多两元素 + 计时），`docs/level_r/preregistration.md`（9.1 第 4 条的清单，先留空位）。
6. 工具试做：用 3D Slicer 装 10 个病灶的演示，请一位读者读一遍并计时，把每例分钟数换成实测。
7. 顺手加 GitHub Actions 跑 CPU 测试。

---

## 10. 既有评审的吸收情况

已被 v2.5 吸收：Gate 0 前置与 H1 重跑（09-16 §1、09-19 §11.3）；热图头替代 DETR（09-16 §2.2，v2.5:219）；不再把 SynthSeg+ 当更强基线（09-16 §2.1，v2.5 全文未再提）；Bprior 与 Bgeo+ 两个危险基线（09-19 §6，v2.5 §10）；逐病灶候选竞争替代全局 K×M（09-19 §5、09-22 §3，v2.5 B3）；Level R 前置、伪参照不定义正确性（09-16 §8、09-19 §3，v2.5 §7 与 §16 的 C1/C2/R）；SKM-TEA 不再承担干净优越性主结论（09-09 §2，v2.5 §15）；净救回在全部病灶上算（09-16 §3b，v2.5 §12.1）；A/U/R 重新定位（09-22 全文，v2.5 §0）。

仍未解决：Gate A′（噪声训练的分割器能否直接消掉冲突，09-19 §8）被 v2.5 移到 R1 之后的 Robustness（v2.5:568–574），与重新定位一致，但意味着第二层机制主张至今没有一个正面证据；B2 与 B3 大概率打平的预期（09-16 §3b:47）v2.5 只写成 No-Go 条件，时间线仍先建 B3/B4 再知道；T1/T1POST 224 卷为何不用（09-16 §2.4）v2.5 §15 未写理由；低分辨率 FLAIR 系列 123 卷（09-16 §4:68，[C4] 实测 44+34+4+8+11+22 = 123）在 v2.5 里没有出现，其上的 220 个小病灶 [C5] 探针从未测过；"病灶名称"在脑上没有可训内容（09-16 §4:70）v2.5 §6 仍把 lesion_type 列为 U 的输出而未注明 FLAIR 本体塌缩；脑侧 U 没有任何设计。

---

## 11. 仓库文档自相矛盾之处

1. STATUS:28 与 v2.5:931："两名读者各读约 2600 例（每例 1.5 分钟约 32 小时）"。1297 × 2 = 2594 是两人合计；每人约 1300 例，1300 × 1.5 分钟才是约 32 小时。"各约 2600"与"约 32 小时"在同一句里互相矛盾。
2. v2.5:930 把"Gwet AC1 ≥ 0.6"与"raw ≥ 0.90"并列为等价备选。按探针宿主分布，AC1 0.6 对应 raw 约 0.65，比现文的 raw 0.80 还松 [C5]。
3. leg 2 规格 spec:165 的 H1（3D 检出 ≥ 300 且逐病灶正率 20% 到 80%）与 `scripts/check_h1.py` 实现（训练折扫描级正率落在 [0.2, 0.5]）不一致；H1_verdict 按代码写。
4. STATUS:23 把"约 79% 小病灶完全在白质内"写成全集事实；它来自按病灶数最多挑出的 24 卷（REPORT:17），且 REPORT:51 的 79% 是查表宿主为白质的比例（614/780），"完全落在"只有中位重叠 1.00 支持。
5. v2.5:931 的设计效应 1.3 到 1.5 沿自 SKM-TEA 每例约 3 个病灶（RESEARCH_PLAN:600）；脑侧实测 m_eff = 29.45，不适用。这不是文字矛盾，是未经检查的沿用。
6. RESEARCH_PLAN:571 的 Bgeo+ 仍写"tree/XGBoost 类"，v2.5:431 已改为不新增 XGBoost 依赖且 xgboost 未安装 [C6]；以 v2.5 为准即可，顺手改掉。
7. v2.5:117 把 ventricle 列为 primary_host，§25 第 4 项自己标为未决，V7 与 v2.4 定的是地标。已在 §25 登记，此处只作提醒。

---

## 附录：重跑命令与原始输出

[C1] HEAD 确认：
```
git log --oneline -3
6848e1b Plan: apply the six mechanical PR #6 review comments and list the five open decisions in v2.5
4f86316 Status 2026-09-22: v2.5 route merged, readers confirmed, Gate 0 is next
779123f Merge pull request #6 from sober235/plan/aur-v2.5-complete-route-2026-09-22
```

[C2] 测试套件（nice 19）：
```
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
311 passed in 33.39s
```

[C3] 代码事实 grep（均在仓库根执行）：
```
grep -rn -i 'nr *- *y\|flipud\|flip_y\|convert_box_csv_to_rss\|voxel_to_world\|world_to_voxel\|box_coordinate_convention\|transform_version' anatobind/ scripts/ tests/
  -> 仅 anatobind/data_engine/seg_frames.py:3（SKM-TEA 分割坐标系的 flip_y，与 fastMRI+ 无关）
grep -rn 'IndependentCandidateHead' --include='*.py' .
  -> tests/test_independent_candidate.py:5,24；anatobind/model/relation.py:5,55（无训练/评估脚本引用）
grep -rn 'CandidateCompetitionHead\|BoundaryAwareBindingHead' --include='*.py' .   -> 无匹配
grep -rn 'HostCompetitionHead' --include='*.py' .                                  -> 无匹配
ls anatobind/data_engine/entity_manifest.py anatobind/eval/relation_metrics.py anatobind/eval/level_r_stats.py -> 三个文件均不存在
grep -rn -i 'sklearn\|HistGradientBoosting\|xgboost\|bprior\|bgeo\|net_rescue\|net rescue\|gwet\|kappa' --include='*.py' anatobind/ scripts/ tests/ -> 无匹配
grep -c 'def test_' tests/test_independent_candidate.py -> 7
ls -a .github .gitlab-ci.yml .travis.yml tox.ini setup.cfg pyproject.toml -> 均不存在
```

[C4] 标注 CSV（`/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/Annotations/`，只读）：
```
wc -l brain.csv knee.csv -> 8214 / 16168（含表头）
brain.csv：total rows 8213；study_level Yes 643；框行 7570；有框卷 476 = AXFLAIR 252 + AXT1 136 + AXT1POST 88
FLAIR 框行 3941；FLAIR 系列：200:67 201:62 202:44 203:34 205:4 206:8 209:11 210:22
```

[C5] `docs/verification/2026-09-22-feasibility-review/flair_counts.py`（探针同款规则：去 study_level、宽高 ≥ 3 px、同卷同标签相邻层 IoU ≥ 0.3 合并；完整原始输出见同目录 `flair_counts_output.txt`，2026-09-22 16:01 在 6848e1b 重跑，与下列数字逐项一致）关键输出：
```
FLAIR merged 3D lesions: 1825；Nonspecific white matter lesion 1240，Lacunar infarct 57，Possible artifact 57
small lesions (NSWML + lacunar): 1297 in 165 volumes；FLAIR volumes with any box: 252
per volume: min 1 q25 2 median 3 q75 7 max 87; top-10 volumes hold 497 (38.3%)
mean 7.86; m_eff = sum(m^2)/sum(m) = 29.45
single-slice small lesions: 1029 (79.3%); max in-plane extent <= 6 mm (0.6875 mm/px): 545 (42.0%)
small lesions in series 200/201: 1077; in low-res variants: 220
20% hold-out = 50 patients: small lesions p5/p50/p95 = 149/248/380; patients with >=1: 28/33/38
DE(ICC 0.02/0.05/0.10/0.20) = 1.57/2.42/3.85/6.69
d_min @ p_disc 0.10: n 260 DE1 0.055, DE2 0.078; n 700 DE1 0.033, DE1.5 0.041; n 1297 DE1 0.025, DE1.5 0.030, DE2 0.035
host prevalence: WM 0.787 cortex 0.187 CSF 0.024 caudate 0.001
raw 0.80: kappa 0.42, AC1 0.77; raw 0.90: kappa 0.71, AC1 0.89 (p_e kappa 0.655, p_e Gwet 0.115)
reader hours (2 readers): pilot150 7.5/12.5/20 h; ~260 13/21.7/34.7 h; 700 35/58.3/93.3 h; 1297 64.8/108.1/172.9 h @1.5/2.5/4 min
```

[C6] 依赖与算力（2026-09-22 15:41）：
```
sklearn 1.9.0; scipy 1.17.1; torch 2.5.1+cu121; monai 1.5.2; xgboost / statsmodels / lightgbm NOT AVAILABLE
nvidia-smi：GPU 0–7 各 14 MiB / 81920 MiB，利用率 0%
```

[C7] 检测器参数量：`Detector2D(**FULL)` -> 3458195（H1_verdict:83 写 3.46 M，一致）。

[C8] SynthSeg 覆盖：`derived/synthseg/fastmri_brain/seg_native` 996 个条目，其中 447 个 AXFLAIR（`data_engine_synthseg.md:33` 记 996/997）。

[C9] 旧导出 `derived/fastmri_knee/leg2`：1176 个条目（1175 卷目录 + `detections/`），`folds.json`、`lesions.csv`、`manifest.csv` 均无 `transform_version`；`detections/fold0..4` 存在，fold0 有 1603 个文件。56 GB 取自 v2.5:182，未重新测量。

[C10] 提交节奏：
```
git log --oneline | wc -l -> 146；首末提交 2026-08-25 / 2026-09-22
按日提交数（节选）：09-07 15、09-08 12、09-11 23、09-12 14、09-13 6、09-14 23、09-15 2、09-22 22
tag：stage/2026-09-07-skmtea-m1-data-engine -> stage/2026-09-13-leg1-skmtea-g2-fail（leg 1 约一周）；
     leg 2 规格 docs/superpowers/specs/2026-09-14-… -> stage/2026-09-15-leg2-fastmri-h1-fail（两天）
```
