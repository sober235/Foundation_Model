# 对"AnatoBind 专家评论"的可行性与正确性审计

日期:2026-09-09。审计对象:用户转来的两段专家评论(第一段写于无法访问仓库时,第二段写于进入私有仓库 `sober235/Foundation_Model` 之后)。
核对基准:本地仓库 `/data0/congcong/code/Project_Doing/foundation_model`,分支 `main`@f85ce73(2026-09-08),`RESEARCH_PLAN.md` v2.1,`docs/architecture_and_novelty_2026-09-08.md`(PDF 源文),`docs/verification/2026-09-08/REPORT.md`,以及 `anatobind/` 全部模型与数据引擎代码。测试:130 passed(本次重跑)。

---

## 0. 结论

1. 专家对仓库现状的每一条事实性断言都核实无误;七条外部引用(TotalSegmentator 脑叶仅 CT、PI-CAI 2025-07 起 1500 例全专家标注、KMAR-50K 1190 人/1444 对/62506 层、Decipher-MR 20 万序列、Sci Rep 2024 坐标位置编码、CheXtriev MICCAI 2024、Cascading Graph Alignment MedIA 2026)全部查到原文。第一段里有四条已被第二段或仓库现状覆盖(见 §1),以第二段为准。
2. 专家漏掉了一件决定 M1 成败的事,我补测了:**把病灶类别用上的查表基线(seg-then-lookup)在全部 309 个分割内实例上组织族级准确率 0.968**,不是仓库 §13.4 写的 0.860。半月板撕裂 101 例组织族级 100%(类别即组织族)。剩余可争空间约 3 个百分点,全部是软骨病变在髌股/股胫接触面的邻接混淆。**M1 现行门槛(组织族级 ABA 提升 ≥ 10 个百分点)对一个公平基线在干净数据上不可达**;按 §9.7 的功效表,3–5 个点的差需要 470–1500 例,手上只有 309。专家的 Gate 1 按其自身判据("gap ≈ 0 → 停")已经接近触发。
3. 这不是模型缺陷,是结构性事实:`located_in` 关系在有分割可用时本质是几何查表。仓库 §9.1 自己也写了"干净数据上 seg-then-lookup 几乎必然不输",但把"14% 空间"算在了一个不看类别的稻草人基线上。真正非平凡的内容只剩三块:退化后预测几何失效时的绑定鲁棒性;无宿主 mask 的关系(韧带 38 例,但它退化成细分类);关系可靠性 E。
4. 专家建议里可直接采纳的:预测几何替换真值几何、B0–B3 四模型对照、硬负样本延后、E 的失效概率定义(附三个补丁)、运动仿真延后、脑叶级本体。需要修改再采纳的:公平基线必须是类别感知 + 最近结构兜底;Gate 1 需重定义;"一周做完"是乐观估计(2–3 周)。不必采纳的:稀疏关系 token(按方案规模不是算力必需)、有界 spacing 归一(§4.1 已有)、typed relation(R_Q 已取消)。
5. 需要用户拍板三件事(§4):M1 判据改法;是否把类别感知基线与修正后的空间写进方案;是否为脑侧重跑 SynthSeg `--parc`。

---

## 1. 逐条核对

### 1.1 仓库层面的断言(第二段)

| 专家断言 | 核实 | 出处 |
|---|---|---|
| 关系真值只来自 `host_label`,不用重叠率;函数刻意不接几何 | 属实 | `losses.py::build_relation_targets`,docstring 与测试 `test_host_ce_target_uses_host_label_not_max_ioa` |
| `ArmBMinimal.forward()` 把 GT mask、GT presence、匹配后的 GT 框送进几何 | 属实 | `armb.py:55-63`,注释 `MINIMAL PATH` |
| 指标键名 `val_host_acc_NOT_EVIDENCE`,脚本自称"NOT an M1 experiment" | 属实 | `train_armb_minimal.py` docstring 与 `evaluate()` |
| G_ij = (Δz,Δy,Δx,‖Δ‖,IoA),φ 拼接 [A_i;U_j;g(G_ij)],2 层配对 Transformer,d=128,K=6,M=8 | 属实 | `relation.py` |
| 损失 = L_mask+L_presence + L_cls+5L1+2GIoU + L_hostCE+L_relBCE,无硬负样本 | 属实 | `losses.py:134` |
| 裁块 (64,128,128),stem (2,4,4),window (4,8,8),MONAI 相对位置偏置,无 M_valid、无 RoPE | 属实 | `backbone.py` docstring,`dataset.py` |
| UNKNOWN(侧别未解)≠ none,整体剔出关系损失 | 属实 | `dataset.py::UNKNOWN_HOST`,`losses.py` ignore_index |
| 噪声/欠采是 k-space 级(加噪、Poisson 欠采、adjoint SENSE),不是图像增广 | 属实 | `skmtea_recon.py` |
| 运动仿真缺失;E 头缺失;S、U_Q 缺失;脑侧未接入训练;前列腺无代码 | 属实 | `docs/architecture_and_novelty_2026-09-08.md` §1 差距表 |
| E* = exp(−NRMSE/τ_E),E = σ(h_E[r_ij; pool(F2,Ω_ij)]) | 属实 | `RESEARCH_PLAN.md` §4.7、§5.2 |
| U_Q 独立全局输出,不产生 R_Q | 属实 | `RESEARCH_PLAN.md` §4.5(1A) |
| "仓库自己的 novelty review 已指出局部保真度 ≠ 关系可判别性" | 属实 | `architecture_and_novelty` §3 风险 3;`RESEARCH_PLAN.md` §5.2 第二口径 |

补一条专家说得不够完整的:关系真值"不由重叠率生成"只对组织族成立。侧别(311 例中 144 例,46%)由 D5 规则用外扩框的重叠比解析,仓库 §13.4 已披露并因此把主判据降到组织族级。见 §2 的进一步发现。

### 1.2 外部事实(第一段)

| 断言 | 核实结果 |
|---|---|
| TotalSegmentator `total_mr` 50 类,脑只有整体 `brain`;`brain_structures`(含 frontal_lobe 等)注明 for CT | 属实(README/CHANGELOG) |
| PI-CAI:2025-07-01 起原 205 例 AI 标注改为专家标注,1500 例全专家;PI-RADS 分数已提供 | 属实(picai_labels;PI-RADS 的"2026-07"日期未能单独核到) |
| KMAR-50K:1190 人、1444 对、62506 层,1.5T/3T 两机 | 属实(Sci Data 2025-07-09) |
| MR-ART 148 健康成人三档运动 | 属实(已知) |
| Decipher-MR:npj Digital Medicine 2026,约 20 万序列/2.2 万 study | 属实(专家给的 203,233/22,594 精确数未单独核到) |
| Sci Rep 2024 坐标位置编码(Das et al.,Siemens;UNETR/SwinUNETR,MRI 梗死分割) | 属实 |
| CheXtriev MICCAI 2024 解剖中心图 Transformer | 属实 |
| Cascading Graph Alignment MedIA 2026(解剖/病理节点 + 关系边) | 属实,但它是 CXR 视觉语言预训练 |
| fastMRI 多线圈 k-space 为 (slices, coils, H, W),2D 多层采集 | 属实(本地 h5 实测 16×20×640×320) |
| SKM-TEA 155 例 qDESS、6 组织分割、16 类病理 3D 框 | 属实 |

### 1.3 第一段中已过时或被覆盖的条目

| 条目 | 现状 |
|---|---|
| 问题三:TotalSegmentator MRI 给不了 left frontal | 仓库从未用 TotalSeg 做脑;用的是 SynthSeg-robust。但实质成立:现有输出是 33 类 aseg 粒度(皮层每半球一个标签),没有脑叶。修法见 §3.9 |
| 问题六:typed relation R^p(located_in / affects_visibility) | v2.1 已取消 R_Q,只剩 located_in;专家第二段 §10 自己也建议维持取消。typed 没必要 |
| 问题五:fastMRI 不能套 3D 刚体 k-space 运动公式 | §6.1 已是多线圈图像域逐 shot 重编码、按 ISMRMRD echo train 取 shot 结构。需补写的只是"2D 多层只做面内逐层运动、层间运动不可表示"这句 |
| 问题二:不要包装成 foundation model | README 与方案 §0 已改称"结构化感知编码器" |
| 问题八:有界 spacing 归一 | §4.1 已有(SKM-TEA 0.3125→0.625 mm,极端 spacing 温和重采样) |
| 术语 equivariance → conditional invariance | v2.1 §1.2 已不称 R 的非平凡等变 |

---

## 2. 新实测:类别感知查表基线的天花板

### 2.1 为什么要测

仓库 §13.4 的"oracle 重叠绑定器"是 `argmax_i IoA(box, M_i)` 遍历全部六个结构。它不知道病灶类别。而任何真实的 seg-then-lookup 管线都有检测器输出的类别,把"半月板撕裂"绑到胫骨软骨上是不会发生的。D5 规则②又把类别与 tissue_id 不相容的 2 条剔了,所以"半月板撕裂 → 组织族=半月板"是构造上的恒等式。

### 2.2 怎么算的

脚本 `docs/verification/2026-09-09/lookup_ceiling.py`,只读 `/data2/congcong/data/FM_data/derived/skmtea/m1/`,155 卷、309 个宿主已知的分割内实例(311 减 2 个 unresolved)。真值 = `host_label`;组织族由 tissue_id 定义(与 REPORT §6 相同)。

```
V0  argmax IoA,全部存在结构                     (仓库 §13.4 的 oracle)
V1  候选限定为类别对应组织:半月板撕裂 → {5,6},软骨病变 → {1,2,3,4};
    原始框 IoA argmax;全零时取最近结构(mm)
V2  同 V1,但 IoA 在 D5 外扩框上算(面内 2、z 4 体素)
V3  只取最近结构,不看重叠
```

### 2.3 结果

| 基线 | 标签级 | 组织族级 | n |
|---|---|---|---|
| V0 全结构 argmax IoA(仓库 oracle) | 0.864 | 0.864 | 309 |
| **V1 类别感知 IoA + 最近兜底** | **0.968** | **0.968** | 309 |
| V2 类别感知外扩 IoA + 最近兜底 | 0.948 | 0.948 | 309 |
| V3 只看最近 | 0.822 | 0.822 | 309 |

按类别(组织族级):

| 类别 | n | V0 | V1 |
|---|---|---|---|
| 半月板撕裂 | 101 | 0.762 | **1.000** |
| 软骨病变 | 208 | 0.913 | 0.952 |

按折(组织族级,V1):fold0 0.965 / fold1 0.964 / fold2 0.984 / fold3 0.955 / fold4 0.970。fold 0 的 V0 = 0.860,与 REPORT §6 一致,说明我的读法与仓库相同。

V1 剩余 10 个错(V2 剩余 16 个)全是软骨病变:股骨 ↔ 胫骨 9 例、髌骨 ↔ 股骨 6 例、1 例零重叠。其中 §13.5 保留的 MTR_110 ann 15(半月板撕裂,框内只有股骨软骨,宿主在 1.25 mm 外)在 V1 下被"最近半月板"兜底答对;§13.5 的理由"重叠率必然答错"只对不看类别、不兜底的查表成立。真正查表答不了的是 MTR_020 ann 67 这类(软骨病变,框内装着标签 4 和 6,标注宿主 2)。

### 2.4 后果

- 组织族级 ABA 的可争空间从 14 个点缩到约 3 个点(oracle mask 下)。
- 这 3 个点来自 15 个左右的接触面病例,是关系建模真能贡献的内容,但按 §9.7 的公式 `n ≈ 7.84 × p_disc / d²`,d=0.03、p_disc≈0.08 需要约 700 例;d=0.05 需 470 例。309 例(有效约 215–240)检不出。
- 侧别对任何重叠类查表都是白送的(V1/V2 侧别 0 错),与 §13.4 结论一致,且比它更强:侧别不是"次要口径",是没有信息的口径。
- 预测 mask 会拉低 V1,但组织族只要粗定位对就够,nnU-Net 在 SKM-TEA 软骨/半月板上 Dice 0.8–0.9,估计干净数据上臂 A 仍在 0.93–0.96。退化条件下两臂一起掉,掉的主要是检测,不是绑定。
- 因此 M1 现行判据("退化条件下组织族级 ABA 提升 ≥ 10 个百分点")大概率给出阴性,而且阴性不说明关系建模没价值,只说明这道题在这份数据上没有 10 个点可赢。**判据必须在动手建臂 A 之前改**,否则两臂建完得到的是一个预先注定的结论。

---

## 3. 对专家各项建议的可行性判断

### 3.1 预测几何替换 GT 几何 —— 采纳

三处 teacher forcing(presence、mask、matched box)改为 `a["presence"].sigmoid()>阈`、`a["masks"].sigmoid()>0.5`、`u["boxes"]` 反归一化。1–2 天。仓库 REPORT §9 待办第 5 项已列。评估口径按 §9.5:漏检计入失败,不只在成功匹配子集上报。

### 3.2 公平 seg-then-lookup —— 采纳,但必须升级

臂 A 的查表规则写死为:类别限定候选 + IoA argmax + 零重叠取最近结构。否则就是 §2 的稻草人,审稿人一眼看穿。同时臂 A 与臂 B 训练同一套退化视图(§9.1 "两臂同监督")。工作量:nnU-Net 3d_fullres 五折(155 卷 256×256×160)约 1–2 GPU 天/折,四张空闲卡并行约 2–3 天;3D 检测(nnDetection 或简化版)再 2–3 天;评估框架 1 周。**合计 2–3 周,不是专家说的一周。**

### 3.3 Gate 1 识别–绑定 gap —— 需重定义

专家的定义(A 对、U 对、R 是否错)对有 mask 头的管线已经有答案:≈ 3%(§2)。仓库 Exp 1 的定义更严格——只对"无 mask 头的纯分类/检测 baseline"测 gap——这才是能测出正数的口径,但审稿人会答"加个分割头就好"。建议把 gap 改写成三段式报告而不是门:(a) 无 mask 头编码器的 gap;(b) 有 mask 头 + 类别感知查表的 gap(≈3%);(c) 退化后预测 mask 下的 gap。门放到 §3.6/§3.7。

### 3.4 B0–B3 四模型 —— 采纳

现有 `relation.py` 直接支持:B2 = `geo` 置零(或去掉 `g`);B1 = 用 MLP 替换 `blocks`;B3 = 现状;B0 = §3.2 的类别感知查表。加一列 B0-naive(V0)只为说明稻草人有多大。

### 3.5 硬负样本延后 —— 采纳

同意先证 Relation Transformer 有无必要,再加 L_hard。计划 A 表已 λ_h=0。

### 3.6 E 改为 P(R̂ 正确 | X) —— 方向采纳,附三个必要补丁

仓库 §5.2 已把"冻结的干净模型在 X^q 上这条关系是否仍正确"列为第二口径并用它定义失效事件;专家的建议等于把第二口径升为训练主目标、NRMSE 降为辅助回归。批评本身(局部保真度 ≠ 关系可判别性)仓库自己已承认。但专家方案没解决三个可行性问题:

1. **交叉拟合。** 失效标签 `1[R̂^q 正确]` 必须来自折外教师(out-of-fold),否则训练集上标签几乎全 1,E 学不到干净图上的失效。五折下自然可做:每折的 E 标签由该折之外训练的 R 产生。
2. **分层评估。** 失效事件集中在 q3/R16,E 若只学"退化档"就能拿高 AUROC。这正是专家自己在第四个问题里担心的,但其 Gate 4 没有按档评估。必须报每一退化档内的 AUROC/AURC,以及对照"只用退化档做预测"的基线。
3. **样本量。** 309 实例 × 6 退化条件 ≈ 1,850 个关系–条件对,失效事件估计几百个。E 头必须很轻(两层 MLP 级),且 Brier/ECE 的置信区间会很宽。

另外要清楚代价:E 一旦主要由失效标签训练,它与 ConfidNet 的差别只剩"外部物理代理 + 关系粒度 + 跨退化迁移",这正是仓库 novelty 表 N1 写的边界。可接受,但 Exp 5 的 learned-failure-prediction 对照就成了必须赢的对照。τ_E 无处定标的问题(09-07 审计②)随 NRMSE 降为辅助项而减轻。

### 3.7 Gate 3 / Gate 4 —— 采纳并改成 M1 的真正门

Gate 3(退化确实造成绑定失效):在 oracle mask 下退化不改变查表结果,失效只能来自预测 mask/框。所以 Gate 3 测的是两臂的感知鲁棒性,应按"检测漏检"和"检出后绑错"分开报。Gate 4(E 优于 softmax/熵/全局与局部 NRMSE/ConfidNet):按 §3.6 分层。这两个门可以承担 M1 的判据角色,见 §4。

### 3.8 稀疏关系 token —— 算术正确,当前不必

K=80、M=100 是专家假设的规模。方案默认 M=20,K 为膝 6 + 脑 SynthSeg 粒度约 32(加 `--parc` 约 64),配对 ≤ 1,300,注意力矩阵 ≤ 1.7M 项/层/样本,GPU 上微不足道。稀疏候选的价值在硬负样本采样,不在算力;放到 §3.5 之后。

### 3.9 脑叶级解剖真值 —— 采纳,修法便宜

现有 `derived/synthseg/*/seg_native` 是 33 类(实测 PDGM-0004:FreeSurfer 标签 2–60,皮层 3/42 各一整块)。SynthSeg 的 `--parc` 与 `--robust` 可同时用(README 命令行同列;"不适用于 --robust"的说明只针对 `--fast`),给 Desikan-Killiany 皮层分区,聚合成脑叶即可。管线 `synthseg_pipeline.py` 目前只传 `--robust`,加一个开关重跑各向同性库(PDGM 501 + BMSR 461 + HCP 1113 = 2,075 卷,单进程约 15 CPU 小时,4 进程约 4 小时)。fastMRI 5 mm 层保持 aseg 粒度。方案 §13.3 Q3 写的 ANTs atlas 配准可以不做。

### 3.10 运动仿真 —— 同意延后;方案需补三句

§6.1 路线正确。补写:fastMRI 2D 多层只做面内逐层刚体运动,层间运动不可表示;SKM-TEA 3D qDESS 的 ky–kz shot 顺序数据集未给,须声明假设;SKM-TEA 原始 k-space 采集支撑只有 38.5%(09-07 实测),X^0 是数据集 `target` 重建而退化图是 adjoint SENSE,参考不一致已在架构文档标出,做 E* 前要统一重建口径。

### 3.11 KMAR / MR-ART 只能证 sim-to-real —— 正确

与方案 §9.4 口径一致。专家建议人工标一个小规模真实运动病理子集,可行且加分,KMAR 有 PD TSE 膝、可标半月板/软骨病灶框。

### 3.12 前列腺 —— 更新一句即可

PI-CAI 2025-07 起 1500 例全专家 csPCa 标注,§7.3 可注明"二阶段外测可用 PI-CAI(lesion → prostate / PZ-TZ)"。第一篇仍不做。

### 3.13 前作与措辞 —— 采纳

Decipher-MR(npj DM 2026)、Sci Rep 2024 坐标 PE、CheXtriev、Cascading Graph Alignment 应进 §2 前作线;物理坐标 RoPE 降为工程属性(方案 §4.2 本就标"断言须消融");"Intervention-Consistent Relational Learning"作总称可用。

---

## 4. 需要用户拍板的三件事

**决定 A:M1 判据怎么改。**

- A1 保留"绑定 vs 查表",但改成退化梯度上预测几何的曲线比较 + 弃权/覆盖率,只报告不设门槛。
- A2 把门换成 Gate 3 + Gate 4:公平管线在退化下确有绑定失效(按漏检/绑错分开),且 E 在每一退化档内显著优于 softmax/熵/NRMSE/ConfidNet。**推荐 A2 作门,A1 作报告。** 这与专家第一段"novelty 收敛到 R + E + 干预"一致,也与仓库 novelty 表 N1/N2 一致;显式关系 token 保留为 E 的载体,不再作为要赢 10 个点的主角。
- A3 维持现判据。我的判断:必然阴性,且阴性不能说明问题。

**决定 B:是否授权修改方案文档。** 把类别感知查表基线写进 §9.1/§9.6,把 §13.4 的"14% 空间"改为"约 3%",§13.5 的理由改为"对类别感知 + 最近兜底的查表仍答错的实例"(MTR_110 ann 15 不再是例子,MTR_020 ann 67 是)。这是文档修改,我没有动。

**决定 C:是否为脑侧重跑 SynthSeg `--parc`。** 只在决定继续脑侧(第二战场 BMSR 多灶)时做;CPU 半天。

---

## 5. 复现

```bash
# 类别感知查表天花板(只读)
cd /data0/congcong/code/Project_Doing/foundation_model
PYTHONNOUSERSITE=1 ~/anaconda3/envs/nvgen/bin/python \
  docs/verification/2026-09-09/lookup_ceiling.py docs/verification/2026-09-09/lookup_ceiling.json \
  | tee docs/verification/2026-09-09/lookup_ceiling_output.txt

# 测试
PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python -m pytest tests/ -q -p no:cacheprovider
# 130 passed in 10.55s(2026-09-09)
```

逐实例结果:`docs/verification/2026-09-09/lookup_ceiling.json`(scan、fold、ann、类别、tissue_id、真值、V0–V3 预测、宿主 IoA、宿主距离);未加工的终端输出:`docs/verification/2026-09-09/lookup_ceiling_output.txt`(2026-09-09 重跑,与首次运行逐数相同)。

未核到的:臂 A 用预测 mask 的真实准确率(需训练 nnU-Net);Decipher-MR 的精确序列数与 PI-RADS 发布日期;PDF 文件本身未打开,以其源文 md 为准。
