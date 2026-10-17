# 冻结 query-CC 解码对照：完整 val 结果（2026-10-17）

**裁决：`query_cc_v1` 在预先固定的工作点下 FAIL，不升为默认解码器。** 完整 405 行 val 已覆盖；三源 U 虽增加命中，但全部超出 S7 误报预算，灵敏度也全部低于 S7−0.05；controlled R 仍低于 B0。A32 解剖掩膜、统一异常 U 实例掩膜、每实例 13 个分侧主宿主 + no-host（R14）的三个目标均不变。

**所有结果均为 NOT_EVIDENCE。** A/R 使用伪参照，未获得人标 Level R 或临床验证；BMSR、ISLES 的 AUR 重采样网格与 S7 native 网格仅近似对应。没有新训练、N7、test 推理或 UI 工作。[冻结实施合同](../../../superpowers/specs/2026-10-17-aur-query-cc-decoder-design.md)；[运行命令、哈希与检查](RUN.json)。

## 比较合同与完整性

- 推理代码 `1874af2`，checkpoint 与数据不变；唯一实验变量是 U 实例解码。presence >0.5、mask ≥0.3、26CC、10 mm³ 下限、mask-NMS IoU ≥0.5、ownership 与无回填规则均冻结。
- BMSR/ISLES/PDGM 数值 score 固定为 **0.70/0.60/0.75**。query 子实例继承原 parent 组件均值，legacy 使用融合概率图组件均值，二者校准不同；这是相同预定数值阈值比较，**不是相同实测 FP 比较**。未扫描或重选阈值；失败只否定本冻结配置，不能外推为所有 query 保留解码均无效。[配对汇总](full_val_paired.json)
- legacy 预测直接复用缓存，仅用 `fixed_legacy_e2e` 在固定 score 过滤后重新进行官方一对一 bbox IoU ≥0.1 匹配；两条保留 binding 新获得 GT 关联，历史端到端匹配数 38→40。A、U 与 controlled R 预测及其汇总不变。[归一化审计](legacy_matching_normalization.json)
- 最终成员为 **202+102+101=405** 个唯一行，来自 1/2、0/4、2/4 三个完成分片。原 0/2 未完成分片排除；其中已验证的 68 条原子记录复用到新分片，保留原始输入哈希、行时间与显存记录。此次仅调整调度，没有修改模型、规则或分数。[重分片记录](scheduling/repartition.json)

## A 与 controlled R：保持原输出，不等于模型问题已解决

405/405 行保存的 A 数值记录完全相同；394 个 A-supervised 行的 13-host macro Dice 为 **0.803457**，entity macro 为 **0.651301**。其余 11 行为 QC 未通过的参照，仅报告、不判门。val 的 A 数值超过 0.80 是原模型的既有结果；历史 **test A=0.7988 仍未过门**，本轮没有新的 test 结果。[精确配对检查](full_val_paired.json)；[A 汇总](query_aggregate_anonymous.json)；[此前状态](STATUS_before_query_cc.md)

716/716 个 controlled R 实例的共同保存字段完全相同，实例集合无差异；ABA-R 为 **0.568436**，低于 B0 的 **0.698324**，G3 仍失败。精确检查排除随 U 改变的 `matched_det/matched_score`；legacy 未保存 query/window origin，且未保存整卷 voxel maps，因此不能声称这些未记录对象逐位相同。[配对检查](full_val_paired.json)；[controlled R 汇总](query_aggregate_anonymous.json)

## U：增加覆盖，同时显著增加误报

下表只使用各源注册的 S7 对照序列；“最低灵敏度”是对应 S7 灵敏度减 0.05。GT 数为 AUR 非 ignored 实例数；FP/scan 按官方匹配计算，ignored-GT 上被豁免的预测不计 FP。[全部原始汇总字段](full_val_paired.json)

| 来源/序列 | 扫描/GT | 命中 legacy→query | query 灵敏度 | FP/scan legacy→query | S7 FP/scan 预算 | 最低灵敏度 | query 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| BMSR/T1c | 41/470 | 2→78 | 0.16596 | 0.04878→17.00000 | 0.87805 | 0.64612 | 双项失败 |
| ISLES/DWI | 20/138 | 4→60 | 0.43478 | 1.20000→540.10000 | 1.60000 | 0.54429 | 双项失败 |
| PDGM/FLAIR | 41/53 | 27→45 | 0.84906 | 0.14634→4.73171 | 0.36585 | 0.85566 | 双项失败 |

按原等效直径分层，表内为“legacy 命中→query 命中 / GT”；这是描述性覆盖分解，不替代完整 FP 预算。

| 来源/序列 | <5 mm | 5–10 mm | ≥10 mm |
|---|---:|---:|---:|
| BMSR/T1c | 1→14 / 279 | 1→39 / 148 | 0→25 / 43 |
| ISLES/DWI | 1→15 / 68 | 2→19 / 41 | 1→26 / 29 |
| PDGM/FLAIR | 0→0 / 4 | 0→0 / 1 | 27→45 / 48 |

上述三个序列的既有命中均保留，但新增命中不足以抵消预算失败。PDGM/T1c 仅作描述：57 GT 中命中 7→34，FP/scan 0.02439→2.70732；它不是 PDGM 的 S7 判门序列。[分层与配对计数](full_val_paired.json)；[query 汇总](query_aggregate_anonymous.json)；[legacy 汇总](legacy_aggregate_anonymous.json)

## 端到端 R：新增检测覆盖不能当作 R 头改善

全部 U-supervised 序列共有 718 个非 ignored GT；其中两项无 controlled R 记录，R 的固定资格分母为 **716**。固定工作点下，legacy/query 分别匹配并绑定 40/216 个合格 GT，正确端到端宿主数为 **27→128 / 716**；全部合格 GT 配对有 102 项获救、1 项受损。变化主要伴随此前漏检实例获得覆盖。[资格分母与状态转移](full_val_paired.json)

两种解码器**共同匹配并绑定的 40 个 GT** 上，正确数为 **27→26**，0 项获救、1 项受损。query 的条件正确率 128/216 与 legacy 的 27/40 来自不同检测子集，不能比较为 R 头准确率改善；共同子集的结果也只适用于该交集。controlled R 本身完全未变。[配对 R 结果](full_val_paired.json)

## FP 形态：事后探索线索，尚非因果定位

该诊断按同一固定 score 与官方匹配区分 TP、FP、ignored-excused；预定 retained-fraction 档为 ≤0.1、(0.1,0.5]、>0.5，物理大小沿用原分层。所有输出均为匿名汇总，不输出患者、逐例、逐 GT 或 origin 表。[诊断合同](analysis/FP_MECHANISM_CONTRACT.md)；[完整形态统计](full_val_fp_morphology.json)

- **ISLES/DWI：10802 个 FP** 中，84.06% 的最终子实例保留体素占原 parent ≤0.1，89.98% 来自产生多个最终子实例的 parent。FP 子实例体积中位数 **36 mm³**；按子实例加权的 parent 大小中位数 **16359 voxels**。这与大量小残余继承 parent score 的模式一致。
- **BMSR/T1c：697 个 FP** 中，仅 **37.30%** 来自多子实例 parent；FP 子实例体积中位数 202 mm³。因此不能用“碎片化”解释全部来源的失败。
- 这里仅观察 ownership/体积过滤后存活的最终子实例。未保存子实例自身概率均值；不能断言残余概率偏低，也不能据此证明 ownership 或 score 继承造成全部 FP。parent 可同时产生 TP 与 FP，按结果分组的 parent 数不互斥；这些事后关联不直接推出 loss 修复。

## A/R 审计保留的独立模型问题

**A：** 共享 A/U 采点可以遗漏小结构正点，当前 loss 在“dense 存在、采样无正点”时会压低该 query 的 mask logits；这是已验证的机制候选。可是含实例的 12 个固定 TRAIN crop 中，标签 18/54 的共享采样均值同为 2.542 个正点，零正点均为 1/24，这些覆盖统计尚未建立 18 单侧缺失的因果解释。正式 dense A 不受 presence 门控制。[A 监督审计](a_supervision_audit/REPORT.md)

**R：** `embed_coarse` 是独立学习投影，没有直接 A/U mask 监督或 dense–coarse 一致性目标；实际权重证实 Stage II 保持随机初始化，Stage III 后改变。它不等同于整个 coarse pixel pyramid 未训练：共享 pyramid 经 dense 分支收到 A/U 监督。该结构失配不证明全部 R 错误由此造成。[R 梯度/权重审计](r_coarse_supervision/REPORT.md)；[CPU 验证](r_coarse_supervision/verification.json)

## 下一步：分别冻结合同的单变量实验，尚未执行

1. **U：** 冻结残余子实例置信度诊断及 GT-free 评分/前瞻校准合同，直接比较子掩膜原概率与继承 parent score；不在本轮 val 扫阈值、重选工作点或同时改 loss。先检验评分与误报机制，再决定是否值得开展新的解码比较。
2. **R：** 固定 crop、query、host/event/sequence tokens 与 head，仅替换几何来源：同 query 的 dense A/U 软概率经预定 patch 对齐平均池化（2×4×4）映到同一 F1 网格；保留 presence 门、host 合并、valid mask 和物理坐标。共同换 A/U 只能检验联合几何来源，不能区分各自贡献；冻结 head 在 coarse 几何上训练过，阴性结果不能单独排除几何问题。
3. **A：** 固定 checkpoint 与 TRAIN crop，对全部 32 个 query 做 dense/sample logits、GT 正点覆盖与 winner-margin 审计，区分监督遗漏、query 拟合/身份问题和输出竞争；不据本轮结果直接重训。

以上均为拟议实验，没有启动新训练、N7 或显示/API 工作。[实施计划](../../../superpowers/plans/2026-10-17-aur-query-cc-decoder.md)；[当前状态](../../../../STATUS.md)；[项目入口](../../../../CLAUDE.md)

## 资源、验证与复现边界

- 完整 val 共 **4302 个自然窗口**，405 行记录时间合计 **4051.38 s**；这不是多进程总 wall time，也不是相对于历史 legacy 的配对速度比较。68 条复用记录保留原始计时。最大记录 CUDA allocated 峰值为 **9.017 GiB**，不代表 reserved 或总设备占用。[资源汇总](full_val_paired.json)；[调度说明](scheduling/repartition.json)
- 143 个 U-supervised 行合计 411605 个合格候选、124749 个 NMS 抑制候选、190134 个完全遮挡候选，最终 135646 个未做来源 score 过滤的实例；这些中间计数不能充当正式检测灵敏度。[解码统计](full_val_paired.json)
- NMS 仅做精确工作剪枝：优化前后六例全部保存结果字段相同；未保存 voxel maps 的限制仍成立。六例实测进程耗时 507.78→406.60 s，是共享机器回放成本，不能归因为孤立算子加速。[等价性核验](nms_live_equivalence.json)
- 完整隐藏 GPU 的 CPU 测试 **1111 passed、2 skipped，157.46 s**；代码、分析脚本、调度修正及 A/R 科学审计的独立复核均无未解决项。完整配对与 FP 诊断脚本退出码均为 0；命令、哈希、跳过项与审阅记录见 [RUN.json](RUN.json)。

本报告仅收匿名汇总；私有 cases、逐行记录与医学数据不纳入 Git。当前配置的失败、仍未解决的 A/R 问题和伪参照证据边界均保留。
