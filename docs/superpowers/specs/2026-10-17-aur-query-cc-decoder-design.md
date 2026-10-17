# AUR query 保留解码对照：固定实施合同

用户要求按当前模型图继续。A32、统一异常实例 U、每实例 R13 + no-host 均保持；模型权重与结构不变。本合同把既有 NEXT_EXPERIMENT 的跨窗与实例规则写定。下列算法细节是实施者在看新结果前采用的默认选择，不冒充用户逐项指定的参数。

## 基线与唯一变量

基线 bbfc4ba；`legacy_max` 保留全部既有行为和默认值。新增显式 profile `query_cc_v1`：只改变 U 实例解码。checkpoint、样本、val 患者、冻结六例与原诊断相同。presence >0.5、mask ≥0.3、26CC、10mm³ 下限、BMSR/ISLES/PDGM score 0.70/0.60/0.75 固定；不训练、不扫阈值、不启用 N7、不推理 test。

## GT-free 候选

每个自然窗口、每个 presence >0.5 的 query，在有效且位于原体积的体素上先取 mask ≥0.3，再做26CC。按共享 `min_voxels_for(voxel_mm3)` 保留组件。每候选记录原始 float32 概率的组件均值（float64 求和）、global bbox、有效体素集合及 `(window_start_zyx, query_index, component_label)`。没有 GT 输入、候选数量上限或宿主条件过滤。

候选以 bbox 内 packed binary mask 存储，避免为大块永久保存 int64 全局体素索引。最多4个 CPU 线程处理 query；任务完成顺序不得改变结果。

## 固定去重、归属与评分

1. 全部候选按 `(-score, window_start_zyx, query_index, component_label)` 字典序排序。
2. 贪心 mask-IoU NMS：候选与更早保留候选的**原始有效全局掩膜** IoU ≥0.5（含等号）时抑制，记录最早的抑制者。仅用 bbox/体积上界加速，不能以 bbox IoU 代替 mask IoU。不同窗口相同 query 编号不是同一身份。
3. 按同一优先级给原始保留掩膜分配全局体素；先到者拥有重叠体素。先完成全部 ownership，再对每个候选残余做26CC并再次应用10mm³下限。细碎丢弃体素**不回填**，不重新NMS；不同来源候选绝不全局 union 成一个实例。
4. 残余子组件按 raster component label 顺序生成连续实例 id。每个子实例继承原候选 score（优先级不因去除重叠而重算），记录原候选、保留体素比例与来源；明确这是 parent score，不称子组件重新算出的平均概率。
5. 最终 score 过滤在上述步骤完成之后执行；过滤后实例 id 连续重排，来源记录同步更新。评价先保留所有分数供固定阈值计分，不能按 GT 或 test 调阈值。

mask NMS=0.5 和无回填均是冻结的初始试验选择，有可能保留重复或舍弃小片段；报告这些代价，不事后换参数救结果。

## A/R 接口保持

A 的 forward、窗口融合、presence 统计和输出数值不改；`lesion_prob` 继续保留原 max/mean 图作参考，不能把它当新实例的评分源。新实例图/rows/解码统计另返回。

R controlled 保持原 GT 中心裁块、query 选择、粗几何、候选竞争和权重不变。端到端 R 仍对最终每个实例重新裁块并选 query，不把 detector-origin query 冒充实际 binding query。每实例同时保存 `detector_origin` 与 `binding_origin={query,window,in_crop_share}`。R 仍输出14类，不改变 token 构造；U 集合变化导致端到端子集变化须明确报告。

所有实际入口（infer、evaluate_case、外部U评价、eval脚本和shard合并）显式传 profile；旧 metadata 缺 profile 视为 legacy。合并拒绝混合 profile/规则，即使checkpoint相同。原脚本不传profile时行为不变。

### 评价工作量与固定工作点匹配

评价中的 A-only 行沿用原 forward，U 无监督时不计算未被计分的候选；这不改变通用推理入口。预测实例的 R 只需绑定固定来源 score 阈值以上的已完成实例，U 仍保留全部原始候选供误报计算，controlled R 仍覆盖全部监督 GT。绑定前临时重排实例 id，之后映回原检测 id，避免空 id 破坏宿主查表。记录绑定 score 下限，聚合器拒绝用更低阈值或未固定阈值解释这些部分绑定记录。

已用合成例确认：在全部检测上先匹配、再做 score 过滤，会让低分高IoU框抢走被保留高分框的真值。本次固定工作点对照在 score 过滤后匹配端到端 GT；保存的 legacy 输出用同一固定阈值重新关联 GT，不重跑或更改预测。A、U检测数和controlled R原门不受这个计分修正影响。旧默认路径保持兼容，报告披露这项端到端计分归一化。

## 验收和运行顺序

- TDD：query间桥接不合并、跨窗重复抑制、IoU等号边界、优先级稳定、有效区裁剪、ownership后分裂/体积下限、无回填、score过滤后id与来源、空输入、并发顺序及来源正确。
- 小型真实模型证明 A 与 legacy 核心输出不变；controlled R 的宿主/概率/query/window不变。`matched_det/matched_score` 可以随U改变，不用它们判断R核心复现。
- 首先独立代码审阅与完整CPU测试（隐藏GPU；已知沙箱socket限制时直接外部运行）。
- 启动前冻结本合同、代码SHA与原始数据哈希。先同一六例 paired replay，记录时间/显存/候选及抑制/残片/误报、原六例 A 与controlledR复现；两小时上限。失败停止归因。
- 根据六例实测吞吐安排完整405行 frozen val，保持同一规则和阈值，逐来源核算U hits/FP及参照预算；同时报A、controlled/端到端R。运行超过一天则按项目规矩先问用户；不为本轮看过的val另改规则。支持可验证分片/恢复，目录与结果不覆盖。
- 任何oracle或六例局部maskIoU不替代正式全卷boxIoU≥0.1评价。没有完整val结果就不能声称解码有效；A/R仍为NOT_EVIDENCE。

## 交付

明确 profile 的代码与回归测试、被冻结的运行合同、六例与完整val匿名比较、耗时/资源、原始输入哈希、下一项单变量模型研究。A五类输出缺失与R粗几何问题各自保留，不因U改善关闭；不做显示/API或扩展目标。
