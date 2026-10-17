# STATUS：2026-10-17（完整 query-CC 对照完成；固定规则未过 U 门）

当前阶段：按已确定架构完成冻结权重、GT-free 的 U 实例解码对照。完整405行val已运行、复现、合并和配对分析；没有重新训练或新test推理。此前状态保存在 `docs/verification/2026-10-17/aur_query_cc/STATUS_before_query_cc.md`。

## 1. 三个目标及当前裁决

**A32解剖掩膜、统一异常U实例掩膜、每个U的R13分侧主宿主+no-host（14概率）不变。** S仅为内部序列上下文。默认解码仍为legacy_max；可选query_cc_v1只保留为可复现实验，不能当作已通过的模型修复。

| 目标 | 本轮完整val结果 | 状态 |
|---|---|---|
| A | 394个A监督行的13-host Dice0.803457；405行保存分数逐项完全不变 | val宏门通过；五类缺失仍未解决，最近历史test0.7988仍不过 |
| U | BMSR/ISLES/PDGM-FLAIR检出2→78、4→60、27→45；FP/scan变为17.0、540.1、4.7317 | 三源均超S7预算，也均低于参考敏感度减0.05；该固定profile不采用 |
| R | 716个controlled实例的共享保存字段完全不变；ABA0.568436对B00.698324 | controlled门仍失败，未修复R |

主报告 `docs/verification/2026-10-17/aur_query_cc/RESULTS.md`；完整命令/哈希/资源 `RUN.json`；匿名配对数据 `full_val_paired.json`；匿名标准汇总 `query_aggregate_anonymous.json`、`legacy_aggregate_anonymous.json`。A/R采用伪参照，本轮整体NOT_EVIDENCE；BMSR/ISLES对S7仍有原生与1mm网格差异。没有Level R人标或临床成功主张。

## 2. 已完成的实现与控制

- 冻结实现代码1874af2，checkpoint SHA256 `38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde`；实际best选自Stage III step3000/72,000crops，80,016crops为已完成训练预算，权重本轮未改。
- query_cc_v1：独立窗口/query的26CC，presence>0.5、mask>=0.3、共享10mm³floor、原mask IoU>=0.5贪心NMS、按固定优先级归属体素后再拆残余组件，子组件继承父分数；无GT挑选、无候选上限、无回填。detector来源和重裁块后实际R query来源分别记录。设计与计划在 `docs/superpowers/{specs,plans}/2026-10-17-aur-query-cc-decoder*`。
- 405/405保存A分数严格相等；716/716controlled R共享字段严格相等。旧缓存未保存A体素图、controlled query/window来源，不能声称这些未保存对象也已逐项验证。
- 旧端到端缓存先匹配再过滤分数的问题已在本对照中统一为先固定阈值过滤、后一对一bbox IoU>=0.1匹配；2个保留检测恢复GT关联，旧匹配池38→40，预测未改变。原始缓存保留。
- 716个controlled合格GT的正确端到端宿主27→128；共同匹配的40个GT正确数27→26（0rescue、1harm）。新增成功主要来自检测覆盖，不能说R头质量提高。U共718个非忽略GT，有2个没有controlled记录；分母分开报告。
- 完整CPU测试1111 passed、2 skipped，157.46秒；隐藏GPU、nice19、线程2，沙箱外解决DataLoader本地socket限制。独立代码/分析审阅的问题均已关闭，A/R科学审计0项可操作问题；细节见报告。

## 3. 误报与 A/R 机制线索

- 数值源分数固定BMSR0.70、ISLES0.60、PDGM0.75；raw父组件均值与legacy融合图均值校准不同。本轮比较不是等观测FP，也没有搜索阈值；不能由此否定所有query保留方案。
- 完整val的事后形态分析：ISLES10,802个FP中，84.06%子组件保留父体积<=10%，89.98%来自多子组件父组件，子体积中位数36mm³。BMSR只有37.30%FP来自多子组件父组件，故碎片不能解释全部失败；未保存子组件概率均值，尚不能直接判定重评分即可修复。见 `full_val_fp_morphology.json`。
- A标签覆盖审计：8个TRAIN行、16裁块、32点采样；共享采样42.282%点落在A忽略区，uniform为1.350%。密集存在但采样无正点时，实际mask损失推动空预测；18/54在病灶裁块的正点均值及零正点频次相同，尚不能据此建立单侧18缺失的因果解释。见 `a_supervision_audit/REPORT.md`。
- R粗投影与dense投影独立，无直接A/U掩膜或一致性监督；Stage II粗投影与seed0初始化逐位相同，Stage III后改变。共享coarse像素金字塔接受A/U训练，不能称整个粗分支未训练。CPU权重/config/历史源码验证已由root和独立审阅者重跑。见 `r_coarse_supervision/REPORT.md`。

## 4. 下一项模型工作

1. **U优先核验残余实例置信度和完整错误组件**，再设计前瞻性、GT-free的评分/校准对照。当前profile保留以便复现，不用本轮val重新挑阈值；不同时改mask训练、ownership和R几何。
2. **R单独做固定token/query/crop/权重的几何来源对照**：仅将预测coarse A/U几何换为同一F1网格上的预测dense A/U几何，预先固定池化、valid、gate和坐标规则，记录逐特征和宿主logit变化。联合A/U替换不能区分二者贡献；冻结R头的分布变化也限制阴性结果解释。该实验尚未执行。
3. **A用同一冻结TRAIN裁块审计全部32query的dense概率和胜出类别margin**，分开检查正点遗漏、身份/优化塌缩及类别竞争，再设计单一采样或损失变量。不要仅据18/54相同的正点均值及零点频次直接跳到改loss。

这些是下一步诊断/实验，不是已完成的修复。三个模型目标始终优先；N7、DICOM/独立2D、网页/API/URL及Level R实际读者工作仍按既有顺序处理。

## 5. 资源与别重做

- 完整val4302自然窗口，累计冻结单行推理4051.38秒，峰值已记录单行CUDA分配9.017GiB；并行墙钟跨度约4282.42秒含调度/空闲，不能与缓存legacy时间作速度对比。六例NMS加速前后全部保存结果相同，实测507.78→406.60秒为共享主机观测。
- 先检查分片的source/sequence/U监督分布。排序后步长2使全部20个DWI落一端；本次保留已完成odd202行，验证原始全部输入哈希后复用even68行，最终以1/2、0/4、2/4三完整分片合并202+102+101=405，原0/2partial不入结果。见 `scheduling/repartition.json`。
- 恢复必须保持code/checkpoint/所有原始输入/规则/参考哈希；重新分片还必须核对未完成行的原冻结输入哈希。逐例/逐query表、影像与Level R盲读材料只留本地private目录，不能入git。
- 数据只读/data2/congcong/data/FM_data；nvgen绝对路径、PYTHONNOUSERSITE=1；CPU测试隐藏GPU。GPU按用户常设授权，先查实时负载/显存/活跃任务及调度可见性，选兼容空卡并保留既有工作负载。
