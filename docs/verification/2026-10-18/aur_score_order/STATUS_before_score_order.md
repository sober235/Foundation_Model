# STATUS：2026-10-17（U 子实例置信度诊断完成；不直接替换评分）

当前阶段：按已确定架构完成原六例val的冻结残余置信度诊断。每遍83个窗口、815个present-query概率平面严格复现，7528个最终子实例已测量，55个controlled R输出及来源完全一致。ISLES的分数继承有数值放大，但BMSR直接换子均值反而增加FP；没有采用评分变体。本轮前状态保存在 `docs/verification/2026-10-17/aur_residual_confidence/STATUS_before_confidence.md`；完整405行val裁决仍有效，本轮没有重跑它、训练或新test推理。

## 1. 三个目标及当前裁决

**A32解剖掩膜、统一异常U实例掩膜、每个U的R13分侧主宿主+no-host（14概率）不变。** S仅为内部序列上下文。默认解码仍为legacy_max；可选query_cc_v1只保留为可复现实验，不能当作已通过的模型修复。

| 目标 | 既有完整val结果（本轮未重跑） | 状态 |
|---|---|---|
| A | 394个A监督行的13-host Dice0.803457；405行保存分数逐项完全不变 | val宏门通过；五类缺失仍未解决，最近历史test0.7988仍不过 |
| U | BMSR/ISLES/PDGM-FLAIR检出2→78、4→60、27→45；FP/scan变为17.0、540.1、4.7317 | 三源均超S7预算，也均低于参考敏感度减0.05；该固定profile不采用 |
| R | 716个controlled实例的共享保存字段完全不变；ABA0.568436对B00.698324 | controlled门仍失败，未修复R |

最新报告 `docs/verification/2026-10-17/aur_residual_confidence/RESULTS.md`，同目录 `RUN.json` 和 `aggregate_anonymous.json` 保存命令/哈希/匿名分层数据。完整405行对照仍见 `docs/verification/2026-10-17/aur_query_cc/RESULTS.md`、同目录 `full_val_paired.json` 和 query/legacy匿名汇总。六例为已暴露开发val，仅机制诊断；A/R采用伪参照，整体NOT_EVIDENCE。没有Level R人标、独立校准/总体过门或临床成功主张。

原六例固定阈值下，parent→子raw均值的事后过滤结果：BMSR命中2/13不变、FP16→23；ISLES命中11/40→10/40、FP1511→171；PDGM命中2/2不变、FP6→0。三种分数校准不同，反事实保留全部原mask/NMS/ownership，仅过滤后重匹配，不是新的正式解码性能结果。

## 2. 已完成的实现与控制

- 最新诊断代码6b26137，原推理/模型/训练36个依赖文件与冻结1874源码一致。7528个子实例测量parent/raw/fused三项分数；原U行、GT、解码统计、legacy检测、六例A保存分数、55个R共享字段及query/window来源精确复现。两遍当前A/U图和上下文精确相同，历史未保存图不作跨轮声明。隐藏GPU的完整缓存恢复核验通过，未加载模型。
- 冻结实现代码1874af2，checkpoint SHA256 `38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde`；实际best选自Stage III step3000/72,000crops，80,016crops为已完成训练预算，权重本轮未改。
- query_cc_v1：独立窗口/query的26CC，presence>0.5、mask>=0.3、共享10mm³floor、原mask IoU>=0.5贪心NMS、按固定优先级归属体素后再拆残余组件，子组件继承父分数；无GT挑选、无候选上限、无回填。detector来源和重裁块后实际R query来源分别记录。设计与计划在 `docs/superpowers/{specs,plans}/2026-10-17-aur-query-cc-decoder*`。
- 405/405保存A分数严格相等；716/716controlled R共享字段严格相等。旧缓存未保存A体素图、controlled query/window来源，不能声称这些未保存对象也已逐项验证。
- 旧端到端缓存先匹配再过滤分数的问题已在本对照中统一为先固定阈值过滤、后一对一bbox IoU>=0.1匹配；2个保留检测恢复GT关联，旧匹配池38→40，预测未改变。原始缓存保留。
- 716个controlled合格GT的正确端到端宿主27→128；共同匹配的40个GT正确数27→26（0rescue、1harm）。新增成功主要来自检测覆盖，不能说R头质量提高。U共718个非忽略GT，有2个没有controlled记录；分母分开报告。
- 最新完整CPU测试1151 passed、2 skipped，158.53秒；40项新增针对性测试通过。隐藏GPU、nice19、线程2，沙箱外解决DataLoader本地socket限制。两项代码审阅发现已通过11项RED回归修复，复审0项未解决；科学审阅记录见最新RUN.json。上轮1111项和A/R审计仍为其对应历史版本的验证记录。

## 3. 误报与 A/R 机制线索

- 数值源分数固定BMSR0.70、ISLES0.60、PDGM0.75；raw父组件均值与legacy融合图均值校准不同。本轮比较不是等观测FP，也没有搜索阈值；不能由此否定所有query保留方案。
- 完整val的既有事后形态分析：ISLES10,802个FP中，84.06%子组件保留父体积<=10%，89.98%来自多子组件父组件，子体积中位数36mm³；该完整集当时没有保存子概率。本轮六例直接测量：ISLES原1511个FP的parent/raw均值为0.63888/0.46675，1350个raw低于0.60；但BMSR原16个FP中14个保留比例>0.5且raw中位数0.71872，自身仍过0.70。直接换raw不能统一修复误报；没有把六例统计外推为完整val统计。
- A标签覆盖审计：8个TRAIN行、16裁块、32点采样；共享采样42.282%点落在A忽略区，uniform为1.350%。密集存在但采样无正点时，实际mask损失推动空预测；18/54在病灶裁块的正点均值及零正点频次相同，尚不能据此建立单侧18缺失的因果解释。见 `a_supervision_audit/REPORT.md`。
- R粗投影与dense投影独立，无直接A/U掩膜或一致性监督；Stage II粗投影与seed0初始化逐位相同，Stage III后改变。共享coarse像素金字塔接受A/U训练，不能称整个粗分支未训练。CPU权重/config/历史源码验证已由root和独立审阅者重跑。见 `r_coarse_supervision/REPORT.md`。

## 4. 下一项模型工作

1. **U置信度核验已完成，下一步预先冻结单变量评分/校准对照**：TRAIN侧按患者分折仅约束校准器，基模型已经见过TRAIN；需另行锁定与模型/校准开发隔离的评价患者，规定分来源FP/敏感度验收。已暴露六例与完整val保持dev身份，不用于挑最终阈值。显式检验BMSR新入选FP和ISLES剩余高分FP；若只能复用既有val，仍只作开发诊断。mask、ownership和R保持固定；当前默认legacy_max和完整405行FAIL不变。
2. **R单独做固定token/query/crop/权重的几何来源对照**：仅将预测coarse A/U几何换为同一F1网格上的预测dense A/U几何，预先固定池化、valid、gate和坐标规则，记录逐特征和宿主logit变化。联合A/U替换不能区分二者贡献；冻结R头的分布变化也限制阴性结果解释。该实验尚未执行。
3. **A用同一冻结TRAIN裁块审计全部32query的dense概率和胜出类别margin**，分开检查正点遗漏、身份/优化塌缩及类别竞争，再设计单一采样或损失变量。不要仅据18/54相同的正点均值及零点频次直接跳到改loss。

这些是下一步诊断/实验，不是已完成的修复。三个模型目标始终优先；N7、DICOM/独立2D、网页/API/URL及Level R实际读者工作仍按既有顺序处理。

## 5. 资源与别重做

- 新六例回放已在空闲A100物理卡3完成：成功wall186.79秒、行时间合计175.66秒、峰值CUDA allocated9.0176GiB，CPU nice19/数值线程1/decoder4。首次外部启动器参数转发失败发生在模型加载前，2.57秒；修正入口并核验help后重启，日志保留。不要重新跑这六例来补已完成的数值测量；相同冻结合同可直接验证恢复。
- 完整val4302自然窗口，累计冻结单行推理4051.38秒，峰值已记录单行CUDA分配9.017GiB；并行墙钟跨度约4282.42秒含调度/空闲，不能与缓存legacy时间作速度对比。六例NMS加速前后全部保存结果相同，实测507.78→406.60秒为共享主机观测。
- 先检查分片的source/sequence/U监督分布。排序后步长2使全部20个DWI落一端；本次保留已完成odd202行，验证原始全部输入哈希后复用even68行，最终以1/2、0/4、2/4三完整分片合并202+102+101=405，原0/2partial不入结果。见 `scheduling/repartition.json`。
- 恢复必须保持code/checkpoint/所有原始输入/规则/参考哈希；重新分片还必须核对未完成行的原冻结输入哈希。逐例/逐query表、影像与Level R盲读材料只留本地private目录，不能入git。
- 数据只读/data2/congcong/data/FM_data；nvgen绝对路径、PYTHONNOUSERSITE=1；CPU测试隐藏GPU。GPU按用户常设授权，先查实时负载/显存/活跃任务及调度可见性，选兼容空卡并保留既有工作负载。
