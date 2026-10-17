# STATUS：2026-10-17（六例建模机制诊断完成，A/U/R 模型门仍未过）

当前研究阶段：用户批准按 2026-10-16 的建模诊断设计继续；工具、冻结六例回放、缓存分析和结果归档均已完成。没有重新训练或改动模型权重。原 2026-10-16 状态完整保存在 `docs/verification/2026-10-17/aur_model_diagnostic/STATUS_2026-10-16_before_replay.md`。

## 1. 已完成且已验证

- **三个目标不变：A32 解剖掩膜、统一异常的 U 实例掩膜、每个 U 的 R13 分侧主宿主 + no-host。** G2-A/G2-U/G3 不变；没有日期节点，模型成功优先。
- 最近训练结果仍是 `unmatched_empty_v1` C2：test A 0.7988 <0.80；U PDGM/BMSR/ISLES 0.6538/0.0146/0.0242，三源不过；controlled R 0.5606 <B0 0.7545。新诊断不改变这些裁决。
- **执行计划** `docs/superpowers/plans/2026-10-16-anatobind-aur-model-diagnostic.md`；**设计** `docs/superpowers/specs/2026-10-16-anatobind-aur-modeling-next-step-design.md`。实现代码到 `b240962`：非扰动 inference observers、全部 query 的阈值掩膜/数值证据、GT/预测组件/R fail-closed 复现、val-only 选例与哈希冻结。
- **独立代码评审完成，三条 Important 均已修复并回归验证。完整 CPU 测试 1065 passed、2 skipped，155.20 秒。** 首次测试卡住是沙箱禁止 DataLoader 共享张量所需的本地 socket；相同测试及完整套件在沙箱外、隐藏 GPU 后通过，未改测试/模型绕过。
- **6/6 验证卷的 U 组件与 GT 几何、55/55 controlled R 的实例集合/宿主/概率/query IoU 复现；282 项 A 保存分数完全一致。** 每源两名患者，83 个自然窗口，A800 GPU 2，180.31 秒、峰值分配 9.018 GiB。两小时硬超时未触发；没有 test 推理或阈值搜索。
- **U：观察到 query max 破坏实例。** 主 mask 阈值0.3下，在完整可见窗口中有好组件（mask IoU≥0.5）的 GT 数：全部 query 18/55、presence 保留 query 15/55、max 图7/55。12/55有同窗保留 query 的好组件被 max 损伤；3/55在所有完整窗口丢掉好的 presence 前候选。所有55GT至少有一个完整可见自然窗口。整 query IoU低可能来自断开假阳块，不能直接归因训练失败。
- **A：五类实际未输出。** 标签5/44/18/26/58在六例中真值均非零、预测均为零。A整卷推理不使用 entity presence筛选，不能用presence解释。完整保存val上这些标签也全部有限Dice为0。
- **R：粗几何与正式dense感知失配。** 55裁块的同网格dense A平均宿主Dice0.7440，coarse ungated/gated约0.0485/0.0487；中心采样控制保留该差距。三个dense选中query较好的绑定，其coarse U前景占约99.93–100%有效crop。局部与整卷GT宿主55/55一致，未观察到这个目标差异。小病灶粗采样有丢失，不能把coarse IoU直接当dense检测性能。
- **所有结果仍 NOT_EVIDENCE。** 六例中33个GT来自一卷，不代表人群；A/R仍是SynthSeg伪参照。不能把GT-best局部组件覆盖当作可部署检测、正式U灵敏度、真实关系或临床验证。

**报告入口：** `docs/verification/2026-10-17/aur_model_diagnostic/RESULTS.md`，运行哈希/资源 `RUN.json`，匿名分组分析 `u/REPORT.md`、`ar/REPORT.md`。完整405行保存val的A/R分析在 `docs/verification/2026-10-16/aur_model_diagnostic/saved_val_ar/REPORT.md`。

## 2. 待解决的模型决定

- 下一项优先候选已收敛为**冻结权重、保留query身份的U解码对照**，完整误报核算先于任何新训练。不是已经完成的修复；跨窗去重、重叠体素归属和R token来源须在实施设计中写定，不能让GT挑候选或根据结果随意选规则。
- A的五类缺失仍需检查监督正点覆盖、dense概率与身份竞争；当前没有证明是训练样本不足或presence过滤。
- R需单独控制几何来源，先固定token/query/U，仅替换A几何来源检验敏感性。当前粗语义差距不等于已证明全部R错误的因果原因；不与U解码改动混在同一实验。
- N7语义头不是本轮选择；旧60k iterations触发与当前240k crops不可直接换算。若后续启用，按原规格先说明并确认。Level R人标仍需真实读者，不能伪造或用伪标签代替。

## 3. 下一步

1. 按 `docs/verification/2026-10-17/aur_model_diagnostic/NEXT_EXPERIMENT.md`，为GT-free、query保留解码写实施设计，明确跨窗去重、评分、重叠归属与R实例/token合同。
2. 固定同一checkpoint，先在原六例核验实例来源和误报，再固定规则在完整val比较同FP预算下的U；A不变，controlled和端到端R分别报告。该对照尚未实现/运行；不能用本次oracle成绩代替。
3. 根据实际收益决定后续单一模型修正。A监督/掩膜竞争、R粗几何来源分别保留，任何一项不过都不能宣布A/U/R完成；不用历史test选模型。
4. 模型成立后再做DICOM几何读入、离线3D验收与独立2D数据/模型/参照；网页/API/公开URL继续后置。

## 4. 坑与别重做

- 六例选例、checkpoint、数据与原评价哈希已冻结。复现失败要停止归因，不放宽容差。公开只收汇总；逐例/逐query/盲读材料留本地。
- 原始局部缓存约13GiB，在 `/home/congcongliu/outputs/aur_model_diagnostic/20261017_replay_b240962/`；浮点代表缓存float16，仅用于核查，不能替代原float32组件score。全部query的0.3/0.5 hard masks及数值测量均保留。
- 控制query的整mask IoU、其独立CC IoU、max后CC IoU、整卷bbox IoU是不同量；局部GT-best选择只作oracle。跨窗平均压掉3.41%的any-window支持GT体素，不是实例灵敏度。
- R所读coarse mask不等于正式dense A/U；saved local A Dice也不等于R的实际粗A。最近邻粗采样会漏小病灶，保留同网格/中心相位控制。
- 原val REPORT的test-set/test-selected是历史模板措辞问题；实际先val选阈值后test固定，不据措辞断言泄漏。
- 数据只读 `/data2/congcong/data/FM_data`；nvgen绝对路径；CPU测试隐藏GPU；确认空卡；不覆盖历史产物。没有本轮资源快照就不把旧卡位当现状。

## 5. 为什么这样推进

本轮证据已经定位到一个可观察的解码信息损失，并排除了若干简单解释；还没有证明GT-free实例解码能兼顾误报和R身份。因此先做冻结权重的单变量对照，而不是同时换loss、训练、U架构和R几何。A32/U实例/R主宿主始终是交付目标，任何诊断工具、oracle或子集好看都不能替代它们。
