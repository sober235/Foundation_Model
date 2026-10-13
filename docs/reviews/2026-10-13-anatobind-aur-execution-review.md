# AnatoBind-MRI A/U/R 执行一致性复核

日期：2026-10-13

复核基准：`feature/aur-brain-iter2-2026-10-13`，HEAD `d31dd5a`。复核时远端 `main` 与该分支指向同一提交。

范围：只读核对决策文档、源码、评估报告和最新状态记录；本次未重跑训练、全量数据评估或测试。

## 结论

**脑部 A/U/R 工程线大体遵循 2026-10-08/09 的局部规格，但 A、U、R 三项评估门均未通过；C2 另有一项明确的预注册流程违约。** 这条线可以作为工程验证和失败诊断，不能作为项目科学目标已经完成的证据。

判断须分两层：

1. **局部脑部实现目标。** 10-08 规格明确限定为脑部 A/U/R 子项目，评估使用 SynthSeg 与病灶伪标签，并标为 `NOT_EVIDENCE`；10-09 后续执行计划又把原先“跳过 Stage I”改为严格 `Stage I → II → III`。当前 SSL-first 执行应按较新的 10-09 决策评判。[脑部规格](../superpowers/specs/2026-10-08-anatobind-brain-aur-design.md) · [SSL-first 执行计划](../superpowers/plans/2026-10-09-anatobind-brain-ssl-first-execution.md)
2. **v2.6 第一篇论文目标。** 该目标要求 Level R 独立人标真值、患者级五折评估和公平关系基线；主要问题是 learned R 是否优于几何方法。当前多来源伪标签实验不能替代这些步骤。[v2.6 计划](../plans/2026-09-22-aur-v2.6-experiment-design-route.md)

## A/U/R 对照

| 子目标 | 规格与执行 | 结果 | 复核判断 |
|---|---|---|---|
| **A：解剖识别** | 局部规格要求 13 类宿主平均 Dice ≥ 0.80；G1 是冻结 backbone 的表征探针，不等同于最终 A 分割门。 | G1 第二轮读出 0.724，对照随机初始化 0.383；病灶可分性 AUC 0.667，对照 0.680。G2-A：C2 0.7961、C0 0.6289，均未达 0.80。G1 的 0.05/0.02 判据仍标为 `PROPOSED`。 | 架构和指标符合局部工程规格；表征探针通过不代表 A 通过。A 门失败。C2 测试结果还受下述流程违约限制。[G1 记录](../verification/2026-10-09/anatobind_brain_ssl_first/g1/G1.md) · [C0 报告](../verification/2026-10-09/anatobind_brain_ssl_first/eval/c0_test_20261010_1533/REPORT.md) · [C2 报告](../verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_test_20261011_0500/REPORT.md) |
| **U：病灶识别** | 局部规格要求各来源在 nnU-Net 同误报工作点下，灵敏度不低于参照 0.05 以上。阈值先在验证集选择，再固定到测试集；全测 1,056 行。 | C2：PDGM FLAIR 0.7538（参照 0.8615）；BMSR T1c 0.0049（参照 0.7316）；ISLES DWI 0.0069（参照 0.6303）。G2-U 失败。最新状态记录称，数据与标签对齐检查正常，异常来自事件输出合并。 | 测量路径符合局部规格，U 目标未达成。源码机制与记录诊断一致：`event_loss` 对所有 query 监督存在性，但 mask loss 仅作用于匈牙利匹配项；推理对过存在概率门限的 query mask 取最大值，再阈值化为连通域。未匹配 query 的散点可能污染合并图。此处是代码路径核验，不是本次重新跑数据所得的独立复现。[U 状态](../../STATUS.md#L60) · [event loss](../../anatobind/aur/losses.py#L122) · [推理合并](../../anatobind/aur/infer.py#L127) |
| **R：病灶–解剖绑定** | 局部 G3 以伪关系标签评估 controlled ABA，门槛为 R ≥ B0。 | C2 ABA(R) 0.5142，B0 0.6692；C0 0.5007，B0 0.6789，均失败。C2 的 53 例差距子集中，R 救回 23 例、弄错 7 例；仅作探索性描述。 | 局部伪标签 G3 失败。评估报告明确写明 controlled truth 来自 SynthSeg lookup。Level R 人工读片尚未开始，故临床关系真值与 R0/R1 仍未验证。[C2 R 报告](../verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_test_20261011_0500/REPORT.md#L41) · [v2.6 R0/R1](../plans/2026-09-22-aur-v2.6-experiment-design-route.md#L99) |

所有 C0/C2 全测指标均基于伪标签并标为 `NOT_EVIDENCE`。不能据此声称临床效能、关系泛化或研究假设成立。

## 执行偏差与当前状态

### Q25 C2 停止条件没有生效

Q25 预先规定：C2 第 1000 步时，冻结宿主读出需 ≥0.65，且验证 A 点 Dice 需 **>0.045**；任一项未过即停止并报告。实测读出 0.6829 通过，A Dice 0.0443 未通过，但训练链仍继续到 Stage II/III 和全测试。`c2v2_check.sh` 只打印 `FAIL - stop and report`，没有自动终止链；当时又无人值守。故 C2 后续指标必须写作“越过一次未通过的预注册检查后所得”，不能当作严格按 Q25 完成的确认性门结果。[Q25 记录](../verification/2026-10-09/anatobind_brain_ssl_first/stage2/STAGE2_V2.md) · [检查脚本](../verification/2026-10-09/anatobind_brain_ssl_first/ops/c2v2_check.sh)

### 配对 C0 与 U 修复

最新 10-13 记录显示，新配方 C0 已在 0、2 号卡启动，`II_BB=5e-5`，与新 C2 配置匹配；这是 Q25 配对比较的一部分。它能回答相同 Stage II backbone 学习率下，预训练与随机初始化的差异，尚不能修复 U 解码失败。[启动记录](../../STATUS.md#L61)

U 的解码/损失调整在规格中属于门失败后的新选择；当前记录列出选项，但尚未形成新的预注册方案。任何改变应先确定验证集诊断、操作点和停止规则，再保留测试集作最终一次评估。

### 状态文档滞后

`STATUS.md` 标题仍写 10-12，旧段落仍称新配方 C0 未启动；同一文件 10-13 追加记录已写明 C0 启动。其 `待用户定` 与 `下一步` 部分也包含旧状态。当前进展应以 10-13 追加记录为准，并同步整理状态文件，避免旧待办覆盖新事实。[STATUS](../../STATUS.md#L1) · [10-13 更新](../../STATUS.md#L60)

## 与 v2.6 第一篇目标的差距

当前局部规格做的是四来源脑部单序列 A/U/R 工程线，约 20% 病人留出、伪标签评估，R 对照为 B0。v2.6 第一篇主实验则要求：

- Level R 双读者及裁定，先通过独立关系真值门 R0；
- 以 reference lesion instances + predicted anatomy 评估 R，脑侧 U 属于次要系统实验；
- 患者级五折与封存测试折；
- 在同一 A/U、split 和关系真值上比较 B0、Bprior、Bgeo+、B1–B4 等基线。

因此，当前脑部 A/U/R 线与局部 10-08 规格一致；作为 v2.6 主论文完成度，它仍是前置工程筛查，Level R、强几何基线与正式患者级评估尚缺。[v2.6 Gate A/U、R0、R1](../plans/2026-09-22-aur-v2.6-experiment-design-route.md#L74)

## 建议的收敛路线

1. 完成已经启动的新配对 C0；把 C2 Q25 违约及所有伪标签边界写进比较结论。
2. 后续门控改成 fail-closed：检查失败时训练链自动退出，避免只打印失败状态。
3. U 先由用户确定是否启动新的解码/损失实验；按验证集预注册规则运行，测试集保持封存。
4. R 的论文级结论转入 v2.6 路线：先完成 Level R pilot/R0，再运行公平关系基线和 R1。
5. 同步 `STATUS.md` 与项目入口文档，明确“脑部工程线”与“v2.6 主论文实验”的关系，并标清尚未用户批准的 `PROPOSED` 超参数。

## 证据文件

- `docs/superpowers/specs/2026-10-08-anatobind-brain-aur-design.md`
- `docs/superpowers/plans/2026-10-09-anatobind-brain-ssl-first-execution.md`
- `docs/plans/2026-09-22-aur-v2.6-experiment-design-route.md`
- `STATUS.md`
- `docs/verification/2026-10-09/anatobind_brain_ssl_first/g1/G1.md`
- `docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c0_test_20261010_1533/REPORT.md`
- `docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_test_20261011_0500/REPORT.md`
- `docs/verification/2026-10-09/anatobind_brain_ssl_first/stage2/STAGE2_V2.md`
