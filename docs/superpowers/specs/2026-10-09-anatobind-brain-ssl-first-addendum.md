# AnatoBind-Brain 设计规格增补：Stage I 必须先于 Stage II/III（2026-10-09）

**决策来源：** 原 [2026-10-08 AUR 设计规格](2026-10-08-anatobind-brain-aur-design.md) 仅为先验证模型功能、跳过 Stage I 的简化实现。新的用户决定是 **完整 Brain MRI Foundation Model 正式路线必须先自监督预训练（Stage I），再训练结构化感知（Stage II），最后训练关系推理（Stage III）**。

**完整实施方案：** [2026-10-09-anatobind-brain-ssl-first-three-stage.md](../plans/2026-10-09-anatobind-brain-ssl-first-three-stage.md)。

## 四卡训练就绪性补充（同日 Review）

在真正提交 Stage I GPU 主训前，先完成 [4×A800 审查报告](../reviews/2026-10-09-anatobind-brain-four-a800-execution-readiness.md) 中的 **P0-1 至 P0-5**。已有单/四卡 BF16 显存结果只针对旧 Part 1 **合成数据**，不能作为 Stage I 两视图吞吐或真实 NIfTI loader 的承诺。需先移植并回归验证 MRI-148 本地已有而当前新分支缺失的 RAS/`entity_present` 更新；默认 NCCL P2P 有记录的初始化问题，当前四卡已证实的兼容设置为 `NCCL_P2P_DISABLE=1`。

Stage I 主线建议先试 **4 GPU × microbatch 2 × accumulate 2 = global 16 source-crops**，320k 曝光约 20k optimizer steps。上述为 probe 选择，不是已经验证的 Stage I 显存/吞吐。要以 G0、无泄漏的 SSL smoke（G0.5）、G1 三项独立审核后才能进入 Stage II。MRI-148 当前文件连接器不具备 GPU 执行接口，本审查未启动训练。

## A. 变更范围（优先于旧 N11 的训练顺序）

| 原设计 | 增补后正式主线 | 不变部分 |
|---|---|---|
| 只训练 Stage II→III，跳过 Stage I | **Stage I→II→III**，每阶段通过 gate 才能进入下一阶段 | 同一 3D Swin、A32/U64/S6、R13+no_host |
| 预训练无实施安排 | 3D MIM + 病人安全对比学习，统一 backbone 参数接口 | 单序列输入、native grid、3D RoPE、M_valid |
| Stage II 随机 Backbone | Stage II 必须从严格校验的 Stage I backbone-only checkpoint 初始化 | A/U/S 联合监督与病人级隔离 |
| Stage III 从 Stage II 继承 | 保持原方案，不变 | crop-local R truth、host competition、hard negatives |
| 只靠伪标签评价 | 另增严格 G1 表征探针与独立人工 Level R 审核 | 伪标签指标依旧标 NOT_EVIDENCE |

## B. 必须明确的工程契约

1. **Stage I 不读取病灶真值作为目标。** MIM 只重建被遮挡的有效前景体素；对比学习把同患者同区域增强视图作为正对，不把同一患者的其他序列/重复块错误当作负样本。
2. **真实遮挡不可见：** 模型的 patch embedding、shifted-window attention、多尺度融合不得在预测时看到被遮挡区域的原始像素；M_valid 与 M_visible 独立定义，并测试“改变隐藏像素不改变输出”。
3. **空间物理一致：** affine、单位、方向、ROI 与增强同步；不默认做左右镜像；MIM 与 Stage II 直接复用相同 3D Swin。
4. **标注患者隔离：** Stage I 在 Stage II/III 的 test/val 病人上不做梯度更新；若额外无标注 MRI 无法排除同患者重复，则不进入主训练清单。
5. **Stage II 严格继承权重：** 显式验证 state_dict 全部 backbone 参数键和形状；不静默跳过未能加载的层，不悄悄将主线退回随机初始化。
6. **r_supervised 必须真正参与 Stage III loss gating**，不能只把配置标志写进数据行。ISLES A 伪标签 QC 不过时，A/R 监督关闭，U/S 继续训练。
7. **Stage I 的 pilot 初值，不是已通过训练验证的最终超参：** mask 0.60，温度 0.20，对比系数 0.10，8k pilot、320k 主训练 crop 曝光；探针确定显存/速率后锁定。
8. **三阶段训练与评估必须可区分：** C0 无预训练，C1 MIM-only，C2 MIM+contrast，C3 R 用 B0 几何查表；至少在种子/划分/监督数据和预算相同的情况下比较。
9. **评价边界：** A 对 SynthSeg 的 Dice、R 对 SynthSeg 派生宿主的 ABA 是伪标签一致性，只能作为工程门。未知病种发现、临床可靠性需新的独立病例及人工宿主标签，不能靠网络结构名称推断。
10. **禁止在 GitHub 推送包含患者数据的 NIfTI、原始病例路径清单、密码、访问令牌或其他可识别元数据。**

## C. Gates

- **G0**：先锁定 patient-level manifests、物理网格/affine/QC、外部来源去重、许可、GPU 基线。
- **G1**：Stage I 遮挡无泄漏、loss 有效、表示不塌缩、未见患者 SSL/冻结轻量 A/U 探针合格；导出 backbone checkpoint。
- **G2**：Stage II A 13 宿主宏 Dice ≥0.80（仅伪标签工程门）；U 在每来源匹配 FP/case 下的敏感度 ≥ nnU-Net −0.05；单独报告小于 5mm 和超 64 实例负荷。
- **G3**：Stage III R controlled ABA ≥ B0；受控/端到端及 rescue/harm 分别报告；独立盲审 Level R 决定临床正确性，不用伪关系自证。

**实施状态：** 本增补和主方案是**待执行的工作定义**，不声称 Stage I/II/III 已训练，亦不声称 G0–G3 已通过。原 Part 1 仍有效，不修改其历史决策记录。
