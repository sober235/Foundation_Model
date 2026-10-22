# Foundation_Model / AnatoBind-MRI

面向 2D/3D MRI 的解剖实体、病灶实例与病灶–解剖关系学习项目。

项目进展统一发布在`main`：先读[STATUS.md](STATUS.md)顶部，再查[当前候选实验报告](docs/verification/2026-10-21/aur_2d_candidates/RESULTS.md)与同目录`RUN.json`。状态以记录的快照时点为准。

## 历史架构图与子模块说明 PDF（2026-09）

**[打开历史 6 页 PDF](output/pdf/anatobind_architecture_explained.pdf)** · [直接下载 PDF](output/pdf/anatobind_architecture_explained.pdf?raw=true)

PDF 包含整体架构图和 loss function、16 个子模块的技术说明与通俗解释、已实现/待实现状态，以及训练阶段和 novelty 分析。中文字体已嵌入，架构图可无损放大。

文件路径：`output/pdf/anatobind_architecture_explained.pdf`。

| 配套资料 | 入口 |
|---|---|
| 完整文字说明与公式 | [架构、loss 与 novelty](docs/architecture_and_novelty_2026-09-08.md) |
| 单页架构图 | [PDF](docs/figures/anatobind_architecture_with_losses.pdf) · [SVG](docs/figures/anatobind_architecture_with_losses.svg) · [PNG](docs/figures/anatobind_architecture_with_losses.png) |
| PDF 生成脚本 | [render_architecture_guide_pdf.py](scripts/render_architecture_guide_pdf.py) |
| 研究方案 | [RESEARCH_PLAN.md](RESEARCH_PLAN.md) |

## 当前实现状态

当前为脑 MRI 的 A32 解剖、U 病灶实例与 R13 宿主＋no-host 结构化感知。3D SSL-first 三阶段训练及全量评估已执行，但原完整模型的 A/U/R 性能门仍未通过。

2026-10-19 已实现可选的统一 2D/3D 模型、单 checkpoint、严格迁移、可恢复交替训练、classic MR 几何读入与原网格推理；完成三位 TRAIN 患者的小配置 I→II→III pilot 和冻结模型 A/R 诊断。正式统一训练、原生 DICOM 验收及 Level R 人标仍未完成，不能把技术闭环当作模型质量达标。

随后完成148训练数据审计，修复35例TRAIN ISLES标签截断丢失，恢复459个合格病灶组件；原数据与历史结果保留。2026-10-20完成正式数据资格、观察级二维weak-R、纯图像SSL、双mode G1和可恢复正式runner，完整CPU1699 passed/5 skipped。全尺寸24步资源probe及CUDA状态恢复通过；2026-10-21首轮正式Stage I完成2048步，但G1为**2D FAIL / 3D PASS**（二维宿主Dice增益0.043545<预设0.05），流程按登记规则停止，II/III未启动。最终双mode A/U/R质量、原生DICOM几何和医生真值仍未完成。

同日后续已接通[正式权重与双模式评价接口](docs/verification/2026-10-21/aur_evaluation_interfaces/RESULTS.md)，CPU全套1768 passed/5 skipped、真实合同字节验证及独立终审通过；没有真实统一监督checkpoint，接口测试不代表质量达标。[二维保存证据审计](docs/verification/2026-10-21/aur_g1_diagnosis/RESULTS.md)排除了未更新/整体缓存退化；[step0对照](docs/verification/2026-10-21/aur_g1_step0_control/RESULTS.md)显示宿主Dice0.248889→0.279637，但三组患者bootstrap区间均跨0。原G1判定保持，下一模型干预尚未开训。

后续已实现[两个独立二维候选](docs/verification/2026-10-21/aur_2d_candidates/RESULTS.md)：仅提高四个二维适配器的学习率，以及保持原学习率的真正二维中间合并。原完整CPU1820 passed/5 skipped；实际初始化、初始3D等价和每臂16步TRAIN恢复均通过。2026-10-22两臂正式Stage I均完成2048步并选中1792：**LR5为2D PASS / 3D PASS，PLANAR为2D FAIL / 3D PASS**，最终完成证据和判定均已核验。LR5停在下一阶段计划审阅前，II/III未启动；G1通过尚不代表最终A/U/R质量达标，原失败记录保留。

最新状态见 [STATUS.md](STATUS.md)及[训练资格与执行报告](docs/verification/2026-10-20/aur_qualified_training/RESULTS.md)，数据组成及最初修复见[数据审计报告](docs/verification/2026-10-19/aur_data_audit/RESULTS.md)，统一模型及数值限制见[此前报告](docs/verification/2026-10-19/aur_unified_input/RESULTS.md)。上述 PDF 描述早期方案，不代表当前实现或性能。
