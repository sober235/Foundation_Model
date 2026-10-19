# Foundation_Model / AnatoBind-MRI

面向 3D MRI 的解剖-异常关系学习项目。

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

最新状态见 [STATUS.md](STATUS.md)，实现、数值限制及证据见[本轮报告](docs/verification/2026-10-19/aur_unified_input/RESULTS.md)。上述 PDF 描述早期方案，不代表当前实现或性能。
