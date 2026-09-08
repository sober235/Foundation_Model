# Foundation_Model / AnatoBind-MRI

面向 3D MRI 的解剖-异常关系学习项目。

## 架构图与子模块说明 PDF

**[打开最新版 6 页 PDF](output/pdf/anatobind_architecture_explained.pdf)** · [直接下载 PDF](output/pdf/anatobind_architecture_explained.pdf?raw=true)

PDF 包含整体架构图和 loss function、16 个子模块的技术说明与通俗解释、已实现/待实现状态，以及训练阶段和 novelty 分析。中文字体已嵌入，架构图可无损放大。

文件路径：`output/pdf/anatobind_architecture_explained.pdf`。

| 配套资料 | 入口 |
|---|---|
| 完整文字说明与公式 | [架构、loss 与 novelty](docs/architecture_and_novelty_2026-09-08.md) |
| 单页架构图 | [PDF](docs/figures/anatobind_architecture_with_losses.pdf) · [SVG](docs/figures/anatobind_architecture_with_losses.svg) · [PNG](docs/figures/anatobind_architecture_with_losses.png) |
| PDF 生成脚本 | [render_architecture_guide_pdf.py](scripts/render_architecture_guide_pdf.py) |
| 研究方案 | [RESEARCH_PLAN.md](RESEARCH_PLAN.md) |

## 当前实现状态

数据引擎和 Arm-B 最小训练路径已有实现；S、U_Q、E 分支与完整配对干预训练仍在方案阶段。当前关系模块使用 GT 几何，已有 smoke-run 指标不能证明绑定或泛化优势。详细边界见上述 PDF。
