"""Build a six-page Chinese PDF guide with the existing vector architecture.

Dependencies: reportlab, PyMuPDF; fonts are supplied by this Linux workstation.
Run: /home/congcongliu/anaconda3/bin/python scripts/render_architecture_guide_pdf.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape

import fitz
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/figures/anatobind_architecture_with_losses.pdf"
OUTPUT = ROOT / "output/pdf/anatobind_architecture_explained.pdf"
PAGE_W, PAGE_H = landscape(A3)
MARGIN, GAP = 36, 22
CARD_W = (PAGE_W - 2 * MARGIN - GAP) / 2
CARD_H = 330
INK = colors.HexColor("#283442")
MUTED = colors.HexColor("#687582")
TEAL = colors.HexColor("#668B80")
AMBER = colors.HexColor("#A0885E")
BLUE = colors.HexColor("#627F9A")

pdfmetrics.registerFont(TTFont("CN", "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf"))
pdfmetrics.registerFont(TTFont("Latin", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("LatinBold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))

BODY = ParagraphStyle("body", fontName="CN", fontSize=13, leading=19,
                      textColor=INK, alignment=TA_LEFT, wordWrap="CJK")
SMALL = ParagraphStyle("small", parent=BODY, fontSize=10.5, leading=15, textColor=MUTED)
PLAIN = ParagraphStyle("plain", parent=BODY, fontSize=12.5, leading=18)
EQUATION = ParagraphStyle("equation", fontName="Latin", fontSize=14, leading=22, textColor=INK)


def module(number, title, english, status, fields, source):
    return dict(number=number, title=title, english=english, status=status,
                fields=fields, source=source)


PAGES = [
    ("数据与骨干", "先把图像、位置和标签整理一致，再提取细节与整体结构；“伪标签”指其他模型生成的参考标注。", [
        module("01", "膝侧数据引擎", "SKM-TEA data engine", "已实现", [
            ("输入", "双回波 qDESS 的混合域 k-space、线圈图、六类分割及含 tissue_id 的病灶框。"),
            ("处理", "核对分割与 H5 方位；沿 ky/kz 逆变换并作 adjoint-SENSE 线圈合并；生成噪声/欠采视图，重采样并按扫描划分五折。"),
            ("输出", "干净/退化 NIfTI、seg.nii.gz、boxes.csv、逐卷 manifest 与 splits.json。"),
            ("监督", "组织族取自标注者 tissue_id；部分内外侧通过分割重叠解析，不能视为完全独立的人工侧别标签。"),
        ], "data_engine/skmtea.py; skmtea_recon.py; splits.py"),
        module("02", "脑侧解剖伪标签", "Brain anatomy label pipeline", "已实现", [
            ("输入", "fastMRI H5 或已有脑 MRI NIfTI；数据包含不同层厚、覆盖范围与空间网格。"),
            ("处理", "依据采集头构建几何；薄层栈按规则补层；调用 SynthSeg-robust，再将离散标签回采到原生影像网格。"),
            ("输出", "与原图形状/仿射一致的解剖伪标签，以及记录成功、缺失、跳过原因的清单。"),
            ("状态", "数据管线已有；统一脑/膝本体及共同训练尚未接入。伪标签质量与左右命名仍须按数据来源说明。"),
        ], "data_engine/fastmri.py; synthseg_pipeline.py; labels.py"),
        module("03", "预处理与训练采样", "Preprocessing and crop dataset", "已实现", [
            ("输入", "当前只读膝侧第一回波干净图、解剖标签、病灶框、宿主标签和逐卷 spacing。"),
            ("处理", "数组和框统一 XYZ->ZYX；0.5/99.5 百分位裁剪后 z-score；围绕病灶选取 64x128x128 裁块，训练时加位置扰动。"),
            ("输出", "image: Bx1x64x128x128；稠密分割/存在状态；变长框、事件类别与宿主列表。"),
            ("边界", "框随裁块裁剪；未解析宿主保留为未知。当前没有任意整卷的 M_valid padding 与滑窗推理闭环。"),
        ], "train/dataset.py: normalise_volume; SkmteaArmBDataset"),
        module("04", "多尺度 3D Swin", "Hierarchical feature backbone", "已实现", [
            ("输入", "方向、强度和 spacing 已对齐的单通道 3D 裁块。"),
            ("处理", "(2,4,4) Conv3D stem；MONAI Swin 的前三个 stage，使用相对位置偏置。F1 为 stem 特征，尚未经过注意力。"),
            ("输出", "F1/F2/F3/F4 通道为 32/64/128/256；默认裁块上的空间尺寸为 32^3、16^3、8^3、4^3。"),
            ("loss / 状态", "当前由下游 loss 反传更新；完整方案的物理 3D RoPE、valid-voxel mask、更宽模型和 SSL 尚未实现。"),
        ], "model/backbone.py: Backbone"),
    ]),
    ("解剖、异常与采集 token", "token 是模型保存信息的数字表示；query 是用于寻找结构或异常的可学习查询。当前 A/U_B 已实现，S/U_Q 待实现。", [
        module("05", "解剖实体 A", "Identity-anchored anatomy queries", "已实现", [
            ("输入", "F4/F3/F2 供 query 交叉注意力使用；F1 供 mask 空间投影。"),
            ("处理", "K=6 固定身份 query，三层 cross-attention，d=128。query 与 F1 投影点积后上采样，同时预测每个结构的存在性。"),
            ("输出", "六类解剖的 mask、presence logits 和 A_i 嵌入；query 身份固定，不做 Hungarian 匹配。"),
            ("loss / 状态", "存在结构的 mask BCE+Dice，以及全部槽位的 presence BCE。当前没有独立身份 CE；跨器官固定并集本体仍待实现。"),
        ], "model/decoders.py: ADecoder; model/losses.py: a_loss"),
        module("06", "采集条件 S", "Acquisition-condition token", "待实现", [
            ("输入", "完整方案中的 F4 masked pooling 特征，以及可用的采集元数据。"),
            ("处理", "全局特征与元数据编码形成 S，预测序列、场强、方向、脂肪抑制；S 将作为关系 token 的条件输入。"),
            ("输出", "全局 S 嵌入及采集属性预测，帮助解释不同对比度下相同组织/异常的外观。"),
            ("loss / 边界", "L_S 为有标签属性的分类监督。输入元数据与监督目标重合时需避免平凡复制；当前关系 phi 未包含 S。"),
        ], "RESEARCH_PLAN.md: sections 4.4 and 4.6"),
        module("07", "生物异常 U_B", "DETR-style biological events", "已实现", [
            ("输入", "F3/F2/F1 多尺度特征，兼顾区域结构与细小异常纹理。"),
            ("处理", "M=8 个 d=128 query，经三层解码输出类别与归一化 3D 框；Hungarian 将预测事件与标注实例对应。"),
            ("输出", "半月板撕裂/软骨病变/no-object logits，中心与尺寸六参数，以及事件嵌入 U_j。"),
            ("loss", "全 query 分类 CE + 5x匹配框 L1 + 2x匹配框 GIoU；no-object 类权重 0.1。未知宿主的框仍可监督检测。"),
        ], "model/decoders.py: UBDecoder; model/losses.py: ub_loss"),
        module("08", "全局退化 U_Q", "Global degradation branch", "待实现", [
            ("输入", "F4 的全局聚合特征；训练标签来自可控干预算子的类型与施加强度。"),
            ("处理", "独立预测 motion、noise、aliasing 的存在与各自强度；多标签形式允许多种退化共存。"),
            ("输出", "每类 presence/severity 与全局退化表示；没有 3D 框、Hungarian 或主宿主，也不输出 R_Q。"),
            ("loss / 边界", "L_Q=类型 BCE+已知强度回归。不同类型强度不可混成同一物理量；零标签仅表示未额外施加退化。当前不输入关系 token。"),
        ], "RESEARCH_PLAN.md: sections 4.5, 5.3 and 6"),
    ]),
    ("显式关系与局部空间支撑", "把异常与可能的宿主逐一配对；“宿主”指异常所属的解剖结构，ROI 指为当前判断选取的局部区域。", [
        module("09", "配对几何 G", "Physical pair geometry", "部分实现", [
            ("输入", "解剖 mask、事件 box 和逐卷物理 spacing。当前取自 GT mask 与匹配后的 GT 框。"),
            ("处理", "求 mask 质心和框中心差，换算为 mm；计算位移范数及框内属于该解剖的体积比例 IoA；整个几何计算 detach。"),
            ("输出", "G_ij=(dz,dy,dx,||delta||,IoA)，形状 BxKxMx5，经 MLP 投影到 64 维。"),
            ("边界", "几何没有独立 loss。主评估需改用预测几何；GT IoA 容易形成查表捷径。方案中的显式侧别/层级通道尚未加入。"),
        ], "model/relation.py: geometry_features; model/armb.py"),
        module("10", "关系 token 与 Transformer", "Pair tokens and relational attention", "部分实现", [
            ("输入", "A_i、U_j 与 g(G_ij)；完整方案还包含 S。"),
            ("处理", "phi 拼接后映射为 d=128 的配对 token；KxM 个槽位经过两层、8-head self-attention，共享实体对间的上下文。"),
            ("输出", "每个候选异常-解剖对的关系表示 r_ij，供关系头及计划中的 E 头读取。"),
            ("loss / 边界", "当前通过宿主 CE 与关系 BCE 学习。attention 仅按 GT 解剖存在性屏蔽；完整事件门控及 S 条件仍待实现。"),
        ], "model/relation.py: RelationModule; _PairBlock"),
        module("11", "关系与主宿主输出", "Soft relations and main-host head", "已实现", [
            ("输入", "每个配对的 r_ij；none 头单独读取事件嵌入。"),
            ("处理", "一支 sigmoid 给每对关系概率；另一支在存在的 K 个解剖候选加 none 上做主宿主 softmax。代码中的 R 本身是 logits。"),
            ("输出", "R_ij 软关系与事件 j 的主宿主分布。多标签关系支持多个关联结构，主宿主给单个主要归属。"),
            ("loss / 边界", "已匹配且宿主已知的事件接受 CE/BCE；当前 in_seg 数据没有积液 none 正样本。none 表示事件确无宿主，不能代替低证据。"),
        ], "model/relation.py: h_R, h_host, h_none; losses.py"),
        module("12", "局部配对 ROI", "Shared spatial support for evidence", "待实现", [
            ("输入", "事件框 B_j、解剖 mask M_i、物理间距和有效体素 M_valid。"),
            ("处理", "保留病灶框，再加入其 8 mm 邻域内的宿主解剖，裁至有效域；不把整个器官并入。F2 池化与 E* 使用同一物理区域。"),
            ("输出", "Omega_ij 与局部特征 F_ij^local；训练固定参考 ROI 并 detach，推理仅从当前图预测的 mask/box 构造 ROI。"),
            ("边界", "8 mm 为待验证默认值。空 ROI 或无有效体素返回无评分；需报告参考/预测 ROI 差距及评分覆盖率。"),
        ], "RESEARCH_PLAN.md: sections 4.7 and 5.2"),
    ]),
    ("证据、物理干预与当前训练", "R 回答“异常属于哪里”，E 试图回答“现在看得清、能判断吗”；GT 指训练时使用的参考标注。", [
        module("13", "关系可观测性 E", "Local relation evidence score", "待实现", [
            ("输入", "关系表示 r_ij 与同一 ROI 内池化得到的 F_ij^local。"),
            ("处理", "MLP+sigmoid 预测 E_ij；参考标签 E*=exp[-局部 NRMSE/tau_E]，由配对幅值图在固定 ROI 内计算并停止梯度。"),
            ("输出", "每条候选关系的局部证据估计，语义与 R 的关系概率不同；高 E 也可能对应可清楚判断的负关系。"),
            ("loss / 边界", "L_obs 回归 E*，L_rank 按同一干预族的实测 E* 排序。保真度是代理，不能直接当作已校准的正确概率。"),
        ], "RESEARCH_PLAN.md: sections 4.7, 5.2 and 5.3"),
        module("14", "受控 k-space 干预", "Perturbation and reconstruction", "部分实现", [
            ("输入", "同一扫描的原始多线圈数据、线圈图和采集/采样掩膜。"),
            ("处理", "现有噪声只加在采集支撑上，幅度按线圈数归一；Poisson 掩膜嵌入 ky/kz 网格后欠采并重建。运动仍待实现。"),
            ("输出", "同病例的 X0、Xq1、Xq2、Xq3 及已知类型/强度。当前已有整卷 NRMSE，尚无关系局部 E*。"),
            ("边界", "完整配对训练要求四视图共享网络和实例身份。需核对 target 与 adjoint-SENSE 的重建/强度差异，避免把差异误作证据损失。"),
        ], "data_engine/skmtea_recon.py; skmtea.py: export_scan"),
        module("15", "匹配与关系监督路由", "Instance matching and target routing", "部分实现", [
            ("输入", "异常预测的类别/框，GT 事件实例、host_label 与解剖存在状态。"),
            ("处理", "Hungarian 代价为 -P(class)+5xL1-2xGIoU；匹配后按 host_label 构造宿主 CE 和配对 BCE 目标。"),
            ("输出", "匹配索引、有效监督掩膜及关系正/负标签。未匹配或宿主未知者不接受关系项，但已标注框仍训练检测。"),
            ("规划", "配对视图必须匹配同一参考实例，不能用相同 query 编号假定同一病灶；外部 E* 门控和 hard-negative loss 尚未实现。"),
        ], "model/losses.py: hungarian_match; build_relation_targets"),
        module("16", "当前总 loss 与训练循环", "Executable Arm-B objective", "已实现", [
            ("组成", "L_ArmB = L_mask + L_presence + L_cls + 5L1 + 2L_GIoU + L_host_CE + L_rel_BCE。L_mask 为 BCE+Dice。"),
            ("归约", "mask 只算存在结构；框项只算匹配实例，L1 对六坐标求和后除以实例数；关系只算有效已知宿主。"),
            ("更新", "AdamW，初始学习率 3e-4、weight decay 0.05，余弦调度；梯度裁剪 1.0；CUDA 上使用 BF16。"),
            ("状态", "当前仅为干净膝裁块 smoke training，仍依赖 GT 几何。已有 34 项相关测试通过，但不能据此证明绑定或泛化优势。"),
        ], "model/losses.py: total_loss; scripts/train_armb_minimal.py"),
    ]),
]


PLAIN_EXPLANATIONS = {
    "01": "像整理训练教材：把同一次扫描的图像、结构地图和病灶标记对齐，确保位置没有张冠李戴。",
    "02": "缺少脑解剖标注时，先请现成分割模型画参考地图。地图可能画错，所以叫“伪标签”，还需核验。",
    "03": "先统一方向和亮度，再裁出病灶附近一小块来训练；标签和框也一起裁，避免图像与答案错位。",
    "04": "像用放大镜看纹理、退远一点看结构：不同尺度各留一份信息，后续模块按需要取用。",
    "05": "给六种解剖各设一个固定名额：每个名额负责找出自己的结构、画出范围，并判断它是否在图中。",
    "06": "先弄清“这张 MRI 是怎么拍的”，再理解亮暗变化；同一组织在不同扫描序列里，外观可以很不一样。",
    "07": "负责找出并框住可疑异常，例如一处撕裂；找到位置之后，还需要关系模块判断它具体属于哪块结构。",
    "08": "回答“图像受了什么干扰、干扰多强”，例如运动或噪声。这些采集干扰不等同于新的生物病变。",
    "09": "给每个候选配对列位置线索：相距多少毫米、病灶框内有多少属于该结构，供关系模型参考。",
    "10": "把“这个异常是否属于这块解剖”单独做成一条待判断记录，再结合其他候选配对的上下文。",
    "11": "例如判断“这处撕裂主要属于内侧还是外侧半月板”。软关系可关联多个结构，主宿主再选择主要归属。",
    "12": "把放大镜对准病灶及附近相关组织；目的是避免整只膝盖的大量正常区域淹没局部证据的变化。",
    "13": "R 像“归属判断”，E 像“这块图像够不够支撑判断”。看不清时应表达不确定；E 的保真度代理仍需验证。",
    "14": "保持病例不变，有控制地加入噪声或减少采样，再看归属判断怎样变化；运动版本仍待实现。",
    "15": "多个异常同时出现时，先把预测框与标注实例一一配对，再给答案计分；不能把 query 编号当作病灶身份。",
    "16": "loss 像分项扣分表：分割、检测、归属答错各自扣分，训练据此改参数；总分下降仍需独立测试验证。",
}


def paragraph(c, content, x, top, width, style=BODY):
    p = Paragraph(content, style)
    _, height = p.wrap(width, 1000)
    p.drawOn(c, x, top - height)
    return top - height


def header(c, title, subtitle, page):
    c.setFillColor(INK)
    c.setFont("CN", 25)
    c.drawString(MARGIN, PAGE_H - 48, title)
    c.setFont("LatinBold", 10)
    c.setFillColor(BLUE)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 43, "AnatoBind-MRI / MODULE GUIDE")
    paragraph(c, escape(subtitle), MARGIN, PAGE_H - 65, PAGE_W - 2 * MARGIN, SMALL)
    c.setStrokeColor(colors.HexColor("#DCE1E5"))
    c.line(MARGIN, 37, PAGE_W - MARGIN, 37)
    c.setFont("Latin", 8.5)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, 22, "Code: f812043  |  Plan: v2.1  |  2026-09-08  |  implementation status explicitly marked")
    c.drawRightString(PAGE_W - MARGIN, 22, f"{page} / 6")
    c.bookmarkPage(f"page{page}")
    c.addOutlineEntry(title, f"page{page}", level=0)


def card_frame(c, x, y, title, english, status, number=None):
    color = TEAL if status == "已实现" else (AMBER if status == "待实现" else BLUE)
    c.setFillColor(colors.HexColor("#FCFCFD"))
    c.setStrokeColor(color)
    c.setLineWidth(0.8)
    c.setDash(4, 3) if status == "待实现" else c.setDash()
    c.roundRect(x, y, CARD_W, CARD_H, 8, fill=1, stroke=1)
    c.setDash()
    if number:
        c.setFont("LatinBold", 12)
        c.setFillColor(color)
        c.drawString(x + 17, y + CARD_H - 31, number)
    c.setFont("CN", 19)
    c.setFillColor(INK)
    c.drawString(x + (49 if number else 17), y + CARD_H - 34, title)
    c.setFont("CN", 11)
    c.setFillColor(color)
    c.drawRightString(x + CARD_W - 17, y + CARD_H - 31, status)
    c.setFont("Latin", 10)
    c.setFillColor(MUTED)
    c.drawString(x + 17, y + CARD_H - 56, english)
    return y + CARD_H - 75


def draw_module(c, item, x, y):
    top = card_frame(c, x, y, item["title"], item["english"], item["status"], item["number"])
    for label, content in item["fields"]:
        text = f'<font color="#627F9A">{escape(label)}：</font>{escape(content)}'
        top = paragraph(c, text, x + 17, top, CARD_W - 34) - 7
    plain_text = '<font color="#52736A">通俗理解：</font>' + escape(PLAIN_EXPLANATIONS[item["number"]])
    note = Paragraph(plain_text, PLAIN)
    _, note_height = note.wrap(CARD_W - 46, 1000)
    if top - note_height < y + 34:
        raise ValueError(f"Module {item['number']} plain-language note overlaps its source")
    c.setFillColor(colors.HexColor("#EEF4F1"))
    c.roundRect(x + 12, top - note_height - 4, CARD_W - 24, note_height + 8,
                5, fill=1, stroke=0)
    note.drawOn(c, x + 23, top - note_height)
    c.setFont("Mono", 7.8)
    c.setFillColor(MUTED)
    c.drawString(x + 17, y + 19, item["source"])


def positions():
    upper = PAGE_H - 103 - CARD_H
    lower = upper - GAP - CARD_H
    return [(MARGIN, upper), (MARGIN + CARD_W + GAP, upper),
            (MARGIN, lower), (MARGIN + CARD_W + GAP, lower)]


def summary_page(c):
    header(c, "完整训练目标与 novelty", "loss 是训练时的错误评分；novelty 指相对已有工作的新增贡献。以下阶段属于完整方案，尚待实现和验证。", 6)
    xy = positions()
    x, y = xy[0]
    top = card_frame(c, x, y, "四阶段 loss 开关", "Stage-specific objectives", "待实现")
    for label, formula in [
            ("I：先学会看 MRI", "L<sub>I</sub> = L<sub>MIM</sub> + L<sub>contrast</sub>"),
            ("II：再认结构与异常", "L<sub>II</sub> = L<sub>A</sub> + L<sub>S</sub> + L<sub>UB</sub> + L<sub>Q</sub>"),
            ("III：学习异常归属", "L<sub>III</sub> = λ<sub>R</sub>L<sub>rel</sub><super>0</super>"),
            ("IV：在退化下判断证据", "L<sub>IV</sub> = λ<sub>R</sub>L<sub>rel</sub> + λ<sub>I</sub>L<sub>int</sub>")]:
        top = paragraph(c, label, x + 17, top, 170) - 4
        top = paragraph(c, formula, x + 35, top, CARD_W - 52, EQUATION) - 7
    paragraph(c, "阶段 IV 关闭 L_sem；L_Q 只计一次，避免同一项重复扣分。", x + 17, y + 36, CARD_W - 34, SMALL)

    x, y = xy[1]
    top = card_frame(c, x, y, "证据门控的干预目标", "Evidence-gated paired learning", "待实现")
    for formula in [
            "L<sub>rel</sub> = L<sub>rel</sub><super>0</super> + mean<sub>q&gt;0</sub>(L<sub>rel</sub><super>q,w</super>)",
            "L<sub>int</sub> = L<sub>cons</sub> + L<sub>Q</sub> + λ<sub>E</sub>L<sub>obs</sub> + λ<sub>m</sub>L<sub>rank</sub>"]:
        top = paragraph(c, formula, x + 17, top, CARD_W - 34, EQUATION) - 9
    for text in [
            "高 E*（保真度代理较高）：暂按证据较充分处理，对关系 BCE、整条宿主 CE 和硬负样本加权，并对齐停止梯度的干净预测。",
            "低 E*：原病灶没有因噪声而消失，但图像可能已不足以判断归属。因此关闭确定性监督与 KL，允许更分散的预测；不改成负类或 none。未知/缺失 E* 也不奖励不确定。",
            "L_obs 让证据评分接近外部参考；L_rank 要求同一 ROI 中实测证据更好的视图排得更高。参考是实测 E*，不是名义退化强度；具体回归/排序形式尚未固化。"]:
        top = paragraph(c, escape(text), x + 17, top, CARD_W - 34) - 8
    if top < y + 17:
        raise ValueError("Intervention summary overflows")

    x, y = xy[2]
    top = card_frame(c, x, y, "三个候选组合贡献", "Novelty requires controlled comparisons", "待验证")
    for text in [
            "N1 局部证据：除了“病灶属于哪里”，还想判断“此处图像够不够支撑归属判断”。每条关系的 E 用局部保真度监督，需胜过置信度、局部质量与失效预测对照。",
            "N2 选择性一致：给同一病例施加可控退化，看得清时尽量维持归属，看不清时允许不确定。需与使用相同退化样本的普通增广比较，证明配对目标有额外价值。",
            "N3 组合泛化：学过不同解剖、异常和退化之后，能否处理没有一起见过的组合，例如训练未见的“膝部+运动退化”。这是要检验的能力，尚无实验结论。",
            "当前最强贡献仍在设计阶段。Swin、关系 Transformer 或额外 confidence head 本身不足以支持首创表述。"]:
        top = paragraph(c, escape(text), x + 17, top, CARD_W - 34) - 9
    if top < y + 17:
        raise ValueError("Novelty summary overflows")

    x, y = xy[3]
    top = card_frame(c, x, y, "证据边界与阅读索引", "Verification and primary references", "已核对")
    top = paragraph(c, "模型源码、研究方案和图中 loss 已逐项对应；已有 34 项相关测试通过。GT 几何下的 smoke-run 指标不是 M1 性能证据。", x + 17, top, CARD_W - 34) - 12
    top = paragraph(c, "阅读顺序：第 1 页看数据流；第 2-5 页看模块；本页核对训练开关与贡献边界。R 是关系预测，E 是证据代理，none、no-object、未知与低 E 不能互换。", x + 17, top, CARD_W - 34) - 12
    for title, url in [
            ("Swin UNETR / medical self-supervision", "https://arxiv.org/abs/2111.14791"),
            ("MRI-CORE / MRI representation learning", "https://arxiv.org/abs/2506.12186"),
            ("SGTR / relation Transformer", "https://arxiv.org/abs/2112.12970"),
            ("Chest ImaGenome / anatomy-finding relations", "https://arxiv.org/abs/2108.00316"),
            ("ConfidNet / learned failure prediction", "https://github.com/valeoai/ConfidNet")]:
        link = f'<font name="Latin" size="10"><link href="{url}" color="#627F9A">{escape(title)}</link></font>'
        top = paragraph(c, link, x + 17, top, CARD_W - 34, SMALL) - 6
    paragraph(c, "详细依据：docs/architecture_and_novelty_2026-09-08.md", x + 17, y + 29, CARD_W - 34, SMALL)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path("/tmp/pdfs")
    tmp_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="anatobind-guide-", dir=tmp_root) as temp:
        base = Path(temp) / "guide_base.pdf"
        c = canvas.Canvas(str(base), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
        c.setTitle("AnatoBind-MRI: 整体架构与子模块说明")
        c.setAuthor("AnatoBind-MRI project")
        c.setSubject("Architecture, module inputs/outputs, loss functions and implementation status")
        header(c, "整体架构图与 loss", "先认结构/异常，再判断归属，最后估计证据是否充足；mask 是结构轮廓，box 是异常框。实线框为已实现，虚线框为待实现。", 1)
        c.showPage()
        for page, (title, subtitle, modules) in enumerate(PAGES, start=2):
            header(c, title, subtitle, page)
            for item, (x, y) in zip(modules, positions()):
                draw_module(c, item, x, y)
            c.showPage()
        summary_page(c)
        c.showPage()
        c.save()
        with fitz.open(base) as guide, fitz.open(SOURCE) as figure:
            guide[0].show_pdf_page(fitz.Rect(26, 98, PAGE_W - 26, PAGE_H - 46), figure, 0)
            guide.set_page_labels([{"startpage": 0, "prefix": "", "style": "D", "firstpagenum": 1}])
            guide.save(OUTPUT, garbage=4, deflate=True)
    print(OUTPUT)
    with fitz.open(OUTPUT) as check:
        assert len(check) == 6
        for number, page in enumerate(check, start=1):
            text = page.get_text()
            assert len(text.strip()) > 100, f"Empty page {number}"
            if 2 <= number <= 5:
                assert text.count("通俗理解") == 4, f"Missing module explanations on page {number}"
            print(f"page {number}: {len(text)} extractable characters")
        print(f"bookmarks: {len(check.get_toc())}")


if __name__ == "__main__":
    main()
