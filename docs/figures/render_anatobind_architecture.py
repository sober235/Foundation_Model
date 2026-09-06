"""Render plan-v2.1 with the four accepted architecture decisions.

Sources: RESEARCH_PLAN.md sections 4, 5 and 8; architecture review 2026-09-06.
This figure describes a proposal, not an implemented or validated model.
Python schematic: forward architecture, paired supervision, and accepted interfaces.
Exports: editable-text SVG and a PNG preview. No measured results are depicted.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as PlotPath


ROOT = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H = 2400, 1780
C = dict(ink="#17263F", muted="#596B83", line="#8190A5", teal="#158879",
         blue="#3578C4", coral="#C96657", purple="#7854AD", amber="#B47B21")
matplotlib.rcParams["svg.fonttype"] = "none"
fig, ax = plt.subplots(figsize=(24, 17.8))
fig.patch.set_facecolor("white")
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


def text(x, y, s, size=12, color=None, bold=False, ha="left", va="center"):
    return ax.text(x, y, s, fontproperties=FontProperties(fname=BOLD if bold else FONT,
                   family="Noto Sans CJK JP", weight="bold" if bold else "normal",
                   size=size), color=color or C["ink"], ha=ha, va=va, zorder=5,
                   linespacing=1.5)


def box(x, y, w, h, fill="white", edge="#D9E1EC", radius=16, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={radius}",
                 facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))


def arrow(points, color=None, dashed=False, width=1.8):
    path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=14,
                 color=color or C["line"], linewidth=width,
                 linestyle=(0, (5, 4)) if dashed else "solid", zorder=3,
                 capstyle="round", joinstyle="round"))


def label(x, y, s, color=None, size=10):
    t = text(x, y, s, size, color or C["muted"], ha="center")
    t.set_bbox(dict(facecolor="white", edgecolor="none", pad=2))


text(55, 57, "AnatoBind-MRI", 30, bold=True)
text(475, 59, "整体模型架构", 26, bold=True)
text(57, 108, "plan-v2 分支 · 方案 v2.1｜已采纳接口修订；完整模型尚未实现或验证", 13, C["muted"])
box(35, 145, 2330, 940, edge="#E2E8F1", radius=22)
text(65, 179, "A  前向推理", 17, bold=True)
text(2325, 179, "结构化输出：{A, S, U_B, U_Q, R_B, E}", 13, C["muted"], ha="right")

# Input and variable-size feature hierarchy.
box(65, 365, 215, 250, "#F0F4FA")
text(172, 398, "MRI 体数据 X", 16, bold=True, ha="center")
for i, grey in enumerate(("#CDD5DF", "#A4B0C0", "#71829A", "#43556F")):
    ax.add_patch(Rectangle((118 + i * 9, 433 - i * 6), 81, 66,
                           facecolor=grey, edgecolor="white", lw=1, zorder=4))
text(172, 516, "D × H × W", 12, ha="center")
text(172, 555, "方向 / 强度归一", 12, ha="center")
text(172, 586, "Padding + M_valid", 11, ha="center")
arrow([(280, 490), (335, 490)])

box(335, 235, 365, 475, "#EDF3FC", "#A6BDDA")
text(517, 270, "变尺寸 3D Swin", 19, C["blue"], True, "center")
text(517, 312, "Conv3D stem · (2, 4, 4)", 12, ha="center")
for i, (name, channel) in enumerate((("F1", 64), ("F2", 128), ("F3", 256), ("F4", 512))):
    yy = 345 + i * 70
    box(372, yy, 291, 49, ("#E2ECFA", "#CFDFF5", "#BBD1F0", "#A8C3E8")[i], "#9CB6D8", 8)
    text(398, yy + 24, f"{name}   ·   {channel} channels", 13, bold=True)
    if i < 3:
        arrow([(517, yy + 49), (517, yy + 70)], C["blue"], width=1.3)
text(517, 654, "物理坐标 3D RoPE", 12, ha="center")
text(517, 685, "+ 局部归一坐标", 12, ha="center")

# Parallel decoder branches; labels define the selected feature levels.
arrow([(700, 440), (745, 440), (745, 297), (795, 297)], C["teal"])
arrow([(663, 579), (715, 579), (715, 535), (795, 535)], C["blue"])
arrow([(700, 550), (735, 550), (735, 745), (795, 745)], C["coral"])
label(747, 275, "F2–F4", C["teal"])
label(749, 517, "F4", C["blue"])
label(747, 790, "F1–F3", C["coral"])

box(795, 218, 350, 185, "#EAF7F3", "#8ABFB3")
text(818, 251, "A  解剖实体解码器", 17, C["teal"], True)
text(818, 291, "跨器官固定 union ontology queries", 10.7)
text(818, 327, "身份 / no-object · mask · A_i", 12)
text(818, 366, "存在性门控 · 无 Hungarian", 11)

box(885, 421, 190, 30, "#F0F5FC", "#C9D9EE", 8)
text(980, 436, "可选元数据", 10, C["blue"], ha="center")
arrow([(980, 451), (980, 465)], C["blue"], width=1.3)
box(795, 465, 350, 140, "#EDF4FD", "#A0BCDE")
text(818, 497, "S  采集条件 token", 17, C["blue"], True)
text(818, 538, "Masked pooling + 元数据条件", 11)
text(818, 578, "序列 · 场强 · 方向 · 脂肪抑制", 11)

box(795, 658, 350, 174, "#FFF0EC", "#DDB0A5")
text(818, 690, "U_B  生物异常解码器", 16, C["coral"], True)
text(818, 730, "DETR · 20 queries · Hungarian", 11)
text(818, 766, "类型 / no-object · 3D 框 · U_j^B", 11)
text(818, 806, "存在性门控后参与关系与主宿主选择", 10.5)

# U_Q is a separate global output; it is not a relation-token input.
arrow([(372, 579), (310, 579), (310, 949), (335, 949)], C["coral"])
label(310, 815, "F4", C["coral"])
box(335, 876, 715, 158, "#FFF0EC", "#DDB0A5")
text(361, 910, "U_Q  全局采集退化头 · 独立输出", 17, C["coral"], True)
text(361, 952, "Masked pooling → 多标签 presence / type + 每类 severity", 12)
text(361, 991, "motion / noise / aliasing；无 bbox / Hungarian / R_Q", 11)

# Anatomy/event/acquisition embeddings meet in the relation module.
arrow([(1145, 297), (1215, 297), (1215, 452), (1270, 452)], C["teal"])
arrow([(1145, 535), (1270, 535)], C["blue"])
arrow([(1145, 720), (1215, 720), (1215, 603), (1270, 603)], C["coral"])
label(1215, 431, "A_i", C["teal"])
label(1205, 516, "S", C["blue"])
label(1215, 626, "U_j^B", C["coral"])
box(1270, 292, 480, 488, "#F4EFFB", "#B7A2D3")
text(1510, 331, "显式关系 token", 19, C["purple"], True, "center")
text(1510, 375, "G_ij ← A mask + U_B box", 13, ha="center")
text(1510, 409, "位移 · IoA · 侧别 · 层级", 12, ha="center")
text(1510, 462, "r^0_ij = φ[A_i; U_j^B; S; g(G_ij)]", 12, ha="center")
for i in range(8):
    box(1372 + i * 34, 504, 24, 28, "#BAA3D8", "#9E83C1", 4, 0.6)
text(1510, 564, "K_union × 20 候选对 · 每对 256 维", 12, ha="center")
text(1510, 601, "缺席 A / U_B 槽位屏蔽", 11, C["purple"], ha="center")
arrow([(1510, 621), (1510, 645)], C["purple"])
box(1342, 649, 336, 64, "#E5D9F4", "#AC8ECD", 10)
text(1510, 681, "关系 Transformer · 2 层", 14, C["purple"], True, "center")
text(1510, 750, "关系表示 r_ij", 15, C["purple"], True, "center")

# Both output heads consume r_ij; E additionally consumes local evidence.
arrow([(1638, 750), (1818, 750), (1818, 422), (1885, 422)], C["purple"])
arrow([(1818, 750), (1885, 750)], C["purple"])
box(1885, 320, 450, 210, "#F4EFFB", "#B7A2D3")
text(1911, 359, "R_B  生物异常关系绑定", 19, C["purple"], True)
text(1911, 407, "异常属于 / 影响哪个解剖？", 12)
text(1911, 454, "sigmoid：软关系", 12)
text(1911, 493, "softmax(K_union + none)：有效主宿主", 11)

box(1885, 620, 450, 220, "#FFF7E7", "#D6B97B")
text(1911, 659, "E  关系可观测性", 19, C["amber"], True)
text(1911, 705, "当前图像证据够不够？", 12)
text(1911, 748, "[r_ij ; F_ij^local] → E_ij", 13)
text(1911, 800, "关系级输出 · 局部保真度代理监督", 11, C["muted"])

# Explicit pair-ROI construction uses BOTH the lesion box and anatomy mask.
# The anatomy contribution is clipped to the lesion neighborhood, not a whole organ.
arrow([(1145, 327), (1175, 327), (1175, 887), (1370, 887), (1370, 914)], C["teal"], width=1.3)
arrow([(1145, 766), (1215, 766), (1215, 860), (1610, 860), (1610, 914)], C["coral"], width=1.3)
label(1295, 887, "A mask", C["teal"], 10)
label(1496, 860, "病灶 3D 框", C["coral"], 10)
box(1190, 914, 530, 120, "#FFF8ED", "#DFC89E", 10)
text(1455, 940, "局部配对区域 Ω_ij", 14, C["amber"], True, "center")
text(1455, 976, "病灶框 ∪（A mask ∩ 病灶 8 mm 邻域）", 11, ha="center")
text(1455, 1009, "裁剪至 M_valid · 8 mm 为待验证默认值", 10, C["muted"], ha="center")
arrow([(1720, 970), (1775, 970)], C["amber"])
box(1775, 914, 310, 120, "#FFF8ED", "#DFC89E", 10)
text(1930, 942, "F2 + Ω_ij", 14, C["amber"], True, "center")
text(1930, 980, "Masked mean → F_ij^local", 11, ha="center")
text(1930, 1012, "输入 / E* 监督使用同一区域", 10, C["muted"], ha="center")
arrow([(663, 439), (685, 439), (685, 847), (1090, 847), (1090, 1056), (1930, 1056), (1930, 1034)], C["amber"])
label(1124, 1056, "F2 局部证据旁路", C["amber"], 10)
arrow([(2085, 970), (2220, 970), (2220, 840)], C["amber"])

# Paired controlled interventions are training-only, with one shared network.
box(35, 1110, 2330, 373, edge="#E2E8F1", radius=22)
text(65, 1144, "B  仅训练：受控物理干预", 17, bold=True)
box(65, 1185, 240, 127, "#F0F4FA")
text(185, 1223, "多线圈 raw k-space", 13, bold=True, ha="center")
text(185, 1271, "同一病例", 12, C["muted"], ha="center")
arrow([(305, 1248), (350, 1248)])
box(350, 1185, 320, 127, "#FFF0EC", "#DDB0A5")
text(510, 1223, "受控退化 + 重建", 15, C["coral"], True, "center")
text(510, 1271, "运动 / 噪声 / 欠采样", 12, ha="center")
arrow([(670, 1248), (715, 1248)])
box(715, 1185, 433, 127, "#F0F4FA")
text(931, 1223, "X^0   X^q1   X^q2   X^q3", 14, bold=True, ha="center")
text(931, 1271, "同一 raw k-space · 不同干预强度", 11, C["muted"], ha="center")
arrow([(1148, 1248), (1193, 1248)])
box(1193, 1185, 299, 127, "#EDF3FC", "#A6BDDA")
text(1342, 1223, "同一网络", 17, C["blue"], True, "center")
text(1342, 1271, "四视图 · 参数共享", 12, ha="center")
arrow([(1492, 1248), (1540, 1248)])
box(1540, 1185, 795, 168, "#FFF8ED", "#DFC89E")
text(1564, 1214, "保留干净视图 L_rel；退化视图 L_rel 由外部 E* 门控", 12)
text(1564, 1250, "E* > τ：确定性关系监督有效", 12)
text(1564, 1286, "E* ≤ τ：关闭确定性关系监督；鼓励不确定性", 12)
text(1564, 1322, "U_Q 学习类型 / 强度；E 回归 + 排序", 12)

box(65, 1353, 1083, 72, "#FFF8ED", "#DFC89E", 10)
text(606, 1376, "干净 / 退化图对 + 固定 Ω_ij → 外部 E*（detach）", 12, C["amber"], ha="center")
text(606, 1407, "各视图共用参考 mask / box；ROI 坐标与 M_valid 同步固定并 detach", 10, C["muted"], ha="center")
arrow([(931, 1312), (931, 1353)], C["amber"], True)
arrow([(1148, 1390), (1516, 1390), (1516, 1335), (1540, 1335)], C["amber"], True)
label(1319, 1371, "门控退化关系监督 + E 监督", C["amber"], 10)
text(65, 1457, "推理与评估：仅当前视图预测 ROI；无干净参考。缺失 / 空 ROI 不输出 E，需报告覆盖率与参考 / 预测 ROI 差距。", 11, C["muted"])

text(65, 1517, "训练顺序", 13, bold=True)
stages = [(65, "I  自监督骨干"), (645, "II  实体 / 事件感知"),
          (1225, "III  关系绑定"), (1805, "IV  干预 / 可观测性")]
for i, (xx, title) in enumerate(stages):
    box(xx, 1543, 530, 54, "#EAF0F8", "#CED8E5", 10)
    text(xx + 265, 1570, title, 14, bold=True, ha="center")
    if i < 3:
        arrow([(xx + 530, 1570), (xx + 578, 1570)])

text(65, 1650, "已采纳", 12, C["teal"], True)
text(209, 1650, "1A 退化独立；2A 统一本体 + 存在性门控；3B E* 门控关系监督；4B 输入 / 监督使用同一局部配对区域。", 11)
text(65, 1690, "验证边界", 12, C["amber"], True)
text(209, 1690, "E* 为保真度代理；需验证预测 ROI 下的失效预测能力。8 mm 邻域与门控阈值仍需验证。", 11)
text(65, 1739, "实线：前向数据流    虚线：训练监督    ·    来源：RESEARCH_PLAN.md v2.1 / 2026-09-06 架构评审", 10, C["muted"])
text(2335, 1739, "设计示意 · 非实验结果", 10, C["muted"], ha="right")

for suffix in ("png", "svg"):
    dest = ROOT / f"anatobind_plan_v2_1_architecture.{suffix}"
    fig.savefig(dest, dpi=180, facecolor=fig.get_facecolor())
    print(dest)
plt.close(fig)
