"""One code-grounded AnatoBind-MRI overview, including implemented/planned losses.

Run from any directory with a Python environment containing Matplotlib.
Solid outlines show the current minimal core; dashed outlines show planned
additions. This is an architecture review artifact, not an experimental result.
Sources and exact mathematical conventions: ../architecture_and_novelty_2026-09-08.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath


OUT = Path(__file__).resolve().parent
STEM = "anatobind_architecture_with_losses"
W, H = 2200, 1680
plt.style.use(OUT / "anatobind_academic.mplstyle")
C = dict(ink="#283442", muted="#65717E", line="#8A96A0",
         blue="#627F9A", teal="#668B80", purple="#867797", amber="#A0885E",
         coral="#AD8275", pale="#F7F8FA")
fig, ax = plt.subplots(figsize=(22, 16.8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")


def txt(x, y, value, size=13, color=None, bold=False, ha="left"):
    return ax.text(x, y, value, fontsize=size, color=color or C["ink"],
                   fontweight="bold" if bold else "normal", ha=ha,
                   va="center", linespacing=1.4, zorder=5)


def box(x, y, w, h, fill="white", edge=None, planned=False, radius=10):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge or "#DCE1E5", linewidth=1.3,
        linestyle=(0, (5, 4)) if planned else "solid", zorder=2))


def arrow(points, color=None, planned=False, dotted=False):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle="-|>", mutation_scale=13, linewidth=1.5,
        color=color or C["line"],
        linestyle=":" if dotted else ((0, (5, 4)) if planned else "solid"),
        capstyle="round", joinstyle="round", zorder=3))


def label(x, y, value, color=None, size=12):
    t = txt(x, y, value, size=size, color=color or C["muted"], ha="center")
    t.set_bbox(dict(facecolor="white", edgecolor="none", pad=1.5))


txt(50, 48, "AnatoBind-MRI", 30, bold=True)
txt(460, 49, "Architecture and loss functions", 25, bold=True)
txt(52, 94, "foundation_model  /  code snapshot f812043  /  research plan v2.1  /  08 Sep 2026", 12, C["muted"])
box(1300, 79, 36, 25, edge=C["blue"], radius=4)
txt(1350, 93, "Implemented core", 12)
box(1620, 79, 36, 25, edge=C["amber"], planned=True, radius=4)
txt(1670, 93, "Planned addition", 12)
txt(1970, 93, "Dotted: supervision", 11, C["muted"])

# A. The executable core and the extensions needed for the full research model.
box(30, 130, 2140, 685)
txt(55, 164, "A   MRI → entities / events → relations → local evidence", 19, bold=True)
txt(55, 207, "DATA", 11, C["muted"], True)
txt(360, 207, "MULTISCALE ENCODER", 11, C["muted"], True)
txt(775, 207, "STRUCTURED TOKENS", 11, C["muted"], True)
txt(1200, 207, "EXPLICIT PAIRS", 11, C["muted"], True)
txt(1700, 207, "RELATION OUTPUTS", 11, C["muted"], True)

box(55, 280, 245, 330, C["pale"])
txt(177, 312, "3D MRI volume", 17, bold=True, ha="center")
for i, shade in enumerate(("#D8DFE5", "#BFCBD5", "#9AABBA", "#72899D")):
    ax.add_patch(Rectangle((117 + 10*i, 353 - 7*i), 85, 63,
                           facecolor=shade, edgecolor="white", linewidth=1, zorder=4))
txt(177, 450, r"$X\in\mathbb{R}^{1\times D\times H\times W}$", 15, ha="center")
txt(177, 494, "Align axes + spacing", 12, ha="center")
txt(177, 523, "Clip / z-score → crop", 12, ha="center")
txt(177, 558, "SKM-TEA: masks, boxes", 11, ha="center")
txt(177, 584, "host_label; 5 scan folds", 11, ha="center")
arrow([(300, 445), (360, 445)])
box(55, 645, 245, 124, C["pale"])
txt(177, 671, "Brain data engine", 13, bold=True, ha="center")
txt(177, 700, "fastMRI / NIfTI → SynthSeg", 10, ha="center")
txt(177, 727, "Native-grid anatomy labels", 10, ha="center")
txt(177, 752, "Model integration: planned", 10, C["muted"], ha="center")
arrow([(177, 645), (177, 610)], planned=True)

box(360, 237, 315, 406, "#EFF3F6", C["blue"])
txt(517, 270, "3D Swin backbone", 18, C["blue"], True, "center")
txt(517, 308, "MONAI • relative position bias", 11, ha="center")
txt(517, 340, "Conv3D stem (2, 4, 4)", 13, ha="center")
for i, (name, channels, stride) in enumerate([
        ("F1", 32, "2 / 4 / 4"), ("F2", 64, "4 / 8 / 8"),
        ("F3", 128, "8 / 16 / 16"), ("F4", 256, "16 / 32 / 32")]):
    y = 370 + i * 57
    box(385, y, 265, 44, ["#E8EEF2", "#DDE6ED", "#CEDBE5", "#BFCEDE"][i], radius=5)
    txt(402, y + 22, f"{name}   C={channels}", 13, bold=True)
    txt(636, y + 22, stride, 11, C["muted"], ha="right")
    if i < 3:
        arrow([(517, y + 44), (517, y + 57)], C["blue"])
txt(517, 610, "F1 = stem map; F2–F4 = stages", 11, ha="center")

arrow([(675, 420), (720, 420), (720, 297), (775, 297)], C["teal"])
arrow([(720, 420), (720, 601), (775, 601)], C["coral"])
arrow([(720, 450), (775, 450)], C["blue"], planned=True)
label(729, 277, "F2–F4", C["teal"], 11)
label(729, 625, "F1–F3", C["coral"], 11)
label(745, 429, "F4", C["blue"], 11)

box(775, 234, 340, 145, "#F0F5F2", C["teal"])
txt(798, 264, r"$A$  Anatomy decoder", 17, C["teal"], True)
txt(798, 298, "K = 6 fixed identity queries", 13)
txt(798, 330, "Mask + presence + embedding", 12)
txt(798, 359, "F1 mask projection; no matching", 11)

box(775, 410, 340, 84, "#EFF3F6", C["blue"], planned=True)
txt(798, 437, r"$S$  Acquisition conditions", 16, C["blue"], True)
txt(798, 469, "F4 pooling + available metadata", 11)

box(775, 531, 340, 140, "#F7F1EE", C["coral"])
txt(798, 562, r"$U_B$  Biological events", 16, C["coral"], True)
txt(798, 598, "M = 8 queries • d = 128", 13)
txt(798, 630, "Class + 3D box + embedding", 12)
txt(798, 655, "DETR-style Hungarian matching", 11)

arrow([(1115, 297), (1160, 297), (1160, 412), (1200, 412)], C["teal"])
arrow([(1115, 601), (1160, 601), (1160, 552), (1200, 552)], C["coral"])
arrow([(1115, 452), (1200, 452)], C["blue"], planned=True)
label(1160, 389, r"$A_i$", C["teal"])
label(1160, 578, r"$U_j$", C["coral"])

box(1200, 300, 400, 336, "#F3F0F6", C["purple"])
txt(1400, 333, "Relation module", 18, C["purple"], True, "center")
txt(1400, 373, r"$G_{ij}=(\Delta_{mm},\|\Delta\|,\mathrm{IoA})$", 16, ha="center")
txt(1400, 419, r"$r^0_{ij}=\phi[A_i;U_j;g(G_{ij})]$", 16, ha="center")
txt(1400, 454, "Full plan: append S", 11, C["blue"], ha="center")
for i in range(8):
    box(1280 + i * 30, 482, 21, 23, "#C6BCD2", C["purple"], radius=3)
txt(1400, 529, "K × M pair tokens", 12, ha="center")
txt(1400, 565, "2-layer relation Transformer", 14, bold=True, ha="center")
txt(1400, 601, "Anatomy-presence attention gate", 11, ha="center")

arrow([(1600, 420), (1650, 420), (1650, 371), (1700, 371)], C["purple"])
box(1700, 300, 435, 150, "#F3F0F6", C["purple"])
txt(1726, 332, r"$R_B$  Abnormality–anatomy relation", 14, C["purple"], True)
txt(1726, 374, r"$R_{ij}=\sigma(h_R(r_{ij}))$", 17)
txt(1726, 414, "Host softmax: present anatomy + none", 12)

arrow([(1600, 570), (1700, 570)], C["amber"], planned=True)
box(1700, 514, 435, 157, "#F8F4EC", C["amber"], planned=True)
txt(1726, 547, r"$E$  Local evidence / observability", 14, C["amber"], True)
txt(1726, 589, r"$E_{ij}=\sigma(h_E[r_{ij};F^{local}_{ij}])$", 16)
txt(1726, 632, "Supervised by local fidelity proxy", 12)

arrow([(517, 643), (517, 691)], C["coral"], planned=True)
box(360, 691, 755, 78, "#F7F1EE", C["coral"], planned=True)
txt(382, 716, r"$U_Q$  Global degradation: motion / noise / aliasing", 15, C["coral"], True)
txt(382, 749, "F4 → multilabel presence + severity; independent of relation tokens", 11)

box(1200, 697, 935, 72, "#F8F4EC", C["amber"], planned=True)
txt(1218, 721, r"$\Omega_{ij}=[B_j\cup(\mathrm{dilate}(B_j,8\,\mathrm{mm})\cap M_i)]\cap M_{valid}$", 16)
txt(1218, 752, r"$F^{local}_{ij}=\mathrm{pool}(F2,\Omega_{ij})$   •   predicted ROI at inference", 12)
arrow([(650, 449), (687, 449), (687, 683), (1138, 683), (1138, 790), (1900, 790), (1900, 769)], C["amber"], planned=True)
label(1490, 790, "F2 local-feature path (planned)", C["amber"], 10)
arrow([(2025, 697), (2025, 671)], C["amber"], planned=True)

box(30, 829, 2140, 73, "#FAF5EC", "#DDCBAE")
txt(53, 853, "CURRENT LIMIT: GT masks + matched GT boxes + GT presence feed relation geometry/gates. Host accuracy is NOT M1 evidence.", 13, C["amber"], True)
txt(53, 880, "Full plan: cross-organ ontology, d=256 / M=20, wider Swin, physical 3D RoPE, valid-voxel + event gates, S / U_Q / E, and staged paired training.", 11, C["muted"])

# B. Already implemented degradation data generation vs planned paired training.
box(30, 922, 2140, 259)
txt(55, 951, "B   Controlled k-space views and external evidence supervision", 18, bold=True)
box(55, 993, 285, 91, C["pale"])
txt(197, 1022, "Same scan: raw k-space", 14, bold=True, ha="center")
txt(197, 1057, "Noise + undersampling: coded", 11, ha="center")
arrow([(340, 1038), (385, 1038)])
box(385, 993, 370, 91, "#F7F1EE", C["coral"])
txt(570, 1022, "Perturb + reconstruct", 15, C["coral"], True, "center")
txt(570, 1057, "Multicoil motion: planned", 12, ha="center")
arrow([(755, 1038), (800, 1038)])
box(800, 993, 345, 91, C["pale"])
txt(972, 1022, r"$X^0,\ X^{q_1},\ X^{q_2},\ X^{q_3}$", 19, ha="center")
txt(972, 1057, "Matched anatomy / event identity", 11, ha="center")
arrow([(1145, 1038), (1190, 1038)], planned=True)
box(1190, 993, 330, 91, "#EFF3F6", C["blue"], planned=True)
txt(1355, 1022, r"Shared $f_\theta$ from panel A", 15, C["blue"], True, "center")
txt(1355, 1057, "Paired training: planned", 12, ha="center")
arrow([(1520, 1038), (1570, 1038)], planned=True)
box(1570, 993, 565, 91, "#F8F4EC", C["amber"], planned=True)
txt(1593, 1022, "High evidence → preserve biological relation", 13)
txt(1593, 1057, "Low evidence → relax certainty; do not relabel", 12)

arrow([(972, 1084), (972, 1110), (595, 1110), (595, 1128)], C["amber"], dotted=True)
txt(57, 1145, r"$E^*_{ij}(q)=\exp[-\mathrm{NRMSE}(X^q,X^0;\Omega_{ij})/\tau_E]$", 18, C["amber"])
txt(1110, 1131, r"$w^q_{ij}=\mathbf{1}[E^*_{ij}(q)>\tau]\,\mathrm{sg}(E^*_{ij}(q))$", 17, C["amber"])
txt(1110, 1162, "Fixed reference ROI; detached E*. Missing evidence is excluded.", 11, C["muted"])
arrow([(880, 1145), (1050, 1145), (1050, 1131), (1090, 1131)], C["amber"], dotted=True)
arrow([(1720, 1131), (1855, 1131), (1855, 1084)], C["amber"], dotted=True)

# C. Actual executable loss first; proposed objective and stage switches below.
box(30, 1201, 2140, 402)
txt(55, 1231, "C   Loss functions: executable objective and full-plan extensions", 18, bold=True)
box(55, 1265, 2080, 76, "#EFF4F1", C["teal"])
txt(78, 1290, r"$\mathcal{L}_{ArmB}=(\mathcal{L}_{BCE}^{mask}+\mathcal{L}_{Dice}+\mathcal{L}_{BCE}^{pres})+(\mathcal{L}_{CE}^{U_B}+5\mathcal{L}_1+2\mathcal{L}_{GIoU})+(\mathcal{L}_{CE}^{host}+\mathcal{L}_{BCE}^{rel})$", 22)
txt(78, 1325, "Implemented: present-anatomy masks; Hungarian-matched boxes; known-host relations. No-object class weight = 0.1; relation weight = 1.", 11, C["muted"])

box(55, 1360, 1185, 221, "#F8F4EC", C["amber"], planned=True)
txt(77, 1385, "Planned evidence-aware objectives", 14, C["amber"], True)
txt(77, 1421, r"$\mathcal{L}_{rel}=\mathcal{L}_{rel}^{0}+\langle\mathcal{L}_{rel}^{q,w}\rangle_{q>0},\qquad\mathcal{L}_{rel}^{0}=\mathcal{L}_{bind}^{0}+\lambda_h\mathcal{L}_{hard}^{0}$", 16)
txt(77, 1461, r"$\mathcal{L}_{int}=\mathcal{L}_{cons}+\mathcal{L}_{Q}+\lambda_E\mathcal{L}_{obs}+\lambda_m\mathcal{L}_{rank}$", 19)
txt(77, 1499, r"High $E^*$: $\mathrm{KL}(\mathrm{sg}(P^0)\Vert P^q)$; low $E^*$: encourage $H(P^q)$.", 14)
txt(77, 1533, r"$\mathcal{L}_{obs}=\mathrm{Reg}(E,\mathrm{sg}(E^*))$; $\mathcal{L}_{rank}$ follows measured $E^*$ order.", 14)
txt(77, 1562, r"Defaults: $\lambda_R=\lambda_I=\lambda_E=1$; $\lambda_h=\lambda_m=0.5$; $\tau=0.5$.", 11, C["muted"])

box(1280, 1360, 855, 221, "#F4F5F7", C["blue"], planned=True)
txt(1302, 1385, "Planned training schedule (losses switched by stage)", 13, C["blue"], True)
for y, stage, formula in [
        (1422, "I   SSL", r"$\mathcal{L}_{MIM}+\mathcal{L}_{contrast}$"),
        (1460, "II  Perception", r"$\mathcal{L}_{A}+\mathcal{L}_{S}+\mathcal{L}_{U_B}+\mathcal{L}_{Q}$"),
        (1498, "III Binding", r"$\lambda_R\mathcal{L}_{rel}^{0}$"),
        (1536, "IV Paired", r"$\lambda_R\mathcal{L}_{rel}+\lambda_I\mathcal{L}_{int}$")]:
    txt(1302, y, stage, 12, bold=True)
    txt(1538, y, formula, 17)
txt(1302, 1565, "Stage IV: semantic loss off; U_Q loss counted once.", 11, C["muted"])

txt(53, 1630, "Interpretation: R predicts relation membership; E estimates local evidence via a fidelity proxy. Empty ROI / missed event → no E score; report coverage.", 12, C["muted"])
txt(53, 1658, "Source: anatobind/{data_engine,model,train} + RESEARCH_PLAN.md §§4–8. Proposed observability and generalization remain unvalidated.", 11, C["muted"])

for suffix in ("svg", "pdf", "png"):
    destination = OUT / f"{STEM}.{suffix}"
    fig.savefig(destination, dpi=200, facecolor="white")
    print(destination)
plt.close(fig)
