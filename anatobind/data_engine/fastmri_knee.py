"""fastMRI+ 膝关节的标注读取、清洗、3D 合并、患者划分与受控退化（leg 2 spec 第 1 节）。

标注是逐层 2D 框，坐标在 320x320 的 RSS 图像空间。3D 病灶由"相邻层、面内 IoU >= 0.3 相连"
的连通分量得到；同一类别族但空间分离的两串因此分成两个病灶。
"""
import csv
from collections import Counter, defaultdict

FAMILIES = ("meniscus", "cartilage", "bone", "ligament", "effusion")
FAMILY_OF_LABEL = {
    "Meniscus Tear": "meniscus",
    "Displaced Meniscal Tissue": "meniscus",
    "Cartilage - Partial Thickness loss/defect": "cartilage",
    "Cartilage - Full Thickness loss/defect": "cartilage",
    "Bone-Fracture/Contusion/dislocation": "bone",
    "Bone- Subchondral edema": "bone",
    "Bone - Lesion": "bone",
    "Ligament - ACL Low Grade sprain": "ligament",
    "Ligament - ACL High Grade Sprain": "ligament",
    "Ligament - MCL Low-Mod Grade Sprain": "ligament",
    "Ligament - MCL High Grade sprain": "ligament",
    "Ligament - PCL Low-Mod grade sprain": "ligament",
    "Ligament - PCL High Grade": "ligament",
    "LCL Complex - Low-Mod Grade Sprain": "ligament",
    "LCL Complex- High Grade Sprain": "ligament",
    "Patellar Retinaculum - High grade sprain": "ligament",
    "Joint Effusion": "effusion",
}
MIN_SIDE = 3


def read_annotations(csv_path):
    """CSV 行 -> 带 family 的整数化行；标签两端的空白必须 strip（原文件里 "Joint Effusion " 带尾空格）。"""
    out = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["study_level"].strip() == "Yes":
                continue
            fam = FAMILY_OF_LABEL.get(r["label"].strip())
            if fam is None:
                continue
            try:
                row = {"file": r["file"], "slice": int(r["slice"]), "x": int(r["x"]), "y": int(r["y"]),
                       "width": int(r["width"]), "height": int(r["height"]), "family": fam}
            except ValueError:
                continue
            out.append(row)
    return out


def clean_boxes(rows):
    kept, dropped = [], Counter()
    for r in rows:
        if r["width"] < MIN_SIDE or r["height"] < MIN_SIDE:
            dropped["too_small"] += 1
            continue
        kept.append(r)
    return kept, dropped


def _iou(a, b):
    ax1, ay1 = a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1 = b["x"] + b["width"], b["y"] + b["height"]
    iw = max(0, min(ax1, bx1) - max(a["x"], b["x"]))
    ih = max(0, min(ay1, by1) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union else 0.0


def merge_to_3d(rows, iou_min=0.3):
    """相邻层、面内 IoU >= iou_min 的框属于同一个 3D 病灶。"""
    lesions = []
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["file"], r["family"])].append(r)
    for (file, family), group in sorted(by_group.items()):
        parent = list(range(len(group)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, a in enumerate(group):
            for j, b in enumerate(group):
                if j <= i or abs(a["slice"] - b["slice"]) != 1:
                    continue
                if _iou(a, b) >= iou_min:
                    parent[find(i)] = find(j)
        comps = defaultdict(list)
        for i in range(len(group)):
            comps[find(i)].append(group[i])
        for members in comps.values():
            lesions.append({
                "file": file, "family": family,
                "z0": min(m["slice"] for m in members), "z1": max(m["slice"] for m in members),
                "x0": min(m["x"] for m in members), "y0": min(m["y"] for m in members),
                "x1": max(m["x"] + m["width"] for m in members),
                "y1": max(m["y"] + m["height"] for m in members),
                "n_boxes": len(members),
            })
    return lesions
