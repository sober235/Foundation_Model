"""逐层检出 -> 3D 病灶 -> 与真值匹配 -> 任务失败标签(leg 2 spec 3.1)。

3D 聚合用与真值同一条规则(相邻层、面内 IoU >= 0.3 相连),因此预测与真值在同一口径上比较。
扫描级标签把漏检计入:逐病灶可靠性头看不见漏检,扫描级头必须看得见。
"""
import numpy as np
import torch

from anatobind.data_engine.fastmri_knee import FAMILIES
from anatobind.eval.matching import iou3d, match
from anatobind.model.dense_head_2d import decode_centres_2d


def detect_volume(model, vol, device, M=8, score_min=0.05, slab=5):
    """vol: (S, H, W) float32。返回逐层检出。"""
    half = slab // 2
    out, globals_ = [], []
    model.eval()
    with torch.no_grad():
        for z in range(vol.shape[0]):
            idx = np.clip(np.arange(z - half, z + half + 1), 0, vol.shape[0] - 1)
            x = torch.from_numpy(np.ascontiguousarray(vol[idx]))[None, None].to(device)
            pred = model(x)
            globals_.append(pred["global_feat"][0].float().cpu().numpy())
            boxes, cls_prob, embed = decode_centres_2d(pred, M)
            for m in range(M):
                c = int(cls_prob[m, :len(FAMILIES)].argmax())
                s = float(cls_prob[m, c])
                if s < score_min:
                    continue
                y0, x0, y1, x1 = (float(v) for v in boxes[m])
                out.append({"slice": z, "family": FAMILIES[c], "score": s,
                            "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                            "embed": embed[m].float().cpu().numpy()})
    return out, np.mean(globals_, axis=0).astype(np.float32)


def _iou2d(a, b):
    iy = max(0.0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    ix = max(0.0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    inter = iy * ix
    union = (a["y1"] - a["y0"]) * (a["x1"] - a["x0"]) + (b["y1"] - b["y0"]) * (b["x1"] - b["x0"]) - inter
    return inter / union if union > 0 else 0.0


def aggregate_to_3d(dets, iou_min=0.3):
    groups = {}
    for d in dets:
        groups.setdefault(d["family"], []).append(d)
    out = []
    for family, group in sorted(groups.items()):
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
                if _iou2d(a, b) >= iou_min:
                    parent[find(i)] = find(j)
        comps = {}
        for i in range(len(group)):
            comps.setdefault(find(i), []).append(group[i])
        for members in comps.values():
            peak = max(members, key=lambda m: m["score"])
            out.append({"family": family, "score": peak["score"],
                        "z0": min(m["slice"] for m in members), "z1": max(m["slice"] for m in members),
                        "y0": min(m["y0"] for m in members), "x0": min(m["x0"] for m in members),
                        "y1": max(m["y1"] for m in members), "x1": max(m["x1"] for m in members),
                        "embed": peak["embed"], "n_slices": len(members)})
    return out


def _corners(d):
    return [d["z0"], d["y0"], d["x0"], d["z1"] + 1, d["y1"], d["x1"]]


def failure_labels(gt, pred, iou=0.1):
    """per_lesion: 每个检出一行,correct = 匹配上且类别族正确。scan_label: 有任何错误(含漏检)则 1。"""
    g = np.array([_corners(L) for L in gt], dtype=float).reshape(-1, 6)
    p = np.array([_corners(d) for d in pred], dtype=float).reshape(-1, 6)
    pairs = match(g, p, iou) if len(g) and len(p) else {}
    gt_of_pred = {q: j for j, q in pairs.items()}
    per = []
    for q, d in enumerate(pred):
        j = gt_of_pred.get(q)
        correct = int(j is not None and gt[j]["family"] == d["family"])
        per.append({"score": d["score"], "correct": correct, "family": d["family"],
                    "box": _corners(d), "embed": d["embed"]})
    n_matched_right = sum(r["correct"] for r in per)
    scan = int(n_matched_right < len(gt) or any(r["correct"] == 0 for r in per))
    return per, scan
