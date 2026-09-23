"""Sensitivity and false positives per scan for 3D lesion boxes (spec 2026-09-23 §4.1).

Matching is one-to-one on 3D IoU >= IOU (anatobind.eval.matching.match), family-agnostic; a hit
whose predicted family differs from the truth still counts as localised, and 'sensitivity_family'
counts only hits with the right family. The operating point is the score threshold with the best
family sensitivity among those with <= FP_MAX false positives per scan; the gate reads it.
"""
import numpy as np

from anatobind.eval.matching import match

IOU = 0.1
FP_MAX = 2.0
GATE_SENSITIVITY = 0.5
THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.05, 1.0, 0.05), 2))


def match_scan(gt, dets, iou=IOU):
    g = np.array([r["box"] for r in gt], float).reshape(-1, 6)
    p = np.array([d["box"] for d in dets], float).reshape(-1, 6)
    return match(g, p, iou)


def _count(scans, thr, iou):
    n_gt = n_hit = n_fam = n_fp = 0
    for s in scans:
        dets = [d for d in s["dets"] if d["score"] >= thr]
        pairs = match_scan(s["gt"], dets, iou)
        n_gt += len(s["gt"])
        n_hit += len(pairs)
        n_fam += sum(1 for g, p in pairs.items() if dets[p]["family"] == s["gt"][g]["family"])
        n_fp += len(dets) - len(pairs)
    return n_gt, n_hit, n_fam, n_fp


def _rate(a, b):
    return a / b if b else 0.0


def sweep(scans, thresholds=THRESHOLDS, iou=IOU):
    rows = []
    for thr in thresholds:
        n_gt, n_hit, n_fam, n_fp = _count(scans, thr, iou)
        rows.append({"thr": float(thr), "n_gt": n_gt, "n_hit": n_hit, "n_hit_family": n_fam, "n_fp": n_fp,
                     "n_scans": len(scans), "sensitivity": _rate(n_hit, n_gt),
                     "sensitivity_family": _rate(n_fam, n_gt), "fp_per_scan": _rate(n_fp, len(scans))})
    return rows


def operating_point(rows, fp_max=FP_MAX):
    ok = [r for r in rows if r["fp_per_scan"] <= fp_max]
    return max(ok, key=lambda r: (r["sensitivity_family"], r["thr"])) if ok else None


def per_family(scans, thr, iou=IOU):
    out = {}
    for s in scans:
        dets = [d for d in s["dets"] if d["score"] >= thr]
        pairs = match_scan(s["gt"], dets, iou)
        for g, r in enumerate(s["gt"]):
            f = out.setdefault(r["family"], {"n_gt": 0, "n_hit": 0, "n_hit_family": 0})
            f["n_gt"] += 1
            if g in pairs:
                f["n_hit"] += 1
                f["n_hit_family"] += int(dets[pairs[g]]["family"] == r["family"])
    for f in out.values():
        f["sensitivity"] = _rate(f["n_hit"], f["n_gt"])
        f["sensitivity_family"] = _rate(f["n_hit_family"], f["n_gt"])
    return out


def gate(rows, fp_max=FP_MAX, sensitivity_min=GATE_SENSITIVITY):
    op = operating_point(rows, fp_max)
    if op is None:
        return {"pass": False, "thr": None, "sensitivity_family": 0.0, "fp_per_scan": None}
    return {"pass": bool(op["sensitivity_family"] >= sensitivity_min), "thr": op["thr"],
            "sensitivity_family": op["sensitivity_family"], "fp_per_scan": op["fp_per_scan"]}
