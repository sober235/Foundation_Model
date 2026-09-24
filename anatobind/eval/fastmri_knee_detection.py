# anatobind/eval/fastmri_knee_detection.py
"""Held-out detection numbers of the leg 2 knee detector after Gate 0 (v2.6 §6.2) on the 3D detections cached by
scripts/cache_detections.py. Matching and the operating point are the SKM-TEA gate's
(anatobind.eval.detection_metrics: one-to-one 3D IoU >= 0.1, <= 2 false positives per scan)."""
import numpy as np

from anatobind.eval.detection_metrics import match_scan
from anatobind.eval.matching import iou3d

SHARED_FAMILIES = ("meniscus", "cartilage", "ligament", "effusion")   # the SKM-TEA families; bone exists only in fastMRI+
THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.01, 1.0, 0.01), 2))


def corners(d):
    """(z0, y0, x0, z1, y1, x1) with z1 exclusive, as anatobind.eval.detect3d.failure_labels builds them."""
    return [d["z0"], d["y0"], d["x0"], d["z1"] + 1, d["y1"], d["x1"]]


def scan_record(file, gt_lesions, dets, normal, families=None):
    """One scan for detection_metrics: truth and detections restricted to `families` (None = all)."""
    keep = (lambda f: True) if families is None else (lambda f: f in families)
    return {"file": file, "normal": bool(normal),
            "gt": [{"box": corners(L), "family": L["family"]} for L in gt_lesions if keep(L["family"])],
            "dets": [{"box": corners(d), "family": d["family"], "score": float(d["score"])} for d in dets if keep(d["family"])]}


def box_volume_ml(box, spacing):
    z0, y0, x0, z1, y1, x1 = box
    return float((z1 - z0) * spacing[0] * (y1 - y0) * spacing[1] * (x1 - x0) * spacing[2]) / 1000.0


def hit_geometry(scan, thr, spacing, iou=0.1):
    """Per hit at threshold thr: centre error in mm (box centres, physical spacing) and 3D IoU."""
    dets = [d for d in scan["dets"] if d["score"] >= thr]
    pairs = match_scan(scan["gt"], dets, iou)
    out = []
    for g, p in pairs.items():
        a, b = np.array(scan["gt"][g]["box"], float), np.array(dets[p]["box"], float)
        ca, cb = (a[:3] + a[3:]) / 2, (b[:3] + b[3:]) / 2
        out.append({"file": scan["file"], "family": scan["gt"][g]["family"], "pred_family": dets[p]["family"],
                    "centre_error_mm": float(np.sqrt((((ca - cb) * np.asarray(spacing, float)) ** 2).sum())),
                    "iou": float(iou3d(a[None], b[None])[0, 0]), "score": dets[p]["score"]})
    return out


def size_terciles(scans, spacing_of, thr, iou=0.1):
    """Localisation hits by family x size tercile (mL), like the SKM-TEA miss analysis."""
    rows = []
    for s in scans:
        dets = [d for d in s["dets"] if d["score"] >= thr]
        pairs = match_scan(s["gt"], dets, iou)
        for g, r in enumerate(s["gt"]):
            rows.append({"family": r["family"], "ml": box_volume_ml(r["box"], spacing_of[s["file"]]), "hit": int(g in pairs)})
    out = {}
    for fam in sorted({r["family"] for r in rows}):
        sel = [r for r in rows if r["family"] == fam]
        ml, hit = np.array([r["ml"] for r in sel]), np.array([r["hit"] for r in sel])
        bins = np.digitize(ml, np.quantile(ml, [1 / 3, 2 / 3]), right=True)
        out[fam] = [{"lo": float(ml[bins == b].min()) if (bins == b).any() else None,
                     "hi": float(ml[bins == b].max()) if (bins == b).any() else None,
                     "n": int((bins == b).sum()), "hit": int(hit[bins == b].sum())} for b in range(3)]
    return out


def patient_coverage(scans, patient_of, thr, iou=0.1):
    """Share of patients with >= 1 lesion that get >= 1 family-correct hit at thr."""
    has, got = set(), set()
    for s in scans:
        if not s["gt"]:
            continue
        p = patient_of[s["file"]]
        has.add(p)
        dets = [d for d in s["dets"] if d["score"] >= thr]
        if any(dets[q]["family"] == s["gt"][g]["family"] for g, q in match_scan(s["gt"], dets, iou).items()):
            got.add(p)
    return {"n_patients": len(has), "covered": len(got), "coverage": len(got) / len(has) if has else 0.0}


def fp_per_normal_scan(scans, thr):
    """Detections per scan on the volumes the radiologist left without any annotation (true negatives)."""
    normals = [s for s in scans if s["normal"]]
    n_fp = sum(sum(1 for d in s["dets"] if d["score"] >= thr) for s in normals)
    return {"n_normal": len(normals), "fp_per_normal_scan": n_fp / len(normals) if normals else 0.0}
